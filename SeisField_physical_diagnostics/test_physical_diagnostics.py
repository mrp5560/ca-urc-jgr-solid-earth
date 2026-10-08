# -*- coding: utf-8 -*-
"""Synthetic tests only. No seismic experimental results are asserted."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

import diagnose_physical_conditions as d


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture(root):
    run, hroot = root/'run', root/'h5'
    (run/'audit').mkdir(parents=True)
    (run/'evaluation_grouped').mkdir()
    hroot.mkdir()
    settings = {'split_column':'split_grouped', 'input_stations':5, 'target_stations':2,
                'test_repeats':3,'seed':20260713,'t0_sec':5,'maximum_correction':1.5}
    cohort, records, truth_rows, draw_rows, pred_rows = [], [], [], [], []
    for e in range(10):
        eid = str(1000+e)
        split = 'train' if e < 4 else 'test'
        p = np.array([0,1,2,3,4,4.5,6,7,8],dtype=np.float32)
        s = p+2
        s[7] = np.nan
        if e == 5:
            s[0] = np.nan
        coords = np.column_stack((33+np.arange(9)*.12, -118+np.arange(9)*.17, np.full(9,20))).astype(np.float32)
        path = hroot/f'{eid}.h5'
        with h5py.File(path,'w') as h:
            h.attrs.update(event_id=eid,first_p_time='2020-01-01T00:00:00Z',magnitude=3.1+e*.1,
                           latitude=33,longitude=-118,depth_km=9)
            h['station_coords']=coords
            h['p_offset_sec']=p
            h['s_offset_sec']=s
            h.create_dataset('station_id',data=np.array([f'CI.S{i}.--' for i in range(9)],object),dtype=h5py.string_dtype('utf-8'))
            h.create_dataset('acceleration',shape=(9,3,20),dtype='float32')
        cohort.append({'event_id':eid,'h5_path':str(path),'split_grouped':split,
                       'sequence_group':f'G{e}','magnitude':3.1+e*.1,'depth_km':9})
        records.append({'event_id':eid,'split':split,'source_sha256':d.digest_file(path)})
        if split != 'test':
            continue
        for rep in range(3):
            rng=np.random.default_rng(d.canonical_seed(f'locked:test:{eid}:repeat:{rep}',settings['seed']))
            ins=rng.choice(np.arange(6),size=5,replace=False)
            ts=rng.choice(np.setdiff1d(np.arange(9),ins),size=2,replace=False)
            draw_rows.append({'event_id':eid,'repeat':rep,'input_station_indices':'|'.join(map(str,ins)),
                              'target_station_indices':'|'.join(map(str,ts))})
            for slot, ti in enumerate(ts):
                row={'event_id':eid,'repeat':rep,'target_slot':slot,'target_station_index':int(ti),
                     'true_log10_pga':1 if ti%2 else .3,'true_log10_pgv':1.2 if ti%2 else .4}
                truth_rows.append(row.copy())
                for q,th in [('pga',.5),('pgv',.6)]:
                    true=row[f'true_log10_{q}']; base=true+(-.55 if ti%2 else .15)
                    risk=.6+.025*int(ti); raw=.2; applied=risk**5*raw
                    row.update({f'is_tail_{q}':true>=th,f'Prefix_Base_log10_{q}':base,
                                f'Prefix_CAURC_log10_{q}':base+applied,
                                f'underprediction_risk_score_{q}':risk,f'raw_correction_amplitude_{q}':raw,
                                f'applied_correction_{q}':applied})
                row['input_station_indices']='|'.join(map(str,ins))
                pred_rows.append(row)
    c=pd.DataFrame(cohort); f=pd.DataFrame(pred_rows)
    c.to_csv(run/'audit/cohort_manifest.csv',index=False)
    pd.DataFrame(truth_rows).to_csv(run/'audit/checked_test_keys_and_truth.csv',index=False)
    pd.DataFrame(draw_rows).to_csv(run/'audit/checked_test_station_draws.csv',index=False)
    f.sample(frac=1,random_state=11).to_csv(run/'evaluation_grouped/locked_prefix_test_predictions.csv',index=False)
    metrics=[]
    for method in d.METHODS:
        for q in d.QUANTITIES:
            for pop in ('overall','non_tail','high_motion_tail'):
                w=f.copy()
                if pop!='overall':
                    w=w.loc[w[f'is_tail_{q}'] if pop=='high_motion_tail' else ~w[f'is_tail_{q}']].copy()
                r=w[f'{method}_log10_{q}']-w[f'true_log10_{q}']
                vals={'mae':abs(r),'bias':r,'under05':(r<=-.5).astype(float),
                      'factor2':(abs(r)<=math.log10(2)).astype(float)}
                for metric,x in vals.items():
                    w['v']=x
                    ev=w.groupby(['event_id','repeat']).v.mean().groupby('event_id').mean()
                    metrics.append({'method':method,'quantity':q,'population':pop,'metric':metric,
                                    'value':ev.mean(),'n_events':len(ev),'n_target_rows':len(w),
                                    'n_event_repeats':len(w[['event_id','repeat']].drop_duplicates()),'n_groups':w.event_id.nunique()})
    pd.DataFrame(metrics).to_csv(run/'evaluation_grouped/paired_metrics.csv',index=False)
    dump(run/'experiment_protocol.json',{'protocol_id':'synthetic-only','settings':settings,'thresholds':[.5,.6],
                                         'event_order_and_content':records})
    dump(run/'evaluation_grouped/evaluation_audit.json',{'snapshot_protocol_id':'synthetic-only',
            'test_events':6,'target_rows':36,'group_column':'sequence_group','selected_gamma':5})
    return run,hroot


class UnitTests(unittest.TestCase):
    def test_indices_preserve_unsorted(self):
        self.assertEqual(d.parse_indices('4|1|6'),(4,1,6))
    def test_indices_reject_duplicate(self):
        with self.assertRaises(ValueError):d.parse_indices('1|2|1')
    def test_indices_reject_float(self):
        with self.assertRaises(ValueError):d.parse_indices('1.0|2')
    def test_indices_reject_missing(self):
        with self.assertRaises(ValueError):d.parse_indices(np.nan)
    def test_strict_bool(self):
        self.assertEqual(d.strict_bool(pd.Series(['true','False','1','0'])).tolist(),[True,False,True,False])
    def test_strict_bool_reject(self):
        with self.assertRaises(ValueError):d.strict_bool(pd.Series(['maybe']))
    def test_phase_boundaries(self):
        p=np.array([6,5,2,2]);s=np.array([8,7,5,4])
        a,b,valid=d.phase_states(p,s,5)
        self.assertEqual(a.tolist(),['Pre-P','Post-P','Post-P','Post-P'])
        self.assertEqual(b.tolist(),['Pre-P','P-to-S','Post-S','Post-S'])
    def test_missing_s_not_assumed_before(self):
        a,b,ok=d.phase_states(np.array([1.,6.]),np.array([np.nan,np.nan]),5)
        self.assertEqual(a.tolist(),['Post-P','Pre-P'])
        self.assertEqual(b.tolist(),['Unknown/invalid PS']*2)
        self.assertFalse(ok.any())
    def test_invalid_pair(self):
        a,b,ok=d.phase_states(np.array([2.,np.nan,-5.,2.]),np.array([1.,3.,0.,2.]),5)
        self.assertFalse(ok.any())
    def test_distance_local_source_convention(self):
        coords=np.array([[0,0,0],[0,1,0],[1,0,0]],np.float32)
        dist=d.distance_matrix(coords,np.array([0]),np.array([1,2]))[:,0]
        np.testing.assert_allclose(dist,[111.32,110.57],rtol=1e-6)
    def test_event_balancing(self):
        f=pd.DataFrame({'event_id':['a','a','a','b'],'repeat':[0,0,1,0]})
        s=d.event_metric(f,np.array([0,2,8,20]),np.array([True]*4))
        self.assertEqual(s['a'],4.5);self.assertEqual(s['b'],20)
    def test_empty_tail_not_zero(self):
        f=pd.DataFrame({'event_id':['a','b'],'repeat':[0,0]})
        s=d.event_metric(f,np.array([1,9]),np.array([True,False]))
        self.assertNotIn('b',s.index)
    def test_bootstrap_constant(self):
        events=['a','b','c','d']; g=pd.Series(['g1','g1','g2','g2'],index=events)
        w=d.bootstrap_weights(events,g,500,11)
        ci=d.ci_for_series(pd.Series(-.1,index=events),events,g,w,2,2)
        self.assertAlmostEqual(ci['ci95_low'],-.1); self.assertAlmostEqual(ci['ci95_high'],-.1)
    def test_sparse_suppression(self):
        events=['a','b'];g=pd.Series(['g1','g2'],index=events);w=d.bootstrap_weights(events,g,100,10)
        ci=d.ci_for_series(pd.Series([1,2],index=events),events,g,w,10,5)
        self.assertEqual(ci['inference_status'],'descriptive_only_sparse')
        self.assertTrue(np.isnan(ci['ci95_low']))
    def test_seed(self):
        expected=int.from_bytes(hashlib.sha256(b'train:123:repeat:0:99').digest()[:8],'little')%(2**32)
        self.assertEqual(d.canonical_seed('train:123:repeat:0',99),expected)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.run,self.hroot=fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def call(self,stage='audit',extra=None,out='diag'):
        cmd=[sys.executable,str(Path(d.__file__)),'--stage',stage,'--run-dir',str(self.run),
             '--h5-root',str(self.hroot),'--out-dir',str(self.root/out),'--bootstrap-replicates','200',
             '--min-events-ci','2','--min-groups-ci','2','--train-geometry-repeats','2']
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        return subprocess.run(cmd+(extra or []),capture_output=True,text=True,env=env,timeout=35)
    def test_full_audit_then_analysis(self):
        before={p:d.digest_file(p) for p in self.root.rglob('*') if p.is_file()}
        a=self.call();self.assertEqual(a.returncode,0,a.stdout+a.stderr)
        b=self.call('analyze');self.assertEqual(b.returncode,0,b.stdout+b.stderr)
        report=d.read_json(self.root/'diag/physical_diagnostics_audit.json')
        self.assertEqual(report['status'],'analysis_complete')
        self.assertEqual(report['target_rows'],36)
        self.assertLess(report['max_global_metric_reconciliation_error'],1e-12)
        for p,h in before.items():self.assertEqual(d.digest_file(p),h)
        result=pd.read_csv(self.root/'diag/physical_stratified_metrics.csv')
        self.assertFalse(result.empty)
        e=pd.read_csv(self.root/'diag/physical_target_diagnostics.csv')
        self.assertIn('Unknown/invalid PS',set(e.target_ps_state))
        self.assertIn('Unknown/partial S',set(e.input_s_state))
    def test_failed_draw_alignment(self):
        p=self.run/'audit/checked_test_station_draws.csv';df=pd.read_csv(p)
        v=df.loc[0,'input_station_indices'].split('|');v[0],v[1]=v[1],v[0]
        df.loc[0,'input_station_indices']='|'.join(v);df.to_csv(p,index=False)
        a=self.call();self.assertEqual(a.returncode,2);self.assertIn('input indices differ',a.stderr)
    def test_label_mismatch(self):
        p=self.run/'evaluation_grouped/locked_prefix_test_predictions.csv';df=pd.read_csv(p)
        df.loc[0,'true_log10_pga']+=1;df.to_csv(p,index=False)
        a=self.call();self.assertEqual(a.returncode,2);self.assertIn('truth audit mismatch',a.stderr)
    def test_source_hash_guard(self):
        with h5py.File(self.hroot/'1000.h5','r+') as h:h.attrs['changed']='yes'
        a=self.call(extra=['--verify-source-hash']);self.assertEqual(a.returncode,2)
        self.assertIn('hash differs',a.stderr)
    def test_global_metric_reconciliation(self):
        p=self.run/'evaluation_grouped/paired_metrics.csv';f=pd.read_csv(p)
        f.loc[0,'value']+=.01;f.to_csv(p,index=False)
        a=self.call();self.assertEqual(a.returncode,2);self.assertIn('reconciliation failed',a.stderr)
    def test_missing_predictions(self):
        (self.run/'evaluation_grouped/locked_prefix_test_predictions.csv').unlink()
        a=self.call();self.assertEqual(a.returncode,2);self.assertIn('file missing',a.stderr)
    def test_configuration_change_refused(self):
        a=self.call();self.assertEqual(a.returncode,0,a.stderr)
        b=self.call(extra=['--train-geometry-repeats','3'])
        self.assertEqual(b.returncode,2);self.assertIn('configuration changed',b.stderr)
    def test_not_inside_training_output(self):
        a=self.call(extra=['--out-dir',str(self.run/'diag')])
        self.assertEqual(a.returncode,2);self.assertIn('separate output directory',a.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
