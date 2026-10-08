#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 7: CA-URC under distribution shifts.

Uses saved metrics only: NO training, inference, re-selection or bootstrap rerun.
Default mode reads the original project CSV outputs. --verified-summary reproduces
an explicitly labelled figure from six-decimal values transcribed programmatically
from the four verified run logs. These are real logged values, not synthetic data.

Required packages: numpy, pandas, matplotlib, Pillow, pymupdf.
Every chart is rendered independently; PDF/PNG/SVG composition does not rescale
any panel. At 183 mm assembled width the original text size is retained.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SCENARIOS = ['grouped', 'chronological', 'unseen_station', 'ridgecrest']
LABELS = {'grouped': 'Grouped', 'chronological': '2021–2024',
          'unseen_station': 'Unseen stations', 'ridgecrest': 'Ridgecrest'}
BOOTSTRAP = {'grouped': 'hierarchical_sequence_event', 'chronological': 'event',
             'unseen_station': 'event', 'ridgecrest': 'event_within_single_heldout_sequence'}
EXPECTED_N = {'grouped': (224, 98, 87), 'chronological': (192, 188, 186),
              'unseen_station': (195, 37, 30), 'ridgecrest': (593, 591, 589)}
# Populations and resampling units must NOT be silently standardized across protocols.
PROTOCOLS = [
 {'scenario':'grouped','evaluation':'Sequence-grouped held-out events','n_total_events':224,
  'n_prediction_rows':44800,'model_policy':'Original grouped base and A4 lock; head epoch 3, gamma 5',
  'threshold_policy':'Grouped training-derived thresholds','bootstrap_unit':BOOTSTRAP['grouped']},
 {'scenario':'chronological','evaluation':'2021–2024 events; train 2010–2018; validation non-Ridgecrest 2019 + 2020',
  'n_total_events':192,'n_prediction_rows':38400,
  'model_policy':'Separately trained temporal base epoch 49 and head epoch 19, gamma 5',
  'threshold_policy':'Temporal training-derived thresholds','bootstrap_unit':'event'},
 {'scenario':'unseen_station','evaluation':'Unseen target role after P-state/distance matching, 20-km caliper',
  'n_total_events':195,'n_prediction_rows':5649,
  'model_policy':'Separately trained station-OOD model; unseen stations excluded from train/validation',
  'threshold_policy':'Station-OOD seen-only training targets','bootstrap_unit':'event'},
 {'scenario':'ridgecrest','evaluation':'Retrospective single-held-out-sequence transfer, 2019 Ridgecrest',
  'n_total_events':593,'n_prediction_rows':118600,
  'model_policy':'Unchanged chronological base/head/gamma; no Ridgecrest re-selection',
  'threshold_policy':'Unchanged temporal training-derived thresholds',
  'bootstrap_unit':BOOTSTRAP['ridgecrest']}
]
VERIFIED_CSV = 'scenario,quantity,population,metric,base_value,ca_urc_value,delta,ci_lower,ci_upper,n_events,n_sequence_groups,n_target_rows,bootstrap_unit,source\ngrouped,pga,overall,mae,0.305941,0.305531,-0.00041,-0.005833,0.003725,224,22.0,44800.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pga,overall,factor2,0.576607,0.575201,-0.001406,-0.009366,0.009052,224,22.0,44800.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pga,overall,under05,0.105112,0.08846,-0.016652,-0.021382,-0.012086,224,22.0,44800.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pga,high_motion_tail,mae,0.639199,0.566925,-0.072274,-0.078435,-0.061038,98,21.0,1363.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pga,high_motion_tail,factor2,0.14576,0.205067,0.059307,0.024329,0.132614,98,21.0,1363.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pga,high_motion_tail,under05,0.63999,0.538049,-0.101942,-0.213588,-0.043684,98,21.0,1363.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pgv,overall,mae,0.265767,0.263486,-0.002281,-0.006958,0.001475,224,22.0,44800.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pgv,overall,factor2,0.672054,0.669687,-0.002366,-0.010463,0.007308,224,22.0,44800.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pgv,overall,under05,0.092388,0.077857,-0.014531,-0.019736,-0.009442,224,22.0,44800.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pgv,high_motion_tail,mae,0.746959,0.64841,-0.098549,-0.113612,-0.083568,87,18.0,1485.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pgv,high_motion_tail,factor2,0.106318,0.20574,0.099422,0.031473,0.171055,87,18.0,1485.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\ngrouped,pgv,high_motion_tail,under05,0.714567,0.60354,-0.111027,-0.213529,-0.051458,87,18.0,1485.0,hierarchical_sequence_event,粘贴的文本 (1)(20260903-062132).txt\nchronological,pga,overall,mae,0.387721,0.386031,-0.001691,-0.003638,0.000218,192,,38400.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pga,overall,factor2,0.474115,0.475286,0.001172,-0.002266,0.004557,192,,38400.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pga,overall,under05,0.159974,0.143698,-0.016276,-0.018385,-0.014245,192,,38400.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pga,high_motion_tail,mae,0.599704,0.567024,-0.032681,-0.034962,-0.030386,188,,4311.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pga,high_motion_tail,factor2,0.261298,0.295363,0.034065,0.02252,0.046629,188,,4311.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pga,high_motion_tail,under05,0.539727,0.497006,-0.042721,-0.053676,-0.032931,188,,4311.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pgv,overall,mae,0.34626,0.345611,-0.000649,-0.00208,0.00075,192,,38400.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pgv,overall,factor2,0.534062,0.531589,-0.002474,-0.005208,0.00026,192,,38400.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pgv,overall,under05,0.146849,0.137083,-0.009766,-0.011276,-0.008307,192,,38400.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pgv,high_motion_tail,mae,0.706155,0.670638,-0.035517,-0.037997,-0.033103,186,,4738.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pgv,high_motion_tail,factor2,0.182651,0.225455,0.042804,0.030693,0.056327,186,,4738.0,event,粘贴的文本 (1)(20260903-084012).txt\nchronological,pgv,high_motion_tail,under05,0.656697,0.610945,-0.045752,-0.055792,-0.036263,186,,4738.0,event,粘贴的文本 (1)(20260903-084012).txt\nridgecrest,pga,overall,mae,0.515864,0.496393,-0.019471,-0.020329,-0.018599,593,1.0,118600.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pga,overall,factor2,0.369098,0.386998,0.017901,0.016273,0.019562,593,1.0,118600.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pga,overall,under05,0.380219,0.353921,-0.026298,-0.027622,-0.024958,593,1.0,118600.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pga,high_motion_tail,mae,0.76755,0.739054,-0.028497,-0.029091,-0.027885,591,1.0,33533.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pga,high_motion_tail,factor2,0.168151,0.186903,0.018752,0.015747,0.021929,591,1.0,33533.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pga,high_motion_tail,under05,0.676127,0.658264,-0.017863,-0.020771,-0.015199,591,1.0,33533.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pgv,overall,mae,0.552683,0.544747,-0.007936,-0.008445,-0.007421,593,1.0,118600.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pgv,overall,factor2,0.374384,0.380514,0.00613,0.005067,0.007184,593,1.0,118600.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pgv,overall,under05,0.407327,0.396695,-0.010632,-0.011417,-0.009865,593,1.0,118600.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pgv,high_motion_tail,mae,0.807832,0.781284,-0.026548,-0.027592,-0.025514,589,1.0,39212.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pgv,high_motion_tail,factor2,0.117679,0.136529,0.01885,0.014675,0.023687,589,1.0,39212.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nridgecrest,pgv,high_motion_tail,under05,0.730476,0.700038,-0.030438,-0.036671,-0.025026,589,1.0,39212.0,event_within_single_heldout_sequence,粘贴的文本 (1)(20260903-091339).txt\nunseen_station,pga,overall,mae,0.357014,0.357947,0.000933,-0.003247,0.004921,195,,5649.0,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pga,overall,factor2,0.520563,0.519295,-0.001268,-0.014597,0.012164,195,,5649.0,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pga,overall,under05,0.142299,0.12045,-0.02185,-0.02868,-0.015856,195,,5649.0,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pga,high_motion_tail,mae,0.86096,0.79973,-0.06123,-0.073219,-0.049192,37,,,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pga,high_motion_tail,factor2,0.07683,0.096487,0.019657,0.000711,0.044034,37,,,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pga,high_motion_tail,under05,0.825991,0.780548,-0.045443,-0.108903,-0.00265,37,,,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pgv,overall,mae,0.30931,0.304324,-0.004986,-0.009288,-0.000875,195,,5649.0,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pgv,overall,factor2,0.625367,0.631233,0.005866,-0.007014,0.018585,195,,5649.0,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pgv,overall,under05,0.137914,0.112146,-0.025768,-0.03375,-0.018498,195,,5649.0,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pgv,high_motion_tail,mae,1.02031,0.960586,-0.059724,-0.074961,-0.045019,30,,,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pgv,high_motion_tail,factor2,0.107958,0.12005,0.012092,0.0,0.035294,30,,,event,粘贴的文本 (1)(20260903-124642).txt\nunseen_station,pgv,high_motion_tail,under05,0.810047,0.784141,-0.025906,-0.055556,-0.001462,30,,,event,粘贴的文本 (1)(20260903-124642).txt\n'
VERIFIED_GAP_CSV = 'caliper_km,comparison_type,group_or_model,quantity,population,metric,n_events,point_delta,ci_lower,ci_upper,ci_excludes_zero,two_sided_bootstrap_sign_p,bootstrap_repetitions,bootstrap_unit\n20.0,unseen_minus_seen,cross_attention_base,pga,overall,mae,195,0.04877,0.033911,0.063764,True,0.0002,10000,event\n20.0,unseen_minus_seen,cross_attention_base,pgv,overall,mae,195,0.015772,0.003409,0.028143,True,0.013399,10000,event\n20.0,unseen_minus_seen,ca_urc,pga,overall,mae,195,0.053925,0.03904,0.068571,True,0.0002,10000,event\n20.0,unseen_minus_seen,ca_urc,pgv,overall,mae,195,0.022836,0.011078,0.035157,True,0.0006,10000,event\n'

CAPTION = """Fig. 7 | Tail-risk reductions persist across distinct distribution-shift evaluations.
a,b, Event-macro high-motion-tail MAE for the Cross-Attention Base and CA-URC,
for PGA and PGV, respectively. Each horizontal segment joins two models evaluated
on the same targets within one protocol, not different domains. Numbers denote
Base -> CA-URC; percentages are point estimates of the relative tail-MAE reduction.
c,d, Paired tail-MAE changes, CA-URC minus Base. e,f, Paired changes in tail severe
underprediction rate U0.5, shown in percentage points. Negative changes favour
CA-URC. Intervals in c–f are the reported 95% paired-bootstrap intervals, NOT
intervals on individual model values. The grouped reference uses hierarchical
sequence-to-event bootstrap; chronological and matched-station evaluations use
event bootstrap; Ridgecrest uses event bootstrap within one held-out sequence.
All use 10,000 replicates. Labels in c–f give the contributing tail-event count.
The station result is for unseen targets under the primary 20-km distance caliper
and exact P-wave-state matching. Chronological and station models were trained
under their own train/validation restrictions; only Ridgecrest reused the frozen
chronological artifacts without retraining or re-selection. Axes share scales
between quantities within each panel pair. Absolute errors across protocols are
descriptive, not a common-population ranking. Ridgecrest is a retrospective
sequence-transfer test, not a real-time 2019 replay. Source values and the
protocol-specific uncertainty units are provided in Supplementary Table 6."""


def require(df: pd.DataFrame, names: list[str], context: str) -> None:
    missing=set(names)-set(df.columns)
    if missing:
        raise ValueError(f'{context}: missing columns {sorted(missing)}')


def read_csv(path: str, registry: dict) -> pd.DataFrame:
    p=Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f'Required saved result not found: {p}\n'
                                'Use the exact output from the earlier experiment; do not substitute another run.')
    raw=p.read_bytes()
    registry[str(p.resolve())]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    return pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig')


def field(row: pd.Series, alternatives: list[str]) -> float:
    for name in alternatives:
        if name in row and pd.notna(row[name]):
            x=float(row[name])
            if np.isfinite(x):
                return x
    raise ValueError(f'Missing finite value in columns {alternatives}')


def select_comparison(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    d=df.copy()
    require(d,['quantity','population','metric'],scenario)
    if 'candidate' in d:
        d=d[d.candidate.astype(str).str.lower().isin(['ca_urc','caurc','a4_under_only'])]
    if 'reference' in d:
        d=d[d.reference.astype(str).str.lower().isin(['cross_attention','cross_attention_base','a2_cross_attention_base'])]
    if 'bootstrap_mode' in d:
        target=BOOTSTRAP[scenario]
        d=d[d.bootstrap_mode.eq(target)]
    if 'bootstrap_repetitions' in d:
        if not (pd.to_numeric(d.bootstrap_repetitions)==10000).all():
            raise ValueError(f'{scenario}: the locked result used 10,000 bootstrap replicates.')
    if 'confidence_level' in d and not np.allclose(d.confidence_level,0.95):
        raise ValueError(f'{scenario}: expected 95% intervals.')
    d=d[d.population.isin(['overall','high_motion_tail']) & d.metric.isin(['mae','under05','factor2'])]
    if d.empty:
        raise ValueError(f'{scenario}: no matching CA-URC vs Cross-Attention rows.')
    return d


def normalize(df: pd.DataFrame, scenario: str, source: str) -> pd.DataFrame:
    rows=[]
    for _,d in select_comparison(df,scenario).iterrows():
        rows.append(dict(scenario=scenario,quantity=str(d.quantity).lower(),population=d.population,
            metric=d.metric,base_value=field(d,['reference_value','base_value']),
            ca_urc_value=field(d,['candidate_value','ca_urc_value']),
            delta=field(d,['point_delta_candidate_minus_reference','mean_delta_caurc_minus_base']),
            ci_lower=field(d,['ci_lower','ci95_low']),ci_upper=field(d,['ci_upper','ci95_high']),
            n_events=int(field(d,['n_paired_events','n_events'])),
            n_sequence_groups=d.get('n_sequence_groups_used',1 if scenario=='ridgecrest' else np.nan),
            n_target_rows=np.nan,bootstrap_unit=BOOTSTRAP[scenario],source=source))
    return pd.DataFrame(rows)


def select_caliper(df: pd.DataFrame, caliper: float, context: str) -> pd.DataFrame:
    require(df,['caliper_km'],context)
    mask=np.isclose(pd.to_numeric(df.caliper_km,errors='raise'),caliper)
    d=df.loc[mask].copy()
    if d.empty:
        raise ValueError(f'{context}: no rows for caliper {caliper:g} km.')
    return d


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    registry={}
    if args.verified_summary:
        return (pd.read_csv(io.StringIO(VERIFIED_CSV)),
                pd.read_csv(io.StringIO(VERIFIED_GAP_CSV)),
                {'input_mode':'verified_six_decimal_run_log_summary',
                 'warning':'Real logged estimates, not a rerun or a target-row pairing audit.'})
    tables=[]
    for scenario,path in [('grouped',args.grouped_bootstrap),('chronological',args.chronological_bootstrap),
                          ('ridgecrest',args.ridgecrest_bootstrap)]:
        tables.append(normalize(read_csv(path,registry),scenario,path))
    s=select_caliper(read_csv(args.station_deltas,registry),args.caliper_km,'station point estimates')
    b=select_caliper(read_csv(args.station_bootstrap,registry),args.caliper_km,'station bootstrap')
    require(s,['target_role','quantity','population','metric','base_value','ca_urc_value',
               'mean_delta_caurc_minus_base','n_paired_events'],'station point estimates')
    require(b,['comparison_type','group_or_model','quantity','population','metric',
               'point_delta','ci_lower','ci_upper','n_events'],'station bootstrap')
    s=s[s.target_role.eq('unseen_station_target') & s.population.isin(['overall','high_motion_tail'])
        & s.metric.isin(['mae','under05','factor2'])]
    b1=b[b.comparison_type.eq('caurc_minus_base') & b.group_or_model.eq('unseen_station_target')]
    keys=['quantity','population','metric']
    joined=s.merge(b1[keys+['point_delta','ci_lower','ci_upper','n_events']],on=keys,
                   how='left',validate='one_to_one')
    rows=[]
    for _,d in joined.iterrows():
        if not np.isfinite(d.ci_lower) or not np.isfinite(d.ci_upper):
            raise ValueError(f'Missing station interval for {tuple(d[k] for k in keys)}.')
        if not np.isclose(d.mean_delta_caurc_minus_base,d.point_delta,atol=2e-6,rtol=0):
            raise ValueError('Station estimate and bootstrap delta disagree.')
        if int(d.n_events)!=int(d.n_paired_events):
            raise ValueError('Station point and bootstrap event counts disagree.')
        rows.append(dict(scenario='unseen_station',quantity=d.quantity,population=d.population,metric=d.metric,
            base_value=d.base_value,ca_urc_value=d.ca_urc_value,delta=d.point_delta,
            ci_lower=d.ci_lower,ci_upper=d.ci_upper,n_events=int(d.n_events),
            n_sequence_groups=np.nan,n_target_rows=np.nan,bootstrap_unit='event',
            source=args.station_deltas+' | '+args.station_bootstrap))
    tables.append(pd.DataFrame(rows))
    gap=b[b.comparison_type.eq('unseen_minus_seen') & b.population.eq('overall') & b.metric.eq('mae')].copy()
    if len(gap)!=4:
        raise ValueError('Expected four overall station-gap rows (two models x two quantities).')
    registry['input_mode']='original_project_summary_csvs'
    return pd.concat(tables,ignore_index=True),gap,registry


def row_at(df: pd.DataFrame, scenario: str, q: str, pop: str, metric: str) -> pd.Series:
    d=df[(df.scenario==scenario)&(df.quantity==q)&(df.population==pop)&(df.metric==metric)]
    if len(d)!=1:
        raise ValueError(f'Expected one row for {scenario}/{q}/{pop}/{metric}; got {len(d)}.')
    return d.iloc[0]


def validate(df: pd.DataFrame, allow_different_run: bool) -> None:
    require(df,['scenario','quantity','population','metric','base_value','ca_urc_value','delta',
                'ci_lower','ci_upper','n_events','bootstrap_unit'],'normalized results')
    if df.duplicated(['scenario','quantity','population','metric']).any():
        raise ValueError('Duplicate comparisons; do not mix bootstrap schemes or matching calipers.')
    cols=['base_value','ca_urc_value','delta','ci_lower','ci_upper','n_events']
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise ValueError('Missing/nonfinite primary metrics.')
    if (df.ci_lower>df.ci_upper).any():
        raise ValueError('Reversed confidence limits.')
    if not np.allclose(df.ca_urc_value-df.base_value,df.delta,atol=2.1e-6,rtol=0):
        raise ValueError('Delta is inconsistent with CA-URC minus Base (beyond six-decimal rounding).')
    for scenario in SCENARIOS:
        for q in ['pga','pgv']:
            for pop in ['overall','high_motion_tail']:
                for metric in ['mae','under05','factor2']:
                    d=row_at(df,scenario,q,pop,metric)
                    if d.bootstrap_unit != BOOTSTRAP[scenario]:
                        raise ValueError(f'{scenario}: wrong resampling unit.')
                    if metric in ['under05','factor2'] and not (0<=d.base_value<=1 and 0<=d.ca_urc_value<=1):
                        raise ValueError('Rates must be fractions, not percentages.')
    if allow_different_run:
        return
    ref=pd.read_csv(io.StringIO(VERIFIED_CSV))
    keys=['scenario','quantity','population','metric']
    for _,d in df.iterrows():
        r=row_at(ref,d.scenario,d.quantity,d.population,d.metric)
        if int(d.n_events)!=int(r.n_events):
            raise ValueError(f'Locked event-count mismatch: {tuple(d[k] for k in keys)}.')
        # Permit Monte Carlo changes in CI, not a different locked point estimate.
        for c in ['base_value','ca_urc_value','delta']:
            if not np.isclose(d[c],r[c],atol=6e-6,rtol=0):
                raise ValueError(f'Locked value mismatch in {tuple(d[k] for k in keys)}, {c}: {d[c]} vs {r[c]}.')


def setup_fonts() -> None:
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,
        'ytick.labelsize':7.3,'legend.fontsize':7,'axes.linewidth':0.7,
        'xtick.major.width':0.7,'ytick.major.width':0.7,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})


def axis_figure(letter: str, title: str, height_mm: float=66.0):
    fig=plt.figure(figsize=(91.5/25.4,height_mm/25.4))
    ax=fig.add_axes([0.295,0.27,0.655,0.53])
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=5)
    ax.set_ylim(3.7,-0.85)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.text(.04,.94,letter,fontsize=11,fontweight='bold',va='top')
    fig.text(.14,.94,title,fontsize=9,va='top')
    return fig,ax


def save_panel(fig, path: Path, dpi: int) -> None:
    for fmt in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+fmt),dpi=dpi)
    plt.close(fig)


def plot_levels(df: pd.DataFrame, q: str, letter: str, out: Path, dpi: int) -> Path:
    fig,ax=axis_figure(letter,q.upper()+' | High-motion-tail MAE')
    rows=[row_at(df,s,q,'high_motion_tail','mae') for s in SCENARIOS]
    y=np.arange(4,dtype=float)
    base=np.array([d.base_value for d in rows]);final=np.array([d.ca_urc_value for d in rows])
    ax.scatter(base,y,marker='o',s=27,label='Cross-Attention Base',zorder=4)
    ax.scatter(final,y,marker='s',s=30,label='CA-URC',zorder=5)
    xs=[];ys=[]
    for i,d in enumerate(rows):
        xs.extend([d.base_value,d.ca_urc_value,np.nan]);ys.extend([i,i,np.nan])
        reduction=-100*d.delta/d.base_value
        ax.text((d.base_value+d.ca_urc_value)/2,i-.25,
            f'{d.base_value:.3f} → {d.ca_urc_value:.3f}  (−{reduction:.1f}%)',
            fontsize=7,ha='center',va='center')
    ax.plot(xs,ys,lw=1.2,alpha=.45,zorder=2)
    ax.set_xlim(.44,1.14);ax.set_xticks([.5,.7,.9,1.1])
    ax.set_yticks(y,[LABELS[s] for s in SCENARIOS])
    ax.set_xlabel(r'Tail MAE ($\log_{10}$ units)')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.015),
               ncol=2,frameon=False,handletextpad=.3,columnspacing=.8)
    path=out/f'Fig7{letter}_tail_mae_{q}';save_panel(fig,path,dpi);return path


def plot_effects(df: pd.DataFrame, q: str, metric: str, letter: str, out: Path, dpi: int) -> Path:
    rate=metric=='under05'; scale=100.0 if rate else 1.0
    title=q.upper()+(' | Tail underprediction change' if rate else ' | Paired tail-MAE change')
    fig,ax=axis_figure(letter,title)
    rows=[row_at(df,s,q,'high_motion_tail',metric) for s in SCENARIOS]
    y=np.arange(4,dtype=float)
    d=np.array([r.delta for r in rows])*scale
    lo=np.array([r.ci_lower for r in rows])*scale;hi=np.array([r.ci_upper for r in rows])*scale
    # Draw explicit interval segments; asymmetric intervals need not contain the point.
    ax.hlines(y,lo,hi,lw=1.4)
    ax.scatter(d,y,s=22,marker='o' if q=='pga' else 's',zorder=4)
    ax.plot([0,0],[-.6,3.5],linestyle='--',lw=.8,alpha=.55)
    labels=[f'{LABELS[s]}\n(n = {int(r.n_events)})' for s,r in zip(SCENARIOS,rows)]
    ax.set_yticks(y,labels)
    if rate:
        ax.set_xlim(-25,1.5);ax.set_xticks([-20,-10,0])
        ax.set_xlabel(r'$\Delta U_{0.5}$ (percentage points)')
    else:
        ax.set_xlim(-.125,.006);ax.set_xticks([-.12,-.08,-.04,0])
        ax.set_xlabel(r'$\Delta$ tail MAE ($\log_{10}$ units)')
    for i,r in enumerate(rows):
        text=f'{r.delta*scale:+.2f}' if rate else f'{r.delta:+.4f}'
        ax.text(r.delta*scale,i-.25,text,fontsize=7,ha='center',va='center')
    fig.text(.61,.04,'← Lower values favour CA-URC',ha='center',fontsize=7.1)
    path=out/f'Fig7{letter}_delta_{metric}_{q}';save_panel(fig,path,dpi);return path


def plot_gap(gap: pd.DataFrame,out: Path,dpi: int) -> list[Path]:
    paths=[]
    for q,letter in [('pga','a'),('pgv','b')]:
        fig,ax=axis_figure(letter,q.upper()+' | Residual station gap',height_mm=60)
        sub=gap[gap.quantity.eq(q)]
        models=['cross_attention_base','ca_urc']
        rows=[]
        for model in models:
            z=sub[sub.group_or_model.eq(model)]
            if len(z)!=1: raise ValueError(f'Station gap row missing: {q} {model}')
            rows.append(z.iloc[0])
        for i,r in enumerate(rows):
            ax.hlines(i,r.ci_lower,r.ci_upper,lw=1.5)
            ax.scatter(r.point_delta,i,marker='o' if i==0 else 's',s=30)
            ax.text(r.point_delta,i-.26,f'{r.point_delta:+.4f}',ha='center',fontsize=7.5)
        ax.plot([0,0],[-.6,1.5],linestyle='--',lw=.8,alpha=.5)
        ax.set_ylim(1.65,-.85);ax.set_xlim(-.006,.081);ax.set_xticks([0,.02,.04,.06,.08])
        ax.set_yticks([0,1],['Base','CA-URC'])
        ax.set_xlabel('Unseen − seen overall MAE')
        fig.text(.59,.055,'20-km matching; 195 paired events',ha='center',fontsize=7)
        path=out/f'FigS_station_gap_{q}';save_panel(fig,path,dpi);paths.append(path)
    return paths


def compose(panels: list[Path], dest: Path, rows: int, dpi: int, footer: str) -> None:
    """Assemble standalone panels at their native size; do not use subplots."""
    try:
        import pymupdf as fitz
    except ImportError:
        try: import fitz
        except ImportError as exc:
            raise ImportError('Install pymupdf for vector PDF composition: python -m pip install pymupdf') from exc
    docs=[fitz.open(str(p.with_suffix('.pdf'))) for p in panels]
    try:
        pw,ph=docs[0][0].rect.width,docs[0][0].rect.height
        footer_h=34
        doc=fitz.open();page=doc.new_page(width=2*pw,height=rows*ph+footer_h)
        for i,d in enumerate(docs):
            x=(i%2)*pw;y=(i//2)*ph
            page.show_pdf_page(fitz.Rect(x,y,x+pw,y+ph),d,0)
        page.insert_textbox(fitz.Rect(12,rows*ph+1,2*pw-12,rows*ph+footer_h-2),
                            footer,fontsize=6.8,fontname='helv',align=1)
        doc.save(str(dest.with_suffix('.pdf')),garbage=4,deflate=True)
        pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False)
        pix.save(str(dest.with_suffix('.png')))
        doc.close()
        # Preserve original Matplotlib vector SVG text/paths; prefix panel IDs.
        NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
        root=ET.Element('{'+NS+'}svg',{'width':f'{2*pw}pt','height':f'{rows*ph+footer_h}pt',
            'viewBox':f'0 0 {2*pw} {rows*ph+footer_h}','version':'1.1'})
        for i,p in enumerate(panels):
            element=ET.fromstring(p.with_suffix('.svg').read_text(encoding='utf-8'))
            mapping={e.get('id'):f'p{i}_{e.get("id")}' for e in element.iter() if e.get('id')}
            for e in element.iter():
                if e.get('id'): e.set('id',mapping[e.get('id')])
                for key,value in list(e.attrib.items()):
                    if key=='id': continue
                    for old,new in mapping.items():
                        value=value.replace(f'url(#{old})',f'url(#{new})')
                        if value==f'#{old}':value=f'#{new}'
                    e.set(key,value)
            group=ET.SubElement(root,'{'+NS+'}g',{'transform':f'translate({i%2*pw},{i//2*ph})'})
            for child in list(element):group.append(child)
        for i,line in enumerate(footer.splitlines()):
            t=ET.SubElement(root,'{'+NS+'}text',{'x':str(pw),'y':str(rows*ph+8+i*8),
                'text-anchor':'middle','font-family':'Arial, sans-serif','font-size':'7'})
            t.text=line
        ET.ElementTree(root).write(dest.with_suffix('.svg'),encoding='utf-8',xml_declaration=True)
    finally:
        for d in docs:d.close()


def write_tables(df: pd.DataFrame,gap: pd.DataFrame,out: Path) -> None:
    df=df.copy()
    df['relative_mae_reduction_percent']=np.where(df.metric.eq('mae'),-100*df.delta/df.base_value,np.nan)
    df['delta_percentage_points']=np.where(df.metric.isin(['under05','factor2']),100*df.delta,np.nan)
    df.to_csv(out/'TableS6_all_saved_effects.csv',index=False)
    pd.DataFrame(PROTOCOLS).to_csv(out/'TableS6_protocols.csv',index=False)
    gap.to_csv(out/'TableS6_residual_station_gap.csv',index=False)
    compact=[]
    for s in SCENARIOS:
        for q in ['pga','pgv']:
            t=row_at(df,s,q,'high_motion_tail','mae');u=row_at(df,s,q,'high_motion_tail','under05')
            a=row_at(df,s,q,'overall','mae')
            compact.append({'Protocol':LABELS[s],'Quantity':q.upper(),'Tail events':int(t.n_events),
                'Overall Base':a.base_value,'Overall CA-URC':a.ca_urc_value,
                'Tail Base':t.base_value,'Tail CA-URC':t.ca_urc_value,'Tail MAE delta':t.delta,
                'Tail MAE CI lower':t.ci_lower,'Tail MAE CI upper':t.ci_upper,
                'Tail MAE reduction (%)':-100*t.delta/t.base_value,'Tail U0.5 Base (%)':100*u.base_value,
                'Tail U0.5 CA-URC (%)':100*u.ca_urc_value,'Bootstrap unit':t.bootstrap_unit})
    table=pd.DataFrame(compact);table.to_csv(out/'TableS6_compact.csv',index=False)
    tex=[]
    for q in ['PGA','PGV']:
        d=table[table.Quantity.eq(q)]
        z=pd.DataFrame({'Protocol':d.Protocol,'Tail events':d['Tail events'],
            'Overall Base / CA-URC':[f'{a:.3f} / {b:.3f}' for a,b in zip(d['Overall Base'],d['Overall CA-URC'])],
            'Tail Base / CA-URC':[f'{a:.3f} / {b:.3f}' for a,b in zip(d['Tail Base'],d['Tail CA-URC'])],
            'Tail delta [95% CI]':[f'{x:+.4f} [{l:+.4f}, {h:+.4f}]' for x,l,h in zip(d['Tail MAE delta'],d['Tail MAE CI lower'],d['Tail MAE CI upper'])],
            'U0.5 Base / CA-URC (%)':[f'{a:.1f} / {b:.1f}' for a,b in zip(d['Tail U0.5 Base (%)'],d['Tail U0.5 CA-URC (%)'])]})
        tex.append('% '+q+'\n'+z.to_latex(index=False,escape=True,column_format='lrllll'))
    (out/'TableS6_compact.tex').write_text('\n'.join(tex),encoding='utf-8')
    lines=['Section 2.6: figures use saved results; no new statistical inference.','']
    for s in SCENARIOS:
        for q in ['pga','pgv']:
            r=row_at(df,s,q,'high_motion_tail','mae')
            lines.append(f'{LABELS[s]} {q.upper()}: {r.base_value:.6f} -> {r.ca_urc_value:.6f}; '
                f'delta {r.delta:+.6f}, CI [{r.ci_lower:+.6f}, {r.ci_upper:+.6f}]; '
                f'n={int(r.n_events)}; unit={r.bootstrap_unit}')
    (out/'Section2_6_verified_numbers.txt').write_text('\n'.join(lines),encoding='utf-8')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--verified-summary',action='store_true',help='Use the bundled REAL six-decimal run-log summaries.')
    p.add_argument('--grouped-bootstrap',default='runs/caurc_sequence_aware_bootstrap/hierarchical_bootstrap_summary.csv')
    p.add_argument('--chronological-bootstrap',default='runs/caurc_chronological_2021_2024/paired_event_bootstrap_2021_2024.csv')
    p.add_argument('--station-deltas',default='runs/caurc_station_ood_matched_geometry/caurc_minus_base_matched_by_role.csv')
    p.add_argument('--station-bootstrap',default='runs/caurc_station_ood_matched_geometry/matched_geometry_event_bootstrap.csv')
    p.add_argument('--ridgecrest-bootstrap',default='runs/caurc_ridgecrest_ood/paired_event_bootstrap_ridgecrest.csv')
    p.add_argument('--caliper-km',type=float,default=20.0)
    p.add_argument('--allow-different-run',action='store_true',help='Opt out of locked-point-value guards; not for reproducing this manuscript.')
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--out-dir',default='figures/nc_section_2_6')
    args=p.parse_args()
    if not np.isclose(args.caliper_km,20.0):
        p.error('This manuscript figure is defined at 20 km; other calipers belong in a separate sensitivity figure.')
    if args.dpi<72 or args.dpi>1200:p.error('--dpi must be between 72 and 1200.')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    df,gap,registry=load_inputs(args);validate(df,args.allow_different_run)
    print('Input mode:',registry['input_mode']);print('Saved-summary consistency audit: PASSED')
    print('No target-row pairing has been re-audited; that requires the original predictions.')
    setup_fonts();parts=out/'panels';parts.mkdir(exist_ok=True)
    panels=[]
    panels.append(plot_levels(df,'pga','a',parts,args.dpi))
    panels.append(plot_levels(df,'pgv','b',parts,args.dpi))
    panels.append(plot_effects(df,'pga','mae','c',parts,args.dpi))
    panels.append(plot_effects(df,'pgv','mae','d',parts,args.dpi))
    panels.append(plot_effects(df,'pga','under05','e',parts,args.dpi))
    panels.append(plot_effects(df,'pgv','under05','f',parts,args.dpi))
    footer=('95% CI units: grouped, sequence -> event; temporal / station, event; Ridgecrest, events in one sequence.\n'
            'Protocols have different evaluation populations; Ridgecrest is retrospective frozen-model transfer.')
    compose(panels,out/'Fig7_distribution_shift',3,args.dpi,footer)
    gp=plot_gap(gap,parts,args.dpi)
    compose(gp,out/'FigS_residual_station_gap',1,args.dpi,
            'Unseen - seen gap after matching; intervals are paired event-bootstrap 95% CIs, not sequence-level inference.')
    write_tables(df,gap,out)
    (out/'Fig7_caption.txt').write_text(CAPTION,encoding='utf-8')
    (out/'FigS_station_gap_caption.txt').write_text(
        'Residual unseen-minus-seen overall MAE after matching at a 20-km distance caliper and exact P-wave state. '
        'PGA and PGV are shown separately. Positive values indicate larger unseen-target errors. '
        'Intervals are 95% event-bootstrap intervals from 195 paired earthquakes. '
        'CA-URC retains tail gains but does not remove this station-domain gap.',encoding='utf-8')
    (out/'run_configuration.json').write_text(json.dumps({'arguments':vars(args),'inputs':registry,
        'resampling_units':BOOTSTRAP,'training_performed':False,'model_selection_performed':False,
        'bootstrap_rerun':False,'new_raw_target_pairing_audit':False,
        'audit_scope':'Saved estimate/count/schema consistency only','relative_reduction_CI':'Not computed from fixed reference denominators',
        'unseen_tail_row_counts':'Not in the inspected log; deliberately left missing'},indent=2),encoding='utf-8')
    print('Results saved to',out.resolve())

if __name__=='__main__':
    main()
