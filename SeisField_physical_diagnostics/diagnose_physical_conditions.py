#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only post-hoc physical-condition diagnostics for the completed PREFIX run.

No torch/ObsPy, model loading, inference, training, waveform preprocessing,
new test draws, target changes, or exclusion of failed metadata are performed.
Only numpy, pandas, h5py and the Python standard library are required.

Run from the user's project root. Defaults match the supplied completed run.
Default stage=audit. `--stage analyze` also writes event-balanced, paired
hierarchical group->event bootstrap statistics. All thresholds for amplitude
come from experiment_protocol.json; distance tertiles use TRAIN metadata only.
These are exploratory retrospective conditional associations, not causal effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

VERSION = 'prefix_physical_diagnostics_v1'
KEYS = ['event_id', 'repeat', 'target_slot', 'target_station_index']
QUANTITIES = ('pga', 'pgv')
METHODS = ('Prefix_Base', 'Prefix_CAURC')
METRICS = ('mae', 'bias', 'under05', 'factor2')
AXES = {
    'target_p_state': ['Pre-P', 'Post-P', 'Unknown/invalid P'],
    'target_ps_state': ['Pre-P', 'P-to-S', 'Post-S', 'Unknown/invalid PS'],
    'nearest_input_band': ['Near', 'Middle', 'Far'],
    'input_s_state': ['0 of K', '1-2 of K', '3+ of K', 'Unknown/partial S'],
}


def text(value: Any) -> str:
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe_json(value), ensure_ascii=False,
                              indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(path)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path: Path, required: tuple[str, ...] | list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={'event_id': str})
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if 'event_id' in df:
        if df.event_id.isna().any():
            raise ValueError(f'{path}: missing event IDs')
        df['event_id'] = df.event_id.str.strip()
    return df


def parse_indices(value: Any) -> tuple[int, ...]:
    # No eval(), no zero-/one-index conversion: exact archived integer indices.
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        raise ValueError('Missing input/target station-index vector')
    tokens = s.split('|')
    if any(not token.strip().isdigit() for token in tokens):
        raise ValueError(f'Expected pipe-separated non-negative integer indices: {s!r}')
    result = tuple(int(v) for v in tokens)
    if len(set(result)) != len(result):
        raise ValueError(f'Duplicate station indices: {result}')
    return result


def strict_bool(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(['true', 'false', '1', '0']).all():
        raise ValueError(f'Invalid boolean values in {series.name}')
    return values.isin(['true', '1']).to_numpy()


def check_keys(df: pd.DataFrame, label: str) -> None:
    for key in KEYS[1:]:
        v = pd.to_numeric(df[key], errors='raise')
        if v.isna().any() or not np.equal(v, np.floor(v)).all() or (v < 0).any():
            raise ValueError(f'{label}: invalid integer {key}')
        df[key] = v.astype(np.int64)
    if df.duplicated(KEYS).any():
        raise ValueError(f'{label}: duplicate keys')
    if df.duplicated(['event_id', 'repeat', 'target_station_index']).any():
        raise ValueError(f'{label}: duplicate target station within a draw')


def canonical_seed(label: str, seed: int) -> int:
    # Exact source63 text:seed convention; used ONLY for TRAIN metadata draws.
    d = hashlib.sha256(f'{label}:{seed}'.encode('utf-8')).digest()
    return int.from_bytes(d[:8], 'little') % (2**32)


def local_xy(coords: np.ndarray, origin_lat: float, origin_lon: float) -> np.ndarray:
    # The supplied source45 convention: lat/lon/elevation, in this order.
    x = (coords[:, 1] - origin_lon) * 111.32 * math.cos(math.radians(origin_lat))
    y = (coords[:, 0] - origin_lat) * 110.57
    return np.stack((x, y), axis=-1)


def distance_matrix(coords: np.ndarray, inputs: np.ndarray,
                    targets: np.ndarray) -> np.ndarray:
    origin_lat = float(coords[inputs, 0].mean())
    origin_lon = float(coords[inputs, 1].mean())
    a = local_xy(coords[inputs], origin_lat, origin_lon)
    b = local_xy(coords[targets], origin_lat, origin_lon)
    return np.sqrt(np.sum((b[:, None, :] - a[None, :, :])**2, axis=-1))


def phase_states(p: np.ndarray, s: np.ndarray, t0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_ok = np.isfinite(p) & (p >= -1e-3)
    ps_ok = p_ok & np.isfinite(s) & (s > p)
    p_state = np.full(p.shape, 'Unknown/invalid P', dtype=object)
    p_state[p_ok & (p > t0)] = 'Pre-P'
    p_state[p_ok & (p <= t0)] = 'Post-P'
    # All THREE detailed states use the SAME valid P/S-pair requirement.
    ps_state = np.full(p.shape, 'Unknown/invalid PS', dtype=object)
    ps_state[ps_ok & (p > t0)] = 'Pre-P'
    ps_state[ps_ok & (p <= t0) & (s > t0)] = 'P-to-S'
    ps_state[ps_ok & (s <= t0)] = 'Post-S'
    return p_state, ps_state, ps_ok


@dataclass
class Meta:
    path: Path
    coords: np.ndarray
    p: np.ndarray
    s: np.ndarray
    station_ids: np.ndarray
    attrs: dict
    fingerprint: str
    has_s_field: bool


def read_meta(row: pd.Series, h5_root: Path, expected_hash: str | None,
              verify_hash: bool, t0: float) -> Meta:
    eid = str(row.event_id)
    # Existing explicit manifest path first, configured root only if unavailable.
    original = Path(str(row.get('h5_path', '')))
    path = original if original.is_file() else h5_root / f'{eid}.h5'
    if not path.is_file():
        raise FileNotFoundError(f'Original HDF5 missing for {eid}: {path}')
    if verify_hash:
        if not expected_hash:
            raise ValueError(f'No archived source hash for {eid}')
        if digest_file(path) != expected_hash:
            raise ValueError(f'Original HDF5 hash differs from training protocol: {eid}')
    with h5py.File(path, 'r') as h:
        needed = {'station_coords', 'p_offset_sec', 'station_id', 'acceleration'}
        if not needed.issubset(h):
            raise ValueError(f'{eid}: missing HDF5 datasets {sorted(needed-set(h))}')
        if text(h.attrs.get('event_id', '')) != eid:
            raise ValueError(f'{eid}: HDF5 event_id mismatch')
        if not str(h.attrs.get('first_p_time', '')).strip():
            raise ValueError(f'{eid}: absent first-P time reference')
        coords = np.asarray(h['station_coords'][:], dtype=np.float32)
        p = np.asarray(h['p_offset_sec'][:], dtype=float)
        has_s = 's_offset_sec' in h
        s = np.asarray(h['s_offset_sec'][:], dtype=float) if has_s else np.full(p.shape, np.nan)
        ids = h['station_id'].asstr()[:]
        n = len(p)
        if coords.shape != (n, 3) or s.shape != (n,) or ids.shape != (n,):
            raise ValueError(f'{eid}: incompatible station array shapes')
        if h['acceleration'].shape[0] != n:
            raise ValueError(f'{eid}: station count mismatch')
        if not np.isfinite(coords).all() or (np.abs(coords[:, 0]) > 90).any() or (np.abs(coords[:, 1]) > 180).any():
            raise ValueError(f'{eid}: invalid lat/lon/elevation metadata')
        attrs = {k: h.attrs.get(k, np.nan) for k in
                 ('magnitude', 'latitude', 'longitude', 'depth_km', 'first_p_time')}
    digest = hashlib.sha256()
    digest.update(eid.encode())
    digest.update(text(attrs['first_p_time']).encode())
    for arr in (coords, p, s):
        digest.update(np.ascontiguousarray(arr).tobytes())
    digest.update('|'.join(map(str, ids)).encode())
    return Meta(path, coords, p, s, ids, attrs, digest.hexdigest(), has_s)


def fit_distance_bins(cohort: pd.DataFrame, split_col: str, settings: dict,
                      metadata, repeats: int) -> tuple[dict, pd.DataFrame]:
    train = cohort.loc[cohort[split_col].eq('train')]
    if train.empty:
        raise ValueError('No training events for training-only distance bins')
    k, q = int(settings['input_stations']), int(settings['target_stations'])
    t0, seed = float(settings['t0_sec']), int(settings['seed'])
    rows = []
    for i, (_, event) in enumerate(train.iterrows(), 1):
        m = metadata(event)
        eligible = np.flatnonzero(np.isfinite(m.p) & (m.p >= -1e-3) & (m.p <= t0))
        if len(eligible) < k or len(m.p) - k < q:
            raise ValueError(f'{event.event_id}: training metadata cannot reproduce K/Q rule')
        for repeat in range(repeats):
            rng = np.random.default_rng(canonical_seed(f'train:{event.event_id}:repeat:{repeat}', seed))
            inputs = rng.choice(eligible, size=k, replace=False)
            targets = rng.choice(np.setdiff1d(np.arange(len(m.p)), inputs), size=q, replace=False)
            distances = distance_matrix(m.coords, inputs, targets).min(axis=1)
            for slot, (idx, dist) in enumerate(zip(targets, distances)):
                rows.append({'event_id': event.event_id, 'repeat': repeat, 'target_slot': slot,
                             'target_station_index': int(idx), 'nearest_input_km': float(dist)})
        if i % 200 == 0 or i == len(train):
            print(f'Training geometry {i}/{len(train)} (metadata only)', flush=True)
    frame = pd.DataFrame(rows)
    edges = np.quantile(frame.nearest_input_km.to_numpy(float), [1/3, 2/3])
    if not np.isfinite(edges).all() or edges[0] <= 0 or edges[1] <= edges[0]:
        raise ValueError('Training distances do not support distinct tertile bins; do not tune using test errors')
    config = {'distance_internal_edges_km': edges.tolist(),
              'labels': ['Near', 'Middle', 'Far'],
              'intervals': ['[0,q1)', '[q1,q2)', '[q2,+inf)'],
              'fit_population': 'retained TRAIN events only; no predictions or amplitudes read',
              'train_events': len(train), 'train_target_draws': len(frame), 'repeats_per_event': repeats,
              'draw_protocol': 'source63 SHA256(train:event_id:repeat:r:seed); input then target draw',
              'weighting': 'equal events/repeats/targets; K,Q and repeats fixed across events'}
    return config, frame


def check_predictions(pred: pd.DataFrame, draws: pd.DataFrame, truth: pd.DataFrame,
                      cohort: pd.DataFrame, settings: dict, thresholds: np.ndarray,
                      evaluation: dict) -> tuple[pd.DataFrame, dict]:
    check_keys(pred, 'predictions')
    check_keys(truth, 'checked_test_keys_and_truth')
    split_col = settings['split_column']
    expected = set(cohort.loc[cohort[split_col].eq('test'), 'event_id'])
    if set(pred.event_id) != expected or set(truth.event_id) != expected:
        raise ValueError('Prediction/truth event set differs from retained TEST cohort')
    k, q, nr = int(settings['input_stations']), int(settings['target_stations']), int(settings['test_repeats'])
    if len(pred) != len(expected) * nr * q or len(pred) != int(evaluation['target_rows']):
        raise ValueError('Prediction row count does not match archived cohort/draw count')
    if len(expected) != int(evaluation['test_events']):
        raise ValueError('Test event count differs from evaluation audit')
    a = pd.MultiIndex.from_frame(pred[KEYS])
    b = pd.MultiIndex.from_frame(truth[KEYS])
    if len(a.difference(b)) or len(b.difference(a)):
        raise ValueError('Prediction keys differ from saved checked truth keys')
    aligned = truth.set_index(KEYS).reindex(a)
    max_truth_diff = 0.0
    for j, quant in enumerate(QUANTITIES):
        target = pd.to_numeric(pred[f'true_log10_{quant}'], errors='raise').to_numpy(float)
        if not np.isfinite(target).all():
            raise ValueError(f'Non-finite {quant} truth')
        diff = float(np.max(np.abs(target - aligned[f'true_log10_{quant}'].to_numpy(float))))
        max_truth_diff = max(max_truth_diff, diff)
        if diff > 1e-5:
            raise ValueError(f'{quant}: prediction/truth audit mismatch {diff}')
        tail = strict_bool(pred[f'is_tail_{quant}'])
        if not np.array_equal(tail, target >= thresholds[j]):
            raise ValueError(f'{quant}: frozen training tail threshold does not match prediction flags')
        pred[f'is_tail_{quant}'] = tail
        for method in METHODS:
            col = f'{method}_log10_{quant}'
            pred[col] = pd.to_numeric(pred[col], errors='raise')
            if not np.isfinite(pred[col].to_numpy(float)).all():
                raise ValueError(f'Non-finite {col}; refusing row deletion')
        delta = pred[f'Prefix_CAURC_log10_{quant}'] - pred[f'Prefix_Base_log10_{quant}']
        if (delta < -1e-6).any() or (delta > float(settings['maximum_correction']) + 1e-6).any():
            raise ValueError(f'{quant}: correction violates recorded one-sided bound')
        cols = [f'underprediction_risk_score_{quant}', f'raw_correction_amplitude_{quant}', f'applied_correction_{quant}']
        if all(c in pred for c in cols):
            risk, raw, applied = (pd.to_numeric(pred[c], errors='raise').to_numpy(float) for c in cols)
            if not all(np.isfinite(v).all() for v in (risk, raw, applied)):
                raise ValueError('Nonfinite risk/correction arrays')
            if ((risk < 0) | (risk > 1)).any():
                raise ValueError('Risk score outside [0,1]')
            if not np.allclose(applied, delta, atol=1e-6, rtol=0):
                raise ValueError('Saved correction differs from final minus base')
            if not np.allclose(applied, risk**float(evaluation['selected_gamma'])*raw, atol=1e-6, rtol=1e-6):
                raise ValueError('Saved risk gate and correction amplitudes do not reproduce applied correction')
    if draws.duplicated(['event_id', 'repeat']).any():
        raise ValueError('Duplicate saved draw rows')
    draws = draws.copy()
    draws['repeat'] = pd.to_numeric(draws['repeat'], errors='raise').astype(int)
    lookup = draws.set_index(['event_id', 'repeat'])
    if set(lookup.index) != set(zip(pred.event_id, pred['repeat'])):
        raise ValueError('Saved draw table and predictions have different event/repeat pairs')
    if 'input_station_indices' not in pred:
        pred = pred.merge(draws[['event_id', 'repeat', 'input_station_indices']],
                          on=['event_id', 'repeat'], how='left', validate='many_to_one')
    for (eid, rep), part in pred.groupby(['event_id', 'repeat'], sort=False):
        saved = lookup.loc[(eid, rep)]
        inputs, targets = parse_indices(saved.input_station_indices), parse_indices(saved.target_station_indices)
        actual = tuple(part.sort_values('target_slot').target_station_index.astype(int))
        if len(inputs) != k or len(targets) != q or set(inputs) & set(targets):
            raise ValueError(f'{eid}/{rep}: invalid saved input/target set')
        if set(part.target_slot) != set(range(q)) or actual != targets:
            raise ValueError(f'{eid}/{rep}: targets differ from the saved ordered draw')
        if any(parse_indices(v) != inputs for v in part.input_station_indices):
            raise ValueError(f'{eid}/{rep}: input indices differ from current-run audit draw')
    for eid, part in pred.groupby('event_id', sort=False):
        if set(part['repeat']) != set(range(nr)):
            raise ValueError(f'{eid}: incomplete repeat range')
    return pred, {'test_events': len(expected), 'prediction_rows': len(pred),
                  'saved_current_run_input_and_target_draws_match': True,
                  'max_abs_truth_diff_to_current_run_audit': max_truth_diff,
                  'original_historical_input_index_comparison': evaluation.get('pairing_audit', {}).get('input_indices_verified_against_archive'),
                  'tail_flags_match_frozen_training_thresholds': True}


def enrich_predictions(pred: pd.DataFrame, cohort: pd.DataFrame, settings: dict,
                       bins: dict, group_col: str, metadata) -> pd.DataFrame:
    t0 = float(settings['t0_sec'])
    lookup = cohort.set_index('event_id', drop=False)
    rows = []
    for ei, (eid, event_rows) in enumerate(pred.groupby('event_id', sort=False), 1):
        event = lookup.loc[eid]
        m = metadata(event)
        p_state, ps_state, ps_ok = phase_states(m.p, m.s, t0)
        p_valid = np.isfinite(m.p) & (m.p >= -1e-3)
        for rep, part in event_rows.groupby('repeat', sort=False):
            inputs = np.array(parse_indices(part.input_station_indices.iloc[0]), dtype=int)
            targets = part.target_station_index.to_numpy(int)
            if min(inputs.min(), targets.min()) < 0 or max(inputs.max(), targets.max()) >= len(m.p):
                raise IndexError(f'{eid}/{rep}: station index exceeds original archive')
            if not np.all(p_valid[inputs] & (m.p[inputs] <= t0)):
                raise ValueError(f'{eid}/{rep}: saved inputs not eligible at snapshot')
            distances = distance_matrix(m.coords, inputs, targets)
            n_s_known = int(ps_ok[inputs].sum())
            # Inputs exclude t_s, so require positive post-S duration in the prefix.
            n_s_arrived = int((ps_ok[inputs] & (m.s[inputs] < t0)).sum())
            n_inputs = len(inputs)
            if n_s_known < n_inputs:
                input_s_state = 'Unknown/partial S'
            elif n_s_arrived == 0:
                input_s_state = '0 of K'
            elif n_s_arrived <= 2:
                input_s_state = '1-2 of K'
            else:
                input_s_state = '3+ of K'
            for ri, (_, r) in enumerate(part.iterrows()):
                idx = int(r.target_station_index)
                record = {key: r[key] for key in KEYS}
                record.update({
                    'station_id': str(m.station_ids[idx]), 'group_id': str(event[group_col]),
                    't0_sec': t0, 'p_offset_sec': float(m.p[idx]), 's_offset_sec': float(m.s[idx]),
                    'p_lead_sec': float(m.p[idx] - t0) if p_valid[idx] else np.nan,
                    's_lead_sec': float(m.s[idx] - t0) if ps_ok[idx] else np.nan,
                    'target_p_state': p_state[idx], 'target_ps_state': ps_state[idx],
                    'valid_p_pick': bool(p_valid[idx]), 'valid_ps_pair': bool(ps_ok[idx]),
                    'nearest_input_km': float(distances[ri].min()),
                    'mean_input_distance_km': float(distances[ri].mean()),
                    'input_s_known_n': n_s_known, 'input_s_arrived_confirmed_n': n_s_arrived,
                    'input_s_unknown_n': n_inputs - n_s_known, 'input_s_state': input_s_state,
                    'mean_input_post_p_duration_sec': float(np.mean(t0 - m.p[inputs])),
                    'magnitude': float(event.get('magnitude', m.attrs['magnitude'])),
                    'depth_km': float(event.get('depth_km', m.attrs['depth_km'])),
                })
                rows.append(record)
        if ei % 50 == 0 or ei == pred.event_id.nunique():
            print(f'Test metadata join {ei}/{pred.event_id.nunique()}', flush=True)
    meta_frame = pd.DataFrame(rows)
    # One-to-one target key merge; no join on row order alone.
    frame = pred.merge(meta_frame, on=KEYS, how='left', validate='one_to_one')
    if len(frame) != len(pred) or frame.station_id.isna().any():
        raise ValueError('Metadata join lost/duplicated target rows')
    cut = np.asarray(bins['distance_internal_edges_km'])
    frame['nearest_input_band'] = np.asarray(['Near', 'Middle', 'Far'], dtype=object)[
        np.searchsorted(cut, frame.nearest_input_km.to_numpy(float), side='right')]
    for q in QUANTITIES:
        base = frame[f'Prefix_Base_log10_{q}'].to_numpy(float) - frame[f'true_log10_{q}'].to_numpy(float)
        final = frame[f'Prefix_CAURC_log10_{q}'].to_numpy(float) - frame[f'true_log10_{q}'].to_numpy(float)
        frame[f'base_residual_{q}'], frame[f'caurc_residual_{q}'] = base, final
        frame[f'absolute_error_change_{q}'] = np.abs(final) - np.abs(base)
        frame[f'base_under05_{q}'], frame[f'caurc_under05_{q}'] = base <= -0.5, final <= -0.5
    return frame


def event_metric(frame: pd.DataFrame, values: np.ndarray, mask: np.ndarray) -> pd.Series:
    if not np.asarray(mask).any():
        return pd.Series(dtype=float)
    work = frame.loc[mask, ['event_id', 'repeat']].copy()
    work['value'] = np.asarray(values)[mask]
    return work.groupby(['event_id', 'repeat'], sort=False)['value'].mean().groupby('event_id', sort=False).mean()


def metric_values(frame: pd.DataFrame, method: str, q: str, metric: str) -> np.ndarray:
    residual = frame[f'{method}_log10_{q}'].to_numpy(float) - frame[f'true_log10_{q}'].to_numpy(float)
    if metric == 'mae':
        return np.abs(residual)
    if metric == 'bias':
        return residual
    if metric == 'under05':
        return (residual <= -0.5).astype(float)
    if metric == 'factor2':
        return (np.abs(residual) <= np.log10(2.0)).astype(float)
    raise ValueError(metric)


def population_mask(frame: pd.DataFrame, q: str, pop: str) -> np.ndarray:
    if pop == 'overall':
        return np.ones(len(frame), dtype=bool)
    tail = frame[f'is_tail_{q}'].to_numpy(bool)
    return tail if pop == 'high_motion_tail' else ~tail


def overall_reconciliation(frame: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for q in QUANTITIES:
            for pop in ('overall', 'non_tail', 'high_motion_tail'):
                mask = population_mask(frame, q, pop)
                for metric in METRICS:
                    ev = event_metric(frame, metric_values(frame, method, q, metric), mask)
                    reference = expected.loc[(expected.method == method) & (expected.quantity == q)
                                             & (expected.population == pop) & (expected.metric == metric)]
                    if len(reference) != 1:
                        raise ValueError(f'Expected exactly one archived metric row: {method,q,pop,metric}')
                    r = reference.iloc[0]
                    actual = float(ev.mean())
                    rows.append({'method': method, 'quantity': q, 'population': pop, 'metric': metric,
                                 'recomputed': actual, 'archived': float(r.value),
                                 'absolute_difference': abs(actual-float(r.value))})
                    if not np.isclose(actual, float(r.value), rtol=0, atol=1e-8):
                        raise ValueError(f'Overall metric reconciliation failed: {method,q,pop,metric}')
                    part = frame.loc[mask]
                    counts = {'n_events': len(ev), 'n_target_rows': len(part),
                              'n_event_repeats': len(part[['event_id', 'repeat']].drop_duplicates()),
                              'n_groups': part.group_id.nunique()}
                    for name, count in counts.items():
                        if name in r and int(r[name]) != count:
                            raise ValueError(f'Archived metric {name} mismatch: {method,q,pop}: {count} vs {r[name]}')
    return pd.DataFrame(rows)


def coverage_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage, counts = [], []
    unique = frame.drop_duplicates(['event_id', 'target_station_index'])
    for label, data in [('target_prediction_rows', frame), ('unique_event_station_pairs', unique)]:
        coverage.append({'basis': label, 'n': len(data),
                         'p_valid_n': int(data.valid_p_pick.sum()),
                         'ps_pair_valid_n': int(data.valid_ps_pair.sum()),
                         'p_unknown_or_invalid_n': int((~data.valid_p_pick).sum()),
                         'ps_unknown_or_invalid_n': int((~data.valid_ps_pair).sum())})
    for axis, levels in AXES.items():
        for level in levels:
            for q in QUANTITIES:
                for pop in ('overall', 'non_tail', 'high_motion_tail'):
                    mask = (frame[axis] == level).to_numpy() & population_mask(frame, q, pop)
                    part = frame.loc[mask]
                    counts.append({'axis': axis, 'level': level, 'quantity': q, 'population': pop,
                                   'n_target_rows': len(part), 'n_events': part.event_id.nunique(),
                                   'n_groups': part.group_id.nunique(),
                                   'n_event_repeats': len(part[['event_id','repeat']].drop_duplicates()),
                                   'n_unique_event_station_pairs': len(part[['event_id','target_station_index']].drop_duplicates()),
                                   'event_magnitude_median': part.drop_duplicates('event_id').magnitude.median() if len(part) else np.nan,
                                   'nearest_input_km_row_median': part.nearest_input_km.median() if len(part) else np.nan,
                                   'observed_log10_row_median': part[f'true_log10_{q}'].median() if len(part) else np.nan})
    return pd.DataFrame(coverage), pd.DataFrame(counts)


def bootstrap_weights(events: list[str], groups_by_event: pd.Series,
                      draws: int, seed: int) -> np.ndarray:
    # Same distribution as the completed run's group->event bootstrap.
    labels = groups_by_event.reindex(events).astype(str).to_numpy()
    groups = list(dict.fromkeys(labels))
    if len(groups) < 2:
        raise ValueError('Need at least two original groups for group-bootstrap')
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=draws)
    weights = np.zeros((draws, len(events)), dtype=float)
    for gi, group in enumerate(groups):
        idx = np.flatnonzero(labels == group)
        n = len(idx)
        for m in np.unique(counts[:, gi]):
            if m == 0:
                continue
            rows = np.flatnonzero(counts[:, gi] == m)
            w = rng.multinomial(n*int(m), np.full(n, 1/n), size=len(rows))
            weights[np.ix_(rows, idx)] = w
    return weights


def ci_for_series(series: pd.Series, events: list[str], groups: pd.Series,
                  weights: np.ndarray, min_events: int, min_groups: int) -> dict:
    arr = series.reindex(events).to_numpy(float)
    valid = np.isfinite(arr)
    n = int(valid.sum())
    ng = int(groups.reindex(events).loc[valid].nunique()) if n else 0
    result = {'ci95_low': np.nan, 'ci95_high': np.nan,
              'bootstrap_valid_replicates': 0, 'inference_status': 'empty'}
    if n == 0:
        return result
    if n < min_events or ng < min_groups:
        result['inference_status'] = 'descriptive_only_sparse'
        return result
    den = weights @ valid.astype(float)
    num = weights @ np.nan_to_num(arr, nan=0.0)
    ok = den > 0
    samples = num[ok] / den[ok]
    result['bootstrap_valid_replicates'] = int(ok.sum())
    if ok.mean() < 0.99:
        result['inference_status'] = 'descriptive_only_many_empty_replicates'
        return result
    lo, hi = np.quantile(samples, [0.025, 0.975])
    result.update(ci95_low=float(lo), ci95_high=float(hi), inference_status='pointwise_exploratory')
    return result


def analyze(frame: pd.DataFrame, cohort: pd.DataFrame, settings: dict, group_col: str,
            counts: pd.DataFrame, out: Path, args) -> None:
    events = cohort.loc[cohort[settings['split_column']].eq('test'), 'event_id'].tolist()
    groups = cohort.set_index('event_id')[group_col].astype(str)
    weights = bootstrap_weights(events, groups, args.bootstrap_replicates, int(settings['seed']) + 77)
    values = {(m,q,metric): metric_values(frame, m, q, metric)
              for m in METHODS for q in QUANTITIES for metric in METRICS}
    rows, event_rows, stored = [], [], {}
    for row_no, c in enumerate(counts.to_dict('records'), 1):
        axis, level, q, pop = (c[k] for k in ('axis','level','quantity','population'))
        mask = (frame[axis] == level).to_numpy() & population_mask(frame, q, pop)
        for metric in METRICS:
            a = event_metric(frame, values[('Prefix_Base',q,metric)], mask)
            b = event_metric(frame, values[('Prefix_CAURC',q,metric)], mask)
            delta = b-a
            stored[(axis,level,q,pop,metric,'base')] = a
            stored[(axis,level,q,pop,metric,'delta')] = delta
            ci = ci_for_series(delta, events, groups, weights, args.min_events_ci, args.min_groups_ci)
            row = {**c, 'metric': metric, 'base_value': float(a.mean()),
                   'caurc_value': float(b.mean()), 'delta_caurc_minus_base': float(delta.mean()),
                   **ci, 'rate_unit': 'fraction' if metric in ('under05','factor2') else 'log10',
                   'relative_mae_reduction_pct': (-100*float(delta.mean())/float(a.mean()))
                   if metric == 'mae' and len(a) and a.mean() > 0 else np.nan}
            rows.append(row)
            for eid in a.index:
                event_rows.append({'axis':axis,'level':level,'quantity':q,'population':pop,'metric':metric,
                                   'event_id':eid,'group_id':groups.loc[eid],
                                   'base_value':a.loc[eid],'caurc_value':b.loc[eid],
                                   'delta_caurc_minus_base':delta.loc[eid]})
        if row_no % 18 == 0 or row_no == len(counts):
            print(f'Stratified statistics {row_no}/{len(counts)}', flush=True)
    stats = pd.DataFrame(rows)
    write_csv(out/'physical_stratified_metrics.csv', stats)
    write_csv(out/'physical_event_metrics.csv', pd.DataFrame(event_rows))
    # Direct between-condition contrasts: only earthquakes represented in BOTH
    # conditions, using the SAME bootstrap weights. Not a causal identification.
    pairs = [('target_p_state','Pre-P','Post-P'),
             ('target_ps_state','P-to-S','Post-S'),
             ('nearest_input_band','Far','Near')]
    contrasts = []
    for axis, left, right in pairs:
        for q in QUANTITIES:
            for pop in ('overall','high_motion_tail'):
                for metric, kind in [('under05','base'), ('bias','base'), ('mae','delta')]:
                    a = stored[(axis,left,q,pop,metric,kind)]
                    b = stored[(axis,right,q,pop,metric,kind)]
                    common = a.index.intersection(b.index)
                    diff = a.loc[common]-b.loc[common]
                    contrasts.append({'axis':axis,'left_level':left,'right_level':right,
                                      'contrast':'left_minus_right','quantity':q,'population':pop,
                                      'metric':metric,'kind':kind,'n_common_events':len(common),
                                      'n_common_groups':groups.reindex(common).nunique(),
                                      'left_mean_common_events':a.loc[common].mean(),
                                      'right_mean_common_events':b.loc[common].mean(),
                                      'mean_difference':diff.mean(),
                                      **ci_for_series(diff,events,groups,weights,args.min_events_ci,args.min_groups_ci),
                                      'rate_unit':'fraction' if metric=='under05' else 'log10'})
    write_csv(out/'physical_common_event_contrasts.csv', pd.DataFrame(contrasts))
    focus = stats.loc[stats.population.eq('high_motion_tail') & stats.metric.isin(['mae','under05'])
                      & stats.axis.isin(['target_p_state','target_ps_state','nearest_input_band'])].copy()
    # Export a presentation-ready table with explicitly labelled percentage units.
    for col in ('base_value','caurc_value','delta_caurc_minus_base','ci95_low','ci95_high'):
        focus[col+'_display'] = np.where(focus.metric.eq('under05'),100*focus[col],focus[col])
    focus['display_unit'] = np.where(focus.metric.eq('under05'), 'percent; changes in percentage points','log10')
    write_csv(out/'physical_primary_summary.csv', focus)
    print('\nPrimary tail-MAE conditional changes (negative = improvement):')
    print(focus.loc[focus.metric.eq('mae'), ['axis','level','quantity','base_value','caurc_value',
          'delta_caurc_minus_base','ci95_low','ci95_high','n_events','n_groups','inference_status']].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit','analyze'], default='audit')
    parser.add_argument('--run-dir', default='runs/snapshot_available_validation_common1603')
    parser.add_argument('--evaluation-subdir', default='evaluation_grouped')
    parser.add_argument('--h5-root', default='data/scedc/processed_full_v4/events')
    parser.add_argument('--out-dir', default='runs/physical_diagnostics_prefix')
    parser.add_argument('--train-geometry-repeats', type=int, default=3)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--min-events-ci', type=int, default=10,
                        help='Pragmatic reporting guard, NOT a universal sufficiency threshold')
    parser.add_argument('--min-groups-ci', type=int, default=5,
                        help='Pragmatic reporting guard; small groups remain descriptive')
    parser.add_argument('--verify-source-hash', action='store_true',
                        help='Rehash entire original HDF5 against training protocol; extra disk I/O, no array preprocessing')
    args = parser.parse_args()
    if args.train_geometry_repeats < 1 or args.bootstrap_replicates < 100:
        raise ValueError('Use positive training metadata repeats and >=100 bootstrap draws')
    if args.min_events_ci < 2 or args.min_groups_ci < 2:
        raise ValueError('Inferential coverage guards must both be >=2')
    run = Path(args.run_dir).resolve()
    out = Path(args.out_dir).resolve()
    if out == run or run in out.parents:
        raise ValueError('Use a separate output directory outside the completed training run')
    src = {'protocol':run/'experiment_protocol.json',
           'cohort':run/'audit/cohort_manifest.csv',
           'draws':run/'audit/checked_test_station_draws.csv',
           'truth':run/'audit/checked_test_keys_and_truth.csv',
           'audit':run/args.evaluation_subdir/'evaluation_audit.json',
           'metrics':run/args.evaluation_subdir/'paired_metrics.csv',
           'predictions':run/args.evaluation_subdir/'locked_prefix_test_predictions.csv'}
    for p in src.values():
        if not p.is_file():
            raise FileNotFoundError(f'Required completed-run file missing: {p}')
    protocol, evaluation = read_json(src['protocol']), read_json(src['audit'])
    if protocol['protocol_id'] != evaluation['snapshot_protocol_id']:
        raise ValueError('Evaluation and training protocol IDs differ')
    settings = protocol['settings']
    split_col, group_col = settings['split_column'], evaluation.get('group_column')
    if not group_col or group_col == split_col:
        raise ValueError('Evaluation must identify an actual group column, not the split label')
    cohort = read_csv(src['cohort'], ['event_id','h5_path',split_col,group_col])
    if cohort.event_id.duplicated().any() or cohort[group_col].isna().any():
        raise ValueError('Duplicated cohort event IDs or missing group IDs')
    cohort[group_col] = cohort[group_col].astype(str).str.strip()
    if cohort[group_col].eq('').any():
        raise ValueError('Blank group ID')
    content = protocol['event_order_and_content']
    saved_splits = {str(r['event_id']): str(r['split']) for r in content}
    if dict(zip(cohort.event_id, cohort[split_col].astype(str))) != saved_splits:
        raise ValueError('Retained cohort/splits differ from completed training protocol')
    hashes = {str(r['event_id']): r.get('source_sha256') for r in content}
    cfg = {'version':VERSION,'snapshot_protocol_id':protocol['protocol_id'],
           'source_hashes':{k:digest_file(v) for k,v in src.items()},
           'script_sha256':digest_file(Path(__file__)),
           'train_geometry_repeats':args.train_geometry_repeats,
           'bootstrap_replicates':args.bootstrap_replicates,
           'min_events_ci':args.min_events_ci,'min_groups_ci':args.min_groups_ci,
           'source_h5_rehash_requested':args.verify_source_hash}
    config_id = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    cfg['diagnostic_config_id'] = config_id
    cfg_path = out/'diagnostic_configuration.json'
    if cfg_path.is_file():
        if read_json(cfg_path).get('diagnostic_config_id') != config_id:
            raise ValueError('Diagnostic configuration changed. Use a new --out-dir; no result-dependent overwrite')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('Nonempty output has no matching configuration; use a new directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(cfg_path,cfg)
    cache: dict[str,Meta] = {}
    def metadata(row):
        eid = str(row.event_id)
        if eid not in cache:
            cache[eid] = read_meta(row,Path(args.h5_root),hashes.get(eid),args.verify_source_hash,
                                  float(settings['t0_sec']))
        return cache[eid]
    # Train-only bin selection precedes reading test prediction VALUES.
    bins, train_geometry = fit_distance_bins(cohort,split_col,settings,metadata,args.train_geometry_repeats)
    write_json(out/'distance_bins_train_only.json',bins)
    write_csv(out/'training_geometry_source.csv',train_geometry)
    print('Distance tertile edges (TRAIN metadata only):', bins['distance_internal_edges_km'])
    pred = read_csv(src['predictions'], KEYS + [f'true_log10_{q}' for q in QUANTITIES]
                    + [f'is_tail_{q}' for q in QUANTITIES]
                    + [f'{m}_log10_{q}' for m in METHODS for q in QUANTITIES])
    draws = read_csv(src['draws'], ['event_id','repeat','input_station_indices','target_station_indices'])
    truth = read_csv(src['truth'], KEYS + ['true_log10_pga','true_log10_pgv'])
    pred, pairing = check_predictions(pred,draws,truth,cohort,settings,np.array(protocol['thresholds']),evaluation)
    enriched = enrich_predictions(pred,cohort,settings,bins,group_col,metadata)
    expected = read_csv(src['metrics'], ['method','quantity','population','metric','value'])
    reconciled = overall_reconciliation(enriched,expected)
    coverage, counts = coverage_tables(enriched)
    meta_rows = [{'event_id':eid,'path':str(m.path),'metadata_sha256':m.fingerprint,'has_s_offset_field':m.has_s_field}
                 for eid,m in cache.items()]
    old_fp = out/'metadata_fingerprints.csv'
    if old_fp.is_file():
        old = read_csv(old_fp,['event_id','metadata_sha256']).set_index('event_id').metadata_sha256.to_dict()
        now = {r['event_id']:r['metadata_sha256'] for r in meta_rows}
        if old != now:
            raise ValueError('Metadata changed since diagnostic audit; use a new output and investigate')
    write_csv(old_fp,pd.DataFrame(meta_rows))
    write_csv(out/'physical_target_diagnostics.csv',enriched)
    write_csv(out/'physical_arrival_coverage.csv',coverage)
    write_csv(out/'physical_stratum_counts.csv',counts)
    write_csv(out/'overall_metric_reconciliation.csv',reconciled)
    report = {'diagnostic_config_id':config_id, 'snapshot_protocol_id':protocol['protocol_id'],
              'status':'metadata_pairing_and_global_metrics_PASS','pairing':pairing,
              'test_events':enriched.event_id.nunique(),'target_rows':len(enriched),
              'test_groups':enriched.group_id.nunique(),'t0_sec':settings['t0_sec'],
              'distance_bins':bins,'source_h5_rehash_performed':args.verify_source_hash,
              'max_global_metric_reconciliation_error':reconciled.absolute_difference.max(),
              'missing_p_and_s_rows_kept_as_unknown':True,'no_model_loading_training_or_inference':True,
              'no_waveform_arrays_read':True,'data_sources':'saved predictions/draws, original HDF5 metadata, retained manifest',
              'scope':'Exploratory retrospective stratification; archived arrivals; no causal attribution',
              'ci_scope':'pointwise hierarchical group->event; no multiple-testing or train-seed uncertainty claim',
              'sparsity_guards':'reporting guards, not proof of adequacy',
              'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'h5py':h5py.__version__}}
    write_json(out/'physical_diagnostics_audit.json',report)
    print('\nMetadata pairing and global metric reconciliation PASS.')
    print(coverage.to_string(index=False))
    if args.stage == 'audit':
        print('Audit only: no stratified outcome tables or bootstrap generated. No model was run.')
        print('Next: same command with --stage analyze. Output:',out)
    else:
        analyze(enriched,cohort,settings,group_col,counts,out,args)
        report['status']='analysis_complete'
        write_json(out/'physical_diagnostics_audit.json',report)
        print('\nPhysical-condition diagnostics complete. No retraining/inference. Output:',out)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, IndexError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(2)
