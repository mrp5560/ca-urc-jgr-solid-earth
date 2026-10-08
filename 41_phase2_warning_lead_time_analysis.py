#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 / Step 1: warning-lead-time audit for locked Causal-SeisField results.

Inputs:
- locked target-level predictions (Base + Power Gate)
- T0=5 s/K=5 manifest
- per-event HDF5 files
- frozen training-only high-motion thresholds

Outputs:
- unique_event_station_lead_times.csv
- locked_predictions_with_lead_times.csv
- lead_time_distribution_summary.csv
- metrics_by_peak_lead_time.csv
- power_minus_base_by_peak_lead_time.csv
- latency_sensitivity.csv
- future_window_edge_audit.csv

Main metric aggregation:
targets -> repeats -> events.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return (
        series.astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes", "y", "t"})
        .to_numpy(dtype=bool)
    )


def resolve_h5(event_id: str, row: pd.Series, h5_root: Path | None) -> Path:
    raw = str(row.get("h5_path", "")).strip()
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    if h5_root is not None:
        p = h5_root / f"{event_id}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot resolve HDF5 for event {event_id}; "
        f"manifest h5_path={raw!r}, h5_root={h5_root}"
    )


def read_strings(h5: h5py.File, name: str, n: int) -> np.ndarray:
    if name not in h5:
        return np.asarray([""] * n, dtype=object)
    try:
        return h5[name].asstr()[:]
    except Exception:
        return np.asarray([str(x) for x in h5[name][:]], dtype=object)


def horizontal_series(x: np.ndarray) -> np.ndarray:
    return np.sqrt(
        x[0].astype(np.float64) ** 2
        + x[1].astype(np.float64) ** 2
    )


def first_crossing(series: np.ndarray, threshold: float, fs: float) -> float:
    idx = np.flatnonzero(series >= float(threshold))
    return float(idx[0] / fs) if len(idx) else np.nan


def physical_leads(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    p_offset: np.ndarray,
    s_offset: np.ndarray,
    station_index: int,
    snapshot: int,
    fs: float,
    t0_sec: float,
    pga_thr: float,
    pgv_thr: float,
    end_guard_sec: float,
) -> dict:
    acc = horizontal_series(acceleration[station_index, :, snapshot:])
    vel = horizontal_series(velocity[station_index, :, snapshot:])
    if not len(acc) or not len(vel):
        raise ValueError("Empty post-snapshot waveform.")

    ipga = int(np.argmax(acc))
    ipgv = int(np.argmax(vel))
    pga = float(acc[ipga])
    pgv = float(vel[ipgv])
    lpga = float(ipga / fs)
    lpgv = float(ipgv / fs)
    duration = float((len(acc) - 1) / fs)

    po = float(p_offset[station_index]) if np.isfinite(p_offset[station_index]) else np.nan
    so = float(s_offset[station_index]) if np.isfinite(s_offset[station_index]) else np.nan

    return {
        "future_pga_mps2_recomputed": pga,
        "future_pgv_mps_recomputed": pgv,
        "true_log10_pga_recomputed": float(np.log10(max(pga, 1e-10))),
        "true_log10_pgv_recomputed": float(np.log10(max(pgv, 1e-12))),
        "lead_to_pga_peak_sec": lpga,
        "lead_to_pgv_peak_sec": lpgv,
        "target_p_offset_sec_h5": po,
        "target_s_offset_sec_h5": so,
        "lead_to_p_arrival_sec": po - t0_sec if np.isfinite(po) else np.nan,
        "lead_to_s_arrival_sec": so - t0_sec if np.isfinite(so) else np.nan,
        "pga_highmotion_crossing_lead_sec": first_crossing(acc, pga_thr, fs),
        "pgv_highmotion_crossing_lead_sec": first_crossing(vel, pgv_thr, fs),
        "future_window_duration_sec": duration,
        "pga_peak_near_window_end": bool(duration - lpga <= end_guard_sec),
        "pgv_peak_near_window_end": bool(duration - lpgv <= end_guard_sec),
    }


def make_bins(values: pd.Series, edges: list[float]) -> pd.Series:
    if edges[0] != 0.0:
        raise ValueError("Lead-bin edges must begin with 0.")
    bins = edges + [np.inf]
    labels = [
        f"{edges[i]:g}-{edges[i+1]:g}s"
        for i in range(len(edges) - 1)
    ] + [f">={edges[-1]:g}s"]
    return pd.cut(values, bins=bins, labels=labels, right=False, include_lowest=True)


def element_metric(truth: np.ndarray, pred: np.ndarray, metric: str) -> np.ndarray:
    r = pred - truth
    a = np.abs(r)
    if metric == "mae":
        return a
    if metric == "bias":
        return r
    if metric == "factor2":
        return (a <= LOG10_FACTOR_2).astype(float)
    if metric == "factor3":
        return (a <= LOG10_FACTOR_3).astype(float)
    if metric == "under03":
        return (r <= -0.3).astype(float)
    if metric == "under05":
        return (r <= -0.5).astype(float)
    raise ValueError(metric)


def canonical_metric(
    frame: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    metric: str,
) -> dict:
    if frame.empty:
        return {
            "value": np.nan, "n_target_rows": 0, "n_event_repeats": 0,
            "n_events": 0, "n_unique_event_station_pairs": 0,
        }
    truth = frame[truth_col].to_numpy(float)
    pred = frame[pred_col].to_numpy(float)
    work = frame[["event_id", "repeat", "target_station_index"]].copy()
    work["value"] = element_metric(truth, pred, metric)

    er = (
        work.groupby(["event_id", "repeat"], sort=False)["value"]
        .mean().reset_index()
    )
    ev = er.groupby("event_id", sort=False)["value"].mean()
    return {
        "value": float(ev.mean()),
        "n_target_rows": int(len(frame)),
        "n_event_repeats": int(len(er)),
        "n_events": int(len(ev)),
        "n_unique_event_station_pairs": int(
            frame[["event_id", "target_station_index"]].drop_duplicates().shape[0]
        ),
    }


def metrics_by_peak_lead(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("pga", "lead_to_pga_peak_sec", "pga_peak_lead_bin"),
        ("pgv", "lead_to_pgv_peak_sec", "pgv_peak_lead_bin"),
    ]
    models = ["base", "power_gate"]
    metrics = ["mae", "bias", "factor2", "factor3", "under03", "under05"]

    for q, lead_col, bin_col in specs:
        populations = {
            "overall": np.ones(len(df), dtype=bool),
            "high_motion_tail": bool_array(df[f"is_tail_{q}"]),
        }
        for population, mask in populations.items():
            sub = df.loc[mask].copy()
            for bin_name, g in sub.groupby(bin_col, observed=True, sort=False):
                for model in models:
                    for metric in metrics:
                        rec = canonical_metric(
                            g,
                            f"true_log10_{q}",
                            f"{model}_log10_{q}",
                            metric,
                        )
                        rows.append({
                            "quantity": q,
                            "population": population,
                            "lead_definition": lead_col,
                            "lead_bin": str(bin_name),
                            "model": model,
                            "metric": metric,
                            "median_lead_sec": float(g[lead_col].median()),
                            **rec,
                        })
    return pd.DataFrame(rows)


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("pga", "pga_peak_lead_bin"),
        ("pgv", "pgv_peak_lead_bin"),
    ]
    for q, bin_col in specs:
        tail = df.loc[bool_array(df[f"is_tail_{q}"])].copy()
        for bin_name, g in tail.groupby(bin_col, observed=True, sort=False):
            truth = g[f"true_log10_{q}"].to_numpy(float)
            for metric in ["mae", "bias", "factor2", "under05"]:
                a = element_metric(truth, g[f"base_log10_{q}"].to_numpy(float), metric)
                b = element_metric(truth, g[f"power_gate_log10_{q}"].to_numpy(float), metric)
                w = g[["event_id", "repeat"]].copy()
                w["a"] = a
                w["b"] = b
                er = w.groupby(["event_id", "repeat"], sort=False)[["a", "b"]].mean()
                ev = er.groupby(level="event_id")[["a", "b"]].mean()
                delta = ev["b"] - ev["a"]
                rows.append({
                    "quantity": q,
                    "population": "high_motion_tail",
                    "lead_bin": str(bin_name),
                    "metric": metric,
                    "n_paired_events": int(len(delta)),
                    "mean_delta_power_minus_base": float(delta.mean()),
                    "median_delta_power_minus_base": float(delta.median()),
                })
    return pd.DataFrame(rows)


def distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = {
        "pga_peak": "lead_to_pga_peak_sec",
        "pgv_peak": "lead_to_pgv_peak_sec",
        "s_arrival": "lead_to_s_arrival_sec",
        "pga_highmotion_crossing": "pga_highmotion_crossing_lead_sec",
        "pgv_highmotion_crossing": "pgv_highmotion_crossing_lead_sec",
    }
    for q in ("pga", "pgv"):
        populations = {
            "overall": np.ones(len(df), dtype=bool),
            "high_motion_tail": bool_array(df[f"is_tail_{q}"]),
        }
        for population, mask in populations.items():
            sub = df.loc[mask]
            for definition, col in definitions.items():
                x = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
                x = x[np.isfinite(x)]
                if not len(x):
                    continue
                rows.append({
                    "quantity": q,
                    "population": population,
                    "lead_definition": definition,
                    "n_target_rows_total": int(len(sub)),
                    "n_with_finite_lead": int(len(x)),
                    "finite_fraction": float(len(x) / max(len(sub), 1)),
                    "median_sec": float(np.median(x)),
                    "p25_sec": float(np.percentile(x, 25)),
                    "p75_sec": float(np.percentile(x, 75)),
                    "p90_sec": float(np.percentile(x, 90)),
                    "fraction_gt_0s": float(np.mean(x > 0)),
                    "fraction_ge_2s": float(np.mean(x >= 2)),
                    "fraction_ge_5s": float(np.mean(x >= 5)),
                    "fraction_ge_10s": float(np.mean(x >= 10)),
                })
    return pd.DataFrame(rows)


def latency_summary(df: pd.DataFrame, latencies: list[float]) -> pd.DataFrame:
    rows = []
    specs = [
        ("pga", "lead_to_pga_peak_sec", "time_to_remaining_peak"),
        ("pgv", "lead_to_pgv_peak_sec", "time_to_remaining_peak"),
        ("pga", "pga_highmotion_crossing_lead_sec", "time_to_highmotion_threshold_crossing"),
        ("pgv", "pgv_highmotion_crossing_lead_sec", "time_to_highmotion_threshold_crossing"),
    ]
    for q, col, definition in specs:
        sub = df.loc[bool_array(df[f"is_tail_{q}"])]
        raw = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
        raw = raw[np.isfinite(raw)]
        for latency in latencies:
            if not len(raw):
                continue
            net = raw - latency
            rows.append({
                "quantity": q,
                "population": "high_motion_tail",
                "lead_definition": definition,
                "assumed_total_latency_sec": float(latency),
                "n_target_rows": int(len(sub)),
                "n_with_defined_lead": int(len(raw)),
                "defined_fraction": float(len(raw) / max(len(sub), 1)),
                "fraction_positive_net_lead": float(np.mean(net > 0)),
                "fraction_net_lead_ge_1s": float(np.mean(net >= 1)),
                "fraction_net_lead_ge_2s": float(np.mean(net >= 2)),
                "fraction_net_lead_ge_5s": float(np.mean(net >= 5)),
                "median_net_lead_sec": float(np.median(net)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        default="./runs/locked_power_gated_test_gamma3/locked_test_predictions.csv",
    )
    p.add_argument(
        "--manifest",
        default="./data/scedc/model_manifests/scenario_t0_5s_k5.csv",
    )
    p.add_argument(
        "--h5-root",
        default="./data/scedc/processed_full_v4/events",
        help="Linux HDF5 root; used if manifest h5_path is a stale Windows path.",
    )
    p.add_argument(
        "--threshold-json",
        default=(
            "runs/tail_gated_compromise_t0_5s_k5/"
            "tail_thresholds_q0.90_t0_5s.json"
        ),
    )
    p.add_argument("--t0-sec", type=float, default=5.0)
    p.add_argument("--lead-bin-edges", default="0,1,2,5,10,20")
    p.add_argument(
        "--latency-seconds",
        default="0,0.5,1,2",
        help="Descriptive assumed total latencies; these are not measured deployment latency.",
    )
    p.add_argument("--end-guard-sec", type=float, default=0.5)
    p.add_argument("--out-dir", default="runs/phase2_warning_lead_time")
    args = p.parse_args()

    pred_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    threshold_path = Path(args.threshold_json)
    h5_root = Path(args.h5_root) if args.h5_root.strip() else None

    for path in [pred_path, manifest_path, threshold_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(
        pred_path,
        dtype={"event_id": str, "target_station_id": str},
    )
    manifest = pd.read_csv(manifest_path, dtype={"event_id": str})
    manifest = manifest.drop_duplicates("event_id").set_index("event_id", drop=False)

    required = {
        "event_id", "repeat", "target_station_index", "target_station_id",
        "true_log10_pga", "true_log10_pgv",
        "base_log10_pga", "base_log10_pgv",
        "power_gate_log10_pga", "power_gate_log10_pgv",
        "is_tail_pga", "is_tail_pgv",
    }
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    pga_log_thr = float(threshold_data["log10_pga_threshold"])
    pgv_log_thr = float(threshold_data["log10_pgv_threshold"])
    pga_thr = 10.0 ** pga_log_thr
    pgv_thr = 10.0 ** pgv_log_thr

    print("=== Phase 2 / Step 1: warning lead-time audit ===")
    print(f"Prediction rows : {len(pred):,}")
    print(f"Events          : {pred['event_id'].nunique():,}")
    print(
        f"Thresholds      : PGA={pga_log_thr:.4f} ({pga_thr:.6g} m/s^2), "
        f"PGV={pgv_log_thr:.4f} ({pgv_thr:.6g} m/s)"
    )

    pairs = (
        pred[["event_id", "target_station_index", "target_station_id"]]
        .drop_duplicates(["event_id", "target_station_index"])
        .reset_index(drop=True)
    )

    physical_rows = []
    mismatches = []
    n_events = pairs["event_id"].nunique()

    for event_number, (event_id, gp) in enumerate(
        pairs.groupby("event_id", sort=False), start=1
    ):
        if event_id not in manifest.index:
            raise KeyError(f"Event {event_id} not found in manifest.")
        row = manifest.loc[event_id]
        h5_path = resolve_h5(event_id, row, h5_root)

        with h5py.File(h5_path, "r") as h5:
            acceleration = np.asarray(h5["acceleration"][:], dtype=np.float32)
            velocity = np.asarray(h5["velocity"][:], dtype=np.float32)
            p_offset = np.asarray(h5["p_offset_sec"][:], dtype=np.float64)
            s_offset = (
                np.asarray(h5["s_offset_sec"][:], dtype=np.float64)
                if "s_offset_sec" in h5
                else np.full(len(p_offset), np.nan)
            )
            station_ids = read_strings(h5, "station_id", len(p_offset))
            fs = float(h5.attrs["sampling_rate_hz"])
            zero = int(h5.attrs["time_zero_index"])

        snapshot = min(
            zero + int(round(args.t0_sec * fs)),
            acceleration.shape[-1] - 1,
        )

        for r in gp.itertuples(index=False):
            idx = int(r.target_station_index)
            if idx < 0 or idx >= len(p_offset):
                raise IndexError(f"Event {event_id}: station index {idx} out of range.")
            expected = str(r.target_station_id).strip()
            actual = str(station_ids[idx]).strip()
            if expected != actual:
                mismatches.append((event_id, idx, expected, actual))
            physical_rows.append({
                "event_id": event_id,
                "target_station_index": idx,
                "target_station_id": expected,
                "h5_station_id": actual,
                "h5_path_resolved": str(h5_path),
                "sampling_rate_hz": fs,
                **physical_leads(
                    acceleration, velocity, p_offset, s_offset, idx,
                    snapshot, fs, args.t0_sec, pga_thr, pgv_thr,
                    args.end_guard_sec,
                ),
            })

        if event_number % 25 == 0 or event_number == n_events:
            print(f"Processed events: {event_number}/{n_events} | pairs={len(physical_rows):,}")

    physical = pd.DataFrame(physical_rows)
    if mismatches:
        pd.DataFrame(
            mismatches,
            columns=["event_id", "target_station_index", "expected", "actual"],
        ).to_csv(out_dir / "station_id_mismatches.csv", index=False)
        raise RuntimeError(
            f"Detected {len(mismatches)} station-ID mismatches; "
            f"see {out_dir/'station_id_mismatches.csv'}"
        )

    physical.to_csv(out_dir / "unique_event_station_lead_times.csv", index=False)

    merged = pred.merge(
        physical.drop(columns=["target_station_id"]),
        on=["event_id", "target_station_index"],
        how="left",
        validate="many_to_one",
    )

    merged["pga_truth_abs_diff"] = np.abs(
        merged["true_log10_pga"].astype(float)
        - merged["true_log10_pga_recomputed"].astype(float)
    )
    merged["pgv_truth_abs_diff"] = np.abs(
        merged["true_log10_pgv"].astype(float)
        - merged["true_log10_pgv_recomputed"].astype(float)
    )
    max_pga = float(merged["pga_truth_abs_diff"].max())
    max_pgv = float(merged["pgv_truth_abs_diff"].max())
    print(f"Truth recomputation max |diff|: PGA={max_pga:.3e}, PGV={max_pgv:.3e}")
    if max(max_pga, max_pgv) > 1e-5:
        merged.loc[
            (merged["pga_truth_abs_diff"] > 1e-5)
            | (merged["pgv_truth_abs_diff"] > 1e-5)
        ].head(100).to_csv(out_dir / "truth_recomputation_mismatches.csv", index=False)
        raise RuntimeError("Locked truth does not match HDF5 recomputation.")

    edges = sorted(
        set(float(x.strip()) for x in args.lead_bin_edges.split(",") if x.strip())
    )
    merged["pga_peak_lead_bin"] = make_bins(merged["lead_to_pga_peak_sec"], edges)
    merged["pgv_peak_lead_bin"] = make_bins(merged["lead_to_pgv_peak_sec"], edges)
    merged.to_csv(out_dir / "locked_predictions_with_lead_times.csv", index=False)

    metric_df = metrics_by_peak_lead(merged)
    metric_df.to_csv(out_dir / "metrics_by_peak_lead_time.csv", index=False)

    delta_df = paired_deltas(merged)
    delta_df.to_csv(out_dir / "power_minus_base_by_peak_lead_time.csv", index=False)

    dist_df = distribution_summary(merged)
    dist_df.to_csv(out_dir / "lead_time_distribution_summary.csv", index=False)

    latencies = [
        float(x.strip()) for x in args.latency_seconds.split(",") if x.strip()
    ]
    lat_df = latency_summary(merged, latencies)
    lat_df.to_csv(out_dir / "latency_sensitivity.csv", index=False)

    edge_df = pd.DataFrame([
        {
            "quantity": "pga",
            "n_rows": int(len(merged)),
            "n_peak_near_window_end": int(bool_array(merged["pga_peak_near_window_end"]).sum()),
            "fraction_peak_near_window_end": float(bool_array(merged["pga_peak_near_window_end"]).mean()),
        },
        {
            "quantity": "pgv",
            "n_rows": int(len(merged)),
            "n_peak_near_window_end": int(bool_array(merged["pgv_peak_near_window_end"]).sum()),
            "fraction_peak_near_window_end": float(bool_array(merged["pgv_peak_near_window_end"]).mean()),
        },
    ])
    edge_df.to_csv(out_dir / "future_window_edge_audit.csv", index=False)

    print("\n=== High-motion lead-time distribution ===")
    show = dist_df.loc[
        (dist_df["population"] == "high_motion_tail")
        & dist_df["lead_definition"].isin([
            "pga_peak", "pgv_peak", "s_arrival",
            "pga_highmotion_crossing", "pgv_highmotion_crossing",
        ])
    ]
    print(show[
        [
            "quantity", "lead_definition", "n_with_finite_lead",
            "median_sec", "p25_sec", "p75_sec",
            "fraction_ge_2s", "fraction_ge_5s", "fraction_ge_10s",
        ]
    ].to_string(index=False))

    print("\n=== High-motion tail MAE by time-to-peak bin ===")
    tail_mae = metric_df.loc[
        (metric_df["population"] == "high_motion_tail")
        & (metric_df["metric"] == "mae")
    ]
    print(
        tail_mae[
            ["quantity", "lead_bin", "model", "value", "n_events", "n_target_rows"]
        ].to_string(index=False)
    )

    print("\n=== HDF5 end-window audit ===")
    print(edge_df.to_string(index=False))

    print("\nOutputs:")
    for name in [
        "unique_event_station_lead_times.csv",
        "locked_predictions_with_lead_times.csv",
        "lead_time_distribution_summary.csv",
        "metrics_by_peak_lead_time.csv",
        "power_minus_base_by_peak_lead_time.csv",
        "latency_sensitivity.csv",
        "future_window_edge_audit.csv",
    ]:
        print(" ", (out_dir / name).resolve())


if __name__ == "__main__":
    main()
