#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-URC Warning Lead-Time analysis.

Final candidate:
    CA-URC = Cross-Attention Underprediction-Risk Correction
    y = y_CA + p_under**gamma * Delta
    grouped validation lock: A4_under_only, epoch=3, gamma=5

NO training / NO model selection.

Inputs:
1) runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv
2) runs/phase2_warning_lead_time/unique_event_station_lead_times.csv

Primary aggregation: targets -> repeats -> events
Default peak-lead bins: 0-1, 1-2, 2-5, 5-10, 10-20, >=20 s
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)
SEVERE_UNDER_THRESHOLD = -0.5


def as_bool_array(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)
    x = s.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}
    unknown = set(x.unique()).difference(allowed)
    if unknown:
        raise ValueError(f"Cannot parse boolean values: {sorted(unknown)[:20]}")
    return x.isin({"true", "1", "yes", "y", "t"}).to_numpy(dtype=bool)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def make_bins(values: pd.Series, edges: list[float]) -> pd.Series:
    edges = sorted(set(float(v) for v in edges))
    if not edges or not np.isclose(edges[0], 0.0):
        raise ValueError("Lead-bin edges must start at 0.")
    bins = edges + [np.inf]
    labels = [f"{edges[i]:g}-{edges[i+1]:g}s" for i in range(len(edges)-1)] + [f">={edges[-1]:g}s"]
    return pd.cut(values, bins=bins, labels=labels, right=False, include_lowest=True)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"event_id": str})
    required = {
        "event_id", "repeat", "target_slot", "target_station_index",
        "true_log10_pga", "true_log10_pgv", "is_tail_pga", "is_tail_pgv",
        "A2_cross_attention_base_log10_pga", "A2_cross_attention_base_log10_pgv",
        "A4_under_only_log10_pga", "A4_under_only_log10_pgv",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    df["event_id"] = df["event_id"].astype(str).str.strip()
    keys = ["event_id", "repeat", "target_station_index"]
    if df.duplicated(keys).any():
        raise ValueError(f"Predictions are not unique on {keys}.")
    df["cross_attention_log10_pga"] = df["A2_cross_attention_base_log10_pga"].astype(float)
    df["cross_attention_log10_pgv"] = df["A2_cross_attention_base_log10_pgv"].astype(float)
    df["ca_urc_log10_pga"] = df["A4_under_only_log10_pga"].astype(float)
    df["ca_urc_log10_pgv"] = df["A4_under_only_log10_pgv"].astype(float)
    return df


def load_physical(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"event_id": str})
    required = {
        "event_id", "target_station_index",
        "lead_to_pga_peak_sec", "lead_to_pgv_peak_sec", "lead_to_s_arrival_sec",
        "pga_highmotion_crossing_lead_sec", "pgv_highmotion_crossing_lead_sec",
        "true_log10_pga_recomputed", "true_log10_pgv_recomputed",
        "pga_peak_near_window_end", "pgv_peak_near_window_end",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Lead-time file missing columns: {sorted(missing)}")
    df["event_id"] = df["event_id"].astype(str).str.strip()
    if df.duplicated(["event_id", "target_station_index"]).any():
        raise ValueError("Lead-time file is not unique on event_id + target_station_index.")
    return df


def merge_and_audit(pred: pd.DataFrame, phys: pd.DataFrame, tol: float):
    keys = ["event_id", "target_station_index"]
    unique_pred = pred[keys].drop_duplicates()
    covered = unique_pred.merge(phys[keys], on=keys, how="inner", validate="one_to_one")
    if len(covered) != len(unique_pred):
        raise RuntimeError(f"Lead-time coverage incomplete: {len(covered)}/{len(unique_pred)} unique event-station pairs.")
    df = pred.merge(phys, on=keys, how="left", validate="many_to_one")
    pga_diff = np.abs(df["true_log10_pga"].to_numpy(float) - df["true_log10_pga_recomputed"].to_numpy(float))
    pgv_diff = np.abs(df["true_log10_pgv"].to_numpy(float) - df["true_log10_pgv_recomputed"].to_numpy(float))
    audit = {
        "prediction_rows": int(len(pred)),
        "events": int(pred["event_id"].nunique()),
        "unique_event_station_pairs": int(len(unique_pred)),
        "lead_time_rows": int(len(phys)),
        "merged_rows": int(len(df)),
        "max_abs_truth_diff_pga": float(np.nanmax(pga_diff)),
        "max_abs_truth_diff_pgv": float(np.nanmax(pgv_diff)),
        "truth_tolerance": float(tol),
    }
    if audit["max_abs_truth_diff_pga"] > tol or audit["max_abs_truth_diff_pgv"] > tol:
        raise RuntimeError(f"Truth recomputation audit failed: {audit}")
    return df, audit


def element_metric(truth: np.ndarray, pred: np.ndarray, metric: str) -> np.ndarray:
    residual = pred - truth
    absolute = np.abs(residual)
    if metric == "mae":
        return absolute
    if metric == "bias":
        return residual
    if metric == "factor2":
        return (absolute <= LOG10_FACTOR_2).astype(float)
    if metric == "factor3":
        return (absolute <= LOG10_FACTOR_3).astype(float)
    if metric == "under03":
        return (residual <= -0.3).astype(float)
    if metric == "under05":
        return (residual <= SEVERE_UNDER_THRESHOLD).astype(float)
    raise ValueError(metric)


def event_values(df: pd.DataFrame, truth_col: str, pred_col: str, metric: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    values = element_metric(df[truth_col].to_numpy(float), df[pred_col].to_numpy(float), metric)
    w = df[["event_id", "repeat"]].copy()
    w["value"] = values
    er = w.groupby(["event_id", "repeat"], sort=False)["value"].mean().reset_index()
    return er.groupby("event_id", sort=False)["value"].mean()


def metrics_by_peak_lead(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = {"cross_attention": "cross_attention", "ca_urc": "ca_urc"}
    specs = [
        ("pga", "lead_to_pga_peak_sec", "pga_peak_lead_bin"),
        ("pgv", "lead_to_pgv_peak_sec", "pgv_peak_lead_bin"),
    ]
    for q, lead_col, bin_col in specs:
        populations = {
            "overall": np.ones(len(df), dtype=bool),
            "high_motion_tail": as_bool_array(df[f"is_tail_{q}"]),
        }
        for pop, mask in populations.items():
            dpop = df.loc[mask].copy()
            for lead_bin, d in dpop.groupby(bin_col, observed=True, sort=False):
                for model, prefix in models.items():
                    for metric in ("mae", "bias", "factor2", "factor3", "under03", "under05"):
                        ev = event_values(d, f"true_log10_{q}", f"{prefix}_log10_{q}", metric)
                        rows.append({
                            "quantity": q, "population": pop, "lead_definition": lead_col,
                            "lead_bin": str(lead_bin), "model": model, "metric": metric,
                            "value": float(ev.mean()), "median_lead_sec": float(d[lead_col].median()),
                            "n_events": int(len(ev)),
                            "n_event_repeats": int(d[["event_id", "repeat"]].drop_duplicates().shape[0]),
                            "n_target_rows": int(len(d)),
                        })
    return pd.DataFrame(rows)


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("pga", "pga_peak_lead_bin"), ("pgv", "pgv_peak_lead_bin")]
    for q, bin_col in specs:
        tail = df.loc[as_bool_array(df[f"is_tail_{q}"])].copy()
        for lead_bin, d in tail.groupby(bin_col, observed=True, sort=False):
            for metric in ("mae", "bias", "factor2", "factor3", "under05"):
                base = event_values(d, f"true_log10_{q}", f"cross_attention_log10_{q}", metric)
                final = event_values(d, f"true_log10_{q}", f"ca_urc_log10_{q}", metric)
                common = base.index.intersection(final.index)
                if len(common) == 0:
                    continue
                b = float(base.loc[common].mean())
                f = float(final.loc[common].mean())
                delta = final.loc[common] - base.loc[common]
                row = {
                    "quantity": q, "population": "high_motion_tail", "lead_bin": str(lead_bin),
                    "metric": metric, "n_paired_events": int(len(common)),
                    "cross_attention_value": b, "ca_urc_value": f,
                    "mean_delta_caurc_minus_cross_attention": float(delta.mean()),
                    "median_delta_caurc_minus_cross_attention": float(delta.median()),
                }
                if metric == "mae":
                    row["relative_change_percent"] = float(100.0 * (f - b) / max(abs(b), 1e-12))
                if metric in {"factor2", "factor3", "under05"}:
                    row["delta_percentage_points"] = float(100.0 * (f - b))
                rows.append(row)
    return pd.DataFrame(rows)


def lead_distribution(df: pd.DataFrame) -> pd.DataFrame:
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
            "high_motion_tail": as_bool_array(df[f"is_tail_{q}"]),
        }
        for pop, mask in populations.items():
            d = df.loc[mask]
            for name, col in definitions.items():
                x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
                x = x[np.isfinite(x)]
                if len(x) == 0:
                    continue
                rows.append({
                    "quantity": q, "population": pop, "lead_definition": name,
                    "n_target_rows_total": int(len(d)), "n_with_finite_lead": int(len(x)),
                    "finite_fraction": float(len(x) / max(len(d), 1)),
                    "median_sec": float(np.median(x)), "p25_sec": float(np.percentile(x, 25)),
                    "p75_sec": float(np.percentile(x, 75)), "p90_sec": float(np.percentile(x, 90)),
                    "fraction_gt_0s": float(np.mean(x > 0)), "fraction_ge_2s": float(np.mean(x >= 2)),
                    "fraction_ge_5s": float(np.mean(x >= 5)), "fraction_ge_10s": float(np.mean(x >= 10)),
                })
    return pd.DataFrame(rows)


def latency_table(df: pd.DataFrame, latencies: list[float]) -> pd.DataFrame:
    rows = []
    specs = [
        ("pga", "lead_to_pga_peak_sec", "time_to_remaining_peak"),
        ("pgv", "lead_to_pgv_peak_sec", "time_to_remaining_peak"),
        ("pga", "pga_highmotion_crossing_lead_sec", "time_to_highmotion_threshold_crossing"),
        ("pgv", "pgv_highmotion_crossing_lead_sec", "time_to_highmotion_threshold_crossing"),
    ]
    for q, col, definition in specs:
        d = df.loc[as_bool_array(df[f"is_tail_{q}"])]
        x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        for latency in latencies:
            net = x - float(latency)
            rows.append({
                "quantity": q, "population": "high_motion_tail", "lead_definition": definition,
                "assumed_total_latency_sec": float(latency), "n_with_defined_lead": int(len(x)),
                "fraction_positive_net_lead": float(np.mean(net > 0)),
                "fraction_net_lead_ge_1s": float(np.mean(net >= 1)),
                "fraction_net_lead_ge_2s": float(np.mean(net >= 2)),
                "fraction_net_lead_ge_5s": float(np.mean(net >= 5)),
                "median_net_lead_sec": float(np.median(net)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv")
    ap.add_argument("--lead-times", default="runs/phase2_warning_lead_time/unique_event_station_lead_times.csv")
    ap.add_argument("--lead-bin-edges", default="0,1,2,5,10,20")
    ap.add_argument("--latency-seconds", default="0,0.5,1,2")
    ap.add_argument("--truth-tolerance", type=float, default=1e-5)
    ap.add_argument("--out-dir", default="runs/caurc_warning_lead_time")
    args = ap.parse_args()

    pred = load_predictions(Path(args.predictions))
    phys = load_physical(Path(args.lead_times))
    df, audit = merge_and_audit(pred, phys, args.truth_tolerance)

    edges = parse_float_list(args.lead_bin_edges)
    latencies = parse_float_list(args.latency_seconds)
    df["pga_peak_lead_bin"] = make_bins(df["lead_to_pga_peak_sec"], edges)
    df["pgv_peak_lead_bin"] = make_bins(df["lead_to_pgv_peak_sec"], edges)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged_path = out / "caurc_predictions_with_lead_times.csv"
    dist_path = out / "lead_time_distribution_summary.csv"
    metrics_path = out / "metrics_by_peak_lead_time.csv"
    delta_path = out / "caurc_minus_cross_attention_by_peak_lead_time.csv"
    latency_path = out / "latency_sensitivity.csv"
    edge_path = out / "future_window_edge_audit.csv"
    audit_path = out / "lead_time_analysis_audit.json"

    df.to_csv(merged_path, index=False)
    dist = lead_distribution(df)
    dist.to_csv(dist_path, index=False)
    metrics = metrics_by_peak_lead(df)
    metrics.to_csv(metrics_path, index=False)
    deltas = paired_deltas(df)
    deltas.to_csv(delta_path, index=False)
    latency = latency_table(df, latencies)
    latency.to_csv(latency_path, index=False)

    edge = pd.DataFrame([
        {
            "quantity": "pga", "n_rows": int(len(df)),
            "n_peak_near_window_end": int(as_bool_array(df["pga_peak_near_window_end"]).sum()),
            "fraction_peak_near_window_end": float(as_bool_array(df["pga_peak_near_window_end"]).mean()),
        },
        {
            "quantity": "pgv", "n_rows": int(len(df)),
            "n_peak_near_window_end": int(as_bool_array(df["pgv_peak_near_window_end"]).sum()),
            "fraction_peak_near_window_end": float(as_bool_array(df["pgv_peak_near_window_end"]).mean()),
        },
    ])
    edge.to_csv(edge_path, index=False)

    audit.update({
        "final_model_name": "CA-URC",
        "final_model_internal_variant": "A4_under_only",
        "grouped_validation_selected_epoch": 3,
        "grouped_validation_selected_gamma": 5.0,
        "lead_bin_edges_sec": edges,
        "latency_sensitivity_sec": latencies,
        "metric_aggregation": "targets -> repeats -> events",
        "model_training_performed": False,
        "model_selection_performed": False,
        "physical_lead_times_reused": True,
    })
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("=== CA-URC Warning Lead-Time Analysis ===")
    print(f"Prediction rows          : {len(pred):,}")
    print(f"Events                   : {pred['event_id'].nunique():,}")
    print(f"Unique event-station     : {audit['unique_event_station_pairs']:,}")
    print("Final candidate          : CA-URC = A4_under_only (epoch=3, gamma=5)")
    print("Truth recomputation max |diff|: PGA=%.3e, PGV=%.3e" % (
        audit["max_abs_truth_diff_pga"], audit["max_abs_truth_diff_pgv"]
    ))

    print("\n=== High-motion lead-time distribution ===")
    print(dist.loc[dist["population"].eq("high_motion_tail"), [
        "quantity", "lead_definition", "n_with_finite_lead", "median_sec", "p25_sec", "p75_sec",
        "fraction_ge_2s", "fraction_ge_5s", "fraction_ge_10s",
    ]].to_string(index=False))

    print("\n=== High-motion Tail MAE by time-to-peak bin ===")
    print(metrics.loc[(metrics["population"] == "high_motion_tail") & (metrics["metric"] == "mae"), [
        "quantity", "lead_bin", "model", "value", "n_events", "n_target_rows",
    ]].to_string(index=False))

    print("\n=== CA-URC minus Cross-Attention: Tail MAE by lead bin ===")
    print(deltas.loc[deltas["metric"].eq("mae"), [
        "quantity", "lead_bin", "n_paired_events", "cross_attention_value", "ca_urc_value",
        "mean_delta_caurc_minus_cross_attention", "relative_change_percent",
    ]].to_string(index=False))

    print("\n=== CA-URC minus Cross-Attention: Tail U0.5 by lead bin ===")
    print(deltas.loc[deltas["metric"].eq("under05"), [
        "quantity", "lead_bin", "n_paired_events", "cross_attention_value", "ca_urc_value",
        "mean_delta_caurc_minus_cross_attention", "delta_percentage_points",
    ]].to_string(index=False))

    print("\n=== HDF5 end-window audit ===")
    print(edge.to_string(index=False))

    print("\nOutputs:")
    for p in [merged_path, dist_path, metrics_path, delta_path, latency_path, edge_path, audit_path]:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
