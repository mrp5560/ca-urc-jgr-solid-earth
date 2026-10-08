#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 1 / Step 1
Canonical metric audit for repeated held-out-target predictions.

Main aggregation used for the manuscript:
1) average target-level metric within each (event, repeat);
2) average repeats within each event;
3) average events with equal weight.

This script also exports the older target-weighted and event-repeat-macro
values so numerical differences can be audited before rewriting the paper.

Default input:
runs/locked_power_gated_test_gamma3/locked_test_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOG10_FACTOR_2 = np.log10(2.0)
LOG10_FACTOR_3 = np.log10(3.0)


def as_bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy(dtype=bool)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
        .to_numpy(dtype=bool)
    )


def discover_models(df: pd.DataFrame) -> list[str]:
    models = []
    suffix = "_log10_pga"
    excluded = {"true"}
    for c in df.columns:
        if c.endswith(suffix):
            prefix = c[: -len(suffix)]
            if prefix in excluded:
                continue
            if f"{prefix}_log10_pgv" in df.columns:
                models.append(prefix)
    preferred = ["base", "linear_gate", "power_gate", "model"]
    ordered = [m for m in preferred if m in models]
    ordered += sorted(m for m in models if m not in ordered)
    return ordered


def population_mask(
    df: pd.DataFrame,
    quantity: str,
    population: str,
) -> np.ndarray:
    n = len(df)
    if population == "overall":
        return np.ones(n, dtype=bool)

    if population == "tail":
        c = f"is_tail_{quantity}"
        return as_bool(df[c]) if c in df.columns else np.zeros(n, dtype=bool)

    if population == "non_tail":
        c = f"is_tail_{quantity}"
        return ~as_bool(df[c]) if c in df.columns else np.zeros(n, dtype=bool)

    if population == "m4plus":
        if "magnitude" not in df.columns:
            return np.zeros(n, dtype=bool)
        return pd.to_numeric(df["magnitude"], errors="coerce").to_numpy() >= 4.0

    if population == "p_wave_reached":
        c = "target_triggered_by_snapshot"
        return as_bool(df[c]) if c in df.columns else np.zeros(n, dtype=bool)

    if population == "not_p_wave_reached":
        c = "target_triggered_by_snapshot"
        return ~as_bool(df[c]) if c in df.columns else np.zeros(n, dtype=bool)

    raise ValueError(population)


def element_metric(
    truth: np.ndarray,
    pred: np.ndarray,
    metric: str,
) -> np.ndarray:
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
    if metric == "under05":
        return (residual <= -0.5).astype(float)
    if metric == "under03":
        return (residual <= -0.3).astype(float)
    if metric == "squared_error":
        return residual ** 2

    raise ValueError(metric)


def aggregate_metric(
    selected: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    metric: str,
) -> dict[str, float]:
    truth = selected[truth_col].to_numpy(dtype=float)
    pred = selected[pred_col].to_numpy(dtype=float)

    if metric == "rmse":
        element = element_metric(truth, pred, "squared_error")
    else:
        element = element_metric(truth, pred, metric)

    work = selected[["event_id", "repeat"]].copy()
    work["element"] = element

    # Target-weighted.
    target_value = float(np.mean(element))
    if metric == "rmse":
        target_value = float(np.sqrt(target_value))

    # Old event-repeat macro: each event-repeat carrying this population
    # receives equal weight.
    er = (
        work.groupby(["event_id", "repeat"], sort=False)["element"]
        .mean()
        .rename("value")
        .reset_index()
    )
    if metric == "rmse":
        er["value"] = np.sqrt(er["value"])

    event_repeat_macro = float(er["value"].mean())

    # Canonical manuscript metric:
    # targets -> repeats -> events.
    event_values = (
        er.groupby("event_id", sort=False)["value"]
        .mean()
    )
    canonical = float(event_values.mean())

    return {
        "target_weighted": target_value,
        "event_repeat_macro": event_repeat_macro,
        "canonical_event_macro": canonical,
        "n_events": int(event_values.shape[0]),
        "n_event_repeats": int(er.shape[0]),
        "n_target_rows": int(len(selected)),
    }


def event_level_values(
    df: pd.DataFrame,
    quantity: str,
    population: str,
    model: str,
    metric: str,
) -> pd.Series:
    mask = population_mask(df, quantity, population)
    selected = df.loc[mask].copy()
    if selected.empty:
        return pd.Series(dtype=float)

    truth = selected[f"true_log10_{quantity}"].to_numpy(dtype=float)
    pred = selected[f"{model}_log10_{quantity}"].to_numpy(dtype=float)

    if metric == "rmse":
        element = element_metric(truth, pred, "squared_error")
    else:
        element = element_metric(truth, pred, metric)

    work = selected[["event_id", "repeat"]].copy()
    work["element"] = element
    er = (
        work.groupby(["event_id", "repeat"], sort=False)["element"]
        .mean()
        .rename("value")
        .reset_index()
    )
    if metric == "rmse":
        er["value"] = np.sqrt(er["value"])

    return er.groupby("event_id", sort=False)["value"].mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        default=(
            'runs/locked_power_gated_test_gamma3/locked_test_predictions.csv'
        ),
    )
    p.add_argument(
        "--out-dir",
        default='runs/phase1_canonical_metric_audit',
    )
    p.add_argument(
        "--models",
        default="auto",
        help="Comma-separated model prefixes or 'auto'.",
    )
    p.add_argument(
        "--reference-model",
        default="base",
    )
    args = p.parse_args()

    path = Path(args.predictions)
    if not path.exists():
        raise FileNotFoundError(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path, dtype={"event_id": str})
    required = {"event_id", "repeat", "true_log10_pga", "true_log10_pgv"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.models.strip().lower() == "auto":
        models = discover_models(df)
    else:
        models = [x.strip() for x in args.models.split(",") if x.strip()]

    if not models:
        raise ValueError("No prediction models discovered.")

    for model in models:
        for quantity in ("pga", "pgv"):
            c = f"{model}_log10_{quantity}"
            if c not in df.columns:
                raise ValueError(f"Missing prediction column: {c}")

    populations = [
        "overall",
        "tail",
        "non_tail",
        "m4plus",
        "p_wave_reached",
        "not_p_wave_reached",
    ]
    metrics = [
        "mae",
        "rmse",
        "bias",
        "factor2",
        "factor3",
        "under03",
        "under05",
    ]

    rows = []
    for model in models:
        for quantity in ("pga", "pgv"):
            for population in populations:
                mask = population_mask(df, quantity, population)
                if not mask.any():
                    continue
                selected = df.loc[mask]
                for metric in metrics:
                    values = aggregate_metric(
                        selected,
                        f"true_log10_{quantity}",
                        f"{model}_log10_{quantity}",
                        metric,
                    )
                    rows.append(
                        {
                            "model": model,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric,
                            **values,
                            "macro_minus_event_repeat": (
                                values["canonical_event_macro"]
                                - values["event_repeat_macro"]
                            ),
                            "macro_minus_target_weighted": (
                                values["canonical_event_macro"]
                                - values["target_weighted"]
                            ),
                        }
                    )

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "canonical_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Paired event-level deltas after repeat averaging.
    reference = args.reference_model
    delta_rows = []
    if reference in models:
        for candidate in [m for m in models if m != reference]:
            for quantity in ("pga", "pgv"):
                for population in populations:
                    for metric in ["mae", "bias", "factor2", "under05"]:
                        a = event_level_values(
                            df, quantity, population, reference, metric
                        )
                        b = event_level_values(
                            df, quantity, population, candidate, metric
                        )
                        common = a.index.intersection(b.index)
                        if len(common) == 0:
                            continue
                        d = b.loc[common].to_numpy() - a.loc[common].to_numpy()
                        delta_rows.append(
                            {
                                "reference_model": reference,
                                "candidate_model": candidate,
                                "quantity": quantity,
                                "population": population,
                                "metric": metric,
                                "n_paired_events": int(len(common)),
                                "mean_delta_candidate_minus_reference": float(
                                    np.mean(d)
                                ),
                                "median_delta_candidate_minus_reference": float(
                                    np.median(d)
                                ),
                            }
                        )

    delta_df = pd.DataFrame(delta_rows)
    delta_path = out_dir / "canonical_paired_deltas.csv"
    delta_df.to_csv(delta_path, index=False)

    key = metrics_df.loc[
        metrics_df["population"].eq("overall")
        & metrics_df["metric"].isin(["mae", "bias", "factor2", "under05"])
    ].copy()

    print("=== Phase 1 / Step 1: canonical metric audit ===")
    print(f"Predictions : {path.resolve()}")
    print(f"Rows        : {len(df):,}")
    print(f"Events      : {df['event_id'].nunique():,}")
    print(f"Event-repeat: {df[['event_id','repeat']].drop_duplicates().shape[0]:,}")
    print(f"Models      : {models}")
    print("\nCanonical overall metrics:")
    print(
        key[
            [
                "model", "quantity", "metric",
                "target_weighted", "event_repeat_macro",
                "canonical_event_macro"
            ]
        ].to_string(index=False)
    )
    print(f"\nMetrics out : {metrics_path.resolve()}")
    print(f"Deltas out  : {delta_path.resolve()}")


if __name__ == "__main__":
    main()
