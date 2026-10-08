#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute missing U0.5 and high-motion-tail metrics from the prediction-level
CSV produced by 13_evaluate_sparse_field_repeated_baselines.py.

Primary aggregation:
    targets -> repeats -> events

Severe underprediction:
    residual = prediction - truth
    U0.5 = P(residual <= -0.5)

High-motion tail:
    truth >= train-only Q90 threshold

Default grouped-main thresholds:
    PGA = -2.011172 log10(m/s^2)
    PGV = -3.307250 log10(m/s)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METHODS = [
    "model",
    "train_median",
    "median_observed",
    "nearest_observed",
    "idw_observed",
]

LOG10_FACTOR2 = np.log10(2.0)
LOG10_FACTOR3 = np.log10(3.0)
SEVERE_UNDER_THRESHOLD = -0.5


def canonical_macro(
    event_ids: np.ndarray,
    repeats: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """targets -> repeats -> events"""
    event_ids = np.asarray(event_ids).astype(str)
    repeats = np.asarray(repeats).astype(int)
    values = np.asarray(values, dtype=float)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        event_ids = event_ids[mask]
        repeats = repeats[mask]
        values = values[mask]

    keep = np.isfinite(values)
    event_ids = event_ids[keep]
    repeats = repeats[keep]
    values = values[keep]

    if values.size == 0:
        return float("nan")

    work = pd.DataFrame(
        {
            "event_id": event_ids,
            "repeat": repeats,
            "value": values,
        }
    )

    event_repeat = (
        work.groupby(["event_id", "repeat"], sort=False)["value"]
        .mean()
        .reset_index()
    )

    event_level = (
        event_repeat.groupby("event_id", sort=False)["value"]
        .mean()
    )

    return float(event_level.mean())


def count_events(
    event_ids: np.ndarray,
    mask: np.ndarray,
) -> int:
    event_ids = np.asarray(event_ids).astype(str)
    mask = np.asarray(mask, dtype=bool)
    return int(pd.Series(event_ids[mask]).nunique())


def metric_bundle(
    truth: np.ndarray,
    pred: np.ndarray,
    event_ids: np.ndarray,
    repeats: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    residual = pred - truth
    absolute = np.abs(residual)

    return {
        "mae": canonical_macro(
            event_ids, repeats, absolute, mask
        ),
        "bias": canonical_macro(
            event_ids, repeats, residual, mask
        ),
        "factor2": canonical_macro(
            event_ids,
            repeats,
            (absolute <= LOG10_FACTOR2).astype(float),
            mask,
        ),
        "factor3": canonical_macro(
            event_ids,
            repeats,
            (absolute <= LOG10_FACTOR3).astype(float),
            mask,
        ),
        "under05": canonical_macro(
            event_ids,
            repeats,
            (residual <= SEVERE_UNDER_THRESHOLD).astype(float),
            mask,
        ),
    }


def resolve_methods(df: pd.DataFrame, requested: list[str]) -> list[str]:
    methods = []

    for method in requested:
        ok = True
        for quantity in ("pga", "pgv"):
            col = f"{method}_log10_{quantity}"
            if col not in df.columns:
                ok = False
                break

        if ok:
            methods.append(method)
        else:
            print(
                f"[skip] {method}: prediction columns are incomplete."
            )

    if not methods:
        raise RuntimeError(
            "No usable methods found in prediction CSV."
        )

    return methods


def audit_old_metrics(
    new_metrics: pd.DataFrame,
    old_metrics_path: Path,
    tolerance: float = 2e-6,
) -> None:
    if not old_metrics_path.exists():
        print(
            f"[audit] old metrics file not found: "
            f"{old_metrics_path}; skipped."
        )
        return

    old = pd.read_csv(old_metrics_path)

    required = {
        "method",
        "quantity",
        "mae_log10_event_macro",
    }

    if not required.issubset(old.columns):
        print(
            "[audit] old metrics file does not have the expected "
            "columns; skipped."
        )
        return

    current = new_metrics.loc[
        (new_metrics["population"] == "overall")
        & (new_metrics["metric"] == "mae"),
        ["method", "quantity", "value"],
    ].rename(
        columns={
            "value": "recomputed_mae"
        }
    )

    merged = old[
        [
            "method",
            "quantity",
            "mae_log10_event_macro",
        ]
    ].merge(
        current,
        on=["method", "quantity"],
        how="inner",
    )

    merged["abs_diff"] = np.abs(
        merged["mae_log10_event_macro"]
        - merged["recomputed_mae"]
    )

    print("\n=== Audit against original metrics_summary.csv ===")
    print(
        merged.to_string(index=False)
    )

    max_diff = (
        float(merged["abs_diff"].max())
        if len(merged)
        else float("nan")
    )

    print(
        f"Maximum overall-MAE difference: {max_diff:.3e}"
    )

    if np.isfinite(max_diff) and max_diff > tolerance:
        raise RuntimeError(
            "Prediction file does not reproduce the supplied "
            "metrics_summary.csv. Do not mix these files."
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "Prediction-level target_predictions.csv from the same run "
            "as metrics_summary.csv."
        ),
    )

    parser.add_argument(
        "--old-metrics",
        default="",
        help=(
            "Optional original metrics_summary.csv for consistency audit."
        ),
    )

    parser.add_argument(
        "--pga-tail-threshold",
        type=float,
        default=-2.011172,
        help="Train-only Q90 threshold in log10 PGA.",
    )

    parser.add_argument(
        "--pgv-tail-threshold",
        type=float,
        default=-3.307250,
        help="Train-only Q90 threshold in log10 PGV.",
    )

    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
    )

    parser.add_argument(
        "--out-dir",
        default="recomputed_missing_metrics",
    )

    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    df = pd.read_csv(
        prediction_path,
        dtype={"event_id": str},
    )

    required = [
        "event_id",
        "repeat",
        "true_log10_pga",
        "true_log10_pgv",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction CSV missing required columns: {missing}"
        )

    methods = resolve_methods(
        df,
        [
            x.strip()
            for x in args.methods.split(",")
            if x.strip()
        ],
    )

    thresholds = {
        "pga": float(args.pga_tail_threshold),
        "pgv": float(args.pgv_tail_threshold),
    }

    rows = []

    for method in methods:
        for quantity in ("pga", "pgv"):
            truth = df[
                f"true_log10_{quantity}"
            ].to_numpy(float)

            pred = df[
                f"{method}_log10_{quantity}"
            ].to_numpy(float)

            finite = (
                np.isfinite(truth)
                & np.isfinite(pred)
            )

            masks = {
                "overall": finite,
                "high_motion_tail": (
                    finite
                    & (
                        truth
                        >= thresholds[quantity]
                    )
                ),
            }

            for population, mask in masks.items():
                bundle = metric_bundle(
                    truth=truth,
                    pred=pred,
                    event_ids=df["event_id"].to_numpy(),
                    repeats=df["repeat"].to_numpy(),
                    mask=mask,
                )

                for metric, value in bundle.items():
                    rows.append(
                        {
                            "method": method,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric,
                            "value": value,
                            "n_events": count_events(
                                df["event_id"].to_numpy(),
                                mask,
                            ),
                            "n_target_rows": int(mask.sum()),
                        }
                    )

    result = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_metrics = (
        out_dir
        / "recomputed_overall_and_tail_metrics.csv"
    )

    result.to_csv(
        output_metrics,
        index=False,
    )

    if args.old_metrics:
        audit_old_metrics(
            result,
            Path(args.old_metrics),
        )

    # Compact table: the quantities needed for the NC benchmark table.
    compact = result.loc[
        result["metric"].isin(
            ["mae", "under05", "factor2"]
        )
    ].pivot_table(
        index=["method", "quantity"],
        columns=["population", "metric"],
        values="value",
        aggfunc="first",
    )

    compact.columns = [
        f"{population}_{metric}"
        for population, metric
        in compact.columns
    ]

    compact = (
        compact.reset_index()
        .sort_values(
            ["method", "quantity"]
        )
    )

    compact_path = (
        out_dir
        / "paper_compact_metrics.csv"
    )

    compact.to_csv(
        compact_path,
        index=False,
    )

    print("\n=== Recomputed metrics ===")
    show = result.loc[
        result["metric"].isin(
            ["mae", "under05", "factor2"]
        ),
        [
            "method",
            "quantity",
            "population",
            "metric",
            "value",
            "n_events",
            "n_target_rows",
        ],
    ]

    print(
        show.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(output_metrics.resolve())
    print(compact_path.resolve())

    print("\nDefinitions:")
    print(
        "  residual = prediction - truth"
    )
    print(
        "  U0.5 = P(residual <= -0.5)"
    )
    print(
        "  aggregation = targets -> repeats -> events"
    )
    print(
        f"  PGA tail threshold = {thresholds['pga']:.6f}"
    )
    print(
        f"  PGV tail threshold = {thresholds['pgv']:.6f}"
    )


if __name__ == "__main__":
    main()
