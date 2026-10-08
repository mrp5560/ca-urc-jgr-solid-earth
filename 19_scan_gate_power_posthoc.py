#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
19_scan_gate_power_posthoc.py

Post-hoc gate-power scan for the tail-risk gated dual-head model.

The trained model used:

    y_original = y_base + p * residual

Without retraining, this script evaluates:

    y_gamma = y_base + p^gamma * residual

Because the saved prediction file contains y_original rather than the raw
residual, the equivalent calculation is:

    y_gamma = y_base + p^(gamma - 1) * (y_original - y_base)

For gamma > 1, low-probability corrections are strongly suppressed while
high-probability corrections are largely retained.

The script:
    1. Reads repeated validation predictions from the dual-head model.
    2. Scans multiple gate powers.
    3. Computes target-weighted and event-macro metrics.
    4. Enforces overall/non-tail/bias constraints.
    5. Selects the feasible power with the lowest tail MAE.
    6. Performs event-level paired bootstrap against the frozen base.
    7. Exports predictions for the selected power.

Expected default input:
    runs/tail_risk_gated_dual_head_t0_5s_k5/
    validation_predictions_overall.csv

Example:
    python 19_scan_gate_power_posthoc.py ^
      --predictions runs\tail_risk_gated_dual_head_t0_5s_k5\validation_predictions_overall.csv ^
      --powers "1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0" ^
      --overall-budget 0.015 ^
      --non-tail-budget 0.005 ^
      --bias-limit 0.05 ^
      --bootstrap-repetitions 2000 ^
      --out-dir runs\gate_power_scan_validation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []

    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue

        value = float(item)
        if value < 1.0:
            raise ValueError(
                "All gate powers must be greater than or equal to 1.0."
            )
        values.append(value)

    if not values:
        raise ValueError("At least one gate power is required.")

    values = sorted(set(values))

    if 1.0 not in values:
        values.insert(0, 1.0)

    return values


def resolve_column(
    frame: pd.DataFrame,
    candidates: list[str],
    logical_name: str,
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise KeyError(
        f"Could not resolve column for {logical_name}. "
        f"Tried: {candidates}"
    )


def resolve_columns(
    frame: pd.DataFrame,
) -> dict[str, str]:
    columns = {
        "event_id": resolve_column(
            frame,
            ["event_id"],
            "event_id",
        ),
        "repeat": resolve_column(
            frame,
            ["repeat", "repeat_index"],
            "repeat",
        ),
    }

    for quantity in ("pga", "pgv"):
        columns[f"true_{quantity}"] = resolve_column(
            frame,
            [
                f"true_{quantity}",
                f"true_log10_{quantity}",
            ],
            f"true {quantity}",
        )
        columns[f"base_{quantity}"] = resolve_column(
            frame,
            [
                f"base_{quantity}",
                f"base_log10_{quantity}",
            ],
            f"base {quantity}",
        )
        columns[f"final_{quantity}"] = resolve_column(
            frame,
            [
                f"final_{quantity}",
                f"final_log10_{quantity}",
            ],
            f"final {quantity}",
        )
        columns[f"probability_{quantity}"] = resolve_column(
            frame,
            [
                f"probability_{quantity}",
                f"tail_probability_{quantity}",
            ],
            f"tail probability {quantity}",
        )
        columns[f"tail_{quantity}"] = resolve_column(
            frame,
            [
                f"tail_{quantity}",
                f"is_tail_{quantity}",
            ],
            f"tail label {quantity}",
        )

    optional_candidates = {
        "magnitude": ["magnitude"],
        "triggered": [
            "target_triggered",
            "target_triggered_by_snapshot",
        ],
        "target_station_id": ["target_station_id"],
    }

    for logical_name, candidates in optional_candidates.items():
        for candidate in candidates:
            if candidate in frame.columns:
                columns[logical_name] = candidate
                break

    return columns


def normalize_frame(
    raw: pd.DataFrame,
    columns: dict[str, str],
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "event_id": raw[columns["event_id"]].astype(str),
        "repeat": pd.to_numeric(
            raw[columns["repeat"]],
            errors="coerce",
        ).fillna(-1).astype(int),
    })

    for quantity in ("pga", "pgv"):
        for prefix in (
            "true",
            "base",
            "final",
            "probability",
        ):
            frame[f"{prefix}_{quantity}"] = pd.to_numeric(
                raw[columns[f"{prefix}_{quantity}"]],
                errors="coerce",
            )

        frame[f"tail_{quantity}"] = (
            raw[columns[f"tail_{quantity}"]]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y", "t"})
        )

    if "magnitude" in columns:
        frame["magnitude"] = pd.to_numeric(
            raw[columns["magnitude"]],
            errors="coerce",
        )
    else:
        frame["magnitude"] = np.nan

    if "triggered" in columns:
        frame["target_triggered"] = (
            raw[columns["triggered"]]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y", "t"})
        )
    else:
        frame["target_triggered"] = False

    if "target_station_id" in columns:
        frame["target_station_id"] = (
            raw[columns["target_station_id"]]
            .astype(str)
        )
    else:
        frame["target_station_id"] = ""

    required_numeric = [
        f"{prefix}_{quantity}"
        for quantity in ("pga", "pgv")
        for prefix in (
            "true",
            "base",
            "final",
            "probability",
        )
    ]

    missing_numeric = frame[required_numeric].isna().any(axis=1)
    if missing_numeric.any():
        count = int(missing_numeric.sum())
        raise ValueError(
            f"{count} rows contain missing required numeric values."
        )

    for quantity in ("pga", "pgv"):
        frame[f"probability_{quantity}"] = np.clip(
            frame[f"probability_{quantity}"].to_numpy(dtype=float),
            0.0,
            1.0,
        )

        original_correction = (
            frame[f"final_{quantity}"].to_numpy(dtype=float)
            - frame[f"base_{quantity}"].to_numpy(dtype=float)
        )

        negative_count = int(
            np.sum(original_correction < -1e-8)
        )
        if negative_count > 0:
            print(
                f"Warning: {negative_count} negative {quantity.upper()} "
                "corrections were clipped to zero."
            )

        frame[f"original_correction_{quantity}"] = np.maximum(
            original_correction,
            0.0,
        )

    return frame


def event_macro_mean(
    frame: pd.DataFrame,
    value_column: str,
    mask: np.ndarray | pd.Series | None = None,
) -> float:
    selected = frame

    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        selected = frame.loc[mask_array]

    if len(selected) == 0:
        return float("nan")

    grouped = (
        selected.groupby(
            ["event_id", "repeat"],
            sort=False,
        )[value_column]
        .mean()
    )

    if len(grouped) == 0:
        return float("nan")

    return float(grouped.mean())


def compute_single_model_metrics(
    frame: pd.DataFrame,
    prediction_prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_rows": int(len(frame)),
        "n_events": int(frame["event_id"].nunique()),
        "n_event_repeats": int(
            frame[
                ["event_id", "repeat"]
            ].drop_duplicates().shape[0]
        ),
    }

    m4plus = (
        frame["magnitude"].to_numpy(dtype=float) >= 4.0
    )
    triggered = frame[
        "target_triggered"
    ].to_numpy(dtype=bool)

    for quantity in ("pga", "pgv"):
        truth = frame[
            f"true_{quantity}"
        ].to_numpy(dtype=float)

        prediction = frame[
            f"{prediction_prefix}_{quantity}"
        ].to_numpy(dtype=float)

        residual = prediction - truth
        absolute = np.abs(residual)

        tail = frame[
            f"tail_{quantity}"
        ].to_numpy(dtype=bool)
        non_tail = ~tail

        abs_column = (
            f"__{prediction_prefix}_abs_{quantity}"
        )
        frame[abs_column] = absolute

        metrics[f"mae_{quantity}"] = float(
            absolute.mean()
        )
        metrics[f"bias_{quantity}"] = float(
            residual.mean()
        )
        metrics[f"factor2_{quantity}"] = float(
            np.mean(absolute <= LOG10_FACTOR_2)
        )

        metrics[f"macro_mae_{quantity}"] = event_macro_mean(
            frame,
            abs_column,
        )
        metrics[f"macro_tail_mae_{quantity}"] = event_macro_mean(
            frame,
            abs_column,
            tail,
        )
        metrics[f"macro_non_tail_mae_{quantity}"] = event_macro_mean(
            frame,
            abs_column,
            non_tail,
        )

        if tail.any():
            metrics[f"tail_bias_{quantity}"] = float(
                residual[tail].mean()
            )
            metrics[f"tail_under03_{quantity}"] = float(
                np.mean(residual[tail] <= -0.3)
            )
            metrics[f"tail_under05_{quantity}"] = float(
                np.mean(residual[tail] <= -0.5)
            )
        else:
            metrics[f"tail_bias_{quantity}"] = float("nan")
            metrics[f"tail_under03_{quantity}"] = float("nan")
            metrics[f"tail_under05_{quantity}"] = float("nan")

        if m4plus.any():
            metrics[f"m4plus_mae_{quantity}"] = float(
                absolute[m4plus].mean()
            )
            metrics[f"m4plus_bias_{quantity}"] = float(
                residual[m4plus].mean()
            )
        else:
            metrics[f"m4plus_mae_{quantity}"] = float("nan")
            metrics[f"m4plus_bias_{quantity}"] = float("nan")

        if triggered.any():
            metrics[f"triggered_mae_{quantity}"] = float(
                absolute[triggered].mean()
            )
            metrics[f"triggered_bias_{quantity}"] = float(
                residual[triggered].mean()
            )
        else:
            metrics[f"triggered_mae_{quantity}"] = float("nan")
            metrics[f"triggered_bias_{quantity}"] = float("nan")

        correction_column = (
            f"{prediction_prefix}_correction_{quantity}"
        )
        if correction_column in frame.columns:
            correction = frame[
                correction_column
            ].to_numpy(dtype=float)

            metrics[f"mean_correction_{quantity}"] = float(
                correction.mean()
            )
            metrics[f"tail_mean_correction_{quantity}"] = (
                float(correction[tail].mean())
                if tail.any()
                else float("nan")
            )
            metrics[f"non_tail_mean_correction_{quantity}"] = (
                float(correction[non_tail].mean())
                if non_tail.any()
                else float("nan")
            )

    return metrics


def add_power_predictions(
    base_frame: pd.DataFrame,
    power: float,
    prediction_prefix: str,
) -> pd.DataFrame:
    frame = base_frame.copy()

    for quantity in ("pga", "pgv"):
        probability = frame[
            f"probability_{quantity}"
        ].to_numpy(dtype=float)

        original_correction = frame[
            f"original_correction_{quantity}"
        ].to_numpy(dtype=float)

        if math.isclose(power, 1.0):
            multiplier = np.ones_like(probability)
        else:
            multiplier = np.power(
                np.clip(probability, 0.0, 1.0),
                power - 1.0,
            )

        correction = multiplier * original_correction
        prediction = (
            frame[f"base_{quantity}"].to_numpy(dtype=float)
            + correction
        )

        frame[
            f"{prediction_prefix}_{quantity}"
        ] = prediction
        frame[
            f"{prediction_prefix}_correction_{quantity}"
        ] = correction
        frame[
            f"{prediction_prefix}_effective_probability_{quantity}"
        ] = np.power(
            probability,
            power,
        )

    return frame


def compute_constraint_status(
    candidate: dict[str, float],
    base: dict[str, float],
    overall_budget: float,
    non_tail_budget: float,
    bias_limit: float,
) -> dict[str, Any]:
    values: dict[str, float] = {}

    for quantity in ("pga", "pgv"):
        values[f"overall_delta_{quantity}"] = (
            candidate[f"macro_mae_{quantity}"]
            - base[f"macro_mae_{quantity}"]
        )
        values[f"non_tail_delta_{quantity}"] = (
            candidate[f"macro_non_tail_mae_{quantity}"]
            - base[f"macro_non_tail_mae_{quantity}"]
        )
        values[f"tail_delta_{quantity}"] = (
            candidate[f"macro_tail_mae_{quantity}"]
            - base[f"macro_tail_mae_{quantity}"]
        )

        values[f"overall_excess_{quantity}"] = max(
            0.0,
            values[f"overall_delta_{quantity}"]
            - overall_budget,
        )
        values[f"non_tail_excess_{quantity}"] = max(
            0.0,
            values[f"non_tail_delta_{quantity}"]
            - non_tail_budget,
        )
        values[f"bias_excess_{quantity}"] = max(
            0.0,
            abs(candidate[f"bias_{quantity}"])
            - bias_limit,
        )

    feasible = all(
        values[key] <= 1e-12
        for key in (
            "overall_excess_pga",
            "overall_excess_pgv",
            "non_tail_excess_pga",
            "non_tail_excess_pgv",
            "bias_excess_pga",
            "bias_excess_pgv",
        )
    )

    normalized_violation = 0.0

    for quantity in ("pga", "pgv"):
        normalized_violation += (
            values[f"overall_excess_{quantity}"]
            / max(overall_budget, 1e-12)
        )
        normalized_violation += (
            values[f"non_tail_excess_{quantity}"]
            / max(non_tail_budget, 1e-12)
        )
        normalized_violation += (
            values[f"bias_excess_{quantity}"]
            / max(bias_limit, 1e-12)
        )

    tail_score = 0.5 * (
        candidate["macro_tail_mae_pga"]
        + candidate["macro_tail_mae_pgv"]
    )
    overall_score = 0.5 * (
        candidate["macro_mae_pga"]
        + candidate["macro_mae_pgv"]
    )

    return {
        **values,
        "feasible": bool(feasible),
        "normalized_violation": float(
            normalized_violation
        ),
        "tail_score": float(tail_score),
        "overall_score": float(overall_score),
    }


def build_event_level_values(
    frame: pd.DataFrame,
    prediction_prefix: str,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        event_id,
        repeat,
    ), group in frame.groupby(
        ["event_id", "repeat"],
        sort=False,
    ):
        for quantity in ("pga", "pgv"):
            truth = group[
                f"true_{quantity}"
            ].to_numpy(dtype=float)

            prediction = group[
                f"{prediction_prefix}_{quantity}"
            ].to_numpy(dtype=float)

            residual = prediction - truth
            absolute = np.abs(residual)

            tail = group[
                f"tail_{quantity}"
            ].to_numpy(dtype=bool)
            non_tail = ~tail

            populations = {
                "overall": np.ones(
                    len(group),
                    dtype=bool,
                ),
                "tail": tail,
                "non_tail": non_tail,
            }

            if group[
                "magnitude"
            ].notna().any():
                populations["m4plus"] = (
                    group["magnitude"].to_numpy(dtype=float)
                    >= 4.0
                )

            populations["triggered"] = group[
                "target_triggered"
            ].to_numpy(dtype=bool)

            for population, mask in populations.items():
                if not mask.any():
                    continue

                rows.extend([
                    {
                        "event_id": str(event_id),
                        "repeat": int(repeat),
                        "model": model_name,
                        "quantity": quantity,
                        "population": population,
                        "metric": "mae",
                        "value": float(
                            absolute[mask].mean()
                        ),
                    },
                    {
                        "event_id": str(event_id),
                        "repeat": int(repeat),
                        "model": model_name,
                        "quantity": quantity,
                        "population": population,
                        "metric": "absolute_bias",
                        "value": float(
                            abs(residual[mask].mean())
                        ),
                    },
                    {
                        "event_id": str(event_id),
                        "repeat": int(repeat),
                        "model": model_name,
                        "quantity": quantity,
                        "population": population,
                        "metric": "under05",
                        "value": float(
                            np.mean(
                                residual[mask] <= -0.5
                            )
                        ),
                    },
                ])

    return pd.DataFrame(rows)


def paired_bootstrap(
    event_values: pd.DataFrame,
    candidate_models: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    keys = (
        event_values[
            ["quantity", "population", "metric"]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for quantity, population, metric in keys:
        subset = event_values.loc[
            (
                event_values["quantity"] == quantity
            )
            & (
                event_values["population"] == population
            )
            & (
                event_values["metric"] == metric
            )
        ]

        pivot = subset.pivot_table(
            index=["event_id", "repeat"],
            columns="model",
            values="value",
            aggfunc="mean",
        )

        if "base" not in pivot.columns:
            continue

        for candidate in candidate_models:
            if candidate not in pivot.columns:
                continue

            paired = pivot[
                ["base", candidate]
            ].dropna()

            if len(paired) == 0:
                continue

            delta = (
                paired[candidate]
                - paired["base"]
            )

            event_delta = (
                delta.groupby(level="event_id")
                .mean()
                .to_numpy(dtype=float)
            )

            bootstrap_means = np.empty(
                repetitions,
                dtype=float,
            )

            for index in range(repetitions):
                sampled = rng.integers(
                    0,
                    len(event_delta),
                    size=len(event_delta),
                )
                bootstrap_means[index] = float(
                    event_delta[sampled].mean()
                )

            rows.append({
                "candidate_model": candidate,
                "quantity": quantity,
                "population": population,
                "metric": metric,
                "n_paired_events": int(
                    len(event_delta)
                ),
                "mean_delta_candidate_minus_base": float(
                    event_delta.mean()
                ),
                "ci95_low": float(
                    np.percentile(
                        bootstrap_means,
                        2.5,
                    )
                ),
                "ci95_high": float(
                    np.percentile(
                        bootstrap_means,
                        97.5,
                    )
                ),
                "probability_candidate_favorable": float(
                    np.mean(
                        bootstrap_means < 0.0
                    )
                ),
                "bootstrap_repetitions": int(
                    repetitions
                ),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "runs/time_clean_tail_risk_gated_t0_5s_k5/"
            "validation_predictions_overall.csv"
        ),
    )
    parser.add_argument(
        "--powers",
        default=(
            "1,1.25,1.5,1.75,2,2.5,3,4,4.5,5,5.5,6,7,8,10,12"
        ),
    )
    parser.add_argument(
        "--overall-budget",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--non-tail-budget",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--bias-limit",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "runs/time_clean_gate_power_scan_validation_extended"
        ),
    )

    args = parser.parse_args()

    prediction_path = Path(
        args.predictions
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{prediction_path}"
        )

    if args.overall_budget <= 0:
        raise ValueError(
            "--overall-budget must be positive."
        )
    if args.non_tail_budget <= 0:
        raise ValueError(
            "--non-tail-budget must be positive."
        )
    if args.bias_limit <= 0:
        raise ValueError(
            "--bias-limit must be positive."
        )
    if args.bootstrap_repetitions < 100:
        raise ValueError(
            "--bootstrap-repetitions must be at least 100."
        )

    powers = parse_float_list(
        args.powers
    )

    output_directory = Path(
        args.out_dir
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = pd.read_csv(
        prediction_path,
        dtype={"event_id": str},
    )
    columns = resolve_columns(raw)
    base_frame = normalize_frame(
        raw,
        columns,
    )

    print(
        "=== Post-hoc gate-power scan ==="
    )
    print(
        f"Predictions       : "
        f"{prediction_path.resolve()}"
    )
    print(
        f"Rows/events       : "
        f"{len(base_frame)}/"
        f"{base_frame['event_id'].nunique()}"
    )
    print(
        f"Powers            : "
        f"{', '.join(str(x) for x in powers)}"
    )
    print(
        "Constraints       : "
        f"overall +{args.overall_budget:.4f}, "
        f"non-tail +{args.non_tail_budget:.4f}, "
        f"|bias| <= {args.bias_limit:.4f}"
    )

    base_metrics = compute_single_model_metrics(
        base_frame.copy(),
        "base",
    )

    summary_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = [
        build_event_level_values(
            base_frame.copy(),
            "base",
            "base",
        )
    ]

    candidate_frames: dict[
        float,
        pd.DataFrame,
    ] = {}

    for power in powers:
        model_name = (
            "gamma_"
            + str(power).replace(".", "p")
        )
        prediction_prefix = model_name

        candidate_frame = add_power_predictions(
            base_frame,
            power,
            prediction_prefix,
        )
        candidate_metrics = (
            compute_single_model_metrics(
                candidate_frame,
                prediction_prefix,
            )
        )
        status = compute_constraint_status(
            candidate_metrics,
            base_metrics,
            overall_budget=(
                args.overall_budget
            ),
            non_tail_budget=(
                args.non_tail_budget
            ),
            bias_limit=args.bias_limit,
        )

        summary_rows.append({
            "gate_power": float(power),
            "model_name": model_name,
            **candidate_metrics,
            **{
                f"constraint_{key}": value
                for key, value
                in status.items()
            },
        })

        candidate_frames[power] = (
            candidate_frame
        )
        event_frames.append(
            build_event_level_values(
                candidate_frame,
                prediction_prefix,
                model_name,
            )
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "gate_power"
    )

    summary.to_csv(
        output_directory
        / "gate_power_scan_summary.csv",
        index=False,
    )

    feasible = summary.loc[
        summary[
            "constraint_feasible"
        ].astype(bool)
    ].copy()

    if len(feasible) > 0:
        feasible = feasible.sort_values(
            [
                "constraint_tail_score",
                "constraint_overall_score",
                "gate_power",
            ]
        )
        selected_row = feasible.iloc[0]
        selection_reason = (
            "lowest tail score among "
            "constraint-feasible powers"
        )
    else:
        selected_candidates = (
            summary.sort_values(
                [
                    "constraint_normalized_violation",
                    "constraint_tail_score",
                    "constraint_overall_score",
                ]
            )
        )
        selected_row = (
            selected_candidates.iloc[0]
        )
        selection_reason = (
            "minimum normalized violation; "
            "no fully feasible power found"
        )

    selected_power = float(
        selected_row["gate_power"]
    )
    selected_model_name = str(
        selected_row["model_name"]
    )

    selected_frame = candidate_frames[
        selected_power
    ].copy()

    selected_columns = [
        "event_id",
        "repeat",
        "target_station_id",
        "magnitude",
        "target_triggered",
    ]

    for quantity in ("pga", "pgv"):
        selected_columns.extend([
            f"true_{quantity}",
            f"tail_{quantity}",
            f"base_{quantity}",
            f"final_{quantity}",
            f"probability_{quantity}",
            f"original_correction_{quantity}",
            f"{selected_model_name}_{quantity}",
            f"{selected_model_name}_correction_{quantity}",
            (
                f"{selected_model_name}_"
                f"effective_probability_{quantity}"
            ),
        ])

    selected_frame[
        selected_columns
    ].to_csv(
        output_directory
        / "validation_predictions_selected_power.csv",
        index=False,
    )

    event_values = pd.concat(
        event_frames,
        ignore_index=True,
    )
    event_values.to_csv(
        output_directory
        / "event_level_metrics_all_powers.csv",
        index=False,
    )

    candidate_names = [
        str(name)
        for name in summary[
            "model_name"
        ].tolist()
    ]

    bootstrap = paired_bootstrap(
        event_values,
        candidate_models=candidate_names,
        repetitions=(
            args.bootstrap_repetitions
        ),
        seed=args.seed,
    )
    bootstrap.to_csv(
        output_directory
        / "paired_bootstrap_all_powers.csv",
        index=False,
    )

    selected_bootstrap = bootstrap.loc[
        bootstrap[
            "candidate_model"
        ].eq(selected_model_name)
    ].copy()
    selected_bootstrap.to_csv(
        output_directory
        / "paired_bootstrap_selected_power.csv",
        index=False,
    )

    selection = {
        "selected_gate_power": (
            selected_power
        ),
        "selected_model_name": (
            selected_model_name
        ),
        "selection_reason": (
            selection_reason
        ),
        "feasible": bool(
            selected_row[
                "constraint_feasible"
            ]
        ),
        "normalized_violation": float(
            selected_row[
                "constraint_normalized_violation"
            ]
        ),
        "overall_budget": float(
            args.overall_budget
        ),
        "non_tail_budget": float(
            args.non_tail_budget
        ),
        "bias_limit": float(
            args.bias_limit
        ),
        "base_metrics": base_metrics,
        "selected_metrics": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (np.bool_, bool),
                )
                else float(value)
                if isinstance(
                    value,
                    (
                        np.floating,
                        float,
                        np.integer,
                        int,
                    ),
                )
                else value
            )
            for key, value
            in selected_row.to_dict().items()
        },
    }

    (
        output_directory
        / "best_gate_power.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "gate_power",
        "constraint_feasible",
        "constraint_normalized_violation",
        "macro_mae_pga",
        "macro_mae_pgv",
        "constraint_overall_delta_pga",
        "constraint_overall_delta_pgv",
        "macro_non_tail_mae_pga",
        "macro_non_tail_mae_pgv",
        "constraint_non_tail_delta_pga",
        "constraint_non_tail_delta_pgv",
        "macro_tail_mae_pga",
        "macro_tail_mae_pgv",
        "constraint_tail_delta_pga",
        "constraint_tail_delta_pgv",
        "bias_pga",
        "bias_pgv",
        "tail_under05_pga",
        "tail_under05_pgv",
        "non_tail_mean_correction_pga",
        "non_tail_mean_correction_pgv",
    ]

    print(
        "\n=== Gate-power scan summary ==="
    )
    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Selected gate power ==="
    )
    print(
        f"power             : "
        f"{selected_power}"
    )
    print(
        f"feasible          : "
        f"{bool(selected_row['constraint_feasible'])}"
    )
    print(
        f"violation         : "
        f"{float(selected_row['constraint_normalized_violation']):.4f}"
    )
    print(
        f"selection reason  : "
        f"{selection_reason}"
    )
    print(
        "overall PGA/PGV  : "
        f"{float(selected_row['macro_mae_pga']):.4f}/"
        f"{float(selected_row['macro_mae_pgv']):.4f}"
    )
    print(
        "non-tail PGA/PGV : "
        f"{float(selected_row['macro_non_tail_mae_pga']):.4f}/"
        f"{float(selected_row['macro_non_tail_mae_pgv']):.4f}"
    )
    print(
        "tail PGA/PGV     : "
        f"{float(selected_row['macro_tail_mae_pga']):.4f}/"
        f"{float(selected_row['macro_tail_mae_pgv']):.4f}"
    )
    print(
        f"\nOutputs: "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
