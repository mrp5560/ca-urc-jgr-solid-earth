#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
51_attention_locked_test_and_canonical_summary.py

FINAL locked test for the attention-pooling Causal-SeisField + power-shaped
tail-risk gate, followed by manuscript-canonical aggregation.

This wrapper does NOT perform model selection.
The following must already be fixed from validation:
    - attention base checkpoint
    - tail-head checkpoint epoch
    - gamma
    - training-only tail thresholds
    - all test sampling settings

Workflow
--------
1. Invoke 20_evaluate_locked_power_gated_test.py with the FINAL attention
   training module (48_train_attention_tail_risk_gated.py).
2. Read locked_test_predictions.csv produced by script 20.
3. Recompute the manuscript metrics using:
       targets -> repeats -> events
   so that each earthquake contributes equal weight.
4. Export compact canonical tables for Base / Linear Gate / Power Gate.

NO test-dependent selection is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)


def canonical_metric(
    frame: pd.DataFrame,
    quantity: str,
    model: str,
    metric: str,
) -> tuple[float, int, int, int]:
    truth = frame[
        f"true_log10_{quantity}"
    ].to_numpy(dtype=float)

    prediction = frame[
        f"{model}_log10_{quantity}"
    ].to_numpy(dtype=float)

    residual = prediction - truth
    absolute = np.abs(residual)

    if metric == "mae":
        values = absolute
    elif metric == "rmse":
        values = residual ** 2
    elif metric == "bias":
        values = residual
    elif metric == "absolute_bias":
        # handled after event aggregation below
        values = residual
    elif metric == "factor2":
        values = (
            absolute <= LOG10_FACTOR_2
        ).astype(float)
    elif metric == "factor3":
        values = (
            absolute <= LOG10_FACTOR_3
        ).astype(float)
    elif metric == "under03":
        values = (
            residual <= -0.3
        ).astype(float)
    elif metric == "under05":
        values = (
            residual <= -0.5
        ).astype(float)
    else:
        raise ValueError(metric)

    work = frame[
        ["event_id", "repeat"]
    ].copy()
    work["value"] = values

    event_repeat = (
        work.groupby(
            ["event_id", "repeat"],
            sort=False,
        )["value"]
        .mean()
        .reset_index()
    )

    event = (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )["value"]
        .mean()
    )

    if metric == "rmse":
        value = float(
            np.sqrt(
                event.mean()
            )
        )
    elif metric == "absolute_bias":
        value = float(
            np.abs(event).mean()
        )
    else:
        value = float(
            event.mean()
        )

    return (
        value,
        int(event.shape[0]),
        int(event_repeat.shape[0]),
        int(frame.shape[0]),
    )


def population_mask(
    frame: pd.DataFrame,
    quantity: str,
    population: str,
) -> np.ndarray:
    n = len(frame)

    if population == "overall":
        return np.ones(
            n,
            dtype=bool,
        )

    if population == "tail":
        return frame[
            f"is_tail_{quantity}"
        ].astype(bool).to_numpy()

    if population == "non_tail":
        return ~frame[
            f"is_tail_{quantity}"
        ].astype(bool).to_numpy()

    if population == "m4plus":
        if "magnitude" not in frame.columns:
            return np.zeros(
                n,
                dtype=bool,
            )
        return (
            pd.to_numeric(
                frame["magnitude"],
                errors="coerce",
            ).to_numpy(dtype=float)
            >= 4.0
        )

    if population == "triggered":
        column = (
            "target_triggered_by_snapshot"
            if "target_triggered_by_snapshot"
            in frame.columns
            else "target_triggered"
        )
        if column not in frame.columns:
            return np.zeros(
                n,
                dtype=bool,
            )
        return (
            frame[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "t",
                }
            )
            .to_numpy()
        )

    if population == "not_triggered":
        return ~population_mask(
            frame,
            quantity,
            "triggered",
        )

    raise ValueError(population)


def build_canonical_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    models = (
        "base",
        "linear_gate",
        "power_gate",
    )

    quantities = (
        "pga",
        "pgv",
    )

    populations = (
        "overall",
        "tail",
        "non_tail",
        "m4plus",
        "triggered",
        "not_triggered",
    )

    metrics = (
        "mae",
        "rmse",
        "bias",
        "absolute_bias",
        "factor2",
        "factor3",
        "under03",
        "under05",
    )

    for model in models:
        for quantity in quantities:
            for population in populations:
                mask = population_mask(
                    predictions,
                    quantity,
                    population,
                )

                if not mask.any():
                    continue

                subset = (
                    predictions.loc[
                        mask
                    ].copy()
                )

                for metric in metrics:
                    (
                        value,
                        n_events,
                        n_event_repeats,
                        n_target_rows,
                    ) = canonical_metric(
                        subset,
                        quantity,
                        model,
                        metric,
                    )

                    rows.append(
                        {
                            "model": model,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric,
                            "canonical_event_balanced": value,
                            "n_events": n_events,
                            "n_event_repeats": n_event_repeats,
                            "n_target_rows": n_target_rows,
                        }
                    )

    return pd.DataFrame(rows)


def paired_event_deltas(
    predictions: pd.DataFrame,
    reference: str = "base",
    candidate: str = "power_gate",
) -> pd.DataFrame:
    rows = []

    populations = (
        "overall",
        "tail",
        "non_tail",
        "m4plus",
        "triggered",
        "not_triggered",
    )

    metrics = (
        "mae",
        "bias",
        "factor2",
        "under05",
    )

    for quantity in (
        "pga",
        "pgv",
    ):
        for population in populations:
            mask = population_mask(
                predictions,
                quantity,
                population,
            )

            if not mask.any():
                continue

            subset = (
                predictions.loc[
                    mask
                ].copy()
            )

            truth = subset[
                f"true_log10_{quantity}"
            ].to_numpy(dtype=float)

            for metric in metrics:
                event_values = {}

                for model in (
                    reference,
                    candidate,
                ):
                    pred = subset[
                        f"{model}_log10_{quantity}"
                    ].to_numpy(dtype=float)

                    residual = pred - truth
                    absolute = np.abs(
                        residual
                    )

                    if metric == "mae":
                        values = absolute
                    elif metric == "bias":
                        values = residual
                    elif metric == "factor2":
                        values = (
                            absolute
                            <= LOG10_FACTOR_2
                        ).astype(float)
                    elif metric == "under05":
                        values = (
                            residual <= -0.5
                        ).astype(float)
                    else:
                        raise ValueError(metric)

                    work = subset[
                        [
                            "event_id",
                            "repeat",
                        ]
                    ].copy()
                    work["value"] = values

                    er = (
                        work.groupby(
                            [
                                "event_id",
                                "repeat",
                            ],
                            sort=False,
                        )["value"]
                        .mean()
                        .reset_index()
                    )

                    ev = (
                        er.groupby(
                            "event_id",
                            sort=False,
                        )["value"]
                        .mean()
                    )

                    event_values[
                        model
                    ] = ev

                common = (
                    event_values[
                        reference
                    ].index.intersection(
                        event_values[
                            candidate
                        ].index
                    )
                )

                delta = (
                    event_values[
                        candidate
                    ].loc[
                        common
                    ]
                    - event_values[
                        reference
                    ].loc[
                        common
                    ]
                )

                rows.append(
                    {
                        "reference_model": reference,
                        "candidate_model": candidate,
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "n_paired_events": int(
                            len(common)
                        ),
                        "reference_value": float(
                            event_values[
                                reference
                            ].loc[
                                common
                            ].mean()
                        ),
                        "candidate_value": float(
                            event_values[
                                candidate
                            ].loc[
                                common
                            ].mean()
                        ),
                        "mean_delta_candidate_minus_reference": float(
                            delta.mean()
                        ),
                        "median_delta_candidate_minus_reference": float(
                            delta.median()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation-script",
        default="20_evaluate_locked_power_gated_test.py",
    )
    parser.add_argument(
        "--training-script",
        default="48_train_attention_tail_risk_gated.py",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--threshold-json",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )
    parser.add_argument(
        "--split-name",
        default="test",
    )
    parser.add_argument(
        "--t0-sec",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--input-stations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--target-stations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--input-pre-sec",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--gate-power",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
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
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--checkpoint-every-events",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "attention_locked_power_gated_test"
        ),
    )
    parser.add_argument(
        "--skip-evaluator",
        action="store_true",
        help=(
            "Only recompute canonical summaries from an "
            "existing locked_test_predictions.csv."
        ),
    )

    args = parser.parse_args()

    output_directory = Path(
        args.out_dir
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_script = Path(
        args.evaluation_script
    )

    training_script = Path(
        args.training_script
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    threshold_path = Path(
        args.threshold_json
    )

    manifest_path = Path(
        args.manifest
    )

    for path in (
        evaluation_script,
        training_script,
        checkpoint_path,
        threshold_path,
        manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    lock_metadata = {
        "training_script": str(
            training_script.resolve()
        ),
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "gate_power": float(
            args.gate_power
        ),
        "threshold_json": str(
            threshold_path.resolve()
        ),
        "manifest": str(
            manifest_path.resolve()
        ),
        "split_column": (
            args.split_column
        ),
        "split_name": (
            args.split_name
        ),
        "t0_sec": int(
            args.t0_sec
        ),
        "input_stations": int(
            args.input_stations
        ),
        "target_stations": int(
            args.target_stations
        ),
        "input_pre_sec": float(
            args.input_pre_sec
        ),
        "repeats": int(
            args.repeats
        ),
        "seed": int(
            args.seed
        ),
        "selection_status": (
            "ALL MODEL AND GATE SELECTION COMPLETED "
            "ON VALIDATION BEFORE THIS TEST"
        ),
        "metric_aggregation_primary": (
            "targets -> repeats -> events"
        ),
    }

    (
        output_directory
        / "pre_test_lock.json"
    ).write_text(
        json.dumps(
            lock_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    prediction_path = (
        output_directory
        / "locked_test_predictions.csv"
    )

    if not args.skip_evaluator:
        command = [
            sys.executable,
            str(
                evaluation_script
            ),
            "--training-script",
            str(
                training_script
            ),
            "--checkpoint",
            str(
                checkpoint_path
            ),
            "--threshold-json",
            str(
                threshold_path
            ),
            "--manifest",
            str(
                manifest_path
            ),
            "--split-column",
            str(
                args.split_column
            ),
            "--split-name",
            str(
                args.split_name
            ),
            "--t0-sec",
            str(
                args.t0_sec
            ),
            "--input-stations",
            str(
                args.input_stations
            ),
            "--target-stations",
            str(
                args.target_stations
            ),
            "--input-pre-sec",
            str(
                args.input_pre_sec
            ),
            "--gate-power",
            str(
                args.gate_power
            ),
            "--repeats",
            str(
                args.repeats
            ),
            "--bootstrap-repetitions",
            str(
                args.bootstrap_repetitions
            ),
            "--seed",
            str(
                args.seed
            ),
            "--device",
            str(
                args.device
            ),
            "--torch-threads",
            str(
                args.torch_threads
            ),
            "--checkpoint-every-events",
            str(
                args.checkpoint_every_events
            ),
            "--out-dir",
            str(
                output_directory
            ),
        ]

        print(
            "=== FINAL locked attention-tail test ==="
        )
        print(
            "Checkpoint : %s"
            % checkpoint_path.resolve()
        )
        print(
            "Gamma      : %.6f"
            % args.gate_power
        )
        print(
            "Split      : %s=%s"
            % (
                args.split_column,
                args.split_name,
            )
        )
        print(
            "Repeats    : %d"
            % args.repeats
        )
        print(
            "Selection  : LOCKED BEFORE TEST"
        )

        subprocess.run(
            command,
            check=True,
        )

    if not prediction_path.exists():
        raise FileNotFoundError(
            prediction_path
        )

    predictions = pd.read_csv(
        prediction_path,
        dtype={
            "event_id": str,
        },
    )

    required = {
        "event_id",
        "repeat",
        "true_log10_pga",
        "true_log10_pgv",
        "base_log10_pga",
        "base_log10_pgv",
        "linear_gate_log10_pga",
        "linear_gate_log10_pgv",
        "power_gate_log10_pga",
        "power_gate_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
    }

    missing = required.difference(
        predictions.columns
    )

    if missing:
        raise ValueError(
            "Locked prediction file missing columns: %s"
            % sorted(missing)
        )

    summary = build_canonical_summary(
        predictions
    )

    summary_path = (
        output_directory
        / "canonical_event_balanced_metrics.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    deltas = paired_event_deltas(
        predictions,
        reference="base",
        candidate="power_gate",
    )

    delta_path = (
        output_directory
        / "canonical_power_minus_base_deltas.csv"
    )

    deltas.to_csv(
        delta_path,
        index=False,
    )

    compact = summary.loc[
        summary[
            "population"
        ].isin(
            [
                "overall",
                "tail",
                "non_tail",
                "triggered",
            ]
        )
        & summary[
            "metric"
        ].isin(
            [
                "mae",
                "bias",
                "factor2",
                "under05",
            ]
        )
    ].copy()

    compact_path = (
        output_directory
        / "canonical_key_metrics.csv"
    )

    compact.to_csv(
        compact_path,
        index=False,
    )

    print(
        "\n=== Canonical event-balanced locked test metrics ==="
    )

    key = summary.loc[
        summary[
            "population"
        ].isin(
            [
                "overall",
                "tail",
            ]
        )
        & summary[
            "metric"
        ].isin(
            [
                "mae",
                "bias",
                "factor2",
                "under05",
            ]
        )
    ][
        [
            "model",
            "quantity",
            "population",
            "metric",
            "canonical_event_balanced",
            "n_events",
            "n_target_rows",
        ]
    ]

    print(
        key.to_string(
            index=False
        )
    )

    print(
        "\n=== Canonical Power minus Base deltas ==="
    )

    key_delta = deltas.loc[
        deltas[
            "population"
        ].isin(
            [
                "overall",
                "tail",
            ]
        )
        & deltas[
            "metric"
        ].isin(
            [
                "mae",
                "factor2",
                "under05",
            ]
        )
    ]

    print(
        key_delta.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )
    for path in (
        prediction_path,
        summary_path,
        compact_path,
        delta_path,
        output_directory
        / "gate_classification_metrics.csv",
        output_directory
        / "paired_bootstrap_deltas.csv",
        output_directory
        / "pre_test_lock.json",
    ):
        print(
            "  %s"
            % path.resolve()
        )


if __name__ == "__main__":
    main()
