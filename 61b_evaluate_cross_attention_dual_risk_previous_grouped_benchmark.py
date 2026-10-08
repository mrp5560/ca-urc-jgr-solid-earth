#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
61b_evaluate_cross_attention_dual_risk_previous_grouped_benchmark.py

Locked evaluation for the NEXT-GENERATION model:
    Frozen Cross-Attention Base + Dual-Risk Power Gate

Final formula:
    y_final = y_base + p_under * p_tail**gamma * Delta

This script performs NO model selection and NO tuning.

Recommended use
---------------
Use only after script 60 has selected:
    selected_dual_risk_head.pt
    selected_epoch_gamma.json

This variant intentionally evaluates on the PREVIOUSLY INSPECTED 224-event
grouped test set. Therefore, it is a retrospective fixed benchmark, NOT a
previously untouched final test for the next-generation model.

It reproduces the exact script-20 locked target draws used in the completed
strong-baseline experiment so that all 44,800 rows are directly comparable.

Primary aggregation:
    targets -> repeats -> events
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)
SEVERE_UNDER_THRESHOLD = -0.5


def load_module(
    path: str | Path,
    name: str,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required module not found: {path.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(
    path: str | Path,
    device: torch.device,
):
    try:
        return torch.load(
            str(path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            str(path),
            map_location=device,
        )


def canonical_macro(
    frame: pd.DataFrame,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[
    float,
    int,
    int,
    int,
]:
    work = frame[
        [
            "event_id",
            "repeat",
        ]
    ].copy()

    work[
        "value"
    ] = np.asarray(
        values,
        dtype=float,
    )

    if mask is not None:
        keep = np.asarray(
            mask,
            dtype=bool,
        )

        work = work.loc[
            keep
        ].copy()

    if work.empty:
        return (
            float("nan"),
            0,
            0,
            0,
        )

    event_repeat = (
        work.groupby(
            [
                "event_id",
                "repeat",
            ],
            sort=False,
        )[
            "value"
        ]
        .mean()
        .reset_index()
    )

    event = (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )[
            "value"
        ]
        .mean()
    )

    return (
        float(
            event.mean()
        ),
        int(
            len(
                event
            )
        ),
        int(
            len(
                event_repeat
            )
        ),
        int(
            len(
                work
            )
        ),
    )


def build_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        truth = frame[
            f"true_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        base = frame[
            f"base_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        final = frame[
            f"final_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        tail = frame[
            f"is_tail_{quantity}"
        ].to_numpy(
            dtype=bool
        )

        for method, prediction in (
            (
                "cross_attention_base",
                base,
            ),
            (
                "dual_risk_gate",
                final,
            ),
        ):
            residual = (
                prediction
                - truth
            )

            absolute = np.abs(
                residual
            )

            populations = {
                "overall": np.ones(
                    len(
                        frame
                    ),
                    dtype=bool,
                ),
                "high_motion_tail": tail,
                "non_tail": ~tail,
            }

            metric_values = {
                "mae": absolute,
                "bias": residual,
                "factor2": (
                    absolute
                    <= LOG10_FACTOR_2
                ).astype(
                    float
                ),
                "factor3": (
                    absolute
                    <= LOG10_FACTOR_3
                ).astype(
                    float
                ),
                "under05": (
                    residual
                    <= SEVERE_UNDER_THRESHOLD
                ).astype(
                    float
                ),
            }

            for (
                population,
                population_mask,
            ) in populations.items():
                if not population_mask.any():
                    continue

                for (
                    metric,
                    values,
                ) in metric_values.items():
                    (
                        value,
                        n_events,
                        n_event_repeats,
                        n_target_rows,
                    ) = canonical_macro(
                        frame,
                        values,
                        population_mask,
                    )

                    rows.append(
                        {
                            "method": (
                                method
                            ),
                            "quantity": (
                                quantity
                            ),
                            "population": (
                                population
                            ),
                            "metric": (
                                metric
                            ),
                            "value": (
                                value
                            ),
                            "n_events": (
                                n_events
                            ),
                            "n_event_repeats": (
                                n_event_repeats
                            ),
                            "n_target_rows": (
                                n_target_rows
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-module",
        default=(
            "60_train_cross_attention_dual_risk_joint_selection.py"
        ),
    )

    parser.add_argument(
        "--baseline-module",
        default=(
            "45_phase2_strong_baseline_suite.py"
        ),
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--h5-root",
        default=(
            "data/processed_full_v4/events"
        ),
    )

    parser.add_argument(
        "--split-column",
        required=True,
    )

    parser.add_argument(
        "--split-label",
        required=True,
    )

    parser.add_argument(
        "--selected-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--selection-json",
        required=True,
    )

    parser.add_argument(
        "--threshold-json",
        required=True,
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
        "--repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        "--previous-locked-predictions",
        default=(
            "runs/"
            "attention_locked_power_gate_gamma7/"
            "locked_test_predictions.csv"
        ),
        help=(
            "Previous 44,800-row locked benchmark predictions, used only "
            "for exact target/truth pairing audit."
        ),
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        if (
            args.device
            == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            args.device
        )

    model_module = load_module(
        args.model_module,
        "dual_risk_model_module",
    )

    baseline_module = load_module(
        args.baseline_module,
        "strong_baseline_module",
    )

    # Exact previous locked-test draw protocol.
    def previous_locked_seed(
        text: str,
        base_seed: int,
    ) -> int:
        text = str(text)

        if text.startswith(
            "strong:test:"
        ):
            text = (
                "locked:"
                + text[
                    len("strong:"):
                ]
            )
        elif text.startswith(
            "strong:"
        ):
            text = text[
                len("strong:"):
            ]

        return (
            model_module
            .canonical_stable_seed(
                text,
                base_seed,
            )
        )

    baseline_module.stable_seed = (
        previous_locked_seed
    )

    selected_checkpoint_path = Path(
        args.selected_checkpoint
    )

    selection_json_path = Path(
        args.selection_json
    )

    threshold_path = Path(
        args.threshold_json
    )

    for path in (
        selected_checkpoint_path,
        selection_json_path,
        threshold_path,
        Path(
            args.manifest
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    selection = json.loads(
        selection_json_path.read_text(
            encoding="utf-8"
        )
    )

    gamma = float(
        selection[
            "selected_gamma"
        ]
    )

    if bool(
        selection.get(
            "selected_at_upper_gamma_boundary",
            False,
        )
    ):
        raise RuntimeError(
            "Selected gamma is marked as an upper-grid boundary. "
            "Do not evaluate a final test before validation-only "
            "gamma expansion is complete."
        )

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds = np.asarray(
        [
            threshold_data[
                "log10_pga_threshold"
            ],
            threshold_data[
                "log10_pgv_threshold"
            ],
        ],
        dtype=np.float32,
    )

    checkpoint = load_checkpoint(
        selected_checkpoint_path,
        device,
    )

    training_args = checkpoint.get(
        "args",
        {},
    )

    hidden_dim = int(
        training_args.get(
            "hidden_dim",
            128,
        )
    )

    attention_heads = int(
        training_args.get(
            "attention_heads",
            4,
        )
    )

    risk_hidden = int(
        training_args.get(
            "risk_hidden",
            64,
        )
    )

    maximum_correction = float(
        training_args.get(
            "maximum_correction",
            1.5,
        )
    )

    # Priors are only needed to initialize the modules before the selected
    # state_dict is loaded. Exact selected parameters then overwrite them.
    dummy_prevalence = np.asarray(
        [
            0.10,
            0.10,
        ],
        dtype=float,
    )

    base = baseline_module.make_model(
        "cross_attention",
        hidden_dim,
        attention_heads,
        50.0,
    ).to(
        device
    )

    model = model_module.CrossAttentionDualRiskGate(
        base=base,
        hidden_dim=hidden_dim,
        risk_hidden=risk_hidden,
        maximum_correction=maximum_correction,
        tail_prevalence=dummy_prevalence,
        under_prevalence=dummy_prevalence,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    model.eval()

    common_dataset = dict(
        manifest=args.manifest,
        h5_root=args.h5_root,
        split_column=args.split_column,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
    )

    dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name=args.split_label,
            training=False,
            repeats=args.repeats,
            **common_dataset,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_payload = {
        "model": (
            "Cross-Attention Dual-Risk Power Gate"
        ),
        "selected_checkpoint": str(
            selected_checkpoint_path.resolve()
        ),
        "selection_json": str(
            selection_json_path.resolve()
        ),
        "selected_epoch": (
            selection.get(
                "selected_epoch"
            )
        ),
        "selected_gamma": (
            gamma
        ),
        "manifest": str(
            Path(
                args.manifest
            ).resolve()
        ),
        "split_column": (
            args.split_column
        ),
        "split_label": (
            args.split_label
        ),
        "repeats": int(
            args.repeats
        ),
        "seed": int(
            args.seed
        ),
        "model_selection_performed_here": False,
        "benchmark_status": "previously inspected retrospective fixed benchmark",
        "previous_test_informed_model_design": True,
        "metric_aggregation": (
            "targets -> repeats -> events"
        ),
    }

    (
        out_dir
        / "pre_evaluation_lock.json"
    ).write_text(
        json.dumps(
            lock_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== Retrospective locked Cross-Attention Dual-Risk grouped benchmark ==="
    )

    print(
        f"Device              : {device}"
    )

    print(
        f"Split               : {args.split_column}={args.split_label}"
    )

    print(
        f"Events              : {len(dataset.frame)}"
    )

    print(
        f"Repeats/event       : {args.repeats}"
    )

    print(
        "Selected head epoch : "
        f"{selection.get('selected_epoch')}"
    )

    print(
        f"Selected gamma      : {gamma:g}"
    )

    print(
        "Model selection     : NONE"
    )

    rows = []

    with torch.inference_mode():
        for batch in loader:
            input_waveforms = batch[
                "input_waveforms"
            ].to(
                device,
                non_blocking=True,
            )

            input_features = batch[
                "input_features"
            ].to(
                device,
                non_blocking=True,
            )

            target_features = batch[
                "target_features"
            ].to(
                device,
                non_blocking=True,
            )

            truth = batch[
                "target_log"
            ].numpy()

            (
                base_prediction,
                risk_feature,
            ) = (
                model_module
                .cross_attention_base_context(
                    model.base,
                    input_waveforms,
                    input_features,
                    target_features,
                )
            )

            head = model.head_from_feature(
                risk_feature
            )

            final_prediction = (
                base_prediction
                + head[
                    "under_probability"
                ]
                * torch.pow(
                    torch.clamp(
                        head[
                            "tail_probability"
                        ],
                        0.0,
                        1.0,
                    ),
                    gamma,
                )
                * head[
                    "residual"
                ]
            )

            base_np = (
                base_prediction
                .cpu()
                .numpy()
            )

            final_np = (
                final_prediction
                .cpu()
                .numpy()
            )

            tail_probability_np = (
                head[
                    "tail_probability"
                ]
                .cpu()
                .numpy()
            )

            under_probability_np = (
                head[
                    "under_probability"
                ]
                .cpu()
                .numpy()
            )

            residual_np = (
                head[
                    "residual"
                ]
                .cpu()
                .numpy()
            )

            repeats = batch[
                "repeat"
            ].numpy()

            station_indices = batch[
                "target_station_index"
            ].numpy()

            event_ids = list(
                batch[
                    "event_id"
                ]
            )

            for b in range(
                truth.shape[
                    0
                ]
            ):
                for q in range(
                    truth.shape[
                        1
                    ]
                ):
                    true_pga = float(
                        truth[
                            b,
                            q,
                            0,
                        ]
                    )

                    true_pgv = float(
                        truth[
                            b,
                            q,
                            1,
                        ]
                    )

                    rows.append(
                        {
                            "event_id": str(
                                event_ids[
                                    b
                                ]
                            ),
                            "repeat": int(
                                repeats[
                                    b
                                ]
                            ),
                            "target_slot": int(
                                q
                            ),
                            "target_station_index": int(
                                station_indices[
                                    b,
                                    q,
                                ]
                            ),
                            "true_log10_pga": (
                                true_pga
                            ),
                            "true_log10_pgv": (
                                true_pgv
                            ),
                            "base_log10_pga": float(
                                base_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "base_log10_pgv": float(
                                base_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "final_log10_pga": float(
                                final_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "final_log10_pgv": float(
                                final_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "tail_probability_pga": float(
                                tail_probability_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "tail_probability_pgv": float(
                                tail_probability_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "under_probability_pga": float(
                                under_probability_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "under_probability_pgv": float(
                                under_probability_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "residual_correction_pga": float(
                                residual_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "residual_correction_pgv": float(
                                residual_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "is_tail_pga": bool(
                                true_pga
                                >= thresholds[
                                    0
                                ]
                            ),
                            "is_tail_pgv": bool(
                                true_pgv
                                >= thresholds[
                                    1
                                ]
                            ),
                            "selected_gamma": (
                                gamma
                            ),
                        }
                    )

    predictions = pd.DataFrame(
        rows
    )

    # Exact pairing audit against the previous 44,800-row locked benchmark.
    previous_path = Path(
        args.previous_locked_predictions
    )

    pairing_audit = {
        "previous_locked_predictions": str(
            previous_path
        ),
        "available": bool(
            previous_path.exists()
        ),
    }

    if previous_path.exists():
        previous = pd.read_csv(
            previous_path,
            dtype={
                "event_id": str,
            },
        )

        keys = [
            "event_id",
            "repeat",
            "target_station_index",
        ]

        required_previous = set(
            keys
            + [
                "true_log10_pga",
                "true_log10_pgv",
            ]
        )

        missing = required_previous.difference(
            previous.columns
        )

        if missing:
            raise ValueError(
                "Previous locked file missing columns: "
                f"{sorted(missing)}"
            )

        if previous.duplicated(
            keys
        ).any():
            raise ValueError(
                "Previous locked benchmark is not unique on "
                f"{keys}."
            )

        if predictions.duplicated(
            keys
        ).any():
            raise ValueError(
                "Current benchmark predictions are not unique on "
                f"{keys}."
            )

        paired = predictions[
            keys
            + [
                "true_log10_pga",
                "true_log10_pgv",
            ]
        ].merge(
            previous[
                keys
                + [
                    "true_log10_pga",
                    "true_log10_pgv",
                ]
            ],
            on=keys,
            how="inner",
            suffixes=(
                "_current",
                "_previous",
            ),
            validate="one_to_one",
        )

        pairing_audit.update(
            {
                "current_rows": int(
                    len(
                        predictions
                    )
                ),
                "previous_rows": int(
                    len(
                        previous
                    )
                ),
                "paired_rows": int(
                    len(
                        paired
                    )
                ),
                "exact_pairing": bool(
                    len(
                        paired
                    )
                    == len(
                        predictions
                    )
                    == len(
                        previous
                    )
                ),
                "max_abs_diff_truth_pga": float(
                    np.max(
                        np.abs(
                            paired[
                                "true_log10_pga_current"
                            ].to_numpy(
                                dtype=float
                            )
                            - paired[
                                "true_log10_pga_previous"
                            ].to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
                "max_abs_diff_truth_pgv": float(
                    np.max(
                        np.abs(
                            paired[
                                "true_log10_pgv_current"
                            ].to_numpy(
                                dtype=float
                            )
                            - paired[
                                "true_log10_pgv_previous"
                            ].to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
            }
        )

        if not pairing_audit[
            "exact_pairing"
        ]:
            raise RuntimeError(
                "Current benchmark targets do not exactly match "
                "the previous locked test."
            )

        if (
            pairing_audit[
                "max_abs_diff_truth_pga"
            ]
            > 1e-5
            or pairing_audit[
                "max_abs_diff_truth_pgv"
            ]
            > 1e-5
        ):
            raise RuntimeError(
                "Pairing keys match but target truths differ from "
                "the previous locked benchmark."
            )

    audit_path = (
        out_dir
        / "previous_locked_pairing_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            pairing_audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Previous locked benchmark pairing audit ==="
    )
    print(
        json.dumps(
            pairing_audit,
            indent=2,
        )
    )

    prediction_path = (
        out_dir
        / "locked_dual_risk_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    metrics = build_metrics(
        predictions
    )

    metrics_path = (
        out_dir
        / "canonical_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    print(
        "\n=== Canonical locked metrics ==="
    )

    display = metrics.loc[
        metrics[
            "population"
        ].isin(
            [
                "overall",
                "high_motion_tail",
            ]
        )
        & metrics[
            "metric"
        ].isin(
            [
                "mae",
                "bias",
                "factor2",
                "under05",
            ]
        )
    ]

    print(
        display.to_string(
            index=False
        )
    )

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            {
                **lock_payload,
                "n_target_rows": int(
                    len(
                        predictions
                    )
                ),
                "n_events": int(
                    predictions[
                        "event_id"
                    ].nunique()
                ),
                "thresholds": {
                    "pga": float(
                        thresholds[
                            0
                        ]
                    ),
                    "pgv": float(
                        thresholds[
                            1
                        ]
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nOutputs:"
    )

    for path in (
        prediction_path,
        metrics_path,
        summary_path,
        out_dir
        / "pre_evaluation_lock.json",
        audit_path,
    ):
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
