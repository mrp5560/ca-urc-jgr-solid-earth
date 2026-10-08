#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
56_evaluate_time_clean_attention_base_locked.py

One-shot locked evaluation of the chronological Attention Base.

Use after:
  53_train_time_extrapolation_attention_base.py

Scientific policy
-----------------
- The Attention Base checkpoint was selected using chronological validation.
- No risk-gated model is evaluated because the chronological validation scan
  produced zero feasible candidates under the pre-specified deployment guard.
- This script performs NO model selection and NO tuning.
- Primary aggregation:
      targets -> repeats -> events.

Typical use:
  split_label=test           -> 2021-2024 chronological extrapolation
  split_label=ridgecrest_ood -> Ridgecrest sequence-level OOD case study
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


def load_module(path: str | Path, name: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: str | Path, device: torch.device):
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


def infer_hidden(checkpoint: dict[str, Any]) -> int:
    args = checkpoint.get("args", {})

    if isinstance(args, dict):
        value = args.get("hidden_dim")
        if value is not None:
            return int(value)

    key = "station_encoder.0.weight"

    if key not in checkpoint["model_state"]:
        raise KeyError(
            f"Cannot infer hidden dimension; missing {key}"
        )

    return int(
        checkpoint["model_state"][key].shape[0]
    )


def canonical_macro(
    frame: pd.DataFrame,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, int, int, int]:
    work = frame[
        ["event_id", "repeat"]
    ].copy()

    work["value"] = np.asarray(
        values,
        dtype=float,
    )

    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        work = work.loc[keep].copy()

    if work.empty:
        return (
            float("nan"),
            0,
            0,
            0,
        )

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

    return (
        float(event.mean()),
        int(len(event)),
        int(len(event_repeat)),
        int(len(work)),
    )


def build_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quantity in ("pga", "pgv"):
        truth = predictions[
            f"true_log10_{quantity}"
        ].to_numpy(dtype=float)

        pred = predictions[
            f"pred_log10_{quantity}"
        ].to_numpy(dtype=float)

        residual = pred - truth
        absolute = np.abs(residual)

        tail = predictions[
            f"is_tail_{quantity}"
        ].to_numpy(dtype=bool)

        non_tail = ~tail

        populations = {
            "overall": np.ones(
                len(predictions),
                dtype=bool,
            ),
            "high_motion_tail": tail,
            "non_tail": non_tail,
        }

        metrics = {
            "mae": absolute,
            "bias": residual,
            "factor2": (
                absolute <= LOG10_FACTOR_2
            ).astype(float),
            "factor3": (
                absolute <= LOG10_FACTOR_3
            ).astype(float),
            "under05": (
                residual <= -0.5
            ).astype(float),
        }

        for population, mask in populations.items():
            if not mask.any():
                continue

            for metric, values in metrics.items():
                (
                    value,
                    n_events,
                    n_event_repeats,
                    n_target_rows,
                ) = canonical_macro(
                    predictions,
                    values,
                    mask,
                )

                rows.append(
                    {
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "value": value,
                        "n_events": n_events,
                        "n_event_repeats": n_event_repeats,
                        "n_target_rows": n_target_rows,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-module",
        default="53_train_time_extrapolation_attention_base.py",
    )

    p.add_argument(
        "--manifest",
        required=True,
    )

    p.add_argument(
        "--split-column",
        default="split_time_clean",
    )

    p.add_argument(
        "--split-label",
        default="test",
    )

    p.add_argument(
        "--base-checkpoint",
        required=True,
    )

    p.add_argument(
        "--threshold-json",
        required=True,
    )

    p.add_argument(
        "--t0-sec",
        type=int,
        default=5,
    )

    p.add_argument(
        "--input-stations",
        type=int,
        default=5,
    )

    p.add_argument(
        "--target-stations",
        type=int,
        default=10,
    )

    p.add_argument(
        "--input-pre-sec",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=20,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )

    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    p.add_argument(
        "--out-dir",
        required=True,
    )

    args = p.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(args.device)

    base_module = load_module(
        args.base_module,
        "time_clean_attention_base_module",
    )

    checkpoint_path = Path(
        args.base_checkpoint
    )

    threshold_path = Path(
        args.threshold_json
    )

    manifest_path = Path(
        args.manifest
    )

    for path in (
        checkpoint_path,
        threshold_path,
        manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    if str(
        threshold_data.get(
            "split_column",
            args.split_column,
        )
    ) != str(args.split_column):
        raise ValueError(
            "Threshold split-column mismatch."
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
        checkpoint_path,
        device,
    )

    hidden = infer_hidden(
        checkpoint
    )

    model = (
        base_module
        .AttentionPoolingSparseFieldModel(
            hidden_dim=hidden
        )
        .to(device)
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.eval()

    dataset = (
        base_module
        .SparseFieldDataset(
            manifest=args.manifest,
            split_column=args.split_column,
            split_label=args.split_label,
            t0_sec=args.t0_sec,
            input_stations=args.input_stations,
            target_stations=args.target_stations,
            input_pre_sec=args.input_pre_sec,
            seed=args.seed,
            training=False,
            repeats=args.repeats,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=False,
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pre_test_lock = {
        "base_checkpoint": str(
            checkpoint_path.resolve()
        ),
        "base_checkpoint_epoch": int(
            checkpoint.get(
                "epoch",
                -1,
            )
        ),
        "threshold_json": str(
            threshold_path.resolve()
        ),
        "split_column": args.split_column,
        "split_label": args.split_label,
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "architecture": (
            "attention_pooling_full"
        ),
        "risk_gate_used": False,
        "risk_gate_reason": (
            "No chronological validation epoch-gamma candidate "
            "satisfied all pre-specified deployment constraints."
        ),
        "model_selection_performed_here": False,
        "metric_aggregation": (
            "targets -> repeats -> events"
        ),
    }

    (
        out_dir
        / "pre_evaluation_lock.json"
    ).write_text(
        json.dumps(
            pre_test_lock,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== Locked chronological Attention-Base evaluation ==="
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
        "Checkpoint epoch    : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        "Tail thresholds     : "
        f"PGA={thresholds[0]:.6f}, "
        f"PGV={thresholds[1]:.6f}"
    )

    print(
        "Risk gate           : NOT DEPLOYED"
    )

    print(
        "Model selection     : NONE"
    )

    # Script 53 already implements the exact canonical evaluation.
    (
        base_metrics,
        predictions,
    ) = base_module.evaluate(
        model,
        loader,
        thresholds,
        device,
    )

    prediction_path = (
        out_dir
        / "locked_attention_base_predictions.csv"
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

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            {
                "split_column": args.split_column,
                "split_label": args.split_label,
                "n_events": int(
                    predictions[
                        "event_id"
                    ].nunique()
                ),
                "n_target_rows": int(
                    len(predictions)
                ),
                "repeats": int(args.repeats),
                "checkpoint_epoch": int(
                    checkpoint.get(
                        "epoch",
                        -1,
                    )
                ),
                "thresholds": {
                    "log10_pga": float(
                        thresholds[0]
                    ),
                    "log10_pgv": float(
                        thresholds[1]
                    ),
                },
                "risk_gate_used": False,
                "metric_aggregation": (
                    "targets -> repeats -> events"
                ),
                "base_module_metrics": base_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    key = metrics.loc[
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
        "\n=== Canonical locked metrics ==="
    )

    print(
        key.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {prediction_path.resolve()}"
    )

    print(
        f"  {metrics_path.resolve()}"
    )

    print(
        f"  {summary_path.resolve()}"
    )

    print(
        f"  {(out_dir / 'pre_evaluation_lock.json').resolve()}"
    )


if __name__ == "__main__":
    main()
