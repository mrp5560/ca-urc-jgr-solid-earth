#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 1 architecture decision audit.

Re-evaluates EXISTING best checkpoints from
40_phase1_final_mean_pooling_ablation_suite.py on the validation split only.

Purpose:
- do NOT use the test set to choose the final architecture;
- compare base / no_p_offset / attention_pooling with more repeated
  validation station draws;
- report overall and training-Q90 high-motion-tail metrics;
- no retraining is performed.

Main aggregation:
targets -> repeats -> events.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


LOG10_FACTOR_2 = math.log10(2.0)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "phase1_ablation_module",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
        .to_numpy(dtype=bool)
    )


def collect_predictions(
    model,
    loader,
    device,
) -> pd.DataFrame:
    model.eval()
    rows = []

    with torch.inference_mode():
        for batch in loader:
            pred = model(
                batch["input_waveforms"].to(device),
                batch["input_features"].to(device),
                batch["target_features"].to(device),
            ).cpu().numpy()

            truth = batch["target_log"].numpy()
            event_ids = list(batch["event_id"])
            repeats = batch["repeat"].numpy()

            for b in range(truth.shape[0]):
                for q in range(truth.shape[1]):
                    rows.append(
                        {
                            "event_id": str(event_ids[b]),
                            "repeat": int(repeats[b]),
                            "target_slot": int(q),
                            "true_log10_pga": float(truth[b, q, 0]),
                            "true_log10_pgv": float(truth[b, q, 1]),
                            "pred_log10_pga": float(pred[b, q, 0]),
                            "pred_log10_pgv": float(pred[b, q, 1]),
                        }
                    )

    return pd.DataFrame(rows)


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
    if metric == "under05":
        return (residual <= -0.5).astype(float)

    raise ValueError(metric)


def canonical_metric(
    frame: pd.DataFrame,
    quantity: str,
    metric: str,
) -> float:
    truth = frame[
        f"true_log10_{quantity}"
    ].to_numpy(dtype=float)
    pred = frame[
        f"pred_log10_{quantity}"
    ].to_numpy(dtype=float)

    values = element_metric(
        truth,
        pred,
        metric,
    )

    work = frame[["event_id", "repeat"]].copy()
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

    return float(event.mean())


def summarize(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    variant: str,
) -> pd.DataFrame:
    rows = []

    for quantity in ("pga", "pgv"):
        truth = frame[
            f"true_log10_{quantity}"
        ].to_numpy(dtype=float)

        masks = {
            "overall": np.ones(len(frame), dtype=bool),
            "high_motion_tail": (
                truth >= float(thresholds[quantity])
            ),
        }

        for population, mask in masks.items():
            subset = frame.loc[mask].copy()
            if subset.empty:
                continue

            for metric in (
                "mae",
                "bias",
                "factor2",
                "under05",
            ):
                rows.append(
                    {
                        "variant": variant,
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "value": canonical_metric(
                            subset,
                            quantity,
                            metric,
                        ),
                        "n_events": int(
                            subset["event_id"].nunique()
                        ),
                        "n_event_repeats": int(
                            subset[
                                ["event_id", "repeat"]
                            ].drop_duplicates().shape[0]
                        ),
                        "n_target_rows": int(len(subset)),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ablation-script",
        default="40_phase1_final_mean_pooling_ablation_suite.py",
    )
    p.add_argument(
        "--manifest",
        default="data/model_manifests/scenario_t0_5s_k5.csv",
    )
    p.add_argument(
        "--checkpoint-root",
        default="runs/phase1_final_mean_pooling_ablation",
    )
    p.add_argument(
        "--threshold-json",
        default=(
            "runs/tail_gated_compromise_t0_5s_k5/"
            "tail_thresholds_q0.90_t0_5s.json"
        ),
    )
    p.add_argument(
        "--variants",
        default="base,no_p_offset,attention_pooling",
    )
    p.add_argument(
        "--split-column",
        default="split_grouped",
    )
    p.add_argument("--t0-sec", type=int, default=5)
    p.add_argument("--input-stations", type=int, default=5)
    p.add_argument("--target-stations", type=int, default=10)
    p.add_argument("--input-pre-sec", type=float, default=2.0)
    p.add_argument("--validation-repeats", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--out-dir",
        default="runs/phase1_validation_architecture_audit",
    )

    args = p.parse_args()

    script_path = Path(args.ablation_script)
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    threshold_path = Path(args.threshold_json)
    if not threshold_path.exists():
        raise FileNotFoundError(threshold_path)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device(args.device)

    module = load_module(script_path)

    threshold_data = json.loads(
        threshold_path.read_text(encoding="utf-8")
    )
    thresholds = {
        "pga": float(threshold_data["log10_pga_threshold"]),
        "pgv": float(threshold_data["log10_pgv_threshold"]),
    }

    variants = [
        x.strip()
        for x in args.variants.split(",")
        if x.strip()
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    print("=== Phase 1 validation-only architecture audit ===")
    print(f"Device              : {device}")
    print(f"Validation repeats  : {args.validation_repeats}")
    print(
        "Thresholds          : "
        f"PGA={thresholds['pga']:.4f}, "
        f"PGV={thresholds['pgv']:.4f}"
    )

    for variant in variants:
        checkpoint_path = (
            Path(args.checkpoint_root)
            / variant
            / "best_model.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
            )

        checkpoint_args = checkpoint.get("args", {})
        hidden_dim = int(
            checkpoint_args.get("hidden_dim", 128)
        )

        model = module.FinalAblationModel(
            variant=variant,
            hidden_dim=hidden_dim,
        ).to(device)
        model.load_state_dict(
            checkpoint["model_state"]
        )

        dataset = module.SparseFieldDataset(
            manifest=args.manifest,
            split_column=args.split_column,
            split_name="validation",
            t0_sec=args.t0_sec,
            input_stations=args.input_stations,
            target_stations=args.target_stations,
            input_pre_sec=args.input_pre_sec,
            seed=args.seed,
            training=False,
            repeats=args.validation_repeats,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        predictions = collect_predictions(
            model,
            loader,
            device,
        )
        predictions.to_csv(
            out_dir / f"{variant}_validation_predictions.csv",
            index=False,
        )

        metrics = summarize(
            predictions,
            thresholds,
            variant,
        )
        all_metrics.append(metrics)

        overall = metrics.loc[
            (metrics["population"] == "overall")
            & (metrics["metric"] == "mae")
        ]
        tail = metrics.loc[
            (metrics["population"] == "high_motion_tail")
            & (metrics["metric"].isin(["mae", "under05"]))
        ]

        print(f"\n--- {variant} ---")
        print(
            "Checkpoint epoch    : "
            f"{checkpoint.get('epoch', 'unknown')}"
        )
        print("Overall validation MAE:")
        print(
            overall[
                ["quantity", "value", "n_events", "n_target_rows"]
            ].to_string(index=False)
        )
        print("High-motion validation:")
        print(
            tail[
                ["quantity", "metric", "value", "n_events", "n_target_rows"]
            ].to_string(index=False)
        )

    summary = pd.concat(
        all_metrics,
        ignore_index=True,
    )
    summary_path = out_dir / "validation_architecture_metrics.csv"
    summary.to_csv(summary_path, index=False)

    # Compact architecture decision table.
    compact_rows = []
    for variant in variants:
        sub = summary.loc[
            summary["variant"].eq(variant)
        ]

        def get(quantity, population, metric):
            row = sub.loc[
                sub["quantity"].eq(quantity)
                & sub["population"].eq(population)
                & sub["metric"].eq(metric)
            ]
            return float(row.iloc[0]["value"])

        pga_mae = get("pga", "overall", "mae")
        pgv_mae = get("pgv", "overall", "mae")

        compact_rows.append(
            {
                "variant": variant,
                "overall_mae_pga": pga_mae,
                "overall_mae_pgv": pgv_mae,
                "overall_selection_score": 0.5 * (
                    pga_mae + pgv_mae
                ),
                "tail_mae_pga": get(
                    "pga", "high_motion_tail", "mae"
                ),
                "tail_mae_pgv": get(
                    "pgv", "high_motion_tail", "mae"
                ),
                "tail_under05_pga": get(
                    "pga", "high_motion_tail", "under05"
                ),
                "tail_under05_pgv": get(
                    "pgv", "high_motion_tail", "under05"
                ),
                "overall_bias_pga": get(
                    "pga", "overall", "bias"
                ),
                "overall_bias_pgv": get(
                    "pgv", "overall", "bias"
                ),
            }
        )

    compact = pd.DataFrame(compact_rows).sort_values(
        "overall_selection_score"
    )
    compact_path = out_dir / "architecture_decision_table.csv"
    compact.to_csv(compact_path, index=False)

    print("\n=== Validation-only architecture decision table ===")
    print(compact.to_string(index=False))
    print(f"\nMetrics : {summary_path.resolve()}")
    print(f"Decision: {compact_path.resolve()}")


if __name__ == "__main__":
    main()
