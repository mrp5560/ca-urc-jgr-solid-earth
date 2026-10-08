#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
73_evaluate_frozen_caurc_ridgecrest_ood.py

Evaluate the already locked chronological CA-URC on the complete 2019
Ridgecrest sequence-level OOD split.

NO training. NO threshold recalculation. NO epoch/gamma selection.

Frozen source:
    runs/caurc_chronological_2021_2024/

Model:
    CA-URC = Cross-Attention Underprediction-Risk Correction
    y_final = y_CA + p_under**gamma * Delta

Development split:
    train      : 2010-2018
    validation : 2019 non-Ridgecrest + 2020
    test       : 2021-2024
    OOD        : complete held-out 2019 Ridgecrest sequence

Metrics:
    targets -> repeats -> events

Statistics:
    paired EVENT-level bootstrap within the single held-out Ridgecrest sequence.
    A sequence-cluster bootstrap is not identifiable because Ridgecrest is one
    held-out sequence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_module(path: str | Path, name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        return torch.load(
            str(path), map_location=device, weights_only=False
        )
    except TypeError:
        return torch.load(str(path), map_location=device)


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_frozen_model(
    args,
    baseline_module,
    ablation_module,
    device: torch.device,
):
    base_ckpt = load_checkpoint(args.base_checkpoint, device)
    head_ckpt = load_checkpoint(args.head_checkpoint, device)
    selection = load_json(args.selection_json)
    threshold_info = load_json(args.threshold_json)

    if not bool(base_ckpt.get("chronological_training", False)):
        raise RuntimeError(
            "Base checkpoint is not chronology-trained."
        )
    if str(base_ckpt.get("variant", "")) != "cross_attention":
        raise RuntimeError(
            "Base checkpoint is not Cross-Attention."
        )
    if str(head_ckpt.get("variant", "")) != "A4_under_only":
        raise RuntimeError(
            "Head checkpoint is not A4_under_only / CA-URC."
        )
    if not bool(head_ckpt.get("chronological_training", False)):
        raise RuntimeError(
            "CA-URC head is not chronology-trained."
        )
    if str(selection.get("model_name", "")) != "CA-URC":
        raise RuntimeError(
            "Selection JSON does not describe CA-URC."
        )
    if str(selection.get("variant", "")) != "A4_under_only":
        raise RuntimeError(
            "Selection JSON variant is not A4_under_only."
        )
    if bool(
        selection.get("selected_at_upper_gamma_boundary", True)
    ):
        raise RuntimeError(
            "Selected gamma is at upper validation boundary; "
            "do not access Ridgecrest."
        )
    if not bool(selection.get("test_accessed", False)):
        raise RuntimeError(
            "2021-2024 locked chronological test is not recorded "
            "as completed."
        )

    selected_epoch = int(selection["selected_epoch"])
    selected_gamma = float(selection["selected_gamma"])

    if int(head_ckpt.get("epoch", -1)) != selected_epoch:
        raise RuntimeError(
            "Selected head checkpoint epoch != selection JSON epoch."
        )

    thresholds = np.asarray(
        [
            threshold_info["log10_pga_threshold"],
            threshold_info["log10_pgv_threshold"],
        ],
        dtype=np.float32,
    )

    base = baseline_module.make_model(
        "cross_attention",
        args.hidden_dim,
        args.attention_heads,
        50.0,
    ).to(device)
    base.load_state_dict(base_ckpt["model_state"], strict=True)
    base.eval()

    tail_prev = np.asarray(
        head_ckpt.get("tail_prevalence", [0.10, 0.10]),
        dtype=float,
    )
    under_prev = np.asarray(
        head_ckpt.get("under_prevalence", [0.10, 0.10]),
        dtype=float,
    )

    model = ablation_module.AblationRiskGate(
        base=base,
        variant="A4_under_only",
        hidden_dim=args.hidden_dim,
        risk_hidden=args.risk_hidden,
        maximum_correction=args.maximum_correction,
        tail_prevalence=tail_prev,
        under_prevalence=under_prev,
    ).to(device)

    model.load_state_dict(
        head_ckpt["model_state"], strict=True
    )
    model.eval()

    audit = {
        "model": "CA-URC",
        "variant": "A4_under_only",
        "formula": "y_CA + p_under^gamma * Delta",
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "selection_json": str(Path(args.selection_json).resolve()),
        "threshold_json": str(Path(args.threshold_json).resolve()),
        "base_epoch": int(base_ckpt.get("epoch", -1)),
        "head_epoch": selected_epoch,
        "gamma": selected_gamma,
        "gamma_upper_boundary": False,
        "log10_pga_threshold": float(thresholds[0]),
        "log10_pgv_threshold": float(thresholds[1]),
        "training_performed": False,
        "selection_performed": False,
        "threshold_recomputed": False,
    }
    return model, selected_gamma, thresholds, audit


def make_shift_summary(
    ridge_metrics: pd.DataFrame,
    chronological_metrics_path: Path,
) -> pd.DataFrame:
    if not chronological_metrics_path.exists():
        return pd.DataFrame()

    chrono = pd.read_csv(chronological_metrics_path)
    ridge = ridge_metrics.copy()

    keys = ["model", "quantity", "population", "metric"]
    chrono = chrono[keys + ["value"]].rename(
        columns={"value": "chronological_2021_2024_value"}
    )
    ridge = ridge[keys + ["value"]].rename(
        columns={"value": "ridgecrest_value"}
    )

    result = chrono.merge(
        ridge, on=keys, how="inner", validate="one_to_one"
    )
    result["ridgecrest_minus_chronological"] = (
        result["ridgecrest_value"]
        - result["chronological_2021_2024_value"]
    )
    result["relative_shift_percent"] = np.nan

    mask = result["metric"].astype(str).eq("mae")
    result.loc[mask, "relative_shift_percent"] = (
        100.0
        * result.loc[mask, "ridgecrest_minus_chronological"]
        / result.loc[
            mask, "chronological_2021_2024_value"
        ].abs().clip(lower=1e-12)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-module",
        default="45_phase2_strong_baseline_suite.py",
    )
    parser.add_argument(
        "--ablation-module",
        default="63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py",
    )
    parser.add_argument(
        "--chrono-module",
        default="68_train_and_evaluate_cadrg_chronological_ridgecrest.py",
    )
    parser.add_argument(
        "--caurc-chrono-module",
        default="72_train_and_evaluate_caurc_chronological_2021_2024.py",
    )

    parser.add_argument(
        "--manifest",
        default="data/model_manifests/scenario_t0_5s_k5_time_clean.csv",
    )
    parser.add_argument(
        "--h5-root",
        default="data/processed_full_v4/events",
    )
    parser.add_argument(
        "--split-column", default="split_time_clean"
    )
    parser.add_argument(
        "--ridgecrest-label", default="ridgecrest_ood"
    )

    parser.add_argument(
        "--base-checkpoint",
        default=(
            "runs/caurc_chronological_2021_2024/"
            "cross_attention_base/best_model.pt"
        ),
    )
    parser.add_argument(
        "--head-checkpoint",
        default=(
            "runs/caurc_chronological_2021_2024/"
            "ca_urc_head/selected_ca_urc_head.pt"
        ),
    )
    parser.add_argument(
        "--selection-json",
        default=(
            "runs/caurc_chronological_2021_2024/"
            "ca_urc_head/selected_epoch_gamma.json"
        ),
    )
    parser.add_argument(
        "--threshold-json",
        default=(
            "runs/caurc_chronological_2021_2024/"
            "chronological_tail_thresholds.json"
        ),
    )
    parser.add_argument(
        "--chronological-metrics",
        default=(
            "runs/caurc_chronological_2021_2024/"
            "chronological_2021_2024_metrics.csv"
        ),
    )

    parser.add_argument("--t0-sec", type=int, default=5)
    parser.add_argument("--input-stations", type=int, default=5)
    parser.add_argument("--target-stations", type=int, default=10)
    parser.add_argument("--input-pre-sec", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--risk-hidden", type=int, default=64)
    parser.add_argument(
        "--maximum-correction", type=float, default=1.5
    )

    # Helper-interface arguments. They are NOT used for training.
    parser.add_argument("--base-epochs", type=int, default=50)
    parser.add_argument(
        "--base-validation-repeats", type=int, default=3
    )
    parser.add_argument(
        "--base-learning-rate", type=float, default=1e-3
    )
    parser.add_argument(
        "--base-weight-decay", type=float, default=1e-4
    )
    parser.add_argument(
        "--base-minimum-learning-rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--base-lr-patience", type=int, default=4
    )
    parser.add_argument(
        "--base-early-stopping-patience", type=int, default=12
    )
    parser.add_argument(
        "--base-minimum-delta", type=float, default=1e-4
    )

    parser.add_argument("--test-repeats", type=int, default=20)
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=10000
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/caurc_ridgecrest_ood",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device(args.device)

    if args.bootstrap_repetitions < 1000:
        raise ValueError(
            "--bootstrap-repetitions should be >= 1000."
        )

    baseline_module = load_module(
        args.baseline_module, "ridge_baseline"
    )
    ablation_module = load_module(
        args.ablation_module, "ridge_ablation"
    )
    chrono_module = load_module(
        args.chrono_module, "ridge_chrono"
    )
    caurc_chrono_module = load_module(
        args.caurc_chrono_module, "ridge_caurc_chrono"
    )

    chrono_module.set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        args.manifest, dtype={"event_id": str}
    )
    split_counts = (
        manifest[args.split_column]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    ridge_count = int(
        manifest[args.split_column]
        .astype(str)
        .eq(args.ridgecrest_label)
        .sum()
    )

    print("=== Frozen CA-URC Ridgecrest Sequence-Level OOD ===")
    print(f"Device                    : {device}")
    print(f"Manifest split counts     : {split_counts}")
    print(f"Ridgecrest events         : {ridge_count}")
    print("Training                  : NONE")
    print("Model selection           : NONE")
    print("Threshold recalculation   : NONE")

    start = time.time()

    model, gamma, thresholds, audit = reconstruct_frozen_model(
        args, baseline_module, ablation_module, device
    )

    audit.update(
        {
            "manifest": str(Path(args.manifest).resolve()),
            "split_column": args.split_column,
            "ridgecrest_label": args.ridgecrest_label,
            "ridgecrest_manifest_events": ridge_count,
            "split_counts": split_counts,
            "metric_aggregation": "targets -> repeats -> events",
            "bootstrap_unit": (
                "event within one held-out Ridgecrest sequence"
            ),
            "sequence_cluster_bootstrap": False,
        }
    )
    audit_path = out_dir / "frozen_model_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("\n=== Frozen artifact audit ===")
    print(f"Base epoch                : {audit['base_epoch']}")
    print(f"CA-URC head epoch         : {audit['head_epoch']}")
    print(f"Gamma                     : {gamma:g}")
    print(
        "Q90 PGA/PGV               : "
        f"{thresholds[0]:.6f}/{thresholds[1]:.6f}"
    )

    # Reproduce the deterministic draw convention used by script 72.
    ablation_module.patch_train_validation_seed_protocol(
        baseline_module
    )

    # evaluate_locked_test() evaluates whatever label is in args.test_label.
    # Set it to Ridgecrest only AFTER all frozen artifact checks are complete.
    args.test_label = args.ridgecrest_label

    print("\n=== FIRST LOCKED RIDGECREST ACCESS ===")
    predictions = caurc_chrono_module.evaluate_locked_test(
        args,
        baseline_module,
        ablation_module,
        chrono_module,
        model,
        gamma,
        thresholds,
        device,
    )

    pred_path = out_dir / "ridgecrest_ood_predictions.csv"
    predictions.to_csv(pred_path, index=False)

    expected_rows = (
        ridge_count * args.test_repeats * args.target_stations
    )
    print(f"Prediction rows           : {len(predictions):,}")
    print(f"Expected rows             : {expected_rows:,}")
    print(
        f"Unique Ridgecrest events  : "
        f"{predictions['event_id'].nunique():,}"
    )

    if len(predictions) != expected_rows:
        raise RuntimeError(
            "Ridgecrest row-count audit failed: "
            f"got={len(predictions)}, expected={expected_rows}."
        )

    metrics = chrono_module.metric_table(predictions)
    metrics["model"] = metrics["model"].replace(
        {"ca_drg": "ca_urc"}
    )
    metrics_path = out_dir / "ridgecrest_ood_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    deltas = chrono_module.cadrg_minus_base_table(predictions)
    deltas = deltas.rename(
        columns={
            "ca_drg_value": "ca_urc_value",
            "mean_delta_cadrg_minus_base":
                "mean_delta_caurc_minus_base",
            "median_delta_cadrg_minus_base":
                "median_delta_caurc_minus_base",
        }
    )
    delta_path = (
        out_dir / "caurc_minus_cross_attention_ridgecrest.csv"
    )
    deltas.to_csv(delta_path, index=False)

    bootstrap = caurc_chrono_module.paired_event_bootstrap(
        predictions,
        chrono_module,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 11003,
    )
    bootstrap["sequence"] = "2019_Ridgecrest"
    bootstrap["bootstrap_unit"] = (
        "event_within_single_heldout_sequence"
    )
    bootstrap_path = (
        out_dir / "paired_event_bootstrap_ridgecrest.csv"
    )
    bootstrap.to_csv(bootstrap_path, index=False)

    shift = make_shift_summary(
        metrics,
        Path(args.chronological_metrics),
    )
    shift_path = (
        out_dir / "chronological_vs_ridgecrest_summary.csv"
    )
    if not shift.empty:
        shift.to_csv(shift_path, index=False)

    access_record = {
        "model": "CA-URC",
        "variant": "A4_under_only",
        "base_epoch": audit["base_epoch"],
        "head_epoch": audit["head_epoch"],
        "gamma": gamma,
        "training_performed": False,
        "selection_performed": False,
        "threshold_recomputed": False,
        "ridgecrest_accessed": True,
        "ridgecrest_events": int(
            predictions["event_id"].nunique()
        ),
        "ridgecrest_rows": int(len(predictions)),
    }
    access_path = (
        out_dir / "ridgecrest_locked_access_record.json"
    )
    access_path.write_text(
        json.dumps(access_record, indent=2),
        encoding="utf-8",
    )

    summary = {
        "status": "completed",
        **access_record,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": (
            "event within single held-out Ridgecrest sequence"
        ),
        "elapsed_seconds": float(time.time() - start),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Ridgecrest canonical metrics ===")
    print(
        metrics.loc[
            metrics["metric"].isin(
                ["mae", "bias", "factor2", "under05"]
            ),
            [
                "model",
                "quantity",
                "population",
                "metric",
                "value",
                "n_events",
                "n_target_rows",
            ],
        ].to_string(index=False)
    )

    print(
        "\n=== CA-URC minus Cross-Attention Base: Ridgecrest ==="
    )
    print(
        deltas.loc[
            deltas["metric"].isin(
                ["mae", "factor2", "under05"]
            )
        ].to_string(index=False)
    )

    print(
        "\n=== Paired event bootstrap within Ridgecrest sequence ==="
    )
    print(
        bootstrap.loc[
            bootstrap["metric"].isin(
                ["mae", "factor2", "under05"]
            )
        ].to_string(index=False)
    )

    if not shift.empty:
        print(
            "\n=== 2021-2024 vs Ridgecrest MAE shift ==="
        )
        print(
            shift.loc[
                shift["metric"].eq("mae")
            ].to_string(index=False)
        )

    print("\nOutputs:")
    output_paths = [
        audit_path,
        pred_path,
        metrics_path,
        delta_path,
        bootstrap_path,
        access_path,
        summary_path,
    ]
    if not shift.empty:
        output_paths.append(shift_path)

    for path in output_paths:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
