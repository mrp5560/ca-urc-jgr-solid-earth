#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
54_time_clean_attention_tail_joint_selection.py

Chronology-safe joint validation selection for:
    tail-head checkpoint epoch + power-gate exponent gamma

Base:
    checkpoint from 53_train_time_extrapolation_attention_base.py

Guarantees:
    - chronological train + validation only
    - 2021-2024 test NOT accessed
    - Ridgecrest OOD NOT accessed
    - chronological training-only Q90 thresholds
    - frozen Attention Base
    - save every tail-head epoch
    - 20-repeat validation epoch x gamma scan
    - canonical aggregation: targets -> repeats -> events
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 1.0:
            raise ValueError("gamma must be >= 1.")
        values.append(value)

    values = sorted(set(values))
    if not values:
        raise ValueError("No gamma values.")
    return values


def infer_hidden(checkpoint) -> int:
    args = checkpoint.get("args", {})
    if isinstance(args, dict) and args.get("hidden_dim") is not None:
        return int(args["hidden_dim"])

    key = "station_encoder.0.weight"
    if key not in checkpoint["model_state"]:
        raise KeyError(
            f"Cannot infer hidden dimension; missing {key}"
        )
    return int(
        checkpoint["model_state"][key].shape[0]
    )


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-module",
        default="53_train_time_extrapolation_attention_base.py",
    )
    p.add_argument(
        "--tail-module",
        default="48_train_attention_tail_risk_gated.py",
    )
    p.add_argument(
        "--selection-module",
        default="49_attention_tail_joint_validation_selection.py",
    )

    p.add_argument("--manifest", required=True)
    p.add_argument("--split-column", default="split_time_clean")
    p.add_argument("--train-label", default="train")
    p.add_argument("--validation-label", default="validation")

    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--threshold-json", required=True)

    p.add_argument("--t0-sec", type=int, default=5)
    p.add_argument("--input-stations", type=int, default=5)
    p.add_argument("--target-stations", type=int, default=10)
    p.add_argument("--input-pre-sec", type=float, default=2.0)

    p.add_argument("--training-validation-repeats", type=int, default=3)
    p.add_argument("--selection-validation-repeats", type=int, default=20)

    p.add_argument("--tail-hidden", type=int, default=64)
    p.add_argument("--maximum-correction", type=float, default=1.5)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--head-eval-batch-size", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)

    p.add_argument("--lambda-classification", type=float, default=0.25)
    p.add_argument("--lambda-regression", type=float, default=1.0)
    p.add_argument("--tail-regression-weight", type=float, default=2.0)
    p.add_argument("--lambda-under", type=float, default=0.50)
    p.add_argument("--lambda-leakage", type=float, default=0.02)
    p.add_argument("--max-positive-weight", type=float, default=9.0)

    p.add_argument(
        "--powers",
        default="1,1.5,2,2.5,3,3.5,4,5,6,7,8,9,10,12,15,20",
    )
    p.add_argument("--overall-budget", type=float, default=0.015)
    p.add_argument("--non-tail-budget", type=float, default=0.005)
    p.add_argument("--bias-limit", type=float, default=0.05)

    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--out-dir",
        default="runs/time_clean_attention_tail_joint_selection",
    )

    args = p.parse_args()

    powers = parse_float_list(args.powers)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device(args.device)

    set_seed(args.seed)

    base_module = load_module(
        args.base_module,
        "time_clean_attention_base_module",
    )
    tail_module = load_module(
        args.tail_module,
        "attention_tail_module",
    )
    selection_module = load_module(
        args.selection_module,
        "attention_selection_module",
    )

    threshold_path = Path(args.threshold_json)
    threshold_data = json.loads(
        threshold_path.read_text(encoding="utf-8")
    )

    if str(
        threshold_data.get("split_column", args.split_column)
    ) != str(args.split_column):
        raise ValueError(
            "Threshold JSON split-column mismatch."
        )

    if str(
        threshold_data.get("train_label", args.train_label)
    ) != str(args.train_label):
        raise ValueError(
            "Threshold JSON train-label mismatch."
        )

    thresholds_np = np.asarray(
        [
            threshold_data["log10_pga_threshold"],
            threshold_data["log10_pgv_threshold"],
        ],
        dtype=np.float32,
    )
    thresholds_tensor = torch.from_numpy(
        thresholds_np
    ).to(device)

    prevalence = float(
        np.clip(
            1.0 - float(threshold_data.get("quantile", 0.90)),
            0.01,
            0.50,
        )
    )

    base_checkpoint_path = Path(args.base_checkpoint)
    base_checkpoint = load_checkpoint(
        base_checkpoint_path,
        device,
    )

    hidden = infer_hidden(base_checkpoint)

    base = base_module.AttentionPoolingSparseFieldModel(
        hidden_dim=hidden
    ).to(device)

    base.load_state_dict(
        base_checkpoint["model_state"],
        strict=True,
    )
    base.eval()

    model = tail_module.TailGateModel(
        base=base,
        hidden=hidden,
        tail_hidden=args.tail_hidden,
        prevalence=prevalence,
        maximum_correction=args.maximum_correction,
    ).to(device)

    # Chronological dataset constructor uses split_label.
    train_dataset = base_module.SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.train_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=True,
    )

    validation_dataset = base_module.SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.training_validation_repeats,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    trainable = [
        x for x in model.parameters()
        if x.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_dir = out_dir / "epoch_checkpoints"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    print(
        "=== Chronological Attention tail-head joint validation selection ==="
    )
    print(f"Device                         : {device}")
    print("Base architecture              : attention_pooling_full")
    print(
        "Base checkpoint epoch          : "
        f"{base_checkpoint.get('epoch', 'unknown')}"
    )
    print(
        "Train/Validation events        : "
        f"{len(train_dataset.frame)}/{len(validation_dataset.frame)}"
    )
    print(
        "Training validation repeats    : "
        f"{args.training_validation_repeats}"
    )
    print(
        "Selection validation repeats   : "
        f"{args.selection_validation_repeats}"
    )
    print(
        "Chronological thresholds       : "
        f"PGA={thresholds_np[0]:.6f}, PGV={thresholds_np[1]:.6f}"
    )
    print(f"Gamma grid                     : {powers}")
    print(
        "Overall/non-tail budgets       : "
        f"{args.overall_budget:.4f} / {args.non_tail_budget:.4f}"
    )
    print(f"Absolute bias limit            : {args.bias_limit:.4f}")
    print(
        "Trainable tail parameters      : "
        f"{sum(x.numel() for x in trainable):,}"
    )
    print("2021-2024 test                 : NOT ACCESSED")
    print("Ridgecrest OOD                 : NOT ACCESSED")

    history = []

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)

        train_metrics = tail_module.train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            thresholds=thresholds_tensor,
            prevalence=prevalence,
            lambda_classification=args.lambda_classification,
            lambda_regression=args.lambda_regression,
            tail_regression_weight=args.tail_regression_weight,
            lambda_under=args.lambda_under,
            lambda_leakage=args.lambda_leakage,
            max_positive_weight=args.max_positive_weight,
        )

        validation_metrics, _ = tail_module.evaluate_model(
            model,
            validation_loader,
            thresholds_np,
            device,
        )

        scheduler.step(
            validation_metrics["selection_score"]
        )

        checkpoint_path = (
            epoch_dir / f"epoch_{epoch:03d}.pt"
        )

        tail_module.save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            args,
            validation_metrics,
            threshold_data,
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{k}": v
                    for k, v in train_metrics.items()
                },
                **{
                    f"validation_{k}": v
                    for k, v in validation_metrics.items()
                },
            }
        )

        pd.DataFrame(history).to_csv(
            out_dir / "training_history.csv",
            index=False,
        )

        print(
            "Epoch %03d | "
            "linear overall=%.4f/%.4f | "
            "linear tail=%.4f/%.4f | "
            "tail U05=%.3f/%.3f | "
            "AUPRC=%.3f/%.3f"
            % (
                epoch,
                validation_metrics["linear_macro_mae_pga"],
                validation_metrics["linear_macro_mae_pgv"],
                validation_metrics["linear_macro_tail_mae_pga"],
                validation_metrics["linear_macro_tail_mae_pgv"],
                validation_metrics["linear_tail_under05_pga"],
                validation_metrics["linear_tail_under05_pgv"],
                validation_metrics["approx_auprc_pga"],
                validation_metrics["approx_auprc_pgv"],
            )
        )

    selection_dataset = base_module.SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.selection_validation_repeats,
    )

    selection_loader = DataLoader(
        selection_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    print(
        "\nBuilding frozen-base 20-repeat chronological validation cache..."
    )

    cache = selection_module.build_validation_cache(
        module=tail_module,
        model=model,
        dataset=selection_dataset,
        loader=selection_loader,
        thresholds_np=thresholds_np,
        device=device,
    )

    base_metrics = selection_module.base_metric_bundle(
        cache
    )

    (out_dir / "base_validation_metrics.json").write_text(
        json.dumps(base_metrics, indent=2),
        encoding="utf-8",
    )

    print(
        "Cached validation target rows    : "
        f"{len(cache['event_id'])}"
    )
    print(
        "Base overall MAE                 : "
        f"PGA={base_metrics['overall_mae_pga']:.4f}, "
        f"PGV={base_metrics['overall_mae_pgv']:.4f}"
    )
    print(
        "Base tail MAE                    : "
        f"PGA={base_metrics['tail_mae_pga']:.4f}, "
        f"PGV={base_metrics['tail_mae_pgv']:.4f}"
    )

    scan_rows = []
    head_cache = {}

    for epoch in range(1, args.epochs + 1):
        checkpoint = load_checkpoint(
            epoch_dir / f"epoch_{epoch:03d}.pt",
            device,
        )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )
        model.eval()

        probability, residual = (
            selection_module.predict_tail_head_from_cache(
                model=model,
                context_input=cache["context_input"],
                device=device,
                batch_size=args.head_eval_batch_size,
            )
        )

        head_cache[epoch] = (
            probability,
            residual,
        )

        for gamma in powers:
            candidate = selection_module.candidate_metrics(
                cache,
                probability,
                residual,
                gamma,
            )

            status = selection_module.feasibility_status(
                candidate,
                base_metrics,
                args.overall_budget,
                args.non_tail_budget,
                args.bias_limit,
            )

            scan_rows.append(
                {
                    "epoch": epoch,
                    "gamma": gamma,
                    **candidate,
                    **status,
                }
            )

    scan = pd.DataFrame(scan_rows)
    scan_path = out_dir / "epoch_gamma_validation_scan.csv"
    scan.to_csv(scan_path, index=False)

    feasible = scan.loc[
        scan["feasible"].astype(bool)
    ].copy()

    if feasible.empty:
        print(
            "\nNO feasible chronological epoch-gamma pair."
        )
        print(
            "Do NOT evaluate 2021-2024 test or Ridgecrest."
        )
        raise RuntimeError(
            f"Inspect {scan_path.resolve()}"
        )

    feasible = feasible.sort_values(
        [
            "mean_tail_mae",
            "mean_overall_mae",
            "gamma",
            "epoch",
        ]
    )

    selected = feasible.iloc[0]
    selected_epoch = int(selected["epoch"])
    selected_gamma = float(selected["gamma"])

    upper_boundary = bool(
        np.isclose(
            selected_gamma,
            max(powers),
        )
    )

    source_checkpoint = (
        epoch_dir / f"epoch_{selected_epoch:03d}.pt"
    )

    selected_checkpoint = (
        out_dir / "selected_tail_head.pt"
    )

    shutil.copy2(
        source_checkpoint,
        selected_checkpoint,
    )

    probability, residual = (
        head_cache[selected_epoch]
    )

    prediction_path = (
        out_dir / "selected_validation_predictions.csv"
    )

    selection_module.save_selected_predictions(
        prediction_path,
        cache,
        probability,
        residual,
        selected_gamma,
    )

    feasible.head(30).to_csv(
        out_dir / "top_feasible_validation_candidates.csv",
        index=False,
    )

    payload = {
        "selected_epoch": selected_epoch,
        "selected_gamma": selected_gamma,
        "selected_at_upper_gamma_boundary": upper_boundary,
        "gamma_grid": powers,
        "selection_objective": (
            "minimum mean PGA/PGV high-motion-tail MAE "
            "among validation-feasible candidates"
        ),
        "constraints": {
            "overall_budget": args.overall_budget,
            "non_tail_budget": args.non_tail_budget,
            "bias_limit": args.bias_limit,
        },
        "metric_aggregation": "targets -> repeats -> events",
        "base_metrics": base_metrics,
        "selected_metrics": {
            key: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else (
                    int(value)
                    if isinstance(value, (int, np.integer))
                    else (
                        float(value)
                        if isinstance(value, (float, np.floating))
                        else value
                    )
                )
            )
            for key, value in selected.to_dict().items()
        },
        "base_checkpoint": str(
            base_checkpoint_path.resolve()
        ),
        "selected_tail_checkpoint": str(
            selected_checkpoint.resolve()
        ),
        "2021_2024_test_evaluated": False,
        "ridgecrest_ood_evaluated": False,
    }

    json_path = (
        out_dir / "selected_epoch_gamma.json"
    )
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== Selected chronological validation-only epoch + gamma ==="
    )
    print(f"Epoch                   : {selected_epoch}")
    print(f"Gamma                   : {selected_gamma:.3f}")
    print(f"At upper grid boundary  : {upper_boundary}")
    print(
        "Overall MAE PGA/PGV     : "
        f"{selected['overall_mae_pga']:.4f} / "
        f"{selected['overall_mae_pgv']:.4f}"
    )
    print(
        "Overall delta PGA/PGV   : "
        f"{selected['overall_delta_pga']:+.4f} / "
        f"{selected['overall_delta_pgv']:+.4f}"
    )
    print(
        "Non-tail delta PGA/PGV  : "
        f"{selected['non_tail_delta_pga']:+.4f} / "
        f"{selected['non_tail_delta_pgv']:+.4f}"
    )
    print(
        "Tail MAE PGA/PGV        : "
        f"{selected['tail_mae_pga']:.4f} / "
        f"{selected['tail_mae_pgv']:.4f}"
    )
    print(
        "Tail delta PGA/PGV      : "
        f"{selected['tail_delta_pga']:+.4f} / "
        f"{selected['tail_delta_pgv']:+.4f}"
    )
    print(
        "Tail U0.5 PGA/PGV       : "
        f"{selected['tail_under05_pga']:.3f} / "
        f"{selected['tail_under05_pgv']:.3f}"
    )
    print(
        "Overall bias PGA/PGV    : "
        f"{selected['bias_pga']:+.4f} / "
        f"{selected['bias_pgv']:+.4f}"
    )
    print(
        "Feasible candidates     : "
        f"{len(feasible)} / {len(scan)}"
    )
    print(
        f"Selected checkpoint     : {selected_checkpoint.resolve()}"
    )
    print(
        f"Selection JSON          : {json_path.resolve()}"
    )
    print(
        f"Validation predictions  : {prediction_path.resolve()}"
    )
    print(
        f"Full scan               : {scan_path.resolve()}"
    )
    print("2021-2024 test          : NOT ACCESSED")
    print("Ridgecrest OOD          : NOT ACCESSED")

    if upper_boundary:
        print(
            "\nWARNING: selected gamma is at the upper grid boundary."
        )
        print(
            "Do NOT evaluate test yet; extend gamma on validation only."
        )
    else:
        print(
            "\nChronological epoch/gamma selection is now locked."
        )


if __name__ == "__main__":
    main()
