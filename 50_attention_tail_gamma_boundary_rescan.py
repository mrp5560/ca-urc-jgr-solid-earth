#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
50_attention_tail_gamma_boundary_rescan.py

Validation-only boundary audit for the attention-tail power gate.

The previous joint validation search selected gamma=6.0, exactly the upper
edge of the pre-specified grid. Before locking the test evaluation, expand
ONLY the gamma grid while keeping:
    - the trained epoch checkpoints fixed,
    - validation data fixed,
    - all performance budgets fixed,
    - the selection objective fixed.

NO retraining is performed.
NO test data are accessed.

Default extended grid:
    1,1.5,2,2.5,3,3.5,4,5,6,7,8,9,10,12,15,20

Selection:
    among feasible epoch-gamma pairs, minimize mean PGA/PGV tail MAE,
    tie-break by mean overall MAE, lower gamma, earlier epoch.

Constraints (unchanged):
    overall MAE increase <= 0.015 for each quantity
    non-tail MAE increase <= 0.005 for each quantity
    |overall bias| <= 0.05 for each quantity

Metric aggregation:
    targets -> repeats -> events.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import %s" % path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 1.0:
            raise ValueError("All gamma values must be >= 1.")
        values.append(value)

    values = sorted(set(values))
    if not values:
        raise ValueError("At least one gamma value is required.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--joint-script",
        default="49_attention_tail_joint_validation_selection.py",
    )
    parser.add_argument(
        "--training-module",
        default="48_train_attention_tail_risk_gated.py",
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
        "--validation-label",
        default="validation",
    )
    parser.add_argument(
        "--base-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--threshold-json",
        required=True,
    )
    parser.add_argument(
        "--epoch-checkpoint-dir",
        default=(
            "runs/"
            "attention_tail_joint_validation_selection/"
            "epoch_checkpoints"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
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
        "--validation-repeats",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--tail-hidden",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--maximum-correction",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--powers",
        default=(
            "1,1.5,2,2.5,3,3.5,4,5,6,"
            "7,8,9,10,12,15,20"
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
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--head-eval-batch-size",
        type=int,
        default=8192,
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
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "attention_tail_gamma_boundary_rescan"
        ),
    )

    args = parser.parse_args()

    if args.overall_budget <= 0:
        raise ValueError("--overall-budget must be positive.")
    if args.non_tail_budget <= 0:
        raise ValueError("--non-tail-budget must be positive.")
    if args.bias_limit <= 0:
        raise ValueError("--bias-limit must be positive.")

    powers = parse_float_list(args.powers)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device(args.device)

    joint_module = load_module(
        Path(args.joint_script),
        "joint_validation_module",
    )
    training_module = load_module(
        Path(args.training_module),
        "attention_tail_training_module",
    )

    training_module.set_global_seed(args.seed)

    threshold_data = json.loads(
        Path(args.threshold_json).read_text(
            encoding="utf-8"
        )
    )

    if (
        str(
            threshold_data.get(
                "split_column",
                args.split_column,
            )
        )
        != str(args.split_column)
    ):
        raise ValueError("Threshold split-column mismatch.")

    thresholds_np = np.asarray(
        [
            threshold_data["log10_pga_threshold"],
            threshold_data["log10_pgv_threshold"],
        ],
        dtype=np.float32,
    )

    base_checkpoint = training_module.load_checkpoint(
        args.base_checkpoint,
        device,
    )

    hidden = training_module._infer_base_hidden(
        base_checkpoint
    )

    base = training_module.MeanBase(
        hidden
    ).to(device)

    base.load_state_dict(
        base_checkpoint["model_state"],
        strict=True,
    )
    base.eval()

    model = training_module.TailGateModel(
        base=base,
        hidden=hidden,
        tail_hidden=args.tail_hidden,
        prevalence=float(
            np.clip(
                1.0
                - float(
                    threshold_data.get(
                        "quantile",
                        0.90,
                    )
                ),
                0.01,
                0.50,
            )
        ),
        maximum_correction=args.maximum_correction,
    ).to(device)

    validation_dataset = training_module.SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_name=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.validation_repeats,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": False,
    }

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    output_directory = Path(args.out_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_directory = Path(
        args.epoch_checkpoint_dir
    )
    if not epoch_directory.exists():
        raise FileNotFoundError(epoch_directory)

    checkpoint_paths = []
    for epoch in range(
        1,
        args.epochs + 1,
    ):
        path = (
            epoch_directory
            / ("epoch_%03d.pt" % epoch)
        )
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint_paths.append(
            (epoch, path)
        )

    print("=== Attention tail gamma-boundary rescan ===")
    print("Device                    : %s" % device)
    print(
        "Validation events         : %d"
        % len(validation_dataset.frame)
    )
    print(
        "Validation repeats        : %d"
        % args.validation_repeats
    )
    print(
        "Existing epoch checkpoints: %d"
        % len(checkpoint_paths)
    )
    print(
        "Extended gamma grid       : %s"
        % powers
    )
    print(
        "Budgets unchanged         : overall=%.4f, non-tail=%.4f, |bias|<=%.4f"
        % (
            args.overall_budget,
            args.non_tail_budget,
            args.bias_limit,
        )
    )
    print("Retraining                : NO")
    print("Test split                : NOT ACCESSED")

    print("\nBuilding frozen-base validation cache...")

    cache = joint_module.build_validation_cache(
        module=training_module,
        model=model,
        dataset=validation_dataset,
        loader=validation_loader,
        thresholds_np=thresholds_np,
        device=device,
    )

    base_metrics = joint_module.base_metric_bundle(
        cache
    )

    print(
        "Cached target rows        : %d"
        % len(cache["event_id"])
    )
    print(
        "Base overall MAE          : PGA=%.4f, PGV=%.4f"
        % (
            base_metrics["overall_mae_pga"],
            base_metrics["overall_mae_pgv"],
        )
    )
    print(
        "Base tail MAE             : PGA=%.4f, PGV=%.4f"
        % (
            base_metrics["tail_mae_pga"],
            base_metrics["tail_mae_pgv"],
        )
    )

    scan_rows = []
    selected_head_cache = {}

    for epoch, checkpoint_path in checkpoint_paths:
        checkpoint = training_module.load_checkpoint(
            checkpoint_path,
            device,
        )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )
        model.eval()

        probability, residual = (
            joint_module
            .predict_tail_head_from_cache(
                model=model,
                context_input=cache["context_input"],
                device=device,
                batch_size=args.head_eval_batch_size,
            )
        )

        selected_head_cache[epoch] = (
            probability,
            residual,
        )

        for gamma in powers:
            metrics = joint_module.candidate_metrics(
                cache,
                probability,
                residual,
                gamma,
            )

            status = joint_module.feasibility_status(
                metrics,
                base_metrics,
                args.overall_budget,
                args.non_tail_budget,
                args.bias_limit,
            )

            scan_rows.append(
                {
                    "epoch": int(epoch),
                    "gamma": float(gamma),
                    **metrics,
                    **status,
                }
            )

    scan = pd.DataFrame(scan_rows)

    scan_path = (
        output_directory
        / "extended_epoch_gamma_validation_scan.csv"
    )
    scan.to_csv(
        scan_path,
        index=False,
    )

    feasible = scan.loc[
        scan["feasible"].astype(bool)
    ].copy()

    if len(feasible) == 0:
        raise RuntimeError(
            "No feasible candidates under unchanged budgets."
        )

    feasible = feasible.sort_values(
        [
            "mean_tail_mae",
            "mean_overall_mae",
            "gamma",
            "epoch",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    selected = feasible.iloc[0]

    selected_epoch = int(
        selected["epoch"]
    )
    selected_gamma = float(
        selected["gamma"]
    )

    source_checkpoint = (
        epoch_directory
        / ("epoch_%03d.pt" % selected_epoch)
    )

    locked_checkpoint = (
        output_directory
        / "selected_tail_head_extended.pt"
    )
    shutil.copy2(
        source_checkpoint,
        locked_checkpoint,
    )

    probability, residual = (
        selected_head_cache[
            selected_epoch
        ]
    )

    prediction_path = (
        output_directory
        / "selected_validation_predictions_extended.csv"
    )

    joint_module.save_selected_predictions(
        prediction_path,
        cache,
        probability,
        residual,
        selected_gamma,
    )

    top_path = (
        output_directory
        / "top_feasible_extended_candidates.csv"
    )
    feasible.head(25).to_csv(
        top_path,
        index=False,
    )

    selected_json_path = (
        output_directory
        / "selected_epoch_gamma_extended.json"
    )

    selected_json_path.write_text(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "selected_gamma": selected_gamma,
                "gamma_grid": powers,
                "grid_max_gamma": float(
                    max(powers)
                ),
                "selected_at_upper_gamma_boundary": bool(
                    np.isclose(
                        selected_gamma,
                        max(powers),
                    )
                ),
                "selection_objective": (
                    "minimum mean PGA/PGV high-motion-tail MAE "
                    "among validation-feasible candidates"
                ),
                "tie_breaks": [
                    "lower mean overall MAE",
                    "lower gamma",
                    "earlier epoch",
                ],
                "constraints": {
                    "overall_budget_log10_each_quantity": (
                        args.overall_budget
                    ),
                    "non_tail_budget_log10_each_quantity": (
                        args.non_tail_budget
                    ),
                    "absolute_bias_limit_each_quantity": (
                        args.bias_limit
                    ),
                },
                "validation_repeats": int(
                    args.validation_repeats
                ),
                "metric_aggregation": (
                    "targets -> repeats -> events"
                ),
                "base_metrics": base_metrics,
                "selected_metrics": {
                    key: (
                        bool(value)
                        if isinstance(
                            value,
                            (bool, np.bool_),
                        )
                        else (
                            int(value)
                            if isinstance(
                                value,
                                (int, np.integer),
                            )
                            else (
                                float(value)
                                if isinstance(
                                    value,
                                    (float, np.floating),
                                )
                                else value
                            )
                        )
                    )
                    for key, value
                    in selected.to_dict().items()
                },
                "source_epoch_checkpoint": str(
                    source_checkpoint.resolve()
                ),
                "locked_checkpoint": str(
                    locked_checkpoint.resolve()
                ),
                "test_split_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Extended validation-only selection ===")
    print(
        "Epoch                   : %d"
        % selected_epoch
    )
    print(
        "Gamma                   : %.3f"
        % selected_gamma
    )
    print(
        "At upper grid boundary  : %s"
        % bool(
            np.isclose(
                selected_gamma,
                max(powers),
            )
        )
    )
    print(
        "Overall MAE PGA/PGV     : %.4f / %.4f"
        % (
            selected["overall_mae_pga"],
            selected["overall_mae_pgv"],
        )
    )
    print(
        "Overall delta PGA/PGV   : %+0.4f / %+0.4f"
        % (
            selected["overall_delta_pga"],
            selected["overall_delta_pgv"],
        )
    )
    print(
        "Non-tail delta PGA/PGV  : %+0.4f / %+0.4f"
        % (
            selected["non_tail_delta_pga"],
            selected["non_tail_delta_pgv"],
        )
    )
    print(
        "Tail MAE PGA/PGV        : %.4f / %.4f"
        % (
            selected["tail_mae_pga"],
            selected["tail_mae_pgv"],
        )
    )
    print(
        "Tail delta PGA/PGV      : %+0.4f / %+0.4f"
        % (
            selected["tail_delta_pga"],
            selected["tail_delta_pgv"],
        )
    )
    print(
        "Tail U0.5 PGA/PGV       : %.3f / %.3f"
        % (
            selected["tail_under05_pga"],
            selected["tail_under05_pgv"],
        )
    )
    print(
        "Overall bias PGA/PGV    : %+0.4f / %+0.4f"
        % (
            selected["bias_pga"],
            selected["bias_pgv"],
        )
    )
    print(
        "Feasible candidates     : %d / %d"
        % (
            len(feasible),
            len(scan),
        )
    )
    print(
        "Locked checkpoint       : %s"
        % locked_checkpoint.resolve()
    )
    print(
        "Selection JSON          : %s"
        % selected_json_path.resolve()
    )
    print(
        "Full scan               : %s"
        % scan_path.resolve()
    )
    print("Retraining              : NO")
    print("Test split              : NOT ACCESSED")


if __name__ == "__main__":
    main()
