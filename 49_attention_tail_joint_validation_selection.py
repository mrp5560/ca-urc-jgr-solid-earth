#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
49_attention_tail_joint_validation_selection.py

Validation-only JOINT selection of:
    tail-head checkpoint epoch + power-gate exponent gamma

for the final attention-pooling Causal-SeisField base.

Why this is needed
------------------
Selecting the tail head only by the intermediate linear correction
    y_linear = y_base + p * Delta
can be too conservative. The final model is
    y_gamma = y_base + p^gamma * Delta,
so a stronger tail head at a later epoch may become preferable once
low-risk corrections are suppressed by gamma > 1.

This script therefore:
1. Re-trains the tail head from the same frozen attention base.
2. Saves EVERY epoch checkpoint.
3. Builds a 20-repeat validation cache from the frozen base ONCE.
4. Evaluates every saved epoch over a pre-specified gamma grid.
5. Enforces pre-specified overall/non-tail/bias budgets.
6. Selects the feasible epoch-gamma pair with the lowest mean tail MAE.
7. Exports the selected checkpoint, gamma, predictions, and scan table.
8. NEVER instantiates the test split.

Default constraints are intentionally inherited from the previous gate scan:
    overall MAE increase <= 0.015 log10 for BOTH PGA and PGV
    non-tail MAE increase <= 0.005 log10 for BOTH PGA and PGV
    |overall bias| <= 0.05 for BOTH PGA and PGV

Main metric aggregation:
    targets -> repeats -> events.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "attention_tail_module",
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
            raise ValueError(
                "All gamma values must be >= 1."
            )
        values.append(value)

    values = sorted(set(values))

    if not values:
        raise ValueError("No gamma values supplied.")

    return values


def canonical_macro(
    event_ids: np.ndarray,
    repeats: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is not None:
        keep = np.asarray(
            mask,
            dtype=bool,
        )
        event_ids = event_ids[keep]
        repeats = repeats[keep]
        values = values[keep]

    if len(values) == 0:
        return float("nan")

    frame = pd.DataFrame(
        {
            "event_id": event_ids.astype(str),
            "repeat": repeats.astype(int),
            "value": values.astype(float),
        }
    )

    event_repeat = (
        frame.groupby(
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


def compute_metrics(
    event_ids: np.ndarray,
    repeats: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    tail_mask: np.ndarray,
) -> dict[str, float]:
    error = prediction - truth
    absolute = np.abs(error)
    non_tail = ~tail_mask

    return {
        "overall_mae": canonical_macro(
            event_ids,
            repeats,
            absolute,
        ),
        "tail_mae": canonical_macro(
            event_ids,
            repeats,
            absolute,
            tail_mask,
        ),
        "non_tail_mae": canonical_macro(
            event_ids,
            repeats,
            absolute,
            non_tail,
        ),
        "bias": canonical_macro(
            event_ids,
            repeats,
            error,
        ),
        "tail_under05": canonical_macro(
            event_ids,
            repeats,
            (
                error <= -0.5
            ).astype(float),
            tail_mask,
        ),
        "factor2": canonical_macro(
            event_ids,
            repeats,
            (
                absolute <= math.log10(2.0)
            ).astype(float),
        ),
    }


def build_validation_cache(
    module,
    model,
    dataset,
    loader,
    thresholds_np: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run the frozen attention base ONCE and cache:
      context input [event latent, target xyz, base prediction]
      truth, event/repeat ids, tail labels.
    """
    model.eval()

    context_parts = []
    truth_parts = []
    base_parts = []
    event_ids = []
    repeats = []
    target_slots = []

    with torch.inference_mode():
        for batch in loader:
            input_waveforms = batch[
                "input_waveforms"
            ].to(device)
            input_features = batch[
                "input_features"
            ].to(device)
            target_features = batch[
                "target_features"
            ].to(device)

            truth = batch[
                "target_log"
            ].numpy()

            (
                event_latent,
                _,
                _,
            ) = model.base.encode_event(
                input_waveforms,
                input_features,
            )

            expanded_event = event_latent[
                :, None, :
            ].expand(
                -1,
                target_features.shape[1],
                -1,
            )

            base_prediction = (
                model.base.query_decoder(
                    torch.cat(
                        [
                            expanded_event,
                            target_features,
                        ],
                        dim=-1,
                    )
                )
            )

            context_input = torch.cat(
                [
                    expanded_event,
                    target_features,
                    base_prediction,
                ],
                dim=-1,
            )

            context_parts.append(
                context_input.cpu().numpy()
            )
            base_parts.append(
                base_prediction.cpu().numpy()
            )
            truth_parts.append(
                truth
            )

            batch_event_ids = list(
                batch["event_id"]
            )
            batch_repeats = batch[
                "repeat"
            ].numpy()

            for b in range(
                truth.shape[0]
            ):
                for q in range(
                    truth.shape[1]
                ):
                    event_ids.append(
                        str(
                            batch_event_ids[b]
                        )
                    )
                    repeats.append(
                        int(
                            batch_repeats[b]
                        )
                    )
                    target_slots.append(
                        int(q)
                    )

    context_input = np.concatenate(
        context_parts,
        axis=0,
    ).reshape(
        -1,
        context_parts[0].shape[-1],
    )

    truth = np.concatenate(
        truth_parts,
        axis=0,
    ).reshape(
        -1,
        2,
    )

    base_prediction = np.concatenate(
        base_parts,
        axis=0,
    ).reshape(
        -1,
        2,
    )

    event_ids = np.asarray(
        event_ids,
        dtype=object,
    )
    repeats = np.asarray(
        repeats,
        dtype=np.int64,
    )
    target_slots = np.asarray(
        target_slots,
        dtype=np.int64,
    )

    tail = (
        truth
        >= thresholds_np[
            None,
            :,
        ]
    )

    if not (
        len(context_input)
        == len(truth)
        == len(base_prediction)
        == len(event_ids)
        == len(repeats)
    ):
        raise RuntimeError(
            "Validation cache length mismatch."
        )

    return {
        "context_input": (
            context_input.astype(
                np.float32
            )
        ),
        "truth": truth.astype(
            np.float64
        ),
        "base": base_prediction.astype(
            np.float64
        ),
        "tail": tail.astype(bool),
        "event_id": event_ids,
        "repeat": repeats,
        "target_slot": target_slots,
    }


def predict_tail_head_from_cache(
    model,
    context_input: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate only the small tail head on cached frozen-base context.
    """
    probability_parts = []
    residual_parts = []

    model.eval()

    with torch.inference_mode():
        for start in range(
            0,
            len(context_input),
            batch_size,
        ):
            stop = min(
                start + batch_size,
                len(context_input),
            )

            x = torch.from_numpy(
                context_input[
                    start:stop
                ]
            ).to(device)

            context = model.context(
                x
            )

            probability = torch.sigmoid(
                model.tail_classifier(
                    context
                )
            )

            residual = (
                model.maximum_correction
                * torch.sigmoid(
                    model.correction_head(
                        context
                    )
                )
            )

            probability_parts.append(
                probability.cpu().numpy()
            )
            residual_parts.append(
                residual.cpu().numpy()
            )

    return (
        np.concatenate(
            probability_parts,
            axis=0,
        ).astype(
            np.float64
        ),
        np.concatenate(
            residual_parts,
            axis=0,
        ).astype(
            np.float64
        ),
    )


def base_metric_bundle(
    cache: dict[str, Any],
) -> dict[str, float]:
    result = {}

    for j, quantity in enumerate(
        ("pga", "pgv")
    ):
        metrics = compute_metrics(
            cache["event_id"],
            cache["repeat"],
            cache["truth"][:, j],
            cache["base"][:, j],
            cache["tail"][:, j],
        )

        for key, value in (
            metrics.items()
        ):
            result[
                "%s_%s"
                % (
                    key,
                    quantity,
                )
            ] = value

    result[
        "mean_tail_mae"
    ] = 0.5 * (
        result[
            "tail_mae_pga"
        ]
        + result[
            "tail_mae_pgv"
        ]
    )

    result[
        "mean_overall_mae"
    ] = 0.5 * (
        result[
            "overall_mae_pga"
        ]
        + result[
            "overall_mae_pgv"
        ]
    )

    return result


def candidate_metrics(
    cache: dict[str, Any],
    probability: np.ndarray,
    residual: np.ndarray,
    gamma: float,
) -> dict[str, float]:
    prediction = (
        cache["base"]
        + np.power(
            np.clip(
                probability,
                0.0,
                1.0,
            ),
            float(gamma),
        )
        * residual
    )

    result = {}

    for j, quantity in enumerate(
        ("pga", "pgv")
    ):
        metrics = compute_metrics(
            cache["event_id"],
            cache["repeat"],
            cache["truth"][:, j],
            prediction[:, j],
            cache["tail"][:, j],
        )

        for key, value in (
            metrics.items()
        ):
            result[
                "%s_%s"
                % (
                    key,
                    quantity,
                )
            ] = value

    result[
        "mean_tail_mae"
    ] = 0.5 * (
        result[
            "tail_mae_pga"
        ]
        + result[
            "tail_mae_pgv"
        ]
    )

    result[
        "mean_overall_mae"
    ] = 0.5 * (
        result[
            "overall_mae_pga"
        ]
        + result[
            "overall_mae_pgv"
        ]
    )

    return result


def feasibility_status(
    candidate: dict[str, float],
    base: dict[str, float],
    overall_budget: float,
    non_tail_budget: float,
    bias_limit: float,
) -> dict[str, Any]:
    values = {}

    for quantity in (
        "pga",
        "pgv",
    ):
        values[
            "overall_delta_%s"
            % quantity
        ] = (
            candidate[
                "overall_mae_%s"
                % quantity
            ]
            - base[
                "overall_mae_%s"
                % quantity
            ]
        )

        values[
            "non_tail_delta_%s"
            % quantity
        ] = (
            candidate[
                "non_tail_mae_%s"
                % quantity
            ]
            - base[
                "non_tail_mae_%s"
                % quantity
            ]
        )

        values[
            "tail_delta_%s"
            % quantity
        ] = (
            candidate[
                "tail_mae_%s"
                % quantity
            ]
            - base[
                "tail_mae_%s"
                % quantity
            ]
        )

    constraints = [
        values[
            "overall_delta_pga"
        ]
        <= overall_budget,
        values[
            "overall_delta_pgv"
        ]
        <= overall_budget,
        values[
            "non_tail_delta_pga"
        ]
        <= non_tail_budget,
        values[
            "non_tail_delta_pgv"
        ]
        <= non_tail_budget,
        abs(
            candidate[
                "bias_pga"
            ]
        )
        <= bias_limit,
        abs(
            candidate[
                "bias_pgv"
            ]
        )
        <= bias_limit,
    ]

    values[
        "feasible"
    ] = bool(
        all(
            constraints
        )
    )

    return values


def save_selected_predictions(
    path: Path,
    cache: dict[str, Any],
    probability: np.ndarray,
    residual: np.ndarray,
    gamma: float,
) -> None:
    power_prediction = (
        cache["base"]
        + np.power(
            np.clip(
                probability,
                0.0,
                1.0,
            ),
            float(gamma),
        )
        * residual
    )

    linear_prediction = (
        cache["base"]
        + probability
        * residual
    )

    frame = pd.DataFrame(
        {
            "event_id": (
                cache["event_id"]
            ),
            "repeat": (
                cache["repeat"]
            ),
            "target_slot": (
                cache[
                    "target_slot"
                ]
            ),
            "true_log10_pga": (
                cache["truth"][
                    :, 0
                ]
            ),
            "true_log10_pgv": (
                cache["truth"][
                    :, 1
                ]
            ),
            "base_log10_pga": (
                cache["base"][
                    :, 0
                ]
            ),
            "base_log10_pgv": (
                cache["base"][
                    :, 1
                ]
            ),
            "tail_probability_pga": (
                probability[
                    :, 0
                ]
            ),
            "tail_probability_pgv": (
                probability[
                    :, 1
                ]
            ),
            "residual_correction_pga": (
                residual[
                    :, 0
                ]
            ),
            "residual_correction_pgv": (
                residual[
                    :, 1
                ]
            ),
            "linear_log10_pga": (
                linear_prediction[
                    :, 0
                ]
            ),
            "linear_log10_pgv": (
                linear_prediction[
                    :, 1
                ]
            ),
            "power_gate_log10_pga": (
                power_prediction[
                    :, 0
                ]
            ),
            "power_gate_log10_pgv": (
                power_prediction[
                    :, 1
                ]
            ),
            "is_tail_pga": (
                cache["tail"][
                    :, 0
                ]
            ),
            "is_tail_pgv": (
                cache["tail"][
                    :, 1
                ]
            ),
            "selected_gamma": (
                np.full(
                    len(
                        cache[
                            "event_id"
                        ]
                    ),
                    float(
                        gamma
                    ),
                )
            ),
        }
    )

    frame.to_csv(
        path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-module",
        default=(
            "48_train_attention_tail_risk_gated.py"
        ),
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
        "--train-label",
        default="train",
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
        "--training-validation-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--selection-validation-repeats",
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
        "--epochs",
        type=int,
        default=40,
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
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lambda-classification",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--lambda-regression",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tail-regression-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--lambda-under",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--lambda-leakage",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--max-positive-weight",
        type=float,
        default=9.0,
    )

    parser.add_argument(
        "--powers",
        default=(
            "1,1.5,2,2.5,3,3.5,4,5,6"
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
        "--seed",
        type=int,
        default=20260713,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        "--out-dir",
        default=(
            "runs/"
            "attention_tail_joint_validation_selection"
        ),
    )

    args = parser.parse_args()

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

    powers = parse_float_list(
        args.powers
    )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        if (
            args.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable."
            )
        device = torch.device(
            args.device
        )

    training_module_path = Path(
        args.training_module
    )

    if not training_module_path.exists():
        raise FileNotFoundError(
            training_module_path
        )

    module = load_module(
        training_module_path
    )

    module.set_global_seed(
        args.seed
    )

    threshold_path = Path(
        args.threshold_json
    )

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds_np = np.asarray(
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

    thresholds_tensor = torch.from_numpy(
        thresholds_np
    ).to(device)

    prevalence = float(
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
    )

    base_checkpoint = (
        module.load_checkpoint(
            args.base_checkpoint,
            device,
        )
    )

    hidden = module._infer_base_hidden(
        base_checkpoint
    )

    base = module.MeanBase(
        hidden
    ).to(device)

    base.load_state_dict(
        base_checkpoint[
            "model_state"
        ],
        strict=True,
    )

    base.eval()

    model = module.TailGateModel(
        base=base,
        hidden=hidden,
        tail_hidden=(
            args.tail_hidden
        ),
        prevalence=prevalence,
        maximum_correction=(
            args.maximum_correction
        ),
    ).to(device)

    train_dataset = (
        module.SparseFieldDataset(
            manifest=args.manifest,
            split_column=(
                args.split_column
            ),
            split_name=(
                args.train_label
            ),
            t0_sec=args.t0_sec,
            input_stations=(
                args.input_stations
            ),
            target_stations=(
                args.target_stations
            ),
            input_pre_sec=(
                args.input_pre_sec
            ),
            seed=args.seed,
            training=True,
        )
    )

    validation_dataset = (
        module.SparseFieldDataset(
            manifest=args.manifest,
            split_column=(
                args.split_column
            ),
            split_name=(
                args.validation_label
            ),
            t0_sec=args.t0_sec,
            input_stations=(
                args.input_stations
            ),
            target_stations=(
                args.target_stations
            ),
            input_pre_sec=(
                args.input_pre_sec
            ),
            seed=args.seed,
            training=False,
            repeats=(
                args.training_validation_repeats
            ),
        )
    )

    loader_arguments = {
        "batch_size": (
            args.batch_size
        ),
        "num_workers": (
            args.num_workers
        ),
        "pin_memory": (
            device.type == "cuda"
        ),
        "persistent_workers": False,
    }

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_arguments,
    )

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        )
    )

    output_directory = Path(
        args.out_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_directory = (
        output_directory
        / "epoch_checkpoints"
    )

    epoch_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Attention tail-head joint validation selection ==="
    )
    print(
        "Device                         : %s"
        % device
    )
    print(
        "Base checkpoint epoch          : %s"
        % base_checkpoint.get(
            "epoch",
            "unknown",
        )
    )
    print(
        "Train/Validation events        : %d/%d"
        % (
            len(
                train_dataset.frame
            ),
            len(
                validation_dataset.frame
            ),
        )
    )
    print(
        "Training validation repeats    : %d"
        % args.training_validation_repeats
    )
    print(
        "Selection validation repeats   : %d"
        % args.selection_validation_repeats
    )
    print(
        "Gamma grid                     : %s"
        % powers
    )
    print(
        "Overall/non-tail budgets       : %.4f / %.4f"
        % (
            args.overall_budget,
            args.non_tail_budget,
        )
    )
    print(
        "Absolute bias limit            : %.4f"
        % args.bias_limit
    )
    print(
        "Test split                     : NOT ACCESSED"
    )

    history_rows = []
    start_time = time.time()

    # ------------------------------------------------------------
    # Train once and save every epoch.
    # ------------------------------------------------------------
    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        training_metrics = (
            module.train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                thresholds=(
                    thresholds_tensor
                ),
                prevalence=(
                    prevalence
                ),
                lambda_classification=(
                    args.lambda_classification
                ),
                lambda_regression=(
                    args.lambda_regression
                ),
                tail_regression_weight=(
                    args.tail_regression_weight
                ),
                lambda_under=(
                    args.lambda_under
                ),
                lambda_leakage=(
                    args.lambda_leakage
                ),
                max_positive_weight=(
                    args.max_positive_weight
                ),
            )
        )

        (
            validation_metrics,
            _,
        ) = module.evaluate_model(
            model,
            validation_loader,
            thresholds_np,
            device,
        )

        scheduler.step(
            validation_metrics[
                "selection_score"
            ]
        )

        checkpoint_path = (
            epoch_directory
            / (
                "epoch_%03d.pt"
                % epoch
            )
        )

        module.save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            args,
            validation_metrics,
            threshold_data,
        )

        history_rows.append(
            {
                "epoch": int(
                    epoch
                ),
                "learning_rate": float(
                    optimizer
                    .param_groups[0][
                        "lr"
                    ]
                ),
                **{
                    "train_%s" % key: value
                    for key, value
                    in training_metrics.items()
                },
                **{
                    "validation_%s" % key: value
                    for key, value
                    in validation_metrics.items()
                },
            }
        )

        pd.DataFrame(
            history_rows
        ).to_csv(
            output_directory
            / "training_history.csv",
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
                validation_metrics[
                    "linear_macro_mae_pga"
                ],
                validation_metrics[
                    "linear_macro_mae_pgv"
                ],
                validation_metrics[
                    "linear_macro_tail_mae_pga"
                ],
                validation_metrics[
                    "linear_macro_tail_mae_pgv"
                ],
                validation_metrics[
                    "linear_tail_under05_pga"
                ],
                validation_metrics[
                    "linear_tail_under05_pgv"
                ],
                validation_metrics[
                    "approx_auprc_pga"
                ],
                validation_metrics[
                    "approx_auprc_pgv"
                ],
            )
        )

    # ------------------------------------------------------------
    # Build the FINAL 20-repeat validation cache ONCE.
    # ------------------------------------------------------------
    selection_dataset = (
        module.SparseFieldDataset(
            manifest=args.manifest,
            split_column=(
                args.split_column
            ),
            split_name=(
                args.validation_label
            ),
            t0_sec=args.t0_sec,
            input_stations=(
                args.input_stations
            ),
            target_stations=(
                args.target_stations
            ),
            input_pre_sec=(
                args.input_pre_sec
            ),
            seed=args.seed,
            training=False,
            repeats=(
                args.selection_validation_repeats
            ),
        )
    )

    selection_loader = DataLoader(
        selection_dataset,
        shuffle=False,
        **loader_arguments,
    )

    print(
        "\nBuilding frozen-base 20-repeat validation cache..."
    )

    cache = build_validation_cache(
        module=module,
        model=model,
        dataset=(
            selection_dataset
        ),
        loader=(
            selection_loader
        ),
        thresholds_np=(
            thresholds_np
        ),
        device=device,
    )

    print(
        "Cached validation target rows    : %d"
        % len(
            cache[
                "event_id"
            ]
        )
    )

    base_metrics = (
        base_metric_bundle(
            cache
        )
    )

    (
        output_directory
        / "base_validation_metrics.json"
    ).write_text(
        json.dumps(
            base_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Base overall MAE                 : "
        "PGA=%.4f, PGV=%.4f"
        % (
            base_metrics[
                "overall_mae_pga"
            ],
            base_metrics[
                "overall_mae_pgv"
            ],
        )
    )
    print(
        "Base tail MAE                    : "
        "PGA=%.4f, PGV=%.4f"
        % (
            base_metrics[
                "tail_mae_pga"
            ],
            base_metrics[
                "tail_mae_pgv"
            ],
        )
    )

    # ------------------------------------------------------------
    # Evaluate EVERY epoch x gamma on validation only.
    # ------------------------------------------------------------
    scan_rows = []
    epoch_head_cache = {}

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        checkpoint_path = (
            epoch_directory
            / (
                "epoch_%03d.pt"
                % epoch
            )
        )

        checkpoint = (
            module.load_checkpoint(
                checkpoint_path,
                device,
            )
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ]
        )
        model.eval()

        (
            probability,
            residual,
        ) = predict_tail_head_from_cache(
            model=model,
            context_input=(
                cache[
                    "context_input"
                ]
            ),
            device=device,
            batch_size=(
                args.head_eval_batch_size
            ),
        )

        epoch_head_cache[
            epoch
        ] = (
            probability,
            residual,
        )

        for gamma in powers:
            metrics = candidate_metrics(
                cache,
                probability,
                residual,
                gamma,
            )

            status = feasibility_status(
                metrics,
                base_metrics,
                args.overall_budget,
                args.non_tail_budget,
                args.bias_limit,
            )

            scan_rows.append(
                {
                    "epoch": int(
                        epoch
                    ),
                    "gamma": float(
                        gamma
                    ),
                    **metrics,
                    **status,
                }
            )

    scan = pd.DataFrame(
        scan_rows
    )

    scan_path = (
        output_directory
        / "epoch_gamma_validation_scan.csv"
    )

    scan.to_csv(
        scan_path,
        index=False,
    )

    feasible = scan.loc[
        scan["feasible"].astype(bool)
    ].copy()

    if len(feasible) == 0:
        print(
            "\nNO feasible epoch-gamma pair satisfied all constraints."
        )
        print(
            "Do not evaluate test. Inspect: %s"
            % scan_path.resolve()
        )
        raise RuntimeError(
            "No feasible validation candidate."
        )

    # Primary objective: lowest mean tail MAE.
    # Tie-break 1: lower mean overall MAE.
    # Tie-break 2: lower gamma.
    # Tie-break 3: earlier epoch.
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

    selected = feasible.iloc[
        0
    ]

    selected_epoch = int(
        selected["epoch"]
    )

    selected_gamma = float(
        selected["gamma"]
    )

    selected_source_checkpoint = (
        epoch_directory
        / (
            "epoch_%03d.pt"
            % selected_epoch
        )
    )

    selected_checkpoint = (
        output_directory
        / "selected_tail_head.pt"
    )

    shutil.copy2(
        selected_source_checkpoint,
        selected_checkpoint,
    )

    (
        probability,
        residual,
    ) = epoch_head_cache[
        selected_epoch
    ]

    prediction_path = (
        output_directory
        / "selected_validation_predictions.csv"
    )

    save_selected_predictions(
        prediction_path,
        cache,
        probability,
        residual,
        selected_gamma,
    )

    selection_payload = {
        "selected_epoch": (
            selected_epoch
        ),
        "selected_gamma": (
            selected_gamma
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
        "gamma_grid": powers,
        "validation_repeats": (
            args.selection_validation_repeats
        ),
        "metric_aggregation": (
            "targets -> repeats -> events"
        ),
        "base_metrics": (
            base_metrics
        ),
        "selected_metrics": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (bool, np.bool_)
                )
                else (
                    int(value)
                    if isinstance(
                        value,
                        (int, np.integer)
                    )
                    else (
                        float(value)
                        if isinstance(
                            value,
                            (
                                float,
                                np.floating,
                            )
                        )
                        else value
                    )
                )
            )
            for key, value
            in selected.to_dict().items()
        },
        "base_checkpoint": str(
            Path(
                args.base_checkpoint
            ).resolve()
        ),
        "selected_tail_checkpoint": str(
            selected_checkpoint.resolve()
        ),
        "test_split_evaluated": False,
    }

    selection_json_path = (
        output_directory
        / "selected_epoch_gamma.json"
    )

    selection_json_path.write_text(
        json.dumps(
            selection_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Compact top candidates.
    top = feasible.head(
        15
    ).copy()

    top_path = (
        output_directory
        / "top_feasible_validation_candidates.csv"
    )

    top.to_csv(
        top_path,
        index=False,
    )

    print(
        "\n=== Selected validation-only epoch + gamma ==="
    )
    print(
        "Epoch                   : %d"
        % selected_epoch
    )
    print(
        "Gamma                   : %.3f"
        % selected_gamma
    )
    print(
        "Overall MAE PGA/PGV     : %.4f / %.4f"
        % (
            selected[
                "overall_mae_pga"
            ],
            selected[
                "overall_mae_pgv"
            ],
        )
    )
    print(
        "Overall delta PGA/PGV   : %+0.4f / %+0.4f"
        % (
            selected[
                "overall_delta_pga"
            ],
            selected[
                "overall_delta_pgv"
            ],
        )
    )
    print(
        "Non-tail delta PGA/PGV  : %+0.4f / %+0.4f"
        % (
            selected[
                "non_tail_delta_pga"
            ],
            selected[
                "non_tail_delta_pgv"
            ],
        )
    )
    print(
        "Tail MAE PGA/PGV        : %.4f / %.4f"
        % (
            selected[
                "tail_mae_pga"
            ],
            selected[
                "tail_mae_pgv"
            ],
        )
    )
    print(
        "Tail delta PGA/PGV      : %+0.4f / %+0.4f"
        % (
            selected[
                "tail_delta_pga"
            ],
            selected[
                "tail_delta_pgv"
            ],
        )
    )
    print(
        "Tail U0.5 PGA/PGV       : %.3f / %.3f"
        % (
            selected[
                "tail_under05_pga"
            ],
            selected[
                "tail_under05_pgv"
            ],
        )
    )
    print(
        "Overall bias PGA/PGV    : %+0.4f / %+0.4f"
        % (
            selected[
                "bias_pga"
            ],
            selected[
                "bias_pgv"
            ],
        )
    )
    print(
        "Feasible candidates     : %d / %d"
        % (
            len(
                feasible
            ),
            len(
                scan
            ),
        )
    )
    print(
        "Selected checkpoint     : %s"
        % selected_checkpoint.resolve()
    )
    print(
        "Selection JSON          : %s"
        % selection_json_path.resolve()
    )
    print(
        "Validation predictions  : %s"
        % prediction_path.resolve()
    )
    print(
        "Full scan               : %s"
        % scan_path.resolve()
    )
    print(
        "Test split              : NOT ACCESSED"
    )

    (
        output_directory
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": (
                    str(device)
                ),
                "gamma_grid": powers,
                "metric_aggregation": (
                    "targets -> repeats -> events"
                ),
                "training_seconds": float(
                    time.time()
                    - start_time
                ),
                "test_split_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
