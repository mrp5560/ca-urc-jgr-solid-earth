#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
34_train_time_clean_tail_risk_gated.py

Chronology-safe tail-risk gated training for Section 4.12.

This script is deliberately compatible with:
    20_evaluate_locked_power_gated_test.py

It exposes the symbols required by that evaluator:
    MeanBase
    TailGateModel
    load_checkpoint

Workflow
--------
1. Load the chronology-safe mean-pooling base checkpoint produced by
   21_train_time_extrapolation_mean_base.py.
2. Freeze the entire base predictor.
3. Train only:
       p_tail(x) in [0, 1]
       Delta_tail(x) >= 0
4. The training-time linear correction is:
       y_linear = y_base + p_tail * Delta_tail
5. Model selection uses ONLY the chronological validation split.
6. After the best checkpoint is selected, a deterministic repeated validation
   file is exported for post-hoc gate-power scanning with
   19_scan_gate_power_posthoc.py.
7. No test or Ridgecrest-OOD event is instantiated by this script.

Important
---------
The tail thresholds must be the training-only thresholds produced by
21_train_time_extrapolation_mean_base.py for the SAME split column.

Expected chronology-safe split:
    train           = 2010-2018
    validation      = 2019 non-Ridgecrest + 2020
    test            = 2021-2024
    ridgecrest_ood  = complete held-out Ridgecrest sequence

Example
-------
python 34_train_time_clean_tail_risk_gated.py ^
  --manifest data\scedc\model_manifests\scenario_t0_5s_k5_time_clean.csv ^
  --split-column split_time_clean ^
  --train-label train ^
  --validation-label validation ^
  --base-checkpoint runs\time_clean_mean_base_t0_5s_k5\best_model.pt ^
  --threshold-json runs\time_clean_mean_base_t0_5s_k5\tail_thresholds_q0.90_t0_5s.json ^
  --epochs 40 ^
  --validation-repeats 3 ^
  --final-validation-repeats 20 ^
  --out-dir runs\time_clean_tail_risk_gated_t0_5s_k5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Load the chronology-safe base module.
# ---------------------------------------------------------------------

def _load_python_module(path: str | Path, name: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required Python module not found: {path.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_THIS_DIR = Path(__file__).resolve().parent
_BASE_SCRIPT = _THIS_DIR / "21_train_time_extrapolation_mean_base.py"
_BASE_MODULE = _load_python_module(
    _BASE_SCRIPT,
    "time_clean_mean_base_module",
)

for _required_symbol in (
    "MeanPoolingSparseFieldModel",
    "SparseFieldDataset",
    "as_bool",
):
    if not hasattr(_BASE_MODULE, _required_symbol):
        raise AttributeError(
            f"{_BASE_SCRIPT.name} is missing {_required_symbol}"
        )


# These symbols are intentionally public because
# 20_evaluate_locked_power_gated_test.py imports them.
MeanBase = _BASE_MODULE.MeanPoolingSparseFieldModel
SparseFieldDataset = _BASE_MODULE.SparseFieldDataset
as_bool = _BASE_MODULE.as_bool


LOG10_FACTOR_2 = math.log10(2.0)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
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


def _infer_base_hidden(
    checkpoint: dict[str, Any],
) -> int:
    args = checkpoint.get("args", {})

    if isinstance(args, dict) and args.get("hidden_dim") is not None:
        return int(args["hidden_dim"])

    state = checkpoint["model_state"]

    for key in (
        "station_encoder.0.weight",
        "waveform_encoder.network.13.weight",
    ):
        if key in state:
            return int(state[key].shape[0])

    raise ValueError(
        "Could not infer hidden_dim from chronology-safe base checkpoint."
    )


# ---------------------------------------------------------------------
# Tail-risk gated model.
# ---------------------------------------------------------------------

class TailGateModel(nn.Module):
    """
    Frozen mean-pooling base + target-specific tail-risk/correction head.

    The architecture is constructed so that the formal locked evaluator can
    infer:
        hidden_dim      from base.station_encoder.0.weight
        tail_hidden_dim from context.0.weight

    Outputs
    -------
    base:
        frozen base prediction [B, Q, 2]
    probability:
        tail probability [B, Q, 2]
    residual:
        non-negative maximum correction [B, Q, 2]
    logits:
        pre-sigmoid tail-risk logits [B, Q, 2]
    linear:
        base + probability * residual
    """

    def __init__(
        self,
        base: nn.Module,
        hidden: int,
        tail_hidden: int,
        prevalence: float,
        maximum_correction: float,
    ):
        super().__init__()

        self.base = base
        self.hidden = int(hidden)
        self.tail_hidden = int(tail_hidden)
        self.maximum_correction = float(maximum_correction)

        # Event latent + target xyz + frozen base PGA/PGV prediction.
        context_input_dim = self.hidden + 3 + 2

        self.context = nn.Sequential(
            nn.Linear(
                context_input_dim,
                self.tail_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.tail_hidden,
                self.tail_hidden,
            ),
            nn.GELU(),
        )

        self.tail_classifier = nn.Linear(
            self.tail_hidden,
            2,
        )

        self.correction_head = nn.Linear(
            self.tail_hidden,
            2,
        )

        prevalence = float(
            np.clip(
                prevalence,
                1e-3,
                1.0 - 1e-3,
            )
        )

        prior_logit = math.log(
            prevalence
            / (
                1.0 - prevalence
            )
        )

        with torch.no_grad():
            self.tail_classifier.bias.fill_(
                prior_logit
            )
            # Small correction at initialization.
            self.correction_head.bias.fill_(
                -2.0
            )

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.base.eval()

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        # Important: keep frozen BatchNorm statistics frozen.
        self.base.eval()

        return self

    def _frozen_base_context(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            (
                batch_size,
                stations,
                channels,
                samples,
            ) = input_waveforms.shape

            waveform_latent = (
                self.base.waveform_encoder(
                    input_waveforms.reshape(
                        batch_size
                        * stations,
                        channels,
                        samples,
                    )
                )
                .reshape(
                    batch_size,
                    stations,
                    -1,
                )
            )

            station_latent = (
                self.base.station_encoder(
                    torch.cat(
                        [
                            waveform_latent,
                            input_features,
                        ],
                        dim=-1,
                    )
                )
            )

            event_latent = (
                station_latent.mean(
                    dim=1
                )
            )

            expanded_event = (
                event_latent[
                    :,
                    None,
                    :,
                ]
                .expand(
                    -1,
                    target_features.shape[1],
                    -1,
                )
            )

            base_prediction = (
                self.base.query_decoder(
                    torch.cat(
                        [
                            expanded_event,
                            target_features,
                        ],
                        dim=-1,
                    )
                )
            )

        return (
            base_prediction,
            event_latent,
        )

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        (
            base_prediction,
            event_latent,
        ) = self._frozen_base_context(
            input_waveforms,
            input_features,
            target_features,
        )

        expanded_event = (
            event_latent[
                :,
                None,
                :,
            ]
            .expand(
                -1,
                target_features.shape[1],
                -1,
            )
        )

        context = self.context(
            torch.cat(
                [
                    expanded_event,
                    target_features,
                    base_prediction,
                ],
                dim=-1,
            )
        )

        logits = self.tail_classifier(
            context
        )

        probability = torch.sigmoid(
            logits
        )

        residual = (
            self.maximum_correction
            * torch.sigmoid(
                self.correction_head(
                    context
                )
            )
        )

        linear = (
            base_prediction
            + probability
            * residual
        )

        return {
            "base": base_prediction,
            "probability": probability,
            "residual": residual,
            "logits": logits,
            "linear": linear,
        }


# ---------------------------------------------------------------------
# Loss and evaluation.
# ---------------------------------------------------------------------

def _tail_labels(
    target: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    return (
        target
        >= thresholds[
            None,
            None,
            :,
        ]
    ).to(
        target.dtype
    )


def train_epoch(
    model: TailGateModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    thresholds: torch.Tensor,
    prevalence: float,
    lambda_classification: float,
    lambda_regression: float,
    tail_regression_weight: float,
    lambda_under: float,
    lambda_leakage: float,
    max_positive_weight: float,
) -> dict[str, float]:
    model.train()

    positive_weight = float(
        np.clip(
            (
                1.0
                - prevalence
            )
            / max(
                prevalence,
                1e-6,
            ),
            1.0,
            max_positive_weight,
        )
    )

    pos_weight = torch.full(
        (
            2,
        ),
        positive_weight,
        dtype=torch.float32,
        device=device,
    )

    totals = {
        "loss": 0.0,
        "classification": 0.0,
        "regression": 0.0,
        "under": 0.0,
        "leakage": 0.0,
    }
    batches = 0

    for batch in loader:
        input_waveforms = batch[
            "input_waveforms"
        ].to(
            device
        )

        input_features = batch[
            "input_features"
        ].to(
            device
        )

        target_features = batch[
            "target_features"
        ].to(
            device
        )

        target = batch[
            "target_log"
        ].to(
            device
        )

        outputs = model(
            input_waveforms,
            input_features,
            target_features,
        )

        labels = _tail_labels(
            target,
            thresholds,
        )

        classification = (
            nn.functional.binary_cross_entropy_with_logits(
                outputs[
                    "logits"
                ],
                labels,
                pos_weight=pos_weight,
            )
        )

        linear_prediction = outputs[
            "linear"
        ]

        elementwise_regression = (
            nn.functional.smooth_l1_loss(
                linear_prediction,
                target,
                reduction="none",
            )
        )

        regression_weight = (
            1.0
            + tail_regression_weight
            * labels
        )

        regression = (
            elementwise_regression
            * regression_weight
        ).mean()

        # Penalize directional underestimation more strongly for tail targets.
        under_gap = (
            target
            - linear_prediction
        ).clamp_min(
            0.0
        )

        if labels.sum() > 0:
            under = (
                under_gap.pow(2)
                * labels
            ).sum() / labels.sum()
        else:
            under = under_gap.new_tensor(
                0.0
            )

        applied_correction = (
            outputs[
                "probability"
            ]
            * outputs[
                "residual"
            ]
        )

        non_tail = (
            1.0
            - labels
        )

        if non_tail.sum() > 0:
            leakage = (
                applied_correction.pow(2)
                * non_tail
            ).sum() / non_tail.sum()
        else:
            leakage = applied_correction.new_tensor(
                0.0
            )

        loss = (
            lambda_classification
            * classification
            + lambda_regression
            * regression
            + lambda_under
            * under
            + lambda_leakage
            * leakage
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter
                in model.parameters()
                if parameter.requires_grad
            ],
            max_norm=5.0,
        )

        optimizer.step()

        totals["loss"] += float(
            loss.detach().cpu()
        )
        totals["classification"] += float(
            classification.detach().cpu()
        )
        totals["regression"] += float(
            regression.detach().cpu()
        )
        totals["under"] += float(
            under.detach().cpu()
        )
        totals["leakage"] += float(
            leakage.detach().cpu()
        )

        batches += 1

    return {
        key: value
        / max(
            batches,
            1,
        )
        for key, value
        in totals.items()
    }


def _event_macro(
    frame: pd.DataFrame,
    value_column: str,
    mask: np.ndarray | None = None,
) -> float:
    selected = frame

    if mask is not None:
        selected = frame.loc[
            np.asarray(
                mask,
                dtype=bool,
            )
        ]

    if len(selected) == 0:
        return float(
            "nan"
        )

    grouped = (
        selected.groupby(
            [
                "event_id",
                "repeat",
            ],
            sort=False,
        )[
            value_column
        ]
        .mean()
    )

    if len(grouped) == 0:
        return float(
            "nan"
        )

    return float(
        grouped.mean()
    )


def evaluate_model(
    model: TailGateModel,
    loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
]:
    model.eval()

    rows: list[
        dict[str, Any]
    ] = []

    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                batch[
                    "input_waveforms"
                ].to(
                    device
                ),
                batch[
                    "input_features"
                ].to(
                    device
                ),
                batch[
                    "target_features"
                ].to(
                    device
                ),
            )

            base = outputs[
                "base"
            ].cpu().numpy()

            probability = outputs[
                "probability"
            ].cpu().numpy()

            residual = outputs[
                "residual"
            ].cpu().numpy()

            final = outputs[
                "linear"
            ].cpu().numpy()

            target = batch[
                "target_log"
            ].numpy()

            magnitude = batch[
                "magnitude"
            ].numpy()

            repeat = batch[
                "repeat"
            ].numpy()

            event_ids = list(
                batch[
                    "event_id"
                ]
            )

            for batch_index in range(
                target.shape[0]
            ):
                for target_index in range(
                    target.shape[1]
                ):
                    true_pga = float(
                        target[
                            batch_index,
                            target_index,
                            0,
                        ]
                    )

                    true_pgv = float(
                        target[
                            batch_index,
                            target_index,
                            1,
                        ]
                    )

                    rows.append(
                        {
                            "event_id": str(
                                event_ids[
                                    batch_index
                                ]
                            ),
                            "repeat": int(
                                repeat[
                                    batch_index
                                ]
                            ),
                            "magnitude": float(
                                magnitude[
                                    batch_index
                                ]
                            ),
                            "true_log10_pga": true_pga,
                            "true_log10_pgv": true_pgv,
                            "base_log10_pga": float(
                                base[
                                    batch_index,
                                    target_index,
                                    0,
                                ]
                            ),
                            "base_log10_pgv": float(
                                base[
                                    batch_index,
                                    target_index,
                                    1,
                                ]
                            ),
                            "final_log10_pga": float(
                                final[
                                    batch_index,
                                    target_index,
                                    0,
                                ]
                            ),
                            "final_log10_pgv": float(
                                final[
                                    batch_index,
                                    target_index,
                                    1,
                                ]
                            ),
                            "tail_probability_pga": float(
                                probability[
                                    batch_index,
                                    target_index,
                                    0,
                                ]
                            ),
                            "tail_probability_pgv": float(
                                probability[
                                    batch_index,
                                    target_index,
                                    1,
                                ]
                            ),
                            "residual_correction_pga": float(
                                residual[
                                    batch_index,
                                    target_index,
                                    0,
                                ]
                            ),
                            "residual_correction_pgv": float(
                                residual[
                                    batch_index,
                                    target_index,
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
                        }
                    )

    frame = pd.DataFrame(
        rows
    )

    if len(frame) == 0:
        raise RuntimeError(
            "No validation prediction rows were generated."
        )

    metrics: dict[
        str,
        float,
    ] = {
        "n_rows": int(
            len(
                frame
            )
        ),
        "n_events": int(
            frame[
                "event_id"
            ].nunique()
        ),
        "n_event_repeats": int(
            frame[
                [
                    "event_id",
                    "repeat",
                ]
            ]
            .drop_duplicates()
            .shape[
                0
            ]
        ),
    }

    for quantity in (
        "pga",
        "pgv",
    ):
        truth = frame[
            f"true_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        base_prediction = frame[
            f"base_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        final_prediction = frame[
            f"final_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        tail = frame[
            f"is_tail_{quantity}"
        ].to_numpy(
            dtype=bool
        )

        non_tail = ~tail

        for model_name, prediction in (
            (
                "base",
                base_prediction,
            ),
            (
                "linear",
                final_prediction,
            ),
        ):
            residual_error = (
                prediction
                - truth
            )

            absolute_error = np.abs(
                residual_error
            )

            abs_column = (
                f"__{model_name}_"
                f"abs_{quantity}"
            )

            frame[
                abs_column
            ] = absolute_error

            metrics[
                f"{model_name}_"
                f"macro_mae_{quantity}"
            ] = _event_macro(
                frame,
                abs_column,
            )

            metrics[
                f"{model_name}_"
                f"macro_tail_mae_{quantity}"
            ] = _event_macro(
                frame,
                abs_column,
                tail,
            )

            metrics[
                f"{model_name}_"
                f"macro_non_tail_mae_{quantity}"
            ] = _event_macro(
                frame,
                abs_column,
                non_tail,
            )

            metrics[
                f"{model_name}_"
                f"bias_{quantity}"
            ] = float(
                residual_error.mean()
            )

            metrics[
                f"{model_name}_"
                f"under05_{quantity}"
            ] = float(
                np.mean(
                    residual_error
                    <= -0.5
                )
            )

            if tail.any():
                metrics[
                    f"{model_name}_"
                    f"tail_under05_{quantity}"
                ] = float(
                    np.mean(
                        residual_error[
                            tail
                        ]
                        <= -0.5
                    )
                )
            else:
                metrics[
                    f"{model_name}_"
                    f"tail_under05_{quantity}"
                ] = float(
                    "nan"
                )

        probability_values = frame[
            f"tail_probability_{quantity}"
        ].to_numpy(
            dtype=float
        )

        labels = tail.astype(
            float
        )

        # Dependency-free validation diagnostics.
        if labels.sum() > 0:
            order = np.argsort(
                -probability_values
            )

            sorted_labels = labels[
                order
            ]

            cumulative_positive = np.cumsum(
                sorted_labels
            )

            rank = np.arange(
                1,
                len(
                    labels
                )
                + 1,
                dtype=float,
            )

            precision = (
                cumulative_positive
                / rank
            )

            recall = (
                cumulative_positive
                / max(
                    labels.sum(),
                    1.0,
                )
            )

            # Trapezoidal approximation to PR area.
            # NumPy 2.x removed np.trapz; np.trapezoid is the supported
            # replacement. Keep a manual fallback for older NumPy releases.
            if hasattr(np, "trapezoid"):
                auprc_value = np.trapezoid(
                    precision,
                    recall,
                )
            else:
                auprc_value = np.sum(
                    0.5
                    * (
                        precision[1:]
                        + precision[:-1]
                    )
                    * (
                        recall[1:]
                        - recall[:-1]
                    )
                )

            metrics[
                f"approx_auprc_{quantity}"
            ] = float(
                auprc_value
            )
        else:
            metrics[
                f"approx_auprc_{quantity}"
            ] = float(
                "nan"
            )

    metrics[
        "selection_score"
    ] = 0.5 * (
        metrics[
            "linear_macro_mae_pga"
        ]
        + metrics[
            "linear_macro_mae_pgv"
        ]
    )

    return (
        metrics,
        frame,
    )


def save_checkpoint(
    path: Path,
    model: TailGateModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    validation_metrics: dict[str, Any],
    threshold_data: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(
                epoch
            ),
            "args": vars(
                args
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "thresholds": threshold_data,
            "test_split_evaluated": False,
        },
        path,
    )


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--split-column",
        required=True,
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
        "--validation-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--final-validation-repeats",
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
        "--early-stopping-patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--minimum-delta",
        type=float,
        default=1e-4,
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
        "--out-dir",
        default=(
            "runs/"
            "time_clean_tail_risk_gated_t0_5s_k5"
        ),
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
            args.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            args.device
        )

    if args.validation_repeats < 1:
        raise ValueError(
            "--validation-repeats must be >= 1."
        )

    if args.final_validation_repeats < 1:
        raise ValueError(
            "--final-validation-repeats must be >= 1."
        )

    set_global_seed(
        args.seed
    )

    output_directory = Path(
        args.out_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_path = Path(
        args.threshold_json
    )

    threshold_data = json.loads(
        threshold_path.read_text(
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
        != str(
            args.split_column
        )
    ):
        raise ValueError(
            "Tail-threshold JSON was produced for a different split column: "
            f"{threshold_data.get('split_column')!r} != {args.split_column!r}"
        )

    if (
        str(
            threshold_data.get(
                "train_label",
                args.train_label,
            )
        )
        != str(
            args.train_label
        )
    ):
        raise ValueError(
            "Tail-threshold JSON was produced for a different train label."
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
    ).to(
        device
    )

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

    base_checkpoint = load_checkpoint(
        args.base_checkpoint,
        device,
    )

    hidden = _infer_base_hidden(
        base_checkpoint
    )

    base = MeanBase(
        hidden
    ).to(
        device
    )

    base.load_state_dict(
        base_checkpoint[
            "model_state"
        ]
    )

    base.eval()

    model = TailGateModel(
        base=base,
        hidden=hidden,
        tail_hidden=args.tail_hidden,
        prevalence=prevalence,
        maximum_correction=(
            args.maximum_correction
        ),
    ).to(
        device
    )

    train_dataset = SparseFieldDataset(
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

    validation_dataset = SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.validation_repeats,
    )

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        )
    )

    loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (
            device.type
            == "cuda"
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

    checkpoint_path = (
        output_directory
        / "best_overall.pt"
    )

    print(
        "=== Chronology-safe tail-risk gated training ==="
    )
    print(
        f"Device                  : {device}"
    )
    print(
        f"Manifest                : {Path(args.manifest).resolve()}"
    )
    print(
        f"Split column            : {args.split_column}"
    )
    print(
        f"Train label             : {args.train_label}"
    )
    print(
        f"Validation label        : {args.validation_label}"
    )
    print(
        f"Train events            : {len(train_dataset.frame)}"
    )
    print(
        f"Validation events       : {len(validation_dataset.frame)}"
    )
    print(
        f"Base checkpoint         : {Path(args.base_checkpoint).resolve()}"
    )
    print(
        "Tail thresholds         : "
        f"PGA={thresholds_np[0]:.4f}, "
        f"PGV={thresholds_np[1]:.4f}"
    )
    print(
        f"Prevalence prior        : {prevalence:.4f}"
    )
    print(
        "Test / Ridgecrest OOD   : NOT ACCESSED"
    )

    history: list[
        dict[str, Any]
    ] = []

    best_score = float(
        "inf"
    )
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        training_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            thresholds=thresholds_tensor,
            prevalence=prevalence,
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

        validation_metrics, _ = (
            evaluate_model(
                model,
                validation_loader,
                thresholds_np,
                device,
            )
        )

        current_score = float(
            validation_metrics[
                "selection_score"
            ]
        )

        scheduler.step(
            current_score
        )

        history_row = {
            "epoch": int(
                epoch
            ),
            "learning_rate": float(
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            ),
            **{
                f"train_{key}": value
                for key, value
                in training_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value
                in validation_metrics.items()
            },
        }

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            output_directory
            / "history.csv",
            index=False,
        )

        if (
            current_score
            < best_score
            - args.minimum_delta
        ):
            best_score = (
                current_score
            )
            best_epoch = int(
                epoch
            )
            stale_epochs = 0

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                args,
                validation_metrics,
                threshold_data,
            )
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d} | "
            f"loss={training_metrics['loss']:.4f} | "
            f"val overall="
            f"{validation_metrics['linear_macro_mae_pga']:.4f}/"
            f"{validation_metrics['linear_macro_mae_pgv']:.4f} | "
            f"val tail="
            f"{validation_metrics['linear_macro_tail_mae_pga']:.4f}/"
            f"{validation_metrics['linear_macro_tail_mae_pgv']:.4f} | "
            f"base="
            f"{validation_metrics['base_macro_mae_pga']:.4f}/"
            f"{validation_metrics['base_macro_mae_pgv']:.4f}"
        )

        if (
            stale_epochs
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}."
            )
            break

    best_checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state"
        ]
    )

    model.eval()

    # Re-evaluate the selected checkpoint with 20 deterministic combinations
    # per validation event. This is the ONLY file used for gamma selection.
    final_validation_dataset = SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.final_validation_repeats,
    )

    final_validation_loader = DataLoader(
        final_validation_dataset,
        shuffle=False,
        **loader_arguments,
    )

    final_metrics, final_predictions = (
        evaluate_model(
            model,
            final_validation_loader,
            thresholds_np,
            device,
        )
    )

    final_predictions.to_csv(
        output_directory
        / "validation_predictions_overall.csv",
        index=False,
    )

    (
        output_directory
        / "validation_metrics_overall.json"
    ).write_text(
        json.dumps(
            {
                "best_epoch": int(
                    best_checkpoint[
                        "epoch"
                    ]
                ),
                "training_seconds": float(
                    time.time()
                    - start_time
                ),
                "validation_repeats": int(
                    args.final_validation_repeats
                ),
                "test_split_evaluated": False,
                **final_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_directory
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(
                    args
                ),
                "resolved_device": str(
                    device
                ),
                "hidden_dim": int(
                    hidden
                ),
                "tail_hidden": int(
                    args.tail_hidden
                ),
                "prevalence": float(
                    prevalence
                ),
                "tail_thresholds": (
                    threshold_data
                ),
                "test_split_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Selected chronology-safe tail-risk checkpoint ==="
    )
    print(
        f"Best epoch              : {best_checkpoint['epoch']}"
    )
    print(
        "Validation overall MAE : "
        f"PGA={final_metrics['linear_macro_mae_pga']:.4f}, "
        f"PGV={final_metrics['linear_macro_mae_pgv']:.4f}"
    )
    print(
        "Validation tail MAE    : "
        f"PGA={final_metrics['linear_macro_tail_mae_pga']:.4f}, "
        f"PGV={final_metrics['linear_macro_tail_mae_pgv']:.4f}"
    )
    print(
        "Frozen-base overall    : "
        f"PGA={final_metrics['base_macro_mae_pga']:.4f}, "
        f"PGV={final_metrics['base_macro_mae_pgv']:.4f}"
    )
    print(
        f"Checkpoint              : {checkpoint_path.resolve()}"
    )
    print(
        "Gate-scan predictions  : "
        f"{(output_directory / 'validation_predictions_overall.csv').resolve()}"
    )
    print(
        "Test / Ridgecrest OOD  : NOT ACCESSED"
    )


if __name__ == "__main__":
    main()
