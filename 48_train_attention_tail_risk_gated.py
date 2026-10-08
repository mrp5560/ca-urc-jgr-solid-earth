#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
48_train_attention_tail_risk_gated.py

Final grouped-split tail-risk training for Causal-SeisField after the
validation-only architecture audit selected the FULL attention-pooling base.

Locked base
-----------
Waveform + geometry + P-offset + lightweight station-attention pooling.

Workflow
--------
1. Load the already selected attention-pooling base checkpoint.
2. Freeze the entire base predictor, including BatchNorm statistics.
3. Train only a target-specific tail-risk head:
       p_tail(x) in [0,1]
       Delta_tail(x) >= 0
4. Training-time diagnostic prediction:
       y_linear = y_base + p_tail * Delta_tail
5. Select the tail-head checkpoint ONLY on grouped validation data.
6. Re-evaluate the selected checkpoint with 20 deterministic validation
   station draws and export:
       validation_predictions_overall.csv
   for subsequent power-gate scanning.
7. The test split is NEVER instantiated by this script.

The module deliberately exposes:
    MeanBase
    TailGateModel
    load_checkpoint
so that 20_evaluate_locked_power_gated_test.py can load the final model.
Despite the historical symbol name "MeanBase", MeanBase below is the
FINAL ATTENTION-POOLING base for evaluator compatibility.
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
# Load script 40 for the exact dataset construction and waveform encoder.
# ---------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BASE_SCRIPT = _THIS_DIR / "40_phase1_final_mean_pooling_ablation_suite.py"


def _load_python_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            "Required Python module not found: %s" % path.resolve()
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            "Cannot import Python module: %s" % path
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_python_module(
    _BASE_SCRIPT,
    "phase1_final_attention_base_module",
)

if not hasattr(_BASE_MODULE, "SparseFieldDataset"):
    raise AttributeError(
        "%s is missing SparseFieldDataset" % _BASE_SCRIPT.name
    )

if not hasattr(_BASE_MODULE, "WaveformEncoder"):
    raise AttributeError(
        "%s is missing WaveformEncoder" % _BASE_SCRIPT.name
    )


SparseFieldDataset = _BASE_MODULE.SparseFieldDataset
WaveformEncoder = _BASE_MODULE.WaveformEncoder

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

    if (
        isinstance(args, dict)
        and args.get("hidden_dim") is not None
    ):
        return int(args["hidden_dim"])

    state = checkpoint["model_state"]

    for key in (
        "station_encoder.0.weight",
        "waveform_encoder.network.13.weight",
    ):
        if key in state:
            return int(state[key].shape[0])

    raise ValueError(
        "Could not infer hidden_dim from attention base checkpoint."
    )


# ---------------------------------------------------------------------
# Final attention-pooling Base.
# ---------------------------------------------------------------------

class AttentionPoolingBase(nn.Module):
    """
    Exact architecture of script-40 variant='attention_pooling'.

    Input station representation:
        waveform embedding + [x, y, third station coordinate, P-offset/T0]

    Aggregation:
        learned scalar station attention.

    Target decoder:
        attention-pooled event latent + target spatial coordinates.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.waveform_encoder = WaveformEncoder(
            self.hidden_dim
        )

        self.station_encoder = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 4,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
        )

        self.attention_score = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim // 2,
            ),
            nn.Tanh(),
            nn.Linear(
                self.hidden_dim // 2,
                1,
            ),
        )

        self.query_decoder = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 3,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                2,
            ),
        )

    def encode_event(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            batch_size,
            station_count,
            channel_count,
            sample_count,
        ) = input_waveforms.shape

        waveform_latent = (
            self.waveform_encoder(
                input_waveforms.reshape(
                    batch_size * station_count,
                    channel_count,
                    sample_count,
                )
            )
            .reshape(
                batch_size,
                station_count,
                -1,
            )
        )

        station_latent = self.station_encoder(
            torch.cat(
                [
                    waveform_latent,
                    input_features,
                ],
                dim=-1,
            )
        )

        attention_weights = torch.softmax(
            self.attention_score(
                station_latent
            ),
            dim=1,
        )

        event_latent = torch.sum(
            attention_weights
            * station_latent,
            dim=1,
        )

        return (
            event_latent,
            station_latent,
            attention_weights,
        )

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        (
            event_latent,
            _,
            _,
        ) = self.encode_event(
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

        return self.query_decoder(
            torch.cat(
                [
                    expanded_event,
                    target_features,
                ],
                dim=-1,
            )
        )


# Historical compatibility name required by script 20.
MeanBase = AttentionPoolingBase


# ---------------------------------------------------------------------
# Tail-risk gated model.
# ---------------------------------------------------------------------

class TailGateModel(nn.Module):
    """
    Frozen ATTENTION base + target-specific tail-risk/correction head.

    Outputs:
        base:
            frozen base prediction [B,Q,2]
        probability:
            tail-risk score [B,Q,2]
        residual:
            non-negative maximum correction [B,Q,2]
        logits:
            pre-sigmoid risk logits [B,Q,2]
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
        self.maximum_correction = float(
            maximum_correction
        )

        # event latent + target xyz + frozen base PGA/PGV.
        context_input_dim = (
            self.hidden + 3 + 2
        )

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
            / (1.0 - prevalence)
        )

        with torch.no_grad():
            self.tail_classifier.bias.fill_(
                prior_logit
            )
            self.correction_head.bias.fill_(
                -2.0
            )

        for parameter in (
            self.base.parameters()
        ):
            parameter.requires_grad = False

        self.base.eval()

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        # Freeze base BatchNorm statistics as well as weights.
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
                event_latent,
                _,
                _,
            ) = self.base.encode_event(
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

        expanded_event = event_latent[
            :, None, :
        ].expand(
            -1,
            target_features.shape[1],
            -1,
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
# Loss.
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
                1.0 - prevalence
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
        (2,),
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
        ].to(device)

        input_features = batch[
            "input_features"
        ].to(device)

        target_features = batch[
            "target_features"
        ].to(device)

        target = batch[
            "target_log"
        ].to(device)

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
            nn.functional
            .binary_cross_entropy_with_logits(
                outputs["logits"],
                labels,
                pos_weight=pos_weight,
            )
        )

        linear_prediction = outputs[
            "linear"
        ]

        elementwise_regression = (
            nn.functional
            .smooth_l1_loss(
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

        # Directional underprediction penalty on high-motion targets.
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
            under = (
                under_gap.new_tensor(
                    0.0
                )
            )

        applied_correction = (
            outputs["probability"]
            * outputs["residual"]
        )

        non_tail = (
            1.0 - labels
        )

        if non_tail.sum() > 0:
            leakage = (
                applied_correction.pow(2)
                * non_tail
            ).sum() / non_tail.sum()
        else:
            leakage = (
                applied_correction.new_tensor(
                    0.0
                )
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
        totals[
            "classification"
        ] += float(
            classification.detach().cpu()
        )
        totals[
            "regression"
        ] += float(
            regression.detach().cpu()
        )
        totals[
            "under"
        ] += float(
            under.detach().cpu()
        )
        totals[
            "leakage"
        ] += float(
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


# ---------------------------------------------------------------------
# Validation metrics.
# ---------------------------------------------------------------------

def _canonical_macro(
    frame: pd.DataFrame,
    value_column: str,
    mask=None,
) -> float:
    """
    Manuscript aggregation:
        targets -> repeats -> events.
    """
    selected = frame

    if mask is not None:
        selected = frame.loc[
            np.asarray(
                mask,
                dtype=bool,
            )
        ]

    if len(selected) == 0:
        return float("nan")

    event_repeat = (
        selected.groupby(
            [
                "event_id",
                "repeat",
            ],
            sort=False,
        )[value_column]
        .mean()
        .reset_index()
    )

    if len(event_repeat) == 0:
        return float("nan")

    event = (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )[value_column]
        .mean()
    )

    if len(event) == 0:
        return float("nan")

    return float(
        event.mean()
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

    rows = []

    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                batch[
                    "input_waveforms"
                ].to(device),
                batch[
                    "input_features"
                ].to(device),
                batch[
                    "target_features"
                ].to(device),
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

            repeat = batch[
                "repeat"
            ].numpy()

            event_ids = list(
                batch[
                    "event_id"
                ]
            )

            for b in range(
                target.shape[0]
            ):
                for q in range(
                    target.shape[1]
                ):
                    true_pga = float(
                        target[
                            b, q, 0
                        ]
                    )
                    true_pgv = float(
                        target[
                            b, q, 1
                        ]
                    )

                    rows.append(
                        {
                            "event_id": str(
                                event_ids[b]
                            ),
                            "repeat": int(
                                repeat[b]
                            ),
                            "target_slot": int(
                                q
                            ),
                            "true_log10_pga": (
                                true_pga
                            ),
                            "true_log10_pgv": (
                                true_pgv
                            ),
                            "base_log10_pga": float(
                                base[
                                    b, q, 0
                                ]
                            ),
                            "base_log10_pgv": float(
                                base[
                                    b, q, 1
                                ]
                            ),
                            "final_log10_pga": float(
                                final[
                                    b, q, 0
                                ]
                            ),
                            "final_log10_pgv": float(
                                final[
                                    b, q, 1
                                ]
                            ),
                            "tail_probability_pga": float(
                                probability[
                                    b, q, 0
                                ]
                            ),
                            "tail_probability_pgv": float(
                                probability[
                                    b, q, 1
                                ]
                            ),
                            "residual_correction_pga": float(
                                residual[
                                    b, q, 0
                                ]
                            ),
                            "residual_correction_pgv": float(
                                residual[
                                    b, q, 1
                                ]
                            ),
                            "is_tail_pga": bool(
                                true_pga
                                >= thresholds[0]
                            ),
                            "is_tail_pgv": bool(
                                true_pgv
                                >= thresholds[1]
                            ),
                        }
                    )

    frame = pd.DataFrame(
        rows
    )

    if len(frame) == 0:
        raise RuntimeError(
            "No validation predictions were generated."
        )

    metrics = {
        "n_rows": int(
            len(frame)
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
            .shape[0]
        ),
    }

    for quantity in (
        "pga",
        "pgv",
    ):
        truth = frame[
            "true_log10_%s"
            % quantity
        ].to_numpy(
            dtype=float
        )

        base_prediction = frame[
            "base_log10_%s"
            % quantity
        ].to_numpy(
            dtype=float
        )

        linear_prediction = frame[
            "final_log10_%s"
            % quantity
        ].to_numpy(
            dtype=float
        )

        tail = frame[
            "is_tail_%s"
            % quantity
        ].to_numpy(
            dtype=bool
        )

        non_tail = ~tail

        for (
            model_name,
            prediction,
        ) in (
            (
                "base",
                base_prediction,
            ),
            (
                "linear",
                linear_prediction,
            ),
        ):
            error = (
                prediction - truth
            )
            absolute = np.abs(
                error
            )

            abs_column = (
                "__%s_abs_%s"
                % (
                    model_name,
                    quantity,
                )
            )

            frame[
                abs_column
            ] = absolute

            metrics[
                "%s_macro_mae_%s"
                % (
                    model_name,
                    quantity,
                )
            ] = _canonical_macro(
                frame,
                abs_column,
            )

            metrics[
                "%s_macro_tail_mae_%s"
                % (
                    model_name,
                    quantity,
                )
            ] = _canonical_macro(
                frame,
                abs_column,
                tail,
            )

            metrics[
                "%s_macro_non_tail_mae_%s"
                % (
                    model_name,
                    quantity,
                )
            ] = _canonical_macro(
                frame,
                abs_column,
                non_tail,
            )

            metrics[
                "%s_bias_%s"
                % (
                    model_name,
                    quantity,
                )
            ] = float(
                error.mean()
            )

            metrics[
                "%s_under05_%s"
                % (
                    model_name,
                    quantity,
                )
            ] = float(
                np.mean(
                    error <= -0.5
                )
            )

            if tail.any():
                # Canonical event-balanced tail U0.5.
                under_column = (
                    "__%s_under05_%s"
                    % (
                        model_name,
                        quantity,
                    )
                )
                frame[
                    under_column
                ] = (
                    error <= -0.5
                ).astype(float)

                metrics[
                    "%s_tail_under05_%s"
                    % (
                        model_name,
                        quantity,
                    )
                ] = _canonical_macro(
                    frame,
                    under_column,
                    tail,
                )
            else:
                metrics[
                    "%s_tail_under05_%s"
                    % (
                        model_name,
                        quantity,
                    )
                ] = float(
                    "nan"
                )

        probability_values = frame[
            "tail_probability_%s"
            % quantity
        ].to_numpy(
            dtype=float
        )

        labels = tail.astype(
            float
        )

        # Dependency-free approximate AUPRC for validation monitoring.
        if labels.sum() > 0:
            order = np.argsort(
                -probability_values
            )
            sorted_labels = (
                labels[order]
            )
            cumulative_positive = (
                np.cumsum(
                    sorted_labels
                )
            )
            rank = np.arange(
                1,
                len(labels) + 1,
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

            if hasattr(
                np,
                "trapezoid",
            ):
                auprc = np.trapezoid(
                    precision,
                    recall,
                )
            else:
                auprc = np.sum(
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
                "approx_auprc_%s"
                % quantity
            ] = float(
                auprc
            )
        else:
            metrics[
                "approx_auprc_%s"
                % quantity
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
            "model_state": (
                model.state_dict()
            ),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "epoch": int(
                epoch
            ),
            "args": vars(
                args
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "thresholds": (
                threshold_data
            ),
            "base_architecture": (
                "attention_pooling_full"
            ),
            "test_split_evaluated": (
                False
            ),
        },
        path,
    )


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
        help=(
            "Best script-40 attention_pooling checkpoint."
        ),
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
            "attention_tail_risk_gated_t0_5s_k5"
        ),
    )

    args = parser.parse_args()

    if args.validation_repeats < 1:
        raise ValueError(
            "--validation-repeats must be >=1."
        )

    if (
        args.final_validation_repeats
        < 1
    ):
        raise ValueError(
            "--final-validation-repeats must be >=1."
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
    if not threshold_path.exists():
        raise FileNotFoundError(
            threshold_path
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
            "Threshold JSON split-column mismatch: %r != %r"
            % (
                threshold_data.get(
                    "split_column"
                ),
                args.split_column,
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

    thresholds_tensor = (
        torch.from_numpy(
            thresholds_np
        ).to(device)
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

    base_checkpoint_path = Path(
        args.base_checkpoint
    )

    if not base_checkpoint_path.exists():
        raise FileNotFoundError(
            base_checkpoint_path
        )

    base_checkpoint = load_checkpoint(
        base_checkpoint_path,
        device,
    )

    checkpoint_variant = str(
        base_checkpoint.get(
            "variant",
            "",
        )
    )

    if (
        checkpoint_variant
        and checkpoint_variant
        != "attention_pooling"
    ):
        raise ValueError(
            "Expected attention_pooling base checkpoint, got %r"
            % checkpoint_variant
        )

    hidden = _infer_base_hidden(
        base_checkpoint
    )

    base = MeanBase(
        hidden
    ).to(
        device
    )

    # Exact script-40 attention checkpoint should load strictly.
    base.load_state_dict(
        base_checkpoint[
            "model_state"
        ],
        strict=True,
    )

    base.eval()

    model = TailGateModel(
        base=base,
        hidden=hidden,
        tail_hidden=(
            args.tail_hidden
        ),
        prevalence=prevalence,
        maximum_correction=(
            args.maximum_correction
        ),
    ).to(
        device
    )

    train_dataset = (
        SparseFieldDataset(
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
        SparseFieldDataset(
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
                args.validation_repeats
            ),
        )
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

    checkpoint_path = (
        output_directory
        / "best_overall.pt"
    )

    print(
        "=== Final attention-base tail-risk gated training ==="
    )
    print(
        "Device                  : %s"
        % device
    )
    print(
        "Base architecture       : attention_pooling_full"
    )
    print(
        "Base checkpoint epoch   : %s"
        % base_checkpoint.get(
            "epoch",
            "unknown",
        )
    )
    print(
        "Base checkpoint         : %s"
        % base_checkpoint_path.resolve()
    )
    print(
        "Train/Validation events : %d/%d"
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
        "Validation repeats      : %d"
        % args.validation_repeats
    )
    print(
        "Final val repeats       : %d"
        % args.final_validation_repeats
    )
    print(
        "Tail thresholds         : PGA=%.4f, PGV=%.4f"
        % (
            thresholds_np[0],
            thresholds_np[1],
        )
    )
    print(
        "Prevalence prior        : %.4f"
        % prevalence
    )
    print(
        "Trainable tail params   : %s"
        % format(
            sum(
                p.numel()
                for p in trainable_parameters
            ),
            ",",
        )
    )
    print(
        "Test split              : NOT ACCESSED"
    )

    history = []
    best_score = float(
        "inf"
    )
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        training_metrics = train_epoch(
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

        (
            validation_metrics,
            _,
        ) = evaluate_model(
            model,
            validation_loader,
            thresholds_np,
            device,
        )

        current_score = float(
            validation_metrics[
                "selection_score"
            ]
        )

        scheduler.step(
            current_score
        )

        row = {
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
                "validation_%s"
                % key: value
                for key, value
                in validation_metrics.items()
            },
        }

        history.append(
            row
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
            "Epoch %03d | "
            "loss=%.4f | "
            "linear overall=%.4f/%.4f | "
            "linear tail=%.4f/%.4f | "
            "tail U05=%.3f/%.3f | "
            "frozen base=%.4f/%.4f | "
            "AUPRC=%.3f/%.3f"
            % (
                epoch,
                training_metrics[
                    "loss"
                ],
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
                    "base_macro_mae_pga"
                ],
                validation_metrics[
                    "base_macro_mae_pgv"
                ],
                validation_metrics[
                    "approx_auprc_pga"
                ],
                validation_metrics[
                    "approx_auprc_pgv"
                ],
            )
        )

        if (
            stale_epochs
            >= args.early_stopping_patience
        ):
            print(
                "Early stopping at epoch %d; best epoch=%d."
                % (
                    epoch,
                    best_epoch,
                )
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

    # -------------------------------------------------------------
    # FINAL validation-only repeated file used for gamma selection.
    # -------------------------------------------------------------
    final_validation_dataset = (
        SparseFieldDataset(
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
                args.final_validation_repeats
            ),
        )
    )

    final_validation_loader = (
        DataLoader(
            final_validation_dataset,
            shuffle=False,
            **loader_arguments,
        )
    )

    (
        final_metrics,
        final_predictions,
    ) = evaluate_model(
        model,
        final_validation_loader,
        thresholds_np,
        device,
    )

    validation_prediction_path = (
        output_directory
        / "validation_predictions_overall.csv"
    )

    final_predictions.to_csv(
        validation_prediction_path,
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
                "base_architecture": (
                    "attention_pooling_full"
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
                **vars(args),
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
                "base_architecture": (
                    "attention_pooling_full"
                ),
                "metric_aggregation": (
                    "targets -> repeats -> events"
                ),
                "test_split_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Selected attention tail-risk checkpoint ==="
    )
    print(
        "Best epoch              : %d"
        % int(
            best_checkpoint[
                "epoch"
            ]
        )
    )
    print(
        "Frozen Base overall MAE : PGA=%.4f, PGV=%.4f"
        % (
            final_metrics[
                "base_macro_mae_pga"
            ],
            final_metrics[
                "base_macro_mae_pgv"
            ],
        )
    )
    print(
        "Frozen Base tail MAE    : PGA=%.4f, PGV=%.4f"
        % (
            final_metrics[
                "base_macro_tail_mae_pga"
            ],
            final_metrics[
                "base_macro_tail_mae_pgv"
            ],
        )
    )
    print(
        "Linear overall MAE      : PGA=%.4f, PGV=%.4f"
        % (
            final_metrics[
                "linear_macro_mae_pga"
            ],
            final_metrics[
                "linear_macro_mae_pgv"
            ],
        )
    )
    print(
        "Linear tail MAE         : PGA=%.4f, PGV=%.4f"
        % (
            final_metrics[
                "linear_macro_tail_mae_pga"
            ],
            final_metrics[
                "linear_macro_tail_mae_pgv"
            ],
        )
    )
    print(
        "Linear tail U0.5        : PGA=%.3f, PGV=%.3f"
        % (
            final_metrics[
                "linear_tail_under05_pga"
            ],
            final_metrics[
                "linear_tail_under05_pgv"
            ],
        )
    )
    print(
        "Risk AUPRC (approx.)    : PGA=%.3f, PGV=%.3f"
        % (
            final_metrics[
                "approx_auprc_pga"
            ],
            final_metrics[
                "approx_auprc_pgv"
            ],
        )
    )
    print(
        "Checkpoint              : %s"
        % checkpoint_path.resolve()
    )
    print(
        "Gate-scan predictions  : %s"
        % validation_prediction_path.resolve()
    )
    print(
        "Test split              : NOT ACCESSED"
    )


if __name__ == "__main__":
    main()
