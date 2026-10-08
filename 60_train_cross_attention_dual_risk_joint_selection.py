#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
60_train_cross_attention_dual_risk_joint_selection.py

Next-generation Tail model:
    Frozen Cross-Attention Base + Dual-Risk Power Gate

Final correction:
    y_final = y_base
              + p_under
              * p_tail**gamma
              * Delta

where:
    p_tail  = P(high-motion tail | causal observations, target)
    p_under = P(base severe-underprediction | causal observations, target)
    Delta   >= 0 is a target-specific correction magnitude.

Scientific protocol
-------------------
1. Reuse the already validation-selected Cross-Attention checkpoint.
2. Freeze the complete Cross-Attention Base.
3. Train only a small dual-risk/correction head.
4. Use TRAIN to estimate label prevalences.
5. Save every head epoch.
6. Use a fixed 20-repeat VALIDATION cache.
7. Jointly select head epoch + gamma on VALIDATION ONLY.
8. Test split is NEVER instantiated or read by this script.
9. Primary aggregation:
       targets -> repeats -> events.

Pre-specified validation guard (defaults)
-----------------------------------------
For BOTH PGA and PGV:
    overall MAE increase <= 0.005 log10
    non-tail MAE increase <= 0.003 log10
    |overall bias| <= 0.05

Among feasible candidates:
    1) minimum mean PGA/PGV tail MAE
    2) minimum mean PGA/PGV tail U0.5
    3) minimum mean overall MAE
    4) lower gamma
    5) earlier epoch

Important
---------
This is a NEXT-GENERATION development model motivated after inspecting the
previous grouped test. Therefore, the already-used 224-event grouped test
must NOT be treated as a new final test for this model. Use a genuinely new
held-out dataset/time period/region for final publication claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)
SEVERE_UNDER_THRESHOLD = -0.5


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def canonical_stable_seed(
    text: str,
    base_seed: int,
) -> int:
    """
    Same hash convention as script 40:
        sha256(f"{text}:{base_seed}")
    """
    digest = hashlib.sha256(
        f"{text}:{base_seed}".encode("utf-8")
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            "little",
        )
        % (2**32)
    )


def patch_strong_dataset_seed_protocol(
    baseline_module,
) -> None:
    """
    45's StrongBaselineDataset internally calls:
        stable_seed("strong:<split>:<event>:<epoch/repeat>", seed)

    The final strong-baseline run patched this to match script 40.
    We repeat that exact TRAIN/VALIDATION convention here.
    """

    def matched_seed(
        text: str,
        base_seed: int,
    ) -> int:
        text = str(text)

        if text.startswith("strong:"):
            text = text[len("strong:"):]

        return canonical_stable_seed(
            text,
            base_seed,
        )

    baseline_module.stable_seed = matched_seed


def parse_float_list(
    text: str,
) -> list[float]:
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
        raise ValueError(
            "No gamma values supplied."
        )

    return values


# ---------------------------------------------------------------------
# Frozen Cross-Attention feature extraction
# ---------------------------------------------------------------------

def cross_attention_base_context(
    base: nn.Module,
    input_waveforms: torch.Tensor,
    input_features: torch.Tensor,
    target_features: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Reproduce CrossAttentionBaseline.forward while exposing a target-specific
    feature vector for the risk head.

    Risk feature:
        [cross-attention context,
         target query embedding,
         raw target xyz,
         frozen base PGA/PGV prediction]

    Dimension:
        hidden + hidden + 3 + 2 = 2*hidden + 5
    """

    with torch.no_grad():
        station = base.encoder(
            input_waveforms,
            input_features,
        )

        attended, _ = (
            base.station_attention(
                station,
                station,
                station,
                need_weights=False,
            )
        )

        station = base.station_norm(
            station + attended
        )

        query = base.query_encoder(
            target_features
        )

        context, _ = (
            base.cross_attention(
                query,
                station,
                station,
                need_weights=False,
            )
        )

        context = base.cross_norm(
            query + context
        )

        base_prediction = base.decoder(
            torch.cat(
                [
                    context,
                    query,
                ],
                dim=-1,
            )
        )

        risk_feature = torch.cat(
            [
                context,
                query,
                target_features,
                base_prediction,
            ],
            dim=-1,
        )

    return (
        base_prediction,
        risk_feature,
    )


# ---------------------------------------------------------------------
# Dual-risk gated head
# ---------------------------------------------------------------------

class CrossAttentionDualRiskGate(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        hidden_dim: int,
        risk_hidden: int,
        maximum_correction: float,
        tail_prevalence: np.ndarray,
        under_prevalence: np.ndarray,
    ):
        super().__init__()

        self.base = base
        self.hidden_dim = int(hidden_dim)
        self.risk_hidden = int(risk_hidden)
        self.maximum_correction = float(
            maximum_correction
        )

        input_dim = (
            2 * self.hidden_dim
            + 5
        )

        self.shared = nn.Sequential(
            nn.Linear(
                input_dim,
                self.risk_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                self.risk_hidden,
                self.risk_hidden,
            ),
            nn.GELU(),
        )

        self.tail_classifier = nn.Linear(
            self.risk_hidden,
            2,
        )

        self.under_classifier = nn.Linear(
            self.risk_hidden,
            2,
        )

        self.correction_head = nn.Linear(
            self.risk_hidden,
            2,
        )

        tail_prevalence = np.clip(
            np.asarray(
                tail_prevalence,
                dtype=float,
            ),
            1e-3,
            1.0 - 1e-3,
        )

        under_prevalence = np.clip(
            np.asarray(
                under_prevalence,
                dtype=float,
            ),
            1e-3,
            1.0 - 1e-3,
        )

        with torch.no_grad():
            tail_prior = np.log(
                tail_prevalence
                / (
                    1.0
                    - tail_prevalence
                )
            )

            under_prior = np.log(
                under_prevalence
                / (
                    1.0
                    - under_prevalence
                )
            )

            self.tail_classifier.bias.copy_(
                torch.as_tensor(
                    tail_prior,
                    dtype=self.tail_classifier.bias.dtype,
                )
            )

            self.under_classifier.bias.copy_(
                torch.as_tensor(
                    under_prior,
                    dtype=self.under_classifier.bias.dtype,
                )
            )

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

        # Keep all frozen Cross-Attention BatchNorm / attention state fixed.
        self.base.eval()
        return self

    def head_from_feature(
        self,
        risk_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.shared(
            risk_feature
        )

        tail_logits = self.tail_classifier(
            hidden
        )

        under_logits = self.under_classifier(
            hidden
        )

        tail_probability = torch.sigmoid(
            tail_logits
        )

        under_probability = torch.sigmoid(
            under_logits
        )

        residual = (
            self.maximum_correction
            * torch.sigmoid(
                self.correction_head(
                    hidden
                )
            )
        )

        return {
            "tail_logits": tail_logits,
            "under_logits": under_logits,
            "tail_probability": tail_probability,
            "under_probability": under_probability,
            "residual": residual,
        }

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        (
            base_prediction,
            risk_feature,
        ) = cross_attention_base_context(
            self.base,
            input_waveforms,
            input_features,
            target_features,
        )

        head = self.head_from_feature(
            risk_feature
        )

        # Training-time dual-risk correction uses gamma=1.
        linear_correction = (
            head[
                "tail_probability"
            ]
            * head[
                "under_probability"
            ]
            * head[
                "residual"
            ]
        )

        linear_prediction = (
            base_prediction
            + linear_correction
        )

        return {
            "base": base_prediction,
            "risk_feature": risk_feature,
            "linear": linear_prediction,
            "linear_correction": linear_correction,
            **head,
        }


# ---------------------------------------------------------------------
# Labels and canonical metrics
# ---------------------------------------------------------------------

def tail_labels(
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


def under_labels(
    base_prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        (
            base_prediction
            - target
        )
        <= SEVERE_UNDER_THRESHOLD
    ).to(
        target.dtype
    )


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

        event_ids = event_ids[
            keep
        ]

        repeats = repeats[
            keep
        ]

        values = values[
            keep
        ]

    if len(values) == 0:
        return float("nan")

    frame = pd.DataFrame(
        {
            "event_id": (
                event_ids.astype(str)
            ),
            "repeat": (
                repeats.astype(int)
            ),
            "value": (
                values.astype(float)
            ),
        }
    )

    event_repeat = (
        frame.groupby(
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

    return float(
        event.mean()
    )


def single_quantity_metrics(
    event_ids: np.ndarray,
    repeats: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    tail_mask: np.ndarray,
) -> dict[str, float]:
    error = (
        prediction
        - truth
    )

    absolute = np.abs(
        error
    )

    non_tail = ~tail_mask

    return {
        "overall_mae": canonical_macro(
            event_ids,
            repeats,
            absolute,
        ),
        "non_tail_mae": canonical_macro(
            event_ids,
            repeats,
            absolute,
            non_tail,
        ),
        "tail_mae": canonical_macro(
            event_ids,
            repeats,
            absolute,
            tail_mask,
        ),
        "bias": canonical_macro(
            event_ids,
            repeats,
            error,
        ),
        "factor2": canonical_macro(
            event_ids,
            repeats,
            (
                absolute
                <= LOG10_FACTOR_2
            ).astype(float),
        ),
        "tail_factor2": canonical_macro(
            event_ids,
            repeats,
            (
                absolute
                <= LOG10_FACTOR_2
            ).astype(float),
            tail_mask,
        ),
        "under05": canonical_macro(
            event_ids,
            repeats,
            (
                error
                <= SEVERE_UNDER_THRESHOLD
            ).astype(float),
        ),
        "tail_under05": canonical_macro(
            event_ids,
            repeats,
            (
                error
                <= SEVERE_UNDER_THRESHOLD
            ).astype(float),
            tail_mask,
        ),
    }


def metric_bundle(
    cache: dict[str, Any],
    prediction: np.ndarray,
) -> dict[str, float]:
    result = {}

    for j, quantity in enumerate(
        (
            "pga",
            "pgv",
        )
    ):
        metrics = single_quantity_metrics(
            cache[
                "event_id"
            ],
            cache[
                "repeat"
            ],
            cache[
                "truth"
            ][
                :,
                j,
            ],
            prediction[
                :,
                j,
            ],
            cache[
                "tail"
            ][
                :,
                j,
            ],
        )

        for key, value in metrics.items():
            result[
                f"{key}_{quantity}"
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
        "mean_tail_under05"
    ] = 0.5 * (
        result[
            "tail_under05_pga"
        ]
        + result[
            "tail_under05_pgv"
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


# ---------------------------------------------------------------------
# TRAIN-only prevalence estimation
# ---------------------------------------------------------------------

def estimate_training_prevalence(
    base: nn.Module,
    loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    tail_positive = np.zeros(
        2,
        dtype=np.float64,
    )

    under_positive = np.zeros(
        2,
        dtype=np.float64,
    )

    total = np.zeros(
        2,
        dtype=np.float64,
    )

    thresholds_tensor = torch.as_tensor(
        thresholds,
        dtype=torch.float32,
        device=device,
    )

    base.eval()

    with torch.inference_mode():
        for batch in loader:
            target = batch[
                "target_log"
            ].to(
                device
            )

            (
                base_prediction,
                _,
            ) = cross_attention_base_context(
                base,
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

            tail = tail_labels(
                target,
                thresholds_tensor,
            )

            under = under_labels(
                base_prediction,
                target,
            )

            tail_positive += (
                tail.sum(
                    dim=(
                        0,
                        1,
                    )
                )
                .cpu()
                .numpy()
            )

            under_positive += (
                under.sum(
                    dim=(
                        0,
                        1,
                    )
                )
                .cpu()
                .numpy()
            )

            total += (
                np.asarray(
                    [
                        target.shape[
                            0
                        ]
                        * target.shape[
                            1
                        ],
                    ]
                    * 2,
                    dtype=np.float64,
                )
            )

    tail_prevalence = (
        tail_positive
        / np.maximum(
            total,
            1.0,
        )
    )

    under_prevalence = (
        under_positive
        / np.maximum(
            total,
            1.0,
        )
    )

    return {
        "tail": tail_prevalence,
        "under": under_prevalence,
        "total_per_quantity": total,
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_epoch(
    model: CrossAttentionDualRiskGate,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    thresholds: torch.Tensor,
    tail_pos_weight: torch.Tensor,
    under_pos_weight: torch.Tensor,
    lambda_tail_classification: float,
    lambda_under_classification: float,
    lambda_regression: float,
    risk_regression_weight: float,
    lambda_directional: float,
    lambda_leakage: float,
) -> dict[str, float]:
    model.train()

    totals = {
        "loss": 0.0,
        "tail_classification": 0.0,
        "under_classification": 0.0,
        "regression": 0.0,
        "directional": 0.0,
        "leakage": 0.0,
    }

    batches = 0

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

        target = batch[
            "target_log"
        ].to(
            device,
            non_blocking=True,
        )

        outputs = model(
            input_waveforms,
            input_features,
            target_features,
        )

        tail = tail_labels(
            target,
            thresholds,
        )

        under = under_labels(
            outputs[
                "base"
            ],
            target,
        )

        high_risk = (
            tail
            * under
        )

        tail_classification = (
            nn.functional
            .binary_cross_entropy_with_logits(
                outputs[
                    "tail_logits"
                ],
                tail,
                pos_weight=(
                    tail_pos_weight
                ),
            )
        )

        under_classification = (
            nn.functional
            .binary_cross_entropy_with_logits(
                outputs[
                    "under_logits"
                ],
                under,
                pos_weight=(
                    under_pos_weight
                ),
            )
        )

        elementwise_regression = (
            nn.functional
            .smooth_l1_loss(
                outputs[
                    "linear"
                ],
                target,
                reduction="none",
            )
        )

        # Emphasize only high-motion targets that the frozen base severely
        # underpredicts, rather than all high-motion targets.
        regression_weight = (
            1.0
            + float(
                risk_regression_weight
            )
            * high_risk
        )

        regression = (
            elementwise_regression
            * regression_weight
        ).mean()

        # Directional penalty only in the high-motion tail.
        under_gap = (
            target
            - outputs[
                "linear"
            ]
        ).clamp_min(
            0.0
        )

        if tail.sum() > 0:
            directional = (
                under_gap.pow(
                    2
                )
                * tail
            ).sum() / tail.sum()
        else:
            directional = (
                under_gap.new_tensor(
                    0.0
                )
            )

        # Corrections should remain small outside the actual
        # high-motion + severe-underprediction set.
        non_high_risk = (
            1.0
            - high_risk
        )

        correction = outputs[
            "linear_correction"
        ]

        if non_high_risk.sum() > 0:
            leakage = (
                correction.pow(
                    2
                )
                * non_high_risk
            ).sum() / non_high_risk.sum()
        else:
            leakage = (
                correction.new_tensor(
                    0.0
                )
            )

        loss = (
            float(
                lambda_tail_classification
            )
            * tail_classification
            + float(
                lambda_under_classification
            )
            * under_classification
            + float(
                lambda_regression
            )
            * regression
            + float(
                lambda_directional
            )
            * directional
            + float(
                lambda_leakage
            )
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

        totals[
            "loss"
        ] += float(
            loss.detach().cpu()
        )

        totals[
            "tail_classification"
        ] += float(
            tail_classification.detach().cpu()
        )

        totals[
            "under_classification"
        ] += float(
            under_classification.detach().cpu()
        )

        totals[
            "regression"
        ] += float(
            regression.detach().cpu()
        )

        totals[
            "directional"
        ] += float(
            directional.detach().cpu()
        )

        totals[
            "leakage"
        ] += float(
            leakage.detach().cpu()
        )

        batches += 1

    return {
        key: (
            value
            / max(
                batches,
                1,
            )
        )
        for key, value in totals.items()
    }


# ---------------------------------------------------------------------
# Fixed validation cache
# ---------------------------------------------------------------------

def build_validation_cache(
    model: CrossAttentionDualRiskGate,
    loader: DataLoader,
    thresholds_np: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    risk_feature_parts = []
    base_parts = []
    truth_parts = []

    event_ids = []
    repeats = []
    target_slots = []
    target_station_indices = []

    with torch.inference_mode():
        for batch in loader:
            target = batch[
                "target_log"
            ].numpy()

            (
                base_prediction,
                risk_feature,
            ) = cross_attention_base_context(
                model.base,
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

            risk_feature_parts.append(
                risk_feature.cpu().numpy()
            )

            base_parts.append(
                base_prediction.cpu().numpy()
            )

            truth_parts.append(
                target
            )

            batch_event_ids = list(
                batch[
                    "event_id"
                ]
            )

            batch_repeats = batch[
                "repeat"
            ].numpy()

            station_indices = batch[
                "target_station_index"
            ].numpy()

            for b in range(
                target.shape[
                    0
                ]
            ):
                for q in range(
                    target.shape[
                        1
                    ]
                ):
                    event_ids.append(
                        str(
                            batch_event_ids[
                                b
                            ]
                        )
                    )

                    repeats.append(
                        int(
                            batch_repeats[
                                b
                            ]
                        )
                    )

                    target_slots.append(
                        int(
                            q
                        )
                    )

                    target_station_indices.append(
                        int(
                            station_indices[
                                b,
                                q,
                            ]
                        )
                    )

    risk_feature = np.concatenate(
        risk_feature_parts,
        axis=0,
    ).reshape(
        -1,
        risk_feature_parts[
            0
        ].shape[
            -1
        ],
    )

    base_prediction = np.concatenate(
        base_parts,
        axis=0,
    ).reshape(
        -1,
        2,
    )

    truth = np.concatenate(
        truth_parts,
        axis=0,
    ).reshape(
        -1,
        2,
    )

    tail = (
        truth
        >= thresholds_np[
            None,
            :,
        ]
    )

    under = (
        (
            base_prediction
            - truth
        )
        <= SEVERE_UNDER_THRESHOLD
    )

    return {
        "risk_feature": (
            risk_feature.astype(
                np.float32
            )
        ),
        "base": (
            base_prediction.astype(
                np.float64
            )
        ),
        "truth": (
            truth.astype(
                np.float64
            )
        ),
        "tail": (
            tail.astype(
                bool
            )
        ),
        "under": (
            under.astype(
                bool
            )
        ),
        "event_id": np.asarray(
            event_ids,
            dtype=object,
        ),
        "repeat": np.asarray(
            repeats,
            dtype=np.int64,
        ),
        "target_slot": np.asarray(
            target_slots,
            dtype=np.int64,
        ),
        "target_station_index": np.asarray(
            target_station_indices,
            dtype=np.int64,
        ),
    }


def predict_head_from_cache(
    model: CrossAttentionDualRiskGate,
    risk_feature: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    tail_parts = []
    under_parts = []
    residual_parts = []

    model.eval()

    with torch.inference_mode():
        for start in range(
            0,
            len(
                risk_feature
            ),
            batch_size,
        ):
            stop = min(
                start
                + batch_size,
                len(
                    risk_feature
                ),
            )

            feature = torch.from_numpy(
                risk_feature[
                    start:stop
                ]
            ).to(
                device
            )

            outputs = model.head_from_feature(
                feature
            )

            tail_parts.append(
                outputs[
                    "tail_probability"
                ]
                .cpu()
                .numpy()
            )

            under_parts.append(
                outputs[
                    "under_probability"
                ]
                .cpu()
                .numpy()
            )

            residual_parts.append(
                outputs[
                    "residual"
                ]
                .cpu()
                .numpy()
            )

    return {
        "tail_probability": np.concatenate(
            tail_parts,
            axis=0,
        ).astype(
            np.float64
        ),
        "under_probability": np.concatenate(
            under_parts,
            axis=0,
        ).astype(
            np.float64
        ),
        "residual": np.concatenate(
            residual_parts,
            axis=0,
        ).astype(
            np.float64
        ),
    }


# ---------------------------------------------------------------------
# Validation selection
# ---------------------------------------------------------------------

def candidate_prediction(
    cache: dict[str, Any],
    head: dict[str, np.ndarray],
    gamma: float,
) -> np.ndarray:
    return (
        cache[
            "base"
        ]
        + head[
            "under_probability"
        ]
        * np.power(
            np.clip(
                head[
                    "tail_probability"
                ],
                0.0,
                1.0,
            ),
            float(
                gamma
            ),
        )
        * head[
            "residual"
        ]
    )


def feasibility_status(
    candidate: dict[str, float],
    base: dict[str, float],
    overall_budget: float,
    non_tail_budget: float,
    bias_limit: float,
) -> dict[str, Any]:
    result = {}

    constraints = []

    for quantity in (
        "pga",
        "pgv",
    ):
        overall_delta = (
            candidate[
                f"overall_mae_{quantity}"
            ]
            - base[
                f"overall_mae_{quantity}"
            ]
        )

        non_tail_delta = (
            candidate[
                f"non_tail_mae_{quantity}"
            ]
            - base[
                f"non_tail_mae_{quantity}"
            ]
        )

        tail_delta = (
            candidate[
                f"tail_mae_{quantity}"
            ]
            - base[
                f"tail_mae_{quantity}"
            ]
        )

        tail_under_delta = (
            candidate[
                f"tail_under05_{quantity}"
            ]
            - base[
                f"tail_under05_{quantity}"
            ]
        )

        result[
            f"overall_delta_{quantity}"
        ] = overall_delta

        result[
            f"non_tail_delta_{quantity}"
        ] = non_tail_delta

        result[
            f"tail_delta_{quantity}"
        ] = tail_delta

        result[
            f"tail_under05_delta_{quantity}"
        ] = tail_under_delta

        constraints.extend(
            [
                (
                    overall_delta
                    <= float(
                        overall_budget
                    )
                ),
                (
                    non_tail_delta
                    <= float(
                        non_tail_budget
                    )
                ),
                (
                    abs(
                        candidate[
                            f"bias_{quantity}"
                        ]
                    )
                    <= float(
                        bias_limit
                    )
                ),
            ]
        )

    result[
        "feasible"
    ] = bool(
        all(
            constraints
        )
    )

    return result


def save_selected_predictions(
    path: Path,
    cache: dict[str, Any],
    head: dict[str, np.ndarray],
    gamma: float,
) -> None:
    prediction = candidate_prediction(
        cache,
        head,
        gamma,
    )

    frame = pd.DataFrame(
        {
            "event_id": (
                cache[
                    "event_id"
                ]
            ),
            "repeat": (
                cache[
                    "repeat"
                ]
            ),
            "target_slot": (
                cache[
                    "target_slot"
                ]
            ),
            "target_station_index": (
                cache[
                    "target_station_index"
                ]
            ),
            "true_log10_pga": (
                cache[
                    "truth"
                ][
                    :,
                    0,
                ]
            ),
            "true_log10_pgv": (
                cache[
                    "truth"
                ][
                    :,
                    1,
                ]
            ),
            "base_log10_pga": (
                cache[
                    "base"
                ][
                    :,
                    0,
                ]
            ),
            "base_log10_pgv": (
                cache[
                    "base"
                ][
                    :,
                    1,
                ]
            ),
            "tail_probability_pga": (
                head[
                    "tail_probability"
                ][
                    :,
                    0,
                ]
            ),
            "tail_probability_pgv": (
                head[
                    "tail_probability"
                ][
                    :,
                    1,
                ]
            ),
            "under_probability_pga": (
                head[
                    "under_probability"
                ][
                    :,
                    0,
                ]
            ),
            "under_probability_pgv": (
                head[
                    "under_probability"
                ][
                    :,
                    1,
                ]
            ),
            "residual_correction_pga": (
                head[
                    "residual"
                ][
                    :,
                    0,
                ]
            ),
            "residual_correction_pgv": (
                head[
                    "residual"
                ][
                    :,
                    1,
                ]
            ),
            "final_log10_pga": (
                prediction[
                    :,
                    0,
                ]
            ),
            "final_log10_pgv": (
                prediction[
                    :,
                    1,
                ]
            ),
            "is_tail_pga": (
                cache[
                    "tail"
                ][
                    :,
                    0,
                ]
            ),
            "is_tail_pgv": (
                cache[
                    "tail"
                ][
                    :,
                    1,
                ]
            ),
            "base_under05_pga": (
                cache[
                    "under"
                ][
                    :,
                    0,
                ]
            ),
            "base_under05_pgv": (
                cache[
                    "under"
                ][
                    :,
                    1,
                ]
            ),
            "selected_gamma": np.full(
                len(
                    cache[
                        "event_id"
                    ]
                ),
                float(
                    gamma
                ),
            ),
        }
    )

    frame.to_csv(
        path,
        index=False,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

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
        help=(
            "Validation-selected cross_attention/best_model.pt "
            "from the completed strong-baseline experiment."
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
        "--prevalence-repeats",
        type=int,
        default=3,
        help=(
            "TRAIN-only fixed repeats used to estimate class prevalences."
        ),
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
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--risk-hidden",
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
        "--lambda-tail-classification",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--lambda-under-classification",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--lambda-regression",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--risk-regression-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--lambda-directional",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--lambda-leakage",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-positive-weight",
        type=float,
        default=9.0,
    )

    parser.add_argument(
        "--powers",
        default=(
            "1,1.5,2,2.5,3,4,5,6,7,8,10,12,15,20"
        ),
    )

    parser.add_argument(
        "--overall-budget",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--non-tail-budget",
        type=float,
        default=0.003,
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
            "cross_attention_dual_risk_joint_selection"
        ),
    )

    args = parser.parse_args()

    powers = parse_float_list(
        args.powers
    )

    if (
        args.hidden_dim
        % args.attention_heads
        != 0
    ):
        raise ValueError(
            "hidden_dim must be divisible by attention_heads."
        )

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

    set_seed(
        args.seed
    )

    baseline_module = load_module(
        args.baseline_module,
        "strong_baseline_module",
    )

    patch_strong_dataset_seed_protocol(
        baseline_module
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

    variant = str(
        base_checkpoint.get(
            "variant",
            "cross_attention",
        )
    )

    if (
        variant
        != "cross_attention"
    ):
        raise ValueError(
            "Base checkpoint must be cross_attention, "
            f"got {variant!r}."
        )

    base = baseline_module.make_model(
        "cross_attention",
        args.hidden_dim,
        args.attention_heads,
        50.0,
    ).to(
        device
    )

    base.load_state_dict(
        base_checkpoint[
            "model_state"
        ],
        strict=True,
    )

    base.eval()

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

    # TRAIN only, fixed repeats, prevalence estimation.
    prevalence_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name=(
                args.train_label
            ),
            training=False,
            repeats=(
                args.prevalence_repeats
            ),
            **common_dataset,
        )
    )

    eval_loader_kwargs = {
        "batch_size": (
            args.batch_size
        ),
        "num_workers": (
            args.num_workers
        ),
        "pin_memory": (
            device.type
            == "cuda"
        ),
        "persistent_workers": (
            args.num_workers
            > 0
        ),
    }

    prevalence_loader = DataLoader(
        prevalence_dataset,
        shuffle=False,
        **eval_loader_kwargs,
    )

    print(
        "=== Cross-Attention Dual-Risk Gate: TRAIN-only prevalence audit ==="
    )

    print(
        f"Device                         : {device}"
    )

    print(
        "Frozen base                    : cross_attention"
    )

    print(
        "Base checkpoint epoch          : "
        f"{base_checkpoint.get('epoch', 'unknown')}"
    )

    print(
        "Base validation score          : "
        f"{base_checkpoint.get('validation_score', float('nan'))}"
    )

    print(
        "Test split                     : NOT ACCESSED"
    )

    prevalence = estimate_training_prevalence(
        base,
        prevalence_loader,
        thresholds_np,
        device,
    )

    tail_prevalence = prevalence[
        "tail"
    ]

    under_prevalence = prevalence[
        "under"
    ]

    print(
        "TRAIN tail prevalence PGA/PGV  : "
        f"{tail_prevalence[0]:.4f}/"
        f"{tail_prevalence[1]:.4f}"
    )

    print(
        "TRAIN U0.5 prevalence PGA/PGV  : "
        f"{under_prevalence[0]:.4f}/"
        f"{under_prevalence[1]:.4f}"
    )

    tail_pos_weight_np = np.clip(
        (
            1.0
            - tail_prevalence
        )
        / np.maximum(
            tail_prevalence,
            1e-6,
        ),
        1.0,
        args.max_positive_weight,
    )

    under_pos_weight_np = np.clip(
        (
            1.0
            - under_prevalence
        )
        / np.maximum(
            under_prevalence,
            1e-6,
        ),
        1.0,
        args.max_positive_weight,
    )

    tail_pos_weight = torch.as_tensor(
        tail_pos_weight_np,
        dtype=torch.float32,
        device=device,
    )

    under_pos_weight = torch.as_tensor(
        under_pos_weight_np,
        dtype=torch.float32,
        device=device,
    )

    model = CrossAttentionDualRiskGate(
        base=base,
        hidden_dim=args.hidden_dim,
        risk_hidden=args.risk_hidden,
        maximum_correction=(
            args.maximum_correction
        ),
        tail_prevalence=(
            tail_prevalence
        ),
        under_prevalence=(
            under_prevalence
        ),
    ).to(
        device
    )

    train_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name=(
                args.train_label
            ),
            training=True,
            repeats=1,
            **common_dataset,
        )
    )

    validation_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name=(
                args.validation_label
            ),
            training=False,
            repeats=(
                args.training_validation_repeats
            ),
            **common_dataset,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
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

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **eval_loader_kwargs,
    )

    trainable = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_dir = (
        out_dir
        / "epoch_checkpoints"
    )

    epoch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out_dir
        / "pre_training_protocol.json"
    ).write_text(
        json.dumps(
            {
                **vars(
                    args
                ),
                "resolved_device": str(
                    device
                ),
                "base_checkpoint": str(
                    base_checkpoint_path.resolve()
                ),
                "base_checkpoint_epoch": (
                    base_checkpoint.get(
                        "epoch"
                    )
                ),
                "tail_thresholds": {
                    "pga": float(
                        thresholds_np[
                            0
                        ]
                    ),
                    "pgv": float(
                        thresholds_np[
                            1
                        ]
                    ),
                },
                "training_tail_prevalence": {
                    "pga": float(
                        tail_prevalence[
                            0
                        ]
                    ),
                    "pgv": float(
                        tail_prevalence[
                            1
                        ]
                    ),
                },
                "training_under05_prevalence": {
                    "pga": float(
                        under_prevalence[
                            0
                        ]
                    ),
                    "pgv": float(
                        under_prevalence[
                            1
                        ]
                    ),
                },
                "tail_pos_weight": {
                    "pga": float(
                        tail_pos_weight_np[
                            0
                        ]
                    ),
                    "pgv": float(
                        tail_pos_weight_np[
                            1
                        ]
                    ),
                },
                "under_pos_weight": {
                    "pga": float(
                        under_pos_weight_np[
                            0
                        ]
                    ),
                    "pgv": float(
                        under_pos_weight_np[
                            1
                        ]
                    ),
                },
                "formula": (
                    "y_final = y_cross_attention + "
                    "p_under * p_tail^gamma * Delta"
                ),
                "selection_uses_test": False,
                "old_grouped_test_is_not_new_final_test": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Dual-risk head training ==="
    )

    print(
        "Train/Validation events        : "
        f"{len(train_dataset.frame)}/"
        f"{len(validation_dataset.frame)}"
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
        "Trainable head parameters      : "
        f"{sum(p.numel() for p in trainable):,}"
    )

    print(
        "Overall/non-tail budgets       : "
        f"{args.overall_budget:.4f}/"
        f"{args.non_tail_budget:.4f}"
    )

    print(
        "Absolute bias limit            : "
        f"{args.bias_limit:.4f}"
    )

    print(
        f"Gamma grid                     : {powers}"
    )

    print(
        "Test split                     : NOT ACCESSED"
    )

    history = []

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            thresholds=(
                thresholds_tensor
            ),
            tail_pos_weight=(
                tail_pos_weight
            ),
            under_pos_weight=(
                under_pos_weight
            ),
            lambda_tail_classification=(
                args.lambda_tail_classification
            ),
            lambda_under_classification=(
                args.lambda_under_classification
            ),
            lambda_regression=(
                args.lambda_regression
            ),
            risk_regression_weight=(
                args.risk_regression_weight
            ),
            lambda_directional=(
                args.lambda_directional
            ),
            lambda_leakage=(
                args.lambda_leakage
            ),
        )

        # 3-repeat validation monitoring only; final selection is later 20-repeat.
        validation_cache = build_validation_cache(
            model,
            validation_loader,
            thresholds_np,
            device,
        )

        head_now = predict_head_from_cache(
            model,
            validation_cache[
                "risk_feature"
            ],
            device,
            args.head_eval_batch_size,
        )

        linear_prediction = (
            validation_cache[
                "base"
            ]
            + head_now[
                "under_probability"
            ]
            * head_now[
                "tail_probability"
            ]
            * head_now[
                "residual"
            ]
        )

        validation_metrics = metric_bundle(
            validation_cache,
            linear_prediction,
        )

        scheduler.step(
            validation_metrics[
                "mean_overall_mae"
            ]
        )

        checkpoint_path = (
            epoch_dir
            / (
                f"epoch_{epoch:03d}.pt"
            )
        )

        torch.save(
            {
                "model_state": (
                    model.state_dict()
                ),
                "epoch": int(
                    epoch
                ),
                "args": vars(
                    args
                ),
                "base_checkpoint": str(
                    base_checkpoint_path.resolve()
                ),
                "base_variant": (
                    "cross_attention"
                ),
                "validation_metrics_linear_gamma1": (
                    validation_metrics
                ),
                "thresholds": (
                    threshold_data
                ),
                "test_split_evaluated": False,
            },
            checkpoint_path,
        )

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "learning_rate": float(
                    optimizer
                    .param_groups[
                        0
                    ][
                        "lr"
                    ]
                ),
                **{
                    f"train_{key}": value
                    for key, value
                    in train_metrics.items()
                },
                **{
                    f"validation_{key}": value
                    for key, value
                    in validation_metrics.items()
                },
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            out_dir
            / "training_history.csv",
            index=False,
        )

        print(
            "Epoch %03d | "
            "loss=%.4f | "
            "val overall=%.4f/%.4f | "
            "tail=%.4f/%.4f | "
            "tail U05=%.3f/%.3f"
            % (
                epoch,
                train_metrics[
                    "loss"
                ],
                validation_metrics[
                    "overall_mae_pga"
                ],
                validation_metrics[
                    "overall_mae_pgv"
                ],
                validation_metrics[
                    "tail_mae_pga"
                ],
                validation_metrics[
                    "tail_mae_pgv"
                ],
                validation_metrics[
                    "tail_under05_pga"
                ],
                validation_metrics[
                    "tail_under05_pgv"
                ],
            )
        )

    # -----------------------------------------------------------------
    # FINAL 20-repeat validation cache.
    # -----------------------------------------------------------------

    selection_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name=(
                args.validation_label
            ),
            training=False,
            repeats=(
                args.selection_validation_repeats
            ),
            **common_dataset,
        )
    )

    selection_loader = DataLoader(
        selection_dataset,
        shuffle=False,
        **eval_loader_kwargs,
    )

    print(
        "\nBuilding frozen Cross-Attention 20-repeat validation cache..."
    )

    selection_cache = build_validation_cache(
        model,
        selection_loader,
        thresholds_np,
        device,
    )

    base_metrics = metric_bundle(
        selection_cache,
        selection_cache[
            "base"
        ],
    )

    (
        out_dir
        / "base_validation_metrics.json"
    ).write_text(
        json.dumps(
            base_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Cached validation target rows  : "
        f"{len(selection_cache['event_id'])}"
    )

    print(
        "Cross-Attention Base overall   : "
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
        "Cross-Attention Base tail      : "
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

    print(
        "Cross-Attention Base tail U05  : "
        "PGA=%.3f, PGV=%.3f"
        % (
            base_metrics[
                "tail_under05_pga"
            ],
            base_metrics[
                "tail_under05_pgv"
            ],
        )
    )

    # -----------------------------------------------------------------
    # VALIDATION-ONLY epoch x gamma scan.
    # -----------------------------------------------------------------

    scan_rows = []
    head_cache = {}

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):
        checkpoint = load_checkpoint(
            epoch_dir
            / f"epoch_{epoch:03d}.pt",
            device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ],
            strict=True,
        )

        model.eval()

        head = predict_head_from_cache(
            model,
            selection_cache[
                "risk_feature"
            ],
            device,
            args.head_eval_batch_size,
        )

        head_cache[
            epoch
        ] = head

        for gamma in powers:
            prediction = candidate_prediction(
                selection_cache,
                head,
                gamma,
            )

            candidate_metrics = metric_bundle(
                selection_cache,
                prediction,
            )

            status = feasibility_status(
                candidate_metrics,
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
                    **candidate_metrics,
                    **status,
                }
            )

    scan = pd.DataFrame(
        scan_rows
    )

    scan_path = (
        out_dir
        / "epoch_gamma_validation_scan.csv"
    )

    scan.to_csv(
        scan_path,
        index=False,
    )

    feasible = scan.loc[
        scan[
            "feasible"
        ].astype(
            bool
        )
    ].copy()

    if feasible.empty:
        print(
            "\nNO feasible Dual-Risk epoch-gamma pair "
            "satisfied all pre-specified validation constraints."
        )

        print(
            f"Inspect: {scan_path.resolve()}"
        )

        print(
            "Test split remains NOT ACCESSED."
        )

        raise RuntimeError(
            "No feasible validation candidate."
        )

    feasible = feasible.sort_values(
        [
            "mean_tail_mae",
            "mean_tail_under05",
            "mean_overall_mae",
            "gamma",
            "epoch",
        ],
        ascending=[
            True,
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
        selected[
            "epoch"
        ]
    )

    selected_gamma = float(
        selected[
            "gamma"
        ]
    )

    upper_boundary = bool(
        np.isclose(
            selected_gamma,
            max(
                powers
            ),
        )
    )

    selected_source = (
        epoch_dir
        / f"epoch_{selected_epoch:03d}.pt"
    )

    selected_checkpoint = (
        out_dir
        / "selected_dual_risk_head.pt"
    )

    shutil.copy2(
        selected_source,
        selected_checkpoint,
    )

    selected_head = head_cache[
        selected_epoch
    ]

    selected_predictions_path = (
        out_dir
        / "selected_validation_predictions.csv"
    )

    save_selected_predictions(
        selected_predictions_path,
        selection_cache,
        selected_head,
        selected_gamma,
    )

    feasible.head(
        50
    ).to_csv(
        out_dir
        / "top_feasible_validation_candidates.csv",
        index=False,
    )

    selection_payload = {
        "model": (
            "Cross-Attention Dual-Risk Power Gate"
        ),
        "formula": (
            "y_final = y_cross_attention + "
            "p_under * p_tail^gamma * Delta"
        ),
        "selected_epoch": (
            selected_epoch
        ),
        "selected_gamma": (
            selected_gamma
        ),
        "selected_at_upper_gamma_boundary": (
            upper_boundary
        ),
        "gamma_grid": powers,
        "selection_objective": (
            "minimum mean PGA/PGV high-motion-tail MAE "
            "among validation-feasible candidates"
        ),
        "tie_breaks": [
            "lower mean tail U0.5",
            "lower mean overall MAE",
            "lower gamma",
            "earlier epoch",
        ],
        "constraints": {
            "overall_mae_increase_each_quantity": (
                args.overall_budget
            ),
            "non_tail_mae_increase_each_quantity": (
                args.non_tail_budget
            ),
            "absolute_overall_bias_each_quantity": (
                args.bias_limit
            ),
        },
        "base_metrics": (
            base_metrics
        ),
        "selected_metrics": {
            key: (
                bool(
                    value
                )
                if isinstance(
                    value,
                    (
                        bool,
                        np.bool_,
                    ),
                )
                else (
                    int(
                        value
                    )
                    if isinstance(
                        value,
                        (
                            int,
                            np.integer,
                        ),
                    )
                    else (
                        float(
                            value
                        )
                        if isinstance(
                            value,
                            (
                                float,
                                np.floating,
                            ),
                        )
                        else value
                    )
                )
            )
            for key, value
            in selected.to_dict().items()
        },
        "base_checkpoint": str(
            base_checkpoint_path.resolve()
        ),
        "selected_head_checkpoint": str(
            selected_checkpoint.resolve()
        ),
        "metric_aggregation": (
            "targets -> repeats -> events"
        ),
        "test_split_evaluated": False,
        "old_grouped_test_is_not_new_final_test": True,
    }

    selection_json = (
        out_dir
        / "selected_epoch_gamma.json"
    )

    selection_json.write_text(
        json.dumps(
            selection_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Selected Cross-Attention Dual-Risk validation model ==="
    )

    print(
        f"Epoch                   : {selected_epoch}"
    )

    print(
        f"Gamma                   : {selected_gamma:.3f}"
    )

    print(
        f"At upper grid boundary  : {upper_boundary}"
    )

    print(
        "Overall MAE PGA/PGV     : "
        "%.4f / %.4f"
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
        "Overall delta PGA/PGV   : "
        "%+.4f / %+.4f"
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
        "Non-tail delta PGA/PGV  : "
        "%+.4f / %+.4f"
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
        "Tail MAE PGA/PGV        : "
        "%.4f / %.4f"
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
        "Tail delta PGA/PGV      : "
        "%+.4f / %+.4f"
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
        "Tail U0.5 PGA/PGV       : "
        "%.3f / %.3f"
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
        "Tail U0.5 delta PGA/PGV : "
        "%+.3f / %+.3f"
        % (
            selected[
                "tail_under05_delta_pga"
            ],
            selected[
                "tail_under05_delta_pgv"
            ],
        )
    )

    print(
        "Overall bias PGA/PGV    : "
        "%+.4f / %+.4f"
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
        "Feasible candidates     : "
        f"{len(feasible)} / {len(scan)}"
    )

    print(
        f"Selected checkpoint     : {selected_checkpoint.resolve()}"
    )

    print(
        f"Selection JSON          : {selection_json.resolve()}"
    )

    print(
        "Validation predictions  : "
        f"{selected_predictions_path.resolve()}"
    )

    print(
        f"Full scan               : {scan_path.resolve()}"
    )

    print(
        "Test split              : NOT ACCESSED"
    )

    if upper_boundary:
        print(
            "\nWARNING: selected gamma is at the upper "
            "validation-grid boundary."
        )

        print(
            "Extend gamma on VALIDATION ONLY before freezing."
        )

    else:
        print(
            "\nDual-Risk development model is now "
            "validation-locked."
        )

        print(
            "Do NOT use the previously inspected 224-event grouped "
            "test as a new final test."
        )


if __name__ == "__main__":
    main()
