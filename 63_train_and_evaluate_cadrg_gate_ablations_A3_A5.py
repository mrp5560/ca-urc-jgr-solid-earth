#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py

Gate-mechanism ablation suite for CA-DRG.

Already available and NOT retrained here:
    A2 = frozen Cross-Attention Base
    A6 = full CA-DRG
         y = y_CA + p_under * p_tail**gamma * Delta
         validation-selected epoch=9, gamma=10 in the completed run

New ablations trained in this script:
    A3 = Cross-Attention + Tail-only Gate
         y = y_CA + p_tail**gamma * Delta

    A4 = Cross-Attention + Under-only Gate
         y = y_CA + p_under**gamma * Delta

    A5 = Cross-Attention + Dual-Risk Gate without leakage regularization
         y = y_CA + p_under * p_tail**gamma * Delta
         lambda_leakage = 0

Protocol
--------
1. Use the SAME already validation-selected Cross-Attention checkpoint.
2. Freeze the entire Cross-Attention backbone.
3. Train only the ablation-specific risk/correction head on TRAIN.
4. Jointly select head epoch + gamma on 20-repeat VALIDATION ONLY.
5. Apply identical pre-specified validation guards:
       overall MAE increase <= 0.005 for BOTH PGA/PGV
       non-tail MAE increase <= 0.003 for BOTH PGA/PGV
       |overall bias| <= 0.05 for BOTH PGA/PGV
6. Among feasible candidates select:
       min mean tail MAE
       -> min mean tail U0.5
       -> min mean overall MAE
       -> lower gamma
       -> earlier epoch
7. After each variant is validation-locked, evaluate once on the exact
   previous 224-event / 20-repeat / 10-target locked grouped test draws.
8. A2 and A6 are loaded from the already generated CA-DRG benchmark CSV
   so all six rows use exactly the same 44,800 target predictions.

Canonical metric aggregation:
    targets -> repeats -> events

Ablation interpretation
-----------------------
A2 -> A3:
    contribution of a tail-only selective correction on Cross-Attention.

A2 -> A4:
    contribution of underprediction-risk-only correction.

A3/A4 -> A6:
    whether combining complementary tail and underprediction risks helps.

A5 -> A6:
    contribution of leakage regularization to preserving non-tail/overall
    performance.

This script is for the gate-mechanism ablation. Network-structure ablation
(self-attention / target-conditioned cross-attention / query conditioning)
should be reported separately.
"""

from __future__ import annotations

import argparse
import copy
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
SEVERE_UNDER_THRESHOLD = -0.5

VARIANTS = (
    "A3_tail_only",
    "A4_under_only",
    "A5_dual_no_leakage",
)


# ---------------------------------------------------------------------
# General utilities
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

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

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


def set_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def parse_float_list(
    text: str,
) -> list[float]:
    values = []

    for token in str(
        text
    ).split(","):
        token = token.strip()

        if not token:
            continue

        value = float(
            token
        )

        if value < 1.0:
            raise ValueError(
                "All gamma values must be >= 1."
            )

        values.append(
            value
        )

    values = sorted(
        set(
            values
        )
    )

    if not values:
        raise ValueError(
            "No gamma values supplied."
        )

    return values


def canonical_stable_seed(
    text: str,
    base_seed: int,
) -> int:
    digest = hashlib.sha256(
        f"{text}:{base_seed}".encode(
            "utf-8"
        )
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            "little",
        )
        % (
            2**32
        )
    )


def patch_train_validation_seed_protocol(
    baseline_module,
) -> None:
    """
    Match the script-40 TRAIN/VALIDATION draw convention used in the
    completed strong-baseline and CA-DRG development experiments.
    """

    def matched_seed(
        text: str,
        base_seed: int,
    ) -> int:
        text = str(
            text
        )

        if text.startswith(
            "strong:"
        ):
            text = text[
                len(
                    "strong:"
                ):
            ]

        return canonical_stable_seed(
            text,
            base_seed,
        )

    baseline_module.stable_seed = (
        matched_seed
    )


def patch_locked_test_seed_protocol(
    baseline_module,
) -> None:
    """
    Reproduce the exact already-used locked test target draws:
        strong:test:* -> locked:test:*
    """

    def locked_seed(
        text: str,
        base_seed: int,
    ) -> int:
        text = str(
            text
        )

        if text.startswith(
            "strong:test:"
        ):
            text = (
                "locked:"
                + text[
                    len(
                        "strong:"
                    ):
                ]
            )

        elif text.startswith(
            "strong:"
        ):
            text = text[
                len(
                    "strong:"
                ):
            ]

        return canonical_stable_seed(
            text,
            base_seed,
        )

    baseline_module.stable_seed = (
        locked_seed
    )


# ---------------------------------------------------------------------
# Frozen Cross-Attention context
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
            station
            + attended
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
            query
            + context
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
# Ablation head
# ---------------------------------------------------------------------

class AblationRiskGate(
    nn.Module
):
    def __init__(
        self,
        base: nn.Module,
        variant: str,
        hidden_dim: int,
        risk_hidden: int,
        maximum_correction: float,
        tail_prevalence: np.ndarray,
        under_prevalence: np.ndarray,
    ):
        super().__init__()

        if variant not in VARIANTS:
            raise ValueError(
                variant
            )

        self.base = base
        self.variant = str(
            variant
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.risk_hidden = int(
            risk_hidden
        )

        self.maximum_correction = float(
            maximum_correction
        )

        input_dim = (
            2
            * self.hidden_dim
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

        self.tail_classifier = (
            nn.Linear(
                self.risk_hidden,
                2,
            )
            if self.uses_tail
            else None
        )

        self.under_classifier = (
            nn.Linear(
                self.risk_hidden,
                2,
            )
            if self.uses_under
            else None
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
            1.0
            - 1e-3,
        )

        under_prevalence = np.clip(
            np.asarray(
                under_prevalence,
                dtype=float,
            ),
            1e-3,
            1.0
            - 1e-3,
        )

        with torch.no_grad():
            if (
                self.tail_classifier
                is not None
            ):
                tail_prior = np.log(
                    tail_prevalence
                    / (
                        1.0
                        - tail_prevalence
                    )
                )

                self.tail_classifier.bias.copy_(
                    torch.as_tensor(
                        tail_prior,
                        dtype=(
                            self
                            .tail_classifier
                            .bias
                            .dtype
                        ),
                    )
                )

            if (
                self.under_classifier
                is not None
            ):
                under_prior = np.log(
                    under_prevalence
                    / (
                        1.0
                        - under_prevalence
                    )
                )

                self.under_classifier.bias.copy_(
                    torch.as_tensor(
                        under_prior,
                        dtype=(
                            self
                            .under_classifier
                            .bias
                            .dtype
                        ),
                    )
                )

            self.correction_head.bias.fill_(
                -2.0
            )

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.base.eval()

    @property
    def uses_tail(
        self,
    ) -> bool:
        return (
            self.variant
            in (
                "A3_tail_only",
                "A5_dual_no_leakage",
            )
        )

    @property
    def uses_under(
        self,
    ) -> bool:
        return (
            self.variant
            in (
                "A4_under_only",
                "A5_dual_no_leakage",
            )
        )

    @property
    def leakage_enabled(
        self,
    ) -> bool:
        return (
            self.variant
            != "A5_dual_no_leakage"
        )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(
            mode
        )

        self.base.eval()

        return self

    def head_from_feature(
        self,
        risk_feature: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        hidden = self.shared(
            risk_feature
        )

        if (
            self.tail_classifier
            is not None
        ):
            tail_logits = (
                self.tail_classifier(
                    hidden
                )
            )

            tail_probability = (
                torch.sigmoid(
                    tail_logits
                )
            )

        else:
            tail_logits = (
                hidden.new_zeros(
                    (
                        *hidden.shape[:-1],
                        2,
                    )
                )
            )

            tail_probability = (
                hidden.new_ones(
                    (
                        *hidden.shape[:-1],
                        2,
                    )
                )
            )

        if (
            self.under_classifier
            is not None
        ):
            under_logits = (
                self.under_classifier(
                    hidden
                )
            )

            under_probability = (
                torch.sigmoid(
                    under_logits
                )
            )

        else:
            under_logits = (
                hidden.new_zeros(
                    (
                        *hidden.shape[:-1],
                        2,
                    )
                )
            )

            under_probability = (
                hidden.new_ones(
                    (
                        *hidden.shape[:-1],
                        2,
                    )
                )
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
            "tail_logits": (
                tail_logits
            ),
            "under_logits": (
                under_logits
            ),
            "tail_probability": (
                tail_probability
            ),
            "under_probability": (
                under_probability
            ),
            "residual": (
                residual
            ),
        }

    def correction(
        self,
        head: dict[
            str,
            torch.Tensor,
        ],
        gamma: float,
    ) -> torch.Tensor:
        if (
            self.variant
            == "A3_tail_only"
        ):
            gate = torch.pow(
                torch.clamp(
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

        elif (
            self.variant
            == "A4_under_only"
        ):
            gate = torch.pow(
                torch.clamp(
                    head[
                        "under_probability"
                    ],
                    0.0,
                    1.0,
                ),
                float(
                    gamma
                ),
            )

        elif (
            self.variant
            == "A5_dual_no_leakage"
        ):
            gate = (
                head[
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
                    float(
                        gamma
                    ),
                )
            )

        else:
            raise ValueError(
                self.variant
            )

        return (
            gate
            * head[
                "residual"
            ]
        )

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
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

        linear_correction = (
            self.correction(
                head,
                gamma=1.0,
            )
        )

        return {
            "base": (
                base_prediction
            ),
            "risk_feature": (
                risk_feature
            ),
            "linear_correction": (
                linear_correction
            ),
            "linear": (
                base_prediction
                + linear_correction
            ),
            **head,
        }


# ---------------------------------------------------------------------
# Labels / metrics
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

    if len(
        values
    ) == 0:
        return float(
            "nan"
        )

    frame = pd.DataFrame(
        {
            "event_id": (
                event_ids.astype(
                    str
                )
            ),
            "repeat": (
                repeats.astype(
                    int
                )
            ),
            "value": (
                values.astype(
                    float
                )
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
) -> dict[
    str,
    float,
]:
    residual = (
        prediction
        - truth
    )

    absolute = np.abs(
        residual
    )

    non_tail = (
        ~tail_mask
    )

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
            residual,
        ),
        "factor2": canonical_macro(
            event_ids,
            repeats,
            (
                absolute
                <= LOG10_FACTOR_2
            ).astype(
                float
            ),
        ),
        "tail_factor2": canonical_macro(
            event_ids,
            repeats,
            (
                absolute
                <= LOG10_FACTOR_2
            ).astype(
                float
            ),
            tail_mask,
        ),
        "under05": canonical_macro(
            event_ids,
            repeats,
            (
                residual
                <= SEVERE_UNDER_THRESHOLD
            ).astype(
                float
            ),
        ),
        "tail_under05": canonical_macro(
            event_ids,
            repeats,
            (
                residual
                <= SEVERE_UNDER_THRESHOLD
            ).astype(
                float
            ),
            tail_mask,
        ),
    }


def metric_bundle(
    cache: dict[
        str,
        Any,
    ],
    prediction: np.ndarray,
) -> dict[
    str,
    float,
]:
    result = {}

    for (
        j,
        quantity,
    ) in enumerate(
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

        for (
            key,
            value,
        ) in metrics.items():
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
# Frozen-base cache / prevalence
# ---------------------------------------------------------------------

def estimate_training_prevalence(
    base: nn.Module,
    loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
) -> dict[
    str,
    np.ndarray,
]:
    thresholds_tensor = (
        torch.as_tensor(
            thresholds,
            dtype=torch.float32,
            device=device,
        )
    )

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

            count = (
                target.shape[
                    0
                ]
                * target.shape[
                    1
                ]
            )

            total += np.asarray(
                [
                    count,
                    count,
                ],
                dtype=np.float64,
            )

    return {
        "tail": (
            tail_positive
            / np.maximum(
                total,
                1.0,
            )
        ),
        "under": (
            under_positive
            / np.maximum(
                total,
                1.0,
            )
        ),
    }


def build_eval_cache(
    base: nn.Module,
    loader: DataLoader,
    thresholds_np: np.ndarray,
    device: torch.device,
) -> dict[
    str,
    Any,
]:
    risk_feature_parts = []
    base_parts = []
    truth_parts = []

    event_ids = []
    repeats = []
    target_slots = []
    target_station_indices = []

    base.eval()

    with torch.inference_mode():
        for batch in loader:
            truth = batch[
                "target_log"
            ].numpy()

            (
                base_prediction,
                risk_feature,
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

            risk_feature_parts.append(
                risk_feature
                .cpu()
                .numpy()
            )

            base_parts.append(
                base_prediction
                .cpu()
                .numpy()
            )

            truth_parts.append(
                truth
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
                truth.shape[
                    0
                ]
            ):
                for q in range(
                    truth.shape[
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
        "risk_feature": risk_feature.astype(
            np.float32
        ),
        "base": base_prediction.astype(
            np.float64
        ),
        "truth": truth.astype(
            np.float64
        ),
        "tail": tail.astype(
            bool
        ),
        "under": under.astype(
            bool
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
    model: AblationRiskGate,
    risk_feature: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[
    str,
    np.ndarray,
]:
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

            feature = (
                torch.from_numpy(
                    risk_feature[
                        start:stop
                    ]
                )
                .to(
                    device
                )
            )

            head = model.head_from_feature(
                feature
            )

            tail_parts.append(
                head[
                    "tail_probability"
                ]
                .cpu()
                .numpy()
            )

            under_parts.append(
                head[
                    "under_probability"
                ]
                .cpu()
                .numpy()
            )

            residual_parts.append(
                head[
                    "residual"
                ]
                .cpu()
                .numpy()
            )

    return {
        "tail_probability": (
            np.concatenate(
                tail_parts,
                axis=0,
            )
            .astype(
                np.float64
            )
        ),
        "under_probability": (
            np.concatenate(
                under_parts,
                axis=0,
            )
            .astype(
                np.float64
            )
        ),
        "residual": (
            np.concatenate(
                residual_parts,
                axis=0,
            )
            .astype(
                np.float64
            )
        ),
    }


def candidate_prediction_numpy(
    variant: str,
    cache: dict[
        str,
        Any,
    ],
    head: dict[
        str,
        np.ndarray,
    ],
    gamma: float,
) -> np.ndarray:
    if (
        variant
        == "A3_tail_only"
    ):
        gate = np.power(
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

    elif (
        variant
        == "A4_under_only"
    ):
        gate = np.power(
            np.clip(
                head[
                    "under_probability"
                ],
                0.0,
                1.0,
            ),
            float(
                gamma
            ),
        )

    elif (
        variant
        == "A5_dual_no_leakage"
    ):
        gate = (
            head[
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
        )

    else:
        raise ValueError(
            variant
        )

    return (
        cache[
            "base"
        ]
        + gate
        * head[
            "residual"
        ]
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_epoch(
    model: AblationRiskGate,
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
) -> dict[
    str,
    float,
]:
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

        if (
            model.variant
            == "A3_tail_only"
        ):
            intended_risk = tail
            directional_mask = tail

        elif (
            model.variant
            == "A4_under_only"
        ):
            intended_risk = under
            directional_mask = under

        elif (
            model.variant
            == "A5_dual_no_leakage"
        ):
            intended_risk = (
                tail
                * under
            )
            # Match the full A6 directional penalty.
            directional_mask = tail

        else:
            raise ValueError(
                model.variant
            )

        zero = target.new_tensor(
            0.0
        )

        if model.uses_tail:
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

        else:
            tail_classification = zero

        if model.uses_under:
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

        else:
            under_classification = zero

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

        regression_weight = (
            1.0
            + float(
                risk_regression_weight
            )
            * intended_risk
        )

        regression = (
            elementwise_regression
            * regression_weight
        ).mean()

        under_gap = (
            target
            - outputs[
                "linear"
            ]
        ).clamp_min(
            0.0
        )

        if (
            directional_mask.sum()
            > 0
        ):
            directional = (
                under_gap.pow(
                    2
                )
                * directional_mask
            ).sum() / (
                directional_mask.sum()
            )

        else:
            directional = zero

        if model.leakage_enabled:
            non_intended = (
                1.0
                - intended_risk
            )

            if (
                non_intended.sum()
                > 0
            ):
                leakage = (
                    outputs[
                        "linear_correction"
                    ].pow(
                        2
                    )
                    * non_intended
                ).sum() / (
                    non_intended.sum()
                )

            else:
                leakage = zero

        else:
            leakage = zero

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
            tail_classification
            .detach()
            .cpu()
        )

        totals[
            "under_classification"
        ] += float(
            under_classification
            .detach()
            .cpu()
        )

        totals[
            "regression"
        ] += float(
            regression
            .detach()
            .cpu()
        )

        totals[
            "directional"
        ] += float(
            directional
            .detach()
            .cpu()
        )

        totals[
            "leakage"
        ] += float(
            leakage
            .detach()
            .cpu()
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
        for (
            key,
            value,
        ) in totals.items()
    }


def feasibility_status(
    candidate: dict[
        str,
        float,
    ],
    base: dict[
        str,
        float,
    ],
    overall_budget: float,
    non_tail_budget: float,
    bias_limit: float,
) -> dict[
    str,
    Any,
]:
    result = {}
    conditions = []

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

        conditions.extend(
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
            conditions
        )
    )

    return result


# ---------------------------------------------------------------------
# Reference A2 / A6 extraction
# ---------------------------------------------------------------------

def bool_array(
    series: pd.Series,
) -> np.ndarray:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.to_numpy(
            dtype=bool
        )

    normalized = (
        series.astype(
            str
        )
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    ).to_numpy(
        dtype=bool
    )


def reference_cache_from_locked_csv(
    path: Path,
) -> tuple[
    dict[
        str,
        Any,
    ],
    np.ndarray,
    np.ndarray,
]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    frame = pd.read_csv(
        path,
        dtype={
            "event_id": str,
        },
    )

    required = {
        "event_id",
        "repeat",
        "target_slot",
        "target_station_index",
        "true_log10_pga",
        "true_log10_pgv",
        "base_log10_pga",
        "base_log10_pgv",
        "final_log10_pga",
        "final_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Locked A2/A6 prediction file missing columns: "
            f"{sorted(missing)}"
        )

    cache = {
        "event_id": frame[
            "event_id"
        ].astype(
            str
        ).to_numpy(
            dtype=object
        ),
        "repeat": frame[
            "repeat"
        ].to_numpy(
            dtype=np.int64
        ),
        "target_slot": frame[
            "target_slot"
        ].to_numpy(
            dtype=np.int64
        ),
        "target_station_index": frame[
            "target_station_index"
        ].to_numpy(
            dtype=np.int64
        ),
        "truth": np.column_stack(
            [
                frame[
                    "true_log10_pga"
                ].to_numpy(
                    dtype=float
                ),
                frame[
                    "true_log10_pgv"
                ].to_numpy(
                    dtype=float
                ),
            ]
        ),
        "tail": np.column_stack(
            [
                bool_array(
                    frame[
                        "is_tail_pga"
                    ]
                ),
                bool_array(
                    frame[
                        "is_tail_pgv"
                    ]
                ),
            ]
        ),
    }

    a2 = np.column_stack(
        [
            frame[
                "base_log10_pga"
            ].to_numpy(
                dtype=float
            ),
            frame[
                "base_log10_pgv"
            ].to_numpy(
                dtype=float
            ),
        ]
    )

    a6 = np.column_stack(
        [
            frame[
                "final_log10_pga"
            ].to_numpy(
                dtype=float
            ),
            frame[
                "final_log10_pgv"
            ].to_numpy(
                dtype=float
            ),
        ]
    )

    return (
        cache,
        a2,
        a6,
    )


# ---------------------------------------------------------------------
# One variant
# ---------------------------------------------------------------------

def make_frozen_base(
    baseline_module,
    checkpoint: dict[
        str,
        Any,
    ],
    hidden_dim: int,
    attention_heads: int,
    device: torch.device,
) -> nn.Module:
    base = baseline_module.make_model(
        "cross_attention",
        hidden_dim,
        attention_heads,
        50.0,
    ).to(
        device
    )

    base.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    for parameter in base.parameters():
        parameter.requires_grad = False

    base.eval()

    return base


def run_variant(
    variant: str,
    args,
    baseline_module,
    base_checkpoint: dict[
        str,
        Any,
    ],
    thresholds_np: np.ndarray,
    thresholds_tensor: torch.Tensor,
    prevalence: dict[
        str,
        np.ndarray,
    ],
    tail_pos_weight: torch.Tensor,
    under_pos_weight: torch.Tensor,
    common_dataset: dict[
        str,
        Any,
    ],
    selection_cache: dict[
        str,
        Any,
    ],
    base_validation_metrics: dict[
        str,
        float,
    ],
    powers: list[
        float
    ],
    device: torch.device,
    variant_seed: int,
    out_dir: Path,
) -> dict[
    str,
    Any,
]:
    print(
        "\n"
        + "="
        * 80
    )

    print(
        f"Training {variant}"
    )

    print(
        "="
        * 80
    )

    set_seed(
        variant_seed
    )

    patch_train_validation_seed_protocol(
        baseline_module
    )

    base = make_frozen_base(
        baseline_module,
        base_checkpoint,
        args.hidden_dim,
        args.attention_heads,
        device,
    )

    model = AblationRiskGate(
        base=base,
        variant=variant,
        hidden_dim=(
            args.hidden_dim
        ),
        risk_hidden=(
            args.risk_hidden
        ),
        maximum_correction=(
            args.maximum_correction
        ),
        tail_prevalence=(
            prevalence[
                "tail"
            ]
        ),
        under_prevalence=(
            prevalence[
                "under"
            ]
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
        variant_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        generator=generator,
        num_workers=(
            args.num_workers
        ),
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
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    trainable = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=(
            args.learning_rate
        ),
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

    variant_dir = (
        out_dir
        / variant
    )

    checkpoint_dir = (
        variant_dir
        / "epoch_checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    effective_lambda_leakage = (
        0.0
        if variant
        == "A5_dual_no_leakage"
        else args.lambda_leakage
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
                effective_lambda_leakage
            ),
        )

        # 3-repeat monitor only.
        monitor_cache = build_eval_cache(
            model.base,
            validation_loader,
            thresholds_np,
            device,
        )

        head_monitor = (
            predict_head_from_cache(
                model,
                monitor_cache[
                    "risk_feature"
                ],
                device,
                args.head_eval_batch_size,
            )
        )

        monitor_prediction = (
            candidate_prediction_numpy(
                variant,
                monitor_cache,
                head_monitor,
                gamma=1.0,
            )
        )

        monitor_metrics = (
            metric_bundle(
                monitor_cache,
                monitor_prediction,
            )
        )

        scheduler.step(
            monitor_metrics[
                "mean_overall_mae"
            ]
        )

        checkpoint_path = (
            checkpoint_dir
            / f"epoch_{epoch:03d}.pt"
        )

        torch.save(
            {
                "variant": (
                    variant
                ),
                "model_state": (
                    model.state_dict()
                ),
                "epoch": (
                    int(
                        epoch
                    )
                ),
                "args": vars(
                    args
                ),
                "effective_lambda_leakage": (
                    float(
                        effective_lambda_leakage
                    )
                ),
                "monitor_metrics_gamma1": (
                    monitor_metrics
                ),
                "test_accessed_during_selection": (
                    False
                ),
            },
            checkpoint_path,
        )

        history.append(
            {
                "epoch": (
                    int(
                        epoch
                    )
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
                    for (
                        key,
                        value,
                    ) in train_metrics.items()
                },
                **{
                    f"validation_monitor_{key}": value
                    for (
                        key,
                        value,
                    ) in monitor_metrics.items()
                },
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            variant_dir
            / "training_history.csv",
            index=False,
        )

        print(
            "Epoch %03d | loss=%.4f | "
            "val overall=%.4f/%.4f | "
            "tail=%.4f/%.4f | "
            "U05=%.3f/%.3f"
            % (
                epoch,
                train_metrics[
                    "loss"
                ],
                monitor_metrics[
                    "overall_mae_pga"
                ],
                monitor_metrics[
                    "overall_mae_pgv"
                ],
                monitor_metrics[
                    "tail_mae_pga"
                ],
                monitor_metrics[
                    "tail_mae_pgv"
                ],
                monitor_metrics[
                    "tail_under05_pga"
                ],
                monitor_metrics[
                    "tail_under05_pgv"
                ],
            )
        )

    # -------------------------------------------------------------
    # 20-repeat validation-only joint epoch x gamma selection.
    # -------------------------------------------------------------

    scan_rows = []
    selected_head_cache = {}

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):
        checkpoint = load_checkpoint(
            checkpoint_dir
            / f"epoch_{epoch:03d}.pt",
            device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ],
            strict=True,
        )

        head = predict_head_from_cache(
            model,
            selection_cache[
                "risk_feature"
            ],
            device,
            args.head_eval_batch_size,
        )

        selected_head_cache[
            epoch
        ] = head

        for gamma in powers:
            prediction = (
                candidate_prediction_numpy(
                    variant,
                    selection_cache,
                    head,
                    gamma,
                )
            )

            metrics = metric_bundle(
                selection_cache,
                prediction,
            )

            status = feasibility_status(
                metrics,
                base_validation_metrics,
                args.overall_budget,
                args.non_tail_budget,
                args.bias_limit,
            )

            scan_rows.append(
                {
                    "variant": (
                        variant
                    ),
                    "epoch": (
                        int(
                            epoch
                        )
                    ),
                    "gamma": (
                        float(
                            gamma
                        )
                    ),
                    **metrics,
                    **status,
                }
            )

    scan = pd.DataFrame(
        scan_rows
    )

    scan.to_csv(
        variant_dir
        / "epoch_gamma_validation_scan.csv",
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
        (
            variant_dir
            / "NO_FEASIBLE_CANDIDATE.txt"
        ).write_text(
            (
                "No epoch/gamma pair satisfied the "
                "pre-specified validation constraints.\n"
            ),
            encoding="utf-8",
        )

        print(
            f"{variant}: NO feasible validation candidate."
        )

        return {
            "variant": variant,
            "feasible": False,
        }

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

    selected_at_upper_boundary = bool(
        np.isclose(
            selected_gamma,
            max(
                powers
            ),
        )
    )

    selected_source = (
        checkpoint_dir
        / (
            f"epoch_{selected_epoch:03d}.pt"
        )
    )

    selected_checkpoint_path = (
        variant_dir
        / "selected_head.pt"
    )

    shutil.copy2(
        selected_source,
        selected_checkpoint_path,
    )

    selection = {
        "variant": (
            variant
        ),
        "selected_epoch": (
            selected_epoch
        ),
        "selected_gamma": (
            selected_gamma
        ),
        "selected_at_upper_gamma_boundary": (
            selected_at_upper_boundary
        ),
        "feasible_candidates": int(
            len(
                feasible
            )
        ),
        "total_candidates": int(
            len(
                scan
            )
        ),
        "base_validation_metrics": (
            base_validation_metrics
        ),
        "selected_validation_metrics": {
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
            for (
                key,
                value,
            ) in selected.to_dict().items()
        },
        "effective_lambda_leakage": float(
            effective_lambda_leakage
        ),
        "selection_used_test": False,
    }

    (
        variant_dir
        / "selected_epoch_gamma.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    feasible.head(
        50
    ).to_csv(
        variant_dir
        / "top_feasible_validation_candidates.csv",
        index=False,
    )

    print(
        "\nSelected %s | epoch=%d | gamma=%g | "
        "boundary=%s | feasible=%d/%d"
        % (
            variant,
            selected_epoch,
            selected_gamma,
            selected_at_upper_boundary,
            len(
                feasible
            ),
            len(
                scan
            ),
        )
    )

    print(
        "Validation overall PGA/PGV : %.4f / %.4f"
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
        "Validation tail PGA/PGV    : %.4f / %.4f"
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
        "Validation tail U05 PGA/PGV: %.3f / %.3f"
        % (
            selected[
                "tail_under05_pga"
            ],
            selected[
                "tail_under05_pgv"
            ],
        )
    )

    return {
        "variant": (
            variant
        ),
        "feasible": True,
        "selected_epoch": (
            selected_epoch
        ),
        "selected_gamma": (
            selected_gamma
        ),
        "selected_at_upper_gamma_boundary": (
            selected_at_upper_boundary
        ),
        "selected_checkpoint": str(
            selected_checkpoint_path
        ),
        "selection": (
            selection
        ),
    }


# ---------------------------------------------------------------------
# Locked test evaluation
# ---------------------------------------------------------------------

def evaluate_variant_on_locked_test(
    variant_result: dict[
        str,
        Any,
    ],
    args,
    baseline_module,
    base_checkpoint: dict[
        str,
        Any,
    ],
    prevalence: dict[
        str,
        np.ndarray,
    ],
    test_cache: dict[
        str,
        Any,
    ],
    device: torch.device,
) -> tuple[
    dict[
        str,
        float,
    ],
    np.ndarray,
]:
    variant = str(
        variant_result[
            "variant"
        ]
    )

    checkpoint = load_checkpoint(
        variant_result[
            "selected_checkpoint"
        ],
        device,
    )

    base = make_frozen_base(
        baseline_module,
        base_checkpoint,
        args.hidden_dim,
        args.attention_heads,
        device,
    )

    model = AblationRiskGate(
        base=base,
        variant=variant,
        hidden_dim=(
            args.hidden_dim
        ),
        risk_hidden=(
            args.risk_hidden
        ),
        maximum_correction=(
            args.maximum_correction
        ),
        tail_prevalence=(
            prevalence[
                "tail"
            ]
        ),
        under_prevalence=(
            prevalence[
                "under"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    head = predict_head_from_cache(
        model,
        test_cache[
            "risk_feature"
        ],
        device,
        args.head_eval_batch_size,
    )

    prediction = (
        candidate_prediction_numpy(
            variant,
            test_cache,
            head,
            float(
                variant_result[
                    "selected_gamma"
                ]
            ),
        )
    )

    metrics = metric_bundle(
        test_cache,
        prediction,
    )

    return (
        metrics,
        prediction,
    )


def prediction_pairing_audit(
    test_cache: dict[
        str,
        Any,
    ],
    reference_cache: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    current = pd.DataFrame(
        {
            "event_id": (
                test_cache[
                    "event_id"
                ].astype(
                    str
                )
            ),
            "repeat": (
                test_cache[
                    "repeat"
                ]
            ),
            "target_station_index": (
                test_cache[
                    "target_station_index"
                ]
            ),
            "true_pga": (
                test_cache[
                    "truth"
                ][
                    :,
                    0,
                ]
            ),
            "true_pgv": (
                test_cache[
                    "truth"
                ][
                    :,
                    1,
                ]
            ),
        }
    )

    reference = pd.DataFrame(
        {
            "event_id": (
                reference_cache[
                    "event_id"
                ].astype(
                    str
                )
            ),
            "repeat": (
                reference_cache[
                    "repeat"
                ]
            ),
            "target_station_index": (
                reference_cache[
                    "target_station_index"
                ]
            ),
            "true_pga": (
                reference_cache[
                    "truth"
                ][
                    :,
                    0,
                ]
            ),
            "true_pgv": (
                reference_cache[
                    "truth"
                ][
                    :,
                    1,
                ]
            ),
        }
    )

    keys = [
        "event_id",
        "repeat",
        "target_station_index",
    ]

    paired = current.merge(
        reference,
        on=keys,
        how="inner",
        suffixes=(
            "_current",
            "_reference",
        ),
        validate="one_to_one",
    )

    audit = {
        "current_rows": int(
            len(
                current
            )
        ),
        "reference_rows": int(
            len(
                reference
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
                current
            )
            == len(
                reference
            )
        ),
        "max_abs_truth_diff_pga": float(
            np.max(
                np.abs(
                    paired[
                        "true_pga_current"
                    ].to_numpy(
                        dtype=float
                    )
                    - paired[
                        "true_pga_reference"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        ),
        "max_abs_truth_diff_pgv": float(
            np.max(
                np.abs(
                    paired[
                        "true_pgv_current"
                    ].to_numpy(
                        dtype=float
                    )
                    - paired[
                        "true_pgv_reference"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        ),
    }

    if not audit[
        "exact_pairing"
    ]:
        raise RuntimeError(
            f"Locked test pairing failed: {audit}"
        )

    if (
        audit[
            "max_abs_truth_diff_pga"
        ]
        > 1e-5
        or audit[
            "max_abs_truth_diff_pgv"
        ]
        > 1e-5
    ):
        raise RuntimeError(
            f"Locked test truth audit failed: {audit}"
        )

    return audit


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
        "--test-label",
        default="test",
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
        "--full-cadrg-predictions",
        required=True,
        help=(
            "Existing 44,800-row locked_dual_risk_predictions.csv. "
            "Provides A2 and A6 reference predictions and exact pairing audit."
        ),
    )

    parser.add_argument(
        "--variants",
        default=(
            ",".join(
                VARIANTS
            )
        ),
        help=(
            "Comma-separated subset of A3_tail_only,"
            "A4_under_only,A5_dual_no_leakage"
        ),
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
        "--test-repeats",
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
            "cadrg_gate_ablation_A3_A5"
        ),
    )

    args = parser.parse_args()

    variants = [
        token.strip()
        for token
        in args.variants.split(
            ","
        )
        if token.strip()
    ]

    unknown = set(
        variants
    ).difference(
        VARIANTS
    )

    if unknown:
        raise ValueError(
            f"Unknown variants: {sorted(unknown)}"
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

    patch_train_validation_seed_protocol(
        baseline_module
    )

    threshold_data = json.loads(
        Path(
            args.threshold_json
        ).read_text(
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

    thresholds_tensor = (
        torch.as_tensor(
            thresholds_np,
            dtype=torch.float32,
            device=device,
        )
    )

    base_checkpoint = load_checkpoint(
        args.base_checkpoint,
        device,
    )

    if (
        str(
            base_checkpoint.get(
                "variant",
                "cross_attention",
            )
        )
        != "cross_attention"
    ):
        raise ValueError(
            "Base checkpoint is not cross_attention."
        )

    common_dataset = {
        "manifest": (
            args.manifest
        ),
        "h5_root": (
            args.h5_root
        ),
        "split_column": (
            args.split_column
        ),
        "t0_sec": (
            args.t0_sec
        ),
        "input_stations": (
            args.input_stations
        ),
        "target_stations": (
            args.target_stations
        ),
        "input_pre_sec": (
            args.input_pre_sec
        ),
        "seed": (
            args.seed
        ),
    }

    # -------------------------------------------------------------
    # TRAIN-only prevalence.
    # -------------------------------------------------------------

    prevalence_base = make_frozen_base(
        baseline_module,
        base_checkpoint,
        args.hidden_dim,
        args.attention_heads,
        device,
    )

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

    prevalence_loader = DataLoader(
        prevalence_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    prevalence = (
        estimate_training_prevalence(
            prevalence_base,
            prevalence_loader,
            thresholds_np,
            device,
        )
    )

    tail_pos_weight_np = np.clip(
        (
            1.0
            - prevalence[
                "tail"
            ]
        )
        / np.maximum(
            prevalence[
                "tail"
            ],
            1e-6,
        ),
        1.0,
        args.max_positive_weight,
    )

    under_pos_weight_np = np.clip(
        (
            1.0
            - prevalence[
                "under"
            ]
        )
        / np.maximum(
            prevalence[
                "under"
            ],
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

    # -------------------------------------------------------------
    # Fixed 20-repeat validation cache from the frozen A2 base.
    # -------------------------------------------------------------

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
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    selection_cache = build_eval_cache(
        prevalence_base,
        selection_loader,
        thresholds_np,
        device,
    )

    base_validation_metrics = (
        metric_bundle(
            selection_cache,
            selection_cache[
                "base"
            ],
        )
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out_dir
        / "protocol.json"
    ).write_text(
        json.dumps(
            {
                **vars(
                    args
                ),
                "variants": (
                    variants
                ),
                "A2_status": (
                    "already available; frozen Cross-Attention Base"
                ),
                "A6_status": (
                    "already available; full CA-DRG"
                ),
                "selection_uses_test": (
                    False
                ),
                "metric_hierarchy": (
                    "targets -> repeats -> events"
                ),
                "tail_prevalence_train": {
                    "pga": float(
                        prevalence[
                            "tail"
                        ][
                            0
                        ]
                    ),
                    "pgv": float(
                        prevalence[
                            "tail"
                        ][
                            1
                        ]
                    ),
                },
                "under_prevalence_train": {
                    "pga": float(
                        prevalence[
                            "under"
                        ][
                            0
                        ]
                    ),
                    "pgv": float(
                        prevalence[
                            "under"
                        ][
                            1
                        ]
                    ),
                },
                "base_validation_metrics": (
                    base_validation_metrics
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== CA-DRG gate ablation A3-A5 ==="
    )

    print(
        f"Device                    : {device}"
    )

    print(
        "A2                        : already available"
    )

    print(
        "A6                        : already available"
    )

    print(
        f"New variants               : {variants}"
    )

    print(
        "Validation Base overall    : "
        "%.4f / %.4f"
        % (
            base_validation_metrics[
                "overall_mae_pga"
            ],
            base_validation_metrics[
                "overall_mae_pgv"
            ],
        )
    )

    print(
        "Validation Base tail       : "
        "%.4f / %.4f"
        % (
            base_validation_metrics[
                "tail_mae_pga"
            ],
            base_validation_metrics[
                "tail_mae_pgv"
            ],
        )
    )

    print(
        "TEST DURING SELECTION      : NOT ACCESSED"
    )

    results = []

    for index, variant in enumerate(
        variants
    ):
        result = run_variant(
            variant=variant,
            args=args,
            baseline_module=(
                baseline_module
            ),
            base_checkpoint=(
                base_checkpoint
            ),
            thresholds_np=(
                thresholds_np
            ),
            thresholds_tensor=(
                thresholds_tensor
            ),
            prevalence=(
                prevalence
            ),
            tail_pos_weight=(
                tail_pos_weight
            ),
            under_pos_weight=(
                under_pos_weight
            ),
            common_dataset=(
                common_dataset
            ),
            selection_cache=(
                selection_cache
            ),
            base_validation_metrics=(
                base_validation_metrics
            ),
            powers=powers,
            device=device,
            variant_seed=(
                args.seed
                + 1000
                * (
                    index
                    + 1
                )
            ),
            out_dir=(
                out_dir
            ),
        )

        results.append(
            result
        )

    feasible_results = [
        result
        for result
        in results
        if result.get(
            "feasible",
            False,
        )
    ]

    if not feasible_results:
        raise RuntimeError(
            "None of A3-A5 produced a feasible validation-locked model."
        )

    boundary_variants = [
        result[
            "variant"
        ]
        for result
        in feasible_results
        if result[
            "selected_at_upper_gamma_boundary"
        ]
    ]

    if boundary_variants:
        print(
            "\nWARNING: selected gamma is at upper boundary for:"
        )

        for variant in boundary_variants:
            print(
                f"  {variant}"
            )

        print(
            "Extend gamma on VALIDATION ONLY for those variants before "
            "using their test metrics in the final ablation table."
        )

    # -------------------------------------------------------------
    # Locked test. Exact same 44,800 rows as A2/A6.
    # -------------------------------------------------------------

    patch_locked_test_seed_protocol(
        baseline_module
    )

    test_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name=(
                args.test_label
            ),
            training=False,
            repeats=(
                args.test_repeats
            ),
            **common_dataset,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    test_base = make_frozen_base(
        baseline_module,
        base_checkpoint,
        args.hidden_dim,
        args.attention_heads,
        device,
    )

    test_cache = build_eval_cache(
        test_base,
        test_loader,
        thresholds_np,
        device,
    )

    (
        reference_cache,
        a2_prediction,
        a6_prediction,
    ) = reference_cache_from_locked_csv(
        Path(
            args.full_cadrg_predictions
        )
    )

    pairing_audit = (
        prediction_pairing_audit(
            test_cache,
            reference_cache,
        )
    )

    (
        out_dir
        / "locked_test_pairing_audit.json"
    ).write_text(
        json.dumps(
            pairing_audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    if (
        len(
            test_cache[
                "event_id"
            ]
        )
        != 44800
    ):
        print(
            "WARNING: locked test row count is not 44,800; "
            f"got {len(test_cache['event_id']):,}."
        )

    # A2/A6 point estimates from already existing exact-paired CSV.
    a2_metrics = metric_bundle(
        reference_cache,
        a2_prediction,
    )

    a6_metrics = metric_bundle(
        reference_cache,
        a6_prediction,
    )

    final_rows = []

    def append_metrics(
        variant: str,
        label: str,
        metrics: dict[
            str,
            float,
        ],
        selected_epoch: int | None,
        selected_gamma: float | None,
        status: str,
    ) -> None:
        row = {
            "variant": (
                variant
            ),
            "label": (
                label
            ),
            "status": (
                status
            ),
            "selected_epoch": (
                selected_epoch
            ),
            "selected_gamma": (
                selected_gamma
            ),
        }

        row.update(
            metrics
        )

        final_rows.append(
            row
        )

    append_metrics(
        "A2_cross_attention_base",
        "A2 Cross-Attention Base",
        a2_metrics,
        None,
        None,
        "existing",
    )

    prediction_columns = {
        "A2_cross_attention_base": (
            a2_prediction
        ),
        "A6_full_cadrg": (
            a6_prediction
        ),
    }

    for result in feasible_results:
        if result[
            "selected_at_upper_gamma_boundary"
        ]:
            # Keep selection result but skip a supposedly final test
            # interpretation until validation-only gamma expansion is done.
            print(
                f"Skipping locked-test interpretation for {result['variant']} "
                "because gamma is at the upper validation boundary."
            )

            continue

        metrics, prediction = (
            evaluate_variant_on_locked_test(
                result,
                args,
                baseline_module,
                base_checkpoint,
                prevalence,
                test_cache,
                device,
            )
        )

        append_metrics(
            result[
                "variant"
            ],
            result[
                "variant"
            ],
            metrics,
            int(
                result[
                    "selected_epoch"
                ]
            ),
            float(
                result[
                    "selected_gamma"
                ]
            ),
            "new_ablation",
        )

        prediction_columns[
            result[
                "variant"
            ]
        ] = prediction

    append_metrics(
        "A6_full_cadrg",
        "A6 Full CA-DRG",
        a6_metrics,
        9,
        10.0,
        "existing",
    )

    final_metrics = pd.DataFrame(
        final_rows
    )

    final_metrics.to_csv(
        out_dir
        / "final_ablation_metrics.csv",
        index=False,
    )

    # Compact manuscript-oriented table.
    compact_columns = [
        "variant",
        "selected_epoch",
        "selected_gamma",
        "overall_mae_pga",
        "tail_mae_pga",
        "tail_under05_pga",
        "tail_factor2_pga",
        "overall_mae_pgv",
        "tail_mae_pgv",
        "tail_under05_pgv",
        "tail_factor2_pgv",
        "non_tail_mae_pga",
        "non_tail_mae_pgv",
        "bias_pga",
        "bias_pgv",
    ]

    compact = final_metrics[
        compact_columns
    ].copy()

    compact.to_csv(
        out_dir
        / "paper_ablation_table.csv",
        index=False,
    )

    # Exact-paired prediction file for later paired bootstrap.
    prediction_frame = pd.DataFrame(
        {
            "event_id": (
                reference_cache[
                    "event_id"
                ]
            ),
            "repeat": (
                reference_cache[
                    "repeat"
                ]
            ),
            "target_slot": (
                reference_cache[
                    "target_slot"
                ]
            ),
            "target_station_index": (
                reference_cache[
                    "target_station_index"
                ]
            ),
            "true_log10_pga": (
                reference_cache[
                    "truth"
                ][
                    :,
                    0,
                ]
            ),
            "true_log10_pgv": (
                reference_cache[
                    "truth"
                ][
                    :,
                    1,
                ]
            ),
            "is_tail_pga": (
                reference_cache[
                    "tail"
                ][
                    :,
                    0,
                ]
            ),
            "is_tail_pgv": (
                reference_cache[
                    "tail"
                ][
                    :,
                    1,
                ]
            ),
        }
    )

    for variant, prediction in (
        prediction_columns.items()
    ):
        prediction_frame[
            f"{variant}_log10_pga"
        ] = prediction[
            :,
            0,
        ]

        prediction_frame[
            f"{variant}_log10_pgv"
        ] = prediction[
            :,
            1,
        ]

    prediction_frame.to_csv(
        out_dir
        / "locked_test_ablation_predictions.csv",
        index=False,
    )

    print(
        "\n=== FINAL A2-A6 gate ablation table ==="
    )

    print(
        compact.to_string(
            index=False
        )
    )

    print(
        "\nPairing audit:"
    )

    print(
        json.dumps(
            pairing_audit,
            indent=2,
        )
    )

    print(
        "\nOutputs:"
    )

    for path in [
        out_dir
        / "paper_ablation_table.csv",
        out_dir
        / "final_ablation_metrics.csv",
        out_dir
        / "locked_test_ablation_predictions.csv",
        out_dir
        / "locked_test_pairing_audit.json",
        out_dir
        / "protocol.json",
    ]:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
