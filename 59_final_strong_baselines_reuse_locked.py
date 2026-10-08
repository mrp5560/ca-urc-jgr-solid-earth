#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
59_final_strong_baselines_reuse_locked.py

FINAL strong-baseline suite for Causal-SeisField under the locked grouped
causal protocol.

Compared methods
----------------
Strictly causal non-learned / propagation:
    median_observed
    nearest_observed
    idw_observed
    plum_like
        Adapted PLUM-like propagation reference. It uses only pre-snapshot
        observed input motion. Distance scale is selected on validation;
        additive log10 offset is fitted on training only. It is NOT claimed
        to be the original PLUM implementation.

Strong learned baselines:
    cross_attention
        Target-conditioned cross-station Transformer-style baseline.

    graph
        Geometry-aware message-passing / target-readout baseline.

    tail_weighted_attention
        Same attention-base architecture as Causal-SeisField, but replaces
        the proposed risk head + power gate with a simple tail-weighted
        regression loss. This is a direct tail-handling baseline.

Locked proposed models:
    attention_base
        REUSED directly from the previously generated locked-test prediction
        file. It is NOT retrained and NOT re-inferred in this script.

    power_gate
        REUSED directly from the same previously generated locked-test
        prediction file. It is NOT retrained and NOT re-inferred here.

Fair-comparison protocol
------------------------
1. Same grouped train/validation/test event split.
2. Same T0, K, Q and input time window.
3. TRAIN/VALIDATION station draws exactly follow script 40.
4. TEST draws follow the locked evaluator (script 20). The proposed
   attention_base and power_gate predictions are read directly from the
   previously locked script-51 output and paired target-by-target.
5. No catalog magnitude or hypocenter is used by any causal baseline here.
6. PLUM-like hyperparameters use train + validation only.
7. Learned baseline checkpoints use validation only.
8. The test split is instantiated only AFTER all baseline tuning is finished.
9. Primary aggregation:
       targets -> repeats -> events.
10. No test-dependent tuning or model selection is performed.

Privileged information reference
--------------------------------
source_oracle_ridge
    Optional empirical source reference using FINAL catalog magnitude,
    hypocenter, and target distance. It is included only as an
    information-rich reference / upper-information comparator. It is NOT
    an operational causal baseline and is NOT claimed to be a standard GMPE.
"""

from __future__ import annotations

import argparse
import hashlib
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
# Dynamic imports of already-audited project modules.
# ---------------------------------------------------------------------

def load_module(path: str | Path, name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required Python module not found: {path.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import Python module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# Exact seed protocols.
# ---------------------------------------------------------------------

def canonical_stable_seed(
    text: str,
    base_seed: int,
) -> int:
    """
    Same hash convention as scripts 40 and 20:
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


def patch_baseline_seed_protocol(
    baseline_module,
) -> None:
    """
    45 used:
        strong:<split>:<event>:<epoch/repeat>
    and a reversed hash-string order.

    For the FINAL suite we deliberately reproduce:
      train/validation -> script 40
      test             -> script 20 locked evaluator

    This lets every strong baseline use the same test station draws as the
    final locked Causal-SeisField evaluation.
    """

    def matched_seed(
        text: str,
        base_seed: int,
    ) -> int:
        text = str(text)

        prefix = "strong:"

        if text.startswith(
            "strong:test:"
        ):
            # strong:test:E:repeat:r
            # -> locked:test:E:repeat:r
            transformed = (
                "locked:"
                + text[
                    len(prefix):
                ]
            )

        elif text.startswith(
            prefix
        ):
            # strong:train:E:epoch:e
            # -> train:E:epoch:e
            # strong:validation:E:repeat:r
            # -> validation:E:repeat:r
            transformed = text[
                len(prefix):
            ]

        else:
            transformed = text

        return canonical_stable_seed(
            transformed,
            base_seed,
        )

    baseline_module.stable_seed = (
        matched_seed
    )


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


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


# ---------------------------------------------------------------------
# Tail-weighted attention baseline.
# ---------------------------------------------------------------------

def train_tail_weighted_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    thresholds: torch.Tensor,
    tail_extra_weight: float,
) -> float:

    model.train()

    total = 0.0
    count = 0

    for batch in loader:
        target = batch[
            "target_log"
        ].to(
            device,
            non_blocking=True,
        )

        prediction = model(
            batch[
                "input_waveforms"
            ].to(
                device,
                non_blocking=True,
            ),
            batch[
                "input_features"
            ].to(
                device,
                non_blocking=True,
            ),
            batch[
                "target_features"
            ].to(
                device,
                non_blocking=True,
            ),
        )

        tail_label = (
            target
            >= thresholds[
                None,
                None,
                :,
            ]
        ).to(
            target.dtype
        )

        elementwise = (
            nn.functional
            .smooth_l1_loss(
                prediction,
                target,
                reduction="none",
            )
        )

        weight = (
            1.0
            + float(
                tail_extra_weight
            )
            * tail_label
        )

        loss = (
            elementwise
            * weight
        ).mean()

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        n = int(
            target.numel()
        )

        total += (
            float(
                loss.detach().cpu()
            )
            * n
        )

        count += n

    return (
        total
        / max(
            count,
            1,
        )
    )


def fresh_train_loader(
    dataset,
    args,
    device: torch.device,
):
    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    return DataLoader(
        dataset,
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


def train_standard_variant(
    baseline_module,
    variant: str,
    args: argparse.Namespace,
    train_dataset,
    validation_loader,
    device: torch.device,
):
    """
    Train cross_attention / graph.

    A fresh train DataLoader is constructed for every model, ensuring the
    same shuffle RNG and the same epoch-conditioned station draws.
    """

    set_seed(
        args.seed
    )

    train_loader = (
        fresh_train_loader(
            train_dataset,
            args,
            device,
        )
    )

    model = baseline_module.make_model(
        variant,
        args.hidden_dim,
        args.attention_heads,
        args.graph_initial_length_scale_km,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=(
                args.minimum_learning_rate
            ),
        )
    )

    variant_dir = (
        Path(
            args.out_dir
        )
        / variant
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        variant_dir
        / "best_model.pt"
    )

    best_score = float(
        "inf"
    )

    best_epoch = 0
    stale = 0
    history = []
    start = time.time()

    print(
        f"\n=== Strong learned baseline: {variant} ==="
    )

    print(
        "Parameters               : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_loss = (
            baseline_module.train_epoch(
                model,
                train_loader,
                optimizer,
                device,
            )
        )

        validation_frame = (
            baseline_module.predict_model(
                model,
                validation_loader,
                device,
            )
        )

        val_pga = (
            baseline_module
            .event_balanced_mae(
                validation_frame,
                "pga",
                "pred_log10_pga",
            )
        )

        val_pgv = (
            baseline_module
            .event_balanced_mae(
                validation_frame,
                "pgv",
                "pred_log10_pgv",
            )
        )

        score = 0.5 * (
            val_pga
            + val_pgv
        )

        scheduler.step(
            score
        )

        lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": float(
                    train_loss
                ),
                "validation_mae_pga": float(
                    val_pga
                ),
                "validation_mae_pgv": float(
                    val_pgv
                ),
                "validation_score": float(
                    score
                ),
                "learning_rate": float(
                    lr
                ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            variant_dir
            / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val PGA={val_pga:.4f} | "
            f"PGV={val_pgv:.4f} | "
            f"score={score:.4f} | "
            f"lr={lr:.2e}"
        )

        if (
            score
            < best_score
            - args.minimum_delta
        ):
            best_score = (
                score
            )

            best_epoch = int(
                epoch
            )

            stale = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "variant": (
                        variant
                    ),
                    "epoch": int(
                        epoch
                    ),
                    "validation_score": float(
                        score
                    ),
                    "validation_mae_pga": float(
                        val_pga
                    ),
                    "validation_mae_pgv": float(
                        val_pgv
                    ),
                    "test_split_evaluated": False,
                },
                best_path,
            )

        else:
            stale += 1

        if (
            stale
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}"
            )

            break

    checkpoint = load_checkpoint(
        best_path,
        device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    model.eval()

    summary = {
        "variant": variant,
        "best_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "best_validation_score": float(
            checkpoint[
                "validation_score"
            ]
        ),
        "parameter_count": int(
            sum(
                p.numel()
                for p in model.parameters()
            )
        ),
        "training_seconds": float(
            time.time()
            - start
        ),
        "test_split_evaluated_during_training": False,
    }

    return (
        model,
        summary,
    )


def train_tail_weighted_attention(
    attention_module,
    baseline_module,
    args: argparse.Namespace,
    train_dataset,
    validation_loader,
    threshold_values: dict[str, float],
    device: torch.device,
):
    """
    Direct tail-loss baseline:
        Attention Base + weighted regression
    but NO tail classifier, NO nonnegative correction head, and NO p^gamma.
    """

    set_seed(
        args.seed
    )

    train_loader = (
        fresh_train_loader(
            train_dataset,
            args,
            device,
        )
    )

    model = (
        attention_module
        .FinalAblationModel(
            "attention_pooling",
            args.hidden_dim,
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=(
                args.minimum_learning_rate
            ),
        )
    )

    thresholds = torch.tensor(
        [
            threshold_values[
                "pga"
            ],
            threshold_values[
                "pgv"
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    variant = (
        "tail_weighted_attention"
    )

    variant_dir = (
        Path(
            args.out_dir
        )
        / variant
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        variant_dir
        / "best_model.pt"
    )

    best_score = float(
        "inf"
    )

    best_epoch = 0
    stale = 0
    history = []
    start = time.time()

    print(
        "\n=== Tail-handling baseline: tail_weighted_attention ==="
    )

    print(
        "Parameters               : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        "Tail extra loss weight   : "
        f"{args.tail_weighted_extra_weight:.3f}"
    )

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_loss = (
            train_tail_weighted_epoch(
                model,
                train_loader,
                optimizer,
                device,
                thresholds,
                args.tail_weighted_extra_weight,
            )
        )

        validation_frame = (
            baseline_module.predict_model(
                model,
                validation_loader,
                device,
            )
        )

        val_pga = (
            baseline_module
            .event_balanced_mae(
                validation_frame,
                "pga",
                "pred_log10_pga",
            )
        )

        val_pgv = (
            baseline_module
            .event_balanced_mae(
                validation_frame,
                "pgv",
                "pred_log10_pgv",
            )
        )

        score = 0.5 * (
            val_pga
            + val_pgv
        )

        scheduler.step(
            score
        )

        lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": float(
                    train_loss
                ),
                "validation_mae_pga": float(
                    val_pga
                ),
                "validation_mae_pgv": float(
                    val_pgv
                ),
                "validation_score": float(
                    score
                ),
                "learning_rate": float(
                    lr
                ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            variant_dir
            / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"weighted train={train_loss:.4f} | "
            f"val PGA={val_pga:.4f} | "
            f"PGV={val_pgv:.4f} | "
            f"score={score:.4f}"
        )

        if (
            score
            < best_score
            - args.minimum_delta
        ):
            best_score = (
                score
            )

            best_epoch = int(
                epoch
            )

            stale = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "variant": (
                        variant
                    ),
                    "epoch": int(
                        epoch
                    ),
                    "validation_score": float(
                        score
                    ),
                    "validation_mae_pga": float(
                        val_pga
                    ),
                    "validation_mae_pgv": float(
                        val_pgv
                    ),
                    "tail_extra_weight": float(
                        args.tail_weighted_extra_weight
                    ),
                    "test_split_evaluated": False,
                },
                best_path,
            )

        else:
            stale += 1

        if (
            stale
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}"
            )
            break

    checkpoint = load_checkpoint(
        best_path,
        device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    model.eval()

    summary = {
        "variant": variant,
        "best_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "best_validation_score": float(
            checkpoint[
                "validation_score"
            ]
        ),
        "parameter_count": int(
            sum(
                p.numel()
                for p in model.parameters()
            )
        ),
        "training_seconds": float(
            time.time()
            - start
        ),
        "tail_extra_weight": float(
            args.tail_weighted_extra_weight
        ),
        "test_split_evaluated_during_training": False,
    }

    return (
        model,
        summary,
    )


# ---------------------------------------------------------------------
# Locked proposed-model prediction reuse and pairing audit.
# ---------------------------------------------------------------------

def merge_locked_proposed_predictions(
    baseline_test_frame: pd.DataFrame,
    locked_prediction_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Merge the already-frozen Attention Base and Power Gate predictions
    into the strong-baseline table.

    Pairing key:
        event_id + repeat + target_station_index

    The same test station-draw seed protocol is enforced for all new
    baselines, so these three fields identify the exact locked target.
    """

    locked_path = Path(
        locked_prediction_path
    )

    if not locked_path.exists():
        raise FileNotFoundError(
            locked_path
        )

    locked = pd.read_csv(
        locked_path,
        dtype={
            "event_id": str,
        },
    )

    required = {
        "event_id",
        "repeat",
        "target_station_index",
        "true_log10_pga",
        "true_log10_pgv",
        "base_log10_pga",
        "base_log10_pgv",
        "power_gate_log10_pga",
        "power_gate_log10_pgv",
    }

    missing = required.difference(
        locked.columns
    )

    if missing:
        raise ValueError(
            "Locked prediction file missing columns: "
            f"{sorted(missing)}"
        )

    keys = [
        "event_id",
        "repeat",
        "target_station_index",
    ]

    if (
        locked.duplicated(
            keys
        ).any()
    ):
        examples = (
            locked.loc[
                locked.duplicated(
                    keys,
                    keep=False,
                ),
                keys,
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            )
        )

        raise ValueError(
            "Locked predictions are not unique on pairing key "
            f"{keys}. Examples: {examples}"
        )

    if (
        baseline_test_frame
        .duplicated(
            keys
        )
        .any()
    ):
        raise ValueError(
            "New strong-baseline test rows are not unique on "
            f"{keys}."
        )

    locked_compact = (
        locked[
            keys
            + [
                "true_log10_pga",
                "true_log10_pgv",
                "base_log10_pga",
                "base_log10_pgv",
                "power_gate_log10_pga",
                "power_gate_log10_pgv",
            ]
        ]
        .rename(
            columns={
                "true_log10_pga": (
                    "locked_true_log10_pga"
                ),
                "true_log10_pgv": (
                    "locked_true_log10_pgv"
                ),
                "base_log10_pga": (
                    "attention_base_log10_pga"
                ),
                "base_log10_pgv": (
                    "attention_base_log10_pgv"
                ),
            }
        )
    )

    merged = baseline_test_frame.merge(
        locked_compact,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    audit = {
        "locked_prediction_path": str(
            locked_path.resolve()
        ),
        "baseline_rows": int(
            len(
                baseline_test_frame
            )
        ),
        "locked_rows": int(
            len(
                locked
            )
        ),
        "paired_rows": int(
            len(
                merged
            )
        ),
        "exact_pairing": bool(
            len(
                merged
            )
            == len(
                baseline_test_frame
            )
            == len(
                locked
            )
        ),
    }

    if not audit[
        "exact_pairing"
    ]:
        baseline_keys = set(
            map(
                tuple,
                baseline_test_frame[
                    keys
                ].astype(
                    {
                        "event_id": str,
                        "repeat": int,
                        "target_station_index": int,
                    }
                ).to_numpy(),
            )
        )

        locked_keys = set(
            map(
                tuple,
                locked[
                    keys
                ].astype(
                    {
                        "event_id": str,
                        "repeat": int,
                        "target_station_index": int,
                    }
                ).to_numpy(),
            )
        )

        audit[
            "n_baseline_only_keys"
        ] = int(
            len(
                baseline_keys
                - locked_keys
            )
        )

        audit[
            "n_locked_only_keys"
        ] = int(
            len(
                locked_keys
                - baseline_keys
            )
        )

        raise RuntimeError(
            "Strong-baseline test targets do not exactly match the "
            "previous locked test targets. Audit: "
            f"{audit}"
        )

    audit[
        "max_abs_diff_truth_pga"
    ] = float(
        np.max(
            np.abs(
                merged[
                    "true_log10_pga"
                ].to_numpy(
                    dtype=float
                )
                - merged[
                    "locked_true_log10_pga"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    audit[
        "max_abs_diff_truth_pgv"
    ] = float(
        np.max(
            np.abs(
                merged[
                    "true_log10_pgv"
                ].to_numpy(
                    dtype=float
                )
                - merged[
                    "locked_true_log10_pgv"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    if (
        audit[
            "max_abs_diff_truth_pga"
        ]
        > 1e-5
        or audit[
            "max_abs_diff_truth_pgv"
        ]
        > 1e-5
    ):
        raise RuntimeError(
            "Target pairing keys match, but recomputed truths differ "
            "from the locked test. This indicates protocol drift. "
            f"Audit: {audit}"
        )

    merged = merged.drop(
        columns=[
            "locked_true_log10_pga",
            "locked_true_log10_pgv",
        ]
    )

    return (
        merged,
        audit,
    )


# ---------------------------------------------------------------------
# Privileged catalog-source information reference.
# ---------------------------------------------------------------------

def parse_float_grid(
    text: str,
) -> list[float]:
    values = []

    for item in str(
        text
    ).split(","):
        item = item.strip()

        if not item:
            continue

        value = float(
            item
        )

        if value < 0:
            raise ValueError(
                "Ridge alpha must be >= 0."
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
            "At least one ridge alpha is required."
        )

    return values


def load_source_metadata(
    manifest_path: str | Path,
    event_metadata_path: str | Path | None,
) -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load final catalog source parameters for an information-rich reference.

    Required:
        event_id, magnitude, latitude, longitude, depth_km, h5_path

    If the scenario manifest lacks some catalog columns, optionally merge
    them from --event-metadata.

    This reference is explicitly NON-CAUSAL with respect to the operational
    information budget because it uses final catalog source parameters.
    """

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "event_id": str,
        },
    )

    if (
        "event_id"
        not in manifest.columns
    ):
        raise ValueError(
            "Manifest lacks event_id."
        )

    required_metadata = [
        "magnitude",
        "latitude",
        "longitude",
        "depth_km",
    ]

    missing = [
        column
        for column
        in required_metadata
        if column
        not in manifest.columns
    ]

    source_description = (
        "scenario manifest"
    )

    metadata_path = None

    if (
        event_metadata_path is not None
        and str(
            event_metadata_path
        ).strip()
    ):
        candidate = Path(
            event_metadata_path
        )

        if candidate.exists():
            metadata_path = (
                candidate
            )

    if (
        missing
        and metadata_path
        is not None
    ):
        metadata = pd.read_csv(
            metadata_path,
            dtype={
                "event_id": str,
            },
        )

        merge_columns = [
            "event_id",
            *[
                column
                for column
                in missing
                if column
                in metadata.columns
            ],
        ]

        if (
            len(
                merge_columns
            )
            > 1
        ):
            manifest = (
                manifest.merge(
                    metadata[
                        merge_columns
                    ]
                    .drop_duplicates(
                        "event_id"
                    ),
                    on="event_id",
                    how="left",
                    validate="many_to_one",
                )
            )

            source_description = (
                "scenario manifest + "
                f"{metadata_path}"
            )

    still_missing = [
        column
        for column
        in required_metadata
        if column
        not in manifest.columns
    ]

    if still_missing:
        raise ValueError(
            "Source reference unavailable; missing catalog columns: "
            f"{still_missing}"
        )

    if (
        "h5_path"
        not in manifest.columns
    ):
        manifest[
            "h5_path"
        ] = ""

    keep = [
        "event_id",
        "h5_path",
        *required_metadata,
    ]

    return (
        manifest[
            keep
        ]
        .drop_duplicates(
            "event_id"
        )
        .reset_index(
            drop=True
        ),
        source_description,
    )


def add_source_features(
    frame: pd.DataFrame,
    source_metadata: pd.DataFrame,
    h5_root: str | Path | None,
    baseline_module,
) -> pd.DataFrame:
    """
    Add privileged final-source features for the catalog-source reference.

    Features:
        M
        M^2
        log10(R_hyp + 1 km)
        M * log10(R_hyp + 1 km)
        log10(R_hyp + 1 km)^2
    """

    metadata = (
        source_metadata.set_index(
            "event_id",
            drop=False,
        )
    )

    result = frame.copy()

    magnitude_values = np.full(
        len(
            result
        ),
        np.nan,
        dtype=float,
    )

    distance_values = np.full(
        len(
            result
        ),
        np.nan,
        dtype=float,
    )

    root = (
        Path(
            h5_root
        )
        if (
            h5_root
            is not None
            and str(
                h5_root
            ).strip()
        )
        else None
    )

    # One HDF5 read per event.
    for event_id, indices in (
        result.groupby(
            "event_id",
            sort=False,
        ).groups.items()
    ):
        event_id = str(
            event_id
        )

        if (
            event_id
            not in metadata.index
        ):
            continue

        meta = metadata.loc[
            event_id
        ]

        if isinstance(
            meta,
            pd.DataFrame,
        ):
            meta = meta.iloc[
                0
            ]

        magnitude = pd.to_numeric(
            meta[
                "magnitude"
            ],
            errors="coerce",
        )

        event_latitude = pd.to_numeric(
            meta[
                "latitude"
            ],
            errors="coerce",
        )

        event_longitude = pd.to_numeric(
            meta[
                "longitude"
            ],
            errors="coerce",
        )

        depth_km = pd.to_numeric(
            meta[
                "depth_km"
            ],
            errors="coerce",
        )

        if not (
            np.isfinite(
                magnitude
            )
            and np.isfinite(
                event_latitude
            )
            and np.isfinite(
                event_longitude
            )
            and np.isfinite(
                depth_km
            )
        ):
            continue

        h5_path = (
            baseline_module
            .resolve_h5_path(
                event_id,
                str(
                    meta.get(
                        "h5_path",
                        "",
                    )
                ),
                root,
            )
        )

        import h5py

        with h5py.File(
            h5_path,
            "r",
        ) as h5:
            coordinates = np.asarray(
                h5[
                    "station_coords"
                ][
                    :
                ],
                dtype=float,
            )

        row_indices = np.asarray(
            list(
                indices
            ),
            dtype=int,
        )

        station_indices = pd.to_numeric(
            result.loc[
                row_indices,
                "target_station_index",
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid_station = (
            np.isfinite(
                station_indices
            )
            & (
                station_indices
                >= 0
            )
            & (
                station_indices
                < len(
                    coordinates
                )
            )
        )

        if not valid_station.any():
            continue

        valid_rows = row_indices[
            valid_station
        ]

        station_indices_int = (
            station_indices[
                valid_station
            ].astype(
                int
            )
        )

        target_coordinates = (
            coordinates[
                station_indices_int
            ]
        )

        lat = target_coordinates[
            :,
            0
        ]

        lon = target_coordinates[
            :,
            1
        ]

        mean_lat = (
            0.5
            * (
                lat
                + float(
                    event_latitude
                )
            )
        )

        x = (
            (
                lon
                - float(
                    event_longitude
                )
            )
            * 111.32
            * np.cos(
                np.deg2rad(
                    mean_lat
                )
            )
        )

        y = (
            (
                lat
                - float(
                    event_latitude
                )
            )
            * 110.57
        )

        epicentral = np.sqrt(
            x ** 2
            + y ** 2
        )

        hypocentral = np.sqrt(
            epicentral ** 2
            + float(
                depth_km
            ) ** 2
        )

        magnitude_values[
            valid_rows
        ] = float(
            magnitude
        )

        distance_values[
            valid_rows
        ] = (
            hypocentral
        )

    result[
        "source_magnitude"
    ] = magnitude_values

    result[
        "source_hypocentral_distance_km"
    ] = distance_values

    log_r = np.log10(
        np.maximum(
            distance_values,
            0.0,
        )
        + 1.0
    )

    result[
        "source_feature_m"
    ] = magnitude_values

    result[
        "source_feature_m2"
    ] = magnitude_values ** 2

    result[
        "source_feature_logr"
    ] = log_r

    result[
        "source_feature_m_logr"
    ] = (
        magnitude_values
        * log_r
    )

    result[
        "source_feature_logr2"
    ] = log_r ** 2

    return result


SOURCE_FEATURE_COLUMNS = [
    "source_feature_m",
    "source_feature_m2",
    "source_feature_logr",
    "source_feature_m_logr",
    "source_feature_logr2",
]


def source_design_matrix(
    frame: pd.DataFrame,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    raw = frame[
        SOURCE_FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    standardized = (
        raw
        - means[
            None,
            :,
        ]
    ) / scales[
        None,
        :,
    ]

    return np.concatenate(
        [
            np.ones(
                (
                    len(
                        standardized
                    ),
                    1,
                ),
                dtype=float,
            ),
            standardized,
        ],
        axis=1,
    )


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    penalty = np.eye(
        x.shape[
            1
        ],
        dtype=float,
    )

    # Do not penalize the intercept.
    penalty[
        0,
        0,
    ] = 0.0

    matrix = (
        x.T
        @ x
        + float(
            alpha
        )
        * penalty
    )

    rhs = (
        x.T
        @ y
    )

    try:
        return np.linalg.solve(
            matrix,
            rhs,
        )
    except np.linalg.LinAlgError:
        return np.linalg.pinv(
            matrix
        ) @ rhs


def fit_source_oracle(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    alphas: list[float],
    baseline_module,
) -> dict[str, Any]:
    """
    Fit an information-rich empirical source reference.

    It is NOT a GMPE and NOT an operational causal baseline.
    It deliberately uses final catalog M and hypocenter as a privileged
    information reference.
    """

    train_valid = (
        np.isfinite(
            train_frame[
                SOURCE_FEATURE_COLUMNS
            ].to_numpy(
                dtype=float
            )
        ).all(
            axis=1
        )
    )

    validation_valid = (
        np.isfinite(
            validation_frame[
                SOURCE_FEATURE_COLUMNS
            ].to_numpy(
                dtype=float
            )
        ).all(
            axis=1
        )
    )

    if (
        not train_valid.all()
        or not validation_valid.all()
    ):
        raise ValueError(
            "Non-finite source features remain in train/validation "
            "source reference rows."
        )

    raw_train = train_frame[
        SOURCE_FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    means = raw_train.mean(
        axis=0
    )

    scales = raw_train.std(
        axis=0
    )

    scales[
        scales
        < 1e-8
    ] = 1.0

    x_train = (
        source_design_matrix(
            train_frame,
            means,
            scales,
        )
    )

    x_validation = (
        source_design_matrix(
            validation_frame,
            means,
            scales,
        )
    )

    configuration = {
        "reference_type": (
            "privileged catalog-source empirical ridge reference"
        ),
        "operational_causal_baseline": False,
        "uses_final_catalog_magnitude": True,
        "uses_final_catalog_hypocenter": True,
        "feature_columns": (
            SOURCE_FEATURE_COLUMNS
        ),
        "feature_means": (
            means.tolist()
        ),
        "feature_scales": (
            scales.tolist()
        ),
        "quantity_models": {},
    }

    for quantity in (
        "pga",
        "pgv",
    ):
        y_train = train_frame[
            f"true_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        scores = {}
        coefficients = {}

        for alpha in alphas:
            beta = ridge_fit(
                x_train,
                y_train,
                alpha,
            )

            coefficients[
                alpha
            ] = beta

            prediction = (
                x_validation
                @ beta
            )

            temporary = (
                validation_frame[
                    [
                        "event_id",
                        "repeat",
                        f"true_log10_{quantity}",
                    ]
                ]
                .copy()
            )

            temporary[
                "candidate_prediction"
            ] = prediction

            score = (
                baseline_module
                .event_balanced_mae(
                    temporary,
                    quantity,
                    "candidate_prediction",
                )
            )

            scores[
                alpha
            ] = float(
                score
            )

        best_alpha = min(
            alphas,
            key=lambda value: (
                scores[
                    value
                ],
                value,
            ),
        )

        configuration[
            "quantity_models"
        ][
            quantity
        ] = {
            "alpha": float(
                best_alpha
            ),
            "coefficients": (
                coefficients[
                    best_alpha
                ].tolist()
            ),
            "validation_mae": float(
                scores[
                    best_alpha
                ]
            ),
            "alpha_grid": {
                str(
                    alpha
                ): float(
                    scores[
                        alpha
                    ]
                )
                for alpha in alphas
            },
        }

    return configuration


def apply_source_oracle(
    frame: pd.DataFrame,
    configuration: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()

    means = np.asarray(
        configuration[
            "feature_means"
        ],
        dtype=float,
    )

    scales = np.asarray(
        configuration[
            "feature_scales"
        ],
        dtype=float,
    )

    x = source_design_matrix(
        result,
        means,
        scales,
    )

    for quantity in (
        "pga",
        "pgv",
    ):
        beta = np.asarray(
            configuration[
                "quantity_models"
            ][
                quantity
            ][
                "coefficients"
            ],
            dtype=float,
        )

        result[
            f"source_oracle_ridge_log10_{quantity}"
        ] = (
            x
            @ beta
        )

    return result



# ---------------------------------------------------------------------
# Main.
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
        "--attention-module",
        default=(
            "40_phase1_final_mean_pooling_ablation_suite.py"
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
        "--threshold-json",
        required=True,
    )

    parser.add_argument(
        "--variants",
        default=(
            "cross_attention,"
            "graph,"
            "tail_weighted_attention"
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
        "--validation-repeats",
        type=int,
        default=3,
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
        "--graph-initial-length-scale-km",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--tail-weighted-extra-weight",
        type=float,
        default=2.0,
        help=(
            "Weight multiplier in "
            "1 + w * I(high-motion-tail)."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--minimum-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--plum-length-scales-km",
        default=(
            "20,40,60,80,120,200,inf"
        ),
    )

    parser.add_argument(
        "--locked-predictions",
        default=(
            "runs/"
            "attention_locked_power_gate_gamma7/"
            "locked_test_predictions.csv"
        ),
        help=(
            "Previously frozen script-51 locked_test_predictions.csv. "
            "Attention Base and Power Gate are reused directly from it."
        ),
    )

    parser.add_argument(
        "--event-metadata",
        default="",
        help=(
            "Optional event metadata CSV used only for the privileged "
            "catalog-source reference if magnitude/latitude/longitude/"
            "depth_km are absent from the scenario manifest."
        ),
    )

    parser.add_argument(
        "--source-ridge-alphas",
        default="0,0.001,0.01,0.1,1,10,100",
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
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "final_strong_baselines_attention"
        ),
    )

    args = parser.parse_args()

    allowed_variants = {
        "cross_attention",
        "graph",
        "tail_weighted_attention",
    }

    variants = [
        x.strip()
        for x in (
            args.variants.split(
                ","
            )
        )
        if x.strip()
    ]

    unknown = set(
        variants
    ).difference(
        allowed_variants
    )

    if unknown:
        raise ValueError(
            f"Unknown variants: {sorted(unknown)}"
        )

    if (
        args.hidden_dim
        % args.attention_heads
        != 0
    ):
        raise ValueError(
            "hidden_dim must be divisible by attention_heads."
        )

    if (
        args.tail_weighted_extra_weight
        < 0
    ):
        raise ValueError(
            "--tail-weighted-extra-weight must be >= 0."
        )

    scales = []

    for text in (
        args.plum_length_scales_km
        .split(
            ","
        )
    ):
        text = text.strip()

        if not text:
            continue

        if (
            text.lower()
            == "inf"
        ):
            scales.append(
                float(
                    "inf"
                )
            )
        else:
            value = float(
                text
            )

            if value <= 0:
                raise ValueError(
                    "PLUM-like scales must be positive."
                )

            scales.append(
                value
            )

    source_alphas = parse_float_grid(
        args.source_ridge_alphas
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

    baseline_module = (
        load_module(
            args.baseline_module,
            "legacy_strong_baseline_module",
        )
    )

    attention_module = (
        load_module(
            args.attention_module,
            "final_attention_module",
        )
    )

    # Critical fairness patch.
    patch_baseline_seed_protocol(
        baseline_module
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
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

    threshold_json = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds = {
        "pga": float(
            threshold_json[
                "log10_pga_threshold"
            ]
        ),
        "pgv": float(
            threshold_json[
                "log10_pgv_threshold"
            ]
        ),
    }

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

    # -------------------------------------------------------------
    # TRAIN + VALIDATION ONLY.
    # -------------------------------------------------------------

    train_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name="train",
            training=True,
            repeats=1,
            **common_dataset,
        )
    )

    train_reference_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name="train",
            training=False,
            repeats=1,
            **common_dataset,
        )
    )

    validation_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name="validation",
            training=False,
            repeats=args.validation_repeats,
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

    train_reference_loader = DataLoader(
        train_reference_dataset,
        shuffle=False,
        **eval_loader_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **eval_loader_kwargs,
    )

    print(
        "=== FINAL strong-baseline suite / tuning stage ==="
    )

    print(
        f"Device                    : {device}"
    )

    print(
        "Train/Validation events   : "
        f"{len(train_dataset.frame)}/"
        f"{len(validation_dataset.frame)}"
    )

    print(
        f"Validation repeats        : {args.validation_repeats}"
    )

    print(
        "Train/Val draw protocol   : script-40 matched"
    )

    print(
        "Test draw protocol        : script-20 locked (not instantiated yet)"
    )

    print(
        "Tail thresholds           : "
        f"PGA={thresholds['pga']:.4f}, "
        f"PGV={thresholds['pgv']:.4f}"
    )

    print(
        "Test split during tuning  : NOT ACCESSED"
    )

    print(
        "Proposed model training   : REUSED; NOT RETRAINED"
    )

    print(
        "Proposed model inference  : REUSED; NOT RE-RUN"
    )

    # -------------------------------------------------------------
    # PLUM-like: train offset + validation length scale.
    # -------------------------------------------------------------

    print(
        "\nCollecting TRAIN/VALIDATION causal propagation rows..."
    )

    train_baseline = (
        baseline_module
        .collect_causal_baselines(
            train_reference_loader,
            args.idw_power,
        )
    )

    validation_baseline = (
        baseline_module
        .collect_causal_baselines(
            validation_loader,
            args.idw_power,
        )
    )

    plum_configuration = (
        baseline_module
        .fit_plum(
            train_baseline,
            validation_baseline,
            scales,
        )
    )

    (
        out_dir
        / "plum_like_configuration.json"
    ).write_text(
        json.dumps(
            plum_configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Privileged catalog-source information reference.
    # This is kept separate conceptually from strict-causal baselines.
    # -------------------------------------------------------------

    source_oracle_available = False
    source_configuration = None
    source_metadata_description = None
    train_source = None
    validation_source = None

    try:
        (
            source_metadata,
            source_metadata_description,
        ) = load_source_metadata(
            args.manifest,
            args.event_metadata,
        )

        train_source = add_source_features(
            train_baseline,
            source_metadata,
            args.h5_root,
            baseline_module,
        )

        validation_source = add_source_features(
            validation_baseline,
            source_metadata,
            args.h5_root,
            baseline_module,
        )

        source_configuration = fit_source_oracle(
            train_source,
            validation_source,
            source_alphas,
            baseline_module,
        )

        source_configuration[
            "metadata_source"
        ] = source_metadata_description

        (
            out_dir
            / "source_oracle_ridge_configuration.json"
        ).write_text(
            json.dumps(
                source_configuration,
                indent=2,
            ),
            encoding="utf-8",
        )

        source_oracle_available = True

        print(
            "\nCatalog-source information reference fitted "
            "using TRAIN + VALIDATION only."
        )

    except Exception as exc:
        print(
            "\n[Warning] Catalog-source information reference "
            f"was skipped: {exc}"
        )

    # -------------------------------------------------------------
    # Learned strong baselines: validation-only checkpoint selection.
    # -------------------------------------------------------------

    trained_models = {}
    learned_summaries = []

    for variant in variants:

        if (
            variant
            == "tail_weighted_attention"
        ):
            (
                model,
                summary,
            ) = train_tail_weighted_attention(
                attention_module,
                baseline_module,
                args,
                train_dataset,
                validation_loader,
                thresholds,
                device,
            )

        else:
            (
                model,
                summary,
            ) = train_standard_variant(
                baseline_module,
                variant,
                args,
                train_dataset,
                validation_loader,
                device,
            )

        trained_models[
            variant
        ] = model

        learned_summaries.append(
            summary
        )

    pd.DataFrame(
        learned_summaries
    ).to_csv(
        out_dir
        / "learned_training_summary.csv",
        index=False,
    )

    print(
        "\n=== ALL BASELINE TUNING FINISHED ==="
    )

    print(
        "Test set has not been used for PLUM selection "
        "or learned checkpoint selection."
    )

    # -------------------------------------------------------------
    # NOW instantiate the locked test split.
    # -------------------------------------------------------------

    test_dataset = (
        baseline_module
        .StrongBaselineDataset(
            split_name="test",
            training=False,
            repeats=args.test_repeats,
            **common_dataset,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **eval_loader_kwargs,
    )

    print(
        "\n=== FINAL matched locked-test stage ==="
    )

    print(
        f"Test events               : {len(test_dataset.frame)}"
    )

    print(
        f"Test repeats/event        : {args.test_repeats}"
    )

    print(
        "Test draw protocol        : script-20 locked"
    )

    # -------------------------------------------------------------
    # Non-learned causal baselines.
    # -------------------------------------------------------------

    print(
        "\nCollecting locked-test causal propagation rows..."
    )

    test_combined = (
        baseline_module
        .collect_causal_baselines(
            test_loader,
            args.idw_power,
        )
    )

    test_combined = (
        baseline_module
        .apply_plum(
            test_combined,
            plum_configuration,
        )
    )

    # -------------------------------------------------------------
    # Reuse the PREVIOUSLY LOCKED proposed-model predictions.
    # No retraining and no re-inference of our model is performed here.
    # -------------------------------------------------------------

    (
        test_combined,
        locked_pairing_audit,
    ) = merge_locked_proposed_predictions(
        test_combined,
        args.locked_predictions,
    )

    key_columns = [
        "event_id",
        "repeat",
        "target_slot",
        "target_station_index",
    ]

    # The locked merge used event/repeat/station index because script 20
    # does not need target_slot as a scientific identity field. target_slot
    # remains available from the newly generated baseline table for merging
    # the new learned baselines.

    if source_oracle_available:
        test_source = add_source_features(
            test_combined,
            source_metadata,
            args.h5_root,
            baseline_module,
        )

        test_source = apply_source_oracle(
            test_source,
            source_configuration,
        )

        source_columns = (
            test_source[
                key_columns
                + [
                    "source_oracle_ridge_log10_pga",
                    "source_oracle_ridge_log10_pgv",
                ]
            ]
        )

        test_combined = test_combined.merge(
            source_columns,
            on=key_columns,
            how="left",
            validate="one_to_one",
        )

    # -------------------------------------------------------------
    # Learned baselines on the SAME locked test draws.
    # -------------------------------------------------------------

    for variant in variants:
        model = trained_models[
            variant
        ]

        prediction = (
            baseline_module
            .predict_model(
                model,
                test_loader,
                device,
            )
        )

        variant_dir = (
            out_dir
            / variant
        )

        prediction.to_csv(
            variant_dir
            / "repeated_locked_test_predictions.csv",
            index=False,
        )

        renamed = (
            prediction[
                key_columns
                + [
                    "pred_log10_pga",
                    "pred_log10_pgv",
                ]
            ]
            .rename(
                columns={
                    "pred_log10_pga": (
                        f"{variant}_log10_pga"
                    ),
                    "pred_log10_pgv": (
                        f"{variant}_log10_pgv"
                    ),
                }
            )
        )

        test_combined = (
            test_combined.merge(
                renamed,
                on=key_columns,
                how="left",
                validate="one_to_one",
            )
        )

        if (
            test_combined[
                f"{variant}_log10_pga"
            ].isna().any()
        ):
            raise RuntimeError(
                f"Pairing failure for {variant}."
            )

    # -------------------------------------------------------------
    # Compact final prediction file.
    # -------------------------------------------------------------

    final_predictions = (
        test_combined.drop(
            columns=[
                "observed_log10_pga_vector",
                "observed_log10_pgv_vector",
                "distance_km_vector",
            ]
        )
    )

    prediction_path = (
        out_dir
        / "final_strong_baseline_reused_locked_predictions.csv"
    )

    final_predictions.to_csv(
        prediction_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Canonical event-balanced metrics.
    # -------------------------------------------------------------

    methods = [
        "median_observed",
        "nearest_observed",
        "idw_observed",
        "plum_like",
    ]

    if source_oracle_available:
        methods.append(
            "source_oracle_ridge"
        )

    methods.extend(
        [
            *variants,
            "attention_base",
            "power_gate",
        ]
    )

    metrics = (
        baseline_module
        .build_metrics(
            final_predictions,
            methods,
            thresholds,
        )
    )

    metrics_path = (
        out_dir
        / "final_strong_baseline_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    paired_vs_base = (
        baseline_module
        .paired_event_deltas(
            final_predictions,
            methods,
            reference=(
                "attention_base"
            ),
            thresholds=(
                thresholds
            ),
        )
    )

    paired_base_path = (
        out_dir
        / "paired_deltas_vs_attention_base.csv"
    )

    paired_vs_base.to_csv(
        paired_base_path,
        index=False,
    )

    paired_vs_power = (
        baseline_module
        .paired_event_deltas(
            final_predictions,
            methods,
            reference=(
                "power_gate"
            ),
            thresholds=(
                thresholds
            ),
        )
    )

    paired_power_path = (
        out_dir
        / "paired_deltas_vs_power_gate.csv"
    )

    paired_vs_power.to_csv(
        paired_power_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Save locked target-pairing audit.
    # -------------------------------------------------------------

    audit = locked_pairing_audit

    audit_path = (
        out_dir
        / "locked_prediction_reuse_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Metadata.
    # -------------------------------------------------------------

    (
        out_dir
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
                "methods": methods,
                "learned_baseline_variants": variants,
                "locked_proposed_predictions": str(
                    Path(
                        args.locked_predictions
                    ).resolve()
                ),
                "proposed_model_retrained": False,
                "proposed_model_reinferred": False,
                "metric_aggregation": (
                    "targets -> repeats -> events"
                ),
                "train_validation_draw_protocol": (
                    "script 40"
                ),
                "test_draw_protocol": (
                    "script 20 locked evaluator"
                ),
                "test_used_for_tuning": False,
                "plum_like_note": (
                    "Adapted strictly causal propagation baseline; "
                    "not exact PLUM."
                ),
                "source_oracle_available": bool(
                    source_oracle_available
                ),
                "source_oracle_note": (
                    "source_oracle_ridge is an information-rich "
                    "reference using final catalog magnitude and "
                    "hypocenter; it is NOT an operational causal baseline "
                    "and is NOT claimed to be a standard GMPE."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------

    print(
        "\n=== Locked prediction reuse / pairing audit ==="
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print(
        "\n=== FINAL strong-baseline overall MAE ==="
    )

    overall_display = metrics.loc[
        (
            metrics[
                "population"
            ]
            == "overall"
        )
        & (
            metrics[
                "metric"
            ]
            == "mae"
        )
    ][
        [
            "method",
            "quantity",
            "value",
            "n_events",
            "n_target_rows",
        ]
    ].copy()

    print(
        overall_display.to_string(
            index=False
        )
    )

    print(
        "\n=== FINAL strong-baseline high-motion-tail MAE ==="
    )

    tail_display = metrics.loc[
        (
            metrics[
                "population"
            ]
            == "high_motion_tail"
        )
        & (
            metrics[
                "metric"
            ]
            == "mae"
        )
    ][
        [
            "method",
            "quantity",
            "value",
            "n_events",
            "n_target_rows",
        ]
    ].copy()

    print(
        tail_display.to_string(
            index=False
        )
    )

    print(
        "\n=== Strong learned baselines minus Attention Base: overall MAE ==="
    )

    candidate_names = set(
        variants
        + [
            "plum_like",
        ]
    )

    display_delta = (
        paired_vs_base.loc[
            (
                paired_vs_base[
                    "population"
                ]
                == "overall"
            )
            & (
                paired_vs_base[
                    "metric"
                ]
                == "mae"
            )
            & (
                paired_vs_base[
                    "candidate_method"
                ].isin(
                    candidate_names
                )
            )
        ]
    )

    print(
        display_delta.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    output_paths = [
        prediction_path,
        metrics_path,
        paired_base_path,
        paired_power_path,
        out_dir
        / "learned_training_summary.csv",
        out_dir
        / "plum_like_configuration.json",
        audit_path,
        out_dir
        / "run_configuration.json",
    ]

    if source_oracle_available:
        output_paths.append(
            out_dir
            / "source_oracle_ridge_configuration.json"
        )

    for path in output_paths:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
