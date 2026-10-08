#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
65_train_and_evaluate_cadrg_station_ood.py

Strict station-level OOD experiment for the final CA-DRG model.

Model:
    Cross-Attention Base
    + Dual-Risk Gate
      y_final = y_base + p_under * p_tail**gamma * Delta

Station-OOD protocol
--------------------
The deterministic station assignment produced by script 42 is REUSED.

Training / validation:
    - OOD station IDs are excluded from BOTH inputs and targets.
    - Inputs come only from P-wave-reached seen stations.
    - Targets come only from remaining seen stations.

Test:
    - Same event and same seen input stations are used for both roles.
    - 3 seen targets + 3 unseen-station targets per event/repeat by default.
    - 20 repeats/event by default.
    - The target draw protocol exactly follows script 52.

New-model training protocol
---------------------------
1. Retrain Cross-Attention from scratch under strict seen-only station OOD.
2. Select the Cross-Attention checkpoint using VALIDATION only.
3. Compute Q90 PGA/PGV thresholds from TRAIN seen-only target draws only.
4. Freeze the selected Cross-Attention Base.
5. Train Dual-Risk head using TRAIN seen-only targets only.
6. Jointly select head epoch + gamma on 20-repeat VALIDATION only with
   the same preservation constraints as the grouped CA-DRG experiment.
7. If gamma is not on the upper grid boundary, evaluate paired seen/unseen
   targets ONCE on the station-OOD test set.

Canonical aggregation:
    targets -> repeats -> events

Required local modules
----------------------
42_phase2_prepare_station_holdout.py     (outputs reused, not imported)
45_phase2_strong_baseline_suite.py       (Cross-Attention architecture)
52_phase2_train_and_evaluate_station_ood_attention.py
60_train_cross_attention_dual_risk_joint_selection.py

Important
---------
Do NOT apply the grouped CA-DRG checkpoint directly to this experiment.
For strict station OOD, both the Cross-Attention backbone and the Dual-Risk
head must be retrained without any OOD station appearing in train/validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(
    text: str,
) -> list[float]:
    values = []

    for token in str(text).split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

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


def as_bool_array(
    series: pd.Series,
) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
        "y",
        "n",
        "t",
        "f",
    }

    unknown = set(
        normalized.unique()
    ).difference(
        allowed
    )

    if unknown:
        raise ValueError(
            "Cannot parse boolean values: "
            f"{sorted(unknown)[:20]}"
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


# ---------------------------------------------------------------------
# Resolve station-OOD scenario HDF5 paths for the local machine
# ---------------------------------------------------------------------

def resolve_scenario_for_local_machine(
    scenario_path: Path,
    h5_root: Path | None,
    output_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        scenario_path,
        dtype={
            "event_id": str,
        },
    )

    required = {
        "event_id",
        "h5_path",
        "split_grouped",
        "trainval_seen_only_eligible",
        "test_paired_seen_ood_eligible",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Station-OOD scenario missing columns: "
            f"{sorted(missing)}"
        )

    resolved = []
    unresolved = []

    for row in frame.itertuples(
        index=False
    ):
        event_id = str(
            row.event_id
        ).strip()

        raw = str(
            getattr(
                row,
                "h5_path",
                "",
            )
        ).strip()

        path = Path(raw) if raw else None

        if (
            path is not None
            and path.exists()
        ):
            resolved.append(
                str(path.resolve())
            )
            continue

        candidate = (
            h5_root
            / f"{event_id}.h5"
            if h5_root is not None
            else None
        )

        if (
            candidate is not None
            and candidate.exists()
        ):
            resolved.append(
                str(candidate.resolve())
            )
            continue

        unresolved.append(
            {
                "event_id": event_id,
                "manifest_h5_path": raw,
            }
        )
        resolved.append(raw)

    if unresolved:
        unresolved_path = (
            output_path.parent
            / "unresolved_h5_paths.csv"
        )

        pd.DataFrame(
            unresolved
        ).to_csv(
            unresolved_path,
            index=False,
        )

        raise FileNotFoundError(
            f"Could not resolve {len(unresolved)} event HDF5 paths. "
            f"See {unresolved_path}"
        )

    frame[
        "h5_path"
    ] = resolved

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    return frame


# ---------------------------------------------------------------------
# Cross-Attention Base
# ---------------------------------------------------------------------

def make_cross_attention(
    baseline_module,
    hidden_dim: int,
    attention_heads: int,
    device: torch.device,
) -> nn.Module:
    return baseline_module.make_model(
        "cross_attention",
        hidden_dim,
        attention_heads,
        50.0,
    ).to(
        device
    )


def train_station_ood_cross_attention_base(
    args,
    baseline_module,
    ood_module,
    scenario_path: Path,
    assignment_path: Path,
    device: torch.device,
    out_dir: Path,
) -> tuple[
    nn.Module,
    dict[str, Any],
]:
    train_dataset = ood_module.SeenOnlyDataset(
        scenario_csv=scenario_path,
        assignment_csv=assignment_path,
        split_column=args.split_column,
        split_name=args.train_label,
        eligibility_column=(
            "trainval_seen_only_eligible"
        ),
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=(
            args.train_target_stations
        ),
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=True,
    )

    validation_dataset = (
        ood_module.SeenOnlyDataset(
            scenario_csv=scenario_path,
            assignment_csv=assignment_path,
            split_column=args.split_column,
            split_name=args.validation_label,
            eligibility_column=(
                "trainval_seen_only_eligible"
            ),
            t0_sec=args.t0_sec,
            input_stations=args.input_stations,
            target_stations=(
                args.train_target_stations
            ),
            input_pre_sec=args.input_pre_sec,
            seed=args.seed,
            training=False,
            repeats=(
                args.base_validation_repeats
            ),
        )
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    loader_kwargs = {
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

    model = make_cross_attention(
        baseline_module,
        args.hidden_dim,
        args.attention_heads,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_learning_rate,
        weight_decay=(
            args.base_weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=(
                args.base_lr_patience
            ),
            min_lr=(
                args.base_minimum_learning_rate
            ),
        )
    )

    base_dir = (
        out_dir
        / "cross_attention_base"
    )

    base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        base_dir
        / "best_model.pt"
    )

    history = []
    best_score = np.inf
    best_epoch = 0
    stale = 0

    print(
        "\n=== Step 4A: strict station-OOD "
        "Cross-Attention Base training ==="
    )

    print(
        f"Train events              : "
        f"{len(train_dataset.frame)}"
    )

    print(
        f"Validation events         : "
        f"{len(validation_dataset.frame)}"
    )

    print(
        f"Validation repeats        : "
        f"{args.base_validation_repeats}"
    )

    print(
        f"Cross-Attention params    : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(
        1,
        args.base_epochs
        + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_loss = (
            ood_module.run_training_epoch(
                model,
                train_loader,
                optimizer,
                device,
            )
        )

        validation_frame = (
            ood_module.validation_predictions(
                model,
                validation_loader,
                device,
            )
        )

        val_pga = (
            ood_module.canonical_mae(
                validation_frame,
                "pga",
            )
        )

        val_pgv = (
            ood_module.canonical_mae(
                validation_frame,
                "pgv",
            )
        )

        val_score = 0.5 * (
            val_pga
            + val_pgv
        )

        scheduler.step(
            val_score
        )

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_loss
                ),
                "validation_mae_pga": (
                    val_pga
                ),
                "validation_mae_pgv": (
                    val_pgv
                ),
                "validation_selection_score": (
                    val_score
                ),
                "learning_rate": (
                    current_lr
                ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            base_dir
            / "history.csv",
            index=False,
        )

        print(
            f"Base epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val PGA={val_pga:.4f} | "
            f"PGV={val_pgv:.4f} | "
            f"score={val_score:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if (
            val_score
            < best_score
            - args.base_minimum_delta
        ):
            best_score = (
                val_score
            )

            best_epoch = (
                epoch
            )

            stale = 0

            torch.save(
                {
                    "variant": (
                        "cross_attention"
                    ),
                    "model_state": (
                        model.state_dict()
                    ),
                    "epoch": (
                        epoch
                    ),
                    "validation_mae_pga": (
                        val_pga
                    ),
                    "validation_mae_pgv": (
                        val_pgv
                    ),
                    "validation_score": (
                        val_score
                    ),
                    "station_ood_training": (
                        True
                    ),
                    "architecture": (
                        "cross_attention"
                    ),
                    "args": vars(
                        args
                    ),
                },
                checkpoint_path,
            )

        else:
            stale += 1

        if (
            stale
            >= args.base_early_stopping_patience
        ):
            print(
                "Base early stopping at "
                f"epoch {epoch}; "
                f"best epoch={best_epoch}"
            )

            break

    checkpoint = load_checkpoint(
        checkpoint_path,
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
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_score": float(
            best_score
        ),
        "best_validation_mae_pga": float(
            checkpoint[
                "validation_mae_pga"
            ]
        ),
        "best_validation_mae_pgv": float(
            checkpoint[
                "validation_mae_pgv"
            ]
        ),
        "train_events": int(
            len(
                train_dataset.frame
            )
        ),
        "validation_events": int(
            len(
                validation_dataset.frame
            )
        ),
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
    }

    (
        base_dir
        / "base_selection_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSelected station-OOD "
        "Cross-Attention Base:"
    )

    print(
        f"  epoch={best_epoch}"
    )

    print(
        "  val PGA/PGV="
        f"{summary['best_validation_mae_pga']:.4f}/"
        f"{summary['best_validation_mae_pgv']:.4f}"
    )

    return (
        model,
        summary,
    )


# ---------------------------------------------------------------------
# TRAIN-only thresholds and frozen-base caches
# ---------------------------------------------------------------------

def make_seen_only_dataset(
    ood_module,
    scenario_path: Path,
    assignment_path: Path,
    args,
    split_name: str,
    training: bool,
    repeats: int,
):
    return ood_module.SeenOnlyDataset(
        scenario_csv=scenario_path,
        assignment_csv=assignment_path,
        split_column=args.split_column,
        split_name=split_name,
        eligibility_column=(
            "trainval_seen_only_eligible"
        ),
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=(
            args.train_target_stations
        ),
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=training,
        repeats=(
            repeats
        ),
    )


def estimate_seen_only_train_thresholds(
    ood_module,
    scenario_path: Path,
    assignment_path: Path,
    args,
    device: torch.device,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    dataset = make_seen_only_dataset(
        ood_module,
        scenario_path,
        assignment_path,
        args,
        split_name=(
            args.train_label
        ),
        training=False,
        repeats=(
            args.threshold_repeats
        ),
    )

    loader = DataLoader(
        dataset,
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

    values = []

    for batch in loader:
        values.append(
            batch[
                "target_log"
            ]
            .numpy()
            .reshape(
                -1,
                2,
            )
        )

    targets = np.concatenate(
        values,
        axis=0,
    )

    thresholds = np.quantile(
        targets,
        args.tail_quantile,
        axis=0,
    ).astype(
        np.float32
    )

    info = {
        "tail_quantile": float(
            args.tail_quantile
        ),
        "threshold_repeats": int(
            args.threshold_repeats
        ),
        "n_train_target_rows": int(
            len(
                targets
            )
        ),
        "log10_pga_threshold": float(
            thresholds[
                0
            ]
        ),
        "log10_pgv_threshold": float(
            thresholds[
                1
            ]
        ),
        "pga_threshold_mps2": float(
            10.0
            ** thresholds[
                0
            ]
        ),
        "pgv_threshold_mps": float(
            10.0
            ** thresholds[
                1
            ]
        ),
        "threshold_source": (
            "station-OOD TRAIN seen-only target draws"
        ),
        "ood_station_labels_used": False,
        "validation_labels_used": False,
        "test_labels_used": False,
    }

    return (
        thresholds,
        info,
    )


def build_frozen_base_cache(
    base: nn.Module,
    loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
    risk_module,
) -> dict[str, Any]:
    risk_feature_parts = []
    base_parts = []
    truth_parts = []

    event_ids = []
    repeats = []

    base.eval()

    with torch.inference_mode():
        for batch in loader:
            truth = (
                batch[
                    "target_log"
                ]
                .numpy()
            )

            (
                base_prediction,
                risk_feature,
            ) = (
                risk_module
                .cross_attention_base_context(
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

            batch_repeats = (
                batch[
                    "repeat"
                ]
                .numpy()
            )

            for b in range(
                truth.shape[
                    0
                ]
            ):
                for _ in range(
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

    risk_features = np.concatenate(
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

    base_predictions = np.concatenate(
        base_parts,
        axis=0,
    ).reshape(
        -1,
        2,
    )

    truths = np.concatenate(
        truth_parts,
        axis=0,
    ).reshape(
        -1,
        2,
    )

    tail = (
        truths
        >= thresholds[
            None,
            :,
        ]
    )

    under = (
        (
            base_predictions
            - truths
        )
        <= SEVERE_UNDER_THRESHOLD
    )

    return {
        "risk_feature": (
            risk_features.astype(
                np.float32
            )
        ),
        "base": (
            base_predictions.astype(
                np.float64
            )
        ),
        "truth": (
            truths.astype(
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
    }


# ---------------------------------------------------------------------
# Dual-Risk head training / validation-only selection
# ---------------------------------------------------------------------

def train_and_select_station_ood_dual_risk(
    args,
    ood_module,
    risk_module,
    base: nn.Module,
    scenario_path: Path,
    assignment_path: Path,
    thresholds: np.ndarray,
    device: torch.device,
    out_dir: Path,
) -> tuple[
    nn.Module | None,
    dict[str, Any],
]:
    head_dir = (
        out_dir
        / "dual_risk_head"
    )

    checkpoint_dir = (
        head_dir
        / "epoch_checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # TRAIN-only deterministic reference cache for class priors.
    prevalence_dataset = (
        make_seen_only_dataset(
            ood_module,
            scenario_path,
            assignment_path,
            args,
            split_name=(
                args.train_label
            ),
            training=False,
            repeats=(
                args.prevalence_repeats
            ),
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

    prevalence_cache = (
        build_frozen_base_cache(
            base,
            prevalence_loader,
            thresholds,
            device,
            risk_module,
        )
    )

    tail_prevalence = (
        prevalence_cache[
            "tail"
        ].mean(
            axis=0
        )
    )

    under_prevalence = (
        prevalence_cache[
            "under"
        ].mean(
            axis=0
        )
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

    model = (
        risk_module
        .CrossAttentionDualRiskGate(
            base=base,
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
                tail_prevalence
            ),
            under_prevalence=(
                under_prevalence
            ),
        )
        .to(
            device
        )
    )

    train_dataset = make_seen_only_dataset(
        ood_module,
        scenario_path,
        assignment_path,
        args,
        split_name=(
            args.train_label
        ),
        training=True,
        repeats=1,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(
        args.seed
        + 777
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        generator=(
            train_generator
        ),
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

    monitor_dataset = make_seen_only_dataset(
        ood_module,
        scenario_path,
        assignment_path,
        args,
        split_name=(
            args.validation_label
        ),
        training=False,
        repeats=(
            args.head_monitor_repeats
        ),
    )

    monitor_loader = DataLoader(
        monitor_dataset,
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

    monitor_cache = (
        build_frozen_base_cache(
            base,
            monitor_loader,
            thresholds,
            device,
            risk_module,
        )
    )

    selection_dataset = (
        make_seen_only_dataset(
            ood_module,
            scenario_path,
            assignment_path,
            args,
            split_name=(
                args.validation_label
            ),
            training=False,
            repeats=(
                args.head_selection_repeats
            ),
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

    selection_cache = (
        build_frozen_base_cache(
            base,
            selection_loader,
            thresholds,
            device,
            risk_module,
        )
    )

    base_validation_metrics = (
        risk_module.metric_bundle(
            selection_cache,
            selection_cache[
                "base"
            ],
        )
    )

    trainable = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.head_learning_rate,
        weight_decay=(
            args.head_weight_decay
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

    threshold_tensor = torch.as_tensor(
        thresholds,
        dtype=torch.float32,
        device=device,
    )

    history = []

    print(
        "\n=== Step 4B: station-OOD "
        "Dual-Risk head training ==="
    )

    print(
        "TRAIN tail prevalence PGA/PGV : "
        f"{tail_prevalence[0]:.4f}/"
        f"{tail_prevalence[1]:.4f}"
    )

    print(
        "TRAIN U0.5 prevalence PGA/PGV : "
        f"{under_prevalence[0]:.4f}/"
        f"{under_prevalence[1]:.4f}"
    )

    print(
        "Frozen Base validation overall: "
        f"{base_validation_metrics['overall_mae_pga']:.4f}/"
        f"{base_validation_metrics['overall_mae_pgv']:.4f}"
    )

    print(
        "Frozen Base validation tail   : "
        f"{base_validation_metrics['tail_mae_pga']:.4f}/"
        f"{base_validation_metrics['tail_mae_pgv']:.4f}"
    )

    for epoch in range(
        1,
        args.head_epochs
        + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_metrics = (
            risk_module.train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                thresholds=(
                    threshold_tensor
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
        )

        head_now = (
            risk_module
            .predict_head_from_cache(
                model,
                monitor_cache[
                    "risk_feature"
                ],
                device,
                args.head_eval_batch_size,
            )
        )

        linear_prediction = (
            monitor_cache[
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

        monitor_metrics = (
            risk_module.metric_bundle(
                monitor_cache,
                linear_prediction,
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
                "model_state": (
                    model.state_dict()
                ),
                "epoch": (
                    epoch
                ),
                "thresholds": {
                    "log10_pga_threshold": float(
                        thresholds[
                            0
                        ]
                    ),
                    "log10_pgv_threshold": float(
                        thresholds[
                            1
                        ]
                    ),
                },
                "tail_prevalence": (
                    tail_prevalence.tolist()
                ),
                "under_prevalence": (
                    under_prevalence.tolist()
                ),
                "monitor_metrics_gamma1": (
                    monitor_metrics
                ),
                "station_ood_training": (
                    True
                ),
                "test_evaluated": (
                    False
                ),
            },
            checkpoint_path,
        )

        history.append(
            {
                "epoch": epoch,
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
                    in train_metrics.items()
                },
                **{
                    f"monitor_{key}": value
                    for key, value
                    in monitor_metrics.items()
                },
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            head_dir
            / "history.csv",
            index=False,
        )

        print(
            f"Head epoch {epoch:03d} | "
            f"loss={train_metrics['loss']:.4f} | "
            "val overall="
            f"{monitor_metrics['overall_mae_pga']:.4f}/"
            f"{monitor_metrics['overall_mae_pgv']:.4f} | "
            "tail="
            f"{monitor_metrics['tail_mae_pga']:.4f}/"
            f"{monitor_metrics['tail_mae_pgv']:.4f} | "
            "U05="
            f"{monitor_metrics['tail_under05_pga']:.3f}/"
            f"{monitor_metrics['tail_under05_pgv']:.3f}"
        )

    powers = parse_float_list(
        args.powers
    )

    scan_rows = []

    for epoch in range(
        1,
        args.head_epochs
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

        head = (
            risk_module
            .predict_head_from_cache(
                model,
                selection_cache[
                    "risk_feature"
                ],
                device,
                args.head_eval_batch_size,
            )
        )

        for gamma in powers:
            prediction = (
                selection_cache[
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
                    gamma,
                )
                * head[
                    "residual"
                ]
            )

            metrics = (
                risk_module.metric_bundle(
                    selection_cache,
                    prediction,
                )
            )

            status = (
                risk_module.feasibility_status(
                    metrics,
                    base_validation_metrics,
                    args.overall_budget,
                    args.non_tail_budget,
                    args.bias_limit,
                )
            )

            scan_rows.append(
                {
                    "epoch": epoch,
                    "gamma": (
                        gamma
                    ),
                    **metrics,
                    **status,
                }
            )

    scan = pd.DataFrame(
        scan_rows
    )

    scan.to_csv(
        head_dir
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
        payload = {
            "status": (
                "no_feasible_candidate"
            ),
            "base_validation_metrics": (
                base_validation_metrics
            ),
            "test_evaluated": (
                False
            ),
        }

        (
            head_dir
            / "selection_failure.json"
        ).write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\nNO station-OOD Dual-Risk "
            "candidate satisfied all "
            "validation constraints."
        )

        return (
            None,
            payload,
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
        checkpoint_dir
        / f"epoch_{selected_epoch:03d}.pt"
    )

    selected_checkpoint_path = (
        head_dir
        / "selected_dual_risk_head.pt"
    )

    shutil.copy2(
        selected_source,
        selected_checkpoint_path,
    )

    selected_checkpoint = (
        load_checkpoint(
            selected_checkpoint_path,
            device,
        )
    )

    model.load_state_dict(
        selected_checkpoint[
            "model_state"
        ],
        strict=True,
    )

    model.eval()

    selection = {
        "status": (
            "selected"
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
                bool(value)
                if isinstance(
                    value,
                    (
                        bool,
                        np.bool_,
                    ),
                )
                else (
                    int(value)
                    if isinstance(
                        value,
                        (
                            int,
                            np.integer,
                        ),
                    )
                    else (
                        float(value)
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
        "test_evaluated": (
            False
        ),
    }

    (
        head_dir
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
        head_dir
        / "top_feasible_validation_candidates.csv",
        index=False,
    )

    print(
        "\n=== Selected station-OOD "
        "Dual-Risk model ==="
    )

    print(
        f"Epoch                   : "
        f"{selected_epoch}"
    )

    print(
        f"Gamma                   : "
        f"{selected_gamma:g}"
    )

    print(
        f"At upper grid boundary  : "
        f"{upper_boundary}"
    )

    print(
        "Overall MAE PGA/PGV     : "
        f"{selected['overall_mae_pga']:.4f}/"
        f"{selected['overall_mae_pgv']:.4f}"
    )

    print(
        "Tail MAE PGA/PGV        : "
        f"{selected['tail_mae_pga']:.4f}/"
        f"{selected['tail_mae_pgv']:.4f}"
    )

    print(
        "Tail U0.5 PGA/PGV       : "
        f"{selected['tail_under05_pga']:.3f}/"
        f"{selected['tail_under05_pgv']:.3f}"
    )

    print(
        f"Feasible candidates     : "
        f"{len(feasible)}/{len(scan)}"
    )

    if upper_boundary:
        print(
            "\nSelected gamma is at the "
            "upper validation-grid boundary."
        )

        print(
            "Extend gamma on VALIDATION ONLY "
            "before station-OOD test evaluation."
        )

        return (
            None,
            selection,
        )

    return (
        model,
        selection,
    )


# ---------------------------------------------------------------------
# Paired seen-vs-unseen CA-DRG test
# ---------------------------------------------------------------------

def evaluate_paired_station_ood_cadrg(
    model: nn.Module,
    selected_gamma: float,
    thresholds: np.ndarray,
    scenario_csv: Path,
    assignment_csv: Path,
    ood_module,
    risk_module,
    args,
    device: torch.device,
) -> pd.DataFrame:
    scenario = pd.read_csv(
        scenario_csv,
        dtype={
            "event_id": str,
        },
    )

    scenario = scenario.loc[
        (
            scenario[
                args.split_column
            ]
            .astype(
                str
            )
            .eq(
                str(
                    args.test_label
                )
            )
        )
        & (
            ood_module.as_bool(
                scenario[
                    "test_paired_seen_ood_eligible"
                ]
            )
        )
    ].copy()

    scenario = scenario.reset_index(
        drop=True
    )

    assignment = pd.read_csv(
        assignment_csv,
        dtype={
            "station_id": str,
        },
    )

    role = dict(
        zip(
            assignment[
                "station_id"
            ].astype(
                str
            ),
            assignment[
                "station_role"
            ].astype(
                str
            ),
        )
    )

    rows = []

    print(
        "\n=== Step 4C: paired "
        "seen-vs-unseen station-OOD test ==="
    )

    print(
        f"Test events             : "
        f"{len(scenario)}"
    )

    print(
        f"Repeats/event           : "
        f"{args.test_repeats}"
    )

    print(
        f"Targets/role            : "
        f"{args.paired_targets}"
    )

    print(
        f"Selected gamma          : "
        f"{selected_gamma:g}"
    )

    model.eval()

    for event_number, event in enumerate(
        scenario.itertuples(
            index=False
        ),
        start=1,
    ):
        event_id = str(
            event.event_id
        )

        h5_path = Path(
            str(
                event.h5_path
            )
        )

        with h5py.File(
            h5_path,
            "r",
        ) as h5:
            acceleration = np.asarray(
                h5[
                    "acceleration"
                ][:],
                dtype=np.float32,
            )

            velocity = np.asarray(
                h5[
                    "velocity"
                ][:],
                dtype=np.float32,
            )

            coordinates = np.asarray(
                h5[
                    "station_coords"
                ][:],
                dtype=np.float32,
            )

            p_offset = np.asarray(
                h5[
                    "p_offset_sec"
                ][:],
                dtype=np.float32,
            )

            station_ids = (
                ood_module.read_station_ids(
                    h5,
                    len(
                        coordinates
                    ),
                )
            )

            sampling_rate = float(
                h5.attrs[
                    "sampling_rate_hz"
                ]
            )

            pre_first_p = float(
                h5.attrs[
                    "pre_first_p_sec"
                ]
            )

            time_zero_index = int(
                h5.attrs[
                    "time_zero_index"
                ]
            )

        roles = np.asarray(
            [
                role.get(
                    str(
                        station_id
                    ),
                    "unknown",
                )
                for station_id
                in station_ids
            ],
            dtype=object,
        )

        if np.any(
            roles
            == "unknown"
        ):
            raise RuntimeError(
                f"Event {event_id}: "
                "missing station assignment."
            )

        seen = (
            roles
            == "seen"
        )

        ood = (
            roles
            == "ood_holdout"
        )

        triggered_seen = np.flatnonzero(
            seen
            & np.isfinite(
                p_offset
            )
            & (
                p_offset
                >= -1e-3
            )
            & (
                p_offset
                <= float(
                    args.t0_sec
                )
            )
        )

        if len(
            triggered_seen
        ) < args.input_stations:
            raise RuntimeError(
                f"Event {event_id}: "
                "insufficient seen inputs."
            )

        for repeat in range(
            args.test_repeats
        ):
            rng = (
                np.random.default_rng(
                    ood_module.stable_seed(
                        (
                            "station_ood:test:"
                            f"{event_id}:"
                            f"repeat:{repeat}"
                        ),
                        args.seed,
                    )
                )
            )

            input_indices = (
                rng.choice(
                    triggered_seen,
                    size=(
                        args.input_stations
                    ),
                    replace=False,
                )
            )

            seen_pool = np.setdiff1d(
                np.flatnonzero(
                    seen
                ),
                input_indices,
                assume_unique=False,
            )

            ood_pool = np.flatnonzero(
                ood
            )

            if (
                len(
                    seen_pool
                )
                < args.paired_targets
                or len(
                    ood_pool
                )
                < args.paired_targets
            ):
                raise RuntimeError(
                    f"Event {event_id}: "
                    "insufficient paired targets."
                )

            seen_targets = (
                rng.choice(
                    seen_pool,
                    size=(
                        args.paired_targets
                    ),
                    replace=False,
                )
            )

            ood_targets = (
                rng.choice(
                    ood_pool,
                    size=(
                        args.paired_targets
                    ),
                    replace=False,
                )
            )

            for (
                target_role,
                target_indices,
            ) in (
                (
                    "seen_target",
                    seen_targets,
                ),
                (
                    "unseen_station_target",
                    ood_targets,
                ),
            ):
                sample = (
                    ood_module
                    .build_model_sample(
                        event_id=(
                            event_id
                        ),
                        repeat_index=(
                            repeat
                        ),
                        acceleration=(
                            acceleration
                        ),
                        velocity=(
                            velocity
                        ),
                        coordinates=(
                            coordinates
                        ),
                        p_offset=(
                            p_offset
                        ),
                        input_indices=(
                            input_indices
                        ),
                        target_indices=(
                            target_indices
                        ),
                        sampling_rate=(
                            sampling_rate
                        ),
                        pre_first_p=(
                            pre_first_p
                        ),
                        time_zero_index=(
                            time_zero_index
                        ),
                        t0_sec=(
                            args.t0_sec
                        ),
                        input_pre_sec=(
                            args.input_pre_sec
                        ),
                    )
                )

                input_coordinates = (
                    coordinates[
                        input_indices
                    ]
                )

                origin_lat = float(
                    input_coordinates[
                        :,
                        0,
                    ].mean()
                )

                origin_lon = float(
                    input_coordinates[
                        :,
                        1,
                    ].mean()
                )

                input_xy = (
                    ood_module.local_xy_km(
                        input_coordinates,
                        origin_lat,
                        origin_lon,
                    )
                )

                target_xy = (
                    ood_module.local_xy_km(
                        coordinates[
                            target_indices
                        ],
                        origin_lat,
                        origin_lon,
                    )
                )

                nearest = (
                    ood_module
                    .pairwise_nearest_distance(
                        target_xy,
                        input_xy,
                    )
                )

                with torch.inference_mode():
                    (
                        base_prediction,
                        risk_feature,
                    ) = (
                        risk_module
                        .cross_attention_base_context(
                            model.base,
                            sample[
                                "input_waveforms"
                            ][
                                None
                            ].to(
                                device
                            ),
                            sample[
                                "input_features"
                            ][
                                None
                            ].to(
                                device
                            ),
                            sample[
                                "target_features"
                            ][
                                None
                            ].to(
                                device
                            ),
                        )
                    )

                    head = (
                        model.head_from_feature(
                            risk_feature
                        )
                    )

                    correction = (
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
                            selected_gamma,
                        )
                        * head[
                            "residual"
                        ]
                    )

                    final_prediction = (
                        base_prediction
                        + correction
                    )

                base_np = (
                    base_prediction[
                        0
                    ]
                    .cpu()
                    .numpy()
                )

                final_np = (
                    final_prediction[
                        0
                    ]
                    .cpu()
                    .numpy()
                )

                tail_probability = (
                    head[
                        "tail_probability"
                    ][
                        0
                    ]
                    .cpu()
                    .numpy()
                )

                under_probability = (
                    head[
                        "under_probability"
                    ][
                        0
                    ]
                    .cpu()
                    .numpy()
                )

                correction_magnitude = (
                    head[
                        "residual"
                    ][
                        0
                    ]
                    .cpu()
                    .numpy()
                )

                applied_correction = (
                    correction[
                        0
                    ]
                    .cpu()
                    .numpy()
                )

                truth = (
                    sample[
                        "target_log"
                    ]
                    .numpy()
                )

                input_station_text = "|".join(
                    str(
                        station_ids[
                            index
                        ]
                    )
                    for index
                    in input_indices
                )

                for (
                    q,
                    station_index,
                ) in enumerate(
                    target_indices
                ):
                    true_pga = float(
                        truth[
                            q,
                            0,
                        ]
                    )

                    true_pgv = float(
                        truth[
                            q,
                            1,
                        ]
                    )

                    rows.append(
                        {
                            "event_id": (
                                event_id
                            ),
                            "repeat": int(
                                repeat
                            ),
                            "target_role": (
                                target_role
                            ),
                            "input_station_ids": (
                                input_station_text
                            ),
                            "target_station_id": str(
                                station_ids[
                                    station_index
                                ]
                            ),
                            "target_station_index": int(
                                station_index
                            ),
                            "target_p_offset_sec": (
                                float(
                                    p_offset[
                                        station_index
                                    ]
                                )
                                if np.isfinite(
                                    p_offset[
                                        station_index
                                    ]
                                )
                                else np.nan
                            ),
                            "target_p_wave_reached": bool(
                                np.isfinite(
                                    p_offset[
                                        station_index
                                    ]
                                )
                                and (
                                    p_offset[
                                        station_index
                                    ]
                                    <= args.t0_sec
                                )
                            ),
                            "nearest_input_distance_km": float(
                                nearest[
                                    q
                                ]
                            ),
                            "true_log10_pga": (
                                true_pga
                            ),
                            "true_log10_pgv": (
                                true_pgv
                            ),
                            "base_log10_pga": float(
                                base_np[
                                    q,
                                    0,
                                ]
                            ),
                            "base_log10_pgv": float(
                                base_np[
                                    q,
                                    1,
                                ]
                            ),
                            "final_log10_pga": float(
                                final_np[
                                    q,
                                    0,
                                ]
                            ),
                            "final_log10_pgv": float(
                                final_np[
                                    q,
                                    1,
                                ]
                            ),
                            "tail_probability_pga": float(
                                tail_probability[
                                    q,
                                    0,
                                ]
                            ),
                            "tail_probability_pgv": float(
                                tail_probability[
                                    q,
                                    1,
                                ]
                            ),
                            "under_probability_pga": float(
                                under_probability[
                                    q,
                                    0,
                                ]
                            ),
                            "under_probability_pgv": float(
                                under_probability[
                                    q,
                                    1,
                                ]
                            ),
                            "correction_capacity_pga": float(
                                correction_magnitude[
                                    q,
                                    0,
                                ]
                            ),
                            "correction_capacity_pgv": float(
                                correction_magnitude[
                                    q,
                                    1,
                                ]
                            ),
                            "applied_correction_pga": float(
                                applied_correction[
                                    q,
                                    0,
                                ]
                            ),
                            "applied_correction_pgv": float(
                                applied_correction[
                                    q,
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
                            "selected_gamma": float(
                                selected_gamma
                            ),
                        }
                    )

        if (
            event_number
            % 25
            == 0
            or event_number
            == len(
                scenario
            )
        ):
            print(
                f"Evaluated "
                f"{event_number}/"
                f"{len(scenario)} events | "
                f"rows={len(rows):,}"
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Canonical OOD metrics
# ---------------------------------------------------------------------

def element_metric(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> np.ndarray:
    residual = (
        prediction
        - truth
    )

    absolute = np.abs(
        residual
    )

    if metric == "mae":
        return absolute

    if metric == "bias":
        return residual

    if metric == "factor2":
        return (
            absolute
            <= LOG10_FACTOR_2
        ).astype(
            float
        )

    if metric == "factor3":
        return (
            absolute
            <= LOG10_FACTOR_3
        ).astype(
            float
        )

    if metric == "under05":
        return (
            residual
            <= SEVERE_UNDER_THRESHOLD
        ).astype(
            float
        )

    raise ValueError(
        metric
    )


def event_level_values(
    frame: pd.DataFrame,
    prediction_column: str,
    truth_column: str,
    metric: str,
) -> pd.Series:
    truth = frame[
        truth_column
    ].to_numpy(
        dtype=float
    )

    prediction = frame[
        prediction_column
    ].to_numpy(
        dtype=float
    )

    values = element_metric(
        truth,
        prediction,
        metric,
    )

    work = frame[
        [
            "event_id",
            "repeat",
        ]
    ].copy()

    work[
        "value"
    ] = values

    event_repeat = (
        work.groupby(
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

    return (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )[
            "value"
        ]
        .mean()
    )


def station_ood_metric_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    models = {
        "cross_attention_base": (
            "base"
        ),
        "ca_drg": (
            "final"
        ),
    }

    for (
        model_name,
        prefix,
    ) in models.items():
        for (
            target_role,
            role_group,
        ) in predictions.groupby(
            "target_role",
            sort=False,
        ):
            for quantity in (
                "pga",
                "pgv",
            ):
                populations = {
                    "overall": (
                        np.ones(
                            len(
                                role_group
                            ),
                            dtype=bool,
                        )
                    ),
                    "high_motion_tail": (
                        as_bool_array(
                            role_group[
                                f"is_tail_{quantity}"
                            ]
                        )
                    ),
                }

                for (
                    population,
                    mask,
                ) in populations.items():
                    subset = (
                        role_group.loc[
                            mask
                        ].copy()
                    )

                    if subset.empty:
                        continue

                    for metric in (
                        "mae",
                        "bias",
                        "factor2",
                        "factor3",
                        "under05",
                    ):
                        event_values = (
                            event_level_values(
                                subset,
                                f"{prefix}_log10_{quantity}",
                                f"true_log10_{quantity}",
                                metric,
                            )
                        )

                        rows.append(
                            {
                                "model": (
                                    model_name
                                ),
                                "target_role": (
                                    target_role
                                ),
                                "quantity": (
                                    quantity
                                ),
                                "population": (
                                    population
                                ),
                                "metric": (
                                    metric
                                ),
                                "value": float(
                                    event_values.mean()
                                ),
                                "n_events": int(
                                    len(
                                        event_values
                                    )
                                ),
                                "n_target_rows": int(
                                    len(
                                        subset
                                    )
                                ),
                                "n_unique_stations": int(
                                    subset[
                                        "target_station_id"
                                    ].nunique()
                                ),
                            }
                        )

    return pd.DataFrame(
        rows
    )


def paired_seen_unseen_deltas(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    models = {
        "cross_attention_base": (
            "base"
        ),
        "ca_drg": (
            "final"
        ),
    }

    for (
        model_name,
        prefix,
    ) in models.items():
        for quantity in (
            "pga",
            "pgv",
        ):
            for population in (
                "overall",
                "high_motion_tail",
            ):
                for metric in (
                    "mae",
                    "bias",
                    "factor2",
                    "factor3",
                    "under05",
                ):
                    role_values = {}

                    for (
                        target_role,
                        role_group,
                    ) in predictions.groupby(
                        "target_role",
                        sort=False,
                    ):
                        if (
                            population
                            == "high_motion_tail"
                        ):
                            role_group = (
                                role_group.loc[
                                    as_bool_array(
                                        role_group[
                                            f"is_tail_{quantity}"
                                        ]
                                    )
                                ].copy()
                            )

                        if role_group.empty:
                            continue

                        role_values[
                            target_role
                        ] = (
                            event_level_values(
                                role_group,
                                f"{prefix}_log10_{quantity}",
                                f"true_log10_{quantity}",
                                metric,
                            )
                        )

                    if not {
                        "seen_target",
                        "unseen_station_target",
                    }.issubset(
                        role_values
                    ):
                        continue

                    seen = role_values[
                        "seen_target"
                    ]

                    unseen = role_values[
                        "unseen_station_target"
                    ]

                    common = (
                        seen.index
                        .intersection(
                            unseen.index
                        )
                    )

                    if len(
                        common
                    ) == 0:
                        continue

                    delta = (
                        unseen.loc[
                            common
                        ]
                        - seen.loc[
                            common
                        ]
                    )

                    seen_value = float(
                        seen.loc[
                            common
                        ].mean()
                    )

                    unseen_value = float(
                        unseen.loc[
                            common
                        ].mean()
                    )

                    record = {
                        "model": (
                            model_name
                        ),
                        "quantity": (
                            quantity
                        ),
                        "population": (
                            population
                        ),
                        "metric": (
                            metric
                        ),
                        "n_paired_events": int(
                            len(
                                common
                            )
                        ),
                        "seen_value": (
                            seen_value
                        ),
                        "unseen_value": (
                            unseen_value
                        ),
                        "mean_delta_unseen_minus_seen": float(
                            delta.mean()
                        ),
                        "median_delta_unseen_minus_seen": float(
                            delta.median()
                        ),
                    }

                    if metric == "mae":
                        record[
                            "relative_ood_gap_percent"
                        ] = float(
                            100.0
                            * (
                                unseen_value
                                - seen_value
                            )
                            / max(
                                abs(
                                    seen_value
                                ),
                                1e-12,
                            )
                        )

                    if metric in {
                        "factor2",
                        "under05",
                    }:
                        record[
                            "delta_percentage_points"
                        ] = float(
                            100.0
                            * (
                                unseen_value
                                - seen_value
                            )
                        )

                    rows.append(
                        record
                    )

    return pd.DataFrame(
        rows
    )


def cadrg_minus_base_by_role(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        target_role,
        role_group,
    ) in predictions.groupby(
        "target_role",
        sort=False,
    ):
        for quantity in (
            "pga",
            "pgv",
        ):
            for population in (
                "overall",
                "high_motion_tail",
            ):
                subset = role_group

                if (
                    population
                    == "high_motion_tail"
                ):
                    subset = (
                        role_group.loc[
                            as_bool_array(
                                role_group[
                                    f"is_tail_{quantity}"
                                ]
                            )
                        ].copy()
                    )

                if subset.empty:
                    continue

                for metric in (
                    "mae",
                    "bias",
                    "factor2",
                    "under05",
                ):
                    base = (
                        event_level_values(
                            subset,
                            f"base_log10_{quantity}",
                            f"true_log10_{quantity}",
                            metric,
                        )
                    )

                    final = (
                        event_level_values(
                            subset,
                            f"final_log10_{quantity}",
                            f"true_log10_{quantity}",
                            metric,
                        )
                    )

                    common = (
                        base.index
                        .intersection(
                            final.index
                        )
                    )

                    delta = (
                        final.loc[
                            common
                        ]
                        - base.loc[
                            common
                        ]
                    )

                    rows.append(
                        {
                            "target_role": (
                                target_role
                            ),
                            "quantity": (
                                quantity
                            ),
                            "population": (
                                population
                            ),
                            "metric": (
                                metric
                            ),
                            "n_paired_events": int(
                                len(
                                    common
                                )
                            ),
                            "base_value": float(
                                base.loc[
                                    common
                                ].mean()
                            ),
                            "ca_drg_value": float(
                                final.loc[
                                    common
                                ].mean()
                            ),
                            "mean_delta_cadrg_minus_base": float(
                                delta.mean()
                            ),
                            "median_delta_cadrg_minus_base": float(
                                delta.median()
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def distance_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions.copy()

    frame[
        "target_p_wave_reached"
    ] = as_bool_array(
        frame[
            "target_p_wave_reached"
        ]
    ).astype(
        float
    )

    return (
        frame.groupby(
            "target_role",
            sort=False,
        )
        .agg(
            n_rows=(
                "nearest_input_distance_km",
                "size",
            ),
            n_events=(
                "event_id",
                "nunique",
            ),
            n_unique_stations=(
                "target_station_id",
                "nunique",
            ),
            median_nearest_input_distance_km=(
                "nearest_input_distance_km",
                "median",
            ),
            p25_nearest_input_distance_km=(
                "nearest_input_distance_km",
                lambda values: float(
                    np.percentile(
                        values,
                        25,
                    )
                ),
            ),
            p75_nearest_input_distance_km=(
                "nearest_input_distance_km",
                lambda values: float(
                    np.percentile(
                        values,
                        75,
                    )
                ),
            ),
            fraction_p_wave_reached=(
                "target_p_wave_reached",
                "mean",
            ),
        )
        .reset_index()
    )


def tail_prevalence_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        role,
        group,
    ) in predictions.groupby(
        "target_role",
        sort=False,
    ):
        for quantity in (
            "pga",
            "pgv",
        ):
            tail = as_bool_array(
                group[
                    f"is_tail_{quantity}"
                ]
            )

            rows.append(
                {
                    "target_role": (
                        role
                    ),
                    "quantity": (
                        quantity
                    ),
                    "n_rows": int(
                        len(
                            group
                        )
                    ),
                    "n_tail_rows": int(
                        tail.sum()
                    ),
                    "tail_prevalence": float(
                        tail.mean()
                    ),
                    "n_tail_events": int(
                        group.loc[
                            tail,
                            "event_id",
                        ].nunique()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ood-module",
        default=(
            "52_phase2_train_and_evaluate_station_ood_attention.py"
        ),
    )

    parser.add_argument(
        "--baseline-module",
        default=(
            "45_phase2_strong_baseline_suite.py"
        ),
    )

    parser.add_argument(
        "--risk-module",
        default=(
            "60_train_cross_attention_dual_risk_joint_selection.py"
        ),
    )

    parser.add_argument(
        "--scenario",
        default=(
            "data/model_manifests/"
            "station_ood/"
            "scenario_station_ood.csv"
        ),
    )

    parser.add_argument(
        "--station-assignment",
        default=(
            "data/model_manifests/"
            "station_ood/"
            "station_holdout_assignment.csv"
        ),
    )

    parser.add_argument(
        "--h5-root",
        default=(
            "data/processed_full_v4/events"
        ),
        help=(
            "Fallback local HDF5 root. Useful when scenario h5_path "
            "contains paths from another machine."
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
        "--train-target-stations",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--paired-targets",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--test-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--input-pre-sec",
        type=float,
        default=2.0,
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

    # Cross-Attention Base training.
    parser.add_argument(
        "--base-epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--base-validation-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--base-learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--base-minimum-learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--base-weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--base-lr-patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--base-early-stopping-patience",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--base-minimum-delta",
        type=float,
        default=1e-4,
    )

    # TRAIN-only tail threshold.
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--threshold-repeats",
        type=int,
        default=10,
    )

    # Dual-Risk head.
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
        "--head-epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--head-weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--head-monitor-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--head-selection-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--prevalence-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--head-eval-batch-size",
        type=int,
        default=8192,
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
        "--batch-size",
        type=int,
        default=8,
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
            "cadrg_station_ood"
        ),
    )

    args = parser.parse_args()

    if not (
        0.5
        < args.tail_quantile
        < 1.0
    ):
        raise ValueError(
            "--tail-quantile must be between 0.5 and 1."
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

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario_input = Path(
        args.scenario
    )

    assignment_path = Path(
        args.station_assignment
    )

    if not scenario_input.exists():
        raise FileNotFoundError(
            scenario_input
        )

    if not assignment_path.exists():
        raise FileNotFoundError(
            assignment_path
        )

    baseline_module = load_module(
        args.baseline_module,
        "station_ood_cross_attention_architecture",
    )

    ood_module = load_module(
        args.ood_module,
        "station_ood_protocol_module",
    )

    risk_module = load_module(
        args.risk_module,
        "station_ood_dual_risk_module",
    )

    resolved_scenario_path = (
        out_dir
        / "resolved_scenario_station_ood.csv"
    )

    h5_root = (
        Path(
            args.h5_root
        )
        if str(
            args.h5_root
        ).strip()
        else None
    )

    scenario_frame = (
        resolve_scenario_for_local_machine(
            scenario_input,
            h5_root,
            resolved_scenario_path,
        )
    )

    assignment = pd.read_csv(
        assignment_path,
        dtype={
            "station_id": str,
        },
    )

    station_role_counts = (
        assignment[
            "station_role"
        ]
        .value_counts()
        .to_dict()
    )

    protocol = {
        **vars(
            args
        ),
        "resolved_device": str(
            device
        ),
        "resolved_scenario": str(
            resolved_scenario_path.resolve()
        ),
        "station_role_counts": (
            station_role_counts
        ),
        "strict_station_ood": True,
        "ood_station_train_input_use": False,
        "ood_station_train_target_use": False,
        "ood_station_validation_input_use": False,
        "ood_station_validation_target_use": False,
        "paired_test_rule": (
            "same event + same repeat + same seen inputs + "
            "equal seen/unseen target budgets"
        ),
        "metric_aggregation": (
            "targets -> repeats -> events"
        ),
        "tail_threshold_rule": (
            "Q90 computed from TRAIN seen-only target draws only"
        ),
        "test_used_for_base_selection": False,
        "test_used_for_gate_selection": False,
    }

    (
        out_dir
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== Step 4: CA-DRG Strict Station OOD ==="
    )

    print(
        f"Device                   : "
        f"{device}"
    )

    print(
        f"Scenario events          : "
        f"{len(scenario_frame)}"
    )

    print(
        f"Station roles            : "
        f"{station_role_counts}"
    )

    print(
        "Train/Val OOD stations  : EXCLUDED"
    )

    start_time = time.time()

    # -------------------------------------------------------------
    # 4A: retrain Cross-Attention Base under strict station OOD.
    # -------------------------------------------------------------

    base, base_summary = (
        train_station_ood_cross_attention_base(
            args,
            baseline_module,
            ood_module,
            resolved_scenario_path,
            assignment_path,
            device,
            out_dir,
        )
    )

    # -------------------------------------------------------------
    # TRAIN seen-only Q90 threshold.
    # -------------------------------------------------------------

    thresholds, threshold_info = (
        estimate_seen_only_train_thresholds(
            ood_module,
            resolved_scenario_path,
            assignment_path,
            args,
            device,
        )
    )

    threshold_path = (
        out_dir
        / "station_ood_tail_thresholds.json"
    )

    threshold_path.write_text(
        json.dumps(
            threshold_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Station-OOD TRAIN-only Q90 thresholds ==="
    )

    print(
        "PGA log10 threshold      : "
        f"{thresholds[0]:.6f}"
    )

    print(
        "PGV log10 threshold      : "
        f"{thresholds[1]:.6f}"
    )

    print(
        "Threshold target rows    : "
        f"{threshold_info['n_train_target_rows']:,}"
    )

    # -------------------------------------------------------------
    # 4B: frozen Base + Dual-Risk head.
    # -------------------------------------------------------------

    model, selection = (
        train_and_select_station_ood_dual_risk(
            args,
            ood_module,
            risk_module,
            base,
            resolved_scenario_path,
            assignment_path,
            thresholds,
            device,
            out_dir,
        )
    )

    if model is None:
        final_status = {
            "status": (
                "stopped_before_test"
            ),
            "reason": (
                "No deployable validation-selected station-OOD "
                "Dual-Risk candidate."
            ),
            "base_summary": (
                base_summary
            ),
            "dual_risk_selection": (
                selection
            ),
            "training_seconds": float(
                time.time()
                - start_time
            ),
        }

        (
            out_dir
            / "run_summary.json"
        ).write_text(
            json.dumps(
                final_status,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\nStation-OOD test was NOT accessed."
        )

        return

    selected_gamma = float(
        selection[
            "selected_gamma"
        ]
    )

    # -------------------------------------------------------------
    # 4C: paired seen/unseen station OOD test.
    # -------------------------------------------------------------

    predictions = (
        evaluate_paired_station_ood_cadrg(
            model=model,
            selected_gamma=(
                selected_gamma
            ),
            thresholds=(
                thresholds
            ),
            scenario_csv=(
                resolved_scenario_path
            ),
            assignment_csv=(
                assignment_path
            ),
            ood_module=(
                ood_module
            ),
            risk_module=(
                risk_module
            ),
            args=(
                args
            ),
            device=(
                device
            ),
        )
    )

    prediction_path = (
        out_dir
        / "repeated_station_ood_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    metrics = (
        station_ood_metric_table(
            predictions
        )
    )

    metrics_path = (
        out_dir
        / "station_ood_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    paired = (
        paired_seen_unseen_deltas(
            predictions
        )
    )

    paired_path = (
        out_dir
        / "station_ood_paired_event_deltas.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    gate_effect = (
        cadrg_minus_base_by_role(
            predictions
        )
    )

    gate_effect_path = (
        out_dir
        / "cadrg_minus_base_by_target_role.csv"
    )

    gate_effect.to_csv(
        gate_effect_path,
        index=False,
    )

    distances = (
        distance_summary(
            predictions
        )
    )

    distance_path = (
        out_dir
        / "station_ood_distance_summary.csv"
    )

    distances.to_csv(
        distance_path,
        index=False,
    )

    prevalence_table = (
        tail_prevalence_table(
            predictions
        )
    )

    prevalence_path = (
        out_dir
        / "station_ood_tail_prevalence.csv"
    )

    prevalence_table.to_csv(
        prevalence_path,
        index=False,
    )

    final_summary = {
        "status": (
            "completed"
        ),
        "training_seconds": float(
            time.time()
            - start_time
        ),
        "base_summary": (
            base_summary
        ),
        "dual_risk_selection": (
            selection
        ),
        "tail_thresholds": (
            threshold_info
        ),
        "test_events": int(
            predictions[
                "event_id"
            ].nunique()
        ),
        "test_repeats": int(
            args.test_repeats
        ),
        "paired_targets_per_role": int(
            args.paired_targets
        ),
        "test_rows": int(
            len(
                predictions
            )
        ),
        "seen_target_rows": int(
            (
                predictions[
                    "target_role"
                ]
                == "seen_target"
            ).sum()
        ),
        "unseen_target_rows": int(
            (
                predictions[
                    "target_role"
                ]
                == "unseen_station_target"
            ).sum()
        ),
        "seen_unique_stations": int(
            predictions.loc[
                predictions[
                    "target_role"
                ]
                == "seen_target",
                "target_station_id",
            ].nunique()
        ),
        "unseen_unique_stations": int(
            predictions.loc[
                predictions[
                    "target_role"
                ]
                == "unseen_station_target",
                "target_station_id",
            ].nunique()
        ),
    }

    (
        out_dir
        / "run_summary.json"
    ).write_text(
        json.dumps(
            final_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Station-OOD canonical metrics ==="
    )

    display_metrics = (
        metrics.loc[
            metrics[
                "metric"
            ].isin(
                [
                    "mae",
                    "bias",
                    "factor2",
                    "under05",
                ]
            )
        ][
            [
                "model",
                "target_role",
                "quantity",
                "population",
                "metric",
                "value",
                "n_events",
                "n_target_rows",
                "n_unique_stations",
            ]
        ]
    )

    print(
        display_metrics.to_string(
            index=False
        )
    )

    print(
        "\n=== Unseen minus seen paired event deltas ==="
    )

    display_paired = (
        paired.loc[
            paired[
                "metric"
            ].isin(
                [
                    "mae",
                    "factor2",
                    "under05",
                ]
            )
        ]
    )

    print(
        display_paired.to_string(
            index=False
        )
    )

    print(
        "\n=== CA-DRG minus Base within each target role ==="
    )

    display_gate = (
        gate_effect.loc[
            gate_effect[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                ]
            )
        ]
    )

    print(
        display_gate.to_string(
            index=False
        )
    )

    print(
        "\n=== Geometry audit ==="
    )

    print(
        distances.to_string(
            index=False
        )
    )

    print(
        "\n=== Tail prevalence by target role ==="
    )

    print(
        prevalence_table.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    for path in (
        out_dir
        / "cross_attention_base"
        / "best_model.pt",
        threshold_path,
        out_dir
        / "dual_risk_head"
        / "selected_dual_risk_head.pt",
        out_dir
        / "dual_risk_head"
        / "selected_epoch_gamma.json",
        prediction_path,
        metrics_path,
        paired_path,
        gate_effect_path,
        distance_path,
        prevalence_path,
        out_dir
        / "run_summary.json",
    ):
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
