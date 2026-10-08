#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
71_train_and_evaluate_caurc_station_ood.py

Strict station-level OOD experiment for the final candidate CA-URC.

CA-URC
------
Cross-Attention Underprediction-Risk Correction

    y_final = y_CA + p_under**gamma * Delta

This is the Station-OOD counterpart of grouped ablation A4_under_only.

Protocol
--------
Station assignment:
    reuse the deterministic seen/OOD station assignment produced by Step 2A.

TRAIN / VALIDATION:
    - OOD station IDs are never inputs.
    - OOD station IDs are never targets.
    - input stations are P-wave-reached SEEN stations only.
    - targets are remaining SEEN stations only.

TEST:
    - same event / repeat / K seen input stations for both target roles;
    - equal budgets of seen and unseen-station targets;
    - default 3 seen + 3 unseen targets, 20 repeats/event;
    - exact target-draw protocol is reused from script 52/65.

Model selection:
    1. strict Station-OOD Cross-Attention Base selected using VALIDATION only;
    2. Q90 thresholds computed from TRAIN seen-only target draws only;
    3. freeze Base;
    4. train only A4 underprediction-risk/correction head on TRAIN seen-only;
    5. jointly select head epoch + gamma on 20-repeat VALIDATION only;
    6. preservation guards:
         overall MAE increase <= 0.005 for BOTH PGA/PGV
         non-tail MAE increase <= 0.003 for BOTH PGA/PGV
         |bias_candidate| <= |bias_base| + 0.005 for BOTH PGA/PGV
    7. among feasible candidates:
         minimum mean tail MAE
         -> minimum mean tail U0.5
         -> minimum mean overall MAE
         -> lower gamma
         -> earlier epoch
    8. test is accessed only after a non-boundary validation candidate is locked.

Important
---------
- The old full CA-DRG Station-OOD head is NOT reused.
- The previously trained strict Station-OOD Cross-Attention Base MAY be reused,
  because it was trained/selected before the risk head and already satisfies
  the exact seen-only Station-OOD protocol.
- Use --retrain-base to force a fresh Station-OOD Cross-Attention run.

Required local scripts
----------------------
45_phase2_strong_baseline_suite.py
52_phase2_train_and_evaluate_station_ood_attention.py
63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py
65_train_and_evaluate_cadrg_station_ood.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


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

        values.append(
            value
        )

    values = sorted(
        set(values)
    )

    if not values:
        raise ValueError(
            "No gamma values supplied."
        )

    return values


def loader_kwargs(
    args,
    device: torch.device,
) -> dict[str, Any]:
    return {
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


# ---------------------------------------------------------------------
# Relative-bias validation guard
# ---------------------------------------------------------------------

def relative_bias_status(
    candidate: dict[str, float],
    base: dict[str, float],
    overall_budget: float,
    non_tail_budget: float,
    bias_margin: float,
) -> dict[str, Any]:
    result = {}
    passes = []

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

        tail_u05_delta = (
            candidate[
                f"tail_under05_{quantity}"
            ]
            - base[
                f"tail_under05_{quantity}"
            ]
        )

        base_bias = float(
            base[
                f"bias_{quantity}"
            ]
        )

        candidate_bias = float(
            candidate[
                f"bias_{quantity}"
            ]
        )

        bias_limit = (
            abs(
                base_bias
            )
            + float(
                bias_margin
            )
        )

        pass_overall = bool(
            overall_delta
            <= float(
                overall_budget
            )
        )

        pass_non_tail = bool(
            non_tail_delta
            <= float(
                non_tail_budget
            )
        )

        pass_bias = bool(
            abs(
                candidate_bias
            )
            <= bias_limit
        )

        result[
            f"overall_delta_{quantity}"
        ] = float(
            overall_delta
        )

        result[
            f"non_tail_delta_{quantity}"
        ] = float(
            non_tail_delta
        )

        result[
            f"tail_delta_{quantity}"
        ] = float(
            tail_delta
        )

        result[
            f"tail_under05_delta_{quantity}"
        ] = float(
            tail_u05_delta
        )

        result[
            f"relative_bias_limit_{quantity}"
        ] = float(
            bias_limit
        )

        result[
            f"pass_overall_{quantity}"
        ] = pass_overall

        result[
            f"pass_non_tail_{quantity}"
        ] = pass_non_tail

        result[
            f"pass_relative_bias_{quantity}"
        ] = pass_bias

        result[
            f"bias_abs_change_vs_base_{quantity}"
        ] = float(
            abs(
                candidate_bias
            )
            - abs(
                base_bias
            )
        )

        passes.extend(
            [
                pass_overall,
                pass_non_tail,
                pass_bias,
            ]
        )

    result[
        "feasible"
    ] = bool(
        all(
            passes
        )
    )

    return result


# ---------------------------------------------------------------------
# Base reuse / training
# ---------------------------------------------------------------------

def get_station_ood_base(
    args,
    baseline_module,
    ood_module,
    station_module,
    scenario_path: Path,
    assignment_path: Path,
    device: torch.device,
    out_dir: Path,
) -> tuple[
    nn.Module,
    dict[str, Any],
]:
    source = Path(
        args.base_checkpoint
    )

    if (
        not args.retrain_base
        and source.exists()
    ):
        checkpoint = load_checkpoint(
            source,
            device,
        )

        if not bool(
            checkpoint.get(
                "station_ood_training",
                False,
            )
        ):
            raise RuntimeError(
                "Existing --base-checkpoint is not marked as "
                "station_ood_training=True. Refusing to reuse it."
            )

        if (
            str(
                checkpoint.get(
                    "architecture",
                    "cross_attention",
                )
            )
            != "cross_attention"
        ):
            raise RuntimeError(
                "Existing Station-OOD Base checkpoint is not "
                "Cross-Attention."
            )

        model = (
            station_module
            .make_cross_attention(
                baseline_module,
                args.hidden_dim,
                args.attention_heads,
                device,
            )
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ],
            strict=True,
        )

        model.eval()

        reused_dir = (
            out_dir
            / "cross_attention_base"
        )

        reused_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        reused_path = (
            reused_dir
            / "best_model.pt"
        )

        shutil.copy2(
            source,
            reused_path,
        )

        summary = {
            "reused": True,
            "source_checkpoint": str(
                source.resolve()
            ),
            "checkpoint": str(
                reused_path.resolve()
            ),
            "best_epoch": int(
                checkpoint.get(
                    "epoch",
                    -1,
                )
            ),
            "best_validation_mae_pga": float(
                checkpoint.get(
                    "validation_mae_pga",
                    np.nan,
                )
            ),
            "best_validation_mae_pgv": float(
                checkpoint.get(
                    "validation_mae_pgv",
                    np.nan,
                )
            ),
            "station_ood_training": True,
            "test_used_for_base_selection": False,
        }

        (
            reused_dir
            / "base_selection_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\n=== Step 5A: reuse strict "
            "Station-OOD Cross-Attention Base ==="
        )

        print(
            f"Source checkpoint       : "
            f"{source.resolve()}"
        )

        print(
            f"Selected base epoch     : "
            f"{summary['best_epoch']}"
        )

        print(
            "Stored val PGA/PGV      : "
            f"{summary['best_validation_mae_pga']:.4f}/"
            f"{summary['best_validation_mae_pgv']:.4f}"
        )

        return (
            model,
            summary,
        )

    print(
        "\nNo reusable strict Station-OOD Base "
        "selected, or --retrain-base was requested."
    )

    return (
        station_module
        .train_station_ood_cross_attention_base(
            args,
            baseline_module,
            ood_module,
            scenario_path,
            assignment_path,
            device,
            out_dir,
        )
    )


# ---------------------------------------------------------------------
# CA-URC head train / validation lock
# ---------------------------------------------------------------------

def train_and_select_caurc(
    args,
    ood_module,
    ablation_module,
    station_module,
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
        / "ca_urc_head"
    )

    checkpoint_dir = (
        head_dir
        / "epoch_checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # TRAIN-only prevalence cache.
    # -------------------------------------------------------------
    prevalence_dataset = (
        station_module
        .make_seen_only_dataset(
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
        shuffle=False,
        **loader_kwargs(
            args,
            device,
        ),
    )

    prevalence_cache = (
        station_module
        .build_frozen_base_cache(
            base,
            prevalence_loader,
            thresholds,
            device,
            ablation_module,
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

    # Match the A4 variant's independent initialization stream.
    station_module.set_seed(
        args.seed
        + 2000
    )

    model = (
        ablation_module
        .AblationRiskGate(
            base=base,
            variant="A4_under_only",
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
        )
        .to(
            device
        )
    )

    # -------------------------------------------------------------
    # TRAIN.
    # -------------------------------------------------------------
    train_dataset = (
        station_module
        .make_seen_only_dataset(
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
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
        + 2777
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs(
            args,
            device,
        ),
    )

    # Small deterministic validation cache only for training monitor.
    monitor_dataset = (
        station_module
        .make_seen_only_dataset(
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
    )

    monitor_loader = DataLoader(
        monitor_dataset,
        shuffle=False,
        **loader_kwargs(
            args,
            device,
        ),
    )

    monitor_cache = (
        station_module
        .build_frozen_base_cache(
            base,
            monitor_loader,
            thresholds,
            device,
            ablation_module,
        )
    )

    # 20-repeat validation used for epoch x gamma lock.
    selection_dataset = (
        station_module
        .make_seen_only_dataset(
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
        shuffle=False,
        **loader_kwargs(
            args,
            device,
        ),
    )

    selection_cache = (
        station_module
        .build_frozen_base_cache(
            base,
            selection_loader,
            thresholds,
            device,
            ablation_module,
        )
    )

    base_validation_metrics = (
        ablation_module.metric_bundle(
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

    thresholds_tensor = torch.as_tensor(
        thresholds,
        dtype=torch.float32,
        device=device,
    )

    print(
        "\n=== Step 5B: strict Station-OOD "
        "CA-URC head training ==="
    )

    print(
        "Model                    : "
        "A4 underprediction-risk only"
    )

    print(
        "Correction               : "
        "p_under^gamma * Delta"
    )

    print(
        "TRAIN tail prevalence     : "
        f"{tail_prevalence[0]:.4f}/"
        f"{tail_prevalence[1]:.4f}"
    )

    print(
        "TRAIN U0.5 prevalence     : "
        f"{under_prevalence[0]:.4f}/"
        f"{under_prevalence[1]:.4f}"
    )

    print(
        "Frozen Base val overall   : "
        f"{base_validation_metrics['overall_mae_pga']:.4f}/"
        f"{base_validation_metrics['overall_mae_pgv']:.4f}"
    )

    print(
        "Frozen Base val bias      : "
        f"{base_validation_metrics['bias_pga']:+.4f}/"
        f"{base_validation_metrics['bias_pgv']:+.4f}"
    )

    print(
        "Frozen Base val tail      : "
        f"{base_validation_metrics['tail_mae_pga']:.4f}/"
        f"{base_validation_metrics['tail_mae_pgv']:.4f}"
    )

    history = []

    for epoch in range(
        1,
        args.head_epochs
        + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_metrics = (
            ablation_module.train_epoch(
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
        )

        head_now = (
            ablation_module
            .predict_head_from_cache(
                model,
                monitor_cache[
                    "risk_feature"
                ],
                device,
                args.head_eval_batch_size,
            )
        )

        monitor_prediction = (
            ablation_module
            .candidate_prediction_numpy(
                "A4_under_only",
                monitor_cache,
                head_now,
                gamma=1.0,
            )
        )

        monitor_metrics = (
            ablation_module.metric_bundle(
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
                "model_state": (
                    model.state_dict()
                ),
                "epoch": int(
                    epoch
                ),
                "variant": (
                    "A4_under_only"
                ),
                "model_name": (
                    "CA-URC"
                ),
                "station_ood_training": (
                    True
                ),
                "strict_seen_only_training": (
                    True
                ),
                "test_evaluated": (
                    False
                ),
                "tail_prevalence": (
                    tail_prevalence.tolist()
                ),
                "under_prevalence": (
                    under_prevalence.tolist()
                ),
                "monitor_metrics_gamma1": (
                    monitor_metrics
                ),
            },
            checkpoint_path,
        )

        history.append(
            {
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

    # -------------------------------------------------------------
    # VALIDATION epoch x gamma scan.
    # -------------------------------------------------------------
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
            ablation_module
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
                ablation_module
                .candidate_prediction_numpy(
                    "A4_under_only",
                    selection_cache,
                    head,
                    gamma=gamma,
                )
            )

            metrics = (
                ablation_module.metric_bundle(
                    selection_cache,
                    prediction,
                )
            )

            status = relative_bias_status(
                candidate=metrics,
                base=(
                    base_validation_metrics
                ),
                overall_budget=(
                    args.overall_budget
                ),
                non_tail_budget=(
                    args.non_tail_budget
                ),
                bias_margin=(
                    args.bias_margin
                ),
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
        head_dir
        / "epoch_gamma_validation_scan.csv"
    )

    scan.to_csv(
        scan_path,
        index=False,
    )

    diagnostic_columns = [
        "pass_overall_pga",
        "pass_non_tail_pga",
        "pass_relative_bias_pga",
        "pass_overall_pgv",
        "pass_non_tail_pgv",
        "pass_relative_bias_pgv",
    ]

    print(
        "\nValidation constraint pass counts:"
    )

    for column in diagnostic_columns:
        print(
            f"  {column:30s}: "
            f"{int(scan[column].sum())}/{len(scan)}"
        )

    print(
        f"  {'all_constraints':30s}: "
        f"{int(scan['feasible'].sum())}/{len(scan)}"
    )

    feasible = scan.loc[
        scan[
            "feasible"
        ].astype(
            bool
        )
    ].copy()

    if feasible.empty:
        failure = {
            "status": (
                "no_feasible_candidate"
            ),
            "model_name": (
                "CA-URC"
            ),
            "variant": (
                "A4_under_only"
            ),
            "base_validation_metrics": (
                base_validation_metrics
            ),
            "overall_budget": float(
                args.overall_budget
            ),
            "non_tail_budget": float(
                args.non_tail_budget
            ),
            "bias_margin": float(
                args.bias_margin
            ),
            "bias_rule": (
                "|bias_candidate| <= "
                "|bias_base| + bias_margin"
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
                failure,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\nNO Station-OOD CA-URC candidate "
            "satisfied all validation constraints."
        )

        print(
            "Station-OOD TEST: NOT ACCESSED"
        )

        return (
            None,
            failure,
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

    feasible.to_csv(
        head_dir
        / "feasible_validation_candidates.csv",
        index=False,
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

    selected_path = (
        head_dir
        / "selected_ca_urc_head.pt"
    )

    shutil.copy2(
        selected_source,
        selected_path,
    )

    selected_checkpoint = (
        load_checkpoint(
            selected_path,
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
        "model_name": (
            "CA-URC"
        ),
        "variant": (
            "A4_under_only"
        ),
        "formula": (
            "y_base + p_under^gamma * Delta"
        ),
        "selected_epoch": int(
            selected_epoch
        ),
        "selected_gamma": float(
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
                    float(value)
                    if isinstance(
                        value,
                        (
                            float,
                            np.floating,
                            int,
                            np.integer,
                        ),
                    )
                    else value
                )
            )
            for key, value
            in selected.to_dict().items()
        },
        "overall_budget": float(
            args.overall_budget
        ),
        "non_tail_budget": float(
            args.non_tail_budget
        ),
        "bias_margin": float(
            args.bias_margin
        ),
        "bias_rule": (
            "|bias_candidate| <= "
            "|bias_base| + bias_margin"
        ),
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

    print(
        "\n=== Selected Station-OOD CA-URC ==="
    )

    print(
        f"Epoch                         : "
        f"{selected_epoch}"
    )

    print(
        f"Gamma                         : "
        f"{selected_gamma:g}"
    )

    print(
        f"At upper gamma boundary       : "
        f"{upper_boundary}"
    )

    print(
        "Overall MAE PGA/PGV           : "
        f"{selected['overall_mae_pga']:.6f}/"
        f"{selected['overall_mae_pgv']:.6f}"
    )

    print(
        "Overall delta PGA/PGV         : "
        f"{selected['overall_delta_pga']:+.6f}/"
        f"{selected['overall_delta_pgv']:+.6f}"
    )

    print(
        "Non-tail delta PGA/PGV        : "
        f"{selected['non_tail_delta_pga']:+.6f}/"
        f"{selected['non_tail_delta_pgv']:+.6f}"
    )

    print(
        "Bias PGA/PGV                  : "
        f"{selected['bias_pga']:+.6f}/"
        f"{selected['bias_pgv']:+.6f}"
    )

    print(
        "Tail MAE PGA/PGV              : "
        f"{selected['tail_mae_pga']:.6f}/"
        f"{selected['tail_mae_pgv']:.6f}"
    )

    print(
        "Tail delta PGA/PGV            : "
        f"{selected['tail_delta_pga']:+.6f}/"
        f"{selected['tail_delta_pgv']:+.6f}"
    )

    print(
        "Tail U0.5 PGA/PGV             : "
        f"{selected['tail_under05_pga']:.4f}/"
        f"{selected['tail_under05_pgv']:.4f}"
    )

    print(
        f"Feasible candidates           : "
        f"{len(feasible)}/{len(scan)}"
    )

    if upper_boundary:
        print(
            "\nSelected gamma is at the upper "
            "VALIDATION grid boundary."
        )

        print(
            "Extend gamma using VALIDATION ONLY "
            "before evaluating Station-OOD test."
        )

        print(
            "Station-OOD TEST: NOT ACCESSED"
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
# Adapter: reuse exact paired Station-OOD evaluator from script 65
# ---------------------------------------------------------------------

class CAURCTestAdapter(
    nn.Module
):
    """
    Script 65's locked evaluator computes:

        correction =
            under_probability
            * tail_probability**gamma
            * residual

    To reuse its exact station-draw/test implementation for CA-URC, map:

        evaluator under_probability = 1
        evaluator tail_probability  = real p_under

    so:
        1 * real_p_under**gamma * residual

    is exactly CA-URC.

    After evaluation, the temporary "tail_probability" CSV columns are
    renamed to the true CA-URC underprediction probabilities.
    """

    def __init__(
        self,
        inner: nn.Module,
    ):
        super().__init__()

        self.inner = inner

    @property
    def base(
        self,
    ):
        return self.inner.base

    def head_from_feature(
        self,
        feature: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        head = (
            self.inner
            .head_from_feature(
                feature
            )
        )

        real_under = head[
            "under_probability"
        ]

        return {
            "under_probability": (
                torch.ones_like(
                    real_under
                )
            ),
            "tail_probability": (
                real_under
            ),
            "residual": (
                head[
                    "residual"
                ]
            ),
        }


def evaluate_station_ood_caurc(
    model: nn.Module,
    selected_gamma: float,
    thresholds: np.ndarray,
    scenario_path: Path,
    assignment_path: Path,
    ood_module,
    ablation_module,
    station_module,
    args,
    device: torch.device,
) -> pd.DataFrame:
    adapter = CAURCTestAdapter(
        model
    ).to(
        device
    )

    predictions = (
        station_module
        .evaluate_paired_station_ood_cadrg(
            model=adapter,
            selected_gamma=(
                selected_gamma
            ),
            thresholds=(
                thresholds
            ),
            scenario_csv=(
                scenario_path
            ),
            assignment_csv=(
                assignment_path
            ),
            ood_module=(
                ood_module
            ),
            risk_module=(
                ablation_module
            ),
            args=args,
            device=device,
        )
    )

    # Script 65 temporarily stores the real CA-URC p_under in its
    # tail_probability fields because of the exact evaluator adapter above.
    predictions[
        "under_probability_pga"
    ] = predictions[
        "tail_probability_pga"
    ].astype(
        float
    )

    predictions[
        "under_probability_pgv"
    ] = predictions[
        "tail_probability_pgv"
    ].astype(
        float
    )

    predictions[
        "gate_probability_pga"
    ] = np.power(
        np.clip(
            predictions[
                "under_probability_pga"
            ].to_numpy(
                float
            ),
            0.0,
            1.0,
        ),
        float(
            selected_gamma
        ),
    )

    predictions[
        "gate_probability_pgv"
    ] = np.power(
        np.clip(
            predictions[
                "under_probability_pgv"
            ].to_numpy(
                float
            ),
            0.0,
            1.0,
        ),
        float(
            selected_gamma
        ),
    )

    predictions = predictions.drop(
        columns=[
            "tail_probability_pga",
            "tail_probability_pgv",
        ]
    )

    predictions[
        "model_name"
    ] = (
        "CA-URC"
    )

    predictions[
        "model_variant"
    ] = (
        "A4_under_only"
    )

    return predictions


# ---------------------------------------------------------------------
# Rename old generic CA-DRG metric labels to final CA-URC terminology
# ---------------------------------------------------------------------

def caurc_metric_table(
    station_module,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    table = (
        station_module
        .station_ood_metric_table(
            predictions
        )
    )

    table[
        "model"
    ] = table[
        "model"
    ].replace(
        {
            "ca_drg": (
                "ca_urc"
            ),
        }
    )

    return table


def caurc_seen_unseen_deltas(
    station_module,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    table = (
        station_module
        .paired_seen_unseen_deltas(
            predictions
        )
    )

    table[
        "model"
    ] = table[
        "model"
    ].replace(
        {
            "ca_drg": (
                "ca_urc"
            ),
        }
    )

    return table


def caurc_minus_base_by_role(
    station_module,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    table = (
        station_module
        .cadrg_minus_base_by_role(
            predictions
        )
    )

    rename = {
        "ca_drg_value": (
            "ca_urc_value"
        ),
        "mean_delta_cadrg_minus_base": (
            "mean_delta_caurc_minus_base"
        ),
        "median_delta_cadrg_minus_base": (
            "median_delta_caurc_minus_base"
        ),
    }

    table = table.rename(
        columns=rename
    )

    return table


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
        "--protocol-module",
        default=(
            "52_phase2_train_and_evaluate_station_ood_attention.py"
        ),
    )

    parser.add_argument(
        "--ablation-module",
        default=(
            "63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py"
        ),
    )

    parser.add_argument(
        "--station-module",
        default=(
            "65_train_and_evaluate_cadrg_station_ood.py"
        ),
    )

    parser.add_argument(
        "--scenario",
        default=(
            "data/model_manifests/station_ood/"
            "scenario_station_ood.csv"
        ),
    )

    parser.add_argument(
        "--station-assignment",
        default=(
            "data/model_manifests/station_ood/"
            "station_holdout_assignment.csv"
        ),
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
        default=(
            "runs/cadrg_station_ood/"
            "cross_attention_base/best_model.pt"
        ),
        help=(
            "Previously trained strict Station-OOD Cross-Attention Base. "
            "Reused only when present and marked station_ood_training=True."
        ),
    )

    parser.add_argument(
        "--retrain-base",
        action="store_true",
        help=(
            "Force retraining of the strict Station-OOD Cross-Attention Base."
        ),
    )

    # Task
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
        "--input-pre-sec",
        type=float,
        default=2.0,
    )

    # Architecture
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

    # Base training
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
        "--base-minimum-learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--base-minimum-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--base-early-stopping-patience",
        type=int,
        default=12,
    )

    # Threshold / prevalence
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

    parser.add_argument(
        "--prevalence-repeats",
        type=int,
        default=3,
    )

    # Head training
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
        "--head-eval-batch-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--lambda-tail-classification",
        type=float,
        default=0.25,
        help=(
            "Ignored by A4 because it has no tail classifier."
        ),
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

    # Validation lock
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
        "--bias-margin",
        type=float,
        default=0.005,
    )

    # Test
    parser.add_argument(
        "--test-repeats",
        type=int,
        default=20,
    )

    # Runtime
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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
            "runs/caurc_station_ood"
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

    scenario_input = Path(
        args.scenario
    )

    assignment_path = Path(
        args.station_assignment
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

    if not scenario_input.exists():
        raise FileNotFoundError(
            scenario_input
        )

    if not assignment_path.exists():
        raise FileNotFoundError(
            assignment_path
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_module = load_module(
        args.baseline_module,
        "caurc_station_ood_baseline_module",
    )

    ood_module = load_module(
        args.protocol_module,
        "caurc_station_ood_protocol_module",
    )

    ablation_module = load_module(
        args.ablation_module,
        "caurc_station_ood_ablation_module",
    )

    station_module = load_module(
        args.station_module,
        "caurc_station_ood_existing_module",
    )

    station_module.set_seed(
        args.seed
    )

    resolved_scenario_path = (
        out_dir
        / "resolved_scenario_station_ood.csv"
    )

    scenario_frame = (
        station_module
        .resolve_scenario_for_local_machine(
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

    print(
        "=== CA-URC Strict Station OOD ==="
    )

    print(
        f"Device                    : "
        f"{device}"
    )

    print(
        f"Scenario events           : "
        f"{len(scenario_frame)}"
    )

    print(
        f"Station roles             : "
        f"{station_role_counts}"
    )

    print(
        "Train/Val OOD stations   : EXCLUDED"
    )

    print(
        "Final model              : "
        "CA-URC = Cross-Attention "
        "Underprediction-Risk Correction"
    )

    print(
        "Formula                  : "
        "y_final = y_CA + p_under^gamma * Delta"
    )

    start_time = time.time()

    # -------------------------------------------------------------
    # 5A: strict Station-OOD Cross-Attention Base.
    # -------------------------------------------------------------
    base, base_summary = get_station_ood_base(
        args,
        baseline_module,
        ood_module,
        station_module,
        resolved_scenario_path,
        assignment_path,
        device,
        out_dir,
    )

    # -------------------------------------------------------------
    # TRAIN seen-only Q90, unchanged definition from earlier OOD.
    # -------------------------------------------------------------
    thresholds, threshold_info = (
        station_module
        .estimate_seen_only_train_thresholds(
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
        "PGA log10 threshold       : "
        f"{thresholds[0]:.6f}"
    )

    print(
        "PGV log10 threshold       : "
        f"{thresholds[1]:.6f}"
    )

    print(
        "Threshold target rows     : "
        f"{threshold_info['n_train_target_rows']:,}"
    )

    # -------------------------------------------------------------
    # 5B: train CA-URC head and lock using validation only.
    # -------------------------------------------------------------
    model, selection = train_and_select_caurc(
        args,
        ood_module,
        ablation_module,
        station_module,
        base,
        resolved_scenario_path,
        assignment_path,
        thresholds,
        device,
        out_dir,
    )

    if model is None:
        summary = {
            "status": (
                "stopped_before_test"
            ),
            "base_summary": (
                base_summary
            ),
            "ca_urc_selection": (
                selection
            ),
            "test_accessed": (
                False
            ),
            "elapsed_seconds": float(
                time.time()
                - start_time
            ),
        }

        (
            out_dir
            / "run_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        return

    selected_gamma = float(
        selection[
            "selected_gamma"
        ]
    )

    # -------------------------------------------------------------
    # 5C: locked paired seen vs unseen Station-OOD test.
    # -------------------------------------------------------------
    predictions = (
        evaluate_station_ood_caurc(
            model=model,
            selected_gamma=(
                selected_gamma
            ),
            thresholds=(
                thresholds
            ),
            scenario_path=(
                resolved_scenario_path
            ),
            assignment_path=(
                assignment_path
            ),
            ood_module=(
                ood_module
            ),
            ablation_module=(
                ablation_module
            ),
            station_module=(
                station_module
            ),
            args=args,
            device=device,
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

    metrics = caurc_metric_table(
        station_module,
        predictions,
    )

    metrics_path = (
        out_dir
        / "station_ood_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    paired = caurc_seen_unseen_deltas(
        station_module,
        predictions,
    )

    paired_path = (
        out_dir
        / "station_ood_paired_event_deltas.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    gate_effect = caurc_minus_base_by_role(
        station_module,
        predictions,
    )

    gate_effect_path = (
        out_dir
        / "caurc_minus_base_by_target_role.csv"
    )

    gate_effect.to_csv(
        gate_effect_path,
        index=False,
    )

    distances = (
        station_module
        .distance_summary(
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

    prevalence = (
        station_module
        .tail_prevalence_table(
            predictions
        )
    )

    prevalence_path = (
        out_dir
        / "station_ood_tail_prevalence.csv"
    )

    prevalence.to_csv(
        prevalence_path,
        index=False,
    )

    # Mark the selection lock as test-evaluated only now.
    selection[
        "test_evaluated"
    ] = True

    selection[
        "test_prediction_path"
    ] = str(
        prediction_path.resolve()
    )

    (
        out_dir
        / "ca_urc_head"
        / "selected_epoch_gamma.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
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
        "strict_station_ood": (
            True
        ),
        "final_model_name": (
            "CA-URC"
        ),
        "final_model_variant": (
            "A4_under_only"
        ),
        "formula": (
            "y_base + p_under^gamma * Delta"
        ),
        "ood_station_train_input_use": (
            False
        ),
        "ood_station_train_target_use": (
            False
        ),
        "ood_station_validation_input_use": (
            False
        ),
        "ood_station_validation_target_use": (
            False
        ),
        "paired_test_rule": (
            "same event + same repeat + same seen inputs + "
            "equal seen/unseen target budgets"
        ),
        "metric_aggregation": (
            "targets -> repeats -> events"
        ),
        "tail_threshold_rule": (
            "Q90 from TRAIN seen-only target draws only"
        ),
        "validation_bias_guard": (
            "|bias_candidate| <= |bias_base| + 0.005"
        ),
        "test_used_for_base_selection": (
            False
        ),
        "test_used_for_head_selection": (
            False
        ),
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

    summary = {
        "status": (
            "completed"
        ),
        "base_summary": (
            base_summary
        ),
        "ca_urc_selection": (
            selection
        ),
        "test_events": int(
            predictions[
                "event_id"
            ].nunique()
        ),
        "test_rows": int(
            len(
                predictions
            )
        ),
        "test_accessed_after_lock": (
            True
        ),
        "elapsed_seconds": float(
            time.time()
            - start_time
        ),
    }

    (
        out_dir
        / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Station-OOD canonical metrics ==="
    )

    print(
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
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Unseen minus seen paired event deltas ==="
    )

    print(
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
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== CA-URC minus Base within each target role ==="
    )

    print(
        gate_effect.loc[
            gate_effect[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                ]
            )
        ].to_string(
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
        "\n=== Tail prevalence ==="
    )

    print(
        prevalence.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    for path in [
        out_dir
        / "cross_attention_base"
        / "best_model.pt",
        threshold_path,
        out_dir
        / "ca_urc_head"
        / "epoch_gamma_validation_scan.csv",
        out_dir
        / "ca_urc_head"
        / "selected_ca_urc_head.pt",
        out_dir
        / "ca_urc_head"
        / "selected_epoch_gamma.json",
        prediction_path,
        metrics_path,
        paired_path,
        gate_effect_path,
        distance_path,
        prevalence_path,
        out_dir
        / "run_configuration.json",
        out_dir
        / "run_summary.json",
    ]:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
