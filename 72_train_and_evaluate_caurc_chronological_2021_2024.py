#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
72_train_and_evaluate_caurc_chronological_2021_2024.py

Strict chronological extrapolation of the final candidate CA-URC to 2021-2024.

CA-URC
------
Cross-Attention Underprediction-Risk Correction

    y_final = y_CA + p_under**gamma * Delta

Chronological protocol
----------------------
TRAIN:
    2010-2018

VALIDATION:
    2019 non-Ridgecrest + 2020

LOCKED TEST:
    2021-2024

Ridgecrest:
    NOT ACCESSED by this script.

Model-selection protocol
------------------------
1. Retrain Cross-Attention Base from chronological TRAIN only.
2. Select Base checkpoint using chronological VALIDATION only.
3. Use the existing chronology-safe TRAIN-only Q90 threshold JSON when
   available. If it is unavailable, compute Q90 using TRAIN only.
4. Freeze Cross-Attention Base.
5. Train only CA-URC (A4_under_only) on chronological TRAIN.
6. Jointly select head epoch + gamma on 20-repeat chronological VALIDATION.
7. Shift-aware preservation constraints:
       overall MAE increase <= 0.005 for PGA and PGV
       non-tail MAE increase <= 0.003 for PGA and PGV
       |bias_candidate| <= |bias_base| + 0.005 for PGA and PGV
8. Among feasible candidates:
       minimum mean tail MAE
       -> minimum mean tail U0.5
       -> minimum mean overall MAE
       -> lower gamma
       -> earlier epoch
9. If selected gamma is at the upper grid boundary, STOP before test.
10. Otherwise freeze everything and evaluate 2021-2024 ONCE.

No 2021-2024 event is instantiated before validation selection is locked.
The complete Ridgecrest sequence is never accessed by this script.

Canonical aggregation
---------------------
targets -> repeats -> events

Required local modules
----------------------
45_phase2_strong_baseline_suite.py
63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py
68_train_and_evaluate_cadrg_chronological_ridgecrest.py

Script 68 is used only for its already-tested chronological Cross-Attention
training, dataset, metric, and relative-bias helper functions. Its CA-DRG
training/evaluation path is NOT called.
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


# ---------------------------------------------------------------------
# TRAIN-only Q90 threshold
# ---------------------------------------------------------------------

def load_or_compute_thresholds(
    args,
    baseline_module,
    chrono_module,
    device: torch.device,
    out_dir: Path,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    source = Path(
        args.threshold_json
    )

    if source.exists():
        payload = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )

        required = {
            "log10_pga_threshold",
            "log10_pgv_threshold",
        }

        missing = required.difference(
            payload
        )

        if missing:
            raise ValueError(
                f"Threshold JSON missing keys: {sorted(missing)}"
            )

        thresholds = np.asarray(
            [
                payload[
                    "log10_pga_threshold"
                ],
                payload[
                    "log10_pgv_threshold"
                ],
            ],
            dtype=np.float32,
        )

        info = {
            **payload,
            "threshold_reused": True,
            "source_json": str(
                source.resolve()
            ),
            "validation_labels_used": False,
            "test_labels_used": False,
            "ridgecrest_labels_used": False,
        }

        print(
            "\nReusing existing chronology-safe "
            "TRAIN-only Q90 thresholds."
        )

    else:
        print(
            "\nWARNING: existing chronological threshold JSON "
            "was not found."
        )

        print(
            "Computing replacement Q90 thresholds using "
            "chronological TRAIN ONLY."
        )

        dataset = chrono_module.make_dataset(
            baseline_module,
            args,
            args.train_label,
            training=False,
            repeats=args.threshold_repeats,
        )

        loader = chrono_module.make_loader(
            dataset,
            args,
            device,
            shuffle=False,
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
                "chronological TRAIN target draws only"
            ),
            "threshold_reused": False,
            "validation_labels_used": False,
            "test_labels_used": False,
            "ridgecrest_labels_used": False,
        }

    output = (
        out_dir
        / "chronological_tail_thresholds.json"
    )

    output.write_text(
        json.dumps(
            info,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        thresholds,
        info,
    )


# ---------------------------------------------------------------------
# CA-URC train / validation selection
# ---------------------------------------------------------------------

def train_and_select_caurc(
    args,
    baseline_module,
    ablation_module,
    chrono_module,
    base: nn.Module,
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
    # TRAIN prevalence.
    # -------------------------------------------------------------
    prevalence_dataset = (
        chrono_module.make_dataset(
            baseline_module,
            args,
            args.train_label,
            training=False,
            repeats=args.prevalence_repeats,
        )
    )

    prevalence_loader = (
        chrono_module.make_loader(
            prevalence_dataset,
            args,
            device,
            shuffle=False,
        )
    )

    prevalence_cache = (
        ablation_module.build_eval_cache(
            base,
            prevalence_loader,
            thresholds,
            device,
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

    # A4 was the second ablation variant; retain the same independent
    # initialization-stream convention.
    variant_seed = (
        args.seed
        + 2000
    )

    ablation_module.set_seed(
        variant_seed
    )

    # Match the TRAIN / VALIDATION draw convention used by A4 development.
    ablation_module.patch_train_validation_seed_protocol(
        baseline_module
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

    train_dataset = (
        chrono_module.make_dataset(
            baseline_module,
            args,
            args.train_label,
            training=True,
            repeats=1,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(
        variant_seed
    )

    train_loader = (
        chrono_module.make_loader(
            train_dataset,
            args,
            device,
            shuffle=True,
            generator=generator,
        )
    )

    # 3-repeat monitor.
    monitor_dataset = (
        chrono_module.make_dataset(
            baseline_module,
            args,
            args.validation_label,
            training=False,
            repeats=args.head_monitor_repeats,
        )
    )

    monitor_loader = (
        chrono_module.make_loader(
            monitor_dataset,
            args,
            device,
            shuffle=False,
        )
    )

    monitor_cache = (
        ablation_module.build_eval_cache(
            model.base,
            monitor_loader,
            thresholds,
            device,
        )
    )

    # 20-repeat selection cache.
    selection_dataset = (
        chrono_module.make_dataset(
            baseline_module,
            args,
            args.validation_label,
            training=False,
            repeats=args.head_selection_repeats,
        )
    )

    selection_loader = (
        chrono_module.make_loader(
            selection_dataset,
            args,
            device,
            shuffle=False,
        )
    )

    selection_cache = (
        ablation_module.build_eval_cache(
            model.base,
            selection_loader,
            thresholds,
            device,
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

    history = []

    print(
        "\n=== Step 6B: chronological "
        "CA-URC head training ==="
    )

    print(
        "Formula                    : "
        "y_final = y_CA + p_under^gamma * Delta"
    )

    print(
        "TRAIN tail prevalence       : "
        f"{tail_prevalence[0]:.4f}/"
        f"{tail_prevalence[1]:.4f}"
    )

    print(
        "TRAIN U0.5 prevalence       : "
        f"{under_prevalence[0]:.4f}/"
        f"{under_prevalence[1]:.4f}"
    )

    print(
        "Frozen Base val overall     : "
        f"{base_validation_metrics['overall_mae_pga']:.4f}/"
        f"{base_validation_metrics['overall_mae_pgv']:.4f}"
    )

    print(
        "Frozen Base val bias        : "
        f"{base_validation_metrics['bias_pga']:+.4f}/"
        f"{base_validation_metrics['bias_pgv']:+.4f}"
    )

    print(
        "Frozen Base val tail        : "
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
                "variant": (
                    "A4_under_only"
                ),
                "model_name": (
                    "CA-URC"
                ),
                "model_state": (
                    model.state_dict()
                ),
                "epoch": int(
                    epoch
                ),
                "chronological_training": (
                    True
                ),
                "strict_train_years": (
                    "2010-2018"
                ),
                "validation_years": (
                    "2019 non-Ridgecrest + 2020"
                ),
                "test_accessed": (
                    False
                ),
                "ridgecrest_accessed": (
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

            status = (
                chrono_module.relative_bias_status(
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

    feasible = scan.loc[
        scan[
            "feasible"
        ].astype(
            bool
        )
    ].copy()

    print(
        "\nChronological validation feasibility:"
    )

    print(
        f"  feasible candidates : "
        f"{len(feasible)}/{len(scan)}"
    )

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
            "test_accessed": (
                False
            ),
            "ridgecrest_accessed": (
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
            "\nNO chronological CA-URC candidate "
            "satisfied all validation constraints."
        )

        print(
            "2021-2024 TEST : NOT ACCESSED"
        )

        print(
            "Ridgecrest OOD : NOT ACCESSED"
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

    selected_checkpoint = load_checkpoint(
        selected_path,
        device,
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
            "y_CA + p_under^gamma * Delta"
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
        "test_accessed": (
            False
        ),
        "ridgecrest_accessed": (
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
        "\n=== Selected chronological CA-URC ==="
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

    if upper_boundary:
        print(
            "\nSelected gamma is at the upper "
            "VALIDATION grid boundary."
        )

        print(
            "Extend gamma using VALIDATION ONLY."
        )

        print(
            "2021-2024 TEST : NOT ACCESSED"
        )

        print(
            "Ridgecrest OOD : NOT ACCESSED"
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
# Locked 2021-2024 inference
# ---------------------------------------------------------------------

def evaluate_locked_test(
    args,
    baseline_module,
    ablation_module,
    chrono_module,
    model: nn.Module,
    selected_gamma: float,
    thresholds: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    # Only now instantiate the locked test split.
    test_dataset = (
        chrono_module.make_dataset(
            baseline_module,
            args,
            args.test_label,
            training=False,
            repeats=args.test_repeats,
        )
    )

    test_loader = (
        chrono_module.make_loader(
            test_dataset,
            args,
            device,
            shuffle=False,
        )
    )

    rows = []

    model.eval()

    with torch.inference_mode():
        for batch in test_loader:
            truth = batch[
                "target_log"
            ].numpy()

            (
                base_prediction,
                risk_feature,
            ) = (
                ablation_module
                .cross_attention_base_context(
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
            )

            head = (
                model.head_from_feature(
                    risk_feature
                )
            )

            gate = torch.pow(
                torch.clamp(
                    head[
                        "under_probability"
                    ],
                    0.0,
                    1.0,
                ),
                float(
                    selected_gamma
                ),
            )

            correction = (
                gate
                * head[
                    "residual"
                ]
            )

            final_prediction = (
                base_prediction
                + correction
            )

            base_np = (
                base_prediction
                .cpu()
                .numpy()
            )

            final_np = (
                final_prediction
                .cpu()
                .numpy()
            )

            under_np = (
                head[
                    "under_probability"
                ]
                .cpu()
                .numpy()
            )

            residual_np = (
                head[
                    "residual"
                ]
                .cpu()
                .numpy()
            )

            correction_np = (
                correction
                .cpu()
                .numpy()
            )

            repeats = (
                batch[
                    "repeat"
                ]
                .numpy()
            )

            target_indices = (
                batch[
                    "target_station_index"
                ]
                .numpy()
            )

            target_slots = (
                batch[
                    "target_slot"
                ]
                .numpy()
            )

            event_ids = list(
                batch[
                    "event_id"
                ]
            )

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
                    true_pga = float(
                        truth[
                            b,
                            q,
                            0,
                        ]
                    )

                    true_pgv = float(
                        truth[
                            b,
                            q,
                            1,
                        ]
                    )

                    rows.append(
                        {
                            "split": (
                                args.test_label
                            ),
                            "event_id": str(
                                event_ids[
                                    b
                                ]
                            ),
                            "repeat": int(
                                repeats[
                                    b
                                ]
                            ),
                            "target_slot": int(
                                target_slots[
                                    b,
                                    q,
                                ]
                            ),
                            "target_station_index": int(
                                target_indices[
                                    b,
                                    q,
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
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "base_log10_pgv": float(
                                base_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "final_log10_pga": float(
                                final_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "final_log10_pgv": float(
                                final_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "under_probability_pga": float(
                                under_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "under_probability_pgv": float(
                                under_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "correction_capacity_pga": float(
                                residual_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "correction_capacity_pgv": float(
                                residual_np[
                                    b,
                                    q,
                                    1,
                                ]
                            ),
                            "applied_correction_pga": float(
                                correction_np[
                                    b,
                                    q,
                                    0,
                                ]
                            ),
                            "applied_correction_pgv": float(
                                correction_np[
                                    b,
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
                            "model_name": (
                                "CA-URC"
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Event-level paired bootstrap after test is locked
# ---------------------------------------------------------------------

def paired_event_bootstrap(
    predictions: pd.DataFrame,
    chrono_module,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        seed
    )

    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        for population in (
            "overall",
            "high_motion_tail",
        ):
            subset = predictions

            if (
                population
                == "high_motion_tail"
            ):
                subset = predictions.loc[
                    chrono_module.bool_array(
                        predictions[
                            f"is_tail_{quantity}"
                        ]
                    )
                ].copy()

            for metric in (
                "mae",
                "bias",
                "factor2",
                "under05",
            ):
                base = (
                    chrono_module.event_values(
                        subset,
                        f"true_log10_{quantity}",
                        f"base_log10_{quantity}",
                        metric,
                    )
                )

                final = (
                    chrono_module.event_values(
                        subset,
                        f"true_log10_{quantity}",
                        f"final_log10_{quantity}",
                        metric,
                    )
                )

                common = (
                    base.index
                    .intersection(
                        final.index
                    )
                )

                base = base.loc[
                    common
                ]

                final = final.loc[
                    common
                ]

                delta = (
                    final
                    - base
                ).to_numpy(
                    dtype=float
                )

                n = len(
                    delta
                )

                if n == 0:
                    continue

                sample_index = rng.integers(
                    0,
                    n,
                    size=(
                        repetitions,
                        n,
                    ),
                )

                distribution = delta[
                    sample_index
                ].mean(
                    axis=1
                )

                low = float(
                    np.quantile(
                        distribution,
                        0.025,
                    )
                )

                high = float(
                    np.quantile(
                        distribution,
                        0.975,
                    )
                )

                prob_le = (
                    np.sum(
                        distribution <= 0
                    )
                    + 1
                ) / (
                    repetitions + 1
                )

                prob_ge = (
                    np.sum(
                        distribution >= 0
                    )
                    + 1
                ) / (
                    repetitions + 1
                )

                p_two = min(
                    1.0,
                    2.0
                    * min(
                        prob_le,
                        prob_ge,
                    ),
                )

                base_value = float(
                    base.mean()
                )

                final_value = float(
                    final.mean()
                )

                point_delta = float(
                    delta.mean()
                )

                record = {
                    "quantity": (
                        quantity
                    ),
                    "population": (
                        population
                    ),
                    "metric": (
                        metric
                    ),
                    "reference": (
                        "cross_attention_base"
                    ),
                    "candidate": (
                        "ca_urc"
                    ),
                    "reference_value": (
                        base_value
                    ),
                    "candidate_value": (
                        final_value
                    ),
                    "point_delta_candidate_minus_reference": (
                        point_delta
                    ),
                    "ci95_low": (
                        low
                    ),
                    "ci95_high": (
                        high
                    ),
                    "ci_excludes_zero": bool(
                        (
                            low > 0
                        )
                        or (
                            high < 0
                        )
                    ),
                    "two_sided_bootstrap_sign_p": float(
                        p_two
                    ),
                    "n_paired_events": int(
                        n
                    ),
                    "bootstrap_repetitions": int(
                        repetitions
                    ),
                }

                if metric == "mae":
                    record[
                        "relative_change_percent"
                    ] = float(
                        100.0
                        * point_delta
                        / max(
                            abs(
                                base_value
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
                        * point_delta
                    )

                rows.append(
                    record
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
        "--baseline-module",
        default=(
            "45_phase2_strong_baseline_suite.py"
        ),
    )

    parser.add_argument(
        "--ablation-module",
        default=(
            "63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py"
        ),
    )

    parser.add_argument(
        "--chrono-module",
        default=(
            "68_train_and_evaluate_cadrg_chronological_ridgecrest.py"
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/model_manifests/"
            "scenario_t0_5s_k5_time_clean.csv"
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
        default="split_time_clean",
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
        "--threshold-json",
        default=(
            "runs/"
            "time_clean_mean_base_t0_5s_k5/"
            "tail_thresholds_q0.90_t0_5s.json"
        ),
    )

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
        "--target-stations",
        type=int,
        default=10,
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

    # Cross-Attention Base
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
        "--base-minimum-learning-rate",
        type=float,
        default=1e-5,
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

    # CA-URC head
    parser.add_argument(
        "--prevalence-repeats",
        type=int,
        default=3,
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
        "--head-eval-batch-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--lambda-tail-classification",
        type=float,
        default=0.25,
        help=(
            "Ignored by A4 because there is no tail classifier."
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

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=10000,
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
            "runs/"
            "caurc_chronological_2021_2024"
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

    manifest_path = Path(
        args.manifest
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    baseline_module = load_module(
        args.baseline_module,
        "caurc_chrono_baseline_module",
    )

    ablation_module = load_module(
        args.ablation_module,
        "caurc_chrono_ablation_module",
    )

    chrono_module = load_module(
        args.chrono_module,
        "caurc_chrono_helper_module",
    )

    chrono_module.set_seed(
        args.seed
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "event_id": str,
        },
    )

    split_counts = (
        manifest[
            args.split_column
        ]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    print(
        "=== CA-URC Chronological Extrapolation "
        "to 2021-2024 ==="
    )

    print(
        f"Device                    : "
        f"{device}"
    )

    print(
        f"Manifest split counts     : "
        f"{split_counts}"
    )

    print(
        "TRAIN                     : 2010-2018"
    )

    print(
        "VALIDATION                : "
        "2019 non-Ridgecrest + 2020"
    )

    print(
        "LOCKED TEST               : 2021-2024"
    )

    print(
        "Ridgecrest OOD            : NOT ACCESSED"
    )

    start_time = time.time()

    # -------------------------------------------------------------
    # 6A: chronological Cross-Attention Base.
    # -------------------------------------------------------------
    base, base_summary = (
        chrono_module
        .train_chronological_base(
            baseline_module,
            args,
            device,
            out_dir,
        )
    )

    # -------------------------------------------------------------
    # TRAIN-only Q90.
    # -------------------------------------------------------------
    thresholds, threshold_info = (
        load_or_compute_thresholds(
            args,
            baseline_module,
            chrono_module,
            device,
            out_dir,
        )
    )

    print(
        "\nChronological Q90 PGA/PGV : "
        f"{thresholds[0]:.6f}/"
        f"{thresholds[1]:.6f}"
    )

    # -------------------------------------------------------------
    # 6B: CA-URC head + VALIDATION lock.
    # -------------------------------------------------------------
    model, selection = (
        train_and_select_caurc(
            args,
            baseline_module,
            ablation_module,
            chrono_module,
            base,
            thresholds,
            device,
            out_dir,
        )
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
            "ridgecrest_accessed": (
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
    # 6C: only now access 2021-2024.
    # -------------------------------------------------------------
    print(
        "\n=== LOCKED 2021-2024 chronological test ==="
    )

    test_predictions = (
        evaluate_locked_test(
            args,
            baseline_module,
            ablation_module,
            chrono_module,
            model,
            selected_gamma,
            thresholds,
            device,
        )
    )

    prediction_path = (
        out_dir
        / "chronological_2021_2024_predictions.csv"
    )

    test_predictions.to_csv(
        prediction_path,
        index=False,
    )

    metrics = (
        chrono_module.metric_table(
            test_predictions
        )
    )

    metrics[
        "model"
    ] = metrics[
        "model"
    ].replace(
        {
            "ca_drg": (
                "ca_urc"
            ),
        }
    )

    metrics_path = (
        out_dir
        / "chronological_2021_2024_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    deltas = (
        chrono_module
        .cadrg_minus_base_table(
            test_predictions
        )
    )

    deltas = deltas.rename(
        columns={
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
    )

    delta_path = (
        out_dir
        / "caurc_minus_cross_attention_2021_2024.csv"
    )

    deltas.to_csv(
        delta_path,
        index=False,
    )

    bootstrap = paired_event_bootstrap(
        test_predictions,
        chrono_module,
        repetitions=(
            args.bootstrap_repetitions
        ),
        seed=(
            args.seed
            + 9001
        ),
    )

    bootstrap_path = (
        out_dir
        / "paired_event_bootstrap_2021_2024.csv"
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    selection[
        "test_accessed"
    ] = True

    selection[
        "test_prediction_path"
    ] = str(
        prediction_path.resolve()
    )

    selection[
        "ridgecrest_accessed"
    ] = False

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
        "split_counts": (
            split_counts
        ),
        "split_definition": {
            "train": (
                "2010-2018"
            ),
            "validation": (
                "2019 non-Ridgecrest + 2020"
            ),
            "test": (
                "2021-2024"
            ),
            "ridgecrest_ood": (
                "not accessed in this script"
            ),
        },
        "final_model_name": (
            "CA-URC"
        ),
        "final_model_variant": (
            "A4_under_only"
        ),
        "formula": (
            "y_CA + p_under^gamma * Delta"
        ),
        "tail_threshold_info": (
            threshold_info
        ),
        "selection_rule": {
            "overall_delta_max": float(
                args.overall_budget
            ),
            "non_tail_delta_max": float(
                args.non_tail_budget
            ),
            "relative_bias_margin": float(
                args.bias_margin
            ),
            "relative_bias_rule": (
                "|bias_candidate| <= "
                "|bias_base| + margin"
            ),
        },
        "test_used_for_model_selection": (
            False
        ),
        "ridgecrest_accessed": (
            False
        ),
        "metric_aggregation": (
            "targets -> repeats -> events"
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
            test_predictions[
                "event_id"
            ].nunique()
        ),
        "test_rows": int(
            len(
                test_predictions
            )
        ),
        "ridgecrest_accessed": (
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

    print(
        "\n=== 2021-2024 canonical metrics ==="
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
        ][
            [
                "model",
                "quantity",
                "population",
                "metric",
                "value",
                "n_events",
                "n_target_rows",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== CA-URC minus Cross-Attention Base ==="
    )

    print(
        deltas.loc[
            deltas[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                    "factor2",
                ]
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Paired event bootstrap: CA-URC vs Base ==="
    )

    print(
        bootstrap.loc[
            bootstrap[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                    "factor2",
                ]
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\nRidgecrest OOD: NOT ACCESSED"
    )

    print(
        "\nOutputs:"
    )

    for path in [
        out_dir
        / "cross_attention_base"
        / "best_model.pt",
        out_dir
        / "chronological_tail_thresholds.json",
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
        delta_path,
        bootstrap_path,
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
