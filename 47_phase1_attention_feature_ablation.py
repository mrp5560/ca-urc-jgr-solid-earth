#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
47_phase1_attention_feature_ablation.py

Validation-only feature ablation for the FINAL attention-pooling architecture.

Why this script exists
----------------------
The previous Phase-1 experiment showed that attention pooling outperformed
mean pooling on the validation set. Therefore, feature ablations must now be
re-run on the attention architecture rather than on the old mean-pooling base.

Variants trained here
---------------------
attention_no_waveform
    Remove waveform information; retain station geometry and P-offset.

attention_no_geometry
    Retain waveform and P-offset; remove input-station geometry and all
    target-coordinate features.

attention_no_p_offset
    Retain waveform and geometry; remove explicit relative P-arrival offset.

Reference model
---------------
attention_full
    The already-trained "attention_pooling" checkpoint produced by
    40_phase1_final_mean_pooling_ablation_suite.py.

Scientific protocol
-------------------
1. New variants use exactly the same dataset construction, station sampling,
   training hyperparameters, split, T0, K, Q, and random seed as script 40.
2. Checkpoint selection for each new variant uses validation only.
3. Final architecture decision is made using a 20-repeat validation audit.
4. This script NEVER instantiates or evaluates the test split.
5. Main aggregation:
       targets -> repeats -> events.

After this script:
- if attention_no_p_offset is not worse than attention_full on both overall
  and high-motion validation metrics, P-offset can be removed;
- otherwise retain P-offset.
- waveform / geometry results become the final architecture ablation evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


LOG10_FACTOR_2 = math.log10(2.0)

VALID_VARIANTS = {
    "attention_no_waveform",
    "attention_no_geometry",
    "attention_no_p_offset",
}


def load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "phase1_script40_module",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AttentionFeatureAblationModel(nn.Module):
    """
    Same station-attention architecture as script-40 attention_pooling,
    changing only the requested information source.
    """

    def __init__(
        self,
        base_module,
        variant: str,
        hidden_dim: int = 128,
    ):
        super().__init__()

        if variant not in VALID_VARIANTS:
            raise ValueError(
                "Unknown variant: %s" % variant
            )

        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.use_waveform = (
            variant != "attention_no_waveform"
        )

        if self.use_waveform:
            self.waveform_encoder = (
                base_module.WaveformEncoder(
                    hidden_dim
                )
            )
            station_input_dim = (
                hidden_dim + 4
            )
        else:
            self.waveform_encoder = None
            station_input_dim = 4

        self.station_encoder = nn.Sequential(
            nn.Linear(
                station_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        # Exactly the same lightweight station-attention pooling
        # used by script-40 "attention_pooling".
        self.attention_score = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

        self.query_decoder = nn.Sequential(
            nn.Linear(
                hidden_dim + 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def _apply_ablation(
        self,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        station_features = (
            input_features
        )
        query_features = (
            target_features
        )

        if (
            self.variant
            == "attention_no_geometry"
        ):
            station_features = (
                station_features.clone()
            )

            # input_features:
            # [local_x, local_y, third_station_coord, p_offset/T0]
            # remove geometry while retaining P-offset.
            station_features[
                ..., :3
            ] = 0.0

            # target_features:
            # [local_x, local_y, third_station_coord]
            query_features = (
                torch.zeros_like(
                    query_features
                )
            )

        elif (
            self.variant
            == "attention_no_p_offset"
        ):
            station_features = (
                station_features.clone()
            )
            station_features[
                ..., 3
            ] = 0.0

        return (
            station_features,
            query_features,
        )

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:

        (
            station_features,
            query_features,
        ) = self._apply_ablation(
            input_features,
            target_features,
        )

        if self.use_waveform:
            (
                batch_size,
                station_count,
                channel_count,
                sample_count,
            ) = input_waveforms.shape

            waveform_latent = (
                self.waveform_encoder(
                    input_waveforms.reshape(
                        batch_size
                        * station_count,
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

            station_input = torch.cat(
                [
                    waveform_latent,
                    station_features,
                ],
                dim=-1,
            )
        else:
            station_input = (
                station_features
            )

        station_latent = (
            self.station_encoder(
                station_input
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

        expanded_event = (
            event_latent[
                :, None, :
            ]
            .expand(
                -1,
                query_features.shape[1],
                -1,
            )
        )

        return self.query_decoder(
            torch.cat(
                [
                    expanded_event,
                    query_features,
                ],
                dim=-1,
            )
        )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
) -> Dict[str, float]:

    training = (
        optimizer is not None
    )
    model.train(training)

    total_loss = 0.0
    total_count = 0

    predictions = []
    targets = []

    for batch in loader:
        input_waveforms = (
            batch[
                "input_waveforms"
            ].to(
                device,
                non_blocking=True,
            )
        )
        input_features = (
            batch[
                "input_features"
            ].to(
                device,
                non_blocking=True,
            )
        )
        target_features = (
            batch[
                "target_features"
            ].to(
                device,
                non_blocking=True,
            )
        )
        target = (
            batch[
                "target_log"
            ].to(
                device,
                non_blocking=True,
            )
        )

        with torch.set_grad_enabled(
            training
        ):
            prediction = model(
                input_waveforms,
                input_features,
                target_features,
            )

            loss = (
                nn.functional
                .smooth_l1_loss(
                    prediction,
                    target,
                )
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )
                optimizer.step()

        count = int(
            target.numel()
        )

        total_loss += (
            float(
                loss.detach().cpu()
            )
            * count
        )
        total_count += count

        predictions.append(
            prediction.detach().cpu()
        )
        targets.append(
            target.detach().cpu()
        )

    prediction_tensor = (
        torch.cat(
            predictions,
            dim=0,
        )
    )
    target_tensor = torch.cat(
        targets,
        dim=0,
    )

    absolute = torch.abs(
        prediction_tensor
        - target_tensor
    )

    return {
        "loss": (
            total_loss
            / max(
                total_count,
                1,
            )
        ),
        "mae_pga": float(
            absolute[
                ..., 0
            ].mean()
        ),
        "mae_pgv": float(
            absolute[
                ..., 1
            ].mean()
        ),
    }


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:

    model.eval()
    rows: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            prediction = model(
                batch[
                    "input_waveforms"
                ].to(device),
                batch[
                    "input_features"
                ].to(device),
                batch[
                    "target_features"
                ].to(device),
            ).cpu().numpy()

            truth = (
                batch[
                    "target_log"
                ].numpy()
            )
            repeat = (
                batch[
                    "repeat"
                ].numpy()
            )
            event_ids = list(
                batch["event_id"]
            )

            for b in range(
                truth.shape[0]
            ):
                for q in range(
                    truth.shape[1]
                ):
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
                            "true_log10_pga": float(
                                truth[
                                    b, q, 0
                                ]
                            ),
                            "true_log10_pgv": float(
                                truth[
                                    b, q, 1
                                ]
                            ),
                            "pred_log10_pga": float(
                                prediction[
                                    b, q, 0
                                ]
                            ),
                            "pred_log10_pgv": float(
                                prediction[
                                    b, q, 1
                                ]
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def element_metric(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> np.ndarray:

    residual = (
        prediction - truth
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
        ).astype(float)

    if metric == "under05":
        return (
            residual <= -0.5
        ).astype(float)

    raise ValueError(metric)


def canonical_metric(
    frame: pd.DataFrame,
    quantity: str,
    metric: str,
) -> float:
    """
    Manuscript aggregation:
    targets -> repeats -> events.
    """

    truth = frame[
        "true_log10_%s" % quantity
    ].to_numpy(dtype=float)

    prediction = frame[
        "pred_log10_%s" % quantity
    ].to_numpy(dtype=float)

    values = element_metric(
        truth,
        prediction,
        metric,
    )

    working = frame[
        [
            "event_id",
            "repeat",
        ]
    ].copy()

    working["value"] = values

    event_repeat = (
        working
        .groupby(
            [
                "event_id",
                "repeat",
            ],
            sort=False,
        )["value"]
        .mean()
        .reset_index()
    )

    event = (
        event_repeat
        .groupby(
            "event_id",
            sort=False,
        )["value"]
        .mean()
    )

    return float(
        event.mean()
    )


def summarize_validation(
    predictions: pd.DataFrame,
    variant: str,
    thresholds: Dict[str, float],
) -> pd.DataFrame:

    rows: List[
        Dict[str, Any]
    ] = []

    for quantity in (
        "pga",
        "pgv",
    ):
        truth = predictions[
            "true_log10_%s"
            % quantity
        ].to_numpy(dtype=float)

        populations = {
            "overall": np.ones(
                len(predictions),
                dtype=bool,
            ),
            "high_motion_tail": (
                truth
                >= float(
                    thresholds[
                        quantity
                    ]
                )
            ),
        }

        for (
            population,
            mask,
        ) in populations.items():

            subset = (
                predictions.loc[
                    mask
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
                rows.append(
                    {
                        "variant": (
                            variant
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
                        "value": (
                            canonical_metric(
                                subset,
                                quantity,
                                metric,
                            )
                        ),
                        "n_events": int(
                            subset[
                                "event_id"
                            ].nunique()
                        ),
                        "n_event_repeats": int(
                            subset[
                                [
                                    "event_id",
                                    "repeat",
                                ]
                            ]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_target_rows": int(
                            len(subset)
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def train_variant(
    base_module,
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:

    set_global_seed(
        args.seed
    )

    train_dataset = (
        base_module
        .SparseFieldDataset(
            manifest=args.manifest,
            split_column=(
                args.split_column
            ),
            split_name="train",
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
        base_module
        .SparseFieldDataset(
            manifest=args.manifest,
            split_column=(
                args.split_column
            ),
            split_name="validation",
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
            device.type == "cuda"
        ),
        "persistent_workers": (
            args.num_workers > 0
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

    model = (
        AttentionFeatureAblationModel(
            base_module=(
                base_module
            ),
            variant=variant,
            hidden_dim=(
                args.hidden_dim
            ),
        )
        .to(device)
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=(
                args.lr_patience
            ),
            min_lr=(
                args.minimum_learning_rate
            ),
        )
    )

    variant_dir = (
        Path(args.out_root)
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

    best_validation_loss = (
        np.inf
    )
    best_epoch = 0
    epochs_without_improvement = (
        0
    )
    history = []
    start_time = time.time()

    print(
        "\n=== Attention feature ablation: %s ==="
        % variant
    )
    print(
        "Parameters           : %s"
        % format(
            sum(
                p.numel()
                for p in model.parameters()
            ),
            ",",
        )
    )
    print(
        "Train/Validation     : %d/%d"
        % (
            len(
                train_dataset.frame
            ),
            len(
                validation_dataset.frame
            ),
        )
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_metrics = (
            run_epoch(
                model,
                validation_loader,
                None,
                device,
            )
        )

        scheduler.step(
            validation_metrics[
                "loss"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history.append(
            {
                "variant": (
                    variant
                ),
                "epoch": epoch,
                "learning_rate": (
                    current_lr
                ),
                "train_loss": (
                    train_metrics[
                        "loss"
                    ]
                ),
                "train_mae_pga": (
                    train_metrics[
                        "mae_pga"
                    ]
                ),
                "train_mae_pgv": (
                    train_metrics[
                        "mae_pgv"
                    ]
                ),
                "validation_loss": (
                    validation_metrics[
                        "loss"
                    ]
                ),
                "validation_mae_pga": (
                    validation_metrics[
                        "mae_pga"
                    ]
                ),
                "validation_mae_pgv": (
                    validation_metrics[
                        "mae_pgv"
                    ]
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
            "Epoch %03d | "
            "train=%.4f | "
            "val=%.4f | "
            "PGA=%.4f | "
            "PGV=%.4f | "
            "lr=%.2e"
            % (
                epoch,
                train_metrics[
                    "loss"
                ],
                validation_metrics[
                    "loss"
                ],
                validation_metrics[
                    "mae_pga"
                ],
                validation_metrics[
                    "mae_pgv"
                ],
                current_lr,
            )
        )

        if (
            validation_metrics[
                "loss"
            ]
            < best_validation_loss
            - args.minimum_delta
        ):
            best_validation_loss = (
                validation_metrics[
                    "loss"
                ]
            )
            best_epoch = epoch
            epochs_without_improvement = (
                0
            )

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "variant": (
                        variant
                    ),
                    "epoch": (
                        epoch
                    ),
                    "args": vars(
                        args
                    ),
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "test_split_evaluated": (
                        False
                    ),
                },
                best_path,
            )

        else:
            (
                epochs_without_improvement
            ) += 1

        if (
            epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                "Early stopping at epoch %d; "
                "best epoch=%d"
                % (
                    epoch,
                    best_epoch,
                )
            )
            break

    return {
        "variant": variant,
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "parameter_count": int(
            sum(
                p.numel()
                for p in model.parameters()
            )
        ),
        "training_seconds": float(
            time.time()
            - start_time
        ),
        "checkpoint": str(
            best_path.resolve()
        ),
    }


def load_new_variant_checkpoint(
    base_module,
    variant: str,
    checkpoint_path: Path,
    hidden_dim: int,
    device: torch.device,
):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model = (
        AttentionFeatureAblationModel(
            base_module=(
                base_module
            ),
            variant=variant,
            hidden_dim=(
                hidden_dim
            ),
        )
        .to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    return (
        model,
        checkpoint,
    )


def load_attention_full_reference(
    base_module,
    checkpoint_path: Path,
    hidden_dim: int,
    device: torch.device,
):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model = (
        base_module
        .FinalAblationModel(
            variant=(
                "attention_pooling"
            ),
            hidden_dim=(
                hidden_dim
            ),
        )
        .to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    return (
        model,
        checkpoint,
    )


def metric_lookup(
    frame: pd.DataFrame,
    variant: str,
    quantity: str,
    population: str,
    metric: str,
) -> float:

    selected = frame.loc[
        frame["variant"].eq(
            variant
        )
        & frame["quantity"].eq(
            quantity
        )
        & frame["population"].eq(
            population
        )
        & frame["metric"].eq(
            metric
        )
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Metric lookup failed: "
            "%s/%s/%s/%s"
            % (
                variant,
                quantity,
                population,
                metric,
            )
        )

    return float(
        selected.iloc[0][
            "value"
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-script",
        default=(
            "40_phase1_final_mean_pooling_ablation_suite.py"
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/model_manifests/"
            "scenario_t0_5s_k5_linux.csv"
        ),
    )

    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )

    parser.add_argument(
        "--attention-full-checkpoint",
        default=(
            "runs/"
            "phase1_final_mean_pooling_ablation/"
            "attention_pooling/"
            "best_model.pt"
        ),
    )

    parser.add_argument(
        "--threshold-json",
        default=(
            "runs/"
            "tail_gated_compromise_t0_5s_k5/"
            "tail_thresholds_q0.90_t0_5s.json"
        ),
    )

    parser.add_argument(
        "--variants",
        default=(
            "attention_no_waveform,"
            "attention_no_geometry,"
            "attention_no_p_offset"
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
        "--training-validation-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--final-validation-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
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
        "--out-root",
        default=(
            "runs/"
            "phase1_attention_feature_ablation"
        ),
    )

    args = parser.parse_args()

    base_script = Path(
        args.base_script
    )
    manifest_path = Path(
        args.manifest
    )
    full_checkpoint_path = Path(
        args.attention_full_checkpoint
    )
    threshold_path = Path(
        args.threshold_json
    )

    for path in (
        base_script,
        manifest_path,
        full_checkpoint_path,
        threshold_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    variants = [
        item.strip()
        for item in (
            args.variants.split(",")
        )
        if item.strip()
    ]

    unknown = set(
        variants
    ).difference(
        VALID_VARIANTS
    )

    if unknown:
        raise ValueError(
            "Unknown variants: %s"
            % sorted(
                unknown
            )
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

    base_module = (
        load_python_module(
            base_script
        )
    )

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds = {
        "pga": float(
            threshold_data[
                "log10_pga_threshold"
            ]
        ),
        "pgv": float(
            threshold_data[
                "log10_pgv_threshold"
            ]
        ),
    }

    output_root = Path(
        args.out_root
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(
                    device
                ),
                "test_split_evaluated": False,
                "main_aggregation": (
                    "targets -> repeats -> events"
                ),
                "architecture_reference": (
                    "script-40 attention_pooling"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== Phase 1: final attention feature ablation ==="
    )
    print(
        "Device                       : %s"
        % device
    )
    print(
        "Training validation repeats  : %d"
        % args.training_validation_repeats
    )
    print(
        "Decision validation repeats  : %d"
        % args.final_validation_repeats
    )
    print(
        "Test split evaluated         : False"
    )
    print(
        "Thresholds                   : "
        "PGA=%.4f, PGV=%.4f"
        % (
            thresholds[
                "pga"
            ],
            thresholds[
                "pgv"
            ],
        )
    )

    training_results = []

    for variant in variants:
        result = train_variant(
            base_module,
            variant,
            args,
            device,
        )
        training_results.append(
            result
        )

        pd.DataFrame(
            training_results
        ).to_csv(
            output_root
            / "training_summary_partial.csv",
            index=False,
        )

    training_summary = pd.DataFrame(
        training_results
    )

    training_summary.to_csv(
        output_root
        / "training_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Validation-only 20-repeat architecture audit.
    # ------------------------------------------------------------
    audit_dataset = (
        base_module
        .SparseFieldDataset(
            manifest=args.manifest,
            split_column=(
                args.split_column
            ),
            split_name="validation",
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

    audit_loader = DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    all_metrics = []

    # Existing full attention reference.
    (
        full_model,
        full_checkpoint,
    ) = load_attention_full_reference(
        base_module=(
            base_module
        ),
        checkpoint_path=(
            full_checkpoint_path
        ),
        hidden_dim=(
            args.hidden_dim
        ),
        device=device,
    )

    full_predictions = (
        collect_predictions(
            full_model,
            audit_loader,
            device,
        )
    )

    full_predictions.to_csv(
        output_root
        / "attention_full_validation_predictions.csv",
        index=False,
    )

    all_metrics.append(
        summarize_validation(
            full_predictions,
            "attention_full",
            thresholds,
        )
    )

    print(
        "\nReference attention_full checkpoint epoch: %s"
        % full_checkpoint.get(
            "epoch",
            "unknown",
        )
    )

    # Newly trained attention feature ablations.
    for variant in variants:
        checkpoint_path = (
            output_root
            / variant
            / "best_model.pt"
        )

        (
            model,
            checkpoint,
        ) = load_new_variant_checkpoint(
            base_module=(
                base_module
            ),
            variant=variant,
            checkpoint_path=(
                checkpoint_path
            ),
            hidden_dim=(
                args.hidden_dim
            ),
            device=device,
        )

        predictions = (
            collect_predictions(
                model,
                audit_loader,
                device,
            )
        )

        predictions.to_csv(
            output_root
            / (
                "%s_validation_predictions.csv"
                % variant
            ),
            index=False,
        )

        all_metrics.append(
            summarize_validation(
                predictions,
                variant,
                thresholds,
            )
        )

        print(
            "%s checkpoint epoch: %s"
            % (
                variant,
                checkpoint.get(
                    "epoch",
                    "unknown",
                ),
            )
        )

    metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    metrics_path = (
        output_root
        / "attention_feature_ablation_validation_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Compact validation-only decision table.
    # ------------------------------------------------------------
    decision_rows = []

    decision_variants = (
        ["attention_full"]
        + variants
    )

    for variant in (
        decision_variants
    ):
        pga_overall = metric_lookup(
            metrics,
            variant,
            "pga",
            "overall",
            "mae",
        )
        pgv_overall = metric_lookup(
            metrics,
            variant,
            "pgv",
            "overall",
            "mae",
        )

        row = {
            "variant": variant,
            "overall_mae_pga": (
                pga_overall
            ),
            "overall_mae_pgv": (
                pgv_overall
            ),
            "overall_selection_score": (
                0.5
                * (
                    pga_overall
                    + pgv_overall
                )
            ),
            "tail_mae_pga": metric_lookup(
                metrics,
                variant,
                "pga",
                "high_motion_tail",
                "mae",
            ),
            "tail_mae_pgv": metric_lookup(
                metrics,
                variant,
                "pgv",
                "high_motion_tail",
                "mae",
            ),
            "tail_under05_pga": metric_lookup(
                metrics,
                variant,
                "pga",
                "high_motion_tail",
                "under05",
            ),
            "tail_under05_pgv": metric_lookup(
                metrics,
                variant,
                "pgv",
                "high_motion_tail",
                "under05",
            ),
            "overall_bias_pga": metric_lookup(
                metrics,
                variant,
                "pga",
                "overall",
                "bias",
            ),
            "overall_bias_pgv": metric_lookup(
                metrics,
                variant,
                "pgv",
                "overall",
                "bias",
            ),
        }

        decision_rows.append(
            row
        )

    decision = pd.DataFrame(
        decision_rows
    )

    full_row = (
        decision.loc[
            decision[
                "variant"
            ].eq(
                "attention_full"
            )
        ]
        .iloc[0]
    )

    for column in (
        "overall_mae_pga",
        "overall_mae_pgv",
        "tail_mae_pga",
        "tail_mae_pgv",
        "tail_under05_pga",
        "tail_under05_pgv",
    ):
        decision[
            "delta_%s_vs_full"
            % column
        ] = (
            decision[
                column
            ]
            - float(
                full_row[
                    column
                ]
            )
        )

    decision = decision.sort_values(
        "overall_selection_score"
    )

    decision_path = (
        output_root
        / "attention_validation_decision_table.csv"
    )

    decision.to_csv(
        decision_path,
        index=False,
    )

    print(
        "\n=== Validation-only final attention decision table ==="
    )

    print(
        decision[
            [
                "variant",
                "overall_mae_pga",
                "overall_mae_pgv",
                "overall_selection_score",
                "tail_mae_pga",
                "tail_mae_pgv",
                "tail_under05_pga",
                "tail_under05_pgv",
                "overall_bias_pga",
                "overall_bias_pgv",
            ]
        ].to_string(
            index=False
        )
    )

    # P-offset focused comparison.
    if (
        "attention_no_p_offset"
        in decision[
            "variant"
        ].values
    ):
        p_offset_row = (
            decision.loc[
                decision[
                    "variant"
                ].eq(
                    "attention_no_p_offset"
                )
            ]
            .iloc[0]
        )

        print(
            "\n=== P-offset decision audit ==="
        )

        for metric_name in (
            "overall_mae_pga",
            "overall_mae_pgv",
            "tail_mae_pga",
            "tail_mae_pgv",
            "tail_under05_pga",
            "tail_under05_pgv",
        ):
            delta = (
                float(
                    p_offset_row[
                        metric_name
                    ]
                )
                - float(
                    full_row[
                        metric_name
                    ]
                )
            )

            print(
                "%-20s no-P-offset minus full = %+0.6f"
                % (
                    metric_name,
                    delta,
                )
            )

    print(
        "\nNo test-set predictions were produced."
    )
    print(
        "Metrics : %s"
        % metrics_path.resolve()
    )
    print(
        "Decision: %s"
        % decision_path.resolve()
    )


if __name__ == "__main__":
    main()
