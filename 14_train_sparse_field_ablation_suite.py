#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
14_train_sparse_field_ablation_suite.py

Ablation suite for the deterministic sparse-to-field PGA/PGV baseline.

Variants
--------
full
    Waveform + station coordinates/elevation + P-offset + target coordinates
    with attention pooling.

coordinate_only
    No waveform encoder. Uses only input-station metadata and target coordinates.
    This is the most important ablation for proving that early waveforms carry
    event-state information beyond spatial/data priors.

waveform_only
    Uses waveform and P-offset, but removes input/target spatial coordinates and
    elevation. Every target receives the same event-level prediction, so this
    tests the importance of spatial conditioning.

mean_pooling
    Same inputs as the full model, but replaces learned attention pooling with
    simple mean pooling.

no_p_offset
    Same as the full model, but removes P-arrival offset from station metadata.

Fair-comparison design
----------------------
For every epoch and event, all variants receive exactly the same deterministic
input/target station combination. Validation and test combinations are fixed.
This avoids sampling noise being mistaken for an architectural difference.

Example
-------
python 14_train_sparse_field_ablation_suite.py ^
  --manifest data\scedc\model_manifests\scenario_t0_5s_k5.csv ^
  --variants full,coordinate_only,waveform_only,mean_pooling,no_p_offset ^
  --epochs 50 ^
  --out-root runs\sparse_field_ablation_t0_5s_k5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


VALID_VARIANTS = {
    "full",
    "coordinate_only",
    "waveform_only",
    "mean_pooling",
    "no_p_offset",
}

LOG10_FACTOR_2 = math.log10(2.0)


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{text}:{base_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_xy_km(
    coordinates: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
) -> np.ndarray:
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]

    x = (
        (longitude - origin_longitude)
        * 111.32
        * math.cos(math.radians(origin_latitude))
    )
    y = (latitude - origin_latitude) * 110.57
    return np.stack([x, y], axis=-1)


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


class SparseFieldAblationDataset(Dataset):
    """
    Event-level sparse-to-field dataset with deterministic epoch sampling.

    Training:
        station combinations change by epoch but are identical across variants.
    Validation/test:
        combinations stay fixed for reproducibility.
    """

    def __init__(
        self,
        manifest: str | Path,
        split_column: str,
        split_name: str,
        t0_sec: int,
        input_stations: int,
        target_stations: int,
        input_pre_sec: float,
        base_seed: int,
        training: bool,
    ):
        self.manifest_path = Path(manifest)
        self.frame = pd.read_csv(
            self.manifest_path,
            dtype={"event_id": str},
        )

        required_columns = {
            "event_id",
            "h5_path",
            split_column,
        }
        missing = required_columns.difference(self.frame.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}"
            )

        self.frame = self.frame.loc[
            self.frame[split_column]
            .astype(str)
            .eq(split_name)
        ].copy()

        self.t0_sec = int(t0_sec)
        self.input_stations = int(input_stations)
        self.target_stations = int(target_stations)
        self.input_pre_sec = float(input_pre_sec)
        self.base_seed = int(base_seed)
        self.training = bool(training)
        self.epoch = 0

        eligible_column = (
            f"eligible_t0_{self.t0_sec}s_k{self.input_stations}"
        )
        if eligible_column in self.frame.columns:
            self.frame = self.frame.loc[
                as_bool(self.frame[eligible_column])
            ].copy()

        self.frame = self.frame.reset_index(drop=True)

        if len(self.frame) == 0:
            raise ValueError(
                f"No events for {split_column}={split_name} "
                f"in {self.manifest_path}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def _random_generator(
        self,
        event_id: str,
        index: int,
    ) -> np.random.Generator:
        if self.training:
            key = (
                f"{event_id}:epoch:{self.epoch}:"
                f"sample:{index}"
            )
        else:
            key = f"{event_id}:fixed-evaluation"

        return np.random.default_rng(
            stable_seed(key, self.base_seed)
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        event = self.frame.iloc[index]
        event_id = str(event["event_id"])
        h5_path = Path(str(event["h5_path"]))
        rng = self._random_generator(event_id, index)

        with h5py.File(h5_path, "r") as h5:
            acceleration = np.asarray(
                h5["acceleration"][:],
                dtype=np.float32,
            )
            velocity = np.asarray(
                h5["velocity"][:],
                dtype=np.float32,
            )
            coordinates = np.asarray(
                h5["station_coords"][:],
                dtype=np.float32,
            )
            p_offset = np.asarray(
                h5["p_offset_sec"][:],
                dtype=np.float32,
            )

            sampling_rate = float(
                h5.attrs["sampling_rate_hz"]
            )
            pre_first_p = float(
                h5.attrs["pre_first_p_sec"]
            )
            time_zero_index = int(
                h5.attrs["time_zero_index"]
            )

        triggered_indices = np.flatnonzero(
            np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (p_offset <= float(self.t0_sec))
        )

        if len(triggered_indices) < self.input_stations:
            raise RuntimeError(
                f"Event {event_id}: only "
                f"{len(triggered_indices)} triggered stations"
            )

        input_indices = rng.choice(
            triggered_indices,
            size=self.input_stations,
            replace=False,
        )

        target_pool = np.setdiff1d(
            np.arange(len(coordinates)),
            input_indices,
            assume_unique=False,
        )

        if len(target_pool) < self.target_stations:
            raise RuntimeError(
                f"Event {event_id}: only "
                f"{len(target_pool)} target stations"
            )

        target_indices = rng.choice(
            target_pool,
            size=self.target_stations,
            replace=False,
        )

        input_start_index = int(
            round(
                (pre_first_p - self.input_pre_sec)
                * sampling_rate
            )
        )
        input_start_index = max(0, input_start_index)

        snapshot_index = (
            time_zero_index
            + int(round(self.t0_sec * sampling_rate))
        )
        snapshot_index = min(
            snapshot_index,
            acceleration.shape[-1] - 1,
        )

        input_waveforms = acceleration[
            input_indices,
            :,
            input_start_index:snapshot_index,
        ]

        # Preserve sign and relative amplitude while compressing the wide range.
        amplitude_scale = 1e-3
        input_waveforms = (
            np.sign(input_waveforms)
            * np.log1p(
                np.abs(input_waveforms)
                / amplitude_scale
            )
        ).astype(np.float32)

        input_coordinates = coordinates[input_indices]
        target_coordinates = coordinates[target_indices]

        origin_latitude = float(
            np.mean(input_coordinates[:, 0])
        )
        origin_longitude = float(
            np.mean(input_coordinates[:, 1])
        )

        input_xy = local_xy_km(
            input_coordinates,
            origin_latitude,
            origin_longitude,
        )
        target_xy = local_xy_km(
            target_coordinates,
            origin_latitude,
            origin_longitude,
        )

        input_features = np.concatenate(
            [
                input_xy / 100.0,
                input_coordinates[:, 2:3] / 2000.0,
                p_offset[input_indices, None]
                / max(float(self.t0_sec), 1.0),
            ],
            axis=1,
        ).astype(np.float32)

        target_features = np.concatenate(
            [
                target_xy / 100.0,
                target_coordinates[:, 2:3] / 2000.0,
            ],
            axis=1,
        ).astype(np.float32)

        future_acceleration = acceleration[
            target_indices,
            :,
            snapshot_index:,
        ]
        future_velocity = velocity[
            target_indices,
            :,
            snapshot_index:,
        ]

        pga = np.max(
            np.sqrt(
                future_acceleration[:, 0, :] ** 2
                + future_acceleration[:, 1, :] ** 2
            ),
            axis=1,
        )
        pgv = np.max(
            np.sqrt(
                future_velocity[:, 0, :] ** 2
                + future_velocity[:, 1, :] ** 2
            ),
            axis=1,
        )

        target_log = np.stack(
            [
                np.log10(np.maximum(pga, 1e-10)),
                np.log10(np.maximum(pgv, 1e-12)),
            ],
            axis=-1,
        ).astype(np.float32)

        return {
            "event_id": event_id,
            "input_waveforms": torch.from_numpy(
                input_waveforms
            ),
            "input_features": torch.from_numpy(
                input_features
            ),
            "target_features": torch.from_numpy(
                target_features
            ),
            "target_log": torch.from_numpy(
                target_log
            ),
        }


class WaveformEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                3,
                32,
                kernel_size=9,
                stride=2,
                padding=4,
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.network(waveforms)


class SparseFieldAblationModel(nn.Module):
    def __init__(
        self,
        variant: str,
        hidden_dim: int = 128,
    ):
        super().__init__()

        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}"
            )

        self.variant = variant
        self.hidden_dim = int(hidden_dim)

        self.use_waveform = (
            variant != "coordinate_only"
        )
        self.pooling = (
            "mean"
            if variant == "mean_pooling"
            else "attention"
        )

        if self.use_waveform:
            self.waveform_encoder = WaveformEncoder(
                hidden_dim
            )
            station_input_dim = hidden_dim + 4
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

        if self.pooling == "attention":
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
        else:
            self.attention_score = None

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
            nn.Linear(hidden_dim, 2),
        )

    def _apply_ablation(
        self,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        station_features = input_features
        query_features = target_features

        if self.variant == "waveform_only":
            # Retain only normalized P offset.
            station_features = station_features.clone()
            station_features[..., :3] = 0.0
            query_features = torch.zeros_like(
                query_features
            )

        elif self.variant == "no_p_offset":
            station_features = station_features.clone()
            station_features[..., 3] = 0.0

        return station_features, query_features

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

            waveform_latent = self.waveform_encoder(
                input_waveforms.reshape(
                    batch_size * station_count,
                    channel_count,
                    sample_count,
                )
            ).reshape(
                batch_size,
                station_count,
                -1,
            )

            station_input = torch.cat(
                [
                    waveform_latent,
                    station_features,
                ],
                dim=-1,
            )
        else:
            station_input = station_features

        station_latent = self.station_encoder(
            station_input
        )

        if self.pooling == "mean":
            event_latent = torch.mean(
                station_latent,
                dim=1,
            )
        else:
            weights = torch.softmax(
                self.attention_score(
                    station_latent
                ),
                dim=1,
            )
            event_latent = torch.sum(
                weights * station_latent,
                dim=1,
            )

        query_count = query_features.shape[1]
        event_expanded = event_latent[
            :,
            None,
            :,
        ].expand(
            -1,
            query_count,
            -1,
        )

        return self.query_decoder(
            torch.cat(
                [
                    event_expanded,
                    query_features,
                ],
                dim=-1,
            )
        )


def compute_batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    residual = prediction - target
    absolute_error = torch.abs(residual)

    return {
        "mae_log10_pga": float(
            absolute_error[..., 0]
            .mean()
            .detach()
            .cpu()
        ),
        "mae_log10_pgv": float(
            absolute_error[..., 1]
            .mean()
            .detach()
            .cpu()
        ),
        "bias_log10_pga": float(
            residual[..., 0]
            .mean()
            .detach()
            .cpu()
        ),
        "bias_log10_pgv": float(
            residual[..., 1]
            .mean()
            .detach()
            .cpu()
        ),
        "factor2_pga": float(
            (
                absolute_error[..., 0]
                <= LOG10_FACTOR_2
            )
            .float()
            .mean()
            .detach()
            .cpu()
        ),
        "factor2_pgv": float(
            (
                absolute_error[..., 1]
                <= LOG10_FACTOR_2
            )
            .float()
            .mean()
            .detach()
            .cpu()
        ),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_function = nn.SmoothL1Loss(
        reduction="mean"
    )

    total_loss = 0.0
    total_target_count = 0

    all_prediction = []
    all_target = []

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
        target = batch["target_log"].to(
            device,
            non_blocking=True,
        )

        with torch.set_grad_enabled(training):
            prediction = model(
                input_waveforms,
                input_features,
                target_features,
            )
            loss = loss_function(
                prediction,
                target,
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

        target_count = int(target.numel())
        total_loss += (
            float(loss.detach().cpu())
            * target_count
        )
        total_target_count += target_count

        all_prediction.append(
            prediction.detach().cpu()
        )
        all_target.append(
            target.detach().cpu()
        )

    prediction_tensor = torch.cat(
        all_prediction,
        dim=0,
    )
    target_tensor = torch.cat(
        all_target,
        dim=0,
    )

    metrics = compute_batch_metrics(
        prediction_tensor,
        target_tensor,
    )
    metrics["loss"] = (
        total_loss
        / max(total_target_count, 1)
    )
    return metrics


def train_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_global_seed(args.seed)

    train_dataset = SparseFieldAblationDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_name="train",
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        base_seed=args.seed,
        training=True,
    )
    validation_dataset = SparseFieldAblationDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_name="validation",
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        base_seed=args.seed,
        training=False,
    )
    test_dataset = SparseFieldAblationDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_name="test",
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        base_seed=args.seed,
        training=False,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        generator=loader_generator,
        persistent_workers=(
            args.num_workers > 0
        ),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = SparseFieldAblationModel(
        variant=variant,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )

    variant_directory = (
        Path(args.out_root) / variant
    )
    variant_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        variant_directory / "best_model.pt"
    )

    print(
        f"\n=== Ablation variant: {variant} ==="
    )
    print(f"Device             : {device}")
    print(
        f"Parameters         : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"Train/Val/Test     : "
        f"{len(train_dataset)}/"
        f"{len(validation_dataset)}/"
        f"{len(test_dataset)}"
    )

    history: list[dict[str, Any]] = []
    best_validation_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            None,
            device,
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "variant": variant,
            "epoch": epoch,
            "learning_rate": (
                current_learning_rate
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
        history.append(row)

        pd.DataFrame(history).to_csv(
            variant_directory / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={validation_metrics['loss']:.4f} | "
            f"PGA={validation_metrics['mae_log10_pga']:.4f} | "
            f"PGV={validation_metrics['mae_log10_pgv']:.4f} | "
            f"lr={current_learning_rate:.2e}"
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss
            - args.min_delta
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "variant": variant,
                    "args": vars(args),
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch "
                f"{epoch}; best epoch={best_epoch}"
            )
            break

    try:
        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        device,
    )

    elapsed_seconds = time.time() - start_time

    result = {
        "variant": variant,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
            )
        ),
        "training_seconds": float(
            elapsed_seconds
        ),
        "train_events": int(
            len(train_dataset)
        ),
        "validation_events": int(
            len(validation_dataset)
        ),
        "test_events": int(
            len(test_dataset)
        ),
        **{
            f"test_{key}": value
            for key, value
            in test_metrics.items()
        },
    }

    (
        variant_directory
        / "test_metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nBest-model test: "
        f"PGA MAE={test_metrics['mae_log10_pga']:.4f}, "
        f"PGV MAE={test_metrics['mae_log10_pgv']:.4f}, "
        f"PGA F2={test_metrics['factor2_pga']:.3f}, "
        f"PGV F2={test_metrics['factor2_pgv']:.3f}"
    )

    return result


def parse_variants(text: str) -> list[str]:
    variants = [
        item.strip()
        for item in str(text).split(",")
        if item.strip()
    ]

    unknown = set(variants).difference(
        VALID_VARIANTS
    )
    if unknown:
        raise ValueError(
            f"Unknown variants: {sorted(unknown)}. "
            f"Valid variants: {sorted(VALID_VARIANTS)}"
        )

    if not variants:
        raise ValueError(
            "At least one variant is required."
        )

    return variants


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "data/scedc/model_manifests_sensitivity/"
            "scenario_t0_5s_k5.csv"
        ),
    )
    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )
    parser.add_argument(
        "--variants",
        default=(
            "mean_pooling"
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
        "--min-learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )
    parser.add_argument(
        "--out-root",
        default=(
            "runs/"
            "observation_budget_t0_5s_k5"
        ),
    )

    args = parser.parse_args()
    variants = parse_variants(
        args.variants
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
                "variants": variants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results = []

    for variant in variants:
        result = train_variant(
            variant=variant,
            args=args,
            device=device,
        )
        results.append(result)

        summary = pd.DataFrame(results)
        summary.to_csv(
            output_root
            / "ablation_summary_partial.csv",
            index=False,
        )

    summary = pd.DataFrame(results)
    summary = summary.sort_values(
        [
            "test_mae_log10_pga",
            "test_mae_log10_pgv",
        ]
    )
    summary.to_csv(
        output_root
        / "ablation_summary.csv",
        index=False,
    )

    display_columns = [
        "variant",
        "best_epoch",
        "parameter_count",
        "test_mae_log10_pga",
        "test_mae_log10_pgv",
        "test_bias_log10_pga",
        "test_bias_log10_pgv",
        "test_factor2_pga",
        "test_factor2_pgv",
    ]

    print(
        "\n=== Ablation summary ==="
    )
    print(
        summary[
            display_columns
        ].to_string(index=False)
    )
    print(
        f"\nOutputs: "
        f"{output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
