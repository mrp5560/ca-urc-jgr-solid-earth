#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
15_train_mean_pooling_tail_suite.py

Mean-pooling and tail-calibration suite for causal sparse-to-field PGA/PGV
forecasting.

Purpose
-------
The previous ablation showed that mean pooling outperformed simple global
attention. This script therefore fixes the architecture to the strongest
mean-pooling baseline and isolates two questions:

1. Does P-arrival offset still help under mean pooling?
2. Can tail-aware loss reduce strong-motion underprediction without seriously
   degrading overall accuracy?

Variants
--------
mean_base
    Waveform + coordinates + P-offset + mean pooling + standard SmoothL1.

mean_no_p_offset
    Same as mean_base, but P-offset is removed. This cleanly isolates the
    contribution of propagation timing under the strongest aggregation method.

tail_weighted
    Mean baseline with target-intensity-aware weighted SmoothL1.

tail_under
    Tail-weighted SmoothL1 plus asymmetric underprediction penalty.

Loss
----
For quantity y in log10 space:

    weight(y) = 1 + lambda_tail * sigmoid((y - q_tail) / tail_scale)

    L_tail = weight(y) * SmoothL1(pred, y)

    L_under = weight(y) * relu(y - pred)^2

The q_tail thresholds are computed from all future training targets at the
selected snapshot and cached to JSON.

Reported metrics
----------------
Overall:
    MAE, bias, factor-of-two accuracy.

Strong-motion tail:
    MAE, bias, Under03, Under05.

Under03:
    fraction with prediction - truth <= -0.3 log10 units (~factor 2 low).

Under05:
    fraction with prediction - truth <= -0.5 log10 units (~factor 3.16 low).

Example
-------
python 15_train_mean_pooling_tail_suite.py ^
  --manifest data\scedc\model_manifests\scenario_t0_5s_k5.csv ^
  --variants mean_no_p_offset,tail_weighted,tail_under ^
  --epochs 50 ^
  --out-root runs\mean_pooling_tail_t0_5s_k5
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
from tqdm import tqdm


VALID_VARIANTS = {
    "mean_base",
    "mean_no_p_offset",
    "tail_weighted",
    "tail_under",
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


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


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


def horizontal_peak(data: np.ndarray) -> np.ndarray:
    return np.max(
        np.sqrt(
            data[:, 0, :] ** 2
            + data[:, 1, :] ** 2
        ),
        axis=1,
    )


class SparseFieldTailDataset(Dataset):
    """
    Fair event-level sampling.

    Training combinations change by epoch, but every model variant receives the
    same event/epoch combinations. Validation and test combinations are fixed.
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
        self.frame = pd.read_csv(
            manifest,
            dtype={"event_id": str},
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
                f"No events for {split_column}={split_name}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def _rng(
        self,
        event_id: str,
        index: int,
    ) -> np.random.Generator:
        if self.training:
            text = (
                f"{event_id}:epoch:{self.epoch}:"
                f"sample:{index}"
            )
        else:
            text = f"{event_id}:fixed-evaluation"

        return np.random.default_rng(
            stable_seed(text, self.base_seed)
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        event = self.frame.iloc[index]
        event_id = str(event["event_id"])
        rng = self._rng(event_id, index)

        with h5py.File(
            str(event["h5_path"]),
            "r",
        ) as h5:
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
                f"Event {event_id} has only "
                f"{len(triggered_indices)} triggered stations."
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
                f"Event {event_id} has only "
                f"{len(target_pool)} target stations."
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

        pga = horizontal_peak(future_acceleration)
        pgv = horizontal_peak(future_velocity)

        target_log = np.stack(
            [
                np.log10(np.maximum(pga, 1e-10)),
                np.log10(np.maximum(pgv, 1e-12)),
            ],
            axis=-1,
        ).astype(np.float32)

        magnitude = pd.to_numeric(
            event.get("magnitude", np.nan),
            errors="coerce",
        )

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
            "magnitude": torch.tensor(
                float(magnitude)
                if np.isfinite(magnitude)
                else np.nan,
                dtype=torch.float32,
            ),
        }


class WaveformEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
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
            nn.Linear(128, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MeanPoolingSparseFieldModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_p_offset: bool,
    ):
        super().__init__()
        self.use_p_offset = bool(use_p_offset)

        self.waveform_encoder = WaveformEncoder(
            hidden_dim
        )
        self.station_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_p_offset:
            input_features = input_features.clone()
            input_features[..., 3] = 0.0

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

        station_latent = self.station_encoder(
            torch.cat(
                [
                    waveform_latent,
                    input_features,
                ],
                dim=-1,
            )
        )

        event_latent = torch.mean(
            station_latent,
            dim=1,
        )

        query_count = target_features.shape[1]
        event_expanded = event_latent[:, None, :].expand(
            -1,
            query_count,
            -1,
        )

        return self.query_decoder(
            torch.cat(
                [
                    event_expanded,
                    target_features,
                ],
                dim=-1,
            )
        )


def compute_tail_thresholds(
    manifest: str | Path,
    split_column: str,
    t0_sec: int,
    quantile: float,
    cache_path: Path,
) -> dict[str, float]:
    """
    Compute q_tail from all future target stations in the training split.

    This is a one-time scan and is cached.
    """
    if cache_path.exists():
        cached = json.loads(
            cache_path.read_text(encoding="utf-8")
        )
        if (
            float(cached["quantile"]) == float(quantile)
            and int(cached["t0_sec"]) == int(t0_sec)
        ):
            return cached

    frame = pd.read_csv(
        manifest,
        dtype={"event_id": str},
    )
    train = frame.loc[
        frame[split_column]
        .astype(str)
        .eq("train")
    ]

    pga_values: list[np.ndarray] = []
    pgv_values: list[np.ndarray] = []

    for event in tqdm(
        train.itertuples(index=False),
        total=len(train),
        desc="compute tail thresholds",
    ):
        with h5py.File(
            str(event.h5_path),
            "r",
        ) as h5:
            acceleration = np.asarray(
                h5["acceleration"][:],
                dtype=np.float32,
            )
            velocity = np.asarray(
                h5["velocity"][:],
                dtype=np.float32,
            )
            sampling_rate = float(
                h5.attrs["sampling_rate_hz"]
            )
            time_zero_index = int(
                h5.attrs["time_zero_index"]
            )

        snapshot_index = (
            time_zero_index
            + int(round(t0_sec * sampling_rate))
        )
        snapshot_index = min(
            snapshot_index,
            acceleration.shape[-1] - 1,
        )

        pga = horizontal_peak(
            acceleration[:, :, snapshot_index:]
        )
        pgv = horizontal_peak(
            velocity[:, :, snapshot_index:]
        )

        pga_values.append(
            np.log10(np.maximum(pga, 1e-10))
        )
        pgv_values.append(
            np.log10(np.maximum(pgv, 1e-12))
        )

    pga_all = np.concatenate(pga_values)
    pgv_all = np.concatenate(pgv_values)

    result = {
        "t0_sec": int(t0_sec),
        "quantile": float(quantile),
        "log10_pga_threshold": float(
            np.quantile(pga_all, quantile)
        ),
        "log10_pgv_threshold": float(
            np.quantile(pgv_all, quantile)
        ),
        "n_training_station_targets": int(
            len(pga_all)
        ),
    }

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


class TailAwareLoss(nn.Module):
    def __init__(
        self,
        variant: str,
        thresholds: torch.Tensor,
        lambda_tail: float,
        tail_scale: float,
        lambda_under: float,
    ):
        super().__init__()
        self.variant = variant
        self.register_buffer(
            "thresholds",
            thresholds.reshape(1, 1, 2),
        )
        self.lambda_tail = float(lambda_tail)
        self.tail_scale = float(tail_scale)
        self.lambda_under = float(lambda_under)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        huber = nn.functional.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
        )

        if self.variant in {
            "tail_weighted",
            "tail_under",
        }:
            weights = (
                1.0
                + self.lambda_tail
                * torch.sigmoid(
                    (
                        target
                        - self.thresholds
                    )
                    / self.tail_scale
                )
            )
        else:
            weights = torch.ones_like(
                target
            )

        weighted_huber = (
            weights * huber
        ).mean()

        under_loss = torch.zeros(
            (),
            device=prediction.device,
        )
        if self.variant == "tail_under":
            under_error = torch.relu(
                target - prediction
            )
            under_loss = (
                weights
                * under_error.pow(2)
            ).mean()

        total = (
            weighted_huber
            + self.lambda_under
            * under_loss
        )

        return total, {
            "weighted_huber": float(
                weighted_huber.detach().cpu()
            ),
            "under_loss": float(
                under_loss.detach().cpu()
            ),
        }


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, float]:
    residual = prediction - target
    absolute_error = torch.abs(residual)

    thresholds = thresholds.reshape(1, 1, 2)
    tail_mask = target >= thresholds

    output = {
        "mae_log10_pga": float(
            absolute_error[..., 0].mean()
        ),
        "mae_log10_pgv": float(
            absolute_error[..., 1].mean()
        ),
        "bias_log10_pga": float(
            residual[..., 0].mean()
        ),
        "bias_log10_pgv": float(
            residual[..., 1].mean()
        ),
        "factor2_pga": float(
            (
                absolute_error[..., 0]
                <= LOG10_FACTOR_2
            ).float().mean()
        ),
        "factor2_pgv": float(
            (
                absolute_error[..., 1]
                <= LOG10_FACTOR_2
            ).float().mean()
        ),
    }

    for quantity_index, quantity_name in enumerate(
        ["pga", "pgv"]
    ):
        current_mask = tail_mask[
            ...,
            quantity_index,
        ]
        current_residual = residual[
            ...,
            quantity_index,
        ]
        current_absolute = absolute_error[
            ...,
            quantity_index,
        ]

        tail_count = int(
            current_mask.sum()
        )
        output[
            f"tail_n_{quantity_name}"
        ] = tail_count

        if tail_count > 0:
            tail_residual = current_residual[
                current_mask
            ]
            tail_absolute = current_absolute[
                current_mask
            ]

            output[
                f"tail_mae_log10_{quantity_name}"
            ] = float(
                tail_absolute.mean()
            )
            output[
                f"tail_bias_log10_{quantity_name}"
            ] = float(
                tail_residual.mean()
            )
            output[
                f"tail_under03_{quantity_name}"
            ] = float(
                (
                    tail_residual <= -0.3
                ).float().mean()
            )
            output[
                f"tail_under05_{quantity_name}"
            ] = float(
                (
                    tail_residual <= -0.5
                ).float().mean()
            )
        else:
            output[
                f"tail_mae_log10_{quantity_name}"
            ] = np.nan
            output[
                f"tail_bias_log10_{quantity_name}"
            ] = np.nan
            output[
                f"tail_under03_{quantity_name}"
            ] = np.nan
            output[
                f"tail_under05_{quantity_name}"
            ] = np.nan

    return output


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_function: TailAwareLoss,
    thresholds: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0

    predictions = []
    targets = []

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

        with torch.set_grad_enabled(training):
            prediction = model(
                input_waveforms,
                input_features,
                target_features,
            )
            loss, _ = loss_function(
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

        total_loss += float(
            loss.detach().cpu()
        )
        total_batches += 1

        predictions.append(
            prediction.detach().cpu()
        )
        targets.append(
            target.detach().cpu()
        )

    prediction_tensor = torch.cat(
        predictions,
        dim=0,
    )
    target_tensor = torch.cat(
        targets,
        dim=0,
    )

    metrics = compute_metrics(
        prediction_tensor,
        target_tensor,
        thresholds.cpu(),
    )
    metrics["loss"] = (
        total_loss
        / max(total_batches, 1)
    )
    return metrics


def train_variant(
    variant: str,
    args: argparse.Namespace,
    thresholds: dict[str, float],
    device: torch.device,
) -> dict[str, Any]:
    set_global_seed(args.seed)

    train_dataset = SparseFieldTailDataset(
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
    validation_dataset = SparseFieldTailDataset(
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
    test_dataset = SparseFieldTailDataset(
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

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    common_loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_arguments,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_arguments,
    )

    use_p_offset = (
        variant != "mean_no_p_offset"
    )
    model = MeanPoolingSparseFieldModel(
        hidden_dim=args.hidden_dim,
        use_p_offset=use_p_offset,
    ).to(device)

    threshold_tensor = torch.tensor(
        [
            thresholds[
                "log10_pga_threshold"
            ],
            thresholds[
                "log10_pgv_threshold"
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    loss_function = TailAwareLoss(
        variant=variant,
        thresholds=threshold_tensor,
        lambda_tail=args.lambda_tail,
        tail_scale=args.tail_scale,
        lambda_under=args.lambda_under,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=args.min_learning_rate,
        )
    )

    output_directory = (
        Path(args.out_root)
        / variant
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_path = (
        output_directory
        / "best_model.pt"
    )

    print(
        f"\n=== Mean-pooling variant: "
        f"{variant} ==="
    )
    print(f"Device            : {device}")
    print(
        f"Train/Val/Test    : "
        f"{len(train_dataset)}/"
        f"{len(validation_dataset)}/"
        f"{len(test_dataset)}"
    )
    print(
        "Tail thresholds   : "
        f"PGA={thresholds['log10_pga_threshold']:.4f}, "
        f"PGV={thresholds['log10_pgv_threshold']:.4f}"
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
            loss_function,
            threshold_tensor,
            device,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            None,
            loss_function,
            threshold_tensor,
            device,
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        row = {
            "variant": variant,
            "epoch": epoch,
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
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
            output_directory / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"val={validation_metrics['loss']:.4f} | "
            f"PGA={validation_metrics['mae_log10_pga']:.4f} | "
            f"PGV={validation_metrics['mae_log10_pgv']:.4f} | "
            f"tail PGA={validation_metrics['tail_mae_log10_pga']:.4f} | "
            f"tail PGV={validation_metrics['tail_mae_log10_pgv']:.4f}"
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
                    "thresholds": thresholds,
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch "
                f"{epoch}; best={best_epoch}"
            )
            break

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

    model.load_state_dict(
        checkpoint["model_state"]
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        loss_function,
        threshold_tensor,
        device,
    )

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
            time.time() - start_time
        ),
        **{
            f"test_{key}": value
            for key, value
            in test_metrics.items()
        },
    }

    (
        output_directory
        / "test_metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nTest result | "
        f"PGA={test_metrics['mae_log10_pga']:.4f}, "
        f"PGV={test_metrics['mae_log10_pgv']:.4f}, "
        f"tail PGA={test_metrics['tail_mae_log10_pga']:.4f}, "
        f"tail PGV={test_metrics['tail_mae_log10_pgv']:.4f}, "
        f"tail PGA bias={test_metrics['tail_bias_log10_pga']:.4f}, "
        f"tail PGV bias={test_metrics['tail_bias_log10_pgv']:.4f}"
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
            f"Unknown variants: {sorted(unknown)}"
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
            "data/scedc/model_manifests/"
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
            "tail_under"
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
        "--tail-quantile",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--lambda-tail",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--tail-scale",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--lambda-under",
        type=float,
        default=0.50,
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
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--out-root",
        default=(
            "runs/"
            "mean_pooling_tail_t0_5s_k5"
        ),
    )

    args = parser.parse_args()
    variants = parse_variants(
        args.variants
    )

    if not 0.5 < args.tail_quantile < 1.0:
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
            args.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable."
            )
        device = torch.device(
            args.device
        )

    output_root = Path(args.out_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = compute_tail_thresholds(
        manifest=args.manifest,
        split_column=args.split_column,
        t0_sec=args.t0_sec,
        quantile=args.tail_quantile,
        cache_path=(
            output_root
            / (
                f"tail_thresholds_"
                f"q{args.tail_quantile:.2f}_"
                f"t0_{args.t0_sec}s.json"
            )
        ),
    )

    (
        output_root
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(args),
                "variants": variants,
                "resolved_device": str(device),
                "tail_thresholds": thresholds,
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
            thresholds=thresholds,
            device=device,
        )
        results.append(result)

        pd.DataFrame(results).to_csv(
            output_root
            / "tail_suite_summary_partial.csv",
            index=False,
        )

    summary = pd.DataFrame(results)
    summary.to_csv(
        output_root
        / "tail_suite_summary.csv",
        index=False,
    )

    display_columns = [
        "variant",
        "best_epoch",
        "test_mae_log10_pga",
        "test_mae_log10_pgv",
        "test_bias_log10_pga",
        "test_bias_log10_pgv",
        "test_tail_mae_log10_pga",
        "test_tail_mae_log10_pgv",
        "test_tail_bias_log10_pga",
        "test_tail_bias_log10_pgv",
        "test_tail_under03_pga",
        "test_tail_under03_pgv",
        "test_tail_under05_pga",
        "test_tail_under05_pgv",
    ]

    print(
        "\n=== Mean-pooling tail suite summary ==="
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
