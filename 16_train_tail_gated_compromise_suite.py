#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
16_train_tail_gated_compromise_suite.py

Tail-gated compromise training for causal sparse-to-field PGA/PGV forecasting.

Key changes relative to the previous tail-aware script
------------------------------------------------------
1. The asymmetric underprediction penalty is gated by target intensity:

       gate(y) = sigmoid((y - q_tail) / tail_scale)

       L_under = mean(gate(y) * relu(y - pred)^2)

   Therefore ordinary weak-motion samples are not globally pushed upward.

2. Every epoch is evaluated with three model-selection criteria:

       overall_score
       tail_score
       compromise_score

   and three checkpoints are saved:

       best_overall.pt
       best_tail.pt
       best_compromise.pt

3. Three recommended tail-calibration presets are provided:

       tail_weighted_mild : lambda_tail=1.0, lambda_under=0.0
       tail_gated_mild    : lambda_tail=1.0, lambda_under=0.15
       tail_gated_medium  : lambda_tail=1.5, lambda_under=0.20

   A standard mean-pooling baseline is also available:

       mean_base          : lambda_tail=0.0, lambda_under=0.0

4. Test predictions and metrics are exported for every saved checkpoint.

Default experiment
------------------
T0 = 5 s
K = 5 input stations
10 held-out target stations
Tail threshold = training-set 90th percentile

Recommended formal run
----------------------
python 16_train_tail_gated_compromise_suite.py ^
  --manifest data\scedc\model_manifests\scenario_t0_5s_k5.csv ^
  --variants mean_base,tail_weighted_mild,tail_gated_mild,tail_gated_medium ^
  --epochs 50 ^
  --out-root runs\tail_gated_compromise_t0_5s_k5
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


PRESETS: dict[str, dict[str, float]] = {
    "mean_base": {
        "lambda_tail": 0.0,
        "lambda_under": 0.0,
    },
    "tail_gated_ultralight": {
        "lambda_tail": 0.25,
        "lambda_under": 0.025,
    },
    "tail_gated_light": {
        "lambda_tail": 0.50,
        "lambda_under": 0.05,
    },
    "tail_weighted_mild": {
        "lambda_tail": 1.0,
        "lambda_under": 0.0,
    },
    "tail_gated_mild": {
        "lambda_tail": 1.0,
        "lambda_under": 0.15,
    },
    "tail_gated_medium": {
        "lambda_tail": 1.5,
        "lambda_under": 0.20,
    },
}

CHECKPOINT_SELECTIONS = (
    "overall",
    "tail",
    "compromise",
)

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

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    if data.shape[-1] == 0:
        raise ValueError("Waveform window is empty.")

    return np.max(
        np.sqrt(
            data[:, 0, :] ** 2
            + data[:, 1, :] ** 2
        ),
        axis=1,
    )


class SparseFieldDataset(Dataset):
    """
    Event-level dataset with fair deterministic station sampling.

    Training:
        The station combination changes with epoch.

    Validation/test:
        The station combination is fixed.

    Because the same seed and epoch are used for all model variants, each
    variant receives exactly the same event/station combinations.
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
        missing_columns = required_columns.difference(
            self.frame.columns
        )
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                f"{sorted(missing_columns)}"
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
            f"eligible_t0_{self.t0_sec}s_"
            f"k{self.input_stations}"
        )
        if eligible_column in self.frame.columns:
            self.frame = self.frame.loc[
                as_bool(self.frame[eligible_column])
            ].copy()

        self.frame = self.frame.reset_index(drop=True)

        if len(self.frame) == 0:
            raise ValueError(
                f"No events for {split_column}={split_name}."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def _rng(
        self,
        event_id: str,
        sample_index: int,
    ) -> np.random.Generator:
        if self.training:
            key = (
                f"{event_id}:epoch:{self.epoch}:"
                f"sample:{sample_index}"
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
        rng = self._rng(event_id, index)

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

            station_ids = h5[
                "station_id"
            ].asstr()[:]

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
                f"Event {event_id}: only "
                f"{len(target_pool)} target stations."
            )

        target_indices = rng.choice(
            target_pool,
            size=self.target_stations,
            replace=False,
        )

        input_start_index = int(
            round(
                (
                    pre_first_p
                    - self.input_pre_sec
                )
                * sampling_rate
            )
        )
        input_start_index = max(
            0,
            input_start_index,
        )

        snapshot_index = (
            time_zero_index
            + int(
                round(
                    self.t0_sec
                    * sampling_rate
                )
            )
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

        pga = horizontal_peak(
            future_acceleration
        )
        pgv = horizontal_peak(
            future_velocity
        )

        target_log = np.stack(
            [
                np.log10(
                    np.maximum(pga, 1e-10)
                ),
                np.log10(
                    np.maximum(pgv, 1e-12)
                ),
            ],
            axis=-1,
        ).astype(np.float32)

        magnitude = pd.to_numeric(
            event.get("magnitude", np.nan),
            errors="coerce",
        )

        target_triggered = (
            np.isfinite(p_offset[target_indices])
            & (
                p_offset[target_indices]
                <= float(self.t0_sec)
            )
        )

        target_station_ids = np.asarray(
            [
                str(station_ids[i])
                for i in target_indices
            ]
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
            "target_triggered": torch.from_numpy(
                target_triggered.astype(np.bool_)
            ),
            "target_p_offset": torch.from_numpy(
                p_offset[target_indices].astype(
                    np.float32
                )
            ),
            "target_station_id": (
                target_station_ids.tolist()
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
            nn.Linear(
                128,
                hidden_dim,
            ),
        )

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(waveforms)


class MeanPoolingSparseFieldModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.waveform_encoder = WaveformEncoder(
            hidden_dim
        )

        self.station_encoder = nn.Sequential(
            nn.Linear(
                hidden_dim + 4,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
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

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
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
                    target_features,
                ],
                dim=-1,
            )
        )


class TailGatedLoss(nn.Module):
    """
    Intensity-weighted Huber with tail-only asymmetric underprediction penalty.
    """

    def __init__(
        self,
        thresholds: torch.Tensor,
        lambda_tail: float,
        lambda_under: float,
        tail_scale: float,
    ):
        super().__init__()

        self.register_buffer(
            "thresholds",
            thresholds.reshape(
                1,
                1,
                2,
            ),
        )

        self.lambda_tail = float(
            lambda_tail
        )
        self.lambda_under = float(
            lambda_under
        )
        self.tail_scale = float(
            tail_scale
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        tail_gate = torch.sigmoid(
            (
                target
                - self.thresholds
            )
            / self.tail_scale
        )

        tail_weight = (
            1.0
            + self.lambda_tail
            * tail_gate
        )

        huber = nn.functional.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
        )

        weighted_huber = (
            tail_weight
            * huber
        ).mean()

        under_error = torch.relu(
            target - prediction
        )

        # Important: no baseline weight of 1 is used here.
        # Weak-motion samples receive almost no asymmetric penalty.
        tail_under_loss = (
            tail_gate
            * under_error.pow(2)
        ).mean()

        total_loss = (
            weighted_huber
            + self.lambda_under
            * tail_under_loss
        )

        diagnostics = {
            "weighted_huber": float(
                weighted_huber
                .detach()
                .cpu()
            ),
            "tail_under_loss": float(
                tail_under_loss
                .detach()
                .cpu()
            ),
            "mean_tail_gate": float(
                tail_gate
                .mean()
                .detach()
                .cpu()
            ),
        }

        return total_loss, diagnostics


def compute_tail_thresholds(
    manifest: str | Path,
    split_column: str,
    t0_sec: int,
    quantile: float,
    cache_path: Path,
) -> dict[str, float]:
    """
    Compute tail thresholds only from the training split.
    """
    if cache_path.exists():
        cached = json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )

        same_quantile = math.isclose(
            float(cached["quantile"]),
            float(quantile),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        same_t0 = (
            int(cached["t0_sec"])
            == int(t0_sec)
        )

        if same_quantile and same_t0:
            print(
                f"Reuse tail thresholds: "
                f"{cache_path.resolve()}"
            )
            return cached

    frame = pd.read_csv(
        manifest,
        dtype={"event_id": str},
    )

    train_frame = frame.loc[
        frame[split_column]
        .astype(str)
        .eq("train")
    ]

    if len(train_frame) == 0:
        raise ValueError(
            "Training split is empty."
        )

    all_log_pga: list[np.ndarray] = []
    all_log_pgv: list[np.ndarray] = []

    for event in tqdm(
        train_frame.itertuples(index=False),
        total=len(train_frame),
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
            + int(
                round(
                    t0_sec
                    * sampling_rate
                )
            )
        )
        snapshot_index = min(
            snapshot_index,
            acceleration.shape[-1] - 1,
        )

        pga = horizontal_peak(
            acceleration[
                :,
                :,
                snapshot_index:,
            ]
        )
        pgv = horizontal_peak(
            velocity[
                :,
                :,
                snapshot_index:,
            ]
        )

        all_log_pga.append(
            np.log10(
                np.maximum(pga, 1e-10)
            )
        )
        all_log_pgv.append(
            np.log10(
                np.maximum(pgv, 1e-12)
            )
        )

    concatenated_pga = np.concatenate(
        all_log_pga
    )
    concatenated_pgv = np.concatenate(
        all_log_pgv
    )

    thresholds = {
        "t0_sec": int(t0_sec),
        "quantile": float(quantile),
        "log10_pga_threshold": float(
            np.quantile(
                concatenated_pga,
                quantile,
            )
        ),
        "log10_pgv_threshold": float(
            np.quantile(
                concatenated_pgv,
                quantile,
            )
        ),
        "n_training_station_targets": int(
            len(concatenated_pga)
        ),
    }

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_path.write_text(
        json.dumps(
            thresholds,
            indent=2,
        ),
        encoding="utf-8",
    )

    return thresholds


def safe_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    selected = values[mask]

    if selected.numel() == 0:
        return float("nan")

    return float(
        selected.mean().item()
    )


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thresholds: torch.Tensor,
    magnitudes: torch.Tensor,
    target_triggered: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    magnitudes = magnitudes.float()
    target_triggered = target_triggered.bool()

    residual = prediction - target
    absolute_error = torch.abs(residual)

    threshold_matrix = thresholds.reshape(
        1,
        1,
        2,
    )
    tail_mask = (
        target
        >= threshold_matrix
    )

    event_count = target.shape[0]
    target_count = target.shape[1]

    magnitude_matrix = magnitudes.reshape(
        event_count,
        1,
    ).expand(
        -1,
        target_count,
    )

    m4_mask = (
        torch.isfinite(magnitude_matrix)
        & (magnitude_matrix >= 4.0)
    )

    output: dict[str, float] = {
        "n_events": int(event_count),
        "n_target_rows": int(
            event_count * target_count
        ),
    }

    for quantity_index, quantity_name in enumerate(
        ("pga", "pgv")
    ):
        current_residual = residual[
            ...,
            quantity_index,
        ]
        current_absolute = absolute_error[
            ...,
            quantity_index,
        ]

        output[
            f"mae_log10_{quantity_name}"
        ] = float(
            current_absolute.mean().item()
        )
        output[
            f"bias_log10_{quantity_name}"
        ] = float(
            current_residual.mean().item()
        )
        output[
            f"factor2_{quantity_name}"
        ] = float(
            (
                current_absolute
                <= LOG10_FACTOR_2
            )
            .float()
            .mean()
            .item()
        )

        current_tail_mask = tail_mask[
            ...,
            quantity_index,
        ]
        tail_residual = current_residual[
            current_tail_mask
        ]
        tail_absolute = current_absolute[
            current_tail_mask
        ]

        output[
            f"tail_n_{quantity_name}"
        ] = int(
            current_tail_mask.sum().item()
        )

        if tail_residual.numel() > 0:
            output[
                f"tail_mae_log10_{quantity_name}"
            ] = float(
                tail_absolute.mean().item()
            )
            output[
                f"tail_bias_log10_{quantity_name}"
            ] = float(
                tail_residual.mean().item()
            )
            output[
                f"tail_under03_{quantity_name}"
            ] = float(
                (
                    tail_residual <= -0.3
                )
                .float()
                .mean()
                .item()
            )
            output[
                f"tail_under05_{quantity_name}"
            ] = float(
                (
                    tail_residual <= -0.5
                )
                .float()
                .mean()
                .item()
            )
        else:
            output[
                f"tail_mae_log10_{quantity_name}"
            ] = float("nan")
            output[
                f"tail_bias_log10_{quantity_name}"
            ] = float("nan")
            output[
                f"tail_under03_{quantity_name}"
            ] = float("nan")
            output[
                f"tail_under05_{quantity_name}"
            ] = float("nan")

        output[
            f"m4plus_n_{quantity_name}"
        ] = int(
            m4_mask.sum().item()
        )
        output[
            f"m4plus_mae_log10_{quantity_name}"
        ] = safe_mean(
            current_absolute,
            m4_mask,
        )
        output[
            f"m4plus_bias_log10_{quantity_name}"
        ] = safe_mean(
            current_residual,
            m4_mask,
        )

        output[
            f"triggered_n_{quantity_name}"
        ] = int(
            target_triggered.sum().item()
        )
        output[
            f"triggered_mae_log10_{quantity_name}"
        ] = safe_mean(
            current_absolute,
            target_triggered,
        )
        output[
            f"triggered_bias_log10_{quantity_name}"
        ] = safe_mean(
            current_residual,
            target_triggered,
        )

        not_triggered = ~target_triggered
        output[
            f"not_triggered_n_{quantity_name}"
        ] = int(
            not_triggered.sum().item()
        )
        output[
            f"not_triggered_mae_log10_{quantity_name}"
        ] = safe_mean(
            current_absolute,
            not_triggered,
        )
        output[
            f"not_triggered_bias_log10_{quantity_name}"
        ] = safe_mean(
            current_residual,
            not_triggered,
        )

    return output


def compute_selection_scores(
    metrics: dict[str, float],
) -> dict[str, float]:
    overall_score = 0.5 * (
        metrics["mae_log10_pga"]
        + metrics["mae_log10_pgv"]
    )

    tail_score = 0.5 * (
        metrics["tail_mae_log10_pga"]
        + metrics["tail_mae_log10_pgv"]
    )

    absolute_bias_score = 0.5 * (
        abs(metrics["bias_log10_pga"])
        + abs(metrics["bias_log10_pgv"])
    )

    compromise_score = (
        0.60 * overall_score
        + 0.30 * tail_score
        + 0.10 * absolute_bias_score
    )

    return {
        "overall_score": float(
            overall_score
        ),
        "tail_score": float(
            tail_score
        ),
        "absolute_bias_score": float(
            absolute_bias_score
        ),
        "compromise_score": float(
            compromise_score
        ),
    }


def flatten_prediction_rows(
    event_ids: list[str],
    target_station_ids: Any,
    prediction: torch.Tensor,
    target: torch.Tensor,
    magnitudes: torch.Tensor,
    target_triggered: torch.Tensor,
    target_p_offset: torch.Tensor,
) -> list[dict[str, Any]]:
    prediction_array = prediction.numpy()
    target_array = target.numpy()
    magnitude_array = magnitudes.numpy()
    triggered_array = target_triggered.numpy()
    p_offset_array = target_p_offset.numpy()

    batch_size = prediction_array.shape[0]
    target_count = prediction_array.shape[1]

    # Default DataLoader collates list[str] per target position into
    # a list of tuples. This converts it back to [batch, target].
    station_id_matrix: list[list[str]] = [
        [""] * target_count
        for _ in range(batch_size)
    ]

    if isinstance(target_station_ids, list):
        if (
            len(target_station_ids)
            == target_count
            and all(
                isinstance(item, (list, tuple))
                for item in target_station_ids
            )
        ):
            for target_index, column in enumerate(
                target_station_ids
            ):
                for batch_index, station_id in enumerate(
                    column
                ):
                    station_id_matrix[
                        batch_index
                    ][target_index] = str(
                        station_id
                    )
        elif (
            len(target_station_ids)
            == batch_size
        ):
            for batch_index, row in enumerate(
                target_station_ids
            ):
                for target_index, station_id in enumerate(
                    row
                ):
                    station_id_matrix[
                        batch_index
                    ][target_index] = str(
                        station_id
                    )

    rows: list[dict[str, Any]] = []

    for batch_index in range(batch_size):
        for target_index in range(target_count):
            true_pga = float(
                target_array[
                    batch_index,
                    target_index,
                    0,
                ]
            )
            true_pgv = float(
                target_array[
                    batch_index,
                    target_index,
                    1,
                ]
            )
            predicted_pga = float(
                prediction_array[
                    batch_index,
                    target_index,
                    0,
                ]
            )
            predicted_pgv = float(
                prediction_array[
                    batch_index,
                    target_index,
                    1,
                ]
            )

            rows.append({
                "event_id": str(
                    event_ids[batch_index]
                ),
                "target_station_id": (
                    station_id_matrix[
                        batch_index
                    ][target_index]
                ),
                "magnitude": float(
                    magnitude_array[batch_index]
                ),
                "target_triggered_by_snapshot": bool(
                    triggered_array[
                        batch_index,
                        target_index,
                    ]
                ),
                "target_p_offset_sec": float(
                    p_offset_array[
                        batch_index,
                        target_index,
                    ]
                ),
                "true_log10_pga": true_pga,
                "pred_log10_pga": predicted_pga,
                "error_log10_pga": (
                    predicted_pga - true_pga
                ),
                "true_log10_pgv": true_pgv,
                "pred_log10_pgv": predicted_pgv,
                "error_log10_pgv": (
                    predicted_pgv - true_pgv
                ),
            })

    return rows


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: TailGatedLoss,
    thresholds: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    collect_predictions: bool = False,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_magnitudes: list[torch.Tensor] = []
    all_triggered: list[torch.Tensor] = []
    prediction_rows: list[dict[str, Any]] = []

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

        prediction_cpu = (
            prediction.detach().cpu()
        )
        target_cpu = (
            target.detach().cpu()
        )
        magnitude_cpu = (
            batch["magnitude"]
            .detach()
            .cpu()
        )
        triggered_cpu = (
            batch["target_triggered"]
            .detach()
            .cpu()
        )

        all_predictions.append(
            prediction_cpu
        )
        all_targets.append(
            target_cpu
        )
        all_magnitudes.append(
            magnitude_cpu
        )
        all_triggered.append(
            triggered_cpu
        )

        if collect_predictions:
            prediction_rows.extend(
                flatten_prediction_rows(
                    event_ids=list(
                        batch["event_id"]
                    ),
                    target_station_ids=(
                        batch[
                            "target_station_id"
                        ]
                    ),
                    prediction=prediction_cpu,
                    target=target_cpu,
                    magnitudes=magnitude_cpu,
                    target_triggered=(
                        triggered_cpu
                    ),
                    target_p_offset=(
                        batch[
                            "target_p_offset"
                        ]
                        .detach()
                        .cpu()
                    ),
                )
            )

    prediction_tensor = torch.cat(
        all_predictions,
        dim=0,
    )
    target_tensor = torch.cat(
        all_targets,
        dim=0,
    )
    magnitude_tensor = torch.cat(
        all_magnitudes,
        dim=0,
    )
    triggered_tensor = torch.cat(
        all_triggered,
        dim=0,
    )

    metrics = compute_metrics(
        prediction=prediction_tensor,
        target=target_tensor,
        thresholds=thresholds.cpu(),
        magnitudes=magnitude_tensor,
        target_triggered=triggered_tensor,
    )

    metrics["loss"] = float(
        total_loss
        / max(total_batches, 1)
    )
    metrics.update(
        compute_selection_scores(metrics)
    )

    prediction_frame = None
    if collect_predictions:
        prediction_frame = pd.DataFrame(
            prediction_rows
        )

    return metrics, prediction_frame


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    variant: str,
    preset: dict[str, float],
    args: argparse.Namespace,
    thresholds: dict[str, float],
    validation_metrics: dict[str, float],
    selection_name: str,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "epoch": int(epoch),
            "variant": variant,
            "preset": preset,
            "selection_name": selection_name,
            "args": vars(args),
            "thresholds": thresholds,
            "validation_metrics": (
                validation_metrics
            ),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )
    return checkpoint


def create_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    SparseFieldDataset,
    SparseFieldDataset,
    SparseFieldDataset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    train_dataset = SparseFieldDataset(
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

    validation_dataset = SparseFieldDataset(
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

    test_dataset = SparseFieldDataset(
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
    loader_generator.manual_seed(
        args.seed
    )

    common_loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (
            device.type == "cuda"
        ),
        # Keep disabled so dataset.set_epoch() is visible to workers.
        "persistent_workers": False,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
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

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
    )


def train_variant(
    variant: str,
    args: argparse.Namespace,
    thresholds: dict[str, float],
    device: torch.device,
) -> list[dict[str, Any]]:
    preset = PRESETS[variant]

    set_global_seed(args.seed)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
    ) = create_loaders(
        args,
        device,
    )

    model = MeanPoolingSparseFieldModel(
        hidden_dim=args.hidden_dim
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

    loss_function = TailGatedLoss(
        thresholds=threshold_tensor,
        lambda_tail=(
            preset["lambda_tail"]
        ),
        lambda_under=(
            preset["lambda_under"]
        ),
        tail_scale=args.tail_scale,
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

    variant_directory = (
        Path(args.out_root)
        / variant
    )
    variant_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_paths = {
        selection: (
            variant_directory
            / f"best_{selection}.pt"
        )
        for selection in CHECKPOINT_SELECTIONS
    }

    best_scores = {
        selection: float("inf")
        for selection in CHECKPOINT_SELECTIONS
    }
    best_epochs = {
        selection: 0
        for selection in CHECKPOINT_SELECTIONS
    }

    history: list[dict[str, Any]] = []
    epochs_without_compromise_improvement = 0
    start_time = time.time()

    print(
        f"\n=== Variant: {variant} ==="
    )
    print(f"Device             : {device}")
    print(
        f"lambda_tail        : "
        f"{preset['lambda_tail']}"
    )
    print(
        f"lambda_under       : "
        f"{preset['lambda_under']}"
    )
    print(
        f"Tail scale         : "
        f"{args.tail_scale}"
    )
    print(
        f"Train/Val/Test     : "
        f"{len(train_dataset)}/"
        f"{len(validation_dataset)}/"
        f"{len(test_dataset)}"
    )
    print(
        f"Parameters         : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        "Tail thresholds    : "
        f"PGA={thresholds['log10_pga_threshold']:.4f}, "
        f"PGV={thresholds['log10_pgv_threshold']:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)

        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            thresholds=threshold_tensor,
            device=device,
            optimizer=optimizer,
            collect_predictions=False,
        )

        validation_metrics, _ = run_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            thresholds=threshold_tensor,
            device=device,
            optimizer=None,
            collect_predictions=False,
        )

        scheduler.step(
            validation_metrics[
                "compromise_score"
            ]
        )

        epoch_row = {
            "variant": variant,
            "epoch": int(epoch),
            "learning_rate": float(
                optimizer.param_groups[0][
                    "lr"
                ]
            ),
            "lambda_tail": float(
                preset["lambda_tail"]
            ),
            "lambda_under": float(
                preset["lambda_under"]
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
        history.append(epoch_row)

        pd.DataFrame(history).to_csv(
            variant_directory
            / "history.csv",
            index=False,
        )

        score_mapping = {
            "overall": validation_metrics[
                "overall_score"
            ],
            "tail": validation_metrics[
                "tail_score"
            ],
            "compromise": validation_metrics[
                "compromise_score"
            ],
        }

        compromise_improved = False

        for selection_name, current_score in (
            score_mapping.items()
        ):
            if (
                current_score
                < best_scores[
                    selection_name
                ]
                - args.min_delta
            ):
                best_scores[
                    selection_name
                ] = float(current_score)
                best_epochs[
                    selection_name
                ] = int(epoch)

                save_checkpoint(
                    path=checkpoint_paths[
                        selection_name
                    ],
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    variant=variant,
                    preset=preset,
                    args=args,
                    thresholds=thresholds,
                    validation_metrics=(
                        validation_metrics
                    ),
                    selection_name=(
                        selection_name
                    ),
                )

                if (
                    selection_name
                    == "compromise"
                ):
                    compromise_improved = True

        if compromise_improved:
            epochs_without_compromise_improvement = 0
        else:
            epochs_without_compromise_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"val overall={validation_metrics['overall_score']:.4f} | "
            f"tail={validation_metrics['tail_score']:.4f} | "
            f"comp={validation_metrics['compromise_score']:.4f} | "
            f"PGA={validation_metrics['mae_log10_pga']:.4f} | "
            f"PGV={validation_metrics['mae_log10_pgv']:.4f} | "
            f"tail PGA={validation_metrics['tail_mae_log10_pga']:.4f} | "
            f"tail PGV={validation_metrics['tail_mae_log10_pgv']:.4f} | "
            f"bias PGA={validation_metrics['bias_log10_pga']:+.4f} | "
            f"bias PGV={validation_metrics['bias_log10_pgv']:+.4f}"
        )

        if (
            epochs_without_compromise_improvement
            >= args.early_stopping_patience
        ):
            print(
                "Early stopping based on "
                f"compromise score at epoch {epoch}; "
                f"best compromise epoch="
                f"{best_epochs['compromise']}."
            )
            break

    result_rows: list[dict[str, Any]] = []

    for selection_name in (
        CHECKPOINT_SELECTIONS
    ):
        checkpoint_path = checkpoint_paths[
            selection_name
        ]

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: "
                f"{checkpoint_path}"
            )

        checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            device,
        )

        test_metrics, predictions = run_epoch(
            model=model,
            loader=test_loader,
            loss_function=loss_function,
            thresholds=threshold_tensor,
            device=device,
            optimizer=None,
            collect_predictions=True,
        )

        if predictions is None:
            raise RuntimeError(
                "Prediction export failed."
            )

        predictions[
            "variant"
        ] = variant
        predictions[
            "checkpoint_selection"
        ] = selection_name
        predictions[
            "checkpoint_epoch"
        ] = int(
            checkpoint["epoch"]
        )

        predictions.to_csv(
            variant_directory
            / (
                f"test_predictions_"
                f"best_{selection_name}.csv"
            ),
            index=False,
        )

        result = {
            "variant": variant,
            "checkpoint_selection": (
                selection_name
            ),
            "checkpoint_epoch": int(
                checkpoint["epoch"]
            ),
            "lambda_tail": float(
                preset["lambda_tail"]
            ),
            "lambda_under": float(
                preset["lambda_under"]
            ),
            "tail_scale": float(
                args.tail_scale
            ),
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter
                    in model.parameters()
                )
            ),
            "training_seconds_total": float(
                time.time() - start_time
            ),
            **{
                f"test_{key}": value
                for key, value
                in test_metrics.items()
            },
        }

        (
            variant_directory
            / (
                f"test_metrics_"
                f"best_{selection_name}.json"
            )
        ).write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

        result_rows.append(result)

        print(
            f"\nTest [{selection_name}] | "
            f"epoch={checkpoint['epoch']} | "
            f"PGA={test_metrics['mae_log10_pga']:.4f} | "
            f"PGV={test_metrics['mae_log10_pgv']:.4f} | "
            f"tail PGA={test_metrics['tail_mae_log10_pga']:.4f} | "
            f"tail PGV={test_metrics['tail_mae_log10_pgv']:.4f} | "
            f"tail PGA bias={test_metrics['tail_bias_log10_pga']:+.4f} | "
            f"tail PGV bias={test_metrics['tail_bias_log10_pgv']:+.4f} | "
            f"overall PGA bias={test_metrics['bias_log10_pga']:+.4f} | "
            f"overall PGV bias={test_metrics['bias_log10_pgv']:+.4f}"
        )

    pd.DataFrame(
        result_rows
    ).to_csv(
        variant_directory
        / "checkpoint_test_summary.csv",
        index=False,
    )

    return result_rows


def parse_variants(
    text: str,
) -> list[str]:
    variants = [
        item.strip()
        for item in str(text).split(",")
        if item.strip()
    ]

    if not variants:
        raise ValueError(
            "At least one variant is required."
        )

    unknown = set(variants).difference(
        PRESETS
    )
    if unknown:
        raise ValueError(
            f"Unknown variants: {sorted(unknown)}. "
            f"Valid variants: {sorted(PRESETS)}"
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
            "tail_gated_ultralight"
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
        "--tail-scale",
        type=float,
        default=0.20,
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
        default=14,
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
        "--torch-threads",
        type=int,
        default=0,
        help=(
            "CPU thread count. "
            "Use 0 to keep the PyTorch default."
        ),
    )
    parser.add_argument(
        "--out-root",
        default=(
            "runs/"
            "tail_gated_compromise_t0_5s_k5"
        ),
    )

    args = parser.parse_args()
    variants = parse_variants(
        args.variants
    )

    if not (
        0.5
        < args.tail_quantile
        < 1.0
    ):
        raise ValueError(
            "--tail-quantile must be "
            "between 0.5 and 1.0."
        )

    if args.tail_scale <= 0:
        raise ValueError(
            "--tail-scale must be positive."
        )

    if args.torch_threads > 0:
        torch.set_num_threads(
            args.torch_threads
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
                "CUDA was requested but is unavailable."
            )

        device = torch.device(
            args.device
        )

    output_root = Path(args.out_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_cache_path = (
        output_root
        / (
            f"tail_thresholds_"
            f"q{args.tail_quantile:.2f}_"
            f"t0_{args.t0_sec}s.json"
        )
    )

    thresholds = compute_tail_thresholds(
        manifest=args.manifest,
        split_column=args.split_column,
        t0_sec=args.t0_sec,
        quantile=args.tail_quantile,
        cache_path=threshold_cache_path,
    )

    configuration = {
        **vars(args),
        "variants": variants,
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(
            torch.cuda.is_available()
        ),
        "preset_configuration": {
            variant: PRESETS[variant]
            for variant in variants
        },
        "tail_thresholds": thresholds,
        "selection_formula": {
            "overall_score": (
                "0.5 * (PGA_MAE + PGV_MAE)"
            ),
            "tail_score": (
                "0.5 * "
                "(tail_PGA_MAE + tail_PGV_MAE)"
            ),
            "compromise_score": (
                "0.60 * overall_score "
                "+ 0.30 * tail_score "
                "+ 0.10 * mean_absolute_bias"
            ),
        },
    }

    (
        output_root
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    all_results: list[dict[str, Any]] = []

    print(
        "=== Tail-gated compromise suite ==="
    )
    print(f"Device             : {device}")
    print(
        f"Variants           : "
        f"{', '.join(variants)}"
    )
    print(
        f"Output root        : "
        f"{output_root.resolve()}"
    )

    for variant in variants:
        variant_results = train_variant(
            variant=variant,
            args=args,
            thresholds=thresholds,
            device=device,
        )

        all_results.extend(
            variant_results
        )

        pd.DataFrame(
            all_results
        ).to_csv(
            output_root
            / "all_results_partial.csv",
            index=False,
        )

    summary = pd.DataFrame(
        all_results
    )

    summary.to_csv(
        output_root
        / "all_checkpoint_results.csv",
        index=False,
    )

    compromise_summary = summary.loc[
        summary[
            "checkpoint_selection"
        ].eq("compromise")
    ].copy()

    compromise_summary = (
        compromise_summary.sort_values(
            [
                "test_compromise_score",
                "test_overall_score",
            ]
        )
    )

    compromise_summary.to_csv(
        output_root
        / "compromise_checkpoint_summary.csv",
        index=False,
    )

    display_columns = [
        "variant",
        "checkpoint_epoch",
        "lambda_tail",
        "lambda_under",
        "test_mae_log10_pga",
        "test_mae_log10_pgv",
        "test_bias_log10_pga",
        "test_bias_log10_pgv",
        "test_tail_mae_log10_pga",
        "test_tail_mae_log10_pgv",
        "test_tail_bias_log10_pga",
        "test_tail_bias_log10_pgv",
        "test_tail_under05_pga",
        "test_tail_under05_pgv",
        "test_m4plus_mae_log10_pga",
        "test_m4plus_mae_log10_pgv",
        "test_compromise_score",
    ]

    print(
        "\n=== Best-compromise checkpoints ==="
    )
    print(
        compromise_summary[
            display_columns
        ].to_string(index=False)
    )

    print(
        f"\nOutputs: "
        f"{output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
