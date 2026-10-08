#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 / Step 3
Strong-baseline suite for Causal-SeisField under the SAME causal protocol.

Compared methods
----------------
Non-learned / propagation baselines:
  median_observed
  nearest_observed
  idw_observed
  plum_like
      A deliberately adapted PLUM-like causal propagation baseline:
      distance-tapered maximum of PRE-SNAPSHOT observed input motion.
      The distance scale is selected on validation only; for each candidate
      scale an additive log10 calibration offset is fitted on training only.
      This is NOT claimed to be the original PLUM algorithm.

Learned baselines:
  base
      Final mean-pooling Causal-SeisField Base.
  cross_attention
      Target-conditioned cross-station Transformer-style baseline.
  graph
      Geometry-aware message-passing graph baseline.

Fair-comparison rules
---------------------
- Same grouped event split.
- Same T0, K, Q and input time window.
- Same deterministic input/target station draws across all learned models.
- Same repeated test draws across ALL methods.
- No magnitude or hypocenter is used by the learned/propagation methods.
- Test set is never used for tuning.
- Main metrics aggregate:
      targets -> repeats -> events.
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


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)

LEARNED_VARIANTS = {
    "base",
    "cross_attention",
    "graph",
}


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{text}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def set_seed(seed: int) -> None:
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


def resolve_h5_path(
    event_id: str,
    raw_path: str,
    h5_root: Path | None,
) -> Path:
    raw_path = str(raw_path).strip()
    if raw_path:
        p = Path(raw_path)
        if p.exists():
            return p

    if h5_root is not None:
        p = h5_root / f"{event_id}.h5"
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Cannot resolve HDF5 for event {event_id}; "
        f"manifest path={raw_path!r}; h5_root={h5_root}"
    )


def local_xy_km(
    coordinates: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
) -> np.ndarray:
    lat = coordinates[:, 0]
    lon = coordinates[:, 1]
    x = (
        (lon - origin_longitude)
        * 111.32
        * math.cos(math.radians(origin_latitude))
    )
    y = (lat - origin_latitude) * 110.57
    return np.stack([x, y], axis=-1)


def horizontal_peak(data: np.ndarray) -> np.ndarray:
    return np.max(
        np.sqrt(
            data[:, 0, :] ** 2
            + data[:, 1, :] ** 2
        ),
        axis=1,
    )


def pairwise_distance_km(
    target_xy: np.ndarray,
    input_xy: np.ndarray,
) -> np.ndarray:
    delta = (
        target_xy[:, None, :]
        - input_xy[None, :, :]
    )
    return np.sqrt(
        np.sum(delta ** 2, axis=-1)
    )


class StrongBaselineDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        h5_root: str | Path | None,
        split_column: str,
        split_name: str,
        t0_sec: int,
        input_stations: int,
        target_stations: int,
        input_pre_sec: float,
        seed: int,
        training: bool,
        repeats: int = 1,
    ):
        frame = pd.read_csv(
            manifest,
            dtype={"event_id": str},
        )
        required = {
            "event_id",
            "h5_path",
            split_column,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Manifest missing columns: {sorted(missing)}"
            )

        frame = frame.loc[
            frame[split_column]
            .astype(str)
            .eq(str(split_name))
        ].copy()

        eligible_column = (
            f"eligible_t0_{int(t0_sec)}s_k{int(input_stations)}"
        )
        if eligible_column in frame.columns:
            frame = frame.loc[
                as_bool(frame[eligible_column])
            ].copy()

        self.frame = frame.reset_index(drop=True)
        self.h5_root = (
            Path(h5_root)
            if h5_root is not None
            and str(h5_root).strip()
            else None
        )
        self.split_name = str(split_name)
        self.t0_sec = int(t0_sec)
        self.input_stations = int(input_stations)
        self.target_stations = int(target_stations)
        self.input_pre_sec = float(input_pre_sec)
        self.seed = int(seed)
        self.training = bool(training)
        self.repeats = (
            1 if training else int(repeats)
        )
        self.epoch = 0

        if len(self.frame) == 0:
            raise ValueError(
                f"No eligible events for {split_name}."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (
            len(self.frame)
            * self.repeats
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            event_index = int(index)
            repeat_index = 0
            suffix = f"epoch:{self.epoch}"
        else:
            event_index = int(
                index // self.repeats
            )
            repeat_index = int(
                index % self.repeats
            )
            suffix = f"repeat:{repeat_index}"

        event = self.frame.iloc[
            event_index
        ]
        event_id = str(
            event["event_id"]
        )

        rng = np.random.default_rng(
            stable_seed(
                (
                    f"strong:{self.split_name}:"
                    f"{event_id}:{suffix}"
                ),
                self.seed,
            )
        )

        h5_path = resolve_h5_path(
            event_id,
            str(event["h5_path"]),
            self.h5_root,
        )

        with h5py.File(
            h5_path,
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

        triggered = np.flatnonzero(
            np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (
                p_offset
                <= float(self.t0_sec)
            )
        )
        if (
            len(triggered)
            < self.input_stations
        ):
            raise RuntimeError(
                f"Event {event_id}: only "
                f"{len(triggered)} triggered stations."
            )

        input_indices = rng.choice(
            triggered,
            size=self.input_stations,
            replace=False,
        )

        target_pool = np.setdiff1d(
            np.arange(
                len(coordinates)
            ),
            input_indices,
            assume_unique=False,
        )
        if (
            len(target_pool)
            < self.target_stations
        ):
            raise RuntimeError(
                f"Event {event_id}: insufficient targets."
            )

        target_indices = rng.choice(
            target_pool,
            size=self.target_stations,
            replace=False,
        )

        input_start = max(
            0,
            int(
                round(
                    (
                        pre_first_p
                        - self.input_pre_sec
                    )
                    * sampling_rate
                )
            ),
        )
        snapshot = min(
            time_zero_index
            + int(
                round(
                    self.t0_sec
                    * sampling_rate
                )
            ),
            acceleration.shape[-1] - 1,
        )

        observed_acceleration = acceleration[
            input_indices,
            :,
            input_start:snapshot,
        ]
        observed_velocity = velocity[
            input_indices,
            :,
            input_start:snapshot,
        ]

        observed_pga = np.maximum(
            horizontal_peak(
                observed_acceleration
            ),
            1e-10,
        )
        observed_pgv = np.maximum(
            horizontal_peak(
                observed_velocity
            ),
            1e-12,
        )

        input_waveforms = (
            np.sign(
                observed_acceleration
            )
            * np.log1p(
                np.abs(
                    observed_acceleration
                ) / 1e-3
            )
        ).astype(np.float32)

        input_coordinates = coordinates[
            input_indices
        ]
        target_coordinates = coordinates[
            target_indices
        ]

        origin_latitude = float(
            input_coordinates[:, 0].mean()
        )
        origin_longitude = float(
            input_coordinates[:, 1].mean()
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
                input_coordinates[
                    :, 2:3
                ] / 2000.0,
                p_offset[
                    input_indices,
                    None,
                ]
                / max(
                    float(self.t0_sec),
                    1.0,
                ),
            ],
            axis=1,
        ).astype(np.float32)

        target_features = np.concatenate(
            [
                target_xy / 100.0,
                target_coordinates[
                    :, 2:3
                ] / 2000.0,
            ],
            axis=1,
        ).astype(np.float32)

        future_pga = horizontal_peak(
            acceleration[
                target_indices,
                :,
                snapshot:,
            ]
        )
        future_pgv = horizontal_peak(
            velocity[
                target_indices,
                :,
                snapshot:,
            ]
        )

        target_log = np.stack(
            [
                np.log10(
                    np.maximum(
                        future_pga,
                        1e-10,
                    )
                ),
                np.log10(
                    np.maximum(
                        future_pgv,
                        1e-12,
                    )
                ),
            ],
            axis=-1,
        ).astype(np.float32)

        distances = pairwise_distance_km(
            target_xy,
            input_xy,
        ).astype(np.float32)

        return {
            "event_id": event_id,
            "repeat": torch.tensor(
                repeat_index,
                dtype=torch.int64,
            ),
            "target_slot": torch.arange(
                self.target_stations,
                dtype=torch.int64,
            ),
            "target_station_index": torch.from_numpy(
                target_indices.astype(
                    np.int64
                )
            ),
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
            "observed_log10_pga": torch.from_numpy(
                np.log10(
                    observed_pga
                ).astype(np.float32)
            ),
            "observed_log10_pgv": torch.from_numpy(
                np.log10(
                    observed_pgv
                ).astype(np.float32)
            ),
            "distances_km": torch.from_numpy(
                distances
            ),
        }


class WaveformEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                3, 32,
                kernel_size=9,
                stride=2,
                padding=4,
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(
                32, 64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(
                64, 128,
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
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class StationEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.waveform_encoder = (
            WaveformEncoder(
                hidden_dim
            )
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

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
    ) -> torch.Tensor:
        b, k, c, t = (
            input_waveforms.shape
        )
        wave = self.waveform_encoder(
            input_waveforms.reshape(
                b * k,
                c,
                t,
            )
        ).reshape(
            b,
            k,
            -1,
        )
        return self.station_encoder(
            torch.cat(
                [
                    wave,
                    input_features,
                ],
                dim=-1,
            )
        )


class MeanPoolingBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = StationEncoder(
            hidden_dim
        )
        self.decoder = nn.Sequential(
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
        station = self.encoder(
            input_waveforms,
            input_features,
        )
        event = station.mean(
            dim=1
        )
        event = event[
            :, None, :
        ].expand(
            -1,
            target_features.shape[1],
            -1,
        )
        return self.decoder(
            torch.cat(
                [
                    event,
                    target_features,
                ],
                dim=-1,
            )
        )


class CrossAttentionBaseline(nn.Module):
    """
    Transformer-style set baseline:
    station self-attention + target-conditioned cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
    ):
        super().__init__()
        self.encoder = StationEncoder(
            hidden_dim
        )
        self.station_attention = (
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=heads,
                batch_first=True,
            )
        )
        self.station_norm = nn.LayerNorm(
            hidden_dim
        )

        self.query_encoder = nn.Sequential(
            nn.Linear(
                3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )
        self.cross_attention = (
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=heads,
                batch_first=True,
            )
        )
        self.cross_norm = nn.LayerNorm(
            hidden_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
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
        station = self.encoder(
            input_waveforms,
            input_features,
        )
        attended, _ = (
            self.station_attention(
                station,
                station,
                station,
                need_weights=False,
            )
        )
        station = self.station_norm(
            station + attended
        )

        query = self.query_encoder(
            target_features
        )
        context, _ = (
            self.cross_attention(
                query,
                station,
                station,
                need_weights=False,
            )
        )
        context = self.cross_norm(
            query + context
        )

        return self.decoder(
            torch.cat(
                [
                    context,
                    query,
                ],
                dim=-1,
            )
        )


class GraphBaseline(nn.Module):
    """
    Geometry-aware message passing over input stations, followed by
    geometry-weighted target readout.
    """

    def __init__(
        self,
        hidden_dim: int,
        initial_length_scale_km: float,
    ):
        super().__init__()
        self.encoder = StationEncoder(
            hidden_dim
        )

        self.self_layers = nn.ModuleList(
            [
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                )
                for _ in range(2)
            ]
        )
        self.neighbor_layers = nn.ModuleList(
            [
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                )
                for _ in range(2)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    hidden_dim
                )
                for _ in range(2)
            ]
        )

        self.log_length_scale = (
            nn.Parameter(
                torch.tensor(
                    math.log(
                        initial_length_scale_km
                    ),
                    dtype=torch.float32,
                )
            )
        )

        self.decoder = nn.Sequential(
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

    def _scale(self) -> torch.Tensor:
        return torch.clamp(
            torch.exp(
                self.log_length_scale
            ),
            min=5.0,
            max=500.0,
        )

    def forward(
        self,
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        station = self.encoder(
            input_waveforms,
            input_features,
        )

        # x/y features are scaled by 100 km in the dataset.
        input_xy = (
            input_features[
                ..., :2
            ] * 100.0
        )
        target_xy = (
            target_features[
                ..., :2
            ] * 100.0
        )

        delta = (
            input_xy[:, :, None, :]
            - input_xy[:, None, :, :]
        )
        distance = torch.sqrt(
            torch.sum(
                delta ** 2,
                dim=-1,
            )
            + 1e-8
        )

        scale = self._scale()
        adjacency = torch.softmax(
            -distance / scale,
            dim=-1,
        )

        for self_layer, neighbor_layer, norm in zip(
            self.self_layers,
            self.neighbor_layers,
            self.norms,
        ):
            neighbor = torch.matmul(
                adjacency,
                station,
            )
            updated = torch.nn.functional.gelu(
                self_layer(station)
                + neighbor_layer(neighbor)
            )
            station = norm(
                station + updated
            )

        target_delta = (
            target_xy[:, :, None, :]
            - input_xy[:, None, :, :]
        )
        target_distance = torch.sqrt(
            torch.sum(
                target_delta ** 2,
                dim=-1,
            )
            + 1e-8
        )
        target_weights = torch.softmax(
            -target_distance / scale,
            dim=-1,
        )
        context = torch.matmul(
            target_weights,
            station,
        )

        return self.decoder(
            torch.cat(
                [
                    context,
                    target_features,
                ],
                dim=-1,
            )
        )


def make_model(
    variant: str,
    hidden_dim: int,
    attention_heads: int,
    graph_length_scale_km: float,
) -> nn.Module:
    if variant == "base":
        return MeanPoolingBase(
            hidden_dim
        )
    if variant == "cross_attention":
        return CrossAttentionBaseline(
            hidden_dim,
            attention_heads,
        )
    if variant == "graph":
        return GraphBaseline(
            hidden_dim,
            graph_length_scale_km,
        )
    raise ValueError(variant)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0

    for batch in loader:
        target = batch[
            "target_log"
        ].to(device)
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
        )

        loss = (
            nn.functional
            .smooth_l1_loss(
                prediction,
                target,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        optimizer.step()

        n = int(target.numel())
        total += (
            float(
                loss.detach().cpu()
            ) * n
        )
        count += n

    return total / max(
        count,
        1,
    )


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []

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

            truth = batch[
                "target_log"
            ].numpy()
            repeats = batch[
                "repeat"
            ].numpy()
            target_indices = batch[
                "target_station_index"
            ].numpy()
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
                                repeats[b]
                            ),
                            "target_slot": int(q),
                            "target_station_index": int(
                                target_indices[
                                    b, q
                                ]
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


def event_balanced_mae(
    frame: pd.DataFrame,
    quantity: str,
    prediction_column: str,
) -> float:
    absolute = np.abs(
        frame[
            prediction_column
        ].to_numpy(dtype=float)
        - frame[
            f"true_log10_{quantity}"
        ].to_numpy(dtype=float)
    )

    work = frame[
        ["event_id", "repeat"]
    ].copy()
    work["absolute"] = absolute

    er = (
        work.groupby(
            ["event_id", "repeat"],
            sort=False,
        )["absolute"]
        .mean()
        .reset_index()
    )
    ev = (
        er.groupby(
            "event_id",
            sort=False,
        )["absolute"]
        .mean()
    )
    return float(
        ev.mean()
    )


def collect_causal_baselines(
    loader: DataLoader,
    idw_power: float,
) -> pd.DataFrame:
    rows = []

    for batch in loader:
        truth = batch[
            "target_log"
        ].numpy()
        observed_pga = batch[
            "observed_log10_pga"
        ].numpy()
        observed_pgv = batch[
            "observed_log10_pgv"
        ].numpy()
        distances = batch[
            "distances_km"
        ].numpy()
        repeats = batch[
            "repeat"
        ].numpy()
        target_indices = batch[
            "target_station_index"
        ].numpy()
        event_ids = list(
            batch["event_id"]
        )

        for b in range(
            truth.shape[0]
        ):
            distance = distances[b]
            nearest = np.argmin(
                distance,
                axis=1,
            )

            # Linear-amplitude IDW, then back to log10.
            pga_linear = (
                10.0 ** observed_pga[b]
            )
            pgv_linear = (
                10.0 ** observed_pgv[b]
            )
            weights = 1.0 / np.maximum(
                distance,
                1e-3,
            ) ** float(idw_power)
            weights /= weights.sum(
                axis=1,
                keepdims=True,
            )

            idw_pga = np.log10(
                np.maximum(
                    weights @ pga_linear,
                    1e-10,
                )
            )
            idw_pgv = np.log10(
                np.maximum(
                    weights @ pgv_linear,
                    1e-12,
                )
            )

            median_pga = float(
                np.median(
                    observed_pga[b]
                )
            )
            median_pgv = float(
                np.median(
                    observed_pgv[b]
                )
            )

            for q in range(
                truth.shape[1]
            ):
                rows.append(
                    {
                        "event_id": str(
                            event_ids[b]
                        ),
                        "repeat": int(
                            repeats[b]
                        ),
                        "target_slot": int(q),
                        "target_station_index": int(
                            target_indices[
                                b, q
                            ]
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
                        "median_observed_log10_pga": (
                            median_pga
                        ),
                        "median_observed_log10_pgv": (
                            median_pgv
                        ),
                        "nearest_observed_log10_pga": float(
                            observed_pga[
                                b,
                                nearest[q],
                            ]
                        ),
                        "nearest_observed_log10_pgv": float(
                            observed_pgv[
                                b,
                                nearest[q],
                            ]
                        ),
                        "idw_observed_log10_pga": float(
                            idw_pga[q]
                        ),
                        "idw_observed_log10_pgv": float(
                            idw_pgv[q]
                        ),
                        "observed_log10_pga_vector": "|".join(
                            f"{x:.9g}"
                            for x in observed_pga[b]
                        ),
                        "observed_log10_pgv_vector": "|".join(
                            f"{x:.9g}"
                            for x in observed_pgv[b]
                        ),
                        "distance_km_vector": "|".join(
                            f"{x:.9g}"
                            for x in distance[q]
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def parse_vector(text: str) -> np.ndarray:
    return np.asarray(
        [
            float(x)
            for x in str(text).split("|")
        ],
        dtype=float,
    )


def plum_raw_prediction(
    observed_log: np.ndarray,
    distances_km: np.ndarray,
    length_scale_km: float,
) -> float:
    if math.isinf(
        length_scale_km
    ):
        adjusted = observed_log
    else:
        # log10(A * exp(-d/L)) = log10(A) - d/(L ln 10)
        adjusted = (
            observed_log
            - distances_km
            / (
                length_scale_km
                * math.log(10.0)
            )
        )
    return float(
        np.max(adjusted)
    )


def fit_plum(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    scales: list[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for quantity in (
        "pga",
        "pgv",
    ):
        train_raw_by_scale = {}
        offsets = {}
        validation_scores = {}

        for scale in scales:
            train_raw = np.empty(
                len(train_frame),
                dtype=float,
            )
            for i, row in enumerate(
                train_frame.itertuples(
                    index=False
                )
            ):
                obs = parse_vector(
                    getattr(
                        row,
                        f"observed_log10_{quantity}_vector",
                    )
                )
                distance = parse_vector(
                    row.distance_km_vector
                )
                train_raw[i] = (
                    plum_raw_prediction(
                        obs,
                        distance,
                        scale,
                    )
                )

            truth_train = train_frame[
                f"true_log10_{quantity}"
            ].to_numpy(dtype=float)

            # Median residual minimizes absolute deviation.
            offset = float(
                np.median(
                    truth_train
                    - train_raw
                )
            )
            offsets[scale] = offset
            train_raw_by_scale[
                scale
            ] = train_raw

            validation_prediction = np.empty(
                len(validation_frame),
                dtype=float,
            )
            for i, row in enumerate(
                validation_frame.itertuples(
                    index=False
                )
            ):
                obs = parse_vector(
                    getattr(
                        row,
                        f"observed_log10_{quantity}_vector",
                    )
                )
                distance = parse_vector(
                    row.distance_km_vector
                )
                validation_prediction[i] = (
                    plum_raw_prediction(
                        obs,
                        distance,
                        scale,
                    )
                    + offset
                )

            temp = validation_frame[
                [
                    "event_id",
                    "repeat",
                    f"true_log10_{quantity}",
                ]
            ].copy()
            temp[
                "candidate_prediction"
            ] = validation_prediction

            score = event_balanced_mae(
                temp,
                quantity,
                "candidate_prediction",
            )
            validation_scores[
                scale
            ] = score

        best_scale = min(
            scales,
            key=lambda x: (
                validation_scores[x],
                float("inf")
                if math.isinf(x)
                else x,
            ),
        )
        result[quantity] = {
            "length_scale_km": (
                "inf"
                if math.isinf(
                    best_scale
                )
                else float(
                    best_scale
                )
            ),
            "offset_log10": float(
                offsets[
                    best_scale
                ]
            ),
            "validation_mae": float(
                validation_scores[
                    best_scale
                ]
            ),
            "grid": {
                (
                    "inf"
                    if math.isinf(s)
                    else str(s)
                ): {
                    "offset_log10": float(
                        offsets[s]
                    ),
                    "validation_mae": float(
                        validation_scores[s]
                    ),
                }
                for s in scales
            },
        }

    return result


def apply_plum(
    frame: pd.DataFrame,
    configuration: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()

    for quantity in (
        "pga",
        "pgv",
    ):
        cfg = configuration[
            quantity
        ]
        scale_raw = cfg[
            "length_scale_km"
        ]
        scale = (
            float("inf")
            if str(scale_raw) == "inf"
            else float(scale_raw)
        )
        offset = float(
            cfg["offset_log10"]
        )

        prediction = np.empty(
            len(result),
            dtype=float,
        )
        for i, row in enumerate(
            result.itertuples(
                index=False
            )
        ):
            obs = parse_vector(
                getattr(
                    row,
                    f"observed_log10_{quantity}_vector",
                )
            )
            distance = parse_vector(
                row.distance_km_vector
            )
            prediction[i] = (
                plum_raw_prediction(
                    obs,
                    distance,
                    scale,
                )
                + offset
            )

        result[
            f"plum_like_log10_{quantity}"
        ] = prediction

    return result


def element_metric(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> np.ndarray:
    residual = prediction - truth
    absolute = np.abs(
        residual
    )
    if metric == "mae":
        return absolute
    if metric == "bias":
        return residual
    if metric == "factor2":
        return (
            absolute <= LOG10_FACTOR_2
        ).astype(float)
    if metric == "factor3":
        return (
            absolute <= LOG10_FACTOR_3
        ).astype(float)
    if metric == "under05":
        return (
            residual <= -0.5
        ).astype(float)
    raise ValueError(metric)


def canonical_metric(
    frame: pd.DataFrame,
    true_col: str,
    pred_col: str,
    metric: str,
) -> dict[str, Any]:
    truth = frame[
        true_col
    ].to_numpy(dtype=float)
    prediction = frame[
        pred_col
    ].to_numpy(dtype=float)
    values = element_metric(
        truth,
        prediction,
        metric,
    )

    work = frame[
        ["event_id", "repeat"]
    ].copy()
    work["value"] = values

    er = (
        work.groupby(
            ["event_id", "repeat"],
            sort=False,
        )["value"]
        .mean()
        .reset_index()
    )
    ev = (
        er.groupby(
            "event_id",
            sort=False,
        )["value"]
        .mean()
    )

    return {
        "value": float(
            ev.mean()
        ),
        "n_events": int(
            len(ev)
        ),
        "n_event_repeats": int(
            len(er)
        ),
        "n_target_rows": int(
            len(frame)
        ),
    }


def build_metrics(
    predictions: pd.DataFrame,
    methods: list[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        truth = predictions[
            f"true_log10_{quantity}"
        ].to_numpy(dtype=float)
        masks = {
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

        for population, mask in masks.items():
            subset = predictions.loc[
                mask
            ].copy()
            if subset.empty:
                continue

            for method in methods:
                pred_col = (
                    f"{method}_log10_{quantity}"
                )
                if pred_col not in subset.columns:
                    continue

                for metric in (
                    "mae",
                    "bias",
                    "factor2",
                    "factor3",
                    "under05",
                ):
                    rec = canonical_metric(
                        subset,
                        f"true_log10_{quantity}",
                        pred_col,
                        metric,
                    )
                    rows.append(
                        {
                            "method": method,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric,
                            **rec,
                        }
                    )

    return pd.DataFrame(
        rows
    )


def paired_event_deltas(
    predictions: pd.DataFrame,
    methods: list[str],
    reference: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        truth_all = predictions[
            f"true_log10_{quantity}"
        ].to_numpy(dtype=float)

        masks = {
            "overall": np.ones(
                len(predictions),
                dtype=bool,
            ),
            "high_motion_tail": (
                truth_all
                >= float(
                    thresholds[
                        quantity
                    ]
                )
            ),
        }

        for population, mask in masks.items():
            subset = predictions.loc[
                mask
            ].copy()

            for metric in (
                "mae",
                "factor2",
                "under05",
            ):
                event_values: dict[
                    str, pd.Series
                ] = {}

                for method in methods:
                    pred_col = (
                        f"{method}_log10_{quantity}"
                    )
                    if pred_col not in subset.columns:
                        continue

                    values = element_metric(
                        subset[
                            f"true_log10_{quantity}"
                        ].to_numpy(dtype=float),
                        subset[
                            pred_col
                        ].to_numpy(dtype=float),
                        metric,
                    )
                    work = subset[
                        [
                            "event_id",
                            "repeat",
                        ]
                    ].copy()
                    work["value"] = (
                        values
                    )
                    er = (
                        work.groupby(
                            [
                                "event_id",
                                "repeat",
                            ],
                            sort=False,
                        )["value"]
                        .mean()
                        .reset_index()
                    )
                    event_values[
                        method
                    ] = (
                        er.groupby(
                            "event_id",
                            sort=False,
                        )["value"]
                        .mean()
                    )

                if reference not in event_values:
                    continue

                for method, candidate in event_values.items():
                    if method == reference:
                        continue
                    ref = event_values[
                        reference
                    ]
                    common = (
                        ref.index.intersection(
                            candidate.index
                        )
                    )
                    delta = (
                        candidate.loc[
                            common
                        ]
                        - ref.loc[
                            common
                        ]
                    )
                    rows.append(
                        {
                            "reference_method": reference,
                            "candidate_method": method,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric,
                            "n_paired_events": int(
                                len(common)
                            ),
                            "mean_delta_candidate_minus_reference": float(
                                delta.mean()
                            ),
                            "median_delta_candidate_minus_reference": float(
                                delta.median()
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def train_learned_variant(
    variant: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    train_dataset: StrongBaselineDataset,
    validation_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(
        args.seed
    )

    model = make_model(
        variant,
        args.hidden_dim,
        args.attention_heads,
        args.graph_initial_length_scale_km,
    ).to(device)

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
        Path(args.out_dir)
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

    best_score = np.inf
    best_epoch = 0
    stale = 0
    history = []
    start = time.time()

    print(
        f"\n=== Learned strong baseline: {variant} ==="
    )
    print(
        f"Parameters : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_frame = predict_model(
            model,
            validation_loader,
            device,
        )
        val_pga = event_balanced_mae(
            validation_frame,
            "pga",
            "pred_log10_pga",
        )
        val_pgv = event_balanced_mae(
            validation_frame,
            "pgv",
            "pred_log10_pgv",
        )
        score = 0.5 * (
            val_pga + val_pgv
        )

        scheduler.step(
            score
        )
        lr = optimizer.param_groups[
            0
        ]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_mae_pga": val_pga,
                "validation_mae_pgv": val_pgv,
                "validation_score": score,
                "learning_rate": lr,
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
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "variant": variant,
                    "epoch": epoch,
                    "args": vars(args),
                    "validation_score": score,
                    "validation_mae_pga": val_pga,
                    "validation_mae_pgv": val_pgv,
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

    try:
        checkpoint = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            best_path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    summary = {
        "variant": variant,
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_score": float(
            best_score
        ),
        "parameter_count": int(
            sum(
                p.numel()
                for p in model.parameters()
            )
        ),
        "training_seconds": float(
            time.time() - start
        ),
    }
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            r"data/scedc/model_manifests/"
            r"scenario_t0_5s_k5.csv"
        ),
    )
    parser.add_argument(
        "--h5-root",
        default=(
            r"data/scedc/processed_full_v4/events"
        ),
    )
    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )
    parser.add_argument(
        "--threshold-json",
        default=(
            r"runs/tail_gated_compromise_t0_5s_k5/"
            r"tail_thresholds_q0.90_t0_5s.json"
        ),
    )
    parser.add_argument(
        "--variants",
        default=(
            "base,cross_attention,graph"
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
            r"runs/phase2_strong_baselines"
        ),
    )

    args = parser.parse_args()

    variants = [
        x.strip()
        for x in args.variants.split(",")
        if x.strip()
    ]
    unknown = set(
        variants
    ).difference(
        LEARNED_VARIANTS
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

    scales = []
    for text in (
        args.plum_length_scales_km
        .split(",")
    ):
        text = text.strip()
        if not text:
            continue
        if text.lower() == "inf":
            scales.append(
                float("inf")
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

    train_dataset = StrongBaselineDataset(
        split_name="train",
        training=True,
        repeats=1,
        **common_dataset,
    )
    train_reference_dataset = StrongBaselineDataset(
        split_name="train",
        training=False,
        repeats=1,
        **common_dataset,
    )
    validation_dataset = StrongBaselineDataset(
        split_name="validation",
        training=False,
        repeats=args.validation_repeats,
        **common_dataset,
    )
    test_dataset = StrongBaselineDataset(
        split_name="test",
        training=False,
        repeats=args.test_repeats,
        **common_dataset,
    )

    print(
        "=== Phase 2 / Step 3: strong baseline suite ==="
    )
    print(f"Device             : {device}")
    print(
        f"Train/Val/Test     : "
        f"{len(train_dataset.frame)}/"
        f"{len(validation_dataset.frame)}/"
        f"{len(test_dataset.frame)}"
    )
    print(
        f"Val/Test repeats   : "
        f"{args.validation_repeats}/"
        f"{args.test_repeats}"
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
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
    train_reference_loader = DataLoader(
        train_reference_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    # ------------------------------------------------------------
    # Strictly causal propagation baselines.
    # ------------------------------------------------------------
    print(
        "\nCollecting causal propagation reference rows..."
    )
    train_baseline = collect_causal_baselines(
        train_reference_loader,
        args.idw_power,
    )
    validation_baseline = collect_causal_baselines(
        validation_loader,
        args.idw_power,
    )
    test_baseline = collect_causal_baselines(
        test_loader,
        args.idw_power,
    )

    plum_configuration = fit_plum(
        train_baseline,
        validation_baseline,
        scales,
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

    test_combined = apply_plum(
        test_baseline,
        plum_configuration,
    )

    # ------------------------------------------------------------
    # Learned baselines.
    # ------------------------------------------------------------
    learned_summaries = []

    key_columns = [
        "event_id",
        "repeat",
        "target_slot",
        "target_station_index",
    ]

    for variant in variants:
        model, summary = (
            train_learned_variant(
                variant,
                args,
                train_loader,
                train_dataset,
                validation_loader,
                device,
            )
        )
        learned_summaries.append(
            summary
        )

        prediction = predict_model(
            model,
            test_loader,
            device,
        )
        prediction.to_csv(
            out_dir
            / variant
            / "repeated_test_predictions.csv",
            index=False,
        )

        # Exact pairing check.
        merge_columns = (
            key_columns
            + [
                "pred_log10_pga",
                "pred_log10_pgv",
            ]
        )
        renamed = prediction[
            merge_columns
        ].rename(
            columns={
                "pred_log10_pga": (
                    f"{variant}_log10_pga"
                ),
                "pred_log10_pgv": (
                    f"{variant}_log10_pgv"
                ),
            }
        )

        test_combined = test_combined.merge(
            renamed,
            on=key_columns,
            how="left",
            validate="one_to_one",
        )

        if test_combined[
            f"{variant}_log10_pga"
        ].isna().any():
            raise RuntimeError(
                f"Pairing failure for {variant}."
            )

    learned_summary_df = pd.DataFrame(
        learned_summaries
    )
    learned_summary_df.to_csv(
        out_dir
        / "learned_training_summary.csv",
        index=False,
    )

    # Remove encoded vectors from the final compact prediction table.
    final_predictions = test_combined.drop(
        columns=[
            "observed_log10_pga_vector",
            "observed_log10_pgv_vector",
            "distance_km_vector",
        ]
    )

    final_predictions_path = (
        out_dir
        / "strong_baseline_test_predictions.csv"
    )
    final_predictions.to_csv(
        final_predictions_path,
        index=False,
    )

    methods = [
        "median_observed",
        "nearest_observed",
        "idw_observed",
        "plum_like",
        *variants,
    ]

    metrics = build_metrics(
        final_predictions,
        methods,
        thresholds,
    )
    metrics_path = (
        out_dir
        / "strong_baseline_metrics.csv"
    )
    metrics.to_csv(
        metrics_path,
        index=False,
    )

    paired = paired_event_deltas(
        final_predictions,
        methods,
        reference="base",
        thresholds=thresholds,
    )
    paired_path = (
        out_dir
        / "strong_baseline_paired_deltas_vs_base.csv"
    )
    paired.to_csv(
        paired_path,
        index=False,
    )

    (
        out_dir
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(
                    device
                ),
                "variants": variants,
                "methods": methods,
                "metric_aggregation": (
                    "targets -> repeats -> events"
                ),
                "plum_like_note": (
                    "Adapted causal propagation baseline; not exact PLUM."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== PLUM-like validation-selected configuration ==="
    )
    print(
        json.dumps(
            plum_configuration,
            indent=2,
        )
    )

    print(
        "\n=== Strong-baseline overall MAE ==="
    )
    display = metrics.loc[
        (
            metrics["population"]
            == "overall"
        )
        & (
            metrics["metric"]
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
    ]
    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n=== Strong-baseline high-motion-tail MAE ==="
    )
    tail_display = metrics.loc[
        (
            metrics["population"]
            == "high_motion_tail"
        )
        & (
            metrics["metric"]
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
    ]
    print(
        tail_display.to_string(
            index=False
        )
    )

    print(
        "\n=== Candidate minus Base paired event deltas: overall MAE ==="
    )
    print(
        paired.loc[
            (
                paired["population"]
                == "overall"
            )
            & (
                paired["metric"]
                == "mae"
            )
        ].to_string(
            index=False
        )
    )

    print("\nOutputs:")
    for p in (
        final_predictions_path,
        metrics_path,
        paired_path,
        out_dir
        / "learned_training_summary.csv",
        out_dir
        / "plum_like_configuration.json",
    ):
        print(
            f"  {p.resolve()}"
        )


if __name__ == "__main__":
    main()
