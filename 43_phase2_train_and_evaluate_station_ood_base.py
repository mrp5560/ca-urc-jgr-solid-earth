#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 / Step 2B
Strict station-level OOD training and paired seen-vs-unseen evaluation
for the final mean-pooling Base model.

Protocol
--------
- 80/20 seen/OOD station assignment is supplied by Step 2A.
- OOD station IDs are NEVER used as training/validation inputs or targets.
- Training/validation inputs are selected only from P-wave-reached SEEN stations.
- Training/validation targets are selected only from remaining SEEN stations.
- Test uses only events satisfying test_paired_seen_ood_eligible.
- For each test event/repeat, the SAME K seen input stations are used to query:
    (a) paired-targets seen target stations
    (b) paired-targets OOD target stations
  so both target groups have identical event/input budgets.
- Metrics use the manuscript aggregation:
    targets -> repeats -> events.

Outputs
-------
best_model.pt
history.csv
repeated_station_ood_predictions.csv
station_ood_metrics.csv
station_ood_paired_event_deltas.csv
station_ood_distance_summary.csv
run_configuration.json
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
        raise ValueError("Empty future waveform window.")
    return np.max(
        np.sqrt(
            data[:, 0, :] ** 2
            + data[:, 1, :] ** 2
        ),
        axis=1,
    )


def read_station_ids(
    h5: h5py.File,
    n_stations: int,
) -> np.ndarray:
    if "station_id" not in h5:
        raise KeyError("HDF5 has no station_id dataset.")
    try:
        ids = h5["station_id"].asstr()[:]
    except Exception:
        ids = np.asarray(
            [str(x) for x in h5["station_id"][:]],
            dtype=object,
        )
    ids = np.asarray(
        [str(x).strip() for x in ids],
        dtype=object,
    )
    if len(ids) != n_stations:
        raise RuntimeError(
            "station_id length does not match station count."
        )
    return ids


class SeenOnlyDataset(Dataset):
    """
    Training:
        one deterministic-but-changing seen-only station draw per event/epoch.
    Validation:
        deterministic repeated seen-only draws per event.
    """

    def __init__(
        self,
        scenario_csv: str | Path,
        assignment_csv: str | Path,
        split_column: str,
        split_name: str,
        eligibility_column: str,
        t0_sec: int,
        input_stations: int,
        target_stations: int,
        input_pre_sec: float,
        seed: int,
        training: bool,
        repeats: int = 1,
    ):
        frame = pd.read_csv(
            scenario_csv,
            dtype={"event_id": str},
        )
        required = {
            "event_id",
            "h5_path",
            split_column,
            eligibility_column,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Scenario missing columns: {sorted(missing)}"
            )

        frame = frame.loc[
            frame[split_column].astype(str).eq(str(split_name))
            & as_bool(frame[eligibility_column])
        ].copy()
        self.frame = frame.reset_index(drop=True)

        assignment = pd.read_csv(
            assignment_csv,
            dtype={"station_id": str},
        )
        if not {
            "station_id",
            "station_role",
        }.issubset(assignment.columns):
            raise ValueError(
                "Station assignment must contain "
                "station_id and station_role."
            )
        self.station_role = dict(
            zip(
                assignment["station_id"].astype(str),
                assignment["station_role"].astype(str),
            )
        )
        self.seen_ids = {
            sid
            for sid, role in self.station_role.items()
            if role == "seen"
        }
        self.ood_ids = {
            sid
            for sid, role in self.station_role.items()
            if role == "ood_holdout"
        }

        self.split_name = str(split_name)
        self.t0_sec = int(t0_sec)
        self.input_stations = int(input_stations)
        self.target_stations = int(target_stations)
        self.input_pre_sec = float(input_pre_sec)
        self.seed = int(seed)
        self.training = bool(training)
        self.repeats = 1 if training else int(repeats)
        self.epoch = 0

        if len(self.frame) == 0:
            raise ValueError(
                f"No eligible events for {split_name}."
            )
        if self.repeats < 1:
            raise ValueError("repeats must be >=1.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame) * self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            event_index = int(index)
            repeat_index = 0
            suffix = f"epoch:{self.epoch}"
        else:
            event_index = int(index // self.repeats)
            repeat_index = int(index % self.repeats)
            suffix = f"repeat:{repeat_index}"

        event = self.frame.iloc[event_index]
        event_id = str(event["event_id"])
        rng = np.random.default_rng(
            stable_seed(
                f"{self.split_name}:{event_id}:{suffix}",
                self.seed,
            )
        )

        h5_path = Path(str(event["h5_path"]))
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
            station_ids = read_station_ids(
                h5,
                len(coordinates),
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

        roles = np.asarray(
            [
                self.station_role.get(str(sid), "unknown")
                for sid in station_ids
            ],
            dtype=object,
        )
        if np.any(roles == "unknown"):
            raise RuntimeError(
                f"Event {event_id}: station assignment missing IDs."
            )

        seen = roles == "seen"
        triggered_seen = np.flatnonzero(
            seen
            & np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (p_offset <= float(self.t0_sec))
        )
        if len(triggered_seen) < self.input_stations:
            raise RuntimeError(
                f"Event {event_id}: only {len(triggered_seen)} "
                f"P-wave-reached seen stations."
            )

        input_indices = rng.choice(
            triggered_seen,
            size=self.input_stations,
            replace=False,
        )

        seen_pool = np.setdiff1d(
            np.flatnonzero(seen),
            input_indices,
            assume_unique=False,
        )
        if len(seen_pool) < self.target_stations:
            raise RuntimeError(
                f"Event {event_id}: only {len(seen_pool)} "
                f"seen target candidates."
            )

        target_indices = rng.choice(
            seen_pool,
            size=self.target_stations,
            replace=False,
        )

        return build_model_sample(
            event_id=event_id,
            repeat_index=repeat_index,
            acceleration=acceleration,
            velocity=velocity,
            coordinates=coordinates,
            p_offset=p_offset,
            input_indices=input_indices,
            target_indices=target_indices,
            sampling_rate=sampling_rate,
            pre_first_p=pre_first_p,
            time_zero_index=time_zero_index,
            t0_sec=self.t0_sec,
            input_pre_sec=self.input_pre_sec,
        )


def build_model_sample(
    event_id: str,
    repeat_index: int,
    acceleration: np.ndarray,
    velocity: np.ndarray,
    coordinates: np.ndarray,
    p_offset: np.ndarray,
    input_indices: np.ndarray,
    target_indices: np.ndarray,
    sampling_rate: float,
    pre_first_p: float,
    time_zero_index: int,
    t0_sec: int,
    input_pre_sec: float,
) -> dict[str, Any]:
    input_start = max(
        0,
        int(
            round(
                (pre_first_p - input_pre_sec)
                * sampling_rate
            )
        ),
    )
    snapshot = min(
        time_zero_index
        + int(round(t0_sec * sampling_rate)),
        acceleration.shape[-1] - 1,
    )

    input_waveforms = acceleration[
        input_indices,
        :,
        input_start:snapshot,
    ]
    input_waveforms = (
        np.sign(input_waveforms)
        * np.log1p(
            np.abs(input_waveforms) / 1e-3
        )
    ).astype(np.float32)

    input_coordinates = coordinates[input_indices]
    target_coordinates = coordinates[target_indices]

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
            input_coordinates[:, 2:3] / 2000.0,
            p_offset[input_indices, None]
            / max(float(t0_sec), 1.0),
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

    pga = horizontal_peak(
        acceleration[
            target_indices,
            :,
            snapshot:,
        ]
    )
    pgv = horizontal_peak(
        velocity[
            target_indices,
            :,
            snapshot:,
        ]
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
        "repeat": torch.tensor(
            repeat_index,
            dtype=torch.int64,
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
    }


class WaveformEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
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
            nn.Linear(128, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class MeanPoolingBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
    ):
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
        b, k, c, t = input_waveforms.shape
        waveform_latent = self.waveform_encoder(
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

        station_latent = self.station_encoder(
            torch.cat(
                [
                    waveform_latent,
                    input_features,
                ],
                dim=-1,
            )
        )
        event_latent = station_latent.mean(
            dim=1
        )

        expanded = event_latent[
            :, None, :
        ].expand(
            -1,
            target_features.shape[1],
            -1,
        )
        return self.query_decoder(
            torch.cat(
                [
                    expanded,
                    target_features,
                ],
                dim=-1,
            )
        )


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        target = batch[
            "target_log"
        ].to(device)

        with torch.set_grad_enabled(training):
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
            loss = nn.functional.smooth_l1_loss(
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

        count = int(target.numel())
        total_loss += (
            float(loss.detach().cpu())
            * count
        )
        total_count += count

    return total_loss / max(
        total_count,
        1,
    )


def validation_predictions(
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
            event_ids = list(
                batch["event_id"]
            )
            repeats = batch[
                "repeat"
            ].numpy()

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
                            "true_log10_pga": float(
                                truth[b, q, 0]
                            ),
                            "true_log10_pgv": float(
                                truth[b, q, 1]
                            ),
                            "pred_log10_pga": float(
                                prediction[b, q, 0]
                            ),
                            "pred_log10_pgv": float(
                                prediction[b, q, 1]
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def canonical_mae(
    frame: pd.DataFrame,
    quantity: str,
) -> float:
    absolute = np.abs(
        frame[
            f"pred_log10_{quantity}"
        ].to_numpy(dtype=float)
        - frame[
            f"true_log10_{quantity}"
        ].to_numpy(dtype=float)
    )
    work = frame[
        ["event_id", "repeat"]
    ].copy()
    work["absolute"] = absolute

    event_repeat = (
        work.groupby(
            ["event_id", "repeat"],
            sort=False,
        )["absolute"]
        .mean()
        .reset_index()
    )
    event = (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )["absolute"]
        .mean()
    )
    return float(
        event.mean()
    )


def canonical_metric(
    frame: pd.DataFrame,
    quantity: str,
    metric: str,
) -> float:
    truth = frame[
        f"true_log10_{quantity}"
    ].to_numpy(dtype=float)
    prediction = frame[
        f"pred_log10_{quantity}"
    ].to_numpy(dtype=float)
    residual = prediction - truth
    absolute = np.abs(residual)

    if metric == "mae":
        element = absolute
    elif metric == "bias":
        element = residual
    elif metric == "factor2":
        element = (
            absolute <= LOG10_FACTOR_2
        ).astype(float)
    elif metric == "factor3":
        element = (
            absolute <= LOG10_FACTOR_3
        ).astype(float)
    elif metric == "under05":
        element = (
            residual <= -0.5
        ).astype(float)
    else:
        raise ValueError(metric)

    work = frame[
        ["event_id", "repeat"]
    ].copy()
    work["value"] = element

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
    return float(
        ev.mean()
    )


def pairwise_nearest_distance(
    target_xy: np.ndarray,
    input_xy: np.ndarray,
) -> np.ndarray:
    difference = (
        target_xy[:, None, :]
        - input_xy[None, :, :]
    )
    return np.sqrt(
        np.sum(
            difference ** 2,
            axis=-1,
        )
    ).min(axis=1)


def evaluate_paired_station_ood(
    model: nn.Module,
    scenario_csv: str | Path,
    assignment_csv: str | Path,
    split_column: str,
    test_label: str,
    eligibility_column: str,
    t0_sec: int,
    input_stations: int,
    paired_targets: int,
    input_pre_sec: float,
    repeats: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    scenario = pd.read_csv(
        scenario_csv,
        dtype={"event_id": str},
    )
    scenario = scenario.loc[
        scenario[
            split_column
        ].astype(str).eq(
            str(test_label)
        )
        & as_bool(
            scenario[
                eligibility_column
            ]
        )
    ].copy()
    scenario = scenario.reset_index(
        drop=True
    )

    assignment = pd.read_csv(
        assignment_csv,
        dtype={"station_id": str},
    )
    role = dict(
        zip(
            assignment["station_id"].astype(str),
            assignment["station_role"].astype(str),
        )
    )

    rows = []

    print("\n=== Paired seen-vs-OOD test ===")
    print(
        f"Test events       : "
        f"{len(scenario)}"
    )
    print(
        f"Repeats/event     : "
        f"{repeats}"
    )
    print(
        f"Targets/group     : "
        f"{paired_targets}"
    )

    model.eval()

    for event_number, event in enumerate(
        scenario.itertuples(index=False),
        start=1,
    ):
        event_id = str(
            event.event_id
        )
        h5_path = Path(
            str(event.h5_path)
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
            station_ids = read_station_ids(
                h5,
                len(coordinates),
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
                    str(sid),
                    "unknown",
                )
                for sid in station_ids
            ],
            dtype=object,
        )
        if np.any(
            roles == "unknown"
        ):
            raise RuntimeError(
                f"Event {event_id}: missing station assignment."
            )

        seen = roles == "seen"
        ood = roles == "ood_holdout"

        triggered_seen = np.flatnonzero(
            seen
            & np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (
                p_offset
                <= float(t0_sec)
            )
        )

        if (
            len(triggered_seen)
            < input_stations
        ):
            raise RuntimeError(
                f"Event {event_id}: insufficient seen inputs."
            )

        for repeat in range(
            repeats
        ):
            rng = np.random.default_rng(
                stable_seed(
                    (
                        f"station_ood:test:"
                        f"{event_id}:repeat:{repeat}"
                    ),
                    seed,
                )
            )

            input_indices = rng.choice(
                triggered_seen,
                size=input_stations,
                replace=False,
            )

            seen_pool = np.setdiff1d(
                np.flatnonzero(seen),
                input_indices,
                assume_unique=False,
            )
            ood_pool = np.flatnonzero(
                ood
            )

            if (
                len(seen_pool)
                < paired_targets
                or len(ood_pool)
                < paired_targets
            ):
                raise RuntimeError(
                    f"Event {event_id}: paired target pools "
                    f"seen={len(seen_pool)}, ood={len(ood_pool)}, "
                    f"required={paired_targets}."
                )

            seen_targets = rng.choice(
                seen_pool,
                size=paired_targets,
                replace=False,
            )
            ood_targets = rng.choice(
                ood_pool,
                size=paired_targets,
                replace=False,
            )

            for target_role, target_indices in (
                (
                    "seen_target",
                    seen_targets,
                ),
                (
                    "unseen_station_target",
                    ood_targets,
                ),
            ):
                sample = build_model_sample(
                    event_id=event_id,
                    repeat_index=repeat,
                    acceleration=acceleration,
                    velocity=velocity,
                    coordinates=coordinates,
                    p_offset=p_offset,
                    input_indices=input_indices,
                    target_indices=target_indices,
                    sampling_rate=sampling_rate,
                    pre_first_p=pre_first_p,
                    time_zero_index=time_zero_index,
                    t0_sec=t0_sec,
                    input_pre_sec=input_pre_sec,
                )

                input_coordinates = coordinates[
                    input_indices
                ]
                origin_lat = float(
                    input_coordinates[
                        :, 0
                    ].mean()
                )
                origin_lon = float(
                    input_coordinates[
                        :, 1
                    ].mean()
                )
                input_xy = local_xy_km(
                    input_coordinates,
                    origin_lat,
                    origin_lon,
                )
                target_xy = local_xy_km(
                    coordinates[
                        target_indices
                    ],
                    origin_lat,
                    origin_lon,
                )
                nearest = (
                    pairwise_nearest_distance(
                        target_xy,
                        input_xy,
                    )
                )

                with torch.inference_mode():
                    prediction = model(
                        sample[
                            "input_waveforms"
                        ][None].to(device),
                        sample[
                            "input_features"
                        ][None].to(device),
                        sample[
                            "target_features"
                        ][None].to(device),
                    )[0].cpu().numpy()

                truth = sample[
                    "target_log"
                ].numpy()

                input_station_text = "|".join(
                    str(
                        station_ids[index]
                    )
                    for index in input_indices
                )

                for q, station_index in enumerate(
                    target_indices
                ):
                    rows.append(
                        {
                            "event_id": event_id,
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
                                    <= t0_sec
                                )
                            ),
                            "nearest_input_distance_km": float(
                                nearest[q]
                            ),
                            "true_log10_pga": float(
                                truth[q, 0]
                            ),
                            "true_log10_pgv": float(
                                truth[q, 1]
                            ),
                            "pred_log10_pga": float(
                                prediction[q, 0]
                            ),
                            "pred_log10_pgv": float(
                                prediction[q, 1]
                            ),
                        }
                    )

        if (
            event_number % 25 == 0
            or event_number == len(scenario)
        ):
            print(
                f"Evaluated {event_number}/"
                f"{len(scenario)} events | "
                f"rows={len(rows):,}"
            )

    return pd.DataFrame(
        rows
    )


def metric_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for target_role, group in predictions.groupby(
        "target_role",
        sort=False,
    ):
        for quantity in (
            "pga",
            "pgv",
        ):
            for metric in (
                "mae",
                "bias",
                "factor2",
                "factor3",
                "under05",
            ):
                rows.append(
                    {
                        "target_role": (
                            target_role
                        ),
                        "quantity": quantity,
                        "metric": metric,
                        "value": canonical_metric(
                            group,
                            quantity,
                            metric,
                        ),
                        "n_events": int(
                            group[
                                "event_id"
                            ].nunique()
                        ),
                        "n_event_repeats": int(
                            group[
                                [
                                    "event_id",
                                    "repeat",
                                ]
                            ]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_target_rows": int(
                            len(group)
                        ),
                        "n_unique_stations": int(
                            group[
                                "target_station_id"
                            ].nunique()
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def paired_event_delta_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        for metric in (
            "mae",
            "bias",
            "factor2",
            "factor3",
            "under05",
        ):
            role_event_values = {}

            for target_role, group in predictions.groupby(
                "target_role",
                sort=False,
            ):
                truth = group[
                    f"true_log10_{quantity}"
                ].to_numpy(dtype=float)
                prediction = group[
                    f"pred_log10_{quantity}"
                ].to_numpy(dtype=float)
                residual = prediction - truth
                absolute = np.abs(
                    residual
                )

                if metric == "mae":
                    element = absolute
                elif metric == "bias":
                    element = residual
                elif metric == "factor2":
                    element = (
                        absolute
                        <= LOG10_FACTOR_2
                    ).astype(float)
                elif metric == "factor3":
                    element = (
                        absolute
                        <= LOG10_FACTOR_3
                    ).astype(float)
                elif metric == "under05":
                    element = (
                        residual <= -0.5
                    ).astype(float)
                else:
                    raise ValueError(
                        metric
                    )

                work = group[
                    [
                        "event_id",
                        "repeat",
                    ]
                ].copy()
                work["value"] = (
                    element
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
                ev = (
                    er.groupby(
                        "event_id",
                        sort=False,
                    )["value"]
                    .mean()
                )
                role_event_values[
                    target_role
                ] = ev

            seen = role_event_values[
                "seen_target"
            ]
            unseen = role_event_values[
                "unseen_station_target"
            ]
            common = seen.index.intersection(
                unseen.index
            )
            delta = (
                unseen.loc[common]
                - seen.loc[common]
            )

            rows.append(
                {
                    "quantity": quantity,
                    "metric": metric,
                    "n_paired_events": int(
                        len(common)
                    ),
                    "seen_value": float(
                        seen.loc[
                            common
                        ].mean()
                    ),
                    "unseen_value": float(
                        unseen.loc[
                            common
                        ].mean()
                    ),
                    "mean_delta_unseen_minus_seen": float(
                        delta.mean()
                    ),
                    "median_delta_unseen_minus_seen": float(
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
    return (
        predictions.groupby(
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
                lambda x: float(
                    np.percentile(
                        x,
                        25,
                    )
                ),
            ),
            p75_nearest_input_distance_km=(
                "nearest_input_distance_km",
                lambda x: float(
                    np.percentile(
                        x,
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        default=(
            r"data/ model_manifests/station_ood/"
            r"scenario_station_ood.csv"
        ),
    )
    parser.add_argument(
        "--station-assignment",
        default=(
            r"data/model_manifests/station_ood/"
            r"station_holdout_assignment.csv"
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
        "--paired-targets",
        type=int,
        default=3,
        help=(
            "Equal number of seen and unseen-station targets "
            "per event-repeat during paired OOD evaluation."
        ),
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
            r"runs/phase2_station_ood_base"
        ),
    )
    args = parser.parse_args()

    set_seed(
        args.seed
    )

    scenario_path = Path(
        args.scenario
    )
    assignment_path = Path(
        args.station_assignment
    )
    if not scenario_path.exists():
        raise FileNotFoundError(
            scenario_path
        )
    if not assignment_path.exists():
        raise FileNotFoundError(
            assignment_path
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

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = {
        **vars(args),
        "resolved_device": str(
            device
        ),
        "station_ood_rule": (
            "OOD station IDs are excluded from all "
            "training/validation inputs and targets."
        ),
        "paired_test_rule": (
            "same event, same repeat, same seen inputs, "
            "equal seen/OOD target counts"
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
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_dataset = SeenOnlyDataset(
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
    validation_dataset = SeenOnlyDataset(
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
            args.validation_repeats
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    common_loader = {
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
        **common_loader,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader,
    )

    model = MeanPoolingBase(
        hidden_dim=args.hidden_dim
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

    best_checkpoint = (
        out_dir / "best_model.pt"
    )
    history = []
    best_score = np.inf
    best_epoch = 0
    stale = 0
    start_time = time.time()

    print(
        "=== Phase 2 / Step 2B: "
        "strict station-level OOD Base training ==="
    )
    print(f"Device              : {device}")
    print(
        f"Train events         : "
        f"{len(train_dataset.frame)}"
    )
    print(
        f"Validation events    : "
        f"{len(validation_dataset.frame)}"
    )
    print(
        f"Validation repeats   : "
        f"{args.validation_repeats}"
    )
    print(
        f"Parameters           : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_dataset.set_epoch(
            epoch
        )

        train_loss = run_training_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_frame = (
            validation_predictions(
                model,
                validation_loader,
                device,
            )
        )
        val_pga = canonical_mae(
            validation_frame,
            "pga",
        )
        val_pgv = canonical_mae(
            validation_frame,
            "pgv",
        )
        val_score = 0.5 * (
            val_pga + val_pgv
        )

        scheduler.step(
            val_score
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_mae_pga": (
                val_pga
            ),
            "validation_mae_pgv": (
                val_pgv
            ),
            "validation_selection_score": (
                val_score
            ),
            "learning_rate": current_lr,
        }
        history.append(
            row
        )
        pd.DataFrame(
            history
        ).to_csv(
            out_dir / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val PGA={val_pga:.4f} | "
            f"PGV={val_pgv:.4f} | "
            f"score={val_score:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if (
            val_score
            < best_score
            - args.minimum_delta
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
                    "model_state": (
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "args": vars(args),
                    "validation_mae_pga": (
                        val_pga
                    ),
                    "validation_mae_pgv": (
                        val_pgv
                    ),
                    "validation_selection_score": (
                        val_score
                    ),
                    "station_ood_training": True,
                },
                best_checkpoint,
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
            best_checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            best_checkpoint,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    predictions = (
        evaluate_paired_station_ood(
            model=model,
            scenario_csv=scenario_path,
            assignment_csv=assignment_path,
            split_column=args.split_column,
            test_label=args.test_label,
            eligibility_column=(
                "test_paired_seen_ood_eligible"
            ),
            t0_sec=args.t0_sec,
            input_stations=args.input_stations,
            paired_targets=args.paired_targets,
            input_pre_sec=args.input_pre_sec,
            repeats=args.test_repeats,
            seed=args.seed,
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

    metrics = metric_table(
        predictions
    )
    metrics_path = (
        out_dir
        / "station_ood_metrics.csv"
    )
    metrics.to_csv(
        metrics_path,
        index=False,
    )

    paired = paired_event_delta_table(
        predictions
    )
    paired_path = (
        out_dir
        / "station_ood_paired_event_deltas.csv"
    )
    paired.to_csv(
        paired_path,
        index=False,
    )

    distances = distance_summary(
        predictions
    )
    distance_path = (
        out_dir
        / "station_ood_distance_summary.csv"
    )
    distances.to_csv(
        distance_path,
        index=False,
    )

    final_summary = {
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_selection_score": float(
            best_score
        ),
        "training_seconds": float(
            time.time() - start_time
        ),
        "train_events": int(
            len(train_dataset.frame)
        ),
        "validation_events": int(
            len(validation_dataset.frame)
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

    print("\n=== Station-OOD canonical metrics ===")
    print(
        metrics[
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
                "target_role",
                "quantity",
                "metric",
                "value",
                "n_events",
                "n_target_rows",
                "n_unique_stations",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n=== Unseen minus seen paired event deltas ===")
    print(
        paired[
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

    print("\n=== Geometry audit ===")
    print(
        distances.to_string(
            index=False
        )
    )

    print("\nOutputs:")
    for path in (
        best_checkpoint,
        prediction_path,
        metrics_path,
        paired_path,
        distance_path,
        out_dir / "history.csv",
    ):
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
