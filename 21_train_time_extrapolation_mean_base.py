#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
21_train_time_extrapolation_mean_base.py

Leakage-safe mean-pooling base-model training for time extrapolation or any
user-specified train/validation split.

Key guarantees
--------------
1. --split-column is required.
2. Only train and validation labels are instantiated.
3. Test-event HDF5 files are never opened.
4. Tail thresholds are computed from the training split only.
5. Validation uses deterministic repeated station combinations.
6. The saved model is compatible with 18_train_tail_risk_gated_dual_head.py.

Example
-------
python 21_train_time_extrapolation_mean_base.py ^
  --manifest data\scedc\model_manifests\scenario_t0_5s_k5.csv ^
  --split-column split_time_extrapolation ^
  --train-label train ^
  --validation-label validation ^
  --validation-repeats 3 ^
  --epochs 50 ^
  --out-dir runs\time_extrapolation_mean_base_t0_5s_k5
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
    x = (
        (coordinates[:, 1] - origin_longitude)
        * 111.32
        * math.cos(math.radians(origin_latitude))
    )
    y = (
        coordinates[:, 0] - origin_latitude
    ) * 110.57

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
    Training:
        one deterministic-but-changing station combination per event per epoch.

    Validation:
        fixed repeated combinations per event.
    """

    def __init__(
        self,
        manifest: str | Path,
        split_column: str,
        split_label: str,
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
                f"Manifest is missing columns: {sorted(missing)}"
            )

        frame = frame.loc[
            frame[split_column]
            .astype(str)
            .eq(str(split_label))
        ].copy()

        eligible_column = (
            f"eligible_t0_{t0_sec}s_k{input_stations}"
        )
        if eligible_column in frame.columns:
            frame = frame.loc[
                as_bool(frame[eligible_column])
            ].copy()

        self.frame = frame.reset_index(drop=True)
        self.split_label = str(split_label)
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
                f"No eligible events for {split_column}={split_label}."
            )
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame) * self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            event_index = int(index)
            repeat_index = 0
            key_suffix = f"epoch:{self.epoch}"
        else:
            event_index = int(index // self.repeats)
            repeat_index = int(index % self.repeats)
            key_suffix = f"repeat:{repeat_index}"

        event = self.frame.iloc[event_index]
        event_id = str(event["event_id"])

        rng = np.random.default_rng(
            stable_seed(
                f"{self.split_label}:{event_id}:{key_suffix}",
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
            & (p_offset <= self.t0_sec)
        )

        if len(triggered) < self.input_stations:
            raise RuntimeError(
                f"Event {event_id}: only {len(triggered)} triggered stations."
            )

        input_indices = rng.choice(
            triggered,
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
                f"Event {event_id}: only {len(target_pool)} targets."
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
            + int(round(self.t0_sec * sampling_rate)),
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

        input_features = np.concatenate(
            [
                local_xy_km(
                    input_coordinates,
                    origin_latitude,
                    origin_longitude,
                ) / 100.0,
                input_coordinates[:, 2:3] / 2000.0,
                p_offset[input_indices, None]
                / max(float(self.t0_sec), 1.0),
            ],
            axis=1,
        ).astype(np.float32)

        target_features = np.concatenate(
            [
                local_xy_km(
                    target_coordinates,
                    origin_latitude,
                    origin_longitude,
                ) / 100.0,
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

        magnitude = pd.to_numeric(
            event.get("magnitude", np.nan),
            errors="coerce",
        )

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
            nn.Conv1d(3, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MeanPoolingSparseFieldModel(nn.Module):
    """
    Compatible with 18_train_tail_risk_gated_dual_head.py / MeanBase.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.waveform_encoder = WaveformEncoder(hidden_dim)

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
        batch_size, stations, channels, samples = (
            input_waveforms.shape
        )

        waveform_latent = self.waveform_encoder(
            input_waveforms.reshape(
                batch_size * stations,
                channels,
                samples,
            )
        ).reshape(batch_size, stations, -1)

        station_latent = self.station_encoder(
            torch.cat(
                [waveform_latent, input_features],
                dim=-1,
            )
        )

        event_latent = station_latent.mean(dim=1)

        expanded_event = event_latent[:, None, :].expand(
            -1,
            target_features.shape[1],
            -1,
        )

        return self.query_decoder(
            torch.cat(
                [expanded_event, target_features],
                dim=-1,
            )
        )


def compute_training_thresholds(
    manifest: str | Path,
    split_column: str,
    train_label: str,
    t0_sec: int,
    quantile: float,
    cache_path: Path,
) -> dict[str, Any]:
    if cache_path.exists():
        cached = json.loads(
            cache_path.read_text(encoding="utf-8")
        )
        if (
            str(cached.get("split_column")) == str(split_column)
            and str(cached.get("train_label")) == str(train_label)
            and int(cached.get("t0_sec", -1)) == int(t0_sec)
            and math.isclose(
                float(cached.get("quantile", -1.0)),
                float(quantile),
                abs_tol=1e-12,
            )
        ):
            print(
                f"Reuse training-only thresholds: {cache_path.resolve()}"
            )
            return cached

    frame = pd.read_csv(
        manifest,
        dtype={"event_id": str},
    )

    if split_column not in frame.columns:
        raise KeyError(
            f"Split column not found: {split_column}"
        )

    train_frame = frame.loc[
        frame[split_column]
        .astype(str)
        .eq(str(train_label))
    ].copy()

    eligible_column = f"eligible_t0_{t0_sec}s_k5"
    # Do not depend on K here; thresholds use all available stations in
    # training events. The event split is still strictly training-only.

    if len(train_frame) == 0:
        raise ValueError("Training split is empty.")

    log_pga: list[np.ndarray] = []
    log_pgv: list[np.ndarray] = []

    for event in tqdm(
        train_frame.itertuples(index=False),
        total=len(train_frame),
        desc="compute training-only tail thresholds",
    ):
        with h5py.File(str(event.h5_path), "r") as h5:
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

        snapshot = min(
            time_zero_index
            + int(round(t0_sec * sampling_rate)),
            acceleration.shape[-1] - 1,
        )

        pga = horizontal_peak(
            acceleration[:, :, snapshot:]
        )
        pgv = horizontal_peak(
            velocity[:, :, snapshot:]
        )

        log_pga.append(
            np.log10(np.maximum(pga, 1e-10))
        )
        log_pgv.append(
            np.log10(np.maximum(pgv, 1e-12))
        )

    pga_values = np.concatenate(log_pga)
    pgv_values = np.concatenate(log_pgv)

    result = {
        "split_column": str(split_column),
        "train_label": str(train_label),
        "t0_sec": int(t0_sec),
        "quantile": float(quantile),
        "log10_pga_threshold": float(
            np.quantile(pga_values, quantile)
        ),
        "log10_pgv_threshold": float(
            np.quantile(pgv_values, quantile)
        ),
        "n_training_station_targets": int(
            len(pga_values)
        ),
        "test_split_evaluated": False,
    }

    cache_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0

    for batch in loader:
        prediction = model(
            batch["input_waveforms"].to(device),
            batch["input_features"].to(device),
            batch["target_features"].to(device),
        )
        target = batch["target_log"].to(device)

        loss = nn.functional.smooth_l1_loss(
            prediction,
            target,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1

    return total_loss / max(batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            prediction = model(
                batch["input_waveforms"].to(device),
                batch["input_features"].to(device),
                batch["target_features"].to(device),
            ).cpu().numpy()

            target = batch["target_log"].numpy()
            magnitude = batch["magnitude"].numpy()
            repeat = batch["repeat"].numpy()
            event_ids = list(batch["event_id"])

            for b in range(target.shape[0]):
                for q in range(target.shape[1]):
                    true_pga = float(target[b, q, 0])
                    true_pgv = float(target[b, q, 1])

                    rows.append({
                        "event_id": str(event_ids[b]),
                        "repeat": int(repeat[b]),
                        "magnitude": float(magnitude[b]),
                        "true_log10_pga": true_pga,
                        "true_log10_pgv": true_pgv,
                        "pred_log10_pga": float(
                            prediction[b, q, 0]
                        ),
                        "pred_log10_pgv": float(
                            prediction[b, q, 1]
                        ),
                        "is_tail_pga": bool(
                            true_pga >= thresholds[0]
                        ),
                        "is_tail_pgv": bool(
                            true_pgv >= thresholds[1]
                        ),
                    })

    frame = pd.DataFrame(rows)
    metrics: dict[str, float] = {
        "n_rows": int(len(frame)),
        "n_events": int(
            frame["event_id"].nunique()
        ),
        "n_event_repeats": int(
            frame[
                ["event_id", "repeat"]
            ].drop_duplicates().shape[0]
        ),
    }

    for quantity in ("pga", "pgv"):
        truth = frame[
            f"true_log10_{quantity}"
        ].to_numpy(dtype=float)
        prediction = frame[
            f"pred_log10_{quantity}"
        ].to_numpy(dtype=float)

        residual = prediction - truth
        absolute = np.abs(residual)
        tail = frame[
            f"is_tail_{quantity}"
        ].to_numpy(dtype=bool)

        error_column = (
            f"absolute_error_{quantity}"
        )
        frame[error_column] = absolute

        event_repeat_mae = (
            frame.groupby(
                ["event_id", "repeat"],
                sort=False,
            )[error_column]
            .mean()
        )

        metrics[
            f"event_macro_mae_{quantity}"
        ] = float(event_repeat_mae.mean())

        metrics[
            f"target_weighted_mae_{quantity}"
        ] = float(absolute.mean())

        metrics[
            f"bias_{quantity}"
        ] = float(residual.mean())

        metrics[
            f"factor2_{quantity}"
        ] = float(
            np.mean(
                absolute <= LOG10_FACTOR_2
            )
        )

        if tail.any():
            tail_frame = frame.loc[tail]
            metrics[
                f"tail_event_macro_mae_{quantity}"
            ] = float(
                tail_frame.groupby(
                    ["event_id", "repeat"],
                    sort=False,
                )[error_column]
                .mean()
                .mean()
            )
            metrics[
                f"tail_bias_{quantity}"
            ] = float(
                residual[tail].mean()
            )
        else:
            metrics[
                f"tail_event_macro_mae_{quantity}"
            ] = float("nan")
            metrics[
                f"tail_bias_{quantity}"
            ] = float("nan")

    metrics["selection_score"] = 0.5 * (
        metrics["event_macro_mae_pga"]
        + metrics["event_macro_mae_pgv"]
    )

    return metrics, frame


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "args": vars(args),
            "validation_metrics": metrics,
            "thresholds": thresholds,
            "test_split_evaluated": False,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "data/scedc/model_manifests/"
            "scenario_t0_5s_k5_time_clean.csv"
        ),
    )
    parser.add_argument(
        "--split-column",
        default="split_time_clean",
        help=(
            "Manifest column containing train/validation/test labels, "
            "for example split_time_extrapolation."
        ),
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
        "--tail-quantile",
        type=float,
        default=0.90,
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
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "time_clean_mean_base_t0_5s_k5 "
        ),
    )

    args = parser.parse_args()

    if args.validation_repeats < 1:
        raise ValueError(
            "--validation-repeats must be at least 1."
        )
    if not 0.5 < args.tail_quantile < 1.0:
        raise ValueError(
            "--tail-quantile must be between 0.5 and 1.0."
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
        device = torch.device(args.device)

    set_global_seed(args.seed)

    output_directory = Path(args.out_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_path = (
        output_directory
        / (
            f"tail_thresholds_"
            f"q{args.tail_quantile:.2f}_"
            f"t0_{args.t0_sec}s.json"
        )
    )

    thresholds = compute_training_thresholds(
        manifest=args.manifest,
        split_column=args.split_column,
        train_label=args.train_label,
        t0_sec=args.t0_sec,
        quantile=args.tail_quantile,
        cache_path=threshold_path,
    )
    threshold_array = np.asarray(
        [
            thresholds[
                "log10_pga_threshold"
            ],
            thresholds[
                "log10_pgv_threshold"
            ],
        ],
        dtype=float,
    )

    train_dataset = SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.train_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=True,
    )
    validation_dataset = SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.validation_repeats,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    common_loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (
            device.type == "cuda"
        ),
        "persistent_workers": False,
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

    model = MeanPoolingSparseFieldModel(
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
            min_lr=args.minimum_learning_rate,
        )
    )

    checkpoint_path = (
        output_directory / "best_model.pt"
    )

    print(
        "=== Time-extrapolation mean-base training ==="
    )
    print(f"Device              : {device}")
    print(
        f"Split column        : {args.split_column}"
    )
    print(
        f"Train label         : {args.train_label}"
    )
    print(
        f"Validation label    : {args.validation_label}"
    )
    print(
        f"Train events        : {len(train_dataset.frame)}"
    )
    print(
        f"Validation events   : {len(validation_dataset.frame)}"
    )
    print(
        f"Validation repeats  : {args.validation_repeats}"
    )
    print(
        f"Validation samples  : {len(validation_dataset)}"
    )
    print(
        "Tail thresholds     : "
        f"PGA={threshold_array[0]:.4f}, "
        f"PGV={threshold_array[1]:.4f}"
    )
    print("Test split          : NOT ACCESSED")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            threshold_array,
            device,
        )

        current_score = float(
            validation_metrics[
                "selection_score"
            ]
        )
        scheduler.step(current_score)

        history.append({
            "epoch": int(epoch),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "train_loss": float(train_loss),
            **{
                f"validation_{key}": value
                for key, value
                in validation_metrics.items()
            },
        })
        pd.DataFrame(history).to_csv(
            output_directory / "history.csv",
            index=False,
        )

        if (
            current_score
            < best_score
            - args.minimum_delta
        ):
            best_score = current_score
            best_epoch = int(epoch)
            stale_epochs = 0

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                args,
                validation_metrics,
                thresholds,
            )
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val score={current_score:.4f} | "
            f"PGA={validation_metrics['event_macro_mae_pga']:.4f} | "
            f"PGV={validation_metrics['event_macro_mae_pgv']:.4f} | "
            f"tail PGA="
            f"{validation_metrics['tail_event_macro_mae_pga']:.4f} | "
            f"tail PGV="
            f"{validation_metrics['tail_event_macro_mae_pgv']:.4f}"
        )

        if (
            stale_epochs
            >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}."
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

    best_metrics, best_predictions = evaluate(
        model,
        validation_loader,
        threshold_array,
        device,
    )

    best_predictions.to_csv(
        output_directory
        / "validation_predictions_best.csv",
        index=False,
    )

    (
        output_directory
        / "validation_metrics_best.json"
    ).write_text(
        json.dumps(
            {
                "best_epoch": int(
                    checkpoint["epoch"]
                ),
                "training_seconds": float(
                    time.time() - start_time
                ),
                "test_split_evaluated": False,
                **best_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_directory
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(device),
                "best_epoch": int(
                    checkpoint["epoch"]
                ),
                "tail_thresholds": thresholds,
                "test_split_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Best validation checkpoint ===")
    print(
        f"Epoch              : {checkpoint['epoch']}"
    )
    print(
        "Overall PGA/PGV    : "
        f"{best_metrics['event_macro_mae_pga']:.4f}/"
        f"{best_metrics['event_macro_mae_pgv']:.4f}"
    )
    print(
        "Tail PGA/PGV       : "
        f"{best_metrics['tail_event_macro_mae_pga']:.4f}/"
        f"{best_metrics['tail_event_macro_mae_pgv']:.4f}"
    )
    print("Test split         : NOT ACCESSED")
    print(
        f"Checkpoint         : {checkpoint_path.resolve()}"
    )
    print(
        f"Threshold JSON     : {threshold_path.resolve()}"
    )
    print(
        f"Outputs            : {output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
