#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12_train_sparse_field_pga_pgv_baseline.py

First deterministic sparse-to-field baseline.

Input:
  K triggered stations observed from first-P - input_pre_sec to first-P + T0.
  Per-station acceleration waveform and local station coordinates.

Targets:
  Future horizontal PGA and PGV at held-out stations, calculated only from
  snapshot time (first-P + T0) to the end of the HDF5 window.

The model does not use catalog magnitude, hypocenter, or target waveform.
It uses:
  station waveform encoder
  station metadata encoder
  attention pooling into an event latent
  coordinate-conditioned query decoder

Default main experiment:
  T0 = 5 s
  K = 5 input stations
  10 randomly selected target stations per event
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def stable_seed(event_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{event_id}:{base_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def local_xy_km(
    coords: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    lat = coords[:, 0]
    lon = coords[:, 1]
    x = (lon - origin_lon) * 111.32 * math.cos(
        math.radians(origin_lat)
    )
    y = (lat - origin_lat) * 110.57
    return np.stack([x, y], axis=-1)


class SparseFieldDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        split_column: str,
        split_name: str,
        t0_sec: int,
        input_stations: int,
        target_stations: int,
        input_pre_sec: float,
        base_seed: int,
        training: bool,
    ):
        self.df = pd.read_csv(manifest, dtype={"event_id": str})
        self.df = self.df.loc[
            self.df[split_column].astype(str).eq(split_name)
        ].reset_index(drop=True)

        self.t0_sec = int(t0_sec)
        self.input_stations = int(input_stations)
        self.target_stations = int(target_stations)
        self.input_pre_sec = float(input_pre_sec)
        self.base_seed = int(base_seed)
        self.training = bool(training)

        eligible_column = (
            f"eligible_t0_{self.t0_sec}s_k{self.input_stations}"
        )
        if eligible_column in self.df.columns:
            self.df = self.df.loc[
                self.df[eligible_column].astype(str).str.lower().isin(
                    {"true", "1", "yes"}
                )
            ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No events for {split_column}={split_name} "
                f"in {manifest}"
            )

    def __len__(self):
        return len(self.df)

    def _rng(self, event_id: str):
        if self.training:
            return np.random.default_rng()
        return np.random.default_rng(
            stable_seed(event_id, self.base_seed)
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]
        event_id = str(row["event_id"])
        h5_path = str(row["h5_path"])
        rng = self._rng(event_id)

        with h5py.File(h5_path, "r") as h5:
            acceleration = np.asarray(
                h5["acceleration"][:],
                dtype=np.float32,
            )
            velocity = np.asarray(
                h5["velocity"][:],
                dtype=np.float32,
            )
            coords = np.asarray(
                h5["station_coords"][:],
                dtype=np.float32,
            )
            p_offset = np.asarray(
                h5["p_offset_sec"][:],
                dtype=np.float32,
            )
            sampling_rate = float(h5.attrs["sampling_rate_hz"])
            pre_first_p = float(h5.attrs["pre_first_p_sec"])
            time_zero_index = int(h5.attrs["time_zero_index"])

        triggered = np.flatnonzero(
            np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (p_offset <= self.t0_sec)
        )
        if len(triggered) < self.input_stations:
            raise RuntimeError(
                f"Event {event_id} has only {len(triggered)} "
                f"triggered stations."
            )

        input_indices = rng.choice(
            triggered,
            size=self.input_stations,
            replace=False,
        )

        target_pool = np.setdiff1d(
            np.arange(len(coords)),
            input_indices,
            assume_unique=False,
        )
        if len(target_pool) < self.target_stations:
            raise RuntimeError(
                f"Event {event_id} has only {len(target_pool)} targets."
            )

        target_indices = rng.choice(
            target_pool,
            size=self.target_stations,
            replace=False,
        )

        input_start = int(
            round(
                (pre_first_p - self.input_pre_sec)
                * sampling_rate
            )
        )
        snapshot_index = time_zero_index + int(
            round(self.t0_sec * sampling_rate)
        )

        input_start = max(0, input_start)
        snapshot_index = min(
            acceleration.shape[-1] - 1,
            snapshot_index,
        )

        input_waveforms = acceleration[
            input_indices,
            :,
            input_start:snapshot_index,
        ]

        # Preserve amplitude while compressing the dynamic range.
        amplitude_scale = 1e-3
        input_waveforms = np.sign(input_waveforms) * np.log1p(
            np.abs(input_waveforms) / amplitude_scale
        )

        input_coords = coords[input_indices]
        target_coords = coords[target_indices]

        origin_lat = float(np.mean(input_coords[:, 0]))
        origin_lon = float(np.mean(input_coords[:, 1]))

        input_xy = local_xy_km(
            input_coords,
            origin_lat,
            origin_lon,
        )
        target_xy = local_xy_km(
            target_coords,
            origin_lat,
            origin_lon,
        )

        input_features = np.concatenate(
            [
                input_xy / 100.0,
                input_coords[:, 2:3] / 2000.0,
                p_offset[input_indices, None]
                / max(float(self.t0_sec), 1.0),
            ],
            axis=1,
        ).astype(np.float32)

        target_features = np.concatenate(
            [
                target_xy / 100.0,
                target_coords[:, 2:3] / 2000.0,
            ],
            axis=1,
        ).astype(np.float32)

        future_acc = acceleration[
            target_indices,
            :,
            snapshot_index:,
        ]
        future_vel = velocity[
            target_indices,
            :,
            snapshot_index:,
        ]

        pga = np.max(
            np.sqrt(
                future_acc[:, 0, :] ** 2
                + future_acc[:, 1, :] ** 2
            ),
            axis=1,
        )
        pgv = np.max(
            np.sqrt(
                future_vel[:, 0, :] ** 2
                + future_vel[:, 1, :] ** 2
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
            "input_waveforms": torch.from_numpy(input_waveforms),
            "input_features": torch.from_numpy(input_features),
            "target_features": torch.from_numpy(target_features),
            "target_log": torch.from_numpy(target_log),
        }


class WaveformEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
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
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        return self.network(x)


class SparseFieldBaseline(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.waveform_encoder = WaveformEncoder(hidden_dim)
        self.station_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
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
        input_waveforms,
        input_features,
        target_features,
    ):
        batch, stations, channels, samples = input_waveforms.shape
        encoded = self.waveform_encoder(
            input_waveforms.reshape(
                batch * stations,
                channels,
                samples,
            )
        ).reshape(batch, stations, -1)

        station_latent = self.station_encoder(
            torch.cat([encoded, input_features], dim=-1)
        )
        weights = torch.softmax(
            self.attention_score(station_latent),
            dim=1,
        )
        event_latent = torch.sum(
            weights * station_latent,
            dim=1,
        )

        queries = target_features.shape[1]
        event_expanded = event_latent[:, None, :].expand(
            -1,
            queries,
            -1,
        )
        return self.query_decoder(
            torch.cat(
                [event_expanded, target_features],
                dim=-1,
            )
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0
    absolute_errors = []

    loss_fn = nn.SmoothL1Loss(reduction="mean")

    for batch in loader:
        input_waveforms = batch["input_waveforms"].to(device)
        input_features = batch["input_features"].to(device)
        target_features = batch["target_features"].to(device)
        target = batch["target_log"].to(device)

        with torch.set_grad_enabled(training):
            prediction = model(
                input_waveforms,
                input_features,
                target_features,
            )
            loss = loss_fn(prediction, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )
                optimizer.step()

        count = target.numel()
        total_loss += float(loss.detach()) * count
        total_count += count
        absolute_errors.append(
            torch.abs(prediction.detach() - target)
            .cpu()
            .numpy()
            .reshape(-1, 2)
        )

    errors = np.concatenate(absolute_errors, axis=0)
    return {
        "loss": total_loss / max(total_count, 1),
        "mae_log10_pga": float(np.mean(errors[:, 0])),
        "mae_log10_pgv": float(np.mean(errors[:, 1])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default=(
            "data/scedc/model_manifests/"
            "scenario_t0_5s_k5.csv"
        ),
    )
    p.add_argument("--split-column", default="split_grouped")
    p.add_argument("--t0-sec", type=int, default=5)
    p.add_argument("--input-stations", type=int, default=5)
    p.add_argument("--target-stations", type=int, default=10)
    p.add_argument("--input-pre-sec", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument(
        "--out-dir",
        default="runs/sparse_field_pga_pgv_t0_5s_k5",
    )
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SparseFieldBaseline(
        hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Deterministic sparse-field baseline ===")
    print(f"Device             : {device}")
    print(f"Train events       : {len(train_dataset)}")
    print(f"Validation events  : {len(validation_dataset)}")
    print(f"Test events        : {len(test_dataset)}")
    print(
        f"Scenario           : T0={args.t0_sec}s, "
        f"K={args.input_stations}, "
        f"targets={args.target_stations}"
    )

    history = []
    best_validation = np.inf
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
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

        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} | "
            f"val PGA MAE={validation_metrics['mae_log10_pga']:.4f} | "
            f"val PGV MAE={validation_metrics['mae_log10_pgv']:.4f}"
        )

        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "validation_metrics": validation_metrics,
                },
                best_path,
            )

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        device,
    )

    (out_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    print("\n=== Best-model test metrics ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
