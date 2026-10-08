#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 1 / Step 3
Final-architecture ablation suite for Causal-SeisField.

All information ablations are built on the FINAL mean-pooling base:
  base
      waveform + geometry + elevation + P-offset + target coordinates
      + mean pooling
  no_waveform
      removes the waveform encoder; retains geometry and P-offset
  no_geometry
      retains waveform and P-offset, removes input/target spatial geometry
  no_p_offset
      retains waveform and geometry, removes relative P-arrival timing
  attention_pooling
      same full inputs as base but replaces mean pooling with attention pooling

Fairness:
- identical deterministic station combinations across all variants;
- training combinations change with epoch;
- validation/test combinations are fixed by event and repeat;
- all variants use the same split, T0, K, Q, optimizer and seed;
- final test metrics use targets -> repeats -> events aggregation.

Default main setting:
T0=5 s, K=5, Q=10.
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

VALID_VARIANTS = {
    "base",
    "no_waveform",
    "no_geometry",
    "no_p_offset",
    "attention_pooling",
}


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{text}:{base_seed}".encode("utf-8")
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


class SparseFieldDataset(Dataset):
    """
    Training:
        one deterministic-but-changing station draw per event per epoch.
    Validation/test:
        fixed repeated station draws per event.
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
                f"No events for {split_column}={split_name}."
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
            draw_key = f"epoch:{self.epoch}"
        else:
            event_index = int(index // self.repeats)
            repeat_index = int(index % self.repeats)
            draw_key = f"repeat:{repeat_index}"

        event = self.frame.iloc[event_index]
        event_id = str(event["event_id"])
        rng = np.random.default_rng(
            stable_seed(
                f"{self.split_name}:{event_id}:{draw_key}",
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
            & (p_offset <= float(self.t0_sec))
        )
        if len(triggered) < self.input_stations:
            raise RuntimeError(
                f"Event {event_id}: only {len(triggered)} "
                f"P-reached stations."
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
                f"Event {event_id}: only {len(target_pool)} "
                f"held-out target candidates."
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
                    (pre_first_p - self.input_pre_sec)
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
            nn.Conv1d(3, 32, 9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FinalAblationModel(nn.Module):
    def __init__(
        self,
        variant: str,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(variant)

        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.use_waveform = variant != "no_waveform"
        self.pooling = (
            "attention"
            if variant == "attention_pooling"
            else "mean"
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
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def apply_ablation(
        self,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        station_features = input_features
        query_features = target_features

        if self.variant == "no_geometry":
            station_features = station_features.clone()
            # x, y, elevation removed; P-offset retained.
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
        station_features, query_features = (
            self.apply_ablation(
                input_features,
                target_features,
            )
        )

        if self.use_waveform:
            b, k, c, t = input_waveforms.shape
            waveform_latent = self.waveform_encoder(
                input_waveforms.reshape(
                    b * k, c, t
                )
            ).reshape(b, k, -1)
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

        if self.pooling == "attention":
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
        else:
            event_latent = station_latent.mean(
                dim=1
            )

        expanded = event_latent[:, None, :].expand(
            -1,
            query_features.shape[1],
            -1,
        )
        return self.query_decoder(
            torch.cat(
                [
                    expanded,
                    query_features,
                ],
                dim=-1,
            )
        )


def batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    residual = prediction - target
    absolute = torch.abs(residual)
    return {
        "mae_pga": float(
            absolute[..., 0].mean().detach().cpu()
        ),
        "mae_pgv": float(
            absolute[..., 1].mean().detach().cpu()
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

    total_loss = 0.0
    total_count = 0
    pred_all = []
    true_all = []

    for batch in loader:
        input_waveforms = batch[
            "input_waveforms"
        ].to(device)
        input_features = batch[
            "input_features"
        ].to(device)
        target_features = batch[
            "target_features"
        ].to(device)
        target = batch[
            "target_log"
        ].to(device)

        with torch.set_grad_enabled(training):
            pred = model(
                input_waveforms,
                input_features,
                target_features,
            )
            loss = nn.functional.smooth_l1_loss(
                pred,
                target,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )
                optimizer.step()

        count = int(target.numel())
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
        pred_all.append(pred.detach().cpu())
        true_all.append(target.detach().cpu())

    pred_tensor = torch.cat(pred_all, dim=0)
    true_tensor = torch.cat(true_all, dim=0)
    metrics = batch_metrics(
        pred_tensor,
        true_tensor,
    )
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def repeated_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []

    with torch.inference_mode():
        for batch in loader:
            pred = model(
                batch["input_waveforms"].to(device),
                batch["input_features"].to(device),
                batch["target_features"].to(device),
            ).cpu().numpy()
            target = batch[
                "target_log"
            ].numpy()
            repeats = batch[
                "repeat"
            ].numpy()
            event_ids = list(
                batch["event_id"]
            )

            for b in range(target.shape[0]):
                for q in range(target.shape[1]):
                    rows.append(
                        {
                            "event_id": str(
                                event_ids[b]
                            ),
                            "repeat": int(
                                repeats[b]
                            ),
                            "target_index_within_repeat": int(q),
                            "true_log10_pga": float(
                                target[b, q, 0]
                            ),
                            "true_log10_pgv": float(
                                target[b, q, 1]
                            ),
                            "pred_log10_pga": float(
                                pred[b, q, 0]
                            ),
                            "pred_log10_pgv": float(
                                pred[b, q, 1]
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def canonical_metrics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    result = {
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
        pred = frame[
            f"pred_log10_{quantity}"
        ].to_numpy(dtype=float)
        residual = pred - truth

        work = frame[
            ["event_id", "repeat"]
        ].copy()
        work["mae"] = np.abs(residual)
        work["bias"] = residual
        work["factor2"] = (
            np.abs(residual) <= LOG10_FACTOR_2
        ).astype(float)

        er = (
            work.groupby(
                ["event_id", "repeat"],
                sort=False,
            )
            .agg(
                mae=("mae", "mean"),
                bias=("bias", "mean"),
                factor2=("factor2", "mean"),
            )
            .reset_index()
        )
        ev = (
            er.groupby(
                "event_id",
                sort=False,
            )[
                ["mae", "bias", "factor2"]
            ]
            .mean()
        )

        result[
            f"canonical_mae_{quantity}"
        ] = float(ev["mae"].mean())
        result[
            f"canonical_bias_{quantity}"
        ] = float(ev["bias"].mean())
        result[
            f"canonical_factor2_{quantity}"
        ] = float(ev["factor2"].mean())

        result[
            f"target_weighted_mae_{quantity}"
        ] = float(np.abs(residual).mean())

    return result


def train_variant(
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)

    train_ds = SparseFieldDataset(
        args.manifest,
        args.split_column,
        "train",
        args.t0_sec,
        args.input_stations,
        args.target_stations,
        args.input_pre_sec,
        args.seed,
        True,
    )
    val_ds = SparseFieldDataset(
        args.manifest,
        args.split_column,
        "validation",
        args.t0_sec,
        args.input_stations,
        args.target_stations,
        args.input_pre_sec,
        args.seed,
        False,
        repeats=args.validation_repeats,
    )
    test_ds = SparseFieldDataset(
        args.manifest,
        args.split_column,
        "test",
        args.t0_sec,
        args.input_stations,
        args.target_stations,
        args.input_pre_sec,
        args.seed,
        False,
        repeats=args.test_repeats,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": (
            args.num_workers > 0
        ),
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **common,
    )

    model = FinalAblationModel(
        variant,
        args.hidden_dim,
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
            min_lr=args.minimum_learning_rate,
        )
    )

    variant_dir = Path(args.out_root) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_dir / "best_model.pt"

    history = []
    best_loss = np.inf
    best_epoch = 0
    stale = 0
    start = time.time()

    print(f"\n=== Final-architecture ablation: {variant} ===")
    print(
        "Train/Val/Test events: "
        f"{len(train_ds.frame)}/"
        f"{len(val_ds.frame)}/"
        f"{len(test_ds.frame)}"
    )
    print(
        f"Validation repeats: {args.validation_repeats} | "
        f"Test repeats: {args.test_repeats}"
    )
    print(
        "Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        tr = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )
        va = run_epoch(
            model,
            val_loader,
            None,
            device,
        )
        scheduler.step(va["loss"])

        row = {
            "epoch": epoch,
            "variant": variant,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"validation_{k}": v for k, v in va.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            variant_dir / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={tr['loss']:.4f} | "
            f"val={va['loss']:.4f} | "
            f"PGA={va['mae_pga']:.4f} | "
            f"PGV={va['mae_pgv']:.4f}"
        )

        if va["loss"] < best_loss - args.minimum_delta:
            best_loss = va["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "epoch": epoch,
                    "args": vars(args),
                    "validation_metrics": va,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print(
                    f"Early stop at {epoch}; "
                    f"best={best_epoch}"
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

    pred = repeated_predictions(
        model,
        test_loader,
        device,
    )
    pred_path = (
        variant_dir
        / "repeated_test_predictions.csv"
    )
    pred.to_csv(
        pred_path,
        index=False,
    )

    metrics = canonical_metrics(pred)
    result = {
        "variant": variant,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "parameter_count": int(
            sum(
                p.numel()
                for p in model.parameters()
            )
        ),
        "training_seconds": float(
            time.time() - start
        ),
        "train_events": int(len(train_ds.frame)),
        "validation_events": int(len(val_ds.frame)),
        "test_events": int(len(test_ds.frame)),
        "validation_repeats": int(
            args.validation_repeats
        ),
        "test_repeats": int(
            args.test_repeats
        ),
        **metrics,
    }

    (
        variant_dir / "test_metrics.json"
    ).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        "Canonical test MAE | "
        f"PGA={metrics['canonical_mae_pga']:.4f} | "
        f"PGV={metrics['canonical_mae_pgv']:.4f}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default=(
            'data/scedc/model_manifests/scenario_t0_5s_k5.csv'
        ),
    )
    p.add_argument(
        "--split-column",
        default="split_grouped",
    )
    p.add_argument(
        "--variants",
        default=(
            "base,no_waveform,no_geometry,"
            "no_p_offset,attention_pooling"
        ),
    )
    p.add_argument("--t0-sec", type=int, default=5)
    p.add_argument("--input-stations", type=int, default=5)
    p.add_argument("--target-stations", type=int, default=10)
    p.add_argument("--input-pre-sec", type=float, default=2.0)
    p.add_argument("--validation-repeats", type=int, default=3)
    p.add_argument("--test-repeats", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr-patience", type=int, default=4)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--minimum-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--out-root",
        default=(
            'runs/phase1_final_mean_pooling_ablation'
        ),
    )
    args = p.parse_args()

    variants = [
        x.strip()
        for x in args.variants.split(",")
        if x.strip()
    ]
    unknown = set(variants).difference(
        VALID_VARIANTS
    )
    if unknown:
        raise ValueError(
            f"Unknown variants: {sorted(unknown)}"
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

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (
        out_root / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(device),
                "variants": variants,
                "canonical_metric": (
                    "targets -> repeats -> events"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results = []
    for variant in variants:
        result = train_variant(
            variant,
            args,
            device,
        )
        results.append(result)
        pd.DataFrame(results).to_csv(
            out_root / "ablation_summary_partial.csv",
            index=False,
        )

    summary = pd.DataFrame(results)
    summary.to_csv(
        out_root / "ablation_summary.csv",
        index=False,
    )

    columns = [
        "variant",
        "best_epoch",
        "parameter_count",
        "train_events",
        "validation_events",
        "test_events",
        "canonical_mae_pga",
        "canonical_mae_pgv",
        "canonical_bias_pga",
        "canonical_bias_pgv",
        "canonical_factor2_pga",
        "canonical_factor2_pgv",
    ]

    print("\n=== Phase 1 final-architecture ablation summary ===")
    print(summary[columns].to_string(index=False))
    print(f"\nOutputs: {out_root.resolve()}")


if __name__ == "__main__":
    main()
