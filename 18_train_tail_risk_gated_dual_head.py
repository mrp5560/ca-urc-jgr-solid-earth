#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18_train_tail_risk_gated_dual_head.py

Frozen mean-base + tail-risk gate + nonnegative residual correction.

Final prediction:
    y_final = y_base + sigmoid(gate) * positive_residual

The script trains only on the train split and selects checkpoints only on
repeated validation combinations. It never reads the test split.
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


LOG2 = math.log10(2.0)


def seed_from(text: str, seed: int) -> int:
    value = hashlib.sha256(
        f"{text}:{seed}".encode()
    ).digest()
    return int.from_bytes(value[:8], "little") % (2**32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def local_xy(
    coordinates: np.ndarray,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    x = (
        (coordinates[:, 1] - longitude)
        * 111.32
        * math.cos(math.radians(latitude))
    )
    y = (
        coordinates[:, 0] - latitude
    ) * 110.57
    return np.stack([x, y], axis=-1)


def horizontal_peak(array: np.ndarray) -> np.ndarray:
    return np.max(
        np.sqrt(
            array[:, 0] ** 2
            + array[:, 1] ** 2
        ),
        axis=-1,
    )


class EventDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        split_column: str,
        split_name: str,
        t0: int,
        k: int,
        targets: int,
        pre_sec: float,
        seed: int,
        training: bool,
        repeats: int = 1,
    ):
        frame = pd.read_csv(
            manifest,
            dtype={"event_id": str},
        )
        frame = frame.loc[
            frame[split_column]
            .astype(str)
            .eq(split_name)
        ].copy()

        eligible = f"eligible_t0_{t0}s_k{k}"
        if eligible in frame:
            frame = frame.loc[
                bool_series(frame[eligible])
            ].copy()

        self.frame = frame.reset_index(drop=True)
        self.t0 = t0
        self.k = k
        self.targets = targets
        self.pre_sec = pre_sec
        self.seed = seed
        self.training = training
        self.repeats = 1 if training else repeats
        self.epoch = 0

        if len(self.frame) == 0:
            raise ValueError(
                f"No events in {split_name}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.frame) * self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            event_index = index
            repeat = 0
        else:
            event_index = index // self.repeats
            repeat = index % self.repeats

        row = self.frame.iloc[event_index]
        event_id = str(row.event_id)

        key = (
            f"train:{event_id}:{self.epoch}"
            if self.training
            else f"validation:{event_id}:{repeat}"
        )
        rng = np.random.default_rng(
            seed_from(key, self.seed)
        )

        with h5py.File(row.h5_path, "r") as h5:
            acc = np.asarray(
                h5["acceleration"][:],
                dtype=np.float32,
            )
            vel = np.asarray(
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
            fs = float(
                h5.attrs["sampling_rate_hz"]
            )
            pre_first_p = float(
                h5.attrs["pre_first_p_sec"]
            )
            zero = int(
                h5.attrs["time_zero_index"]
            )

        triggered = np.flatnonzero(
            np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (p_offset <= self.t0)
        )
        if len(triggered) < self.k:
            raise RuntimeError(
                f"{event_id}: insufficient inputs"
            )

        input_ids = rng.choice(
            triggered,
            self.k,
            replace=False,
        )
        pool = np.setdiff1d(
            np.arange(len(coords)),
            input_ids,
        )
        target_ids = rng.choice(
            pool,
            self.targets,
            replace=False,
        )

        start = max(
            0,
            int(
                round(
                    (
                        pre_first_p
                        - self.pre_sec
                    )
                    * fs
                )
            ),
        )
        snapshot = min(
            zero + int(round(self.t0 * fs)),
            acc.shape[-1] - 1,
        )

        wave = acc[
            input_ids,
            :,
            start:snapshot,
        ]
        wave = (
            np.sign(wave)
            * np.log1p(np.abs(wave) / 1e-3)
        ).astype(np.float32)

        input_coords = coords[input_ids]
        target_coords = coords[target_ids]
        lat0 = float(
            input_coords[:, 0].mean()
        )
        lon0 = float(
            input_coords[:, 1].mean()
        )

        input_features = np.concatenate(
            [
                local_xy(
                    input_coords,
                    lat0,
                    lon0,
                ) / 100.0,
                input_coords[:, 2:3] / 2000.0,
                p_offset[input_ids, None]
                / max(self.t0, 1),
            ],
            axis=1,
        ).astype(np.float32)

        target_features = np.concatenate(
            [
                local_xy(
                    target_coords,
                    lat0,
                    lon0,
                ) / 100.0,
                target_coords[:, 2:3] / 2000.0,
            ],
            axis=1,
        ).astype(np.float32)

        pga = horizontal_peak(
            acc[target_ids, :, snapshot:]
        )
        pgv = horizontal_peak(
            vel[target_ids, :, snapshot:]
        )
        target = np.stack(
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
            row.get("magnitude", np.nan),
            errors="coerce",
        )
        target_triggered = (
            np.isfinite(p_offset[target_ids])
            & (p_offset[target_ids] <= self.t0)
        )

        return {
            "event_id": event_id,
            "repeat": repeat,
            "wave": torch.from_numpy(wave),
            "input_features": torch.from_numpy(
                input_features
            ),
            "target_features": torch.from_numpy(
                target_features
            ),
            "target": torch.from_numpy(target),
            "magnitude": torch.tensor(
                float(magnitude)
                if np.isfinite(magnitude)
                else np.nan
            ),
            "triggered": torch.from_numpy(
                target_triggered
            ),
        }


class WaveEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                3, 32, 9, 2, 4
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(
                32, 64, 7, 2, 3
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(
                64, 128, 5, 2, 2
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, hidden),
        )

    def forward(self, x):
        return self.network(x)


class MeanBase(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.waveform_encoder = WaveEncoder(
            hidden
        )
        self.station_encoder = nn.Sequential(
            nn.Linear(hidden + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.query_decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def features(
        self,
        wave,
        input_features,
        target_features,
    ):
        batch, stations, channels, samples = (
            wave.shape
        )
        encoded = self.waveform_encoder(
            wave.reshape(
                batch * stations,
                channels,
                samples,
            )
        ).reshape(batch, stations, -1)

        event = self.station_encoder(
            torch.cat(
                [encoded, input_features],
                dim=-1,
            )
        ).mean(dim=1)

        expanded = event[:, None].expand(
            -1,
            target_features.shape[1],
            -1,
        )
        base = self.query_decoder(
            torch.cat(
                [expanded, target_features],
                dim=-1,
            )
        )
        return expanded, base

    def forward(
        self,
        wave,
        input_features,
        target_features,
    ):
        return self.features(
            wave,
            input_features,
            target_features,
        )[1]


class TailGateModel(nn.Module):
    def __init__(
        self,
        base: MeanBase,
        hidden: int,
        tail_hidden: int,
        prevalence: float,
        maximum_correction: float,
    ):
        super().__init__()
        self.base = base
        self.maximum_correction = (
            maximum_correction
        )

        for parameter in base.parameters():
            parameter.requires_grad = False
        base.eval()

        self.context = nn.Sequential(
            nn.Linear(
                hidden + 3 + 2,
                tail_hidden,
            ),
            nn.GELU(),
            nn.LayerNorm(tail_hidden),
            nn.Linear(
                tail_hidden,
                tail_hidden,
            ),
            nn.GELU(),
        )
        self.gate = nn.Linear(
            tail_hidden,
            2,
        )
        self.residual = nn.Linear(
            tail_hidden,
            2,
        )

        bias = math.log(
            prevalence / (1 - prevalence)
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(
            self.gate.bias,
            bias,
        )
        nn.init.zeros_(
            self.residual.weight
        )
        nn.init.constant_(
            self.residual.bias,
            -4.0,
        )

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self,
        wave,
        input_features,
        target_features,
    ):
        with torch.no_grad():
            expanded, base = self.base.features(
                wave,
                input_features,
                target_features,
            )

        context = self.context(
            torch.cat(
                [
                    expanded,
                    target_features,
                    base,
                ],
                dim=-1,
            )
        )
        logits = self.gate(context)
        probability = torch.sigmoid(logits)
        residual = (
            self.maximum_correction
            * torch.sigmoid(
                self.residual(context)
            )
        )
        correction = probability * residual

        return {
            "base": base,
            "logits": logits,
            "probability": probability,
            "residual": residual,
            "correction": correction,
            "final": base + correction,
        }


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (
        values * mask
    ).sum() / mask.sum().clamp_min(1.0)


class Objective(nn.Module):
    def __init__(
        self,
        thresholds,
        positive_weight,
        gate_weight,
        tail_weight,
        residual_weight,
        preserve_weight,
    ):
        super().__init__()
        self.register_buffer(
            "thresholds",
            thresholds.reshape(1, 1, 2),
        )
        self.register_buffer(
            "positive_weight",
            positive_weight,
        )
        self.gate_weight = gate_weight
        self.tail_weight = tail_weight
        self.residual_weight = (
            residual_weight
        )
        self.preserve_weight = (
            preserve_weight
        )

    def forward(self, output, target):
        tail = (
            target >= self.thresholds
        ).float()
        tail_mask = tail.bool()
        non_tail = ~tail_mask

        gate_loss = (
            nn.functional
            .binary_cross_entropy_with_logits(
                output["logits"],
                tail,
                pos_weight=(
                    self.positive_weight
                ),
            )
        )

        tail_loss = masked_mean(
            nn.functional.smooth_l1_loss(
                output["final"],
                target,
                reduction="none",
            ),
            tail_mask,
        )

        residual_target = torch.relu(
            target - output["base"]
        )
        residual_loss = masked_mean(
            nn.functional.smooth_l1_loss(
                output["residual"],
                residual_target,
                reduction="none",
            ),
            tail_mask,
        )

        preserve_loss = masked_mean(
            (
                output["final"]
                - output["base"]
            ) ** 2,
            non_tail,
        )

        total = (
            self.gate_weight * gate_loss
            + self.tail_weight * tail_loss
            + self.residual_weight
            * residual_loss
            + self.preserve_weight
            * preserve_loss
        )

        return total, {
            "loss": total.item(),
            "gate": gate_loss.item(),
            "tail": tail_loss.item(),
            "residual": residual_loss.item(),
            "preserve": preserve_loss.item(),
        }


def event_macro(
    frame: pd.DataFrame,
    column: str,
    mask: str | None = None,
) -> float:
    selected = (
        frame
        if mask is None
        else frame.loc[frame[mask]]
    )
    return float(
        selected.groupby(
            ["event_id", "repeat"]
        )[column]
        .mean()
        .mean()
    )


def calculate_metrics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    result = {}

    for quantity in ("pga", "pgv"):
        truth = frame[
            f"true_{quantity}"
        ].to_numpy()
        tail = frame[
            f"tail_{quantity}"
        ].to_numpy(bool)

        for prefix in ("base", "final"):
            prediction = frame[
                f"{prefix}_{quantity}"
            ].to_numpy()
            error = prediction - truth
            absolute = np.abs(error)

            column = (
                f"{prefix}_abs_{quantity}"
            )
            frame[column] = absolute

            result[
                f"{prefix}_mae_{quantity}"
            ] = absolute.mean()
            result[
                f"{prefix}_bias_{quantity}"
            ] = error.mean()
            result[
                f"{prefix}_tail_under05_{quantity}"
            ] = np.mean(
                error[tail] <= -0.5
            )
            result[
                f"{prefix}_macro_{quantity}"
            ] = event_macro(
                frame,
                column,
            )
            result[
                f"{prefix}_tail_macro_{quantity}"
            ] = event_macro(
                frame,
                column,
                f"tail_{quantity}",
            )
            result[
                f"{prefix}_non_tail_macro_{quantity}"
            ] = event_macro(
                frame,
                column,
                f"non_tail_{quantity}",
            )

        labels = tail.astype(int)
        probability = frame[
            f"probability_{quantity}"
        ].to_numpy()

        result[
            f"auprc_{quantity}"
        ] = average_precision_score(
            labels,
            probability,
        )
        result[
            f"auroc_{quantity}"
        ] = roc_auc_score(
            labels,
            probability,
        )
        result[
            f"brier_{quantity}"
        ] = brier_score_loss(
            labels,
            probability,
        )

    return result


def constraints(
    metrics,
    overall_budget,
    non_tail_budget,
    bias_limit,
):
    excess = []

    for quantity in ("pga", "pgv"):
        excess.extend([
            max(
                0.0,
                metrics[
                    f"final_macro_{quantity}"
                ]
                - metrics[
                    f"base_macro_{quantity}"
                ]
                - overall_budget,
            ),
            max(
                0.0,
                metrics[
                    f"final_non_tail_macro_"
                    f"{quantity}"
                ]
                - metrics[
                    f"base_non_tail_macro_"
                    f"{quantity}"
                ]
                - non_tail_budget,
            ),
            max(
                0.0,
                abs(
                    metrics[
                        f"final_bias_{quantity}"
                    ]
                )
                - bias_limit,
            ),
        ])

    normalized = (
        excess[0] / overall_budget
        + excess[1] / non_tail_budget
        + excess[2] / bias_limit
        + excess[3] / overall_budget
        + excess[4] / non_tail_budget
        + excess[5] / bias_limit
    )

    tail_score = np.mean([
        metrics["final_tail_macro_pga"],
        metrics["final_tail_macro_pgv"],
    ])
    overall_score = np.mean([
        metrics["final_macro_pga"],
        metrics["final_macro_pgv"],
    ])
    gate_score = np.mean([
        metrics["auprc_pga"],
        metrics["auprc_pgv"],
    ])

    return {
        "feasible": all(
            value <= 1e-12
            for value in excess
        ),
        "violation": normalized,
        "tail_score": tail_score,
        "overall_score": overall_score,
        "gate_score": gate_score,
        "objective": (
            tail_score
            + 5.0 * normalized
        ),
    }


def train_epoch(
    model,
    loader,
    objective,
    optimizer,
    device,
):
    model.train()
    totals = {}
    batches = 0

    for batch in loader:
        output = model(
            batch["wave"].to(device),
            batch[
                "input_features"
            ].to(device),
            batch[
                "target_features"
            ].to(device),
        )
        loss, diagnostics = objective(
            output,
            batch["target"].to(device),
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter
                in model.parameters()
                if parameter.requires_grad
            ],
            5.0,
        )
        optimizer.step()

        for key, value in (
            diagnostics.items()
        ):
            totals[key] = (
                totals.get(key, 0)
                + value
            )
        batches += 1

    return {
        key: value / batches
        for key, value in totals.items()
    }


def evaluate(
    model,
    loader,
    thresholds,
    device,
):
    model.eval()
    rows = []

    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["wave"].to(device),
                batch[
                    "input_features"
                ].to(device),
                batch[
                    "target_features"
                ].to(device),
            )

            target = batch["target"].numpy()
            base = output["base"].cpu().numpy()
            final = output["final"].cpu().numpy()
            probability = (
                output["probability"]
                .cpu()
                .numpy()
            )
            magnitude = (
                batch["magnitude"]
                .numpy()
            )
            event_ids = list(
                batch["event_id"]
            )
            repeats = (
                batch["repeat"].numpy()
            )

            for b in range(target.shape[0]):
                for q in range(target.shape[1]):
                    pga = float(
                        target[b, q, 0]
                    )
                    pgv = float(
                        target[b, q, 1]
                    )
                    rows.append({
                        "event_id": event_ids[b],
                        "repeat": int(repeats[b]),
                        "magnitude": float(
                            magnitude[b]
                        ),
                        "true_pga": pga,
                        "true_pgv": pgv,
                        "tail_pga": (
                            pga >= thresholds[0]
                        ),
                        "tail_pgv": (
                            pgv >= thresholds[1]
                        ),
                        "non_tail_pga": (
                            pga < thresholds[0]
                        ),
                        "non_tail_pgv": (
                            pgv < thresholds[1]
                        ),
                        "base_pga": float(
                            base[b, q, 0]
                        ),
                        "base_pgv": float(
                            base[b, q, 1]
                        ),
                        "final_pga": float(
                            final[b, q, 0]
                        ),
                        "final_pgv": float(
                            final[b, q, 1]
                        ),
                        "probability_pga": float(
                            probability[b, q, 0]
                        ),
                        "probability_pgv": float(
                            probability[b, q, 1]
                        ),
                    })

    frame = pd.DataFrame(rows)
    return calculate_metrics(frame), frame


def load_checkpoint(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    args,
    metrics,
    status,
):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": (
            optimizer.state_dict()
        ),
        "epoch": epoch,
        "args": vars(args),
        "validation_metrics": metrics,
        "constraint_status": status,
    }, path)


def main():
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
        "--base-checkpoint",
        default=(
            "runs/mean_pooling_tail_t0_5s_k5/"
            "mean_base/best_model.pt"
        ),
    )
    parser.add_argument(
        "--threshold-json",
        default=(
            "runs/tail_gated_compromise_t0_5s_k5/"
            "tail_thresholds_q0.90_t0_5s.json"
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
        "--tail-hidden",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--maximum-correction",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--lambda-gate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda-tail",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda-residual",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--lambda-preserve",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--overall-budget",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--non-tail-budget",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--bias-limit",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
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
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "tail_risk_gated_dual_head_t0_5s_k5"
        ),
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if (
            args.device == "auto"
            and torch.cuda.is_available()
        )
        else (
            "cpu"
            if args.device == "auto"
            else args.device
        )
    )
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_data = json.loads(
        Path(args.threshold_json)
        .read_text(encoding="utf-8")
    )
    thresholds = np.array([
        threshold_data[
            "log10_pga_threshold"
        ],
        threshold_data[
            "log10_pgv_threshold"
        ],
    ], dtype=np.float32)
    prevalence = max(
        0.01,
        1.0
        - threshold_data.get(
            "quantile",
            0.9,
        ),
    )
    positive_weight = torch.tensor(
        [
            (1 - prevalence) / prevalence,
            (1 - prevalence) / prevalence,
        ],
        dtype=torch.float32,
        device=device,
    )

    base_checkpoint = load_checkpoint(
        args.base_checkpoint,
        device,
    )
    hidden = int(
        base_checkpoint.get(
            "args",
            {},
        ).get(
            "hidden_dim",
            base_checkpoint[
                "model_state"
            ][
                "station_encoder.0.weight"
            ].shape[0],
        )
    )

    base = MeanBase(hidden).to(device)
    base.load_state_dict(
        base_checkpoint["model_state"]
    )

    model = TailGateModel(
        base,
        hidden,
        args.tail_hidden,
        prevalence,
        args.maximum_correction,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    objective = Objective(
        torch.tensor(
            thresholds,
            dtype=torch.float32,
            device=device,
        ),
        positive_weight,
        args.lambda_gate,
        args.lambda_tail,
        args.lambda_residual,
        args.lambda_preserve,
    ).to(device)

    train_data = EventDataset(
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
    validation_data = EventDataset(
        args.manifest,
        args.split_column,
        "validation",
        args.t0_sec,
        args.input_stations,
        args.target_stations,
        args.input_pre_sec,
        args.seed,
        False,
        args.validation_repeats,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(
        "=== Tail-risk gated dual-head ==="
    )
    print(f"Device             : {device}")
    print(
        f"Train/validation   : "
        f"{len(train_data)}/"
        f"{len(validation_data)}"
    )
    print(
        f"Validation repeats : "
        f"{args.validation_repeats}"
    )
    print(
        f"Trainable params   : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    paths = {
        name: out_dir / f"best_{name}.pt"
        for name in (
            "overall",
            "tail",
            "gate",
            "feasible",
            "min_violation",
        )
    }
    best = {
        "overall": np.inf,
        "tail": np.inf,
        "gate": -np.inf,
        "feasible": np.inf,
        "min_violation": (
            np.inf,
            np.inf,
        ),
    }

    history = []
    best_objective = np.inf
    stale = 0
    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_data.set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            objective,
            optimizer,
            device,
        )
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            thresholds,
            device,
        )
        status = constraints(
            validation_metrics,
            args.overall_budget,
            args.non_tail_budget,
            args.bias_limit,
        )

        history.append({
            "epoch": epoch,
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
            **{
                f"constraint_{key}": value
                for key, value
                in status.items()
            },
        })
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv",
            index=False,
        )

        if status["overall_score"] < best["overall"]:
            best["overall"] = status[
                "overall_score"
            ]
            save_checkpoint(
                paths["overall"],
                model,
                optimizer,
                epoch,
                args,
                validation_metrics,
                status,
            )

        if status["tail_score"] < best["tail"]:
            best["tail"] = status[
                "tail_score"
            ]
            save_checkpoint(
                paths["tail"],
                model,
                optimizer,
                epoch,
                args,
                validation_metrics,
                status,
            )

        if status["gate_score"] > best["gate"]:
            best["gate"] = status[
                "gate_score"
            ]
            save_checkpoint(
                paths["gate"],
                model,
                optimizer,
                epoch,
                args,
                validation_metrics,
                status,
            )

        if (
            status["feasible"]
            and status["tail_score"]
            < best["feasible"]
        ):
            best["feasible"] = status[
                "tail_score"
            ]
            save_checkpoint(
                paths["feasible"],
                model,
                optimizer,
                epoch,
                args,
                validation_metrics,
                status,
            )

        pair = (
            status["violation"],
            status["tail_score"],
        )
        if pair < best["min_violation"]:
            best["min_violation"] = pair
            save_checkpoint(
                paths["min_violation"],
                model,
                optimizer,
                epoch,
                args,
                validation_metrics,
                status,
            )

        if status["objective"] < best_objective:
            best_objective = status[
                "objective"
            ]
            stale = 0
        else:
            stale += 1

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"overall={status['overall_score']:.4f} | "
            f"tail={status['tail_score']:.4f} | "
            f"violation={status['violation']:.3f} | "
            f"feasible={status['feasible']} | "
            f"AUPRC={status['gate_score']:.4f}"
        )

        if stale >= args.patience:
            print(
                f"Early stopping at {epoch}"
            )
            break

    results = []
    for name, path in paths.items():
        if not path.exists():
            continue

        checkpoint = load_checkpoint(
            path,
            device,
        )
        model.load_state_dict(
            checkpoint["model_state"]
        )
        metrics, predictions = evaluate(
            model,
            validation_loader,
            thresholds,
            device,
        )
        status = constraints(
            metrics,
            args.overall_budget,
            args.non_tail_budget,
            args.bias_limit,
        )

        predictions.to_csv(
            out_dir
            / f"validation_predictions_{name}.csv",
            index=False,
        )

        result = {
            "selection": name,
            "epoch": checkpoint["epoch"],
            **metrics,
            **{
                f"constraint_{key}": value
                for key, value
                in status.items()
            },
        }
        results.append(result)

        print(
            f"\n{name}: "
            f"epoch={checkpoint['epoch']}, "
            f"feasible={status['feasible']}, "
            f"overall="
            f"{metrics['final_macro_pga']:.4f}/"
            f"{metrics['final_macro_pgv']:.4f}, "
            f"tail="
            f"{metrics['final_tail_macro_pga']:.4f}/"
            f"{metrics['final_tail_macro_pgv']:.4f}, "
            f"AUPRC="
            f"{metrics['auprc_pga']:.4f}/"
            f"{metrics['auprc_pgv']:.4f}"
        )

    pd.DataFrame(results).to_csv(
        out_dir
        / "checkpoint_validation_summary.csv",
        index=False,
    )

    configuration = {
        **vars(args),
        "device": str(device),
        "thresholds": threshold_data,
        "test_split_evaluated": False,
        "training_seconds": (
            time.time() - start_time
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

    if paths["feasible"].exists():
        print(
            "\nFeasible checkpoint: "
            f"{paths['feasible'].resolve()}"
        )
    else:
        print(
            "\nNo feasible checkpoint. "
            "Use min_violation only for diagnosis."
        )

    print(
        f"\nOutputs: {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
