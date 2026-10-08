#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
17_evaluate_tail_checkpoints_repeated.py

Repeated paired evaluation for sparse-to-field PGA/PGV checkpoints.

This script evaluates several checkpoints on exactly the same:
    - test events,
    - input-station combinations,
    - target-station combinations,
    - repeated random seeds.

It is intended to determine whether strong-motion tail improvements are stable
rather than an artifact of one fixed station sample.

Default comparison
------------------
1. mean_base:
   runs/mean_pooling_tail_t0_5s_k5/mean_base/best_model.pt

2. tail_gated_mild checkpoints:
   runs/tail_gated_compromise_t0_5s_k5/tail_gated_mild/best_overall.pt
   runs/tail_gated_compromise_t0_5s_k5/tail_gated_mild/best_tail.pt
   runs/tail_gated_compromise_t0_5s_k5/tail_gated_mild/best_compromise.pt

Main outputs
------------
paired_target_predictions.csv
model_metrics_summary.csv
event_level_metrics.csv
paired_bootstrap_deltas.csv
evaluation_metadata.json

Paired bootstrap interpretation
-------------------------------
For MAE, absolute bias, and Under05:
    delta = candidate - reference
    delta < 0 means the candidate is better.

For factor-of-two accuracy:
    delta = candidate - reference
    delta > 0 means the candidate is better.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


LOG10_FACTOR_2 = math.log10(2.0)

DEFAULT_MODELS = (
    "mean_base=runs/mean_pooling_tail_t0_5s_k5/mean_base/best_model.pt;"
    "light_overall=runs/tail_gated_compromise_t0_5s_k5/tail_gated_light/best_overall.pt;"
    "light_tail=runs/tail_gated_compromise_t0_5s_k5/tail_gated_light/best_tail.pt;light_compromise=runs/tail_gated_compromise_t0_5s_k5/tail_gated_light/best_compromise.pt"
)


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{text}:{base_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


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

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.network(waveforms)


class MeanPoolingSparseFieldModel(nn.Module):
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
        batch_size, station_count, channel_count, sample_count = (
            input_waveforms.shape
        )

        waveform_latent = self.waveform_encoder(
            input_waveforms.reshape(
                batch_size * station_count,
                channel_count,
                sample_count,
            )
        ).reshape(batch_size, station_count, -1)

        station_latent = self.station_encoder(
            torch.cat([waveform_latent, input_features], dim=-1)
        )

        event_latent = station_latent.mean(dim=1)

        query_count = target_features.shape[1]
        expanded_event = event_latent[:, None, :].expand(
            -1, query_count, -1
        )

        return self.query_decoder(
            torch.cat([expanded_event, target_features], dim=-1)
        )


def parse_model_specification(specification: str) -> dict[str, Path]:
    models: dict[str, Path] = {}

    for item in specification.split(";"):
        item = item.strip()
        if not item:
            continue

        if "=" not in item:
            raise ValueError(
                "Each --models entry must use name=checkpoint_path."
            )

        name, path_text = item.split("=", maxsplit=1)
        name = name.strip()
        checkpoint_path = Path(path_text.strip())

        if not name:
            raise ValueError("Model name cannot be empty.")
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")

        models[name] = checkpoint_path

    if len(models) < 2:
        raise ValueError(
            "At least two models are required for paired evaluation."
        )

    return models


def torch_load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
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

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint is not a dictionary: {checkpoint_path}"
        )
    if "model_state" not in checkpoint:
        raise KeyError(
            f"Checkpoint has no model_state: {checkpoint_path}"
        )

    return checkpoint


def infer_hidden_dim(checkpoint: dict[str, Any]) -> int:
    checkpoint_args = checkpoint.get("args", {})
    if (
        isinstance(checkpoint_args, dict)
        and "hidden_dim" in checkpoint_args
    ):
        return int(checkpoint_args["hidden_dim"])

    state = checkpoint["model_state"]

    for key in (
        "station_encoder.0.weight",
        "waveform_encoder.network.13.weight",
    ):
        if key in state:
            return int(state[key].shape[0])

    raise ValueError(
        "Unable to infer hidden_dim from checkpoint."
    )


def load_models(
    model_paths: dict[str, Path],
    device: torch.device,
) -> tuple[
    dict[str, MeanPoolingSparseFieldModel],
    dict[str, dict[str, Any]],
]:
    models: dict[str, MeanPoolingSparseFieldModel] = {}
    checkpoints: dict[str, dict[str, Any]] = {}

    for name, checkpoint_path in model_paths.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for {name}: {checkpoint_path}"
            )

        checkpoint = torch_load_checkpoint(
            checkpoint_path,
            device,
        )
        hidden_dim = infer_hidden_dim(checkpoint)

        model = MeanPoolingSparseFieldModel(
            hidden_dim=hidden_dim
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        models[name] = model
        checkpoints[name] = checkpoint

        print(
            f"Loaded {name}: {checkpoint_path.resolve()} "
            f"(hidden_dim={hidden_dim})"
        )

    return models, checkpoints


def resolve_thresholds(
    threshold_json: str | None,
    checkpoints: dict[str, dict[str, Any]],
) -> dict[str, float]:
    if threshold_json:
        threshold_path = Path(threshold_json)
        if not threshold_path.exists():
            raise FileNotFoundError(
                f"Threshold JSON not found: {threshold_path}"
            )
        thresholds = json.loads(
            threshold_path.read_text(encoding="utf-8")
        )
    else:
        thresholds = None
        for checkpoint in checkpoints.values():
            candidate = checkpoint.get("thresholds")
            if isinstance(candidate, dict):
                thresholds = candidate
                break

        if thresholds is None:
            raise ValueError(
                "No threshold JSON was supplied and no checkpoint "
                "contains tail thresholds."
            )

    required = {
        "log10_pga_threshold",
        "log10_pgv_threshold",
    }
    missing = required.difference(thresholds)
    if missing:
        raise ValueError(
            f"Tail thresholds are missing: {sorted(missing)}"
        )

    return {
        "log10_pga_threshold": float(
            thresholds["log10_pga_threshold"]
        ),
        "log10_pgv_threshold": float(
            thresholds["log10_pgv_threshold"]
        ),
        "quantile": float(thresholds.get("quantile", np.nan)),
        "t0_sec": int(thresholds.get("t0_sec", -1)),
    }


def prepare_model_inputs(
    acceleration: np.ndarray,
    coordinates: np.ndarray,
    p_offset: np.ndarray,
    input_indices: np.ndarray,
    target_indices: np.ndarray,
    input_start_index: int,
    snapshot_index: int,
    t0_sec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_waveforms = acceleration[
        input_indices,
        :,
        input_start_index:snapshot_index,
    ]

    amplitude_scale = 1e-3
    input_waveforms = (
        np.sign(input_waveforms)
        * np.log1p(np.abs(input_waveforms) / amplitude_scale)
    ).astype(np.float32)

    input_coordinates = coordinates[input_indices]
    target_coordinates = coordinates[target_indices]

    origin_latitude = float(np.mean(input_coordinates[:, 0]))
    origin_longitude = float(np.mean(input_coordinates[:, 1]))

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

    return input_waveforms, input_features, target_features


def metric_from_group(
    group: pd.DataFrame,
    model_name: str,
    quantity: str,
    metric_name: str,
) -> float:
    truth_column = f"true_log10_{quantity}"
    prediction_column = f"{model_name}_log10_{quantity}"

    residual = (
        group[prediction_column].to_numpy(dtype=float)
        - group[truth_column].to_numpy(dtype=float)
    )
    absolute_error = np.abs(residual)

    if metric_name == "mae":
        return float(np.mean(absolute_error))
    if metric_name == "bias":
        return float(np.mean(residual))
    if metric_name == "absolute_bias":
        return float(abs(np.mean(residual)))
    if metric_name == "factor2":
        return float(
            np.mean(absolute_error <= LOG10_FACTOR_2)
        )
    if metric_name == "under05":
        return float(np.mean(residual <= -0.5))

    raise ValueError(f"Unknown metric: {metric_name}")


def filter_metric_rows(
    frame: pd.DataFrame,
    quantity: str,
    population: str,
) -> pd.DataFrame:
    if population == "overall":
        return frame
    if population == "tail":
        return frame.loc[frame[f"is_tail_{quantity}"]]
    if population == "m4plus":
        return frame.loc[frame["magnitude"] >= 4.0]
    if population == "triggered":
        return frame.loc[
            frame["target_triggered_by_snapshot"]
        ]
    if population == "not_triggered":
        return frame.loc[
            ~frame["target_triggered_by_snapshot"]
        ]

    raise ValueError(f"Unknown population: {population}")


def build_event_level_metrics(
    predictions: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    populations = (
        "overall",
        "tail",
        "m4plus",
        "triggered",
        "not_triggered",
    )
    metrics = (
        "mae",
        "bias",
        "absolute_bias",
        "factor2",
        "under05",
    )

    for (
        event_id,
        repeat_index,
    ), event_repeat in predictions.groupby(
        ["event_id", "repeat"],
        sort=False,
    ):
        magnitude = float(event_repeat["magnitude"].iloc[0])

        for model_name in model_names:
            for quantity in ("pga", "pgv"):
                for population in populations:
                    selected = filter_metric_rows(
                        event_repeat,
                        quantity,
                        population,
                    )
                    if len(selected) == 0:
                        continue

                    for metric_name in metrics:
                        value = metric_from_group(
                            selected,
                            model_name,
                            quantity,
                            metric_name,
                        )

                        rows.append({
                            "event_id": str(event_id),
                            "repeat": int(repeat_index),
                            "magnitude": magnitude,
                            "model": model_name,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric_name,
                            "value": value,
                            "n_target_rows": int(len(selected)),
                        })

    return pd.DataFrame(rows)


def summarize_models(
    predictions: pd.DataFrame,
    event_metrics: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    populations = (
        "overall",
        "tail",
        "m4plus",
        "triggered",
        "not_triggered",
    )
    metrics = (
        "mae",
        "bias",
        "absolute_bias",
        "factor2",
        "under05",
    )

    for model_name in model_names:
        for quantity in ("pga", "pgv"):
            for population in populations:
                selected = filter_metric_rows(
                    predictions,
                    quantity,
                    population,
                )
                if len(selected) == 0:
                    continue

                for metric_name in metrics:
                    target_weighted = metric_from_group(
                        selected,
                        model_name,
                        quantity,
                        metric_name,
                    )

                    event_subset = event_metrics.loc[
                        (
                            event_metrics["model"]
                            == model_name
                        )
                        & (
                            event_metrics["quantity"]
                            == quantity
                        )
                        & (
                            event_metrics["population"]
                            == population
                        )
                        & (
                            event_metrics["metric"]
                            == metric_name
                        )
                    ]

                    rows.append({
                        "model": model_name,
                        "quantity": quantity,
                        "population": population,
                        "metric": metric_name,
                        "target_weighted": float(
                            target_weighted
                        ),
                        "event_repeat_macro": float(
                            event_subset["value"].mean()
                        ),
                        "n_events": int(
                            event_subset["event_id"].nunique()
                        ),
                        "n_event_repeats": int(
                            len(event_subset)
                        ),
                        "n_target_rows": int(len(selected)),
                    })

    return pd.DataFrame(rows)


def paired_bootstrap(
    event_metrics: pd.DataFrame,
    model_names: list[str],
    reference_name: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output_rows: list[dict[str, Any]] = []

    candidate_names = [
        model_name
        for model_name in model_names
        if model_name != reference_name
    ]

    metric_keys = (
        event_metrics[
            ["quantity", "population", "metric"]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for quantity, population, metric_name in metric_keys:
        metric_frame = event_metrics.loc[
            (
                event_metrics["quantity"]
                == quantity
            )
            & (
                event_metrics["population"]
                == population
            )
            & (
                event_metrics["metric"]
                == metric_name
            )
        ]

        pivot = metric_frame.pivot_table(
            index=["event_id", "repeat"],
            columns="model",
            values="value",
            aggfunc="mean",
        )

        if reference_name not in pivot.columns:
            continue

        for candidate_name in candidate_names:
            if candidate_name not in pivot.columns:
                continue

            paired = pivot[
                [reference_name, candidate_name]
            ].dropna()

            if len(paired) == 0:
                continue

            paired = paired.copy()
            paired["delta"] = (
                paired[candidate_name]
                - paired[reference_name]
            )

            event_deltas = (
                paired["delta"]
                .groupby(level="event_id")
                .mean()
                .to_numpy(dtype=float)
            )

            if len(event_deltas) == 0:
                continue

            bootstrap_means = np.empty(
                repetitions,
                dtype=float,
            )

            for bootstrap_index in range(repetitions):
                sampled_indices = rng.integers(
                    0,
                    len(event_deltas),
                    size=len(event_deltas),
                )
                bootstrap_means[bootstrap_index] = np.mean(
                    event_deltas[sampled_indices]
                )

            observed_delta = float(np.mean(event_deltas))

            if metric_name == "factor2":
                favorable_probability = float(
                    np.mean(bootstrap_means > 0.0)
                )
                direction = (
                    "positive delta favors candidate"
                )
            else:
                favorable_probability = float(
                    np.mean(bootstrap_means < 0.0)
                )
                direction = (
                    "negative delta favors candidate"
                )

            output_rows.append({
                "reference_model": reference_name,
                "candidate_model": candidate_name,
                "quantity": quantity,
                "population": population,
                "metric": metric_name,
                "n_paired_events": int(len(event_deltas)),
                "n_paired_event_repeats": int(len(paired)),
                "mean_delta_candidate_minus_reference": (
                    observed_delta
                ),
                "ci95_low": float(
                    np.percentile(
                        bootstrap_means,
                        2.5,
                    )
                ),
                "ci95_high": float(
                    np.percentile(
                        bootstrap_means,
                        97.5,
                    )
                ),
                "probability_candidate_favorable": (
                    favorable_probability
                ),
                "interpretation": direction,
                "bootstrap_repetitions": int(repetitions),
            })

    return pd.DataFrame(output_rows)


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
        "--split-name",
        default="validation",
        choices=["train", "validation", "test"],
    )

    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Semicolon-separated entries: "
            "name=checkpoint_path;name=checkpoint_path"
        ),
    )
    parser.add_argument(
        "--reference-model",
        default="mean_base",
    )
    parser.add_argument(
        "--threshold-json",
        default="./runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json",
        help=(
            "Optional threshold JSON. "
            "If omitted, thresholds are read "
            "from a checkpoint."
        ),
    )

    parser.add_argument("--t0-sec", type=int, default=5)
    parser.add_argument("--input-stations", type=int, default=5)
    parser.add_argument(
        "--target-stations",
        type=int,
        default=10,
        help="Use 0 to evaluate all non-input stations.",
    )
    parser.add_argument(
        "--input-pre-sec",
        type=float,
        default=2.0,
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    parser.add_argument("--seed", type=int, default=20260713)

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--checkpoint-every-events",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        default="runs/tail_checkpoint_repeated_validation_light",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )
    if args.bootstrap_repetitions < 100:
        raise ValueError(
            "--bootstrap-repetitions should be at least 100."
        )
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

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
        device = torch.device(args.device)

    output_directory = Path(args.out_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_paths = parse_model_specification(
        args.models
    )

    if args.reference_model not in model_paths:
        raise ValueError(
            "Reference model is absent from "
            f"--models: {args.reference_model}"
        )

    models, checkpoints = load_models(
        model_paths,
        device,
    )
    model_names = list(models)

    thresholds = resolve_thresholds(
        args.threshold_json,
        checkpoints,
    )

    manifest = pd.read_csv(
        args.manifest,
        dtype={"event_id": str},
    )

    events = manifest.loc[
        manifest[args.split_column]
        .astype(str)
        .eq(args.split_name)
    ].copy()

    eligible_column = (
        f"eligible_t0_{args.t0_sec}s_"
        f"k{args.input_stations}"
    )
    if eligible_column in events.columns:
        events = events.loc[
            as_bool(events[eligible_column])
        ].copy()

    events = events.reset_index(drop=True)

    if args.max_events is not None:
        events = events.iloc[:args.max_events].copy()

    if len(events) == 0:
        raise ValueError(
            "No eligible evaluation events."
        )

    print("\n=== Repeated paired evaluation ===")
    print(f"Device             : {device}")
    print(f"Split              : {args.split_name}")
    print(f"Events             : {len(events)}")
    print(f"Repeats/event      : {args.repeats}")
    print(f"Models             : {', '.join(model_names)}")
    print(f"Reference          : {args.reference_model}")
    print(
        "Tail thresholds    : "
        f"PGA={thresholds['log10_pga_threshold']:.4f}, "
        f"PGV={thresholds['log10_pgv_threshold']:.4f}"
    )

    prediction_rows: list[dict[str, Any]] = []

    for event_number, event in enumerate(
        tqdm(
            events.itertuples(index=False),
            total=len(events),
            desc="paired evaluation",
        ),
        start=1,
    ):
        event_id = str(event.event_id)
        h5_path = Path(str(event.h5_path))

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
            & (
                p_offset
                <= float(args.t0_sec)
            )
        )

        if len(triggered_indices) < args.input_stations:
            continue

        input_start_index = int(
            round(
                (
                    pre_first_p
                    - args.input_pre_sec
                )
                * sampling_rate
            )
        )
        input_start_index = max(0, input_start_index)

        snapshot_index = (
            time_zero_index
            + int(
                round(
                    args.t0_sec
                    * sampling_rate
                )
            )
        )
        snapshot_index = min(
            snapshot_index,
            acceleration.shape[-1] - 1,
        )

        magnitude = pd.to_numeric(
            getattr(event, "magnitude", np.nan),
            errors="coerce",
        )
        magnitude = (
            float(magnitude)
            if np.isfinite(magnitude)
            else np.nan
        )

        sequence_group = pd.to_numeric(
            getattr(event, "sequence_group", -1),
            errors="coerce",
        )
        sequence_group = (
            int(sequence_group)
            if np.isfinite(sequence_group)
            else -1
        )

        for repeat_index in range(args.repeats):
            rng = np.random.default_rng(
                stable_seed(
                    (
                        f"{args.split_name}:"
                        f"{event_id}:"
                        f"repeat:{repeat_index}"
                    ),
                    args.seed,
                )
            )

            input_indices = rng.choice(
                triggered_indices,
                size=args.input_stations,
                replace=False,
            )

            target_pool = np.setdiff1d(
                np.arange(len(coordinates)),
                input_indices,
                assume_unique=False,
            )

            if args.target_stations == 0:
                target_indices = target_pool
            else:
                if len(target_pool) < args.target_stations:
                    continue

                target_indices = rng.choice(
                    target_pool,
                    size=args.target_stations,
                    replace=False,
                )

            (
                input_waveforms,
                input_features,
                target_features,
            ) = prepare_model_inputs(
                acceleration=acceleration,
                coordinates=coordinates,
                p_offset=p_offset,
                input_indices=input_indices,
                target_indices=target_indices,
                input_start_index=input_start_index,
                snapshot_index=snapshot_index,
                t0_sec=args.t0_sec,
            )

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

            true_log_pga = np.log10(
                np.maximum(
                    horizontal_peak(future_acceleration),
                    1e-10,
                )
            )
            true_log_pgv = np.log10(
                np.maximum(
                    horizontal_peak(future_velocity),
                    1e-12,
                )
            )

            waveform_tensor = (
                torch.from_numpy(input_waveforms)[None]
                .to(device)
            )
            input_feature_tensor = (
                torch.from_numpy(input_features)[None]
                .to(device)
            )
            target_feature_tensor = (
                torch.from_numpy(target_features)[None]
                .to(device)
            )

            model_predictions: dict[str, np.ndarray] = {}

            with torch.inference_mode():
                for model_name, model in models.items():
                    model_predictions[model_name] = (
                        model(
                            waveform_tensor,
                            input_feature_tensor,
                            target_feature_tensor,
                        )[0]
                        .cpu()
                        .numpy()
                    )

            input_station_text = "|".join(
                str(station_ids[index])
                for index in input_indices
            )

            for (
                local_target_index,
                station_index,
            ) in enumerate(target_indices):
                target_triggered = bool(
                    np.isfinite(p_offset[station_index])
                    and (
                        p_offset[station_index]
                        <= args.t0_sec
                    )
                )

                row: dict[str, Any] = {
                    "event_id": event_id,
                    "repeat": int(repeat_index),
                    "sequence_group": sequence_group,
                    "magnitude": magnitude,
                    "input_station_ids": input_station_text,
                    "target_station_id": str(
                        station_ids[station_index]
                    ),
                    "target_station_index": int(
                        station_index
                    ),
                    "target_p_offset_sec": (
                        float(p_offset[station_index])
                        if np.isfinite(
                            p_offset[station_index]
                        )
                        else np.nan
                    ),
                    "target_triggered_by_snapshot": (
                        target_triggered
                    ),
                    "true_log10_pga": float(
                        true_log_pga[local_target_index]
                    ),
                    "true_log10_pgv": float(
                        true_log_pgv[local_target_index]
                    ),
                    "is_tail_pga": bool(
                        true_log_pga[local_target_index]
                        >= thresholds[
                            "log10_pga_threshold"
                        ]
                    ),
                    "is_tail_pgv": bool(
                        true_log_pgv[local_target_index]
                        >= thresholds[
                            "log10_pgv_threshold"
                        ]
                    ),
                }

                for model_name, prediction in (
                    model_predictions.items()
                ):
                    row[
                        f"{model_name}_log10_pga"
                    ] = float(
                        prediction[
                            local_target_index,
                            0,
                        ]
                    )
                    row[
                        f"{model_name}_log10_pgv"
                    ] = float(
                        prediction[
                            local_target_index,
                            1,
                        ]
                    )

                prediction_rows.append(row)

        if (
            event_number
            % max(
                1,
                args.checkpoint_every_events,
            )
            == 0
            or event_number == len(events)
        ):
            pd.DataFrame(prediction_rows).to_csv(
                output_directory
                / "paired_predictions_partial.csv",
                index=False,
            )

    predictions = pd.DataFrame(prediction_rows)

    if len(predictions) == 0:
        raise RuntimeError(
            "No prediction rows were generated."
        )

    prediction_path = (
        output_directory
        / "paired_target_predictions.csv"
    )
    predictions.to_csv(
        prediction_path,
        index=False,
    )

    event_metrics = build_event_level_metrics(
        predictions,
        model_names,
    )
    event_metrics.to_csv(
        output_directory
        / "event_level_metrics.csv",
        index=False,
    )

    summary = summarize_models(
        predictions,
        event_metrics,
        model_names,
    )
    summary.to_csv(
        output_directory
        / "model_metrics_summary.csv",
        index=False,
    )

    bootstrap = paired_bootstrap(
        event_metrics=event_metrics,
        model_names=model_names,
        reference_name=args.reference_model,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    bootstrap.to_csv(
        output_directory
        / "paired_bootstrap_deltas.csv",
        index=False,
    )

    metadata = {
        "arguments": vars(args),
        "resolved_device": str(device),
        "model_paths": {
            name: str(path.resolve())
            for name, path in model_paths.items()
        },
        "reference_model": args.reference_model,
        "tail_thresholds": thresholds,
        "n_events": int(
            predictions["event_id"].nunique()
        ),
        "n_event_repeats": int(
            predictions[
                ["event_id", "repeat"]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "n_target_rows": int(len(predictions)),
    }

    (
        output_directory
        / "evaluation_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    key_summary = summary.loc[
        (
            summary["population"]
            .isin(
                [
                    "overall",
                    "tail",
                    "m4plus",
                    "triggered",
                ]
            )
        )
        & (
            summary["metric"]
            .isin(
                [
                    "mae",
                    "bias",
                    "under05",
                ]
            )
        )
    ]

    print("\n=== Model metrics summary ===")
    print(
        key_summary[
            [
                "model",
                "quantity",
                "population",
                "metric",
                "event_repeat_macro",
                "target_weighted",
                "n_events",
                "n_target_rows",
            ]
        ].to_string(index=False)
    )

    key_bootstrap = bootstrap.loc[
        (
            bootstrap["population"]
            .isin(
                [
                    "overall",
                    "tail",
                    "m4plus",
                    "triggered",
                ]
            )
        )
        & (
            bootstrap["metric"]
            .isin(
                [
                    "mae",
                    "absolute_bias",
                    "under05",
                ]
            )
        )
    ]

    print("\n=== Paired bootstrap deltas ===")
    print(
        key_bootstrap[
            [
                "candidate_model",
                "quantity",
                "population",
                "metric",
                "mean_delta_candidate_minus_reference",
                "ci95_low",
                "ci95_high",
                "probability_candidate_favorable",
                "n_paired_events",
            ]
        ].to_string(index=False)
    )

    print(f"\nPredictions: {prediction_path.resolve()}")
    print(f"Outputs    : {output_directory.resolve()}")


if __name__ == "__main__":
    main()
