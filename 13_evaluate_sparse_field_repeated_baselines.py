#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
13_evaluate_sparse_field_repeated_baselines.py

Repeated, event-clustered evaluation for the deterministic sparse-to-field
PGA/PGV model.

This script:

1. Loads the best checkpoint from
   12_train_sparse_field_pga_pgv_baseline.py.
2. Repeats input/target station sampling for every test event.
3. Exports one row per target station prediction.
4. Compares the neural model with causal baselines:
      - training-target median
      - median observed input motion
      - nearest observed input station
      - inverse-distance weighting of observed input stations
5. Reports target-weighted and event-macro metrics.
6. Computes event-cluster bootstrap 95% confidence intervals.
7. Reports performance by magnitude, target distance, and whether the target
   station had triggered by the snapshot.

Important causal rule
---------------------
The nearest-station and IDW baselines use only motion observed before the
snapshot time. They do not use the future peak motion at input stations.

Default experiment
------------------
T0 = 5 s, K = 5 input stations, 10 target stations, 20 repeats per event.

Example
-------
python 13_evaluate_sparse_field_repeated_baselines.py ^
  --manifest data\scedc\model_manifests\scenario_t0_5s_k5.csv ^
  --checkpoint runs\sparse_field_pga_pgv_t0_5s_k5\best_model.pt ^
  --split-name test ^
  --repeats 20 ^
  --out-dir runs\sparse_field_pga_pgv_t0_5s_k5\repeated_test
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
LOG10_FACTOR_3 = math.log10(3.0)


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{text}:{base_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def local_xy_km(
    coords: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    latitude = coords[:, 0]
    longitude = coords[:, 1]

    x = (
        (longitude - origin_lon)
        * 111.32
        * math.cos(math.radians(origin_lat))
    )
    y = (latitude - origin_lat) * 110.57
    return np.stack([x, y], axis=-1)


def pairwise_distance_km(
    target_xy: np.ndarray,
    input_xy: np.ndarray,
) -> np.ndarray:
    difference = (
        target_xy[:, None, :]
        - input_xy[None, :, :]
    )
    return np.sqrt(
        np.sum(difference ** 2, axis=-1)
    )


def horizontal_peak(data: np.ndarray) -> np.ndarray:
    """
    data: [N, 3, T]
    returns horizontal vector peak [N]
    """
    if data.shape[-1] == 0:
        raise ValueError("Empty waveform window.")
    return np.max(
        np.sqrt(
            data[:, 0, :] ** 2
            + data[:, 1, :] ** 2
        ),
        axis=1,
    )


class WaveformEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SparseFieldBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 128):
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
        input_waveforms: torch.Tensor,
        input_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, stations, channels, samples = (
            input_waveforms.shape
        )

        encoded = self.waveform_encoder(
            input_waveforms.reshape(
                batch * stations,
                channels,
                samples,
            )
        ).reshape(batch, stations, -1)

        station_latent = self.station_encoder(
            torch.cat(
                [encoded, input_features],
                dim=-1,
            )
        )
        attention_weights = torch.softmax(
            self.attention_score(station_latent),
            dim=1,
        )
        event_latent = torch.sum(
            attention_weights * station_latent,
            dim=1,
        )

        query_count = target_features.shape[1]
        expanded_event = event_latent[:, None, :].expand(
            -1,
            query_count,
            -1,
        )

        return self.query_decoder(
            torch.cat(
                [expanded_event, target_features],
                dim=-1,
            )
        )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SparseFieldBaseline, dict[str, Any]]:
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

    checkpoint_args = dict(checkpoint.get("args", {}))
    hidden_dim = int(
        checkpoint_args.get("hidden_dim", 128)
    )

    model = SparseFieldBaseline(
        hidden_dim=hidden_dim
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint_args


def future_log_targets(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    snapshot_index: int,
    station_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if station_indices is None:
        station_indices = np.arange(
            acceleration.shape[0]
        )

    future_acceleration = acceleration[
        station_indices,
        :,
        snapshot_index:,
    ]
    future_velocity = velocity[
        station_indices,
        :,
        snapshot_index:,
    ]

    pga = horizontal_peak(future_acceleration)
    pgv = horizontal_peak(future_velocity)

    return (
        np.log10(np.maximum(pga, 1e-10)),
        np.log10(np.maximum(pgv, 1e-12)),
    )


def compute_training_target_medians(
    manifest: pd.DataFrame,
    split_column: str,
    t0_sec: int,
    cache_path: Path,
) -> dict[str, float]:
    if cache_path.exists():
        return json.loads(
            cache_path.read_text(encoding="utf-8")
        )

    train_events = manifest.loc[
        manifest[split_column]
        .astype(str)
        .eq("train")
    ]

    all_pga_log = []
    all_pgv_log = []

    for row in tqdm(
        train_events.itertuples(index=False),
        total=len(train_events),
        desc="compute train target medians",
    ):
        h5_path = Path(str(row.h5_path))

        with h5py.File(h5_path, "r") as h5:
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

        pga_log, pgv_log = future_log_targets(
            acceleration,
            velocity,
            snapshot_index,
        )
        all_pga_log.append(pga_log)
        all_pgv_log.append(pgv_log)

    result = {
        "train_median_log10_pga": float(
            np.median(
                np.concatenate(all_pga_log)
            )
        ),
        "train_median_log10_pgv": float(
            np.median(
                np.concatenate(all_pgv_log)
            )
        ),
        "n_training_station_targets": int(
            sum(len(x) for x in all_pga_log)
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


def prepare_model_inputs(
    acceleration: np.ndarray,
    coordinates: np.ndarray,
    p_offset: np.ndarray,
    input_indices: np.ndarray,
    target_indices: np.ndarray,
    input_start: int,
    snapshot_index: int,
    t0_sec: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    input_waveforms = acceleration[
        input_indices,
        :,
        input_start:snapshot_index,
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

    return (
        input_waveforms,
        input_features,
        target_features,
        input_xy,
        target_xy,
    )


def causal_spatial_baselines(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    input_indices: np.ndarray,
    input_xy: np.ndarray,
    target_xy: np.ndarray,
    input_start: int,
    snapshot_index: int,
    idw_power: float,
) -> dict[str, np.ndarray]:
    observed_acceleration = acceleration[
        input_indices,
        :,
        input_start:snapshot_index,
    ]
    observed_velocity = velocity[
        input_indices,
        :,
        input_start:snapshot_index,
    ]

    observed_pga = np.maximum(
        horizontal_peak(observed_acceleration),
        1e-10,
    )
    observed_pgv = np.maximum(
        horizontal_peak(observed_velocity),
        1e-12,
    )

    distances = pairwise_distance_km(
        target_xy,
        input_xy,
    )
    nearest_input = np.argmin(
        distances,
        axis=1,
    )

    nearest_pga = observed_pga[nearest_input]
    nearest_pgv = observed_pgv[nearest_input]

    weights = 1.0 / np.maximum(
        distances,
        1e-3,
    ) ** idw_power
    weights /= np.sum(
        weights,
        axis=1,
        keepdims=True,
    )

    idw_pga = weights @ observed_pga
    idw_pgv = weights @ observed_pgv

    median_pga = np.full(
        len(target_xy),
        np.median(observed_pga),
        dtype=np.float64,
    )
    median_pgv = np.full(
        len(target_xy),
        np.median(observed_pgv),
        dtype=np.float64,
    )

    return {
        "nearest_distance_km": np.min(
            distances,
            axis=1,
        ),
        "nearest_observed_log10_pga": np.log10(
            np.maximum(nearest_pga, 1e-10)
        ),
        "nearest_observed_log10_pgv": np.log10(
            np.maximum(nearest_pgv, 1e-12)
        ),
        "idw_observed_log10_pga": np.log10(
            np.maximum(idw_pga, 1e-10)
        ),
        "idw_observed_log10_pgv": np.log10(
            np.maximum(idw_pgv, 1e-12)
        ),
        "median_observed_log10_pga": np.log10(
            np.maximum(median_pga, 1e-10)
        ),
        "median_observed_log10_pgv": np.log10(
            np.maximum(median_pgv, 1e-12)
        ),
    }


def correlation(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    if (
        len(true_values) < 2
        or np.std(true_values) == 0
        or np.std(predicted_values) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            true_values,
            predicted_values,
        )[0, 1]
    )


def spearman_correlation(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    true_ranks = pd.Series(
        true_values
    ).rank(method="average").to_numpy()
    predicted_ranks = pd.Series(
        predicted_values
    ).rank(method="average").to_numpy()

    return correlation(
        true_ranks,
        predicted_ranks,
    )


def metric_record(
    frame: pd.DataFrame,
    method: str,
    quantity: str,
    grouping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    true_column = f"true_log10_{quantity}"
    prediction_column = (
        f"{method}_log10_{quantity}"
    )

    true_values = frame[
        true_column
    ].to_numpy(dtype=float)
    predicted_values = frame[
        prediction_column
    ].to_numpy(dtype=float)

    residual = predicted_values - true_values
    absolute_error = np.abs(residual)

    event_mae = (
        pd.DataFrame({
            "event_id": frame["event_id"].astype(str),
            "absolute_error": absolute_error,
        })
        .groupby("event_id")["absolute_error"]
        .mean()
    )

    result = {
        "method": method,
        "quantity": quantity,
        "n_target_rows": int(len(frame)),
        "n_events": int(
            frame["event_id"].nunique()
        ),
        "mae_log10_target_weighted": float(
            np.mean(absolute_error)
        ),
        "mae_log10_event_macro": float(
            event_mae.mean()
        ),
        "median_absolute_error_log10": float(
            np.median(absolute_error)
        ),
        "rmse_log10": float(
            np.sqrt(np.mean(residual ** 2))
        ),
        "bias_log10_prediction_minus_true": float(
            np.mean(residual)
        ),
        "pearson": correlation(
            true_values,
            predicted_values,
        ),
        "spearman": spearman_correlation(
            true_values,
            predicted_values,
        ),
        "factor2_accuracy": float(
            np.mean(
                absolute_error <= LOG10_FACTOR_2
            )
        ),
        "factor3_accuracy": float(
            np.mean(
                absolute_error <= LOG10_FACTOR_3
            )
        ),
        "equivalent_multiplicative_mae": float(
            10.0 ** np.mean(absolute_error)
        ),
    }

    if grouping:
        result.update(grouping)

    return result


def bootstrap_event_macro_metrics(
    frame: pd.DataFrame,
    methods: list[str],
    quantities: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output_rows = []

    event_ids = np.asarray(
        sorted(
            frame["event_id"]
            .astype(str)
            .unique()
        )
    )

    for method in methods:
        for quantity in quantities:
            true_column = f"true_log10_{quantity}"
            prediction_column = (
                f"{method}_log10_{quantity}"
            )

            working = frame[
                [
                    "event_id",
                    true_column,
                    prediction_column,
                ]
            ].copy()

            residual = (
                working[prediction_column]
                - working[true_column]
            )
            working["absolute_error"] = np.abs(
                residual
            )
            working["bias"] = residual
            working["factor2"] = (
                working["absolute_error"]
                <= LOG10_FACTOR_2
            ).astype(float)

            event_statistics = (
                working.groupby("event_id")
                .agg(
                    mae=("absolute_error", "mean"),
                    bias=("bias", "mean"),
                    factor2=("factor2", "mean"),
                )
                .reindex(event_ids)
            )

            samples_mae = np.empty(
                repetitions,
                dtype=float,
            )
            samples_bias = np.empty(
                repetitions,
                dtype=float,
            )
            samples_factor2 = np.empty(
                repetitions,
                dtype=float,
            )

            values = event_statistics.to_numpy(
                dtype=float
            )
            number_of_events = len(values)

            for bootstrap_index in range(
                repetitions
            ):
                selected = rng.integers(
                    0,
                    number_of_events,
                    size=number_of_events,
                )
                sampled = values[selected]
                samples_mae[bootstrap_index] = (
                    np.mean(sampled[:, 0])
                )
                samples_bias[bootstrap_index] = (
                    np.mean(sampled[:, 1])
                )
                samples_factor2[
                    bootstrap_index
                ] = np.mean(sampled[:, 2])

            for metric_name, values_array in [
                (
                    "mae_log10_event_macro",
                    samples_mae,
                ),
                (
                    "bias_log10_event_macro",
                    samples_bias,
                ),
                (
                    "factor2_accuracy_event_macro",
                    samples_factor2,
                ),
            ]:
                output_rows.append({
                    "method": method,
                    "quantity": quantity,
                    "metric": metric_name,
                    "bootstrap_repetitions": repetitions,
                    "estimate": float(
                        np.mean(values_array)
                    ),
                    "ci95_low": float(
                        np.percentile(
                            values_array,
                            2.5,
                        )
                    ),
                    "ci95_high": float(
                        np.percentile(
                            values_array,
                            97.5,
                        )
                    ),
                })

    return pd.DataFrame(output_rows)


def add_group_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["magnitude_bin"] = pd.cut(
        result["magnitude"],
        bins=[
            -np.inf,
            3.3,
            3.6,
            4.0,
            5.0,
            6.0,
            np.inf,
        ],
        labels=[
            "M3.0-3.2",
            "M3.3-3.5",
            "M3.6-3.9",
            "M4.0-4.9",
            "M5.0-5.9",
            "M6+",
        ],
        right=False,
    ).astype(str)

    result["nearest_input_distance_bin"] = pd.cut(
        result["nearest_input_distance_km"],
        bins=[
            -np.inf,
            25.0,
            50.0,
            100.0,
            150.0,
            np.inf,
        ],
        labels=[
            "<25 km",
            "25-50 km",
            "50-100 km",
            "100-150 km",
            ">=150 km",
        ],
        right=False,
    ).astype(str)

    result["target_trigger_state"] = np.where(
        result["target_triggered_by_snapshot"],
        "triggered_by_snapshot",
        "not_triggered_by_snapshot",
    )

    return result


def grouped_metric_table(
    frame: pd.DataFrame,
    methods: list[str],
    quantities: list[str],
) -> pd.DataFrame:
    output = []

    grouping_specs = [
        ("magnitude_bin",),
        ("nearest_input_distance_bin",),
        ("target_trigger_state",),
    ]

    for grouping_columns in grouping_specs:
        grouped = frame.groupby(
            list(grouping_columns),
            dropna=False,
        )

        for group_key, group_frame in grouped:
            if not isinstance(
                group_key,
                tuple,
            ):
                group_key = (group_key,)

            grouping_values = {
                "grouping_variable": "|".join(
                    grouping_columns
                ),
                "grouping_value": "|".join(
                    str(value)
                    for value in group_key
                ),
            }

            for method in methods:
                for quantity in quantities:
                    output.append(
                        metric_record(
                            group_frame,
                            method,
                            quantity,
                            grouping=grouping_values,
                        )
                    )

    return pd.DataFrame(output)


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
        "--checkpoint",
        default=(
            "runs/sparse_field_pga_pgv_t0_5s_k5/"
            "best_model.pt"
        ),
    )
    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )
    parser.add_argument(
        "--split-name",
        default="test",
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
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=1000,
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
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "runs/sparse_field_pga_pgv_t0_5s_k5/"
            "repeated_test"
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    checkpoint_path = Path(args.checkpoint)
    output_directory = Path(args.out_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        manifest_path,
        dtype={"event_id": str},
    )
    evaluation_events = manifest.loc[
        manifest[args.split_column]
        .astype(str)
        .eq(args.split_name)
    ].reset_index(drop=True)

    eligible_column = (
        f"eligible_t0_{args.t0_sec}s_"
        f"k{args.input_stations}"
    )
    if eligible_column in evaluation_events.columns:
        evaluation_events = evaluation_events.loc[
            evaluation_events[
                eligible_column
            ]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        ].reset_index(drop=True)

    if args.max_events is not None:
        evaluation_events = evaluation_events.iloc[
            :args.max_events
        ].copy()

    if len(evaluation_events) == 0:
        raise ValueError(
            "No eligible evaluation events."
        )
    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
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
        device = torch.device(args.device)

    model, checkpoint_args = load_checkpoint(
        checkpoint_path,
        device,
    )

    train_median_cache = (
        output_directory
        / f"train_target_medians_t0_{args.t0_sec}s.json"
    )
    train_medians = compute_training_target_medians(
        manifest,
        split_column=args.split_column,
        t0_sec=args.t0_sec,
        cache_path=train_median_cache,
    )

    print(
        "=== Repeated sparse-field evaluation ==="
    )
    print(
        f"Manifest            : "
        f"{manifest_path.resolve()}"
    )
    print(
        f"Checkpoint          : "
        f"{checkpoint_path.resolve()}"
    )
    print(f"Device              : {device}")
    print(
        f"Evaluation split    : "
        f"{args.split_column}={args.split_name}"
    )
    print(
        f"Events              : "
        f"{len(evaluation_events)}"
    )
    print(f"Repeats/event       : {args.repeats}")
    print(
        f"Scenario            : "
        f"T0={args.t0_sec}s, "
        f"K={args.input_stations}, "
        f"targets={args.target_stations or 'all'}"
    )
    print(
        "Train target median : "
        f"PGA={train_medians['train_median_log10_pga']:.4f}, "
        f"PGV={train_medians['train_median_log10_pgv']:.4f}"
    )

    prediction_rows: list[dict[str, Any]] = []

    for event_number, event_row in enumerate(
        tqdm(
            evaluation_events.itertuples(
                index=False
            ),
            total=len(evaluation_events),
            desc="evaluate events",
        ),
        start=1,
    ):
        event_id = str(event_row.event_id)
        h5_path = Path(str(event_row.h5_path))

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
            channel_families = h5[
                "channel_family"
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
            & (p_offset <= args.t0_sec)
        )
        if (
            len(triggered_indices)
            < args.input_stations
        ):
            continue

        input_start = int(
            round(
                (
                    pre_first_p
                    - args.input_pre_sec
                )
                * sampling_rate
            )
        )
        input_start = max(0, input_start)

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

        for repeat_index in range(
            args.repeats
        ):
            rng = np.random.default_rng(
                stable_seed(
                    f"{event_id}:repeat:{repeat_index}",
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
                if (
                    len(target_pool)
                    < args.target_stations
                ):
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
                input_xy,
                target_xy,
            ) = prepare_model_inputs(
                acceleration=acceleration,
                coordinates=coordinates,
                p_offset=p_offset,
                input_indices=input_indices,
                target_indices=target_indices,
                input_start=input_start,
                snapshot_index=snapshot_index,
                t0_sec=args.t0_sec,
            )

            with torch.inference_mode():
                model_prediction = model(
                    torch.from_numpy(
                        input_waveforms
                    )[None].to(device),
                    torch.from_numpy(
                        input_features
                    )[None].to(device),
                    torch.from_numpy(
                        target_features
                    )[None].to(device),
                )[0].cpu().numpy()

            true_pga_log, true_pgv_log = (
                future_log_targets(
                    acceleration,
                    velocity,
                    snapshot_index,
                    target_indices,
                )
            )
            baseline = causal_spatial_baselines(
                acceleration=acceleration,
                velocity=velocity,
                input_indices=input_indices,
                input_xy=input_xy,
                target_xy=target_xy,
                input_start=input_start,
                snapshot_index=snapshot_index,
                idw_power=args.idw_power,
            )

            input_station_text = "|".join(
                str(station_ids[index])
                for index in input_indices
            )

            event_metadata = {
                "magnitude": float(
                    getattr(
                        event_row,
                        "magnitude",
                        np.nan,
                    )
                ),
                "year": int(
                    getattr(
                        event_row,
                        "year",
                        -1,
                    )
                ),
                "sequence_group": int(
                    getattr(
                        event_row,
                        "sequence_group",
                        -1,
                    )
                ),
            }

            for local_target_index, station_index in enumerate(
                target_indices
            ):
                prediction_rows.append({
                    "event_id": event_id,
                    "repeat": repeat_index,
                    "target_station_index": int(
                        station_index
                    ),
                    "target_station_id": str(
                        station_ids[station_index]
                    ),
                    "target_channel_family": str(
                        channel_families[
                            station_index
                        ]
                    ),
                    "input_station_ids": input_station_text,
                    "t0_sec": args.t0_sec,
                    "input_station_count": args.input_stations,
                    "target_station_count_per_repeat": int(
                        len(target_indices)
                    ),
                    "target_latitude": float(
                        coordinates[
                            station_index,
                            0,
                        ]
                    ),
                    "target_longitude": float(
                        coordinates[
                            station_index,
                            1,
                        ]
                    ),
                    "target_elevation_m": float(
                        coordinates[
                            station_index,
                            2,
                        ]
                    ),
                    "target_p_offset_sec": float(
                        p_offset[station_index]
                    )
                    if np.isfinite(
                        p_offset[station_index]
                    )
                    else np.nan,
                    "target_triggered_by_snapshot": bool(
                        np.isfinite(
                            p_offset[station_index]
                        )
                        and p_offset[station_index]
                        <= args.t0_sec
                    ),
                    "nearest_input_distance_km": float(
                        baseline[
                            "nearest_distance_km"
                        ][local_target_index]
                    ),
                    "true_log10_pga": float(
                        true_pga_log[
                            local_target_index
                        ]
                    ),
                    "true_log10_pgv": float(
                        true_pgv_log[
                            local_target_index
                        ]
                    ),
                    "model_log10_pga": float(
                        model_prediction[
                            local_target_index,
                            0,
                        ]
                    ),
                    "model_log10_pgv": float(
                        model_prediction[
                            local_target_index,
                            1,
                        ]
                    ),
                    "train_median_log10_pga": float(
                        train_medians[
                            "train_median_log10_pga"
                        ]
                    ),
                    "train_median_log10_pgv": float(
                        train_medians[
                            "train_median_log10_pgv"
                        ]
                    ),
                    "median_observed_log10_pga": float(
                        baseline[
                            "median_observed_log10_pga"
                        ][local_target_index]
                    ),
                    "median_observed_log10_pgv": float(
                        baseline[
                            "median_observed_log10_pgv"
                        ][local_target_index]
                    ),
                    "nearest_observed_log10_pga": float(
                        baseline[
                            "nearest_observed_log10_pga"
                        ][local_target_index]
                    ),
                    "nearest_observed_log10_pgv": float(
                        baseline[
                            "nearest_observed_log10_pgv"
                        ][local_target_index]
                    ),
                    "idw_observed_log10_pga": float(
                        baseline[
                            "idw_observed_log10_pga"
                        ][local_target_index]
                    ),
                    "idw_observed_log10_pgv": float(
                        baseline[
                            "idw_observed_log10_pgv"
                        ][local_target_index]
                    ),
                    **event_metadata,
                })

        if (
            event_number
            % max(
                1,
                args.checkpoint_every_events,
            )
            == 0
            or event_number
            == len(evaluation_events)
        ):
            partial = pd.DataFrame(
                prediction_rows
            )
            partial.to_csv(
                output_directory
                / "predictions_partial.csv",
                index=False,
            )

    predictions = pd.DataFrame(
        prediction_rows
    )
    if len(predictions) == 0:
        raise RuntimeError(
            "No prediction rows were generated."
        )

    predictions["true_pga_mps2"] = (
        10.0
        ** predictions["true_log10_pga"]
    )
    predictions["true_pga_g"] = (
        predictions["true_pga_mps2"]
        / 9.80665
    )
    predictions["true_pgv_mps"] = (
        10.0
        ** predictions["true_log10_pgv"]
    )
    predictions["true_pgv_cms"] = (
        predictions["true_pgv_mps"]
        * 100.0
    )
    predictions["model_pga_mps2"] = (
        10.0
        ** predictions["model_log10_pga"]
    )
    predictions["model_pgv_mps"] = (
        10.0
        ** predictions["model_log10_pgv"]
    )

    predictions = add_group_columns(
        predictions
    )

    prediction_path = (
        output_directory
        / "target_predictions.csv"
    )
    predictions.to_csv(
        prediction_path,
        index=False,
    )

    methods = [
        "model",
        "train_median",
        "median_observed",
        "nearest_observed",
        "idw_observed",
    ]
    quantities = ["pga", "pgv"]

    summary_rows = []
    for method in methods:
        for quantity in quantities:
            summary_rows.append(
                metric_record(
                    predictions,
                    method,
                    quantity,
                )
            )

    metrics_summary = pd.DataFrame(
        summary_rows
    )
    metrics_summary.to_csv(
        output_directory
        / "metrics_summary.csv",
        index=False,
    )

    bootstrap = bootstrap_event_macro_metrics(
        predictions,
        methods=methods,
        quantities=quantities,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    bootstrap.to_csv(
        output_directory
        / "bootstrap_event_ci.csv",
        index=False,
    )

    grouped_metrics = grouped_metric_table(
        predictions,
        methods=methods,
        quantities=quantities,
    )
    grouped_metrics.to_csv(
        output_directory
        / "grouped_metrics.csv",
        index=False,
    )

    event_metrics_rows = []
    for (
        event_id,
        repeat_index,
    ), event_repeat in predictions.groupby(
        ["event_id", "repeat"]
    ):
        for method in methods:
            for quantity in quantities:
                record = metric_record(
                    event_repeat,
                    method,
                    quantity,
                )
                record.update({
                    "event_id": event_id,
                    "repeat": int(
                        repeat_index
                    ),
                    "magnitude": float(
                        event_repeat[
                            "magnitude"
                        ].iloc[0]
                    ),
                    "sequence_group": int(
                        event_repeat[
                            "sequence_group"
                        ].iloc[0]
                    ),
                })
                event_metrics_rows.append(record)

    pd.DataFrame(
        event_metrics_rows
    ).to_csv(
        output_directory
        / "event_repeat_metrics.csv",
        index=False,
    )

    run_metadata = {
        "arguments": vars(args),
        "checkpoint_arguments": checkpoint_args,
        "device": str(device),
        "n_evaluation_events": int(
            predictions["event_id"].nunique()
        ),
        "n_target_prediction_rows": int(
            len(predictions)
        ),
        "methods": methods,
        "train_target_medians": train_medians,
    }
    (
        output_directory
        / "evaluation_metadata.json"
    ).write_text(
        json.dumps(
            run_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "method",
        "quantity",
        "mae_log10_target_weighted",
        "mae_log10_event_macro",
        "rmse_log10",
        "bias_log10_prediction_minus_true",
        "factor2_accuracy",
        "pearson",
        "spearman",
        "equivalent_multiplicative_mae",
    ]

    print(
        "\n=== Repeated evaluation summary ==="
    )
    print(
        metrics_summary[
            display_columns
        ].to_string(index=False)
    )

    print(
        "\n=== Event-cluster bootstrap 95% CI ==="
    )
    print(
        bootstrap.loc[
            bootstrap["metric"]
            == "mae_log10_event_macro"
        ][
            [
                "method",
                "quantity",
                "estimate",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nPredictions : "
        f"{prediction_path.resolve()}"
    )
    print(
        f"Outputs     : "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
