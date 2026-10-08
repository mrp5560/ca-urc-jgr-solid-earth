#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
20_evaluate_locked_power_gated_test.py

Locked formal evaluation for the tail-risk gated dual-head model.

The model and calibration are fixed before test evaluation:

    checkpoint = best_overall.pt
    gate power = gamma = 3.0

Predictions:
    base:
        y_base

    linear gate diagnostic:
        y_linear = y_base + p * residual

    locked power-calibrated model:
        y_power = y_base + p^gamma * residual

The script:
    1. Loads the exact architecture from
       18_train_tail_risk_gated_dual_head.py.
    2. Reads each test-event HDF5 only once.
    3. Uses identical input/target station combinations for all models.
    4. Repeats each event 20 times by default.
    5. Computes overall, tail, non-tail, M4+, triggered, and not-triggered
       metrics.
    6. Evaluates the tail-risk gate with AUPRC, AUROC, Brier score, ECE,
       precision, recall, and F1.
    7. Performs event-level paired bootstrap against the frozen base.
    8. Exports per-target predictions and all summary tables.

No model or gate-power selection is performed in this script.

Example
-------
python 20_evaluate_locked_power_gated_test.py ^
  --training-script 18_train_tail_risk_gated_dual_head.py ^
  --checkpoint runs\tail_risk_gated_dual_head_t0_5s_k5\best_overall.pt ^
  --threshold-json runs\tail_gated_compromise_t0_5s_k5\tail_thresholds_q0.90_t0_5s.json ^
  --manifest data\scedc\model_manifests\scenario_t0_5s_k5.csv ^
  --split-name test ^
  --gate-power 3.0 ^
  --repeats 20 ^
  --bootstrap-repetitions 2000 ^
  --out-dir runs\locked_power_gated_test_gamma3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)


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


def load_python_module(
    path: str | Path,
) -> ModuleType:
    module_path = Path(path)

    if not module_path.exists():
        raise FileNotFoundError(
            f"Training script not found: {module_path}"
        )

    specification = importlib.util.spec_from_file_location(
        "tail_risk_training_module",
        module_path,
    )
    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"Could not import training script: {module_path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(module)

    required = (
        "MeanBase",
        "TailGateModel",
        "load_checkpoint",
    )
    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]
    if missing:
        raise AttributeError(
            "Training script is missing required symbols: "
            f"{missing}"
        )

    return module


def infer_architecture(
    checkpoint: dict[str, Any],
) -> tuple[int, int, float]:
    arguments = checkpoint.get("args", {})
    state = checkpoint["model_state"]

    hidden = None
    if isinstance(arguments, dict):
        hidden = arguments.get("hidden_dim")

    if hidden is None:
        for key in (
            "base.station_encoder.0.weight",
            "base.waveform_encoder.network.13.weight",
        ):
            if key in state:
                hidden = int(state[key].shape[0])
                break

    if hidden is None:
        raise ValueError(
            "Could not infer base hidden dimension."
        )

    tail_hidden = None
    if isinstance(arguments, dict):
        tail_hidden = arguments.get(
            "tail_hidden",
            arguments.get("tail_hidden_dim"),
        )

    if tail_hidden is None:
        key = "context.0.weight"
        if key in state:
            tail_hidden = int(state[key].shape[0])

    if tail_hidden is None:
        raise ValueError(
            "Could not infer tail hidden dimension."
        )

    maximum_correction = 1.5
    if isinstance(arguments, dict):
        maximum_correction = float(
            arguments.get(
                "maximum_correction",
                maximum_correction,
            )
        )

    return (
        int(hidden),
        int(tail_hidden),
        float(maximum_correction),
    )


def load_locked_model(
    module: ModuleType,
    checkpoint_path: str | Path,
    threshold_data: dict[str, Any],
    device: torch.device,
):
    checkpoint = module.load_checkpoint(
        checkpoint_path,
        device,
    )

    hidden, tail_hidden, maximum_correction = (
        infer_architecture(checkpoint)
    )

    prevalence = float(
        np.clip(
            1.0
            - float(
                threshold_data.get(
                    "quantile",
                    0.90,
                )
            ),
            0.01,
            0.50,
        )
    )

    base = module.MeanBase(
        hidden
    ).to(device)

    model = module.TailGateModel(
        base=base,
        hidden=hidden,
        tail_hidden=tail_hidden,
        prevalence=prevalence,
        maximum_correction=maximum_correction,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    return model, checkpoint, {
        "hidden_dim": hidden,
        "tail_hidden_dim": tail_hidden,
        "maximum_correction": maximum_correction,
        "prevalence_used_for_construction": prevalence,
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

    input_waveforms = (
        np.sign(input_waveforms)
        * np.log1p(
            np.abs(input_waveforms)
            / 1e-3
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

    input_features = np.concatenate(
        [
            local_xy_km(
                input_coordinates,
                origin_latitude,
                origin_longitude,
            ) / 100.0,
            input_coordinates[:, 2:3] / 2000.0,
            p_offset[input_indices, None]
            / max(float(t0_sec), 1.0),
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

    return (
        input_waveforms,
        input_features,
        target_features,
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = 10,
) -> float:
    labels = labels.astype(float)
    probabilities = probabilities.astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    result = 0.0

    for index in range(number_of_bins):
        lower = edges[index]
        upper = edges[index + 1]

        if index == number_of_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())
        if count == 0:
            continue

        result += (
            count
            / len(labels)
            * abs(
                probabilities[mask].mean()
                - labels[mask].mean()
            )
        )

    return float(result)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = labels.astype(int)
    probabilities = probabilities.astype(float)

    prevalence = float(labels.mean())

    if labels.min() == labels.max():
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )
        auprc = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "prevalence": prevalence,
        "auprc": auprc,
        "auroc": auroc,
        "brier": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "ece10": (
            expected_calibration_error(
                labels,
                probabilities,
                number_of_bins=10,
            )
        ),
        "precision_at_05": float(
            precision
        ),
        "recall_at_05": float(
            recall
        ),
        "f1_at_05": float(f1),
    }


def population_mask(
    frame: pd.DataFrame,
    quantity: str,
    population: str,
) -> np.ndarray:
    if population == "overall":
        return np.ones(
            len(frame),
            dtype=bool,
        )
    if population == "tail":
        return frame[
            f"is_tail_{quantity}"
        ].to_numpy(dtype=bool)
    if population == "non_tail":
        return ~frame[
            f"is_tail_{quantity}"
        ].to_numpy(dtype=bool)
    if population == "m4plus":
        return (
            frame["magnitude"]
            .to_numpy(dtype=float)
            >= 4.0
        )
    if population == "triggered":
        return frame[
            "target_triggered_by_snapshot"
        ].to_numpy(dtype=bool)
    if population == "not_triggered":
        return ~frame[
            "target_triggered_by_snapshot"
        ].to_numpy(dtype=bool)

    raise ValueError(
        f"Unknown population: {population}"
    )


def metric_value(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> float:
    residual = prediction - truth
    absolute = np.abs(residual)

    if metric == "mae":
        return float(absolute.mean())
    if metric == "rmse":
        return float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        )
    if metric == "bias":
        return float(residual.mean())
    if metric == "absolute_bias":
        return float(
            abs(residual.mean())
        )
    if metric == "factor2":
        return float(
            np.mean(
                absolute
                <= LOG10_FACTOR_2
            )
        )
    if metric == "factor3":
        return float(
            np.mean(
                absolute
                <= LOG10_FACTOR_3
            )
        )
    if metric == "under03":
        return float(
            np.mean(
                residual <= -0.3
            )
        )
    if metric == "under05":
        return float(
            np.mean(
                residual <= -0.5
            )
        )

    raise ValueError(
        f"Unknown metric: {metric}"
    )


def build_event_level_metrics(
    predictions: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    populations = (
        "overall",
        "tail",
        "non_tail",
        "m4plus",
        "triggered",
        "not_triggered",
    )
    metrics = (
        "mae",
        "rmse",
        "bias",
        "absolute_bias",
        "factor2",
        "factor3",
        "under03",
        "under05",
    )

    for (
        event_id,
        repeat_index,
    ), event_repeat in predictions.groupby(
        ["event_id", "repeat"],
        sort=False,
    ):
        for model_name in model_names:
            for quantity in ("pga", "pgv"):
                truth_all = event_repeat[
                    f"true_log10_{quantity}"
                ].to_numpy(dtype=float)

                prediction_all = event_repeat[
                    f"{model_name}_log10_{quantity}"
                ].to_numpy(dtype=float)

                for population in populations:
                    mask = population_mask(
                        event_repeat,
                        quantity,
                        population,
                    )
                    if not mask.any():
                        continue

                    for metric in metrics:
                        rows.append({
                            "event_id": str(
                                event_id
                            ),
                            "repeat": int(
                                repeat_index
                            ),
                            "model": model_name,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric,
                            "value": metric_value(
                                truth_all[mask],
                                prediction_all[mask],
                                metric,
                            ),
                            "n_target_rows": int(
                                mask.sum()
                            ),
                        })

    return pd.DataFrame(rows)


def summarize_metrics(
    predictions: pd.DataFrame,
    event_metrics: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    populations = (
        "overall",
        "tail",
        "non_tail",
        "m4plus",
        "triggered",
        "not_triggered",
    )
    metrics = (
        "mae",
        "rmse",
        "bias",
        "absolute_bias",
        "factor2",
        "factor3",
        "under03",
        "under05",
    )

    for model_name in model_names:
        for quantity in ("pga", "pgv"):
            truth_all = predictions[
                f"true_log10_{quantity}"
            ].to_numpy(dtype=float)

            prediction_all = predictions[
                f"{model_name}_log10_{quantity}"
            ].to_numpy(dtype=float)

            for population in populations:
                mask = population_mask(
                    predictions,
                    quantity,
                    population,
                )
                if not mask.any():
                    continue

                for metric in metrics:
                    subset = event_metrics.loc[
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
                            == metric
                        )
                    ]

                    rows.append({
                        "model": model_name,
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "target_weighted": metric_value(
                            truth_all[mask],
                            prediction_all[mask],
                            metric,
                        ),
                        "event_repeat_macro": float(
                            subset["value"].mean()
                        ),
                        "n_events": int(
                            subset["event_id"].nunique()
                        ),
                        "n_event_repeats": int(
                            len(subset)
                        ),
                        "n_target_rows": int(
                            mask.sum()
                        ),
                    })

    return pd.DataFrame(rows)


def paired_bootstrap(
    event_metrics: pd.DataFrame,
    candidate_models: list[str],
    reference_model: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    keys = (
        event_metrics[
            ["quantity", "population", "metric"]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    for quantity, population, metric in keys:
        subset = event_metrics.loc[
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
                == metric
            )
        ]

        pivot = subset.pivot_table(
            index=["event_id", "repeat"],
            columns="model",
            values="value",
            aggfunc="mean",
        )

        if reference_model not in pivot.columns:
            continue

        for candidate in candidate_models:
            if candidate not in pivot.columns:
                continue

            paired = pivot[
                [reference_model, candidate]
            ].dropna()

            if len(paired) == 0:
                continue

            delta = (
                paired[candidate]
                - paired[reference_model]
            )

            event_delta = (
                delta.groupby(level="event_id")
                .mean()
                .to_numpy(dtype=float)
            )

            bootstrap_means = np.empty(
                repetitions,
                dtype=float,
            )

            for index in range(repetitions):
                sampled = rng.integers(
                    0,
                    len(event_delta),
                    size=len(event_delta),
                )
                bootstrap_means[index] = float(
                    event_delta[
                        sampled
                    ].mean()
                )

            higher_is_better = metric in (
                "factor2",
                "factor3",
            )

            if higher_is_better:
                favorable = float(
                    np.mean(
                        bootstrap_means > 0.0
                    )
                )
                interpretation = (
                    "positive delta favors candidate"
                )
            else:
                favorable = float(
                    np.mean(
                        bootstrap_means < 0.0
                    )
                )
                interpretation = (
                    "negative delta favors candidate"
                )

            rows.append({
                "reference_model": reference_model,
                "candidate_model": candidate,
                "quantity": quantity,
                "population": population,
                "metric": metric,
                "n_paired_events": int(
                    len(event_delta)
                ),
                "mean_delta_candidate_minus_reference": float(
                    event_delta.mean()
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
                "probability_candidate_favorable": favorable,
                "interpretation": interpretation,
                "bootstrap_repetitions": int(
                    repetitions
                ),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-script",
        default=(
            "18_train_tail_risk_gated_dual_head.py"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/tail_risk_gated_dual_head_t0_5s_k5/"
            "best_overall.pt"
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
        "--gate-power",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
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
        default=(
            "runs/locked_power_gated_test_gamma3"
        ),
    )

    args = parser.parse_args()

    if args.gate_power < 1.0:
        raise ValueError(
            "--gate-power must be at least 1.0."
        )
    if args.repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )
    if args.bootstrap_repetitions < 100:
        raise ValueError(
            "--bootstrap-repetitions must be at least 100."
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

    output_directory = Path(
        args.out_dir
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_path = Path(
        args.threshold_json
    )
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold JSON not found: "
            f"{threshold_path}"
        )

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )
    thresholds = np.asarray(
        [
            float(
                threshold_data[
                    "log10_pga_threshold"
                ]
            ),
            float(
                threshold_data[
                    "log10_pgv_threshold"
                ]
            ),
        ],
        dtype=float,
    )

    module = load_python_module(
        args.training_script
    )
    model, checkpoint, architecture = (
        load_locked_model(
            module,
            args.checkpoint,
            threshold_data,
            device,
        )
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
            as_bool(
                events[eligible_column]
            )
        ].copy()

    events = events.reset_index(
        drop=True
    )

    if args.max_events is not None:
        events = events.iloc[
            :args.max_events
        ].copy()

    if len(events) == 0:
        raise ValueError(
            "No eligible evaluation events."
        )

    print(
        "=== Locked power-gated test evaluation ==="
    )
    print(f"Device             : {device}")
    print(
        f"Checkpoint         : "
        f"{Path(args.checkpoint).resolve()}"
    )
    print(
        f"Checkpoint epoch   : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        f"Gate power         : "
        f"{args.gate_power}"
    )
    print(
        f"Split              : "
        f"{args.split_column}={args.split_name}"
    )
    print(
        f"Events             : "
        f"{len(events)}"
    )
    print(
        f"Repeats/event      : "
        f"{args.repeats}"
    )
    print(
        "Tail thresholds    : "
        f"PGA={thresholds[0]:.4f}, "
        f"PGV={thresholds[1]:.4f}"
    )

    prediction_rows: list[
        dict[str, Any]
    ] = []

    for event_number, event in enumerate(
        tqdm(
            events.itertuples(index=False),
            total=len(events),
            desc="locked test evaluation",
        ),
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

        if (
            len(triggered_indices)
            < args.input_stations
        ):
            continue

        input_start_index = max(
            0,
            int(
                round(
                    (
                        pre_first_p
                        - args.input_pre_sec
                    )
                    * sampling_rate
                )
            ),
        )

        snapshot_index = min(
            time_zero_index
            + int(
                round(
                    args.t0_sec
                    * sampling_rate
                )
            ),
            acceleration.shape[-1] - 1,
        )

        magnitude = pd.to_numeric(
            getattr(
                event,
                "magnitude",
                np.nan,
            ),
            errors="coerce",
        )
        magnitude = (
            float(magnitude)
            if np.isfinite(magnitude)
            else np.nan
        )

        sequence_group = pd.to_numeric(
            getattr(
                event,
                "sequence_group",
                -1,
            ),
            errors="coerce",
        )
        sequence_group = (
            int(sequence_group)
            if np.isfinite(sequence_group)
            else -1
        )

        for repeat_index in range(
            args.repeats
        ):
            rng = np.random.default_rng(
                stable_seed(
                    (
                        f"locked:{args.split_name}:"
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
                np.arange(
                    len(coordinates)
                ),
                input_indices,
                assume_unique=False,
            )

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

            true_pga = np.log10(
                np.maximum(
                    horizontal_peak(
                        future_acceleration
                    ),
                    1e-10,
                )
            )
            true_pgv = np.log10(
                np.maximum(
                    horizontal_peak(
                        future_velocity
                    ),
                    1e-12,
                )
            )

            with torch.inference_mode():
                outputs = model(
                    torch.from_numpy(
                        input_waveforms
                    )[None].to(device),
                    torch.from_numpy(
                        input_features
                    )[None].to(device),
                    torch.from_numpy(
                        target_features
                    )[None].to(device),
                )

                base_prediction = outputs[
                    "base"
                ][0]
                probability = outputs[
                    "probability"
                ][0]
                residual = outputs[
                    "residual"
                ][0]

                linear_prediction = (
                    base_prediction
                    + probability
                    * residual
                )
                power_prediction = (
                    base_prediction
                    + probability.pow(
                        args.gate_power
                    )
                    * residual
                )

                base_array = (
                    base_prediction
                    .cpu()
                    .numpy()
                )
                probability_array = (
                    probability
                    .cpu()
                    .numpy()
                )
                residual_array = (
                    residual
                    .cpu()
                    .numpy()
                )
                linear_array = (
                    linear_prediction
                    .cpu()
                    .numpy()
                )
                power_array = (
                    power_prediction
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
            ) in enumerate(
                target_indices
            ):
                target_triggered = bool(
                    np.isfinite(
                        p_offset[
                            station_index
                        ]
                    )
                    and (
                        p_offset[
                            station_index
                        ]
                        <= args.t0_sec
                    )
                )

                prediction_rows.append({
                    "event_id": event_id,
                    "repeat": int(
                        repeat_index
                    ),
                    "sequence_group": (
                        sequence_group
                    ),
                    "magnitude": magnitude,
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
                    "target_triggered_by_snapshot": (
                        target_triggered
                    ),
                    "true_log10_pga": float(
                        true_pga[
                            local_target_index
                        ]
                    ),
                    "true_log10_pgv": float(
                        true_pgv[
                            local_target_index
                        ]
                    ),
                    "is_tail_pga": bool(
                        true_pga[
                            local_target_index
                        ]
                        >= thresholds[0]
                    ),
                    "is_tail_pgv": bool(
                        true_pgv[
                            local_target_index
                        ]
                        >= thresholds[1]
                    ),
                    "tail_probability_pga": float(
                        probability_array[
                            local_target_index,
                            0,
                        ]
                    ),
                    "tail_probability_pgv": float(
                        probability_array[
                            local_target_index,
                            1,
                        ]
                    ),
                    "residual_correction_pga": float(
                        residual_array[
                            local_target_index,
                            0,
                        ]
                    ),
                    "residual_correction_pgv": float(
                        residual_array[
                            local_target_index,
                            1,
                        ]
                    ),
                    "linear_applied_correction_pga": float(
                        linear_array[
                            local_target_index,
                            0,
                        ]
                        - base_array[
                            local_target_index,
                            0,
                        ]
                    ),
                    "linear_applied_correction_pgv": float(
                        linear_array[
                            local_target_index,
                            1,
                        ]
                        - base_array[
                            local_target_index,
                            1,
                        ]
                    ),
                    "power_applied_correction_pga": float(
                        power_array[
                            local_target_index,
                            0,
                        ]
                        - base_array[
                            local_target_index,
                            0,
                        ]
                    ),
                    "power_applied_correction_pgv": float(
                        power_array[
                            local_target_index,
                            1,
                        ]
                        - base_array[
                            local_target_index,
                            1,
                        ]
                    ),
                    "base_log10_pga": float(
                        base_array[
                            local_target_index,
                            0,
                        ]
                    ),
                    "base_log10_pgv": float(
                        base_array[
                            local_target_index,
                            1,
                        ]
                    ),
                    "linear_gate_log10_pga": float(
                        linear_array[
                            local_target_index,
                            0,
                        ]
                    ),
                    "linear_gate_log10_pgv": float(
                        linear_array[
                            local_target_index,
                            1,
                        ]
                    ),
                    "power_gate_log10_pga": float(
                        power_array[
                            local_target_index,
                            0,
                        ]
                    ),
                    "power_gate_log10_pgv": float(
                        power_array[
                            local_target_index,
                            1,
                        ]
                    ),
                })

        if (
            event_number
            % max(
                1,
                args.checkpoint_every_events,
            )
            == 0
            or event_number
            == len(events)
        ):
            pd.DataFrame(
                prediction_rows
            ).to_csv(
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

    prediction_path = (
        output_directory
        / "locked_test_predictions.csv"
    )
    predictions.to_csv(
        prediction_path,
        index=False,
    )

    model_names = [
        "base",
        "linear_gate",
        "power_gate",
    ]

    event_metrics = (
        build_event_level_metrics(
            predictions,
            model_names,
        )
    )
    event_metrics.to_csv(
        output_directory
        / "event_level_metrics.csv",
        index=False,
    )

    summary = summarize_metrics(
        predictions,
        event_metrics,
        model_names,
    )
    summary.to_csv(
        output_directory
        / "model_metrics_summary.csv",
        index=False,
    )

    gate_rows: list[
        dict[str, Any]
    ] = []

    for quantity in ("pga", "pgv"):
        labels = predictions[
            f"is_tail_{quantity}"
        ].to_numpy(dtype=int)

        probabilities = predictions[
            f"tail_probability_{quantity}"
        ].to_numpy(dtype=float)

        metrics = classification_metrics(
            labels,
            probabilities,
        )

        gate_rows.append({
            "quantity": quantity,
            "n_rows": int(
                len(labels)
            ),
            "n_positive": int(
                labels.sum()
            ),
            **metrics,
        })

    gate_metrics = pd.DataFrame(
        gate_rows
    )
    gate_metrics.to_csv(
        output_directory
        / "gate_classification_metrics.csv",
        index=False,
    )

    bootstrap = paired_bootstrap(
        event_metrics=event_metrics,
        candidate_models=[
            "linear_gate",
            "power_gate",
        ],
        reference_model="base",
        repetitions=(
            args.bootstrap_repetitions
        ),
        seed=args.seed,
    )
    bootstrap.to_csv(
        output_directory
        / "paired_bootstrap_deltas.csv",
        index=False,
    )

    metadata = {
        "arguments": vars(args),
        "resolved_device": str(
            device
        ),
        "checkpoint_epoch": (
            checkpoint.get(
                "epoch",
                None,
            )
        ),
        "architecture": architecture,
        "tail_thresholds": (
            threshold_data
        ),
        "n_events": int(
            predictions[
                "event_id"
            ].nunique()
        ),
        "n_event_repeats": int(
            predictions[
                [
                    "event_id",
                    "repeat",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "n_target_rows": int(
            len(predictions)
        ),
        "locked_before_test": {
            "checkpoint": str(
                Path(
                    args.checkpoint
                ).resolve()
            ),
            "gate_power": float(
                args.gate_power
            ),
            "selection_source": (
                "validation gate-power scan"
            ),
        },
    }

    (
        output_directory
        / "evaluation_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    key_summary = summary.loc[
        (
            summary["population"].isin(
                [
                    "overall",
                    "tail",
                    "non_tail",
                    "m4plus",
                    "triggered",
                ]
            )
        )
        & (
            summary["metric"].isin(
                [
                    "mae",
                    "bias",
                    "under05",
                    "factor2",
                ]
            )
        )
    ]

    print(
        "\n=== Locked test metrics ==="
    )
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
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Gate classification ==="
    )
    print(
        gate_metrics.to_string(
            index=False
        )
    )

    key_bootstrap = bootstrap.loc[
        (
            bootstrap["candidate_model"]
            == "power_gate"
        )
        & (
            bootstrap["population"].isin(
                [
                    "overall",
                    "tail",
                    "non_tail",
                    "m4plus",
                    "triggered",
                ]
            )
        )
        & (
            bootstrap["metric"].isin(
                [
                    "mae",
                    "absolute_bias",
                    "under05",
                    "factor2",
                ]
            )
        )
    ]

    print(
        "\n=== Power-gate paired bootstrap vs base ==="
    )
    print(
        key_bootstrap[
            [
                "quantity",
                "population",
                "metric",
                "mean_delta_candidate_minus_reference",
                "ci95_low",
                "ci95_high",
                "probability_candidate_favorable",
                "n_paired_events",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nPredictions: "
        f"{prediction_path.resolve()}"
    )
    print(
        f"Outputs    : "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
