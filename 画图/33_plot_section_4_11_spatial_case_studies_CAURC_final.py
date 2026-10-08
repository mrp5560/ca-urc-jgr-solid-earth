#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
33_plot_section_4_11_spatial_case_studies_CAURC_final.py

Section 4.11
Spatial case studies for the FINAL Causal-SeisField + CA-URC model.

Final manuscript model
----------------------
Cross-Attention Base checkpoint:
    runs/final_strong_baselines_reuse_locked/
    cross_attention/best_model.pt

CA-URC A4 under-only head:
    runs/cadrg_gate_ablation_A3_A5/
    A4_under_only/selected_head.pt

Locked selection:
    variant = A4_under_only
    epoch   = 3
    gamma   = 5

Final prediction:
    y_CA-URC = y_CA + p_under**gamma * Delta

Final locked grouped-test predictions:
    runs/cadrg_gate_ablation_A3_A5/
    locked_test_ablation_predictions.csv

Grouped test manifest:
    data/model_manifests/scenario_t0_5s_k5_linux.csv

TRAIN-only high-motion thresholds:
    runs/tail_gated_compromise_t0_5s_k5/
    tail_thresholds_q0.90_t0_5s.json

HDF5 root:
    data/processed_full_v4/events

Model source files:
    45_phase2_strong_baseline_suite.py
    63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py

Scientific interpretation
-------------------------
1. Representative and high-motion cases are selected from the final locked
   grouped-test target predictions.
2. The exact deterministic five-input / ten-target locked test draw is
   reproduced and audited.
3. The frozen final model is queried at all remaining non-input stations only
   for descriptive spatial visualisation.
4. Triangulated coloured surfaces are visual guides through actual held-out
   station queries; they are not dense model outputs or continuous ground truth.
5. Quantitative metrics remain station based.

Example
-------
Run from the project root, e.g. /mnt/世界模型:

python 33_plot_section_4_11_spatial_case_studies_CAURC_final.py \
  --out-dir figures/section_4_11_caurc \
  --dpi 600
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------
# FINAL Causal-SeisField + CA-URC locked model
# ---------------------------------------------------------------------

EXPECTED_ROWS = 44_800
EXPECTED_EVENTS = 224
EXPECTED_PGA_TAIL_ROWS = 1_363
EXPECTED_PGV_TAIL_ROWS = 1_485

TRUTH_COLUMNS = {
    "pga": "true_log10_pga",
    "pgv": "true_log10_pgv",
}

TAIL_COLUMNS = {
    "pga": "is_tail_pga",
    "pgv": "is_tail_pgv",
}

BASE_COLUMNS = {
    "pga": "A2_cross_attention_base_log10_pga",
    "pgv": "A2_cross_attention_base_log10_pgv",
}

FINAL_COLUMNS = {
    "pga": "A4_under_only_log10_pga",
    "pgv": "A4_under_only_log10_pgv",
}



# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def load_module(path: str | Path, name: str):
    path = Path(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
                "t",
            }
        )
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
        * math.cos(
            math.radians(
                origin_latitude
            )
        )
    )

    y = (
        latitude - origin_latitude
    ) * 110.57

    return np.column_stack(
        [
            x,
            y,
        ]
    )


def detect_epicenter(
    row: pd.Series,
) -> tuple[
    float | None,
    float | None,
]:
    latitude_columns = [
        "event_latitude",
        "latitude",
        "lat",
        "event_lat",
        "hypocenter_latitude",
    ]

    longitude_columns = [
        "event_longitude",
        "longitude",
        "lon",
        "event_lon",
        "hypocenter_longitude",
    ]

    latitude = None
    longitude = None

    for column in latitude_columns:
        if column not in row.index:
            continue

        value = pd.to_numeric(
            row[column],
            errors="coerce",
        )

        if np.isfinite(value):
            latitude = float(
                value
            )
            break

    for column in longitude_columns:
        if column not in row.index:
            continue

        value = pd.to_numeric(
            row[column],
            errors="coerce",
        )

        if np.isfinite(value):
            longitude = float(
                value
            )
            break

    return (
        latitude,
        longitude,
    )


# ---------------------------------------------------------------------
# Case-selection statistics
# ---------------------------------------------------------------------


def build_case_stats(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build event-repeat case statistics from the FINAL locked prediction table.

    FINAL model:
        Cross-Attention Base + A4_under_only CA-URC head

        y_CA-URC = y_CA + p_under**gamma * Delta

    The locked table is expected to contain:
        A2_cross_attention_base_log10_{pga,pgv}
        A4_under_only_log10_{pga,pgv}

    Case selection is performed only from these locked target rows.
    """
    required = [
        "event_id",
        "repeat",
        "target_slot",
        "target_station_index",
        *TRUTH_COLUMNS.values(),
        *TAIL_COLUMNS.values(),
        *BASE_COLUMNS.values(),
        *FINAL_COLUMNS.values(),
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            "Final locked prediction CSV is missing columns:\n"
            + "\n".join(missing)
        )

    working = dataframe.copy()
    working["event_id"] = working["event_id"].astype(str).str.strip()
    working["is_tail_pga"] = as_bool(working["is_tail_pga"])
    working["is_tail_pgv"] = as_bool(working["is_tail_pgv"])

    # Manuscript guard: this figure must use the same locked grouped test.
    audit = {
        "rows": int(len(working)),
        "events": int(working["event_id"].nunique()),
        "pga_tail_rows": int(working["is_tail_pga"].sum()),
        "pgv_tail_rows": int(working["is_tail_pgv"].sum()),
    }
    expected = {
        "rows": EXPECTED_ROWS,
        "events": EXPECTED_EVENTS,
        "pga_tail_rows": EXPECTED_PGA_TAIL_ROWS,
        "pgv_tail_rows": EXPECTED_PGV_TAIL_ROWS,
    }
    problems = [
        f"{key}: observed={audit[key]}, expected={value}"
        for key, value in expected.items()
        if int(audit[key]) != int(value)
    ]
    if problems:
        raise RuntimeError(
            "Final locked grouped-test audit failed:\n  "
            + "\n  ".join(problems)
        )

    rows = []

    for (event_id, repeat), group in working.groupby(
        ["event_id", "repeat"],
        sort=False,
    ):
        group = group.sort_values("target_slot", kind="stable")

        true_pga = group[TRUTH_COLUMNS["pga"]].to_numpy(dtype=float)
        true_pgv = group[TRUTH_COLUMNS["pgv"]].to_numpy(dtype=float)

        base_pga = group[BASE_COLUMNS["pga"]].to_numpy(dtype=float)
        base_pgv = group[BASE_COLUMNS["pgv"]].to_numpy(dtype=float)

        final_pga = group[FINAL_COLUMNS["pga"]].to_numpy(dtype=float)
        final_pgv = group[FINAL_COLUMNS["pgv"]].to_numpy(dtype=float)

        tail_pga = group[TAIL_COLUMNS["pga"]].to_numpy(dtype=bool)
        tail_pgv = group[TAIL_COLUMNS["pgv"]].to_numpy(dtype=bool)

        base_overall = 0.5 * (
            np.mean(np.abs(base_pga - true_pga))
            + np.mean(np.abs(base_pgv - true_pgv))
        )

        final_overall = 0.5 * (
            np.mean(np.abs(final_pga - true_pga))
            + np.mean(np.abs(final_pgv - true_pgv))
        )

        base_tail_errors: list[float] = []
        final_tail_errors: list[float] = []

        if tail_pga.any():
            base_tail_errors.extend(
                np.abs(base_pga[tail_pga] - true_pga[tail_pga])
            )
            final_tail_errors.extend(
                np.abs(final_pga[tail_pga] - true_pga[tail_pga])
            )

        if tail_pgv.any():
            base_tail_errors.extend(
                np.abs(base_pgv[tail_pgv] - true_pgv[tail_pgv])
            )
            final_tail_errors.extend(
                np.abs(final_pgv[tail_pgv] - true_pgv[tail_pgv])
            )

        base_tail_mae = (
            float(np.mean(base_tail_errors))
            if base_tail_errors
            else np.nan
        )
        final_tail_mae = (
            float(np.mean(final_tail_errors))
            if final_tail_errors
            else np.nan
        )

        rows.append(
            {
                "event_id": str(event_id),
                "repeat": int(repeat),
                "n_tail": int((tail_pga | tail_pgv).sum()),
                "base_overall": float(base_overall),
                "caurc_overall": float(final_overall),
                "base_tail": base_tail_mae,
                "caurc_tail": final_tail_mae,
                "tail_gain": (
                    base_tail_mae - final_tail_mae
                    if np.isfinite(base_tail_mae)
                    and np.isfinite(final_tail_mae)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def choose_pairs(
    statistics: pd.DataFrame,
    event_a: str | None = None,
    repeat_a: int | None = None,
    event_b: str | None = None,
    repeat_b: int | None = None,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """
    Case A: representative case.
      Default:
        Prefer event-repeat units with <=1 high-motion target and select
        the unit whose base overall error is closest to the median.

    Case B: high-motion target case.
      Default:
        Prefer event-repeat units with >=2 high-motion targets and select
        the largest tail gain, with n_tail and base-tail error as tie-breakers.
    """

    def manual(
        event_id: str,
        repeat: int | None,
        high_motion_case: bool,
    ) -> pd.Series:
        subset = statistics.loc[
            statistics[
                "event_id"
            ]
            .astype(str)
            .eq(
                str(
                    event_id
                )
            )
        ].copy()

        if repeat is not None:
            subset = subset.loc[
                subset[
                    "repeat"
                ].eq(
                    repeat
                )
            ]

        if len(
            subset
        ) == 0:
            raise ValueError(
                "Requested event/repeat not found: "
                f"{event_id}/{repeat}"
            )

        if repeat is not None:
            return subset.iloc[
                0
            ]

        if high_motion_case:
            subset[
                "_tail_gain"
            ] = subset[
                "tail_gain"
            ].fillna(
                -1.0e9
            )

            return (
                subset.sort_values(
                    [
                        "_tail_gain",
                        "n_tail",
                        "base_tail",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ],
                )
                .iloc[0]
            )

        return (
            subset.sort_values(
                "base_overall",
                ascending=True,
            )
            .iloc[0]
        )

    if event_a is not None:
        representative = manual(
            event_a,
            repeat_a,
            False,
        )
    else:
        candidates = statistics.loc[
            statistics[
                "n_tail"
            ]
            <= 1
        ].copy()

        if len(
            candidates
        ) == 0:
            candidates = (
                statistics.copy()
            )

        median_error = float(
            candidates[
                "base_overall"
            ].median()
        )

        candidates[
            "_distance_to_median"
        ] = np.abs(
            candidates[
                "base_overall"
            ]
            - median_error
        )

        representative = (
            candidates.sort_values(
                [
                    "_distance_to_median",
                    "base_overall",
                ]
            )
            .iloc[0]
        )

    if event_b is not None:
        high_motion = manual(
            event_b,
            repeat_b,
            True,
        )
    else:
        candidates = statistics.loc[
            statistics[
                "n_tail"
            ]
            >= 2
        ].copy()

        if len(
            candidates
        ) == 0:
            candidates = statistics.loc[
                statistics[
                    "n_tail"
                ]
                >= 1
            ].copy()

        if len(
            candidates
        ) == 0:
            candidates = (
                statistics.copy()
            )

        candidates[
            "_tail_gain"
        ] = candidates[
            "tail_gain"
        ].fillna(
            -1.0e9
        )

        high_motion = (
            candidates.sort_values(
                [
                    "_tail_gain",
                    "n_tail",
                    "base_tail",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

    return (
        representative,
        high_motion,
    )


# ---------------------------------------------------------------------
# Reconstruct full held-out station query for selected event-repeat
# ---------------------------------------------------------------------


def load_checkpoint(
    path: str | Path,
    device: torch.device,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return torch.load(
            str(path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            str(path),
            map_location=device,
        )


def configure_torch_determinism() -> dict[str, Any]:
    """
    Stabilise frozen-model inference and disable TF32 / fused SDPA paths when
    possible. Cross-backend differences are still audited against the archived
    locked prediction table.
    """
    audit: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        audit["deterministic_algorithms"] = True
    except Exception as exc:
        audit["deterministic_algorithms"] = f"unavailable: {exc}"

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    return audit


def reconstruct_final_model(
    baseline_module_path: Path,
    ablation_module_path: Path,
    base_checkpoint_path: Path,
    head_checkpoint_path: Path,
    selection_json_path: Path,
    device_name: str,
    hidden_dim: int,
    attention_heads: int,
    risk_hidden: int,
    maximum_correction: float,
):
    """
    Reconstruct the FINAL manuscript model:

        Cross-Attention Base checkpoint
        + A4_under_only CA-URC head checkpoint

        y_final = y_CA + p_under**gamma * Delta

    The final grouped manuscript lock is enforced:
        variant = A4_under_only
        epoch   = 3
        gamma   = 5
    """
    baseline_module = load_module(
        baseline_module_path,
        "fig411_final_baseline_module",
    )
    ablation_module = load_module(
        ablation_module_path,
        "fig411_final_caurc_module",
    )

    if device_name == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device(device_name)

    base_checkpoint = load_checkpoint(
        base_checkpoint_path,
        device,
    )
    head_checkpoint = load_checkpoint(
        head_checkpoint_path,
        device,
    )

    selection = json.loads(
        Path(selection_json_path).read_text(
            encoding="utf-8"
        )
    )

    variant = str(
        selection.get(
            "variant",
            "",
        )
    )
    gamma = float(
        selection.get(
            "selected_gamma",
            np.nan,
        )
    )
    selected_epoch = int(
        selection.get(
            "selected_epoch",
            -1,
        )
    )

    if variant != "A4_under_only":
        raise RuntimeError(
            f"Expected final variant A4_under_only, found {variant!r}."
        )

    if selected_epoch != 3:
        raise RuntimeError(
            f"Expected final selected epoch=3, found {selected_epoch}."
        )

    if not np.isclose(gamma, 5.0):
        raise RuntimeError(
            f"Expected final gamma=5, found {gamma}."
        )

    if str(head_checkpoint.get("variant", "")) != "A4_under_only":
        raise RuntimeError(
            "Head checkpoint is not the final A4_under_only variant."
        )

    if int(head_checkpoint.get("epoch", -1)) != selected_epoch:
        raise RuntimeError(
            "Head checkpoint epoch does not match selected_epoch_gamma.json."
        )

    head_args = head_checkpoint.get("args", {}) or {}

    hidden_dim = int(
        head_args.get(
            "hidden_dim",
            hidden_dim,
        )
    )
    attention_heads = int(
        head_args.get(
            "attention_heads",
            attention_heads,
        )
    )
    risk_hidden = int(
        head_args.get(
            "risk_hidden",
            risk_hidden,
        )
    )
    maximum_correction = float(
        head_args.get(
            "maximum_correction",
            maximum_correction,
        )
    )

    base = ablation_module.make_frozen_base(
        baseline_module,
        base_checkpoint,
        hidden_dim,
        attention_heads,
        device,
    )

    model = ablation_module.AblationRiskGate(
        base=base,
        variant="A4_under_only",
        hidden_dim=hidden_dim,
        risk_hidden=risk_hidden,
        maximum_correction=maximum_correction,
        tail_prevalence=np.asarray([0.1, 0.1]),
        under_prevalence=np.asarray([0.1, 0.1]),
    ).to(device)

    model.load_state_dict(
        head_checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    audit = {
        "device": str(device),
        "variant": "A4_under_only",
        "selected_epoch": selected_epoch,
        "selected_gamma": gamma,
        "hidden_dim": hidden_dim,
        "attention_heads": attention_heads,
        "risk_hidden": risk_hidden,
        "maximum_correction": maximum_correction,
        "base_checkpoint_variant": str(
            base_checkpoint.get(
                "variant",
                "cross_attention",
            )
        ),
        "head_checkpoint_variant": str(
            head_checkpoint.get(
                "variant",
                "",
            )
        ),
    }

    return (
        baseline_module,
        ablation_module,
        model,
        gamma,
        device,
        audit,
    )


def resolve_h5_path(
    event_id: str,
    manifest_row: pd.Series,
    h5_root: Path | None,
) -> Path:
    """
    Resolve the selected event HDF5 on Linux.

    The manifest may still contain an old local path, so H5_ROOT is always
    available as a manuscript-server fallback.
    """
    raw = str(
        manifest_row.get(
            "h5_path",
            "",
        )
    ).strip()

    if raw and raw.lower() not in {"nan", "none"}:
        candidate = Path(raw)
        if candidate.exists():
            return candidate

    if h5_root is not None:
        candidates = [
            h5_root / f"{event_id}.h5",
            h5_root / str(event_id) / f"{event_id}.h5",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Only two selected cases are resolved, so this final fallback is cheap
        # enough and protects against one extra directory level.
        matches = list(
            h5_root.glob(
                f"**/{event_id}.h5"
            )
        )
        if len(matches) == 1:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot resolve HDF5 for event {event_id}. "
        f"manifest h5_path={raw!r}; h5_root={h5_root}"
    )


def prepare_model_arrays(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    coordinates: np.ndarray,
    p_offset: np.ndarray,
    input_indices: np.ndarray,
    target_indices: np.ndarray,
    sampling_rate: float,
    pre_first_p: float,
    time_zero_index: int,
    t0_sec: float,
    input_pre_sec: float,
) -> dict[str, np.ndarray | int]:
    """
    Reproduce the final Causal-SeisField model inputs used by the locked test.
    """
    input_start = max(
        0,
        int(
            round(
                (
                    pre_first_p
                    - input_pre_sec
                )
                * sampling_rate
            )
        ),
    )

    snapshot = min(
        time_zero_index
        + int(
            round(
                t0_sec
                * sampling_rate
            )
        ),
        acceleration.shape[-1]
        - 1,
    )

    observed_acceleration = acceleration[
        input_indices,
        :,
        input_start:snapshot,
    ]

    input_waveforms = (
        np.sign(
            observed_acceleration
        )
        * np.log1p(
            np.abs(
                observed_acceleration
            )
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

    future_acceleration = acceleration[
        target_indices,
        :,
        snapshot:,
    ]
    future_velocity = velocity[
        target_indices,
        :,
        snapshot:,
    ]

    future_pga = np.max(
        np.sqrt(
            future_acceleration[:, 0, :] ** 2
            + future_acceleration[:, 1, :] ** 2
        ),
        axis=1,
    )

    future_pgv = np.max(
        np.sqrt(
            future_velocity[:, 0, :] ** 2
            + future_velocity[:, 1, :] ** 2
        ),
        axis=1,
    )

    truth = np.column_stack(
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
        ]
    ).astype(np.float32)

    return {
        "input_waveforms": input_waveforms,
        "input_features": input_features,
        "target_features": target_features,
        "truth": truth,
        "snapshot": snapshot,
    }


def predict_targets(
    model,
    arrays: dict[str, Any],
    gamma: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    FINAL inference:
        y_CA-URC = y_CA + model.correction(output, gamma)

    For A4_under_only:
        correction = p_under**gamma * Delta
    """
    with torch.inference_mode():
        output = model(
            torch.from_numpy(
                arrays["input_waveforms"]
            )[None].to(device),
            torch.from_numpy(
                arrays["input_features"]
            )[None].to(device),
            torch.from_numpy(
                arrays["target_features"]
            )[None].to(device),
        )

        correction = model.correction(
            output,
            gamma=float(gamma),
        )

        base = output["base"]
        final = base + correction

    return {
        "base": base[0].detach().cpu().numpy(),
        "final": final[0].detach().cpu().numpy(),
        "correction": correction[0].detach().cpu().numpy(),
        "under_probability": (
            output["under_probability"][0]
            .detach()
            .cpu()
            .numpy()
        ),
        "raw_correction_amplitude": (
            output["residual"][0]
            .detach()
            .cpu()
            .numpy()
        ),
    }


def predict_targets_individually(
    model,
    acceleration: np.ndarray,
    velocity: np.ndarray,
    coordinates: np.ndarray,
    p_offset: np.ndarray,
    input_indices: np.ndarray,
    target_indices: np.ndarray,
    sampling_rate: float,
    pre_first_p: float,
    time_zero_index: int,
    t0_sec: float,
    input_pre_sec: float,
    gamma: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Query one held-out target at a time so the descriptive map is not affected
    by target-query batch shape.
    """
    outputs: dict[str, list[np.ndarray]] = {
        "base": [],
        "final": [],
        "correction": [],
        "under_probability": [],
        "raw_correction_amplitude": [],
    }
    truths: list[np.ndarray] = []

    for target_index in np.asarray(
        target_indices,
        dtype=int,
    ):
        arrays = prepare_model_arrays(
            acceleration,
            velocity,
            coordinates,
            p_offset,
            input_indices,
            np.asarray(
                [target_index],
                dtype=int,
            ),
            sampling_rate,
            pre_first_p,
            time_zero_index,
            t0_sec,
            input_pre_sec,
        )

        prediction = predict_targets(
            model,
            arrays,
            gamma,
            device,
        )

        truths.append(
            np.asarray(
                arrays["truth"][0],
                dtype=np.float64,
            )
        )

        for key in outputs:
            outputs[key].append(
                np.asarray(
                    prediction[key][0],
                    dtype=np.float64,
                )
            )

    merged = {
        key: np.stack(
            value,
            axis=0,
        )
        for key, value in outputs.items()
    }

    truth = np.stack(
        truths,
        axis=0,
    )

    return (
        merged,
        truth,
    )


def case_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    tail: np.ndarray,
) -> dict[str, float | int]:
    residual = prediction - truth
    absolute = np.abs(residual)

    output: dict[str, float | int] = {}

    for index, quantity in enumerate(
        ["pga", "pgv"]
    ):
        tail_mask = tail[:, index]

        output[f"{quantity}_mae"] = float(
            np.mean(
                absolute[:, index]
            )
        )

        output[f"{quantity}_tail_n"] = int(
            tail_mask.sum()
        )

        output[f"{quantity}_tail_mae"] = (
            float(
                np.mean(
                    absolute[
                        tail_mask,
                        index,
                    ]
                )
            )
            if tail_mask.any()
            else np.nan
        )

        output[f"{quantity}_bias"] = float(
            np.mean(
                residual[:, index]
            )
        )

    return output


def infer_all_heldout(
    pair: pd.Series,
    locked_predictions: pd.DataFrame,
    manifest_row: pd.Series,
    h5_root: Path | None,
    baseline_module,
    model,
    gamma: float,
    device: torch.device,
    thresholds: np.ndarray,
    split_label: str,
    seed: int,
    t0_sec: float,
    input_stations: int,
    target_stations: int,
    input_pre_sec: float,
    audit_tolerance: float,
    query_tolerance: float,
) -> dict:
    """
    Reconstruct the exact locked grouped-test station draw, audit the final
    Cross-Attention Base + A4 CA-URC model against archived predictions, then
    query all remaining held-out stations for descriptive spatial maps.
    """
    event_id = str(
        pair["event_id"]
    )
    repeat = int(
        pair["repeat"]
    )

    h5_path = resolve_h5_path(
        event_id,
        manifest_row,
        h5_root,
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
        station_ids = (
            h5["station_id"].asstr()[:]
            if "station_id" in h5
            else np.asarray(
                [str(i) for i in range(len(coordinates))],
                dtype=object,
            )
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
        & (p_offset <= float(t0_sec))
    )

    if len(triggered) < int(input_stations):
        raise RuntimeError(
            f"Event {event_id} has only {len(triggered)} eligible input stations."
        )

    random_seed = baseline_module.stable_seed(
        f"strong:{split_label}:{event_id}:repeat:{repeat}",
        int(seed),
    )

    rng = np.random.default_rng(
        random_seed
    )

    input_indices = rng.choice(
        triggered,
        size=int(input_stations),
        replace=False,
    )

    target_pool = np.setdiff1d(
        np.arange(len(coordinates)),
        input_indices,
        assume_unique=False,
    )

    locked_target_indices = rng.choice(
        target_pool,
        size=int(target_stations),
        replace=False,
    )

    locked_group = locked_predictions.loc[
        locked_predictions["event_id"].astype(str).eq(event_id)
        & locked_predictions["repeat"].eq(repeat)
    ].sort_values(
        "target_slot",
        kind="stable",
    )

    if len(locked_group) != int(target_stations):
        raise RuntimeError(
            f"Selected event-repeat {event_id}/{repeat} has "
            f"{len(locked_group)} locked targets; expected {target_stations}."
        )

    stored_targets = locked_group[
        "target_station_index"
    ].to_numpy(dtype=int)

    if not np.array_equal(
        locked_target_indices.astype(int),
        stored_targets,
    ):
        raise RuntimeError(
            "\nLocked-test station draw reproduction failed.\n"
            f"Event       : {event_id}\n"
            f"Repeat      : {repeat}\n"
            f"Random seed : {int(random_seed)}\n"
            f"Input idx   : {input_indices.astype(int).tolist()}\n"
            f"Generated   : {locked_target_indices.astype(int).tolist()}\n"
            f"Stored      : {stored_targets.tolist()}\n"
        )

    # Exact locked-target audit.
    locked_arrays = prepare_model_arrays(
        acceleration,
        velocity,
        coordinates,
        p_offset,
        input_indices,
        locked_target_indices,
        sampling_rate,
        pre_first_p,
        time_zero_index,
        t0_sec,
        input_pre_sec,
    )

    locked_output = predict_targets(
        model,
        locked_arrays,
        gamma,
        device,
    )

    stored_truth = np.column_stack(
        [
            locked_group[TRUTH_COLUMNS["pga"]].to_numpy(float),
            locked_group[TRUTH_COLUMNS["pgv"]].to_numpy(float),
        ]
    )

    stored_base = np.column_stack(
        [
            locked_group[BASE_COLUMNS["pga"]].to_numpy(float),
            locked_group[BASE_COLUMNS["pgv"]].to_numpy(float),
        ]
    )

    stored_final = np.column_stack(
        [
            locked_group[FINAL_COLUMNS["pga"]].to_numpy(float),
            locked_group[FINAL_COLUMNS["pgv"]].to_numpy(float),
        ]
    )

    reproduction = {
        "max_abs_truth_diff": float(
            np.max(
                np.abs(
                    locked_arrays["truth"]
                    - stored_truth
                )
            )
        ),
        "max_abs_base_diff": float(
            np.max(
                np.abs(
                    locked_output["base"]
                    - stored_base
                )
            )
        ),
        "max_abs_final_diff": float(
            np.max(
                np.abs(
                    locked_output["final"]
                    - stored_final
                )
            )
        ),
    }

    truth_tolerance = min(
        float(audit_tolerance),
        2e-6,
    )
    hard_prediction_ceiling = 1e-3

    if reproduction["max_abs_truth_diff"] > truth_tolerance:
        raise RuntimeError(
            "Truth reconstruction failed: "
            f"{reproduction}"
        )

    if (
        reproduction["max_abs_base_diff"] > hard_prediction_ceiling
        or reproduction["max_abs_final_diff"] > hard_prediction_ceiling
    ):
        raise RuntimeError(
            "Frozen final-model reproduction differs too much from the "
            f"locked archive: {reproduction}"
        )

    archive_drift = max(
        reproduction["max_abs_base_diff"],
        reproduction["max_abs_final_diff"],
    )

    if archive_drift > float(audit_tolerance):
        print(
            "WARNING: small cross-backend archive drift: "
            f"{archive_drift:.3e} > {audit_tolerance:.3e}, "
            f"but < {hard_prediction_ceiling:.3e}."
        )

    # Query-set independence audit on the ten locked targets.
    single_diffs: list[float] = []

    for slot, target_index in enumerate(
        locked_target_indices
    ):
        one_arrays = prepare_model_arrays(
            acceleration,
            velocity,
            coordinates,
            p_offset,
            input_indices,
            np.asarray(
                [target_index],
                dtype=int,
            ),
            sampling_rate,
            pre_first_p,
            time_zero_index,
            t0_sec,
            input_pre_sec,
        )

        one_output = predict_targets(
            model,
            one_arrays,
            gamma,
            device,
        )

        single_diffs.extend(
            [
                float(
                    np.max(
                        np.abs(
                            one_output["base"][0]
                            - locked_output["base"][slot]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            one_output["final"][0]
                            - locked_output["final"][slot]
                        )
                    )
                ),
            ]
        )

    max_query_drift = float(
        max(single_diffs)
        if single_diffs
        else 0.0
    )

    if max_query_drift > 1e-3:
        raise RuntimeError(
            "Target predictions depend too strongly on query batch shape: "
            f"{max_query_drift:.3e}"
        )

    if max_query_drift > float(query_tolerance):
        print(
            "WARNING: small query-set drift detected: "
            f"{max_query_drift:.3e}. "
            "All-held-out maps will use target-by-target inference."
        )

    # Descriptive all-held-out query.
    all_target_indices = target_pool.astype(int)

    all_output, truth = predict_targets_individually(
        model,
        acceleration,
        velocity,
        coordinates,
        p_offset,
        input_indices,
        all_target_indices,
        sampling_rate,
        pre_first_p,
        time_zero_index,
        t0_sec,
        input_pre_sec,
        gamma,
        device,
    )

    tail = truth >= thresholds[None, :]

    epicenter_latitude, epicenter_longitude = detect_epicenter(
        manifest_row
    )

    if (
        epicenter_latitude is None
        or epicenter_longitude is None
    ):
        origin_latitude = float(
            coordinates[:, 0].mean()
        )
        origin_longitude = float(
            coordinates[:, 1].mean()
        )
        epicenter_xy = None
    else:
        origin_latitude = epicenter_latitude
        origin_longitude = epicenter_longitude
        epicenter_xy = np.asarray(
            [0.0, 0.0]
        )

    xy = local_xy_km(
        coordinates,
        origin_latitude,
        origin_longitude,
    )

    magnitude = pd.to_numeric(
        manifest_row.get(
            "magnitude",
            np.nan,
        ),
        errors="coerce",
    )

    target_triggered = (
        np.isfinite(
            p_offset[
                all_target_indices
            ]
        )
        & (
            p_offset[
                all_target_indices
            ]
            <= t0_sec
        )
    )

    return {
        "event_id": event_id,
        "repeat": repeat,
        "magnitude": (
            float(magnitude)
            if np.isfinite(magnitude)
            else np.nan
        ),
        "xy_target": xy[
            all_target_indices
        ],
        "xy_input": xy[
            input_indices
        ],
        "epicenter": epicenter_xy,
        "truth": truth,
        "base": all_output["base"],
        "final": all_output["final"],
        "correction": all_output["correction"],
        "under_probability": all_output["under_probability"],
        "raw_correction_amplitude": all_output["raw_correction_amplitude"],
        "tail": tail,
        "target_triggered": target_triggered,
        "base_metrics": case_metrics(
            truth,
            all_output["base"],
            tail,
        ),
        "final_metrics": case_metrics(
            truth,
            all_output["final"],
            tail,
        ),
        "station_ids_target": [
            str(
                station_ids[index]
            )
            for index in all_target_indices
        ],
        "reproduction_audit": {
            **reproduction,
            "max_query_set_dependence": max_query_drift,
            "random_seed": int(random_seed),
            "input_indices": input_indices.astype(int).tolist(),
            "locked_target_indices": locked_target_indices.astype(int).tolist(),
            "h5_path": str(h5_path.resolve()),
        },
    }


def shifted_log(
    values: np.ndarray,
    quantity_index: int,
) -> np.ndarray:
    """
    Model values:
        PGA log10(m/s^2)
        PGV log10(m/s)

    Display:
        PGA in g
        PGV in cm/s
    """
    if quantity_index == 0:
        return (
            values
            - math.log10(
                9.80665
            )
        )

    return (
        values
        + 2.0
    )


def shared_limits(
    cases: list[dict],
    quantity_index: int,
) -> tuple[
    float,
    float,
]:
    values = np.concatenate(
        [
            shifted_log(
                case[key][
                    :,
                    quantity_index,
                ],
                quantity_index,
            )
            for case
            in cases
            for key
            in (
                "truth",
                "base",
                "final",
            )
        ]
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:
        raise ValueError(
            "No finite plotting values."
        )

    low, high = np.percentile(
        values,
        [
            2,
            98,
        ],
    )

    if (
        high - low
        < 0.2
    ):
        middle = 0.5 * (
            high + low
        )

        low = (
            middle - 0.1
        )

        high = (
            middle + 0.1
        )

    return (
        float(
            low
        ),
        float(
            high
        ),
    )


def physical_tick_label(
    log_value: float,
) -> str:
    value = (
        10.0
        ** log_value
    )

    if value >= 100:
        return (
            f"{value:.0f}"
        )

    if value >= 10:
        return (
            f"{value:.1f}"
        )

    if value >= 1:
        return (
            f"{value:.2f}"
        )

    if value >= 0.1:
        return (
            f"{value:.2f}"
        )

    if value >= 0.01:
        return (
            f"{value:.3f}"
        )

    return (
        f"{value:.1e}"
    )


def map_kwargs(
    cmap: str | None,
) -> dict:
    if (
        cmap is None
        or str(
            cmap
        ).strip() == ""
        or str(
            cmap
        ).lower() == "default"
    ):
        return {}

    return {
        "cmap": cmap
    }


def draw_map(
    axis: plt.Axes,
    case: dict,
    values: np.ndarray,
    quantity_index: int,
    low: float,
    high: float,
    cmap: str | None,
    annotation: str | None = None,
) -> None:
    xy = case[
        "xy_target"
    ]

    x = xy[
        :,
        0,
    ]

    y = xy[
        :,
        1,
    ]

    z = shifted_log(
        values,
        quantity_index,
    )

    finite = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
        & np.isfinite(
            z
        )
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    z = z[
        finite
    ]

    levels = np.linspace(
        low,
        high,
        80,
    )

    color_kwargs = map_kwargs(
        cmap
    )

    try:
        triangulation = (
            mtri.Triangulation(
                x,
                y,
            )
        )

        axis.tricontourf(
            triangulation,
            np.clip(
                z,
                low,
                high,
            ),
            levels=levels,
            extend="both",
            **color_kwargs,
        )

    except Exception:
        axis.scatter(
            x,
            y,
            c=np.clip(
                z,
                low,
                high,
            ),
            vmin=low,
            vmax=high,
            s=20,
            **color_kwargs,
        )

    # Held-out target stations:
    # use small open markers so the spatial surface remains visible.
    axis.scatter(
        x,
        y,
        s=6,
        facecolors="none",
        edgecolors="black",
        linewidths=0.32,
        alpha=0.50,
        zorder=5,
    )

    # Input stations:
    input_xy = case[
        "xy_input"
    ]

    axis.scatter(
        input_xy[
            :,
            0,
        ],
        input_xy[
            :,
            1,
        ],
        marker="^",
        s=38,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=8,
    )

    # Epicenter if catalog coordinates are available.
    if (
        case[
            "epicenter"
        ]
        is not None
    ):
        axis.scatter(
            [
                0.0
            ],
            [
                0.0
            ],
            marker="*",
            s=92,
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
            zorder=9,
        )

    if annotation:
        axis.text(
            0.03,
            0.04,
            annotation,
            transform=(
                axis.transAxes
            ),
            fontsize=8.0,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": (
                    "round,pad=0.20"
                ),
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 0.6,
                "alpha": 0.90,
            },
        )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.tick_params(
        labelsize=8,
        direction="out",
        length=3,
    )

    for spine in (
        axis.spines.values()
    ):
        spine.set_linewidth(
            0.8
        )


def metric_annotation(
    metrics: dict,
    quantity: str,
) -> str:
    text = (
        "MAE="
        f"{metrics[f'{quantity}_mae']:.3f}"
    )

    if (
        metrics[
            f"{quantity}_tail_n"
        ]
        > 0
        and np.isfinite(
            metrics[
                f"{quantity}_tail_mae"
            ]
        )
    ):
        text += (
            "\nTail MAE="
            f"{metrics[f'{quantity}_tail_mae']:.3f}"
        )

    return text


def add_column_colorbar(
    figure: plt.Figure,
    axis: plt.Axes,
    low: float,
    high: float,
    label: str,
    cmap: str | None,
) -> None:
    position = (
        axis.get_position()
    )

    colorbar_height = 0.014

    colorbar_axis = (
        figure.add_axes(
            [
                position.x0,
                0.060,
                position.width,
                colorbar_height,
            ]
        )
    )

    kwargs = map_kwargs(
        cmap
    )

    scalar_mappable = ScalarMappable(
        norm=Normalize(
            low,
            high,
        ),
        **kwargs,
    )

    scalar_mappable.set_array(
        []
    )

    colorbar = (
        figure.colorbar(
            scalar_mappable,
            cax=colorbar_axis,
            orientation="horizontal",
            extend="both",
        )
    )

    ticks = np.linspace(
        low,
        high,
        4,
    )

    colorbar.set_ticks(
        ticks
    )

    colorbar.set_ticklabels(
        [
            physical_tick_label(
                tick
            )
            for tick
            in ticks
        ]
    )

    colorbar.ax.tick_params(
        labelsize=7.3,
        length=2.5,
    )

    colorbar.set_label(
        label,
        fontsize=8.2,
        labelpad=2.0,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    # ============================================================
    # FINAL Causal-SeisField + CA-URC paths
    # Run from the project root, e.g. /mnt/世界模型
    # ============================================================

    parser.add_argument(
        "--predictions",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "locked_test_ablation_predictions.csv"
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "model_manifests/"
            "scenario_t0_5s_k5_linux.csv"
        ),
    )

    parser.add_argument(
        "--h5-root",
        default=(
            'data/scedc/processed_full_v4/events'
        ),
    )

    parser.add_argument(
        "--base-checkpoint",
        default=(
            "runs/final_strong_baselines_reuse_locked/"
            "cross_attention/best_model.pt"
        ),
    )

    parser.add_argument(
        "--head-checkpoint",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "A4_under_only/selected_head.pt"
        ),
    )

    parser.add_argument(
        "--selection-json",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "A4_under_only/selected_epoch_gamma.json"
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
        "--baseline-module",
        default="45_phase2_strong_baseline_suite.py",
    )

    parser.add_argument(
        "--ablation-module",
        default=(
            "63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py"
        ),
    )

    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )

    parser.add_argument(
        "--test-label",
        default="test",
    )

    parser.add_argument(
        "--event-a",
        default=None,
    )

    parser.add_argument(
        "--repeat-a",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--event-b",
        default=None,
    )

    parser.add_argument(
        "--repeat-b",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )

    parser.add_argument(
        "--t0-sec",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--input-pre-sec",
        type=float,
        default=2.0,
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
        "--risk-hidden",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--maximum-correction",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--audit-tolerance",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--query-tolerance",
        type=float,
        default=2e-4,
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
        "--cmap",
        default="turbo",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "figures/section_4_11_caurc"
        ),
    )

    args = parser.parse_args()

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Resolve final-model resources.
    # -----------------------------------------------------------------
    predictions_path = Path(
        args.predictions
    )
    manifest_path = Path(
        args.manifest
    )
    h5_root = Path(
        args.h5_root
    )
    base_checkpoint_path = Path(
        args.base_checkpoint
    )
    head_checkpoint_path = Path(
        args.head_checkpoint
    )
    selection_json_path = Path(
        args.selection_json
    )
    threshold_path = Path(
        args.threshold_json
    )
    baseline_module_path = Path(
        args.baseline_module
    )
    ablation_module_path = Path(
        args.ablation_module
    )

    required_paths = [
        predictions_path,
        manifest_path,
        h5_root,
        base_checkpoint_path,
        head_checkpoint_path,
        selection_json_path,
        threshold_path,
        baseline_module_path,
        ablation_module_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    configure_torch_determinism()

    predictions = pd.read_csv(
        predictions_path,
        dtype={
            "event_id": str,
        },
    )

    statistics = build_case_stats(
        predictions
    )

    statistics_path = (
        out_dir
        / "Fig_4_11_event_repeat_case_statistics_CAURC.csv"
    )
    statistics.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8-sig",
    )

    (
        representative_pair,
        high_motion_pair,
    ) = choose_pairs(
        statistics,
        event_a=args.event_a,
        repeat_a=args.repeat_a,
        event_b=args.event_b,
        repeat_b=args.repeat_b,
    )

    manifest_frame = pd.read_csv(
        manifest_path,
        dtype={
            "event_id": str,
        },
    )

    if args.split_column in manifest_frame.columns:
        manifest_frame = manifest_frame.loc[
            manifest_frame[
                args.split_column
            ].astype(str).eq(
                str(
                    args.test_label
                )
            )
        ].copy()

    manifest_frame["event_id"] = (
        manifest_frame["event_id"]
        .astype(str)
        .str.strip()
    )

    manifest = (
        manifest_frame
        .drop_duplicates(
            "event_id"
        )
        .set_index(
            "event_id",
            drop=False,
        )
    )

    for pair_name, pair in (
        (
            "representative",
            representative_pair,
        ),
        (
            "high-motion",
            high_motion_pair,
        ),
    ):
        event_id = str(
            pair[
                "event_id"
            ]
        )

        if event_id not in manifest.index:
            raise KeyError(
                f"{pair_name} event {event_id} "
                "is absent from the grouped test manifest."
            )

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds = np.asarray(
        [
            threshold_data[
                "log10_pga_threshold"
            ],
            threshold_data[
                "log10_pgv_threshold"
            ],
        ],
        dtype=float,
    )

    (
        baseline_module,
        ablation_module,
        model,
        gamma,
        device,
        model_audit,
    ) = reconstruct_final_model(
        baseline_module_path,
        ablation_module_path,
        base_checkpoint_path,
        head_checkpoint_path,
        selection_json_path,
        args.device,
        args.hidden_dim,
        args.attention_heads,
        args.risk_hidden,
        args.maximum_correction,
    )

    # CRITICAL: Script 63 patched the grouped locked-test seed protocol before
    # producing locked_test_ablation_predictions.csv. Reapply the same patch
    # before regenerating the five input and ten target stations.
    if not hasattr(
        ablation_module,
        "patch_locked_test_seed_protocol",
    ):
        raise AttributeError(
            f"{ablation_module_path.name} does not expose "
            "patch_locked_test_seed_protocol()."
        )

    ablation_module.patch_locked_test_seed_protocol(
        baseline_module
    )

    representative_case = infer_all_heldout(
        representative_pair,
        predictions,
        manifest.loc[
            str(
                representative_pair[
                    "event_id"
                ]
            )
        ],
        h5_root,
        baseline_module,
        model,
        gamma,
        device,
        thresholds,
        str(args.test_label),
        args.seed,
        args.t0_sec,
        args.input_stations,
        args.target_stations,
        args.input_pre_sec,
        args.audit_tolerance,
        args.query_tolerance,
    )

    high_motion_case = infer_all_heldout(
        high_motion_pair,
        predictions,
        manifest.loc[
            str(
                high_motion_pair[
                    "event_id"
                ]
            )
        ],
        h5_root,
        baseline_module,
        model,
        gamma,
        device,
        thresholds,
        str(args.test_label),
        args.seed,
        args.t0_sec,
        args.input_stations,
        args.target_stations,
        args.input_pre_sec,
        args.audit_tolerance,
        args.query_tolerance,
    )

    cases = [
        representative_case,
        representative_case,
        high_motion_case,
        high_motion_case,
    ]

    pga_low, pga_high = shared_limits(
        [
            representative_case,
            high_motion_case,
        ],
        0,
    )

    pgv_low, pgv_high = shared_limits(
        [
            representative_case,
            high_motion_case,
        ],
        1,
    )

    # -----------------------------------------------------------------
    # Publication figure.
    # Keep the original submission layout with the FINAL A4 under-only CA-URC model.
    # -----------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    )

    figure, axes = plt.subplots(
        3,
        4,
        figsize=(
            13.2,
            9.35,
        ),
    )

    rows = [
        (
            "truth",
            "Ground truth",
        ),
        (
            "base",
            "Cross-Attention Base",
        ),
        (
            "final",
            "CA-URC",
        ),
    ]

    quantity_indices = [
        0,
        1,
        0,
        1,
    ]

    for row_index, (
        key,
        row_label,
    ) in enumerate(
        rows
    ):
        for column_index in range(
            4
        ):
            case = cases[
                column_index
            ]

            quantity_index = quantity_indices[
                column_index
            ]

            quantity = (
                "pga"
                if quantity_index == 0
                else "pgv"
            )

            if quantity_index == 0:
                low = pga_low
                high = pga_high
            else:
                low = pgv_low
                high = pgv_high

            annotation = None

            if key in (
                "base",
                "final",
            ):
                metrics_key = (
                    "base_metrics"
                    if key == "base"
                    else "final_metrics"
                )

                annotation = metric_annotation(
                    case[
                        metrics_key
                    ],
                    quantity,
                )

            draw_map(
                axes[
                    row_index,
                    column_index,
                ],
                case,
                case[
                    key
                ][
                    :,
                    quantity_index,
                ],
                quantity_index,
                low,
                high,
                args.cmap,
                annotation,
            )

            if row_index == 0:
                axes[
                    row_index,
                    column_index,
                ].set_title(
                    (
                        "PGA"
                        if quantity_index == 0
                        else "PGV"
                    ),
                    fontweight="bold",
                )

            if row_index < 2:
                axes[
                    row_index,
                    column_index,
                ].set_xticklabels(
                    []
                )
            else:
                axes[
                    row_index,
                    column_index,
                ].set_xlabel(
                    "X (km)"
                )

            if column_index in (
                0,
                2,
            ):
                axes[
                    row_index,
                    column_index,
                ].set_ylabel(
                    "Y (km)"
                )
            else:
                axes[
                    row_index,
                    column_index,
                ].set_yticklabels(
                    []
                )

    # Row labels.
    for row_index, (
        _,
        row_label,
    ) in enumerate(
        rows
    ):
        position = axes[
            row_index,
            0,
        ].get_position()

        figure.text(
            0.025,
            0.5
            * (
                position.y0
                + position.y1
            ),
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    representative_magnitude = (
        f"M{representative_case['magnitude']:.1f}"
        if np.isfinite(
            representative_case[
                "magnitude"
            ]
        )
        else ""
    )

    high_motion_magnitude = (
        f"M{high_motion_case['magnitude']:.1f}"
        if np.isfinite(
            high_motion_case[
                "magnitude"
            ]
        )
        else ""
    )

    figure.text(
        0.285,
        0.975,
        (
            "(a) Representative case  "
            f"{representative_case['event_id']}  "
            f"{representative_magnitude}"
        ),
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    figure.text(
        0.755,
        0.975,
        (
            "(b) High-motion target case  "
            f"{high_motion_case['event_id']}  "
            f"{high_motion_magnitude}"
        ),
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    figure.subplots_adjust(
        left=0.072,
        right=0.988,
        bottom=0.145,
        top=0.92,
        wspace=0.06,
        hspace=0.055,
    )

    for column_index in range(
        4
    ):
        quantity_index = quantity_indices[
            column_index
        ]

        if quantity_index == 0:
            low = pga_low
            high = pga_high
            label = "Future PGA (g)"
        else:
            low = pgv_low
            high = pgv_high
            label = "Future PGV (cm/s)"

        add_column_colorbar(
            figure,
            axes[
                2,
                column_index,
            ],
            low,
            high,
            label,
            args.cmap,
        )

    png_path = (
        out_dir
        / "Fig_4_11_spatial_case_studies_CAURC.png"
    )
    svg_path = (
        out_dir
        / "Fig_4_11_spatial_case_studies_CAURC.svg"
    )
    pdf_path = (
        out_dir
        / "Fig_4_11_spatial_case_studies_CAURC.pdf"
    )

    figure.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # -----------------------------------------------------------------
    # Save selected-case summary using final CA-URC naming.
    # -----------------------------------------------------------------
    summary = pd.DataFrame(
        [
            {
                "case": "representative",
                "event_id": representative_case["event_id"],
                "repeat": representative_case["repeat"],
                "magnitude": representative_case["magnitude"],
                "n_all_heldout_targets": int(
                    len(
                        representative_case[
                            "truth"
                        ]
                    )
                ),
                "n_triggered_targets": int(
                    representative_case[
                        "target_triggered"
                    ].sum()
                ),
                **{
                    "base_" + key: value
                    for key, value
                    in representative_case[
                        "base_metrics"
                    ].items()
                },
                **{
                    "caurc_" + key: value
                    for key, value
                    in representative_case[
                        "final_metrics"
                    ].items()
                },
            },
            {
                "case": "high_motion_target",
                "event_id": high_motion_case["event_id"],
                "repeat": high_motion_case["repeat"],
                "magnitude": high_motion_case["magnitude"],
                "n_all_heldout_targets": int(
                    len(
                        high_motion_case[
                            "truth"
                        ]
                    )
                ),
                "n_triggered_targets": int(
                    high_motion_case[
                        "target_triggered"
                    ].sum()
                ),
                **{
                    "base_" + key: value
                    for key, value
                    in high_motion_case[
                        "base_metrics"
                    ].items()
                },
                **{
                    "caurc_" + key: value
                    for key, value
                    in high_motion_case[
                        "final_metrics"
                    ].items()
                },
            },
        ]
    )

    summary[
        "pga_overall_gain"
    ] = (
        summary[
            "base_pga_mae"
        ]
        - summary[
            "caurc_pga_mae"
        ]
    )

    summary[
        "pgv_overall_gain"
    ] = (
        summary[
            "base_pgv_mae"
        ]
        - summary[
            "caurc_pgv_mae"
        ]
    )

    summary[
        "pga_tail_gain"
    ] = (
        summary[
            "base_pga_tail_mae"
        ]
        - summary[
            "caurc_pga_tail_mae"
        ]
    )

    summary[
        "pgv_tail_gain"
    ] = (
        summary[
            "base_pgv_tail_mae"
        ]
        - summary[
            "caurc_pgv_tail_mae"
        ]
    )

    summary_path = (
        out_dir
        / "Fig_4_11_selected_case_summary_CAURC.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit_path = (
        out_dir
        / "Fig_4_11_final_model_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            {
                "final_model": "Causal-SeisField + CA-URC",
                "formula": "y_CA-URC = y_CA + p_under^gamma * Delta",
                "variant": "A4_under_only",
                "selected_epoch": 3,
                "selected_gamma": 5,
                "model_audit": model_audit,
                "representative_reproduction": representative_case[
                    "reproduction_audit"
                ],
                "high_motion_reproduction": high_motion_case[
                    "reproduction_audit"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Section 4.11 FINAL Causal-SeisField + CA-URC ==="
    )
    print(
        f"Device           : {device}"
    )
    print(
        "Variant          : A4_under_only"
    )
    print(
        "Selected epoch   : 3"
    )
    print(
        f"Selected gamma   : {gamma:g}"
    )
    print(
        f"Base checkpoint  : {base_checkpoint_path.resolve()}"
    )
    print(
        f"Head checkpoint  : {head_checkpoint_path.resolve()}"
    )
    print(
        f"Selection JSON   : {selection_json_path.resolve()}"
    )
    print(
        "\nSelected cases:"
    )
    print(
        summary.to_string(
            index=False
        )
    )
    print(
        "\nSaved:"
    )

    for path in (
        png_path,
        svg_path,
        pdf_path,
        summary_path,
        statistics_path,
        audit_path,
    ):
        print(
            path.resolve()
        )

if __name__ == "__main__":
    main()
