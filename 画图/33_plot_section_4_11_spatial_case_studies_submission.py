#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
32_plot_section_4_11_spatial_case_studies_final.py

Section 4.11
Spatial case studies illustrate selective correction of future
high-motion regions.

This is the publication-ready successor to:
    23_plot_wavecast_style_spatial_cases_fixed.py

Final submission styling
------------------------
1. Uses turbo for the ground-motion intensity scale.
2. Held-out stations are small open black circles.
3. Input stations are white triangles with black edges.
4. Epicenter is a white star with a black edge.
5. "Hazardous-tail event" is replaced by "High-motion target case".
6. The long triangulation explanation is removed from inside the figure
   and should be placed in the manuscript caption.
7. One unambiguous colorbar is shown below each map column.
8. Row labels are "Ground truth", "Frozen base", and
   "Power gate (gamma=3)".
9. Exports PNG, SVG, PDF, selected-case summary, and full selection statistics.
10. Keeps the original reproducible case-selection rule:
   - representative case: near median base error, preferably <=1 tail target;
   - high-motion case: multiple tail targets and large tail improvement.

Important scientific interpretation
-----------------------------------
The colored surfaces are triangulated from station-level held-out queries
for visualization only. They are NOT dense model outputs and are NOT a
continuous ground-truth wavefield. Quantitative metrics are calculated at
the actual held-out station locations.

Default inputs
--------------
Predictions:
    runs/locked_power_gated_test_gamma3/locked_test_predictions.csv

Manifest:
    data/scedc/model_manifests/scenario_t0_5s_k5.csv

Checkpoint:
    runs/tail_risk_gated_dual_head_t0_5s_k5/best_overall.pt

Tail thresholds:
    runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json

Outputs
-------
Fig_4_11_spatial_case_studies.png
Fig_4_11_spatial_case_studies.svg
Fig_4_11_spatial_case_studies.pdf
Fig_4_11_selected_case_summary.csv
Fig_4_11_event_repeat_case_statistics.csv

Example: automatic reproducible case selection
----------------------------------------------
python 32_plot_section_4_11_spatial_case_studies_final.py ^
  --predictions "runs\\locked_power_gated_test_gamma3\\locked_test_predictions.csv" ^
  --manifest "data\\scedc\\model_manifests\\scenario_t0_5s_k5.csv" ^
  --checkpoint "runs\\tail_risk_gated_dual_head_t0_5s_k5\\best_overall.pt" ^
  --threshold-json "runs\\tail_gated_compromise_t0_5s_k5\\tail_thresholds_q0.90_t0_5s.json" ^
  --out-dir "figures\\section_4_11" ^
  --dpi 600

Example: reproduce specific cases
---------------------------------
python 32_plot_section_4_11_spatial_case_studies_final.py ^
  --event-a 39463528 ^
  --event-b 15201961 ^
  --out-dir "figures\\section_4_11" ^
  --dpi 600

If an event is specified without --repeat-a/--repeat-b:
- representative event: choose its repeat with lower base overall error;
- high-motion event: choose its repeat with the largest tail gain.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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
    required = [
        "event_id",
        "repeat",
        "magnitude",
        "input_station_ids",
        "true_log10_pga",
        "true_log10_pgv",
        "base_log10_pga",
        "base_log10_pgv",
        "power_gate_log10_pga",
        "power_gate_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            "Prediction CSV is missing columns:\n"
            + "\n".join(
                missing
            )
        )

    working = dataframe.copy()

    working[
        "is_tail_pga"
    ] = as_bool(
        working[
            "is_tail_pga"
        ]
    )

    working[
        "is_tail_pgv"
    ] = as_bool(
        working[
            "is_tail_pgv"
        ]
    )

    rows = []

    for (
        event_id,
        repeat,
    ), group in working.groupby(
        [
            "event_id",
            "repeat",
        ],
        sort=False,
    ):
        true_pga = group[
            "true_log10_pga"
        ].to_numpy(
            dtype=float
        )

        true_pgv = group[
            "true_log10_pgv"
        ].to_numpy(
            dtype=float
        )

        base_pga = group[
            "base_log10_pga"
        ].to_numpy(
            dtype=float
        )

        base_pgv = group[
            "base_log10_pgv"
        ].to_numpy(
            dtype=float
        )

        power_pga = group[
            "power_gate_log10_pga"
        ].to_numpy(
            dtype=float
        )

        power_pgv = group[
            "power_gate_log10_pgv"
        ].to_numpy(
            dtype=float
        )

        tail_pga = group[
            "is_tail_pga"
        ].to_numpy(
            dtype=bool
        )

        tail_pgv = group[
            "is_tail_pgv"
        ].to_numpy(
            dtype=bool
        )

        base_overall = 0.5 * (
            np.mean(
                np.abs(
                    base_pga
                    - true_pga
                )
            )
            + np.mean(
                np.abs(
                    base_pgv
                    - true_pgv
                )
            )
        )

        power_overall = 0.5 * (
            np.mean(
                np.abs(
                    power_pga
                    - true_pga
                )
            )
            + np.mean(
                np.abs(
                    power_pgv
                    - true_pgv
                )
            )
        )

        base_tail_errors = []
        power_tail_errors = []

        if tail_pga.any():
            base_tail_errors.extend(
                np.abs(
                    base_pga[
                        tail_pga
                    ]
                    - true_pga[
                        tail_pga
                    ]
                )
            )

            power_tail_errors.extend(
                np.abs(
                    power_pga[
                        tail_pga
                    ]
                    - true_pga[
                        tail_pga
                    ]
                )
            )

        if tail_pgv.any():
            base_tail_errors.extend(
                np.abs(
                    base_pgv[
                        tail_pgv
                    ]
                    - true_pgv[
                        tail_pgv
                    ]
                )
            )

            power_tail_errors.extend(
                np.abs(
                    power_pgv[
                        tail_pgv
                    ]
                    - true_pgv[
                        tail_pgv
                    ]
                )
            )

        base_tail_mae = (
            float(
                np.mean(
                    base_tail_errors
                )
            )
            if base_tail_errors
            else np.nan
        )

        power_tail_mae = (
            float(
                np.mean(
                    power_tail_errors
                )
            )
            if power_tail_errors
            else np.nan
        )

        magnitude = pd.to_numeric(
            group[
                "magnitude"
            ].iloc[0],
            errors="coerce",
        )

        rows.append(
            {
                "event_id": str(
                    event_id
                ),
                "repeat": int(
                    repeat
                ),
                "magnitude": (
                    float(
                        magnitude
                    )
                    if np.isfinite(
                        magnitude
                    )
                    else np.nan
                ),
                "n_tail": int(
                    (
                        tail_pga
                        | tail_pgv
                    ).sum()
                ),
                "base_overall": float(
                    base_overall
                ),
                "power_overall": float(
                    power_overall
                ),
                "base_tail": (
                    base_tail_mae
                ),
                "power_tail": (
                    power_tail_mae
                ),
                "tail_gain": (
                    base_tail_mae
                    - power_tail_mae
                    if np.isfinite(
                        base_tail_mae
                    )
                    and np.isfinite(
                        power_tail_mae
                    )
                    else np.nan
                ),
                "input_station_ids": str(
                    group[
                        "input_station_ids"
                    ].iloc[0]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


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

def station_indices(
    station_ids: np.ndarray,
    station_text: str,
) -> np.ndarray:
    lookup = {
        str(
            station
        ): index
        for index, station
        in enumerate(
            station_ids
        )
    }

    selected_ids = [
        item.strip()
        for item in str(
            station_text
        ).split("|")
        if item.strip()
    ]

    missing = [
        station
        for station
        in selected_ids
        if station
        not in lookup
    ]

    if missing:
        raise ValueError(
            "Input station IDs not found in HDF5: "
            f"{missing}"
        )

    return np.asarray(
        [
            lookup[
                station
            ]
            for station
            in selected_ids
        ],
        dtype=int,
    )


def infer_all_heldout(
    pair: pd.Series,
    manifest_row: pd.Series,
    model,
    evaluation_module,
    device: torch.device,
    t0_sec: float,
    input_pre_sec: float,
    gate_power: float,
    thresholds: np.ndarray,
) -> dict:
    h5_path = Path(
        str(
            manifest_row[
                "h5_path"
            ]
        )
    )

    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 not found: "
            f"{h5_path}"
        )

    with h5py.File(
        h5_path,
        "r",
    ) as h5:
        acceleration = np.asarray(
            h5[
                "acceleration"
            ][:],
            dtype=np.float32,
        )

        velocity = np.asarray(
            h5[
                "velocity"
            ][:],
            dtype=np.float32,
        )

        coordinates = np.asarray(
            h5[
                "station_coords"
            ][:],
            dtype=np.float32,
        )

        p_offset = np.asarray(
            h5[
                "p_offset_sec"
            ][:],
            dtype=np.float32,
        )

        station_ids = (
            h5[
                "station_id"
            ]
            .asstr()
            [:]
        )

        sampling_rate_hz = float(
            h5.attrs[
                "sampling_rate_hz"
            ]
        )

        pre_first_p_sec = float(
            h5.attrs[
                "pre_first_p_sec"
            ]
        )

        time_zero_index = int(
            h5.attrs[
                "time_zero_index"
            ]
        )

    input_indices = station_indices(
        station_ids,
        pair[
            "input_station_ids"
        ],
    )

    target_indices = np.setdiff1d(
        np.arange(
            len(
                station_ids
            )
        ),
        input_indices,
    )

    input_start_index = max(
        0,
        int(
            round(
                (
                    pre_first_p_sec
                    - input_pre_sec
                )
                * sampling_rate_hz
            )
        ),
    )

    snapshot_index = min(
        time_zero_index
        + int(
            round(
                t0_sec
                * sampling_rate_hz
            )
        ),
        acceleration.shape[-1]
        - 1,
    )

    (
        input_waveforms,
        input_features,
        target_features,
    ) = (
        evaluation_module
        .prepare_model_inputs(
            acceleration=acceleration,
            coordinates=coordinates,
            p_offset=p_offset,
            input_indices=input_indices,
            target_indices=target_indices,
            input_start_index=(
                input_start_index
            ),
            snapshot_index=(
                snapshot_index
            ),
            t0_sec=t0_sec,
        )
    )

    true_pga = np.log10(
        np.maximum(
            evaluation_module
            .horizontal_peak(
                acceleration[
                    target_indices,
                    :,
                    snapshot_index:,
                ]
            ),
            1.0e-10,
        )
    )

    true_pgv = np.log10(
        np.maximum(
            evaluation_module
            .horizontal_peak(
                velocity[
                    target_indices,
                    :,
                    snapshot_index:,
                ]
            ),
            1.0e-12,
        )
    )

    truth = np.column_stack(
        [
            true_pga,
            true_pgv,
        ]
    )

    with torch.inference_mode():
        output = model(
            torch.from_numpy(
                input_waveforms
            )[None].to(
                device
            ),
            torch.from_numpy(
                input_features
            )[None].to(
                device
            ),
            torch.from_numpy(
                target_features
            )[None].to(
                device
            ),
        )

        base = (
            output[
                "base"
            ][0]
            .cpu()
            .numpy()
        )

        probability = (
            output[
                "probability"
            ][0]
            .cpu()
            .numpy()
        )

        residual = (
            output[
                "residual"
            ][0]
            .cpu()
            .numpy()
        )

        power = (
            base
            + (
                probability
                ** gate_power
            )
            * residual
        )

    (
        epicenter_latitude,
        epicenter_longitude,
    ) = detect_epicenter(
        manifest_row
    )

    if (
        epicenter_latitude
        is None
        or epicenter_longitude
        is None
    ):
        origin_latitude = float(
            coordinates[
                :,
                0,
            ].mean()
        )

        origin_longitude = float(
            coordinates[
                :,
                1,
            ].mean()
        )

        epicenter_xy = None
    else:
        origin_latitude = (
            epicenter_latitude
        )

        origin_longitude = (
            epicenter_longitude
        )

        epicenter_xy = np.asarray(
            [
                0.0,
                0.0,
            ]
        )

    xy = local_xy_km(
        coordinates,
        origin_latitude,
        origin_longitude,
    )

    tail = (
        truth
        >= thresholds[
            None,
            :
        ]
    )

    target_triggered = (
        np.isfinite(
            p_offset[
                target_indices
            ]
        )
        & (
            p_offset[
                target_indices
            ]
            <= t0_sec
        )
    )

    def metrics(
        prediction: np.ndarray,
    ) -> dict:
        residual_error = (
            prediction
            - truth
        )

        absolute_error = np.abs(
            residual_error
        )

        output_metrics = {}

        for index, quantity in enumerate(
            [
                "pga",
                "pgv",
            ]
        ):
            output_metrics[
                f"{quantity}_mae"
            ] = float(
                absolute_error[
                    :,
                    index,
                ].mean()
            )

            tail_mask = tail[
                :,
                index,
            ]

            output_metrics[
                f"{quantity}_tail_n"
            ] = int(
                tail_mask.sum()
            )

            output_metrics[
                f"{quantity}_tail_mae"
            ] = (
                float(
                    absolute_error[
                        tail_mask,
                        index,
                    ].mean()
                )
                if tail_mask.any()
                else np.nan
            )

            output_metrics[
                f"{quantity}_bias"
            ] = float(
                residual_error[
                    :,
                    index,
                ].mean()
            )

        return output_metrics

    magnitude = pd.to_numeric(
        manifest_row.get(
            "magnitude",
            np.nan,
        ),
        errors="coerce",
    )

    return {
        "event_id": str(
            pair[
                "event_id"
            ]
        ),
        "repeat": int(
            pair[
                "repeat"
            ]
        ),
        "magnitude": (
            float(
                magnitude
            )
            if np.isfinite(
                magnitude
            )
            else np.nan
        ),
        "xy_target": xy[
            target_indices
        ],
        "xy_input": xy[
            input_indices
        ],
        "epicenter": (
            epicenter_xy
        ),
        "truth": truth,
        "base": base,
        "power": power,
        "probability": probability,
        "tail": tail,
        "target_triggered": (
            target_triggered
        ),
        "base_metrics": metrics(
            base
        ),
        "power_metrics": metrics(
            power
        ),
        "station_ids_target": [
            str(
                station_ids[
                    index
                ]
            )
            for index
            in target_indices
        ],
    }


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

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
                "power",
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

    parser.add_argument(
        "--predictions",
        default=(
            'runs/locked_power_gated_test_gamma3/locked_test_predictions.csv'
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            r"model_manifests"
            r"\scenario_t0_5s_k5.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            'runs/tail_risk_gated_dual_head_t0_5s_k5/best_overall.pt'
        ),
    )

    parser.add_argument(
        "--threshold-json",
        default=(
            'runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json'
        ),
    )

    parser.add_argument(
        "--training-script",
        default=(
            "18_train_tail_risk_gated_dual_head.py"
        ),
    )

    parser.add_argument(
        "--evaluation-script",
        default=(
            "20_evaluate_locked_power_gated_test.py"
        ),
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
        "--gate-power",
        type=float,
        default=3.0,
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

    # No fixed color map is imposed by default.
    # Pass --cmap <name> if a specific map is required.
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
            'figures/section_4_11'
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

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    prediction_path = Path(
        args.predictions
    )

    manifest_path = Path(
        args.manifest
    )

    threshold_path = Path(
        args.threshold_json
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    for path, name in (
        (
            prediction_path,
            "predictions",
        ),
        (
            manifest_path,
            "manifest",
        ),
        (
            threshold_path,
            "tail-threshold JSON",
        ),
        (
            checkpoint_path,
            "checkpoint",
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: "
                f"{path.resolve()}"
            )

    predictions = pd.read_csv(
        prediction_path,
        dtype={
            "event_id": str
        },
    )

    statistics = build_case_stats(
        predictions
    )

    statistics_path = (
        out_dir
        / "Fig_4_11_event_repeat_case_statistics_submission.csv"
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

    manifest = (
        pd.read_csv(
            manifest_path,
            dtype={
                "event_id": str
            },
        )
        .drop_duplicates(
            "event_id"
        )
        .set_index(
            "event_id"
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
                f"{pair_name} event "
                f"{event_id} is not in manifest."
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

    evaluation_module = load_module(
        args.evaluation_script,
        "locked_eval_4_11",
    )

    training_module = (
        evaluation_module
        .load_python_module(
            args.training_script
        )
    )

    (
        model,
        checkpoint,
        architecture,
    ) = (
        evaluation_module
        .load_locked_model(
            training_module,
            str(
                checkpoint_path
            ),
            threshold_data,
            device,
        )
    )

    representative_case = (
        infer_all_heldout(
            representative_pair,
            manifest.loc[
                str(
                    representative_pair[
                        "event_id"
                    ]
                )
            ],
            model,
            evaluation_module,
            device,
            t0_sec=(
                args.t0_sec
            ),
            input_pre_sec=(
                args.input_pre_sec
            ),
            gate_power=(
                args.gate_power
            ),
            thresholds=(
                thresholds
            ),
        )
    )

    high_motion_case = (
        infer_all_heldout(
            high_motion_pair,
            manifest.loc[
                str(
                    high_motion_pair[
                        "event_id"
                    ]
                )
            ],
            model,
            evaluation_module,
            device,
            t0_sec=(
                args.t0_sec
            ),
            input_pre_sec=(
                args.input_pre_sec
            ),
            gate_power=(
                args.gate_power
            ),
            thresholds=(
                thresholds
            ),
        )
    )

    cases = [
        representative_case,
        representative_case,
        high_motion_case,
        high_motion_case,
    ]

    pga_low, pga_high = (
        shared_limits(
            [
                representative_case,
                high_motion_case,
            ],
            0,
        )
    )

    pgv_low, pgv_high = (
        shared_limits(
            [
                representative_case,
                high_motion_case,
            ],
            1,
        )
    )

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
            "Frozen base",
        ),
        (
            "power",
            (
                "Power gate "
                rf"($\gamma={args.gate_power:g}$)"
            ),
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

            quantity_index = (
                quantity_indices[
                    column_index
                ]
            )

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
                "power",
            ):
                annotation = (
                    metric_annotation(
                        case[
                            f"{key}_metrics"
                        ],
                        quantity,
                    )
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
                        if quantity_index
                        == 0
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
        position = (
            axes[
                row_index,
                0,
            ]
            .get_position()
        )

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
        (
            f"M"
            f"{representative_case['magnitude']:.1f}"
        )
        if np.isfinite(
            representative_case[
                "magnitude"
            ]
        )
        else ""
    )

    high_motion_magnitude = (
        (
            f"M"
            f"{high_motion_case['magnitude']:.1f}"
        )
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

    # Leave room for four unambiguous colorbars.
    figure.subplots_adjust(
        left=0.072,
        right=0.988,
        bottom=0.145,
        top=0.92,
        wspace=0.06,
        hspace=0.055,
    )

    # One colorbar per column; PGA uses one shared scale across the two cases
    # and PGV uses one shared scale across the two cases.
    for column_index in range(
        4
    ):
        quantity_index = (
            quantity_indices[
                column_index
            ]
        )

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

    # No long explanatory sentence is drawn inside the figure.
    # Put the following in the manuscript caption instead:
    #
    # "Filled surfaces are triangulated from actual held-out station
    # queries for visualization only; all quantitative metrics are
    # computed at the original station locations."

    png_path = (
        out_dir
        / "Fig_4_11_spatial_case_studies_submission.png"
    )

    svg_path = (
        out_dir
        / "Fig_4_11_spatial_case_studies_submission.svg"
    )

    pdf_path = (
        out_dir
        / "Fig_4_11_spatial_case_studies_submission.pdf"
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

    summary = pd.DataFrame(
        [
            {
                "case": (
                    "representative"
                ),
                "event_id": (
                    representative_case[
                        "event_id"
                    ]
                ),
                "repeat": (
                    representative_case[
                        "repeat"
                    ]
                ),
                "magnitude": (
                    representative_case[
                        "magnitude"
                    ]
                ),
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
                    (
                        "base_"
                        + key
                    ): value
                    for key, value
                    in representative_case[
                        "base_metrics"
                    ].items()
                },
                **{
                    (
                        "power_"
                        + key
                    ): value
                    for key, value
                    in representative_case[
                        "power_metrics"
                    ].items()
                },
            },
            {
                "case": (
                    "high_motion_target"
                ),
                "event_id": (
                    high_motion_case[
                        "event_id"
                    ]
                ),
                "repeat": (
                    high_motion_case[
                        "repeat"
                    ]
                ),
                "magnitude": (
                    high_motion_case[
                        "magnitude"
                    ]
                ),
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
                    (
                        "base_"
                        + key
                    ): value
                    for key, value
                    in high_motion_case[
                        "base_metrics"
                    ].items()
                },
                **{
                    (
                        "power_"
                        + key
                    ): value
                    for key, value
                    in high_motion_case[
                        "power_metrics"
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
            "power_pga_mae"
        ]
    )

    summary[
        "pgv_overall_gain"
    ] = (
        summary[
            "base_pgv_mae"
        ]
        - summary[
            "power_pgv_mae"
        ]
    )

    summary[
        "pga_tail_gain"
    ] = (
        summary[
            "base_pga_tail_mae"
        ]
        - summary[
            "power_pga_tail_mae"
        ]
    )

    summary[
        "pgv_tail_gain"
    ] = (
        summary[
            "base_pgv_tail_mae"
        ]
        - summary[
            "power_pgv_tail_mae"
        ]
    )

    summary_path = (
        out_dir
        / "Fig_4_11_selected_case_summary_submission.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== Section 4.11 selected cases ==="
    )

    print(
        f"Device      : {device}"
    )

    print(
        f"Gate power  : {args.gate_power:g}"
    )

    print(
        f"Checkpoint  : {checkpoint_path.resolve()}"
    )

    print(
        f"Architecture: {architecture}"
    )

    print(
        "\nSelection summary:"
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
    ):
        print(
            path.resolve()
        )


if __name__ == "__main__":
    main()
