#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
64_cadrg_warning_lead_time_analysis.py

CA-DRG Warning Lead-Time analysis.

Compared models:
    A2 = frozen Cross-Attention Base
    A6 = full CA-DRG

Input:
    locked_dual_risk_predictions.csv from script 61b

Lead-time definitions are identical to the old Power-Gate analysis:
    lead_to_pga_peak_sec
    lead_to_pgv_peak_sec
    lead_to_s_arrival_sec
    pga_highmotion_crossing_lead_sec
    pgv_highmotion_crossing_lead_sec

Primary aggregation:
    targets -> repeats -> events

Fast mode:
    Reuse the old unique_event_station_lead_times.csv because physical
    lead times depend only on the event/target waveforms, not on the model.

Fallback mode:
    Recompute all physical lead times from HDF5.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)


def bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {
        "true", "false", "1", "0",
        "yes", "no", "y", "n", "t", "f",
    }

    unknown = set(normalized.unique()).difference(allowed)
    if unknown:
        raise ValueError(
            f"Cannot parse boolean values: {sorted(unknown)[:20]}"
        )

    return normalized.isin(
        {"true", "1", "yes", "y", "t"}
    ).to_numpy(dtype=bool)


def read_strings(
    h5: h5py.File,
    name: str,
    n: int,
) -> np.ndarray:
    if name not in h5:
        return np.asarray([""] * n, dtype=object)

    try:
        return h5[name].asstr()[:]
    except Exception:
        return np.asarray(
            [str(x) for x in h5[name][:]],
            dtype=object,
        )


def resolve_h5(
    event_id: str,
    row: pd.Series,
    h5_root: Path | None,
) -> Path:
    raw = str(row.get("h5_path", "")).strip()

    if raw:
        p = Path(raw)
        if p.exists():
            return p

    if h5_root is not None:
        candidates = [
            h5_root / f"{event_id}.h5",
            h5_root / str(event_id) / f"{event_id}.h5",
        ]

        for p in candidates:
            if p.exists():
                return p

    raise FileNotFoundError(
        f"Cannot resolve HDF5 for event {event_id}; "
        f"manifest h5_path={raw!r}, h5_root={h5_root}"
    )


def horizontal_series(
    x: np.ndarray,
) -> np.ndarray:
    return np.sqrt(
        x[0].astype(np.float64) ** 2
        + x[1].astype(np.float64) ** 2
    )


def first_crossing(
    series: np.ndarray,
    threshold: float,
    fs: float,
) -> float:
    indices = np.flatnonzero(
        series >= float(threshold)
    )

    if len(indices) == 0:
        return np.nan

    return float(indices[0] / fs)


def physical_leads(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    p_offset: np.ndarray,
    s_offset: np.ndarray,
    station_index: int,
    snapshot: int,
    fs: float,
    t0_sec: float,
    pga_threshold: float,
    pgv_threshold: float,
    end_guard_sec: float,
) -> dict[str, Any]:
    acc = horizontal_series(
        acceleration[
            station_index,
            :,
            snapshot:,
        ]
    )

    vel = horizontal_series(
        velocity[
            station_index,
            :,
            snapshot:,
        ]
    )

    if len(acc) == 0 or len(vel) == 0:
        raise ValueError(
            "Empty post-snapshot waveform."
        )

    pga_index = int(np.argmax(acc))
    pgv_index = int(np.argmax(vel))

    pga = float(acc[pga_index])
    pgv = float(vel[pgv_index])

    pga_peak_lead = float(
        pga_index / fs
    )

    pgv_peak_lead = float(
        pgv_index / fs
    )

    duration = float(
        (len(acc) - 1) / fs
    )

    po = (
        float(p_offset[station_index])
        if np.isfinite(
            p_offset[station_index]
        )
        else np.nan
    )

    so = (
        float(s_offset[station_index])
        if np.isfinite(
            s_offset[station_index]
        )
        else np.nan
    )

    return {
        "future_pga_mps2_recomputed": (
            pga
        ),
        "future_pgv_mps_recomputed": (
            pgv
        ),
        "true_log10_pga_recomputed": float(
            np.log10(
                max(
                    pga,
                    1e-10,
                )
            )
        ),
        "true_log10_pgv_recomputed": float(
            np.log10(
                max(
                    pgv,
                    1e-12,
                )
            )
        ),
        "lead_to_pga_peak_sec": (
            pga_peak_lead
        ),
        "lead_to_pgv_peak_sec": (
            pgv_peak_lead
        ),
        "target_p_offset_sec_h5": (
            po
        ),
        "target_s_offset_sec_h5": (
            so
        ),
        "lead_to_p_arrival_sec": (
            po - t0_sec
            if np.isfinite(po)
            else np.nan
        ),
        "lead_to_s_arrival_sec": (
            so - t0_sec
            if np.isfinite(so)
            else np.nan
        ),
        "pga_highmotion_crossing_lead_sec": (
            first_crossing(
                acc,
                pga_threshold,
                fs,
            )
        ),
        "pgv_highmotion_crossing_lead_sec": (
            first_crossing(
                vel,
                pgv_threshold,
                fs,
            )
        ),
        "future_window_duration_sec": (
            duration
        ),
        "pga_peak_near_window_end": bool(
            duration
            - pga_peak_lead
            <= end_guard_sec
        ),
        "pgv_peak_near_window_end": bool(
            duration
            - pgv_peak_lead
            <= end_guard_sec
        ),
    }


def compute_unique_leads(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    h5_root: Path | None,
    t0_sec: float,
    pga_threshold: float,
    pgv_threshold: float,
    end_guard_sec: float,
) -> pd.DataFrame:
    pairs = (
        predictions[
            [
                "event_id",
                "target_station_index",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    physical_rows = []
    n_events = pairs[
        "event_id"
    ].nunique()

    for event_number, (
        event_id,
        group,
    ) in enumerate(
        pairs.groupby(
            "event_id",
            sort=False,
        ),
        start=1,
    ):
        if event_id not in manifest.index:
            raise KeyError(
                f"Event {event_id} not found in manifest."
            )

        manifest_row = (
            manifest.loc[event_id]
        )

        h5_path = resolve_h5(
            event_id,
            manifest_row,
            h5_root,
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

            p_offset = np.asarray(
                h5[
                    "p_offset_sec"
                ][:],
                dtype=np.float64,
            )

            s_offset = (
                np.asarray(
                    h5[
                        "s_offset_sec"
                    ][:],
                    dtype=np.float64,
                )
                if "s_offset_sec" in h5
                else np.full(
                    len(p_offset),
                    np.nan,
                )
            )

            station_ids = (
                read_strings(
                    h5,
                    "station_id",
                    len(p_offset),
                )
            )

            fs = float(
                h5.attrs[
                    "sampling_rate_hz"
                ]
            )

            zero = int(
                h5.attrs[
                    "time_zero_index"
                ]
            )

        snapshot = min(
            zero
            + int(
                round(
                    t0_sec
                    * fs
                )
            ),
            acceleration.shape[
                -1
            ]
            - 1,
        )

        for row in group.itertuples(
            index=False
        ):
            station_index = int(
                row.target_station_index
            )

            if (
                station_index < 0
                or station_index
                >= len(p_offset)
            ):
                raise IndexError(
                    f"Event {event_id}: "
                    f"station index "
                    f"{station_index} "
                    f"out of range."
                )

            physical_rows.append(
                {
                    "event_id": str(
                        event_id
                    ),
                    "target_station_index": (
                        station_index
                    ),
                    "target_station_id": str(
                        station_ids[
                            station_index
                        ]
                    ).strip(),
                    "h5_path_resolved": str(
                        h5_path
                    ),
                    "sampling_rate_hz": (
                        fs
                    ),
                    **physical_leads(
                        acceleration=(
                            acceleration
                        ),
                        velocity=(
                            velocity
                        ),
                        p_offset=(
                            p_offset
                        ),
                        s_offset=(
                            s_offset
                        ),
                        station_index=(
                            station_index
                        ),
                        snapshot=(
                            snapshot
                        ),
                        fs=(
                            fs
                        ),
                        t0_sec=(
                            t0_sec
                        ),
                        pga_threshold=(
                            pga_threshold
                        ),
                        pgv_threshold=(
                            pgv_threshold
                        ),
                        end_guard_sec=(
                            end_guard_sec
                        ),
                    ),
                }
            )

        if (
            event_number % 25 == 0
            or event_number
            == n_events
        ):
            print(
                "Processed events: "
                f"{event_number}/"
                f"{n_events} | "
                "unique pairs="
                f"{len(physical_rows):,}"
            )

    return pd.DataFrame(
        physical_rows
    )


def make_bins(
    values: pd.Series,
    edges: list[float],
) -> pd.Series:
    if (
        not edges
        or edges[0]
        != 0.0
    ):
        raise ValueError(
            "Lead-bin edges must begin with 0."
        )

    bins = edges + [
        np.inf
    ]

    labels = [
        (
            f"{edges[i]:g}-"
            f"{edges[i + 1]:g}s"
        )
        for i in range(
            len(edges) - 1
        )
    ] + [
        f">={edges[-1]:g}s"
    ]

    return pd.cut(
        values,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )


def element_metric(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> np.ndarray:
    residual = (
        prediction
        - truth
    )

    absolute = np.abs(
        residual
    )

    if metric == "mae":
        return absolute

    if metric == "bias":
        return residual

    if metric == "factor2":
        return (
            absolute
            <= LOG10_FACTOR_2
        ).astype(
            float
        )

    if metric == "factor3":
        return (
            absolute
            <= LOG10_FACTOR_3
        ).astype(
            float
        )

    if metric == "under03":
        return (
            residual
            <= -0.3
        ).astype(
            float
        )

    if metric == "under05":
        return (
            residual
            <= -0.5
        ).astype(
            float
        )

    raise ValueError(
        metric
    )


def canonical_metric(
    frame: pd.DataFrame,
    truth_col: str,
    prediction_col: str,
    metric: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "value": np.nan,
            "n_target_rows": 0,
            "n_event_repeats": 0,
            "n_events": 0,
            "n_unique_event_station_pairs": 0,
        }

    truth = frame[
        truth_col
    ].to_numpy(
        float
    )

    prediction = frame[
        prediction_col
    ].to_numpy(
        float
    )

    work = frame[
        [
            "event_id",
            "repeat",
            "target_station_index",
        ]
    ].copy()

    work[
        "value"
    ] = element_metric(
        truth,
        prediction,
        metric,
    )

    event_repeat = (
        work.groupby(
            [
                "event_id",
                "repeat",
            ],
            sort=False,
        )[
            "value"
        ]
        .mean()
        .reset_index()
    )

    event = (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )[
            "value"
        ]
        .mean()
    )

    return {
        "value": float(
            event.mean()
        ),
        "n_target_rows": int(
            len(frame)
        ),
        "n_event_repeats": int(
            len(event_repeat)
        ),
        "n_events": int(
            len(event)
        ),
        "n_unique_event_station_pairs": int(
            frame[
                [
                    "event_id",
                    "target_station_index",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
    }


def metrics_by_peak_lead(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    specs = [
        (
            "pga",
            "lead_to_pga_peak_sec",
            "pga_peak_lead_bin",
        ),
        (
            "pgv",
            "lead_to_pgv_peak_sec",
            "pgv_peak_lead_bin",
        ),
    ]

    models = {
        "cross_attention": (
            "base"
        ),
        "ca_drg": (
            "final"
        ),
    }

    metrics = [
        "mae",
        "bias",
        "factor2",
        "factor3",
        "under03",
        "under05",
    ]

    for (
        quantity,
        lead_col,
        bin_col,
    ) in specs:
        populations = {
            "overall": np.ones(
                len(frame),
                dtype=bool,
            ),
            "high_motion_tail": (
                bool_array(
                    frame[
                        f"is_tail_{quantity}"
                    ]
                )
            ),
        }

        for (
            population,
            mask,
        ) in populations.items():
            subset = frame.loc[
                mask
            ].copy()

            for (
                bin_name,
                group,
            ) in subset.groupby(
                bin_col,
                observed=True,
                sort=False,
            ):
                for (
                    model_name,
                    prefix,
                ) in models.items():
                    for metric in metrics:
                        result = (
                            canonical_metric(
                                group,
                                f"true_log10_{quantity}",
                                f"{prefix}_log10_{quantity}",
                                metric,
                            )
                        )

                        rows.append(
                            {
                                "quantity": (
                                    quantity
                                ),
                                "population": (
                                    population
                                ),
                                "lead_definition": (
                                    lead_col
                                ),
                                "lead_bin": str(
                                    bin_name
                                ),
                                "model": (
                                    model_name
                                ),
                                "metric": (
                                    metric
                                ),
                                "median_lead_sec": float(
                                    group[
                                        lead_col
                                    ].median()
                                ),
                                **result,
                            }
                        )

    return pd.DataFrame(
        rows
    )


def paired_deltas_by_peak_lead(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    specs = [
        (
            "pga",
            "pga_peak_lead_bin",
        ),
        (
            "pgv",
            "pgv_peak_lead_bin",
        ),
    ]

    for (
        quantity,
        bin_col,
    ) in specs:
        tail = frame.loc[
            bool_array(
                frame[
                    f"is_tail_{quantity}"
                ]
            )
        ].copy()

        for (
            bin_name,
            group,
        ) in tail.groupby(
            bin_col,
            observed=True,
            sort=False,
        ):
            truth = group[
                f"true_log10_{quantity}"
            ].to_numpy(
                float
            )

            base_prediction = group[
                f"base_log10_{quantity}"
            ].to_numpy(
                float
            )

            final_prediction = group[
                f"final_log10_{quantity}"
            ].to_numpy(
                float
            )

            for metric in [
                "mae",
                "bias",
                "factor2",
                "under05",
            ]:
                base_values = (
                    element_metric(
                        truth,
                        base_prediction,
                        metric,
                    )
                )

                final_values = (
                    element_metric(
                        truth,
                        final_prediction,
                        metric,
                    )
                )

                work = group[
                    [
                        "event_id",
                        "repeat",
                    ]
                ].copy()

                work[
                    "base"
                ] = base_values

                work[
                    "ca_drg"
                ] = final_values

                event_repeat = (
                    work.groupby(
                        [
                            "event_id",
                            "repeat",
                        ],
                        sort=False,
                    )[
                        [
                            "base",
                            "ca_drg",
                        ]
                    ]
                    .mean()
                )

                event = (
                    event_repeat.groupby(
                        level=(
                            "event_id"
                        )
                    )[
                        [
                            "base",
                            "ca_drg",
                        ]
                    ]
                    .mean()
                )

                delta = (
                    event[
                        "ca_drg"
                    ]
                    - event[
                        "base"
                    ]
                )

                base_mean = float(
                    event[
                        "base"
                    ].mean()
                )

                final_mean = float(
                    event[
                        "ca_drg"
                    ].mean()
                )

                mean_delta = float(
                    delta.mean()
                )

                record = {
                    "quantity": (
                        quantity
                    ),
                    "population": (
                        "high_motion_tail"
                    ),
                    "lead_bin": str(
                        bin_name
                    ),
                    "metric": (
                        metric
                    ),
                    "n_paired_events": int(
                        len(delta)
                    ),
                    "base_value": (
                        base_mean
                    ),
                    "ca_drg_value": (
                        final_mean
                    ),
                    "mean_delta_cadrg_minus_base": (
                        mean_delta
                    ),
                    "median_delta_cadrg_minus_base": float(
                        delta.median()
                    ),
                }

                if metric == "mae":
                    record[
                        "relative_change_percent"
                    ] = float(
                        100.0
                        * mean_delta
                        / max(
                            abs(
                                base_mean
                            ),
                            1e-12,
                        )
                    )

                if metric in {
                    "factor2",
                    "under05",
                }:
                    record[
                        "delta_percentage_points"
                    ] = float(
                        100.0
                        * mean_delta
                    )

                rows.append(
                    record
                )

    return pd.DataFrame(
        rows
    )


def distribution_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    definitions = {
        "pga_peak": (
            "lead_to_pga_peak_sec"
        ),
        "pgv_peak": (
            "lead_to_pgv_peak_sec"
        ),
        "s_arrival": (
            "lead_to_s_arrival_sec"
        ),
        "pga_highmotion_crossing": (
            "pga_highmotion_crossing_lead_sec"
        ),
        "pgv_highmotion_crossing": (
            "pgv_highmotion_crossing_lead_sec"
        ),
    }

    for quantity in (
        "pga",
        "pgv",
    ):
        populations = {
            "overall": np.ones(
                len(frame),
                dtype=bool,
            ),
            "high_motion_tail": (
                bool_array(
                    frame[
                        f"is_tail_{quantity}"
                    ]
                )
            ),
        }

        for (
            population,
            mask,
        ) in populations.items():
            subset = frame.loc[
                mask
            ]

            for (
                definition,
                column,
            ) in definitions.items():
                values = pd.to_numeric(
                    subset[
                        column
                    ],
                    errors="coerce",
                ).to_numpy(
                    float
                )

                values = values[
                    np.isfinite(
                        values
                    )
                ]

                if len(
                    values
                ) == 0:
                    continue

                rows.append(
                    {
                        "quantity": (
                            quantity
                        ),
                        "population": (
                            population
                        ),
                        "lead_definition": (
                            definition
                        ),
                        "n_target_rows_total": int(
                            len(
                                subset
                            )
                        ),
                        "n_with_finite_lead": int(
                            len(
                                values
                            )
                        ),
                        "finite_fraction": float(
                            len(
                                values
                            )
                            / max(
                                len(
                                    subset
                                ),
                                1,
                            )
                        ),
                        "median_sec": float(
                            np.median(
                                values
                            )
                        ),
                        "p25_sec": float(
                            np.percentile(
                                values,
                                25,
                            )
                        ),
                        "p75_sec": float(
                            np.percentile(
                                values,
                                75,
                            )
                        ),
                        "p90_sec": float(
                            np.percentile(
                                values,
                                90,
                            )
                        ),
                        "fraction_gt_0s": float(
                            np.mean(
                                values > 0
                            )
                        ),
                        "fraction_ge_2s": float(
                            np.mean(
                                values >= 2
                            )
                        ),
                        "fraction_ge_5s": float(
                            np.mean(
                                values >= 5
                            )
                        ),
                        "fraction_ge_10s": float(
                            np.mean(
                                values >= 10
                            )
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def latency_summary(
    frame: pd.DataFrame,
    latencies: list[float],
) -> pd.DataFrame:
    rows = []

    specs = [
        (
            "pga",
            "lead_to_pga_peak_sec",
            "time_to_remaining_peak",
        ),
        (
            "pgv",
            "lead_to_pgv_peak_sec",
            "time_to_remaining_peak",
        ),
        (
            "pga",
            "pga_highmotion_crossing_lead_sec",
            "time_to_highmotion_threshold_crossing",
        ),
        (
            "pgv",
            "pgv_highmotion_crossing_lead_sec",
            "time_to_highmotion_threshold_crossing",
        ),
    ]

    for (
        quantity,
        column,
        definition,
    ) in specs:
        subset = frame.loc[
            bool_array(
                frame[
                    f"is_tail_{quantity}"
                ]
            )
        ]

        values = pd.to_numeric(
            subset[
                column
            ],
            errors="coerce",
        ).to_numpy(
            float
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        for latency in latencies:
            if len(
                values
            ) == 0:
                continue

            net = (
                values
                - float(
                    latency
                )
            )

            rows.append(
                {
                    "quantity": (
                        quantity
                    ),
                    "population": (
                        "high_motion_tail"
                    ),
                    "lead_definition": (
                        definition
                    ),
                    "assumed_total_latency_sec": float(
                        latency
                    ),
                    "n_target_rows": int(
                        len(
                            subset
                        )
                    ),
                    "n_with_defined_lead": int(
                        len(
                            values
                        )
                    ),
                    "defined_fraction": float(
                        len(
                            values
                        )
                        / max(
                            len(
                                subset
                            ),
                            1,
                        )
                    ),
                    "fraction_positive_net_lead": float(
                        np.mean(
                            net > 0
                        )
                    ),
                    "fraction_net_lead_ge_1s": float(
                        np.mean(
                            net >= 1
                        )
                    ),
                    "fraction_net_lead_ge_2s": float(
                        np.mean(
                            net >= 2
                        )
                    ),
                    "fraction_net_lead_ge_5s": float(
                        np.mean(
                            net >= 5
                        )
                    ),
                    "median_net_lead_sec": float(
                        np.median(
                            net
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def validate_reused_leads(
    predictions: pd.DataFrame,
    physical: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "event_id",
        "target_station_index",
        "lead_to_pga_peak_sec",
        "lead_to_pgv_peak_sec",
        "lead_to_s_arrival_sec",
        "pga_highmotion_crossing_lead_sec",
        "pgv_highmotion_crossing_lead_sec",
        "true_log10_pga_recomputed",
        "true_log10_pgv_recomputed",
        "pga_peak_near_window_end",
        "pgv_peak_near_window_end",
    }

    missing = required.difference(
        physical.columns
    )

    if missing:
        raise ValueError(
            "Reused lead-time file missing columns: "
            f"{sorted(missing)}"
        )

    physical = physical.copy()

    physical[
        "event_id"
    ] = (
        physical[
            "event_id"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    if physical.duplicated(
        [
            "event_id",
            "target_station_index",
        ]
    ).any():
        raise ValueError(
            "Reused lead-time file is not unique on "
            "event_id + target_station_index."
        )

    prediction_pairs = (
        predictions[
            [
                "event_id",
                "target_station_index",
            ]
        ]
        .drop_duplicates()
    )

    paired = prediction_pairs.merge(
        physical[
            [
                "event_id",
                "target_station_index",
            ]
        ],
        on=[
            "event_id",
            "target_station_index",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(
        paired
    ) != len(
        prediction_pairs
    ):
        raise RuntimeError(
            "Reused lead-time file does not cover every "
            "CA-DRG event-station pair: "
            f"{len(paired):,}/"
            f"{len(prediction_pairs):,}."
        )

    return physical


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "runs/"
            "cross_attention_dual_risk_previous_grouped_benchmark/"
            "locked_dual_risk_predictions.csv"
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/model_manifests/"
            "scenario_t0_5s_k5_linux.csv"
        ),
    )

    parser.add_argument(
        "--h5-root",
        default=(
            "data/processed_full_v4/events"
        ),
    )

    parser.add_argument(
        "--threshold-json",
        default=(
            "runs/"
            "tail_gated_compromise_t0_5s_k5/"
            "tail_thresholds_q0.90_t0_5s.json"
        ),
    )

    parser.add_argument(
        "--reuse-lead-times",
        default="",
        help=(
            "Optional old unique_event_station_lead_times.csv. "
            "When supplied, HDF5 lead-time recomputation is skipped."
        ),
    )

    parser.add_argument(
        "--t0-sec",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--lead-bin-edges",
        default="0,1,2,5,10,20",
    )

    parser.add_argument(
        "--latency-seconds",
        default="0,0.5,1,2",
    )

    parser.add_argument(
        "--end-guard-sec",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "cadrg_warning_lead_time"
        ),
    )

    args = parser.parse_args()

    prediction_path = Path(
        args.predictions
    )

    manifest_path = Path(
        args.manifest
    )

    threshold_path = Path(
        args.threshold_json
    )

    h5_root = (
        Path(
            args.h5_root
        )
        if str(
            args.h5_root
        ).strip()
        else None
    )

    for path in [
        prediction_path,
        manifest_path,
        threshold_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        prediction_path,
        dtype={
            "event_id": str,
        },
    )

    required_predictions = {
        "event_id",
        "repeat",
        "target_station_index",
        "true_log10_pga",
        "true_log10_pgv",
        "base_log10_pga",
        "base_log10_pgv",
        "final_log10_pga",
        "final_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
    }

    missing = required_predictions.difference(
        predictions.columns
    )

    if missing:
        raise ValueError(
            "Predictions missing columns: "
            f"{sorted(missing)}"
        )

    predictions[
        "event_id"
    ] = (
        predictions[
            "event_id"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    if predictions.duplicated(
        [
            "event_id",
            "repeat",
            "target_station_index",
        ]
    ).any():
        raise ValueError(
            "Prediction file is not unique on "
            "event_id + repeat + target_station_index."
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "event_id": str,
        },
    )

    manifest[
        "event_id"
    ] = (
        manifest[
            "event_id"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    manifest = (
        manifest
        .drop_duplicates(
            "event_id"
        )
        .set_index(
            "event_id",
            drop=False,
        )
    )

    threshold_data = json.loads(
        threshold_path.read_text(
            encoding="utf-8"
        )
    )

    pga_log_threshold = float(
        threshold_data[
            "log10_pga_threshold"
        ]
    )

    pgv_log_threshold = float(
        threshold_data[
            "log10_pgv_threshold"
        ]
    )

    pga_threshold = (
        10.0
        ** pga_log_threshold
    )

    pgv_threshold = (
        10.0
        ** pgv_log_threshold
    )

    unique_pair_count = int(
        predictions[
            [
                "event_id",
                "target_station_index",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "=== CA-DRG Warning Lead-Time Analysis ==="
    )

    print(
        f"Prediction rows          : "
        f"{len(predictions):,}"
    )

    print(
        f"Events                   : "
        f"{predictions['event_id'].nunique():,}"
    )

    print(
        f"Unique event-station     : "
        f"{unique_pair_count:,}"
    )

    print(
        f"T0                       : "
        f"{args.t0_sec:g} s"
    )

    print(
        "Thresholds               : "
        f"PGA={pga_log_threshold:.6f} "
        f"({pga_threshold:.6g} m/s^2), "
        f"PGV={pgv_log_threshold:.6f} "
        f"({pgv_threshold:.6g} m/s)"
    )

    reuse_path = (
        Path(
            args.reuse_lead_times
        )
        if str(
            args.reuse_lead_times
        ).strip()
        else None
    )

    if reuse_path is not None:
        if not reuse_path.exists():
            raise FileNotFoundError(
                reuse_path
            )

        print(
            "Reusing physical lead times: "
            f"{reuse_path.resolve()}"
        )

        physical = pd.read_csv(
            reuse_path,
            dtype={
                "event_id": str,
            },
        )

        physical = validate_reused_leads(
            predictions,
            physical,
        )

    else:
        print(
            "Recomputing physical lead times from HDF5 ..."
        )

        physical = compute_unique_leads(
            predictions=(
                predictions
            ),
            manifest=(
                manifest
            ),
            h5_root=(
                h5_root
            ),
            t0_sec=(
                args.t0_sec
            ),
            pga_threshold=(
                pga_threshold
            ),
            pgv_threshold=(
                pgv_threshold
            ),
            end_guard_sec=(
                args.end_guard_sec
            ),
        )

    physical_path = (
        out_dir
        / "unique_event_station_lead_times.csv"
    )

    physical.to_csv(
        physical_path,
        index=False,
    )

    merged = predictions.merge(
        physical,
        on=[
            "event_id",
            "target_station_index",
        ],
        how="left",
        validate="many_to_one",
    )

    if (
        merged[
            "lead_to_pga_peak_sec"
        ].isna().all()
    ):
        raise RuntimeError(
            "Physical lead-time merge failed."
        )

    merged[
        "pga_truth_abs_diff"
    ] = np.abs(
        merged[
            "true_log10_pga"
        ].astype(
            float
        )
        - merged[
            "true_log10_pga_recomputed"
        ].astype(
            float
        )
    )

    merged[
        "pgv_truth_abs_diff"
    ] = np.abs(
        merged[
            "true_log10_pgv"
        ].astype(
            float
        )
        - merged[
            "true_log10_pgv_recomputed"
        ].astype(
            float
        )
    )

    max_pga_diff = float(
        merged[
            "pga_truth_abs_diff"
        ].max()
    )

    max_pgv_diff = float(
        merged[
            "pgv_truth_abs_diff"
        ].max()
    )

    print(
        "Truth recomputation max |diff|: "
        f"PGA={max_pga_diff:.3e}, "
        f"PGV={max_pgv_diff:.3e}"
    )

    if max(
        max_pga_diff,
        max_pgv_diff,
    ) > 1e-5:
        mismatch_path = (
            out_dir
            / "truth_recomputation_mismatches.csv"
        )

        merged.loc[
            (
                merged[
                    "pga_truth_abs_diff"
                ]
                > 1e-5
            )
            | (
                merged[
                    "pgv_truth_abs_diff"
                ]
                > 1e-5
            )
        ].head(
            200
        ).to_csv(
            mismatch_path,
            index=False,
        )

        raise RuntimeError(
            "CA-DRG truth does not match HDF5 recomputation. "
            f"See {mismatch_path}"
        )

    edges = sorted(
        set(
            float(
                token.strip()
            )
            for token
            in args.lead_bin_edges.split(
                ","
            )
            if token.strip()
        )
    )

    merged[
        "pga_peak_lead_bin"
    ] = make_bins(
        merged[
            "lead_to_pga_peak_sec"
        ],
        edges,
    )

    merged[
        "pgv_peak_lead_bin"
    ] = make_bins(
        merged[
            "lead_to_pgv_peak_sec"
        ],
        edges,
    )

    merged_path = (
        out_dir
        / "cadrg_predictions_with_lead_times.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
    )

    metric_frame = (
        metrics_by_peak_lead(
            merged
        )
    )

    metric_path = (
        out_dir
        / "metrics_by_peak_lead_time.csv"
    )

    metric_frame.to_csv(
        metric_path,
        index=False,
    )

    delta_frame = (
        paired_deltas_by_peak_lead(
            merged
        )
    )

    delta_path = (
        out_dir
        / "cadrg_minus_cross_attention_by_peak_lead_time.csv"
    )

    delta_frame.to_csv(
        delta_path,
        index=False,
    )

    distribution_frame = (
        distribution_summary(
            merged
        )
    )

    distribution_path = (
        out_dir
        / "lead_time_distribution_summary.csv"
    )

    distribution_frame.to_csv(
        distribution_path,
        index=False,
    )

    latencies = [
        float(
            token.strip()
        )
        for token
        in args.latency_seconds.split(
            ","
        )
        if token.strip()
    ]

    latency_frame = (
        latency_summary(
            merged,
            latencies,
        )
    )

    latency_path = (
        out_dir
        / "latency_sensitivity.csv"
    )

    latency_frame.to_csv(
        latency_path,
        index=False,
    )

    edge_frame = pd.DataFrame(
        [
            {
                "quantity": "pga",
                "n_rows": int(
                    len(
                        merged
                    )
                ),
                "n_peak_near_window_end": int(
                    bool_array(
                        merged[
                            "pga_peak_near_window_end"
                        ]
                    ).sum()
                ),
                "fraction_peak_near_window_end": float(
                    bool_array(
                        merged[
                            "pga_peak_near_window_end"
                        ]
                    ).mean()
                ),
            },
            {
                "quantity": "pgv",
                "n_rows": int(
                    len(
                        merged
                    )
                ),
                "n_peak_near_window_end": int(
                    bool_array(
                        merged[
                            "pgv_peak_near_window_end"
                        ]
                    ).sum()
                ),
                "fraction_peak_near_window_end": float(
                    bool_array(
                        merged[
                            "pgv_peak_near_window_end"
                        ]
                    ).mean()
                ),
            },
        ]
    )

    edge_path = (
        out_dir
        / "future_window_edge_audit.csv"
    )

    edge_frame.to_csv(
        edge_path,
        index=False,
    )

    print(
        "\n=== High-motion lead-time distribution ==="
    )

    show_distribution = (
        distribution_frame.loc[
            (
                distribution_frame[
                    "population"
                ]
                == "high_motion_tail"
            )
            & distribution_frame[
                "lead_definition"
            ].isin(
                [
                    "pga_peak",
                    "pgv_peak",
                    "s_arrival",
                    "pga_highmotion_crossing",
                    "pgv_highmotion_crossing",
                ]
            )
        ]
    )

    print(
        show_distribution[
            [
                "quantity",
                "lead_definition",
                "n_with_finite_lead",
                "median_sec",
                "p25_sec",
                "p75_sec",
                "fraction_ge_2s",
                "fraction_ge_5s",
                "fraction_ge_10s",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== High-motion Tail MAE by time-to-peak bin ==="
    )

    tail_mae = (
        metric_frame.loc[
            (
                metric_frame[
                    "population"
                ]
                == "high_motion_tail"
            )
            & (
                metric_frame[
                    "metric"
                ]
                == "mae"
            )
        ]
    )

    print(
        tail_mae[
            [
                "quantity",
                "lead_bin",
                "model",
                "value",
                "n_events",
                "n_target_rows",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== CA-DRG minus Cross-Attention: Tail MAE by lead bin ==="
    )

    tail_delta_mae = (
        delta_frame.loc[
            delta_frame[
                "metric"
            ]
            == "mae"
        ][
            [
                "quantity",
                "lead_bin",
                "n_paired_events",
                "base_value",
                "ca_drg_value",
                "mean_delta_cadrg_minus_base",
                "relative_change_percent",
            ]
        ]
    )

    print(
        tail_delta_mae.to_string(
            index=False
        )
    )

    print(
        "\n=== HDF5 end-window audit ==="
    )

    print(
        edge_frame.to_string(
            index=False
        )
    )

    audit = {
        "prediction_rows": int(
            len(
                predictions
            )
        ),
        "events": int(
            predictions[
                "event_id"
            ].nunique()
        ),
        "unique_event_station_pairs": (
            unique_pair_count
        ),
        "physical_rows": int(
            len(
                physical
            )
        ),
        "merged_rows": int(
            len(
                merged
            )
        ),
        "max_abs_truth_diff_pga": (
            max_pga_diff
        ),
        "max_abs_truth_diff_pgv": (
            max_pgv_diff
        ),
        "t0_sec": float(
            args.t0_sec
        ),
        "lead_bin_edges": (
            edges
        ),
        "pga_log10_tail_threshold": (
            pga_log_threshold
        ),
        "pgv_log10_tail_threshold": (
            pgv_log_threshold
        ),
        "physical_lead_times_reused": bool(
            reuse_path
            is not None
        ),
        "reused_path": (
            str(
                reuse_path.resolve()
            )
            if reuse_path
            is not None
            else None
        ),
        "models": {
            "A2": (
                "Cross-Attention Base"
            ),
            "A6": (
                "CA-DRG"
            ),
        },
        "metric_aggregation": (
            "targets -> repeats -> events"
        ),
    }

    audit_path = (
        out_dir
        / "lead_time_analysis_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nOutputs:"
    )

    for path in [
        physical_path,
        merged_path,
        distribution_path,
        metric_path,
        delta_path,
        latency_path,
        edge_path,
        audit_path,
    ]:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
