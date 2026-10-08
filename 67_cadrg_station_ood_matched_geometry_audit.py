#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
67_cadrg_station_ood_matched_geometry_audit.py

Post-hoc matched-geometry audit for the strict CA-DRG Station-OOD test.

NO model is retrained and NO model/gamma selection is performed.

Why this audit
--------------
The strict Station-OOD test already uses:
    - the same event,
    - the same repeat,
    - the same K seen input stations,
    - equal seen/unseen target budgets.

The remaining geometric imbalance is that seen and unseen target stations can
have different nearest-input distances and different P-wave-reached states.

Matching variables
------------------
ONLY variables independent of prediction error / PGA/PGV are used:
    1) exact target P-wave-reached state;
    2) nearest-input distance.

This script improves on a forced 3-vs-3 permutation match by using a distance
caliper. Pairs outside the caliper are discarded, so the matched cohort
actually becomes geometrically closer.

Default sensitivity:
    10 km, 20 km, 30 km

The 20-km result is printed as the primary descriptive audit, while all three
calipers are saved so the result is not tied to a single arbitrary threshold.

Metrics
-------
Models:
    Cross-Attention Base
    CA-DRG

Roles:
    seen_target
    unseen_station_target

Populations:
    overall
    high_motion_tail

Metrics:
    MAE
    bias
    Factor2
    Factor3
    U0.5

Canonical aggregation:
    matched targets -> repeats -> events

Important:
- Matching NEVER uses true PGA/PGV, tail label, residual, model prediction,
  magnitude, or any outcome-derived quantity.
- Tail metrics are computed only AFTER matching.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)
SEVERE_UNDER_THRESHOLD = -0.5


def as_bool_scalar(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def as_bool_array(
    series: pd.Series,
) -> np.ndarray:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.to_numpy(
            dtype=bool
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
        .to_numpy(
            dtype=bool
        )
    )


def parse_float_list(
    text: str,
) -> list[float]:
    values = []

    for token in str(text).split(","):
        token = token.strip()

        if token:
            value = float(token)

            if value <= 0:
                raise ValueError(
                    "Distance calipers must be > 0."
                )

            values.append(
                value
            )

    values = sorted(
        set(values)
    )

    if not values:
        raise ValueError(
            "No distance calipers supplied."
        )

    return values


# ---------------------------------------------------------------------
# Exact-state + distance-caliper matching
# ---------------------------------------------------------------------

def best_partial_assignment_one_state(
    seen_positions: list[int],
    unseen_positions: list[int],
    seen_distances: np.ndarray,
    unseen_distances: np.ndarray,
    caliper_km: float,
) -> list[tuple[int, int]]:
    """
    Find a maximum-cardinality, minimum-total-distance matching for one
    P-wave state.

    Since there are only three targets/role/event-repeat in the experiment,
    exhaustive enumeration is trivial and deterministic.
    """
    if (
        len(seen_positions) == 0
        or len(unseen_positions) == 0
    ):
        return []

    maximum_size = min(
        len(seen_positions),
        len(unseen_positions),
    )

    best_pairs = []
    best_key = None

    for size in range(
        maximum_size,
        0,
        -1,
    ):
        found_for_size = False

        for seen_subset in itertools.combinations(
            seen_positions,
            size,
        ):
            for unseen_subset in itertools.combinations(
                unseen_positions,
                size,
            ):
                for unseen_perm in itertools.permutations(
                    unseen_subset
                ):
                    pairs = list(
                        zip(
                            seen_subset,
                            unseen_perm,
                        )
                    )

                    differences = [
                        abs(
                            float(
                                seen_distances[
                                    seen_pos
                                ]
                            )
                            - float(
                                unseen_distances[
                                    unseen_pos
                                ]
                            )
                        )
                        for (
                            seen_pos,
                            unseen_pos,
                        ) in pairs
                    ]

                    if any(
                        difference
                        > float(
                            caliper_km
                        )
                        for difference
                        in differences
                    ):
                        continue

                    found_for_size = True

                    key = (
                        float(
                            sum(
                                differences
                            )
                        ),
                        float(
                            max(
                                differences
                            )
                        ),
                        tuple(
                            pairs
                        ),
                    )

                    if (
                        best_key is None
                        or key < best_key
                    ):
                        best_key = key
                        best_pairs = pairs

        if found_for_size:
            break

    return [
        (
            int(
                seen_pos
            ),
            int(
                unseen_pos
            ),
        )
        for (
            seen_pos,
            unseen_pos,
        ) in best_pairs
    ]


def match_event_repeat(
    seen: pd.DataFrame,
    unseen: pd.DataFrame,
    caliper_km: float,
) -> list[tuple[int, int]]:
    seen_distance = (
        seen[
            "nearest_input_distance_km"
        ].to_numpy(
            dtype=float
        )
    )

    unseen_distance = (
        unseen[
            "nearest_input_distance_km"
        ].to_numpy(
            dtype=float
        )
    )

    seen_state = np.asarray(
        [
            as_bool_scalar(
                value
            )
            for value
            in seen[
                "target_p_wave_reached"
            ]
        ],
        dtype=bool,
    )

    unseen_state = np.asarray(
        [
            as_bool_scalar(
                value
            )
            for value
            in unseen[
                "target_p_wave_reached"
            ]
        ],
        dtype=bool,
    )

    pairs = []

    for state in (
        False,
        True,
    ):
        seen_positions = (
            np.flatnonzero(
                seen_state == state
            )
            .astype(
                int
            )
            .tolist()
        )

        unseen_positions = (
            np.flatnonzero(
                unseen_state == state
            )
            .astype(
                int
            )
            .tolist()
        )

        pairs.extend(
            best_partial_assignment_one_state(
                seen_positions=(
                    seen_positions
                ),
                unseen_positions=(
                    unseen_positions
                ),
                seen_distances=(
                    seen_distance
                ),
                unseen_distances=(
                    unseen_distance
                ),
                caliper_km=(
                    caliper_km
                ),
            )
        )

    return pairs


def build_matched_pairs(
    predictions: pd.DataFrame,
    caliper_km: float,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    rows = []

    total_groups = 0
    groups_with_pair = 0
    groups_full_three_pairs = 0
    groups_zero_pairs = 0

    grouped = predictions.groupby(
        [
            "event_id",
            "repeat",
        ],
        sort=False,
    )

    for (
        event_id,
        repeat,
    ), group in grouped:
        total_groups += 1

        seen = (
            group.loc[
                group[
                    "target_role"
                ].eq(
                    "seen_target"
                )
            ]
            .reset_index(
                drop=True
            )
        )

        unseen = (
            group.loc[
                group[
                    "target_role"
                ].eq(
                    "unseen_station_target"
                )
            ]
            .reset_index(
                drop=True
            )
        )

        if (
            seen.empty
            or unseen.empty
        ):
            groups_zero_pairs += 1
            continue

        assignments = (
            match_event_repeat(
                seen=seen,
                unseen=unseen,
                caliper_km=(
                    caliper_km
                ),
            )
        )

        if len(
            assignments
        ) == 0:
            groups_zero_pairs += 1
            continue

        groups_with_pair += 1

        if len(
            assignments
        ) == min(
            len(
                seen
            ),
            len(
                unseen
            ),
        ):
            groups_full_three_pairs += 1

        for pair_index, (
            seen_position,
            unseen_position,
        ) in enumerate(
            assignments
        ):
            s = seen.iloc[
                seen_position
            ]

            u = unseen.iloc[
                unseen_position
            ]

            s_state = as_bool_scalar(
                s[
                    "target_p_wave_reached"
                ]
            )

            u_state = as_bool_scalar(
                u[
                    "target_p_wave_reached"
                ]
            )

            distance_difference = abs(
                float(
                    s[
                        "nearest_input_distance_km"
                    ]
                )
                - float(
                    u[
                        "nearest_input_distance_km"
                    ]
                )
            )

            record = {
                "caliper_km": float(
                    caliper_km
                ),
                "event_id": str(
                    event_id
                ),
                "repeat": int(
                    repeat
                ),
                "pair_index": int(
                    pair_index
                ),
                "seen_station_id": str(
                    s[
                        "target_station_id"
                    ]
                ),
                "unseen_station_id": str(
                    u[
                        "target_station_id"
                    ]
                ),
                "seen_distance_km": float(
                    s[
                        "nearest_input_distance_km"
                    ]
                ),
                "unseen_distance_km": float(
                    u[
                        "nearest_input_distance_km"
                    ]
                ),
                "absolute_distance_difference_km": float(
                    distance_difference
                ),
                "seen_p_wave_reached": bool(
                    s_state
                ),
                "unseen_p_wave_reached": bool(
                    u_state
                ),
                "p_wave_state_match": bool(
                    s_state
                    == u_state
                ),
            }

            for quantity in (
                "pga",
                "pgv",
            ):
                for side, source in (
                    (
                        "seen",
                        s,
                    ),
                    (
                        "unseen",
                        u,
                    ),
                ):
                    record[
                        f"{side}_true_log10_{quantity}"
                    ] = float(
                        source[
                            f"true_log10_{quantity}"
                        ]
                    )

                    record[
                        f"{side}_base_log10_{quantity}"
                    ] = float(
                        source[
                            f"base_log10_{quantity}"
                        ]
                    )

                    record[
                        f"{side}_final_log10_{quantity}"
                    ] = float(
                        source[
                            f"final_log10_{quantity}"
                        ]
                    )

                    record[
                        f"{side}_is_tail_{quantity}"
                    ] = bool(
                        as_bool_scalar(
                            source[
                                f"is_tail_{quantity}"
                            ]
                        )
                    )

            rows.append(
                record
            )

    pairs = pd.DataFrame(
        rows
    )

    audit = {
        "caliper_km": float(
            caliper_km
        ),
        "total_event_repeat_groups": int(
            total_groups
        ),
        "groups_with_at_least_one_pair": int(
            groups_with_pair
        ),
        "groups_with_full_three_pairs": int(
            groups_full_three_pairs
        ),
        "groups_with_zero_pairs": int(
            groups_zero_pairs
        ),
        "matched_pairs": int(
            len(
                pairs
            )
        ),
        "matched_events": int(
            pairs[
                "event_id"
            ].nunique()
            if not pairs.empty
            else 0
        ),
        "matched_event_repeats": int(
            pairs[
                [
                    "event_id",
                    "repeat",
                ]
            ]
            .drop_duplicates()
            .shape[0]
            if not pairs.empty
            else 0
        ),
    }

    return (
        pairs,
        audit,
    )


# ---------------------------------------------------------------------
# Long format and canonical metrics
# ---------------------------------------------------------------------

def pairs_to_long(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for row in pairs.itertuples(
        index=False
    ):
        for (
            role,
            prefix,
        ) in (
            (
                "seen_target",
                "seen",
            ),
            (
                "unseen_station_target",
                "unseen",
            ),
        ):
            item = {
                "caliper_km": float(
                    row.caliper_km
                ),
                "event_id": str(
                    row.event_id
                ),
                "repeat": int(
                    row.repeat
                ),
                "pair_index": int(
                    row.pair_index
                ),
                "target_role": (
                    role
                ),
                "target_station_id": str(
                    getattr(
                        row,
                        f"{prefix}_station_id",
                    )
                ),
                "nearest_input_distance_km": float(
                    getattr(
                        row,
                        f"{prefix}_distance_km",
                    )
                ),
                "target_p_wave_reached": bool(
                    getattr(
                        row,
                        f"{prefix}_p_wave_reached",
                    )
                ),
            }

            for quantity in (
                "pga",
                "pgv",
            ):
                item[
                    f"true_log10_{quantity}"
                ] = float(
                    getattr(
                        row,
                        (
                            f"{prefix}_"
                            f"true_log10_{quantity}"
                        ),
                    )
                )

                item[
                    f"base_log10_{quantity}"
                ] = float(
                    getattr(
                        row,
                        (
                            f"{prefix}_"
                            f"base_log10_{quantity}"
                        ),
                    )
                )

                item[
                    f"final_log10_{quantity}"
                ] = float(
                    getattr(
                        row,
                        (
                            f"{prefix}_"
                            f"final_log10_{quantity}"
                        ),
                    )
                )

                item[
                    f"is_tail_{quantity}"
                ] = bool(
                    getattr(
                        row,
                        (
                            f"{prefix}_"
                            f"is_tail_{quantity}"
                        ),
                    )
                )

            rows.append(
                item
            )

    return pd.DataFrame(
        rows
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

    if metric == "under05":
        return (
            residual
            <= SEVERE_UNDER_THRESHOLD
        ).astype(
            float
        )

    raise ValueError(
        metric
    )


def event_values(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    metric: str,
) -> pd.Series:
    truth = frame[
        truth_column
    ].to_numpy(
        dtype=float
    )

    prediction = frame[
        prediction_column
    ].to_numpy(
        dtype=float
    )

    values = element_metric(
        truth,
        prediction,
        metric,
    )

    work = frame[
        [
            "event_id",
            "repeat",
        ]
    ].copy()

    work[
        "value"
    ] = values

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

    return (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )[
            "value"
        ]
        .mean()
    )


def metric_table(
    matched: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    models = {
        "cross_attention_base": (
            "base"
        ),
        "ca_drg": (
            "final"
        ),
    }

    for caliper, caliper_frame in matched.groupby(
        "caliper_km",
        sort=True,
    ):
        for (
            model_name,
            prefix,
        ) in models.items():
            for (
                target_role,
                role_frame,
            ) in caliper_frame.groupby(
                "target_role",
                sort=False,
            ):
                for quantity in (
                    "pga",
                    "pgv",
                ):
                    populations = {
                        "overall": np.ones(
                            len(
                                role_frame
                            ),
                            dtype=bool,
                        ),
                        "high_motion_tail": (
                            as_bool_array(
                                role_frame[
                                    f"is_tail_{quantity}"
                                ]
                            )
                        ),
                    }

                    for (
                        population,
                        mask,
                    ) in populations.items():
                        subset = role_frame.loc[
                            mask
                        ].copy()

                        if subset.empty:
                            continue

                        for metric in (
                            "mae",
                            "bias",
                            "factor2",
                            "factor3",
                            "under05",
                        ):
                            ev = event_values(
                                subset,
                                f"true_log10_{quantity}",
                                f"{prefix}_log10_{quantity}",
                                metric,
                            )

                            rows.append(
                                {
                                    "caliper_km": float(
                                        caliper
                                    ),
                                    "model": (
                                        model_name
                                    ),
                                    "target_role": (
                                        target_role
                                    ),
                                    "quantity": (
                                        quantity
                                    ),
                                    "population": (
                                        population
                                    ),
                                    "metric": (
                                        metric
                                    ),
                                    "value": float(
                                        ev.mean()
                                    ),
                                    "n_events": int(
                                        len(
                                            ev
                                        )
                                    ),
                                    "n_event_repeats": int(
                                        subset[
                                            [
                                                "event_id",
                                                "repeat",
                                            ]
                                        ]
                                        .drop_duplicates()
                                        .shape[0]
                                    ),
                                    "n_target_rows": int(
                                        len(
                                            subset
                                        )
                                    ),
                                    "n_unique_stations": int(
                                        subset[
                                            "target_station_id"
                                        ].nunique()
                                    ),
                                }
                            )

    return pd.DataFrame(
        rows
    )


def unseen_minus_seen_table(
    matched: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    models = {
        "cross_attention_base": "base",
        "ca_drg": "final",
    }

    for caliper, caliper_frame in matched.groupby(
        "caliper_km",
        sort=True,
    ):
        for (
            model_name,
            prefix,
        ) in models.items():
            for quantity in (
                "pga",
                "pgv",
            ):
                for population in (
                    "overall",
                    "high_motion_tail",
                ):
                    for metric in (
                        "mae",
                        "bias",
                        "factor2",
                        "factor3",
                        "under05",
                    ):
                        role_values = {}

                        for role in (
                            "seen_target",
                            "unseen_station_target",
                        ):
                            subset = (
                                caliper_frame.loc[
                                    caliper_frame[
                                        "target_role"
                                    ].eq(
                                        role
                                    )
                                ].copy()
                            )

                            if (
                                population
                                == "high_motion_tail"
                            ):
                                subset = subset.loc[
                                    as_bool_array(
                                        subset[
                                            f"is_tail_{quantity}"
                                        ]
                                    )
                                ].copy()

                            if subset.empty:
                                continue

                            role_values[
                                role
                            ] = event_values(
                                subset,
                                f"true_log10_{quantity}",
                                f"{prefix}_log10_{quantity}",
                                metric,
                            )

                        if not {
                            "seen_target",
                            "unseen_station_target",
                        }.issubset(
                            role_values
                        ):
                            continue

                        seen = role_values[
                            "seen_target"
                        ]

                        unseen = role_values[
                            "unseen_station_target"
                        ]

                        common = (
                            seen.index
                            .intersection(
                                unseen.index
                            )
                        )

                        if len(
                            common
                        ) == 0:
                            continue

                        seen_value = float(
                            seen.loc[
                                common
                            ].mean()
                        )

                        unseen_value = float(
                            unseen.loc[
                                common
                            ].mean()
                        )

                        delta = (
                            unseen.loc[
                                common
                            ]
                            - seen.loc[
                                common
                            ]
                        )

                        record = {
                            "caliper_km": float(
                                caliper
                            ),
                            "model": (
                                model_name
                            ),
                            "quantity": (
                                quantity
                            ),
                            "population": (
                                population
                            ),
                            "metric": (
                                metric
                            ),
                            "n_paired_events": int(
                                len(
                                    common
                                )
                            ),
                            "seen_value": (
                                seen_value
                            ),
                            "unseen_value": (
                                unseen_value
                            ),
                            "mean_delta_unseen_minus_seen": float(
                                delta.mean()
                            ),
                            "median_delta_unseen_minus_seen": float(
                                delta.median()
                            ),
                        }

                        if metric == "mae":
                            record[
                                "relative_ood_gap_percent"
                            ] = float(
                                100.0
                                * (
                                    unseen_value
                                    - seen_value
                                )
                                / max(
                                    abs(
                                        seen_value
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
                                * (
                                    unseen_value
                                    - seen_value
                                )
                            )

                        rows.append(
                            record
                        )

    return pd.DataFrame(
        rows
    )


def cadrg_minus_base_table(
    matched: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for caliper, caliper_frame in matched.groupby(
        "caliper_km",
        sort=True,
    ):
        for role in (
            "seen_target",
            "unseen_station_target",
        ):
            role_frame = (
                caliper_frame.loc[
                    caliper_frame[
                        "target_role"
                    ].eq(
                        role
                    )
                ].copy()
            )

            for quantity in (
                "pga",
                "pgv",
            ):
                for population in (
                    "overall",
                    "high_motion_tail",
                ):
                    subset = role_frame

                    if (
                        population
                        == "high_motion_tail"
                    ):
                        subset = role_frame.loc[
                            as_bool_array(
                                role_frame[
                                    f"is_tail_{quantity}"
                                ]
                            )
                        ].copy()

                    if subset.empty:
                        continue

                    for metric in (
                        "mae",
                        "bias",
                        "factor2",
                        "under05",
                    ):
                        base = event_values(
                            subset,
                            f"true_log10_{quantity}",
                            f"base_log10_{quantity}",
                            metric,
                        )

                        final = event_values(
                            subset,
                            f"true_log10_{quantity}",
                            f"final_log10_{quantity}",
                            metric,
                        )

                        common = (
                            base.index
                            .intersection(
                                final.index
                            )
                        )

                        delta = (
                            final.loc[
                                common
                            ]
                            - base.loc[
                                common
                            ]
                        )

                        base_value = float(
                            base.loc[
                                common
                            ].mean()
                        )

                        final_value = float(
                            final.loc[
                                common
                            ].mean()
                        )

                        record = {
                            "caliper_km": float(
                                caliper
                            ),
                            "target_role": (
                                role
                            ),
                            "quantity": (
                                quantity
                            ),
                            "population": (
                                population
                            ),
                            "metric": (
                                metric
                            ),
                            "n_paired_events": int(
                                len(
                                    common
                                )
                            ),
                            "base_value": (
                                base_value
                            ),
                            "ca_drg_value": (
                                final_value
                            ),
                            "mean_delta_cadrg_minus_base": float(
                                delta.mean()
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
                                * (
                                    final_value
                                    - base_value
                                )
                                / max(
                                    abs(
                                        base_value
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
                                * (
                                    final_value
                                    - base_value
                                )
                            )

                        rows.append(
                            record
                        )

    return pd.DataFrame(
        rows
    )


def geometry_summary(
    pairs: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if pairs.empty:
        return {
            **audit,
            "fraction_p_wave_state_matched": (
                np.nan
            ),
            "median_seen_distance_km": (
                np.nan
            ),
            "median_unseen_distance_km": (
                np.nan
            ),
            "median_abs_distance_difference_km": (
                np.nan
            ),
            "p75_abs_distance_difference_km": (
                np.nan
            ),
            "p90_abs_distance_difference_km": (
                np.nan
            ),
            "mean_abs_distance_difference_km": (
                np.nan
            ),
        }

    return {
        **audit,
        "fraction_p_wave_state_matched": float(
            pairs[
                "p_wave_state_match"
            ].mean()
        ),
        "median_seen_distance_km": float(
            pairs[
                "seen_distance_km"
            ].median()
        ),
        "median_unseen_distance_km": float(
            pairs[
                "unseen_distance_km"
            ].median()
        ),
        "median_abs_distance_difference_km": float(
            pairs[
                "absolute_distance_difference_km"
            ].median()
        ),
        "p75_abs_distance_difference_km": float(
            pairs[
                "absolute_distance_difference_km"
            ].quantile(
                0.75
            )
        ),
        "p90_abs_distance_difference_km": float(
            pairs[
                "absolute_distance_difference_km"
            ].quantile(
                0.90
            )
        ),
        "mean_abs_distance_difference_km": float(
            pairs[
                "absolute_distance_difference_km"
            ].mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "runs/"
            "cadrg_station_ood_relative_bias_selection/"
            "repeated_station_ood_predictions.csv"
        ),
    )

    parser.add_argument(
        "--distance-calipers-km",
        default="10,20,30",
    )

    parser.add_argument(
        "--primary-caliper-km",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "cadrg_station_ood_relative_bias_selection/"
            "matched_geometry_audit"
        ),
    )

    args = parser.parse_args()

    prediction_path = Path(
        args.predictions
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            prediction_path
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calipers = parse_float_list(
        args.distance_calipers_km
    )

    if not any(
        np.isclose(
            args.primary_caliper_km,
            value,
        )
        for value in calipers
    ):
        raise ValueError(
            "--primary-caliper-km must be included in "
            "--distance-calipers-km."
        )

    predictions = pd.read_csv(
        prediction_path,
        dtype={
            "event_id": str,
            "target_station_id": str,
        },
    )

    required = {
        "event_id",
        "repeat",
        "target_role",
        "target_station_id",
        "target_p_wave_reached",
        "nearest_input_distance_km",
        "true_log10_pga",
        "true_log10_pgv",
        "base_log10_pga",
        "base_log10_pgv",
        "final_log10_pga",
        "final_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
    }

    missing = required.difference(
        predictions.columns
    )

    if missing:
        raise ValueError(
            "Predictions missing columns: "
            f"{sorted(missing)}"
        )

    roles = set(
        predictions[
            "target_role"
        ].astype(
            str
        ).unique()
    )

    expected_roles = {
        "seen_target",
        "unseen_station_target",
    }

    if not expected_roles.issubset(
        roles
    ):
        raise ValueError(
            f"Expected roles {expected_roles}; got {roles}"
        )

    # Pair-budget audit.
    counts = (
        predictions.groupby(
            [
                "event_id",
                "repeat",
                "target_role",
            ]
        )
        .size()
        .unstack(
            "target_role"
        )
    )

    if (
        counts[
            "seen_target"
        ]
        .ne(
            counts[
                "unseen_station_target"
            ]
        )
        .any()
    ):
        raise RuntimeError(
            "Seen/unseen target budgets are not equal "
            "within every event-repeat."
        )

    all_pairs = []
    all_long = []
    geometry_rows = []

    for caliper in calipers:
        pairs, audit = build_matched_pairs(
            predictions,
            caliper,
        )

        if pairs.empty:
            print(
                f"WARNING: no pairs at {caliper:g} km caliper."
            )
            continue

        all_pairs.append(
            pairs
        )

        long_frame = pairs_to_long(
            pairs
        )

        all_long.append(
            long_frame
        )

        geometry_rows.append(
            geometry_summary(
                pairs,
                audit,
            )
        )

    if not all_pairs:
        raise RuntimeError(
            "No matched pairs were produced at any caliper."
        )

    pairs_all = pd.concat(
        all_pairs,
        ignore_index=True,
    )

    matched_all = pd.concat(
        all_long,
        ignore_index=True,
    )

    geometry = pd.DataFrame(
        geometry_rows
    )

    metrics = metric_table(
        matched_all
    )

    unseen_seen = (
        unseen_minus_seen_table(
            matched_all
        )
    )

    gate_effect = (
        cadrg_minus_base_table(
            matched_all
        )
    )

    pairs_path = (
        out_dir
        / "matched_target_pairs_all_calipers.csv"
    )

    matched_long_path = (
        out_dir
        / "matched_targets_long_all_calipers.csv"
    )

    geometry_path = (
        out_dir
        / "matched_geometry_summary.csv"
    )

    metrics_path = (
        out_dir
        / "matched_station_ood_metrics.csv"
    )

    unseen_seen_path = (
        out_dir
        / "matched_unseen_minus_seen_deltas.csv"
    )

    gate_effect_path = (
        out_dir
        / "matched_cadrg_minus_base_by_role.csv"
    )

    pairs_all.to_csv(
        pairs_path,
        index=False,
    )

    matched_all.to_csv(
        matched_long_path,
        index=False,
    )

    geometry.to_csv(
        geometry_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    unseen_seen.to_csv(
        unseen_seen_path,
        index=False,
    )

    gate_effect.to_csv(
        gate_effect_path,
        index=False,
    )

    primary = float(
        args.primary_caliper_km
    )

    primary_geometry = (
        geometry.loc[
            np.isclose(
                geometry[
                    "caliper_km"
                ],
                primary,
            )
        ]
    )

    primary_metrics = (
        metrics.loc[
            np.isclose(
                metrics[
                    "caliper_km"
                ],
                primary,
            )
        ]
    )

    primary_unseen_seen = (
        unseen_seen.loc[
            np.isclose(
                unseen_seen[
                    "caliper_km"
                ],
                primary,
            )
        ]
    )

    primary_gate = (
        gate_effect.loc[
            np.isclose(
                gate_effect[
                    "caliper_km"
                ],
                primary,
            )
        ]
    )

    print(
        "=== CA-DRG Station-OOD matched geometry audit ==="
    )

    print(
        f"Input rows              : "
        f"{len(predictions):,}"
    )

    print(
        f"Events                  : "
        f"{predictions['event_id'].nunique():,}"
    )

    print(
        "Matching variables      : "
        "exact P-wave state + nearest-input distance"
    )

    print(
        f"Caliper sensitivity     : "
        f"{calipers} km"
    )

    print(
        f"Primary descriptive     : "
        f"{primary:g} km"
    )

    print(
        "\n=== Matching quality: all calipers ==="
    )

    print(
        geometry.to_string(
            index=False
        )
    )

    print(
        f"\n=== PRIMARY {primary:g}-km matched geometry ==="
    )

    print(
        primary_geometry.to_string(
            index=False
        )
    )

    print(
        f"\n=== PRIMARY {primary:g}-km matched canonical metrics ==="
    )

    display_metrics = (
        primary_metrics.loc[
            primary_metrics[
                "metric"
            ].isin(
                [
                    "mae",
                    "factor2",
                    "under05",
                ]
            )
        ][
            [
                "model",
                "target_role",
                "quantity",
                "population",
                "metric",
                "value",
                "n_events",
                "n_target_rows",
                "n_unique_stations",
            ]
        ]
    )

    print(
        display_metrics.to_string(
            index=False
        )
    )

    print(
        f"\n=== PRIMARY {primary:g}-km unseen minus seen ==="
    )

    display_gap = (
        primary_unseen_seen.loc[
            primary_unseen_seen[
                "metric"
            ].isin(
                [
                    "mae",
                    "factor2",
                    "under05",
                ]
            )
        ]
    )

    print(
        display_gap.to_string(
            index=False
        )
    )

    print(
        f"\n=== PRIMARY {primary:g}-km CA-DRG minus Base ==="
    )

    display_gate = (
        primary_gate.loc[
            primary_gate[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                ]
            )
        ]
    )

    print(
        display_gate.to_string(
            index=False
        )
    )

    protocol = {
        "prediction_file": str(
            prediction_path.resolve()
        ),
        "matching_variables": [
            "target_p_wave_reached",
            "nearest_input_distance_km",
        ],
        "outcome_variables_used_for_matching": (
            False
        ),
        "distance_calipers_km": (
            calipers
        ),
        "primary_descriptive_caliper_km": (
            primary
        ),
        "matching_objective": (
            "within exact P-wave state, maximize retained pairs "
            "within caliper then minimize total absolute nearest-input "
            "distance difference"
        ),
        "metric_aggregation": (
            "matched targets -> repeats -> events"
        ),
        "models": [
            "cross_attention_base",
            "ca_drg",
        ],
        "test_predictions_reused": (
            True
        ),
        "model_retraining": (
            False
        ),
    }

    (
        out_dir
        / "matched_audit_protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nOutputs:"
    )

    for path in [
        pairs_path,
        matched_long_path,
        geometry_path,
        metrics_path,
        unseen_seen_path,
        gate_effect_path,
        out_dir
        / "matched_audit_protocol.json",
    ]:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
