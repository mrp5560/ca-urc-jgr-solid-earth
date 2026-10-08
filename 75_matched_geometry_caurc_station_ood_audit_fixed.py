#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
75_matched_geometry_caurc_station_ood_audit.py

Outcome-independent matched-geometry audit for the completed strict
CA-URC Station-OOD experiment.

Input
-----
runs/caurc_station_ood/repeated_station_ood_predictions.csv

Matching rule
-------------
For every (event_id, repeat):

1. split targets by exact target P-wave state:
       target_p_wave_reached = False / True

2. within each state, match:
       seen_target <-> unseen_station_target

3. matching uses ONLY:
       nearest_input_distance_km
   and the exact P-wave state.

4. for each caliper (default 10, 20, 30 km):
       maximize number of retained pairs
       then minimize total |distance_seen - distance_unseen|

Matching NEVER uses:
    true PGA / PGV
    prediction
    prediction error
    tail label
    underprediction label

Primary paper caliper:
    20 km

Sensitivity:
    10 km and 30 km

Main analyses
-------------
A. Geometry balance after matching.

B. Canonical matched-target metrics:
       targets -> repeats -> events
   for:
       Cross-Attention Base
       CA-URC
   separately at:
       seen_target
       unseen_station_target

C. CA-URC minus Base within each matched target role.
   This tests whether the CA-URC improvement at unseen stations remains
   after geometry/P-wave-state control.

D. Direct unseen-minus-seen matched-pair OOD gap.
   Pair-level outcome differences are aggregated:
       matched pairs -> repeats -> events

   Populations:
       overall
       both_high_motion_tail
   "both_high_motion_tail" is defined AFTER matching and is used only
   for analysis; it does not affect pair construction.

E. Event-level paired bootstrap, 10,000 repetitions by default.

Interpretation
--------------
- For MAE / U0.5:
      CA-URC - Base < 0 favors CA-URC.
- For Factor-2:
      CA-URC - Base > 0 favors CA-URC.
- For unseen - seen MAE / U0.5:
      positive = residual Station-OOD gap after geometry matching.

This script performs no training and no model selection.
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
SEVERE_UNDER_THRESHOLD = -0.5


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_float_list(text: str) -> list[float]:
    values = []

    for token in str(text).split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

        if value <= 0:
            raise ValueError(
                "All calipers must be > 0 km."
            )

        values.append(
            value
        )

    values = sorted(
        set(values)
    )

    if not values:
        raise ValueError(
            "No calipers supplied."
        )

    return values


def as_bool_array(
    series: pd.Series,
) -> np.ndarray:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.to_numpy(
            dtype=bool
        )

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    true_set = {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }

    false_set = {
        "false",
        "0",
        "no",
        "n",
        "f",
    }

    unknown = set(
        normalized.unique()
    ).difference(
        true_set
        | false_set
    )

    if unknown:
        raise ValueError(
            "Cannot parse boolean values: "
            f"{sorted(unknown)[:20]}"
        )

    return normalized.isin(
        true_set
    ).to_numpy(
        dtype=bool
    )


def load_predictions(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    frame = pd.read_csv(
        path,
        dtype={
            "event_id": str,
            "target_station_id": str,
            "input_station_ids": str,
        },
    )

    required = {
        "event_id",
        "repeat",
        "target_role",
        "input_station_ids",
        "target_station_id",
        "target_station_index",
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
        frame.columns
    )

    if missing:
        raise ValueError(
            "Station-OOD prediction file missing columns: "
            f"{sorted(missing)}"
        )

    frame[
        "event_id"
    ] = (
        frame[
            "event_id"
        ]
        .astype(str)
        .str.strip()
    )

    frame[
        "target_role"
    ] = (
        frame[
            "target_role"
        ]
        .astype(str)
        .str.strip()
    )

    allowed_roles = {
        "seen_target",
        "unseen_station_target",
    }

    bad_roles = set(
        frame[
            "target_role"
        ].unique()
    ).difference(
        allowed_roles
    )

    if bad_roles:
        raise ValueError(
            f"Unexpected target roles: {sorted(bad_roles)}"
        )

    frame[
        "target_p_wave_reached_bool"
    ] = as_bool_array(
        frame[
            "target_p_wave_reached"
        ]
    )

    frame[
        "is_tail_pga_bool"
    ] = as_bool_array(
        frame[
            "is_tail_pga"
        ]
    )

    frame[
        "is_tail_pgv_bool"
    ] = as_bool_array(
        frame[
            "is_tail_pgv"
        ]
    )

    frame[
        "nearest_input_distance_km"
    ] = pd.to_numeric(
        frame[
            "nearest_input_distance_km"
        ],
        errors="raise",
    )

    if not np.all(
        np.isfinite(
            frame[
                "nearest_input_distance_km"
            ].to_numpy(
                float
            )
        )
    ):
        raise ValueError(
            "nearest_input_distance_km contains non-finite values."
        )

    frame[
        "_row_id"
    ] = np.arange(
        len(
            frame
        ),
        dtype=np.int64,
    )

    return frame


# ---------------------------------------------------------------------
# Protocol audit
# ---------------------------------------------------------------------

def audit_station_ood_protocol(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    group_columns = [
        "event_id",
        "repeat",
    ]

    grouped = frame.groupby(
        group_columns,
        sort=False,
    )

    bad_input_groups = 0
    bad_role_budget_groups = 0

    seen_counts = []
    unseen_counts = []

    for _, group in grouped:
        if (
            group[
                "input_station_ids"
            ]
            .astype(str)
            .nunique()
            != 1
        ):
            bad_input_groups += 1

        role_counts = (
            group[
                "target_role"
            ]
            .value_counts()
        )

        seen = int(
            role_counts.get(
                "seen_target",
                0,
            )
        )

        unseen = int(
            role_counts.get(
                "unseen_station_target",
                0,
            )
        )

        seen_counts.append(
            seen
        )

        unseen_counts.append(
            unseen
        )

        if (
            seen
            != unseen
            or seen
            == 0
        ):
            bad_role_budget_groups += 1

    audit = {
        "rows": int(
            len(
                frame
            )
        ),
        "events": int(
            frame[
                "event_id"
            ].nunique()
        ),
        "event_repeat_groups": int(
            frame[
                group_columns
            ]
            .drop_duplicates()
            .shape[
                0
            ]
        ),
        "seen_rows": int(
            (
                frame[
                    "target_role"
                ]
                == "seen_target"
            ).sum()
        ),
        "unseen_rows": int(
            (
                frame[
                    "target_role"
                ]
                == "unseen_station_target"
            ).sum()
        ),
        "unique_seen_stations": int(
            frame.loc[
                frame[
                    "target_role"
                ]
                == "seen_target",
                "target_station_id",
            ].nunique()
        ),
        "unique_unseen_stations": int(
            frame.loc[
                frame[
                    "target_role"
                ]
                == "unseen_station_target",
                "target_station_id",
            ].nunique()
        ),
        "bad_same_input_station_groups": int(
            bad_input_groups
        ),
        "bad_equal_role_budget_groups": int(
            bad_role_budget_groups
        ),
        "seen_targets_per_group_unique": sorted(
            set(
                seen_counts
            )
        ),
        "unseen_targets_per_group_unique": sorted(
            set(
                unseen_counts
            )
        ),
    }

    if (
        bad_input_groups
        > 0
    ):
        raise RuntimeError(
            "Station-OOD protocol audit failed: "
            "seen/unseen targets do not always share "
            "the same input station set."
        )

    if (
        bad_role_budget_groups
        > 0
    ):
        raise RuntimeError(
            "Station-OOD protocol audit failed: "
            "seen/unseen target budgets are not equal."
        )

    return audit


# ---------------------------------------------------------------------
# Exact maximum-cardinality / minimum-cost matching
# ---------------------------------------------------------------------

def best_small_bipartite_matching(
    seen: pd.DataFrame,
    unseen: pd.DataFrame,
    caliper_km: float,
) -> list[
    tuple[
        int,
        int,
        float,
    ]
]:
    """
    Exact brute-force matcher.

    Typical Station-OOD group has <=3 seen and <=3 unseen targets,
    so exhaustive search is trivial and avoids external dependencies.

    Objective:
        1) maximize number of pairs
        2) minimize total absolute distance difference
        3) deterministic lexicographic tie-break
    """

    if (
        seen.empty
        or unseen.empty
    ):
        return []

    # IMPORTANT:
    # pandas.itertuples() renames column names beginning with "_" to
    # positional field names such as "_1", so "_row_id" cannot be
    # accessed reliably as a namedtuple attribute.  Use plain tuples
    # with explicit columns instead.
    seen_rows = list(
        seen[
            [
                "_row_id",
                "nearest_input_distance_km",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    unseen_rows = list(
        unseen[
            [
                "_row_id",
                "nearest_input_distance_km",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    m = len(
        seen_rows
    )

    n = len(
        unseen_rows
    )

    max_k = min(
        m,
        n,
    )

    best_pairs = None
    best_cost = None
    best_tie = None

    for k in range(
        max_k,
        0,
        -1,
    ):
        found_for_k = False

        for seen_indices in itertools.combinations(
            range(
                m
            ),
            k,
        ):
            for unseen_indices in itertools.combinations(
                range(
                    n
                ),
                k,
            ):
                for unseen_perm in itertools.permutations(
                    unseen_indices
                ):
                    pairs = []
                    valid = True
                    total_cost = 0.0

                    tie_key = []

                    for si, ui in zip(
                        seen_indices,
                        unseen_perm,
                    ):
                        seen_row = seen_rows[
                            si
                        ]

                        unseen_row = unseen_rows[
                            ui
                        ]

                        distance_difference = abs(
                            float(
                                seen_row[
                                    1
                                ]
                            )
                            - float(
                                unseen_row[
                                    1
                                ]
                            )
                        )

                        if (
                            distance_difference
                            > float(
                                caliper_km
                            )
                            + 1e-12
                        ):
                            valid = False
                            break

                        total_cost += (
                            distance_difference
                        )

                        seen_row_id = int(
                            seen_row[
                                0
                            ]
                        )

                        unseen_row_id = int(
                            unseen_row[
                                0
                            ]
                        )

                        pairs.append(
                            (
                                seen_row_id,
                                unseen_row_id,
                                float(
                                    distance_difference
                                ),
                            )
                        )

                        tie_key.append(
                            (
                                seen_row_id,
                                unseen_row_id,
                            )
                        )

                    if not valid:
                        continue

                    found_for_k = True

                    tie_key = tuple(
                        sorted(
                            tie_key
                        )
                    )

                    if (
                        best_pairs is None
                        or total_cost
                        < best_cost
                        - 1e-12
                        or (
                            abs(
                                total_cost
                                - best_cost
                            )
                            <= 1e-12
                            and tie_key
                            < best_tie
                        )
                    ):
                        best_pairs = pairs
                        best_cost = (
                            total_cost
                        )
                        best_tie = (
                            tie_key
                        )

        if found_for_k:
            break

    return (
        best_pairs
        if best_pairs is not None
        else []
    )


def build_matches(
    frame: pd.DataFrame,
    caliper_km: float,
) -> pd.DataFrame:
    rows = []

    pair_counter = 0

    grouped = frame.groupby(
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
        input_station_ids = str(
            group[
                "input_station_ids"
            ].iloc[
                0
            ]
        )

        for p_state in (
            False,
            True,
        ):
            state_group = group.loc[
                group[
                    "target_p_wave_reached_bool"
                ]
                == p_state
            ]

            seen = state_group.loc[
                state_group[
                    "target_role"
                ]
                == "seen_target"
            ]

            unseen = state_group.loc[
                state_group[
                    "target_role"
                ]
                == "unseen_station_target"
            ]

            matches = (
                best_small_bipartite_matching(
                    seen,
                    unseen,
                    caliper_km,
                )
            )

            for (
                seen_row_id,
                unseen_row_id,
                distance_difference,
            ) in matches:
                seen_row = frame.loc[
                    frame[
                        "_row_id"
                    ]
                    == seen_row_id
                ].iloc[
                    0
                ]

                unseen_row = frame.loc[
                    frame[
                        "_row_id"
                    ]
                    == unseen_row_id
                ].iloc[
                    0
                ]

                rows.append(
                    {
                        "pair_id": int(
                            pair_counter
                        ),
                        "event_id": str(
                            event_id
                        ),
                        "repeat": int(
                            repeat
                        ),
                        "input_station_ids": (
                            input_station_ids
                        ),
                        "p_wave_reached": bool(
                            p_state
                        ),
                        "seen_row_id": int(
                            seen_row_id
                        ),
                        "unseen_row_id": int(
                            unseen_row_id
                        ),
                        "seen_station_id": str(
                            seen_row[
                                "target_station_id"
                            ]
                        ),
                        "unseen_station_id": str(
                            unseen_row[
                                "target_station_id"
                            ]
                        ),
                        "seen_station_index": int(
                            seen_row[
                                "target_station_index"
                            ]
                        ),
                        "unseen_station_index": int(
                            unseen_row[
                                "target_station_index"
                            ]
                        ),
                        "seen_distance_km": float(
                            seen_row[
                                "nearest_input_distance_km"
                            ]
                        ),
                        "unseen_distance_km": float(
                            unseen_row[
                                "nearest_input_distance_km"
                            ]
                        ),
                        "abs_distance_difference_km": float(
                            distance_difference
                        ),
                        "caliper_km": float(
                            caliper_km
                        ),
                    }
                )

                pair_counter += 1

    return pd.DataFrame(
        rows
    )


def build_matched_long(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    by_row = (
        frame.set_index(
            "_row_id",
            drop=False,
        )
    )

    rows = []

    for pair in pairs.itertuples(
        index=False
    ):
        for (
            role,
            row_id,
        ) in (
            (
                "seen_target",
                int(
                    pair.seen_row_id
                ),
            ),
            (
                "unseen_station_target",
                int(
                    pair.unseen_row_id
                ),
            ),
        ):
            source = by_row.loc[
                row_id
            ]

            record = (
                source.to_dict()
            )

            record.update(
                {
                    "pair_id": int(
                        pair.pair_id
                    ),
                    "matched_role": (
                        role
                    ),
                    "matched_p_wave_reached": bool(
                        pair.p_wave_reached
                    ),
                    "matched_abs_distance_difference_km": float(
                        pair
                        .abs_distance_difference_km
                    ),
                    "caliper_km": float(
                        pair.caliper_km
                    ),
                }
            )

            rows.append(
                record
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Matching balance
# ---------------------------------------------------------------------

def matching_balance_summary(
    original: pd.DataFrame,
    pairs: pd.DataFrame,
    caliper_km: float,
) -> dict[str, Any]:
    total_groups = (
        original[
            [
                "event_id",
                "repeat",
            ]
        ]
        .drop_duplicates()
        .shape[
            0
        ]
    )

    if pairs.empty:
        return {
            "caliper_km": float(
                caliper_km
            ),
            "total_event_repeat_groups": int(
                total_groups
            ),
            "matched_event_repeat_groups": 0,
            "matched_group_fraction": 0.0,
            "matched_events": 0,
            "matched_pairs": 0,
        }

    seen_original = original.loc[
        original[
            "target_role"
        ]
        == "seen_target"
    ]

    unseen_original = original.loc[
        original[
            "target_role"
        ]
        == "unseen_station_target"
    ]

    result = {
        "caliper_km": float(
            caliper_km
        ),
        "total_event_repeat_groups": int(
            total_groups
        ),
        "matched_event_repeat_groups": int(
            pairs[
                [
                    "event_id",
                    "repeat",
                ]
            ]
            .drop_duplicates()
            .shape[
                0
            ]
        ),
        "matched_group_fraction": float(
            pairs[
                [
                    "event_id",
                    "repeat",
                ]
            ]
            .drop_duplicates()
            .shape[
                0
            ]
            / max(
                total_groups,
                1,
            )
        ),
        "matched_events": int(
            pairs[
                "event_id"
            ].nunique()
        ),
        "matched_pairs": int(
            len(
                pairs
            )
        ),
        "median_abs_distance_difference_km": float(
            pairs[
                "abs_distance_difference_km"
            ].median()
        ),
        "p25_abs_distance_difference_km": float(
            pairs[
                "abs_distance_difference_km"
            ].quantile(
                0.25
            )
        ),
        "p75_abs_distance_difference_km": float(
            pairs[
                "abs_distance_difference_km"
            ].quantile(
                0.75
            )
        ),
        "p90_abs_distance_difference_km": float(
            pairs[
                "abs_distance_difference_km"
            ].quantile(
                0.90
            )
        ),
        "mean_seen_distance_before_km": float(
            seen_original[
                "nearest_input_distance_km"
            ].mean()
        ),
        "mean_unseen_distance_before_km": float(
            unseen_original[
                "nearest_input_distance_km"
            ].mean()
        ),
        "mean_unseen_minus_seen_before_km": float(
            unseen_original[
                "nearest_input_distance_km"
            ].mean()
            - seen_original[
                "nearest_input_distance_km"
            ].mean()
        ),
        "mean_seen_distance_matched_km": float(
            pairs[
                "seen_distance_km"
            ].mean()
        ),
        "mean_unseen_distance_matched_km": float(
            pairs[
                "unseen_distance_km"
            ].mean()
        ),
        "mean_unseen_minus_seen_matched_km": float(
            pairs[
                "unseen_distance_km"
            ].mean()
            - pairs[
                "seen_distance_km"
            ].mean()
        ),
        "median_seen_distance_matched_km": float(
            pairs[
                "seen_distance_km"
            ].median()
        ),
        "median_unseen_distance_matched_km": float(
            pairs[
                "unseen_distance_km"
            ].median()
        ),
        "p_wave_state_exact_matching": True,
        "matching_used_outcomes": False,
    }

    return result


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------

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


def event_values_from_rows(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    metric: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(
            dtype=float
        )

    values = element_metric(
        frame[
            truth_column
        ].to_numpy(
            float
        ),
        frame[
            prediction_column
        ].to_numpy(
            float
        ),
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


def matched_role_metric_table(
    matched_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    models = {
        "cross_attention_base": (
            "base"
        ),
        "ca_urc": (
            "final"
        ),
    }

    for (
        role,
        role_frame,
    ) in matched_long.groupby(
        "target_role",
        sort=False,
    ):
        for (
            model_name,
            prefix,
        ) in models.items():
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
                        role_frame[
                            f"is_tail_{quantity}_bool"
                        ]
                        .to_numpy(
                            dtype=bool
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
                        "under05",
                    ):
                        event_values = (
                            event_values_from_rows(
                                subset,
                                f"true_log10_{quantity}",
                                f"{prefix}_log10_{quantity}",
                                metric,
                            )
                        )

                        rows.append(
                            {
                                "caliper_km": float(
                                    role_frame[
                                        "caliper_km"
                                    ].iloc[
                                        0
                                    ]
                                ),
                                "model": (
                                    model_name
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
                                "value": float(
                                    event_values.mean()
                                ),
                                "n_events": int(
                                    len(
                                        event_values
                                    )
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


def caurc_minus_base_by_matched_role(
    matched_long: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[
        tuple,
        pd.Series,
    ],
]:
    rows = []
    event_delta_store = {}

    for (
        role,
        role_frame,
    ) in matched_long.groupby(
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
                    role_frame[
                        f"is_tail_{quantity}_bool"
                    ]
                    .to_numpy(
                        dtype=bool
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
                    "under05",
                ):
                    base = (
                        event_values_from_rows(
                            subset,
                            f"true_log10_{quantity}",
                            f"base_log10_{quantity}",
                            metric,
                        )
                    )

                    caurc = (
                        event_values_from_rows(
                            subset,
                            f"true_log10_{quantity}",
                            f"final_log10_{quantity}",
                            metric,
                        )
                    )

                    common = (
                        base.index
                        .intersection(
                            caurc.index
                        )
                    )

                    delta = (
                        caurc.loc[
                            common
                        ]
                        - base.loc[
                            common
                        ]
                    )

                    key = (
                        float(
                            role_frame[
                                "caliper_km"
                            ].iloc[
                                0
                            ]
                        ),
                        "caurc_minus_base",
                        role,
                        quantity,
                        population,
                        metric,
                    )

                    event_delta_store[
                        key
                    ] = delta

                    base_value = float(
                        base.loc[
                            common
                        ].mean()
                    )

                    caurc_value = float(
                        caurc.loc[
                            common
                        ].mean()
                    )

                    record = {
                        "caliper_km": float(
                            role_frame[
                                "caliper_km"
                            ].iloc[
                                0
                            ]
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
                        "ca_urc_value": (
                            caurc_value
                        ),
                        "mean_delta_caurc_minus_base": float(
                            delta.mean()
                        ),
                        "median_delta_caurc_minus_base": float(
                            delta.median()
                        ),
                    }

                    if metric == "mae":
                        record[
                            "relative_change_percent"
                        ] = float(
                            100.0
                            * (
                                caurc_value
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
                                caurc_value
                                - base_value
                            )
                        )

                    rows.append(
                        record
                    )

    return (
        pd.DataFrame(
            rows
        ),
        event_delta_store,
    )


# ---------------------------------------------------------------------
# Direct matched-pair unseen-minus-seen gap
# ---------------------------------------------------------------------

def pairwise_metric_delta_frame(
    original: pd.DataFrame,
    pairs: pd.DataFrame,
    quantity: str,
    model_prefix: str,
    metric: str,
    population: str,
) -> pd.DataFrame:
    by_row = original.set_index(
        "_row_id",
        drop=False,
    )

    rows = []

    for pair in pairs.itertuples(
        index=False
    ):
        seen = by_row.loc[
            int(
                pair.seen_row_id
            )
        ]

        unseen = by_row.loc[
            int(
                pair.unseen_row_id
            )
        ]

        if (
            population
            == "both_high_motion_tail"
        ):
            if not (
                bool(
                    seen[
                        f"is_tail_{quantity}_bool"
                    ]
                )
                and bool(
                    unseen[
                        f"is_tail_{quantity}_bool"
                    ]
                )
            ):
                continue

        elif (
            population
            != "overall"
        ):
            raise ValueError(
                population
            )

        seen_value = float(
            element_metric(
                np.asarray(
                    [
                        seen[
                            f"true_log10_{quantity}"
                        ]
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        seen[
                            f"{model_prefix}_log10_{quantity}"
                        ]
                    ],
                    dtype=float,
                ),
                metric,
            )[
                0
            ]
        )

        unseen_value = float(
            element_metric(
                np.asarray(
                    [
                        unseen[
                            f"true_log10_{quantity}"
                        ]
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        unseen[
                            f"{model_prefix}_log10_{quantity}"
                        ]
                    ],
                    dtype=float,
                ),
                metric,
            )[
                0
            ]
        )

        rows.append(
            {
                "pair_id": int(
                    pair.pair_id
                ),
                "event_id": str(
                    pair.event_id
                ),
                "repeat": int(
                    pair.repeat
                ),
                "seen_value": (
                    seen_value
                ),
                "unseen_value": (
                    unseen_value
                ),
                "delta_unseen_minus_seen": float(
                    unseen_value
                    - seen_value
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def aggregate_pair_delta_to_events(
    pair_delta: pd.DataFrame,
) -> pd.Series:
    if pair_delta.empty:
        return pd.Series(
            dtype=float
        )

    event_repeat = (
        pair_delta.groupby(
            [
                "event_id",
                "repeat",
            ],
            sort=False,
        )[
            "delta_unseen_minus_seen"
        ]
        .mean()
        .reset_index()
    )

    return (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )[
            "delta_unseen_minus_seen"
        ]
        .mean()
    )


def unseen_minus_seen_pair_table(
    original: pd.DataFrame,
    pairs: pd.DataFrame,
    caliper_km: float,
) -> tuple[
    pd.DataFrame,
    dict[
        tuple,
        pd.Series,
    ],
]:
    rows = []
    event_delta_store = {}

    models = {
        "cross_attention_base": (
            "base"
        ),
        "ca_urc": (
            "final"
        ),
    }

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
                "both_high_motion_tail",
            ):
                for metric in (
                    "mae",
                    "bias",
                    "factor2",
                    "under05",
                ):
                    pair_delta = (
                        pairwise_metric_delta_frame(
                            original,
                            pairs,
                            quantity,
                            prefix,
                            metric,
                            population,
                        )
                    )

                    event_delta = (
                        aggregate_pair_delta_to_events(
                            pair_delta
                        )
                    )

                    if event_delta.empty:
                        continue

                    key = (
                        float(
                            caliper_km
                        ),
                        "unseen_minus_seen",
                        model_name,
                        quantity,
                        population,
                        metric,
                    )

                    event_delta_store[
                        key
                    ] = (
                        event_delta
                    )

                    rows.append(
                        {
                            "caliper_km": float(
                                caliper_km
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
                            "n_matched_pairs": int(
                                len(
                                    pair_delta
                                )
                            ),
                            "n_paired_events": int(
                                len(
                                    event_delta
                                )
                            ),
                            "mean_delta_unseen_minus_seen": float(
                                event_delta.mean()
                            ),
                            "median_event_delta_unseen_minus_seen": float(
                                event_delta.median()
                            ),
                        }
                    )

    return (
        pd.DataFrame(
            rows
        ),
        event_delta_store,
    )


# ---------------------------------------------------------------------
# Event bootstrap
# ---------------------------------------------------------------------

def bootstrap_event_delta(
    event_delta: pd.Series,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | bool | int]:
    values = event_delta.to_numpy(
        dtype=float
    )

    n = len(
        values
    )

    if n == 0:
        return {
            "n_events": 0,
            "point_delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_excludes_zero": False,
            "two_sided_bootstrap_sign_p": np.nan,
        }

    rng = np.random.default_rng(
        seed
    )

    sample_indices = rng.integers(
        0,
        n,
        size=(
            repetitions,
            n,
        ),
    )

    distribution = values[
        sample_indices
    ].mean(
        axis=1
    )

    alpha = (
        1.0
        - confidence_level
    )

    ci_lower = float(
        np.quantile(
            distribution,
            alpha
            / 2.0,
        )
    )

    ci_upper = float(
        np.quantile(
            distribution,
            1.0
            - alpha
            / 2.0,
        )
    )

    probability_le_zero = (
        np.sum(
            distribution
            <= 0.0
        )
        + 1.0
    ) / (
        repetitions
        + 1.0
    )

    probability_ge_zero = (
        np.sum(
            distribution
            >= 0.0
        )
        + 1.0
    ) / (
        repetitions
        + 1.0
    )

    p_two_sided = min(
        1.0,
        2.0
        * min(
            probability_le_zero,
            probability_ge_zero,
        ),
    )

    return {
        "n_events": int(
            n
        ),
        "point_delta": float(
            values.mean()
        ),
        "ci_lower": (
            ci_lower
        ),
        "ci_upper": (
            ci_upper
        ),
        "ci_excludes_zero": bool(
            (
                ci_lower
                > 0.0
            )
            or (
                ci_upper
                < 0.0
            )
        ),
        "two_sided_bootstrap_sign_p": float(
            p_two_sided
        ),
    }


def build_bootstrap_table(
    event_delta_stores: list[
        dict[
            tuple,
            pd.Series,
        ]
    ],
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    rows = []

    merged_store = {}

    for store in event_delta_stores:
        merged_store.update(
            store
        )

    for index, (
        key,
        event_delta,
    ) in enumerate(
        merged_store.items()
    ):
        (
            caliper,
            comparison_type,
            group_or_model,
            quantity,
            population,
            metric,
        ) = key

        stats = bootstrap_event_delta(
            event_delta,
            repetitions,
            seed
            + index
            * 9973,
            confidence_level,
        )

        rows.append(
            {
                "caliper_km": float(
                    caliper
                ),
                "comparison_type": (
                    comparison_type
                ),
                "group_or_model": (
                    group_or_model
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
                **stats,
                "bootstrap_repetitions": int(
                    repetitions
                ),
                "bootstrap_unit": (
                    "event"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Paper-focused tables
# ---------------------------------------------------------------------

def build_primary_paper_table(
    caurc_minus_base: pd.DataFrame,
    unseen_minus_seen: pd.DataFrame,
    primary_caliper: float,
) -> pd.DataFrame:
    rows = []

    # Primary question 1:
    # Does CA-URC still improve completely unseen stations after matching?
    subset = caurc_minus_base.loc[
        (
            np.isclose(
                caurc_minus_base[
                    "caliper_km"
                ],
                primary_caliper,
            )
        )
        & (
            caurc_minus_base[
                "target_role"
            ]
            == "unseen_station_target"
        )
        & (
            caurc_minus_base[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                    "factor2",
                ]
            )
        )
    ].copy()

    for row in subset.itertuples(
        index=False
    ):
        rows.append(
            {
                "analysis": (
                    "CA-URC vs Base on geometry-matched unseen targets"
                ),
                "caliper_km": float(
                    row.caliper_km
                ),
                "quantity": (
                    row.quantity
                ),
                "population": (
                    row.population
                ),
                "metric": (
                    row.metric
                ),
                "reference_value": float(
                    row.base_value
                ),
                "candidate_value": float(
                    row.ca_urc_value
                ),
                "delta": float(
                    row.mean_delta_caurc_minus_base
                ),
                "relative_change_percent": (
                    float(
                        row.relative_change_percent
                    )
                    if hasattr(
                        row,
                        "relative_change_percent"
                    )
                    and np.isfinite(
                        row.relative_change_percent
                    )
                    else np.nan
                ),
                "delta_percentage_points": (
                    float(
                        row.delta_percentage_points
                    )
                    if hasattr(
                        row,
                        "delta_percentage_points"
                    )
                    and np.isfinite(
                        row.delta_percentage_points
                    )
                    else np.nan
                ),
                "n_events": int(
                    row.n_paired_events
                ),
            }
        )

    # Primary question 2:
    # Residual matched unseen-vs-seen gap at 20 km.
    subset2 = unseen_minus_seen.loc[
        (
            np.isclose(
                unseen_minus_seen[
                    "caliper_km"
                ],
                primary_caliper,
            )
        )
        & (
            unseen_minus_seen[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                ]
            )
        )
    ].copy()

    for row in subset2.itertuples(
        index=False
    ):
        rows.append(
            {
                "analysis": (
                    "Residual matched unseen-minus-seen OOD gap"
                ),
                "caliper_km": float(
                    row.caliper_km
                ),
                "quantity": (
                    row.quantity
                ),
                "population": (
                    row.population
                ),
                "metric": (
                    row.metric
                ),
                "reference_value": np.nan,
                "candidate_value": np.nan,
                "delta": float(
                    row.mean_delta_unseen_minus_seen
                ),
                "relative_change_percent": np.nan,
                "delta_percentage_points": (
                    100.0
                    * float(
                        row.mean_delta_unseen_minus_seen
                    )
                    if row.metric
                    == "under05"
                    else np.nan
                ),
                "n_events": int(
                    row.n_paired_events
                ),
                "model": (
                    row.model
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "runs/"
            "caurc_station_ood/"
            "repeated_station_ood_predictions.csv"
        ),
    )

    parser.add_argument(
        "--calipers-km",
        default="10,20,30",
    )

    parser.add_argument(
        "--primary-caliper-km",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "caurc_station_ood_matched_geometry"
        ),
    )

    args = parser.parse_args()

    if (
        args.bootstrap_repetitions
        < 1000
    ):
        raise ValueError(
            "--bootstrap-repetitions should be >= 1000."
        )

    if not (
        0.0
        < args.confidence_level
        < 1.0
    ):
        raise ValueError(
            "--confidence-level must be in (0, 1)."
        )

    calipers = parse_float_list(
        args.calipers_km
    )

    if not any(
        np.isclose(
            caliper,
            args.primary_caliper_km,
        )
        for caliper
        in calipers
    ):
        raise ValueError(
            "--primary-caliper-km must be included "
            "in --calipers-km."
        )

    predictions = load_predictions(
        Path(
            args.predictions
        )
    )

    protocol_audit = (
        audit_station_ood_protocol(
            predictions
        )
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== CA-URC Matched-Geometry Station-OOD Audit ==="
    )

    print(
        f"Prediction rows           : "
        f"{len(predictions):,}"
    )

    print(
        f"Events                    : "
        f"{predictions['event_id'].nunique():,}"
    )

    print(
        "Event-repeat groups       : "
        f"{protocol_audit['event_repeat_groups']:,}"
    )

    print(
        "Seen / unseen rows        : "
        f"{protocol_audit['seen_rows']:,}/"
        f"{protocol_audit['unseen_rows']:,}"
    )

    print(
        "Matching variables        : "
        "exact P-wave state + nearest-input distance ONLY"
    )

    print(
        "Outcome-dependent matching: FALSE"
    )

    print(
        f"Calipers                  : "
        f"{calipers}"
    )

    print(
        f"Primary caliper           : "
        f"{args.primary_caliper_km:g} km"
    )

    balance_rows = []

    all_role_metrics = []
    all_caurc_minus_base = []
    all_unseen_minus_seen = []

    bootstrap_stores = []

    for caliper in calipers:
        print(
            "\n"
            + "="
            * 82
        )

        print(
            f"Matching caliper = "
            f"{caliper:g} km"
        )

        print(
            "="
            * 82
        )

        pairs = build_matches(
            predictions,
            caliper,
        )

        pair_path = (
            out_dir
            / (
                "matched_pairs_"
                f"{caliper:g}km.csv"
            )
        )

        pairs.to_csv(
            pair_path,
            index=False,
        )

        balance = matching_balance_summary(
            predictions,
            pairs,
            caliper,
        )

        balance_rows.append(
            balance
        )

        print(
            "Matched groups / total    : "
            f"{balance['matched_event_repeat_groups']:,}/"
            f"{balance['total_event_repeat_groups']:,}"
        )

        print(
            f"Matched pairs             : "
            f"{balance['matched_pairs']:,}"
        )

        print(
            f"Matched events            : "
            f"{balance['matched_events']:,}"
        )

        print(
            "Median |distance diff|    : "
            f"{balance.get('median_abs_distance_difference_km', np.nan):.3f} km"
        )

        print(
            "Mean unseen-seen distance : "
            f"{balance.get('mean_unseen_minus_seen_matched_km', np.nan):+.3f} km"
        )

        matched_long = build_matched_long(
            predictions,
            pairs,
        )

        if matched_long.empty:
            continue

        role_metrics = (
            matched_role_metric_table(
                matched_long
            )
        )

        all_role_metrics.append(
            role_metrics
        )

        (
            caurc_minus_base,
            store1,
        ) = (
            caurc_minus_base_by_matched_role(
                matched_long
            )
        )

        all_caurc_minus_base.append(
            caurc_minus_base
        )

        bootstrap_stores.append(
            store1
        )

        (
            unseen_minus_seen,
            store2,
        ) = (
            unseen_minus_seen_pair_table(
                predictions,
                pairs,
                caliper,
            )
        )

        all_unseen_minus_seen.append(
            unseen_minus_seen
        )

        bootstrap_stores.append(
            store2
        )

    balance_df = pd.DataFrame(
        balance_rows
    )

    balance_path = (
        out_dir
        / "matching_balance_summary.csv"
    )

    balance_df.to_csv(
        balance_path,
        index=False,
    )

    role_metrics_df = pd.concat(
        all_role_metrics,
        ignore_index=True,
    )

    role_metrics_path = (
        out_dir
        / "matched_role_metrics.csv"
    )

    role_metrics_df.to_csv(
        role_metrics_path,
        index=False,
    )

    caurc_minus_base_df = pd.concat(
        all_caurc_minus_base,
        ignore_index=True,
    )

    caurc_minus_base_path = (
        out_dir
        / "caurc_minus_base_matched_by_role.csv"
    )

    caurc_minus_base_df.to_csv(
        caurc_minus_base_path,
        index=False,
    )

    unseen_minus_seen_df = pd.concat(
        all_unseen_minus_seen,
        ignore_index=True,
    )

    unseen_minus_seen_path = (
        out_dir
        / "unseen_minus_seen_matched_pair_deltas.csv"
    )

    unseen_minus_seen_df.to_csv(
        unseen_minus_seen_path,
        index=False,
    )

    bootstrap_df = build_bootstrap_table(
        bootstrap_stores,
        args.bootstrap_repetitions,
        args.seed,
        args.confidence_level,
    )

    bootstrap_path = (
        out_dir
        / "matched_geometry_event_bootstrap.csv"
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    paper_table = build_primary_paper_table(
        caurc_minus_base_df,
        unseen_minus_seen_df,
        args.primary_caliper_km,
    )

    paper_table_path = (
        out_dir
        / "paper_matched_geometry_table.csv"
    )

    paper_table.to_csv(
        paper_table_path,
        index=False,
    )

    configuration = {
        **vars(
            args
        ),
        "calipers_km": (
            calipers
        ),
        "protocol_audit": (
            protocol_audit
        ),
        "matching_objective": (
            "maximize retained pairs, then minimize total absolute "
            "nearest-input-distance difference"
        ),
        "exact_p_wave_state_matching": (
            True
        ),
        "matching_features": [
            "target_p_wave_reached",
            "nearest_input_distance_km",
        ],
        "matching_excludes": [
            "true PGA/PGV",
            "predicted PGA/PGV",
            "prediction error",
            "tail labels",
            "underprediction labels",
        ],
        "metric_aggregation": (
            "matched targets/pairs -> repeats -> events"
        ),
        "bootstrap_unit": (
            "event"
        ),
        "training_performed": (
            False
        ),
        "model_selection_performed": (
            False
        ),
    }

    config_path = (
        out_dir
        / "run_configuration.json"
    )

    config_path.write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit_path = (
        out_dir
        / "protocol_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            protocol_audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Console: primary 20-km result.
    # -------------------------------------------------------------
    primary = float(
        args.primary_caliper_km
    )

    print(
        "\n"
        + "="
        * 90
    )

    print(
        f"PRIMARY {primary:g}-km MATCHED-GEOMETRY RESULTS"
    )

    print(
        "="
        * 90
    )

    primary_balance = balance_df.loc[
        np.isclose(
            balance_df[
                "caliper_km"
            ],
            primary,
        )
    ]

    print(
        "\n=== Matching balance ==="
    )

    print(
        primary_balance.to_string(
            index=False
        )
    )

    primary_effect = (
        caurc_minus_base_df.loc[
            (
                np.isclose(
                    caurc_minus_base_df[
                        "caliper_km"
                    ],
                    primary,
                )
            )
            & (
                caurc_minus_base_df[
                    "metric"
                ].isin(
                    [
                        "mae",
                        "under05",
                        "factor2",
                    ]
                )
            )
        ]
    )

    print(
        "\n=== CA-URC minus Base on matched targets ==="
    )

    print(
        primary_effect[
            [
                "target_role",
                "quantity",
                "population",
                "metric",
                "n_paired_events",
                "base_value",
                "ca_urc_value",
                "mean_delta_caurc_minus_base",
                "relative_change_percent",
                "delta_percentage_points",
            ]
        ].to_string(
            index=False
        )
    )

    primary_gap = (
        unseen_minus_seen_df.loc[
            (
                np.isclose(
                    unseen_minus_seen_df[
                        "caliper_km"
                    ],
                    primary,
                )
            )
            & (
                unseen_minus_seen_df[
                    "metric"
                ].isin(
                    [
                        "mae",
                        "under05",
                        "factor2",
                    ]
                )
            )
        ]
    )

    print(
        "\n=== Residual unseen-minus-seen gap after matching ==="
    )

    print(
        primary_gap.to_string(
            index=False
        )
    )

    primary_bootstrap = bootstrap_df.loc[
        (
            np.isclose(
                bootstrap_df[
                    "caliper_km"
                ],
                primary,
            )
        )
        & (
            bootstrap_df[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                    "factor2",
                ]
            )
        )
    ].copy()

    print(
        "\n=== 20-km event bootstrap ==="
    )

    print(
        primary_bootstrap.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    for path in [
        balance_path,
        role_metrics_path,
        caurc_minus_base_path,
        unseen_minus_seen_path,
        bootstrap_path,
        paper_table_path,
        config_path,
        audit_path,
    ]:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
