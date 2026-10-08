#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
62_sequence_aware_paired_bootstrap_cadrg.py

Sequence-aware paired statistical inference for the final CA-DRG grouped test.

Primary question
----------------
Does CA-DRG improve over the SAME paired target predictions from:
    1) frozen Cross-Attention Base
    2) Graph baseline
    3) Tail-weighted Attention
    4) previous Power Gate

without treating correlated earthquakes from the same seismic sequence as
independent realizations?

Inputs
------
A) CA-DRG locked predictions:
   runs/cross_attention_dual_risk_previous_grouped_benchmark/
       locked_dual_risk_predictions.csv

B) Completed strong-baseline paired predictions:
   runs/final_strong_baselines_reuse_locked/
       final_strong_baseline_reused_locked_predictions.csv

C) Grouped scenario manifest containing:
       event_id
       split_grouped
       sequence_group

Statistical hierarchy
---------------------
Target rows -> repeat mean -> event mean -> sequence-aware bootstrap

Two bootstrap analyses are exported:

1. sequence_cluster
   Sample sequence groups with replacement. When a sequence is selected,
   all of its event-level paired differences are included.

2. hierarchical_sequence_event
   First sample sequence groups with replacement, then sample events with
   replacement within every selected sequence. This is the primary
   sequence-aware analysis recommended for the manuscript.

The point estimate always remains the original canonical event-balanced
paired difference:
    mean_event(candidate - reference)

Bootstrap resampling is used ONLY for uncertainty / CI / p-values.

Metrics
-------
overall and high_motion_tail:
    MAE       : lower is better
    signed bias
    Factor-2  : higher is better
    U0.5      : lower is better

Delta convention
----------------
    delta = CA-DRG - reference

Therefore:
    MAE/U0.5: negative delta favors CA-DRG
    Factor-2: positive delta favors CA-DRG
    bias: signed difference; no automatic "better" direction

Important
---------
Repeats are NOT treated as independent earthquakes. They are averaged
inside each event before sequence-aware resampling.

This script performs NO training and NO model selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
SEVERE_UNDER_THRESHOLD = -0.5


METHOD_COLUMNS = {
    "ca_drg": {
        "pga": "final_log10_pga",
        "pgv": "final_log10_pgv",
        "source": "new",
    },
    "cross_attention": {
        "pga": "base_log10_pga",
        "pgv": "base_log10_pgv",
        "source": "new",
    },
    "graph": {
        "pga": "graph_log10_pga",
        "pgv": "graph_log10_pgv",
        "source": "baseline",
    },
    "tail_weighted_attention": {
        "pga": "tail_weighted_attention_log10_pga",
        "pgv": "tail_weighted_attention_log10_pgv",
        "source": "baseline",
    },
    "old_power_gate": {
        "pga": "power_gate_log10_pga",
        "pgv": "power_gate_log10_pgv",
        "source": "baseline",
    },
}


DEFAULT_COMPARISONS = (
    "ca_drg:cross_attention,"
    "ca_drg:graph,"
    "ca_drg:tail_weighted_attention,"
    "ca_drg:old_power_gate"
)


def parse_comparisons(
    text: str,
) -> list[tuple[str, str]]:
    """
    candidate:reference,candidate:reference,...
    """
    pairs = []

    for item in str(text).split(","):
        item = item.strip()

        if not item:
            continue

        if ":" not in item:
            raise ValueError(
                "Each comparison must be candidate:reference; "
                f"got {item!r}."
            )

        candidate, reference = [
            part.strip()
            for part in item.split(
                ":",
                1,
            )
        ]

        if candidate not in METHOD_COLUMNS:
            raise ValueError(
                f"Unknown candidate method: {candidate}"
            )

        if reference not in METHOD_COLUMNS:
            raise ValueError(
                f"Unknown reference method: {reference}"
            )

        if candidate == reference:
            raise ValueError(
                f"Candidate and reference are identical: {candidate}"
            )

        pairs.append(
            (
                candidate,
                reference,
            )
        )

    if not pairs:
        raise ValueError(
            "No comparisons supplied."
        )

    return pairs


def as_str_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
    )


def as_bool_array(
    series: pd.Series,
) -> np.ndarray:
    """
    Robustly parse CSV booleans. Avoid bool("False") == True.
    """
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

    allowed = {
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
        "y",
        "n",
        "t",
        "f",
    }

    unknown = set(
        normalized.unique()
    ).difference(
        allowed
    )

    if unknown:
        raise ValueError(
            "Cannot parse boolean values: "
            f"{sorted(unknown)[:20]}"
        )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    ).to_numpy(
        dtype=bool
    )


def load_and_pair_predictions(
    new_path: Path,
    baseline_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    if not new_path.exists():
        raise FileNotFoundError(
            new_path
        )

    if not baseline_path.exists():
        raise FileNotFoundError(
            baseline_path
        )

    new = pd.read_csv(
        new_path,
        dtype={
            "event_id": str,
        },
    )

    baseline = pd.read_csv(
        baseline_path,
        dtype={
            "event_id": str,
        },
    )

    keys = [
        "event_id",
        "repeat",
        "target_station_index",
    ]

    required_new = {
        *keys,
        "true_log10_pga",
        "true_log10_pgv",
        "base_log10_pga",
        "base_log10_pgv",
        "final_log10_pga",
        "final_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
    }

    required_baseline = {
        *keys,
        "true_log10_pga",
        "true_log10_pgv",
        "cross_attention_log10_pga",
        "cross_attention_log10_pgv",
        "graph_log10_pga",
        "graph_log10_pgv",
        "tail_weighted_attention_log10_pga",
        "tail_weighted_attention_log10_pgv",
        "power_gate_log10_pga",
        "power_gate_log10_pgv",
    }

    missing_new = (
        required_new
        .difference(
            new.columns
        )
    )

    missing_baseline = (
        required_baseline
        .difference(
            baseline.columns
        )
    )

    if missing_new:
        raise ValueError(
            "CA-DRG prediction file missing columns: "
            f"{sorted(missing_new)}"
        )

    if missing_baseline:
        raise ValueError(
            "Strong-baseline prediction file missing columns: "
            f"{sorted(missing_baseline)}"
        )

    new[
        "event_id"
    ] = as_str_series(
        new[
            "event_id"
        ]
    )

    baseline[
        "event_id"
    ] = as_str_series(
        baseline[
            "event_id"
        ]
    )

    if new.duplicated(
        keys
    ).any():
        raise ValueError(
            "CA-DRG predictions are not unique on "
            f"{keys}."
        )

    if baseline.duplicated(
        keys
    ).any():
        raise ValueError(
            "Strong-baseline predictions are not unique on "
            f"{keys}."
        )

    baseline_keep = (
        keys
        + [
            "true_log10_pga",
            "true_log10_pgv",
            "cross_attention_log10_pga",
            "cross_attention_log10_pgv",
            "graph_log10_pga",
            "graph_log10_pgv",
            "tail_weighted_attention_log10_pga",
            "tail_weighted_attention_log10_pgv",
            "power_gate_log10_pga",
            "power_gate_log10_pgv",
        ]
    )

    paired = new.merge(
        baseline[
            baseline_keep
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_baseline",
        ),
    )

    audit = {
        "new_prediction_path": str(
            new_path.resolve()
        ),
        "baseline_prediction_path": str(
            baseline_path.resolve()
        ),
        "new_rows": int(
            len(
                new
            )
        ),
        "baseline_rows": int(
            len(
                baseline
            )
        ),
        "paired_rows": int(
            len(
                paired
            )
        ),
        "exact_row_pairing": bool(
            len(
                paired
            )
            == len(
                new
            )
            == len(
                baseline
            )
        ),
    }

    if not audit[
        "exact_row_pairing"
    ]:
        raise RuntimeError(
            "Prediction files do not contain the exact same paired rows. "
            f"Audit: {audit}"
        )

    for quantity in (
        "pga",
        "pgv",
    ):
        truth_current = paired[
            f"true_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        truth_baseline = paired[
            f"true_log10_{quantity}_baseline"
        ].to_numpy(
            dtype=float
        )

        audit[
            f"max_abs_truth_diff_{quantity}"
        ] = float(
            np.max(
                np.abs(
                    truth_current
                    - truth_baseline
                )
            )
        )

        base_current = paired[
            f"base_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        base_baseline = paired[
            f"cross_attention_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        audit[
            f"max_abs_cross_attention_diff_{quantity}"
        ] = float(
            np.max(
                np.abs(
                    base_current
                    - base_baseline
                )
            )
        )

    tolerance = 1e-5

    for key, value in audit.items():
        if (
            key.startswith(
                "max_abs_"
            )
            and float(
                value
            )
            > tolerance
        ):
            raise RuntimeError(
                "Pairing/model audit failed: "
                f"{key}={value:.6e} > {tolerance:.1e}"
            )

    paired = paired.drop(
        columns=[
            "true_log10_pga_baseline",
            "true_log10_pgv_baseline",
        ]
    )

    return (
        paired,
        audit,
    )


def attach_sequence_groups(
    frame: pd.DataFrame,
    manifest_path: Path,
    split_column: str,
    test_label: str,
    sequence_column: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "event_id": str,
        },
    )

    required = {
        "event_id",
        split_column,
        sequence_column,
    }

    missing = required.difference(
        manifest.columns
    )

    if missing:
        raise ValueError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    manifest[
        "event_id"
    ] = as_str_series(
        manifest[
            "event_id"
        ]
    )

    test_manifest = (
        manifest.loc[
            manifest[
                split_column
            ]
            .astype(str)
            .eq(
                str(
                    test_label
                )
            ),
            [
                "event_id",
                sequence_column,
            ],
        ]
        .drop_duplicates(
            "event_id"
        )
        .copy()
    )

    if test_manifest[
        "event_id"
    ].duplicated().any():
        raise ValueError(
            "Manifest has duplicate event IDs after test filtering."
        )

    result = frame.merge(
        test_manifest,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    if result[
        sequence_column
    ].isna().any():
        missing_events = (
            result.loc[
                result[
                    sequence_column
                ].isna(),
                "event_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Missing sequence-group assignment for prediction events: "
            f"{missing_events[:20]}"
        )

    result[
        "sequence_group"
    ] = result[
        sequence_column
    ].astype(str)

    if (
        sequence_column
        != "sequence_group"
    ):
        result = result.drop(
            columns=[
                sequence_column
            ]
        )

    event_group = (
        result[
            [
                "event_id",
                "sequence_group",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "sequence_group",
                "event_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if event_group[
        "event_id"
    ].duplicated().any():
        raise RuntimeError(
            "An event maps to multiple sequence groups."
        )

    return (
        result,
        event_group,
    )


def metric_values(
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


def build_event_level_metrics(
    frame: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    """
    Exact manuscript hierarchy:
        target -> repeat -> event

    For tail metrics, only repeats containing at least one tail target
    contribute, exactly matching the existing canonical metric code.
    """

    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        truth = frame[
            f"true_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        tail = as_bool_array(
            frame[
                f"is_tail_{quantity}"
            ]
        )

        populations = {
            "overall": np.ones(
                len(
                    frame
                ),
                dtype=bool,
            ),
            "high_motion_tail": (
                tail
            ),
        }

        for method in methods:
            column = (
                METHOD_COLUMNS[
                    method
                ][
                    quantity
                ]
            )

            prediction = frame[
                column
            ].to_numpy(
                dtype=float
            )

            for metric in (
                "mae",
                "bias",
                "factor2",
                "under05",
            ):
                values = metric_values(
                    truth,
                    prediction,
                    metric,
                )

                for (
                    population,
                    mask,
                ) in populations.items():
                    if not mask.any():
                        continue

                    work = frame.loc[
                        mask,
                        [
                            "event_id",
                            "repeat",
                            "sequence_group",
                        ],
                    ].copy()

                    work[
                        "value"
                    ] = values[
                        mask
                    ]

                    event_repeat = (
                        work.groupby(
                            [
                                "event_id",
                                "repeat",
                                "sequence_group",
                            ],
                            sort=False,
                            as_index=False,
                        )[
                            "value"
                        ]
                        .mean()
                    )

                    event = (
                        event_repeat.groupby(
                            [
                                "event_id",
                                "sequence_group",
                            ],
                            sort=False,
                            as_index=False,
                        )[
                            "value"
                        ]
                        .mean()
                    )

                    event[
                        "method"
                    ] = method

                    event[
                        "quantity"
                    ] = quantity

                    event[
                        "population"
                    ] = population

                    event[
                        "metric"
                    ] = metric

                    rows.append(
                        event
                    )

    if not rows:
        raise RuntimeError(
            "No event-level metric rows were generated."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_delta_matrix(
    event_metrics: pd.DataFrame,
    event_group: pd.DataFrame,
    comparisons: list[
        tuple[
            str,
            str,
        ]
    ],
) -> tuple[
    np.ndarray,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    all_events = (
        event_group[
            "event_id"
        ]
        .astype(str)
        .tolist()
    )

    event_to_position = {
        event_id: index
        for index, event_id
        in enumerate(
            all_events
        )
    }

    columns = []
    vectors = []

    for (
        candidate,
        reference,
    ) in comparisons:
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
                    "under05",
                ):
                    subset = event_metrics.loc[
                        (
                            event_metrics[
                                "quantity"
                            ]
                            == quantity
                        )
                        & (
                            event_metrics[
                                "population"
                            ]
                            == population
                        )
                        & (
                            event_metrics[
                                "metric"
                            ]
                            == metric
                        )
                        & (
                            event_metrics[
                                "method"
                            ].isin(
                                [
                                    candidate,
                                    reference,
                                ]
                            )
                        )
                    ].copy()

                    pivot = subset.pivot(
                        index="event_id",
                        columns="method",
                        values="value",
                    )

                    if (
                        candidate
                        not in pivot.columns
                        or reference
                        not in pivot.columns
                    ):
                        raise RuntimeError(
                            "Missing paired event metrics for "
                            f"{candidate} vs {reference}, "
                            f"{quantity}, {population}, {metric}."
                        )

                    paired = pivot[
                        [
                            candidate,
                            reference,
                        ]
                    ].dropna()

                    if paired.empty:
                        raise RuntimeError(
                            "No paired events for "
                            f"{candidate} vs {reference}, "
                            f"{quantity}, {population}, {metric}."
                        )

                    delta = (
                        paired[
                            candidate
                        ]
                        - paired[
                            reference
                        ]
                    )

                    vector = np.full(
                        len(
                            all_events
                        ),
                        np.nan,
                        dtype=np.float64,
                    )

                    for (
                        event_id,
                        value,
                    ) in delta.items():
                        vector[
                            event_to_position[
                                str(
                                    event_id
                                )
                            ]
                        ] = float(
                            value
                        )

                    candidate_value = float(
                        paired[
                            candidate
                        ].mean()
                    )

                    reference_value = float(
                        paired[
                            reference
                        ].mean()
                    )

                    point_delta = float(
                        delta.mean()
                    )

                    groups_used = (
                        event_group.loc[
                            event_group[
                                "event_id"
                            ].isin(
                                paired.index.astype(
                                    str
                                )
                            ),
                            "sequence_group",
                        ]
                        .nunique()
                    )

                    metadata = {
                        "candidate": candidate,
                        "reference": reference,
                        "quantity": quantity,
                        "population": (
                            population
                        ),
                        "metric": metric,
                        "candidate_value": (
                            candidate_value
                        ),
                        "reference_value": (
                            reference_value
                        ),
                        "point_delta_candidate_minus_reference": (
                            point_delta
                        ),
                        "n_paired_events": int(
                            len(
                                paired
                            )
                        ),
                        "n_sequence_groups_used": int(
                            groups_used
                        ),
                    }

                    if metric == "mae":
                        metadata[
                            "relative_change_percent"
                        ] = (
                            100.0
                            * point_delta
                            / max(
                                abs(
                                    reference_value
                                ),
                                1e-12,
                            )
                        )

                    elif metric in (
                        "factor2",
                        "under05",
                    ):
                        metadata[
                            "delta_percentage_points"
                        ] = (
                            100.0
                            * point_delta
                        )

                    vectors.append(
                        vector
                    )

                    columns.append(
                        metadata
                    )

    matrix = np.stack(
        vectors,
        axis=1,
    )

    return (
        matrix,
        columns,
    )


def group_index_arrays(
    event_group: pd.DataFrame,
) -> tuple[
    list[str],
    list[np.ndarray],
]:
    event_group = (
        event_group.reset_index(
            drop=True
        )
    )

    group_names = []
    index_arrays = []

    for (
        group,
        subset,
    ) in event_group.groupby(
        "sequence_group",
        sort=True,
    ):
        group_names.append(
            str(
                group
            )
        )

        index_arrays.append(
            subset.index.to_numpy(
                dtype=np.int64
            )
        )

    return (
        group_names,
        index_arrays,
    )


def sequence_cluster_bootstrap(
    delta_matrix: np.ndarray,
    group_indices: list[np.ndarray],
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """
    Standard cluster bootstrap at the seismic-sequence level.

    Sequence groups are sampled with replacement. Every event from each
    selected sequence contributes to the replicate statistic.
    """

    rng = np.random.default_rng(
        seed
    )

    n_groups = len(
        group_indices
    )

    n_columns = (
        delta_matrix.shape[
            1
        ]
    )

    group_sum = np.zeros(
        (
            n_groups,
            n_columns,
        ),
        dtype=np.float64,
    )

    group_count = np.zeros(
        (
            n_groups,
            n_columns,
        ),
        dtype=np.float64,
    )

    for g, indices in enumerate(
        group_indices
    ):
        values = delta_matrix[
            indices,
            :,
        ]

        finite = np.isfinite(
            values
        )

        group_sum[
            g,
            :,
        ] = np.nansum(
            values,
            axis=0,
        )

        group_count[
            g,
            :,
        ] = finite.sum(
            axis=0
        )

    group_draw_counts = (
        rng.multinomial(
            n_groups,
            np.full(
                n_groups,
                1.0
                / n_groups,
            ),
            size=repetitions,
        )
        .astype(
            np.float64
        )
    )

    numerator = (
        group_draw_counts
        @ group_sum
    )

    denominator = (
        group_draw_counts
        @ group_count
    )

    result = np.full(
        (
            repetitions,
            n_columns,
        ),
        np.nan,
        dtype=np.float64,
    )

    valid = (
        denominator
        > 0
    )

    result[
        valid
    ] = (
        numerator[
            valid
        ]
        / denominator[
            valid
        ]
    )

    return result


def hierarchical_sequence_event_bootstrap(
    delta_matrix: np.ndarray,
    group_indices: list[np.ndarray],
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """
    Two-level hierarchical bootstrap:
        sequence group -> event

    For each bootstrap replicate:
      1. draw G sequence groups with replacement;
      2. if group g is selected c times, draw c*n_g events with replacement
         from that group;
      3. compute the event-balanced paired delta over all sampled events.

    All outcome columns share the exact same resampling realization.
    """

    rng = np.random.default_rng(
        seed
    )

    n_groups = len(
        group_indices
    )

    n_columns = (
        delta_matrix.shape[
            1
        ]
    )

    result = np.full(
        (
            repetitions,
            n_columns,
        ),
        np.nan,
        dtype=np.float64,
    )

    probabilities = np.full(
        n_groups,
        1.0
        / n_groups,
        dtype=np.float64,
    )

    group_draw_counts = (
        rng.multinomial(
            n_groups,
            probabilities,
            size=repetitions,
        )
    )

    for bootstrap_index in range(
        repetitions
    ):
        numerator = np.zeros(
            n_columns,
            dtype=np.float64,
        )

        denominator = np.zeros(
            n_columns,
            dtype=np.float64,
        )

        counts = group_draw_counts[
            bootstrap_index
        ]

        selected_groups = np.flatnonzero(
            counts
            > 0
        )

        for group_index in selected_groups:
            event_indices = group_indices[
                group_index
            ]

            n_events = len(
                event_indices
            )

            n_draws = int(
                counts[
                    group_index
                ]
                * n_events
            )

            sampled_local = (
                rng.integers(
                    0,
                    n_events,
                    size=n_draws,
                )
            )

            sampled_event_indices = (
                event_indices[
                    sampled_local
                ]
            )

            values = delta_matrix[
                sampled_event_indices,
                :,
            ]

            finite = np.isfinite(
                values
            )

            numerator += np.nansum(
                values,
                axis=0,
            )

            denominator += finite.sum(
                axis=0
            )

        valid = (
            denominator
            > 0
        )

        result[
            bootstrap_index,
            valid,
        ] = (
            numerator[
                valid
            ]
            / denominator[
                valid
            ]
        )

    return result


def bootstrap_summary(
    distributions: np.ndarray,
    metadata: list[
        dict[
            str,
            Any,
        ]
    ],
    mode: str,
    confidence_level: float,
) -> pd.DataFrame:
    alpha = (
        1.0
        - float(
            confidence_level
        )
    )

    lower_q = (
        alpha
        / 2.0
    )

    upper_q = (
        1.0
        - alpha
        / 2.0
    )

    rows = []

    for column_index, meta in enumerate(
        metadata
    ):
        values = distributions[
            :,
            column_index,
        ]

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(
            values
        ) == 0:
            raise RuntimeError(
                "No valid bootstrap replicates for "
                f"{meta}"
            )

        lower = float(
            np.quantile(
                values,
                lower_q,
            )
        )

        upper = float(
            np.quantile(
                values,
                upper_q,
            )
        )

        point = float(
            meta[
                "point_delta_candidate_minus_reference"
            ]
        )

        n = len(
            values
        )

        # Plus-one correction prevents an estimated p-value of exactly zero.
        probability_le_zero = (
            (
                np.sum(
                    values
                    <= 0.0
                )
                + 1.0
            )
            / (
                n
                + 1.0
            )
        )

        probability_ge_zero = (
            (
                np.sum(
                    values
                    >= 0.0
                )
                + 1.0
            )
            / (
                n
                + 1.0
            )
        )

        p_two_sided = min(
            1.0,
            2.0
            * min(
                probability_le_zero,
                probability_ge_zero,
            ),
        )

        metric = str(
            meta[
                "metric"
            ]
        )

        if metric in (
            "mae",
            "under05",
        ):
            preferred_direction = (
                "negative"
            )

            probability_candidate_better = float(
                np.mean(
                    values
                    < 0.0
                )
            )

            point_candidate_better = bool(
                point
                < 0.0
            )

        elif metric == "factor2":
            preferred_direction = (
                "positive"
            )

            probability_candidate_better = float(
                np.mean(
                    values
                    > 0.0
                )
            )

            point_candidate_better = bool(
                point
                > 0.0
            )

        else:
            preferred_direction = (
                "signed_bias_no_single_preferred_direction"
            )

            probability_candidate_better = float(
                "nan"
            )

            point_candidate_better = None

        row = {
            **meta,
            "bootstrap_mode": (
                mode
            ),
            "confidence_level": float(
                confidence_level
            ),
            "bootstrap_repetitions_valid": int(
                n
            ),
            "bootstrap_mean_delta": float(
                np.mean(
                    values
                )
            ),
            "bootstrap_median_delta": float(
                np.median(
                    values
                )
            ),
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_excludes_zero": bool(
                (
                    lower
                    > 0.0
                )
                or (
                    upper
                    < 0.0
                )
            ),
            "two_sided_bootstrap_sign_p": float(
                p_two_sided
            ),
            "preferred_direction_for_candidate": (
                preferred_direction
            ),
            "point_candidate_better": (
                point_candidate_better
            ),
            "bootstrap_probability_candidate_better": (
                probability_candidate_better
            ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--new-predictions",
        default=(
            "runs/"
            "cross_attention_dual_risk_previous_grouped_benchmark/"
            "locked_dual_risk_predictions.csv"
        ),
    )

    parser.add_argument(
        "--baseline-predictions",
        default=(
            "runs/"
            "final_strong_baselines_reuse_locked/"
            "final_strong_baseline_reused_locked_predictions.csv"
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
        "--split-column",
        default="split_grouped",
    )

    parser.add_argument(
        "--test-label",
        default="test",
    )

    parser.add_argument(
        "--sequence-column",
        default="sequence_group",
    )

    parser.add_argument(
        "--comparisons",
        default=(
            DEFAULT_COMPARISONS
        ),
        help=(
            "Comma-separated candidate:reference pairs."
        ),
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
        "--save-distributions",
        action="store_true",
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "runs/"
            "cadrg_sequence_aware_bootstrap"
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
        0.5
        < args.confidence_level
        < 1.0
    ):
        raise ValueError(
            "--confidence-level must be between 0.5 and 1."
        )

    comparisons = parse_comparisons(
        args.comparisons
    )

    methods = sorted(
        {
            method
            for pair in comparisons
            for method in pair
        }
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired, pairing_audit = (
        load_and_pair_predictions(
            Path(
                args.new_predictions
            ),
            Path(
                args.baseline_predictions
            ),
        )
    )

    # Add strong-baseline method columns into the unified frame.
    for method in methods:
        if (
            METHOD_COLUMNS[
                method
            ][
                "source"
            ]
            == "baseline"
        ):
            # These columns already came from the baseline merge.
            pass

    paired, event_group = (
        attach_sequence_groups(
            paired,
            Path(
                args.manifest
            ),
            args.split_column,
            args.test_label,
            args.sequence_column,
        )
    )

    # Ensure the prediction frame exposes every selected method column.
    for method in methods:
        for quantity in (
            "pga",
            "pgv",
        ):
            column = METHOD_COLUMNS[
                method
            ][
                quantity
            ]

            if column not in paired.columns:
                raise ValueError(
                    f"Missing prediction column for {method}: {column}"
                )

    sequence_summary = (
        event_group.groupby(
            "sequence_group",
            sort=True,
        )
        .agg(
            n_events=(
                "event_id",
                "nunique",
            )
        )
        .reset_index()
        .sort_values(
            [
                "n_events",
                "sequence_group",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    sequence_summary.to_csv(
        out_dir
        / "test_sequence_group_summary.csv",
        index=False,
    )

    pairing_audit.update(
        {
            "n_events": int(
                event_group[
                    "event_id"
                ].nunique()
            ),
            "n_sequence_groups": int(
                event_group[
                    "sequence_group"
                ].nunique()
            ),
            "largest_sequence_events": int(
                sequence_summary[
                    "n_events"
                ].max()
            ),
            "comparisons": [
                {
                    "candidate": candidate,
                    "reference": reference,
                }
                for (
                    candidate,
                    reference,
                ) in comparisons
            ],
        }
    )

    (
        out_dir
        / "pairing_and_sequence_audit.json"
    ).write_text(
        json.dumps(
            pairing_audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== CA-DRG sequence-aware paired bootstrap ==="
    )

    print(
        f"Paired target rows        : {len(paired):,}"
    )

    print(
        "Events                    : "
        f"{event_group['event_id'].nunique():,}"
    )

    print(
        "Sequence groups           : "
        f"{event_group['sequence_group'].nunique():,}"
    )

    print(
        "Largest sequence          : "
        f"{sequence_summary['n_events'].max():,} events"
    )

    print(
        "Truth pairing max diff    : "
        "PGA=%.3e, PGV=%.3e"
        % (
            pairing_audit[
                "max_abs_truth_diff_pga"
            ],
            pairing_audit[
                "max_abs_truth_diff_pgv"
            ],
        )
    )

    print(
        "Cross-Attention audit     : "
        "PGA=%.3e, PGV=%.3e"
        % (
            pairing_audit[
                "max_abs_cross_attention_diff_pga"
            ],
            pairing_audit[
                "max_abs_cross_attention_diff_pgv"
            ],
        )
    )

    event_metrics = (
        build_event_level_metrics(
            paired,
            methods,
        )
    )

    event_metrics.to_csv(
        out_dir
        / "event_level_canonical_metrics.csv",
        index=False,
    )

    (
        delta_matrix,
        metadata,
    ) = build_delta_matrix(
        event_metrics,
        event_group,
        comparisons,
    )

    (
        group_names,
        group_indices,
    ) = group_index_arrays(
        event_group
    )

    if (
        len(
            group_names
        )
        < 2
    ):
        raise RuntimeError(
            "Sequence-aware bootstrap requires at least two sequence groups."
        )

    print(
        "\nRunning sequence-cluster bootstrap..."
    )

    cluster_distribution = (
        sequence_cluster_bootstrap(
            delta_matrix,
            group_indices,
            args.bootstrap_repetitions,
            args.seed,
        )
    )

    cluster_summary = (
        bootstrap_summary(
            cluster_distribution,
            metadata,
            "sequence_cluster",
            args.confidence_level,
        )
    )

    cluster_summary.to_csv(
        out_dir
        / "sequence_cluster_bootstrap_summary.csv",
        index=False,
    )

    print(
        "Running hierarchical sequence->event bootstrap..."
    )

    hierarchical_distribution = (
        hierarchical_sequence_event_bootstrap(
            delta_matrix,
            group_indices,
            args.bootstrap_repetitions,
            args.seed
            + 1,
        )
    )

    hierarchical_summary = (
        bootstrap_summary(
            hierarchical_distribution,
            metadata,
            "hierarchical_sequence_event",
            args.confidence_level,
        )
    )

    hierarchical_summary.to_csv(
        out_dir
        / "hierarchical_bootstrap_summary.csv",
        index=False,
    )

    combined = pd.concat(
        [
            hierarchical_summary,
            cluster_summary,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        out_dir
        / "all_sequence_aware_bootstrap_summary.csv",
        index=False,
    )

    if args.save_distributions:
        labels = np.asarray(
            [
                (
                    f"{item['candidate']}__vs__"
                    f"{item['reference']}__"
                    f"{item['quantity']}__"
                    f"{item['population']}__"
                    f"{item['metric']}"
                )
                for item in metadata
            ],
            dtype=object,
        )

        np.savez_compressed(
            out_dir
            / "bootstrap_distributions.npz",
            labels=labels,
            sequence_cluster=(
                cluster_distribution
            ),
            hierarchical_sequence_event=(
                hierarchical_distribution
            ),
        )

    (
        out_dir
        / "run_configuration.json"
    ).write_text(
        json.dumps(
            {
                **vars(
                    args
                ),
                "comparisons_parsed": [
                    {
                        "candidate": candidate,
                        "reference": reference,
                    }
                    for (
                        candidate,
                        reference,
                    ) in comparisons
                ],
                "main_inference": (
                    "hierarchical_sequence_event"
                ),
                "sensitivity_inference": (
                    "sequence_cluster"
                ),
                "metric_hierarchy": (
                    "targets -> repeats -> events -> "
                    "sequence-aware bootstrap"
                ),
                "delta_definition": (
                    "candidate - reference"
                ),
                "repeats_treated_as_independent_events": False,
                "test_used_for_training": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Main manuscript display:
    # CA-DRG vs Cross-Attention, hierarchical inference.
    main_display = hierarchical_summary.loc[
        (
            hierarchical_summary[
                "candidate"
            ]
            == "ca_drg"
        )
        & (
            hierarchical_summary[
                "reference"
            ]
            == "cross_attention"
        )
        & (
            hierarchical_summary[
                "metric"
            ].isin(
                [
                    "mae",
                    "factor2",
                    "under05",
                    "bias",
                ]
            )
        )
    ][
        [
            "quantity",
            "population",
            "metric",
            "reference_value",
            "candidate_value",
            "point_delta_candidate_minus_reference",
            "ci_lower",
            "ci_upper",
            "two_sided_bootstrap_sign_p",
            "n_paired_events",
            "n_sequence_groups_used",
        ]
    ].copy()

    print(
        "\n=== PRIMARY: hierarchical sequence->event bootstrap ==="
    )

    print(
        "Delta = CA-DRG - Cross-Attention Base"
    )

    print(
        main_display.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    output_paths = [
        out_dir
        / "hierarchical_bootstrap_summary.csv",
        out_dir
        / "sequence_cluster_bootstrap_summary.csv",
        out_dir
        / "all_sequence_aware_bootstrap_summary.csv",
        out_dir
        / "event_level_canonical_metrics.csv",
        out_dir
        / "test_sequence_group_summary.csv",
        out_dir
        / "pairing_and_sequence_audit.json",
        out_dir
        / "run_configuration.json",
    ]

    if args.save_distributions:
        output_paths.append(
            out_dir
            / "bootstrap_distributions.npz"
        )

    for path in output_paths:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
