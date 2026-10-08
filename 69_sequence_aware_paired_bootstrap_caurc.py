#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
69_sequence_aware_paired_bootstrap_caurc.py

Sequence-aware paired statistical inference for the locked CA-URC grouped test.

Final candidate model
---------------------
CA-URC = Cross-Attention Underprediction-Risk Correction

    y_final = y_CA + p_under**gamma * Delta

The grouped benchmark has already been validation-locked:
    A4_under_only
    selected epoch = 3
    selected gamma = 5

This script performs NO training and NO model selection.

Primary comparisons
-------------------
1) CA-URC vs Cross-Attention Base
2) CA-URC vs Graph baseline
3) CA-URC vs Tail-weighted Attention
4) CA-URC vs previous Power Gate
5) CA-URC vs former full CA-DRG

Inputs
------
A) Gate-ablation locked predictions:
   runs/cadrg_gate_ablation_A3_A5/
       locked_test_ablation_predictions.csv

   Required final-model columns:
       A4_under_only_log10_pga
       A4_under_only_log10_pgv

   Also contains exact-paired:
       A2_cross_attention_base_log10_*
       A6_full_cadrg_log10_*

B) Strong-baseline exact-paired predictions:
   runs/final_strong_baselines_reuse_locked/
       final_strong_baseline_reused_locked_predictions.csv

C) Grouped scenario manifest containing:
       event_id
       split_grouped
       sequence_group

Statistical hierarchy
----------------------
target rows -> repeat mean -> event mean -> sequence-aware bootstrap

Primary inference:
    hierarchical sequence -> event bootstrap

Sensitivity:
    sequence-cluster bootstrap

Delta convention:
    delta = CA-URC - reference

Therefore:
    MAE / U0.5 : negative favors CA-URC
    Factor-2   : positive favors CA-URC
    signed bias: interpreted relative to zero; no automatic preferred sign

The point estimate remains the original canonical event-balanced paired delta.
Bootstrap is used only for uncertainty / CI / p-values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_COLUMNS = {
    "ca_urc": {
        "pga": "A4_under_only_log10_pga",
        "pgv": "A4_under_only_log10_pgv",
    },
    "cross_attention": {
        "pga": "cross_attention_log10_pga",
        "pgv": "cross_attention_log10_pgv",
    },
    "graph": {
        "pga": "graph_log10_pga",
        "pgv": "graph_log10_pgv",
    },
    "tail_weighted_attention": {
        "pga": "tail_weighted_attention_log10_pga",
        "pgv": "tail_weighted_attention_log10_pgv",
    },
    "old_power_gate": {
        "pga": "power_gate_log10_pga",
        "pgv": "power_gate_log10_pgv",
    },
    "ca_drg": {
        "pga": "A6_full_cadrg_log10_pga",
        "pgv": "A6_full_cadrg_log10_pgv",
    },
}


DEFAULT_COMPARISONS = (
    "ca_urc:cross_attention,"
    "ca_urc:graph,"
    "ca_urc:tail_weighted_attention,"
    "ca_urc:old_power_gate,"
    "ca_urc:ca_drg"
)


def load_module(
    path: str | Path,
    name: str,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required module not found: {path.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def parse_comparisons(
    text: str,
) -> list[tuple[str, str]]:
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
                "Candidate and reference must differ."
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


def load_and_pair_predictions(
    ablation_path: Path,
    baseline_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    if not ablation_path.exists():
        raise FileNotFoundError(
            ablation_path
        )

    if not baseline_path.exists():
        raise FileNotFoundError(
            baseline_path
        )

    ablation = pd.read_csv(
        ablation_path,
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

    required_ablation = {
        *keys,
        "target_slot",
        "true_log10_pga",
        "true_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
        "A2_cross_attention_base_log10_pga",
        "A2_cross_attention_base_log10_pgv",
        "A4_under_only_log10_pga",
        "A4_under_only_log10_pgv",
        "A6_full_cadrg_log10_pga",
        "A6_full_cadrg_log10_pgv",
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

    missing_ablation = (
        required_ablation
        .difference(
            ablation.columns
        )
    )

    missing_baseline = (
        required_baseline
        .difference(
            baseline.columns
        )
    )

    if missing_ablation:
        raise ValueError(
            "Ablation prediction file missing columns: "
            f"{sorted(missing_ablation)}"
        )

    if missing_baseline:
        raise ValueError(
            "Strong-baseline prediction file missing columns: "
            f"{sorted(missing_baseline)}"
        )

    ablation[
        "event_id"
    ] = as_str_series(
        ablation[
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

    if ablation.duplicated(
        keys
    ).any():
        raise ValueError(
            "Ablation predictions are not unique on "
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

    paired = ablation.merge(
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
        "ablation_prediction_path": str(
            ablation_path.resolve()
        ),
        "baseline_prediction_path": str(
            baseline_path.resolve()
        ),
        "ablation_rows": int(
            len(
                ablation
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
                ablation
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
            "Prediction files do not contain the exact "
            f"same paired rows. Audit={audit}"
        )

    tolerance = 1e-5

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

        cross_ablation = paired[
            f"A2_cross_attention_base_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        cross_baseline = paired[
            f"cross_attention_log10_{quantity}"
        ].to_numpy(
            dtype=float
        )

        audit[
            f"max_abs_cross_attention_diff_{quantity}"
        ] = float(
            np.max(
                np.abs(
                    cross_ablation
                    - cross_baseline
                )
            )
        )

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
                f"{key}={value:.6e} > "
                f"{tolerance:.1e}"
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


def compact_primary_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paper-oriented primary table:
    only hierarchical sequence->event results,
    with MAE / U0.5 / Factor2.
    """
    keep = summary.loc[
        (
            summary[
                "bootstrap_mode"
            ]
            == "hierarchical_sequence_event"
        )
        & (
            summary[
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

    columns = [
        "candidate",
        "reference",
        "quantity",
        "population",
        "metric",
        "reference_value",
        "candidate_value",
        "point_delta_candidate_minus_reference",
        "relative_change_percent",
        "delta_percentage_points",
        "ci_lower",
        "ci_upper",
        "ci_excludes_zero",
        "two_sided_bootstrap_sign_p",
        "bootstrap_probability_candidate_better",
        "n_paired_events",
        "n_sequence_groups_used",
    ]

    existing = [
        column
        for column in columns
        if column in keep.columns
    ]

    return keep[
        existing
    ]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-module",
        default=(
            "62_sequence_aware_paired_bootstrap_cadrg.py"
        ),
        help=(
            "Existing tested sequence-aware bootstrap implementation. "
            "Its bootstrap functions are reused; no training is performed."
        ),
    )

    parser.add_argument(
        "--ablation-predictions",
        default=(
            "runs/"
            "cadrg_gate_ablation_A3_A5/"
            "locked_test_ablation_predictions.csv"
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
            "caurc_sequence_aware_bootstrap"
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

    bootstrap_module = load_module(
        args.bootstrap_module,
        "caurc_sequence_bootstrap_core",
    )

    # Reuse the tested statistical implementation from script 62,
    # but replace its method-column map with CA-URC/A4.
    bootstrap_module.METHOD_COLUMNS = {
        method: {
            "pga": METHOD_COLUMNS[
                method
            ][
                "pga"
            ],
            "pgv": METHOD_COLUMNS[
                method
            ][
                "pgv"
            ],
            "source": "paired",
        }
        for method in METHOD_COLUMNS
    }

    paired, pairing_audit = (
        load_and_pair_predictions(
            Path(
                args.ablation_predictions
            ),
            Path(
                args.baseline_predictions
            ),
        )
    )

    paired, event_group = (
        bootstrap_module
        .attach_sequence_groups(
            paired,
            Path(
                args.manifest
            ),
            args.split_column,
            args.test_label,
            args.sequence_column,
        )
    )

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
                    f"Missing prediction column for "
                    f"{method}: {column}"
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

    pairing_audit.update(
        {
            "final_candidate_name": (
                "CA-URC"
            ),
            "final_candidate_internal_column": (
                "A4_under_only"
            ),
            "grouped_validation_selected_epoch": (
                3
            ),
            "grouped_validation_selected_gamma": (
                5.0
            ),
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

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_summary.to_csv(
        out_dir
        / "test_sequence_group_summary.csv",
        index=False,
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
        "=== CA-URC sequence-aware paired bootstrap ==="
    )

    print(
        f"Paired target rows        : "
        f"{len(paired):,}"
    )

    print(
        f"Events                    : "
        f"{event_group['event_id'].nunique():,}"
    )

    print(
        f"Sequence groups           : "
        f"{event_group['sequence_group'].nunique():,}"
    )

    print(
        f"Largest sequence          : "
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

    print(
        "Final candidate           : "
        "CA-URC = A4_under_only "
        "(epoch=3, gamma=5)"
    )

    event_metrics = (
        bootstrap_module
        .build_event_level_metrics(
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
    ) = (
        bootstrap_module
        .build_delta_matrix(
            event_metrics,
            event_group,
            comparisons,
        )
    )

    (
        group_names,
        group_indices,
    ) = (
        bootstrap_module
        .group_index_arrays(
            event_group
        )
    )

    if len(
        group_names
    ) < 2:
        raise RuntimeError(
            "Sequence-aware bootstrap requires "
            "at least two sequence groups."
        )

    print(
        "\nRunning sequence-cluster bootstrap..."
    )

    cluster_distribution = (
        bootstrap_module
        .sequence_cluster_bootstrap(
            delta_matrix,
            group_indices,
            args.bootstrap_repetitions,
            args.seed,
        )
    )

    cluster_summary = (
        bootstrap_module
        .bootstrap_summary(
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
        "Running hierarchical "
        "sequence->event bootstrap..."
    )

    hierarchical_distribution = (
        bootstrap_module
        .hierarchical_sequence_event_bootstrap(
            delta_matrix,
            group_indices,
            args.bootstrap_repetitions,
            args.seed
            + 1,
        )
    )

    hierarchical_summary = (
        bootstrap_module
        .bootstrap_summary(
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

    compact = compact_primary_table(
        combined
    )

    compact.to_csv(
        out_dir
        / "paper_primary_bootstrap_table.csv",
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
                "final_model_name": (
                    "CA-URC"
                ),
                "final_model_formula": (
                    "y_CA + p_under^gamma * Delta"
                ),
                "grouped_validation_lock": {
                    "variant": (
                        "A4_under_only"
                    ),
                    "epoch": 3,
                    "gamma": 5.0,
                },
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
                    "CA-URC - reference"
                ),
                "repeats_treated_as_independent_events": (
                    False
                ),
                "model_training_performed": (
                    False
                ),
                "model_selection_performed": (
                    False
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Main display: CA-URC vs Cross-Attention Base.
    # -------------------------------------------------------------

    primary = hierarchical_summary.loc[
        (
            hierarchical_summary[
                "candidate"
            ]
            == "ca_urc"
        )
        & (
            hierarchical_summary[
                "reference"
            ]
            == "cross_attention"
        )
    ][
        [
            "quantity",
            "population",
            "metric",
            "reference_value",
            "candidate_value",
            "point_delta_candidate_minus_reference",
            "relative_change_percent",
            "delta_percentage_points",
            "ci_lower",
            "ci_upper",
            "ci_excludes_zero",
            "two_sided_bootstrap_sign_p",
            "bootstrap_probability_candidate_better",
            "n_paired_events",
            "n_sequence_groups_used",
        ]
    ].copy()

    print(
        "\n=== PRIMARY: hierarchical "
        "sequence->event bootstrap ==="
    )

    print(
        "Delta = CA-URC - Cross-Attention Base"
    )

    print(
        primary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Tail-only comparison against every reference.
    # -------------------------------------------------------------

    tail_display = hierarchical_summary.loc[
        (
            hierarchical_summary[
                "candidate"
            ]
            == "ca_urc"
        )
        & (
            hierarchical_summary[
                "population"
            ]
            == "high_motion_tail"
        )
        & (
            hierarchical_summary[
                "metric"
            ].isin(
                [
                    "mae",
                    "under05",
                    "factor2",
                ]
            )
        )
    ][
        [
            "reference",
            "quantity",
            "metric",
            "reference_value",
            "candidate_value",
            "point_delta_candidate_minus_reference",
            "relative_change_percent",
            "delta_percentage_points",
            "ci_lower",
            "ci_upper",
            "ci_excludes_zero",
            "two_sided_bootstrap_sign_p",
            "bootstrap_probability_candidate_better",
            "n_paired_events",
        ]
    ].copy()

    print(
        "\n=== CA-URC high-motion-tail "
        "comparisons vs all references ==="
    )

    print(
        tail_display.to_string(
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
        / "paper_primary_bootstrap_table.csv",
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
