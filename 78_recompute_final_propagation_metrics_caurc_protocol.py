#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
78_recompute_final_propagation_metrics_caurc_protocol.py

Recompute the FINAL locked-test metrics for:
    1) nearest_observed
    2) idw_observed
    3) plum_like

using exactly the same evaluation principle as the final CA-URC grouped test:

    - exact same 44,800 locked target rows
    - same event / repeat / target pairing
    - same CA-URC high-motion-tail masks
    - same severe-underprediction definition:
          residual = prediction - truth
          U0.5 = P(residual <= -0.5)
    - same canonical aggregation:
          targets -> repeats -> events
    - same sequence-aware uncertainty:
          sequence -> event hierarchical paired bootstrap

IMPORTANT
---------
This script performs NO training, NO model selection, NO threshold recalculation,
and NO re-sampling of stations.

It reads:
    runs/final_strong_baselines_reuse_locked/
        final_strong_baseline_reused_locked_predictions.csv

and exact-pairs them with:
    runs/cadrg_gate_ablation_A3_A5/
        locked_test_ablation_predictions.csv

The latter supplies the locked CA-URC truth values and is_tail_pga/is_tail_pgv
flags, so the final propagation baselines are evaluated on exactly the same
tail population as CA-URC.

Primary paper outputs
---------------------
1) final_propagation_canonical_metrics.csv
2) paper_propagation_table.csv
3) caurc_vs_propagation_hierarchical_bootstrap.csv
4) caurc_vs_propagation_sequence_cluster_bootstrap.csv
5) pairing_and_tail_audit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODS = [
    "nearest_observed",
    "idw_observed",
    "plum_like",
]

DISPLAY_NAMES = {
    "nearest_observed": "Nearest observed",
    "idw_observed": "IDW observed",
    "plum_like": "PLUM-like",
    "ca_urc": "CA-URC",
}


def load_module(path: str | Path, name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_string(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
    )


def robust_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    unknown = set(text.unique()).difference(mapping)
    if unknown:
        raise ValueError(
            f"Cannot parse boolean values: {sorted(unknown)}"
        )

    return text.map(mapping).to_numpy(dtype=bool)


def load_and_pair(
    strong_path: Path,
    caurc_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not strong_path.exists():
        raise FileNotFoundError(strong_path)

    if not caurc_path.exists():
        raise FileNotFoundError(caurc_path)

    strong = pd.read_csv(
        strong_path,
        dtype={"event_id": str},
    )

    caurc = pd.read_csv(
        caurc_path,
        dtype={"event_id": str},
    )

    strong["event_id"] = as_string(
        strong["event_id"]
    )
    caurc["event_id"] = as_string(
        caurc["event_id"]
    )

    keys = [
        "event_id",
        "repeat",
        "target_station_index",
    ]

    required_strong = {
        *keys,
        "true_log10_pga",
        "true_log10_pgv",
    }

    for method in METHODS:
        required_strong.update(
            {
                f"{method}_log10_pga",
                f"{method}_log10_pgv",
            }
        )

    required_caurc = {
        *keys,
        "target_slot",
        "true_log10_pga",
        "true_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
        "A4_under_only_log10_pga",
        "A4_under_only_log10_pgv",
    }

    missing_strong = (
        required_strong
        .difference(strong.columns)
    )
    missing_caurc = (
        required_caurc
        .difference(caurc.columns)
    )

    if missing_strong:
        raise ValueError(
            "Strong-baseline prediction file missing columns: "
            f"{sorted(missing_strong)}"
        )

    if missing_caurc:
        raise ValueError(
            "CA-URC prediction file missing columns: "
            f"{sorted(missing_caurc)}"
        )

    if strong.duplicated(keys).any():
        raise RuntimeError(
            f"Strong predictions are not unique on {keys}."
        )

    if caurc.duplicated(keys).any():
        raise RuntimeError(
            f"CA-URC predictions are not unique on {keys}."
        )

    strong_keep = (
        keys
        + [
            "true_log10_pga",
            "true_log10_pgv",
        ]
        + [
            f"{method}_log10_{quantity}"
            for method in METHODS
            for quantity in ("pga", "pgv")
        ]
    )

    caurc_keep = (
        keys
        + [
            "target_slot",
            "true_log10_pga",
            "true_log10_pgv",
            "is_tail_pga",
            "is_tail_pgv",
            "A4_under_only_log10_pga",
            "A4_under_only_log10_pgv",
        ]
    )

    paired = caurc[
        caurc_keep
    ].merge(
        strong[
            strong_keep
        ],
        on=keys,
        how="inner",
        suffixes=(
            "",
            "_strong",
        ),
        validate="one_to_one",
    )

    truth_diff_pga = float(
        np.max(
            np.abs(
                paired["true_log10_pga"].to_numpy(float)
                - paired["true_log10_pga_strong"].to_numpy(float)
            )
        )
    )

    truth_diff_pgv = float(
        np.max(
            np.abs(
                paired["true_log10_pgv"].to_numpy(float)
                - paired["true_log10_pgv_strong"].to_numpy(float)
            )
        )
    )

    paired = paired.drop(
        columns=[
            "true_log10_pga_strong",
            "true_log10_pgv_strong",
        ]
    )

    audit = {
        "strong_rows": int(len(strong)),
        "caurc_rows": int(len(caurc)),
        "paired_rows": int(len(paired)),
        "strong_unique_events": int(
            strong["event_id"].nunique()
        ),
        "caurc_unique_events": int(
            caurc["event_id"].nunique()
        ),
        "paired_unique_events": int(
            paired["event_id"].nunique()
        ),
        "truth_max_abs_diff_pga": truth_diff_pga,
        "truth_max_abs_diff_pgv": truth_diff_pgv,
    }

    return paired, audit


def attach_sequence_groups(
    frame: pd.DataFrame,
    manifest_path: Path,
    split_column: str,
    test_label: str,
    sequence_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(
        manifest_path,
        dtype={"event_id": str},
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
            f"Manifest missing columns: {sorted(missing)}"
        )

    manifest["event_id"] = as_string(
        manifest["event_id"]
    )

    test = manifest.loc[
        manifest[
            split_column
        ].astype(str)
        == str(test_label),
        [
            "event_id",
            sequence_column,
        ],
    ].copy()

    test = test.rename(
        columns={
            sequence_column: "sequence_group"
        }
    )

    if test["event_id"].duplicated().any():
        raise RuntimeError(
            "Manifest test split contains duplicated event_id."
        )

    frame = frame.merge(
        test,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    if frame["sequence_group"].isna().any():
        missing_events = (
            frame.loc[
                frame["sequence_group"].isna(),
                "event_id",
            ]
            .drop_duplicates()
            .tolist()
        )
        raise RuntimeError(
            "Some paired test events have no sequence_group: "
            f"{missing_events[:20]}"
        )

    event_order = (
        frame[
            [
                "event_id",
                "sequence_group",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "event_id",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return frame, event_order


def canonical_summary(
    event_metrics: pd.DataFrame,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        event_metrics.groupby(
            [
                "method",
                "quantity",
                "population",
                "metric",
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            value=("value", "mean"),
            n_events=("event_id", "nunique"),
        )
    )

    tail_pga = robust_bool(
        paired["is_tail_pga"]
    )
    tail_pgv = robust_bool(
        paired["is_tail_pgv"]
    )

    row_counts = {
        ("pga", "overall"): int(len(paired)),
        ("pgv", "overall"): int(len(paired)),
        ("pga", "high_motion_tail"): int(
            tail_pga.sum()
        ),
        ("pgv", "high_motion_tail"): int(
            tail_pgv.sum()
        ),
    }

    summary["n_target_rows"] = [
        row_counts[
            (
                quantity,
                population,
            )
        ]
        for quantity, population
        in zip(
            summary["quantity"],
            summary["population"],
        )
    ]

    return summary


def make_paper_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for method in METHODS:
        row = {
            "method": method,
            "label": DISPLAY_NAMES[method],
        }

        for quantity in (
            "pga",
            "pgv",
        ):
            for population, prefix in (
                ("overall", "overall"),
                ("high_motion_tail", "tail"),
            ):
                for metric in (
                    "mae",
                    "bias",
                    "factor2",
                    "under05",
                ):
                    subset = summary.loc[
                        (
                            summary["method"]
                            == method
                        )
                        & (
                            summary["quantity"]
                            == quantity
                        )
                        & (
                            summary["population"]
                            == population
                        )
                        & (
                            summary["metric"]
                            == metric
                        )
                    ]

                    if subset.empty:
                        value = float("nan")
                    else:
                        value = float(
                            subset.iloc[0]["value"]
                        )

                    row[
                        f"{quantity}_{prefix}_{metric}"
                    ] = value

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-module",
        default="62_sequence_aware_paired_bootstrap_cadrg.py",
    )

    parser.add_argument(
        "--strong-predictions",
        default=(
            "runs/final_strong_baselines_reuse_locked/"
            "final_strong_baseline_reused_locked_predictions.csv"
        ),
    )

    parser.add_argument(
        "--caurc-predictions",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "locked_test_ablation_predictions.csv"
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
        "--expected-rows",
        type=int,
        default=44800,
    )

    parser.add_argument(
        "--expected-events",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--expected-pga-tail-rows",
        type=int,
        default=1363,
    )

    parser.add_argument(
        "--expected-pgv-tail-rows",
        type=int,
        default=1485,
    )

    parser.add_argument(
        "--expected-pga-tail-events",
        type=int,
        default=98,
    )

    parser.add_argument(
        "--expected-pgv-tail-events",
        type=int,
        default=87,
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "runs/final_propagation_metrics_caurc_protocol"
        ),
    )

    args = parser.parse_args()

    bootstrap_module = load_module(
        args.bootstrap_module,
        "sequence_bootstrap_module",
    )

    # Patch the audited metric code with exactly the methods needed here.
    bootstrap_module.METHOD_COLUMNS = {
        "ca_urc": {
            "pga": "A4_under_only_log10_pga",
            "pgv": "A4_under_only_log10_pgv",
            "source": "caurc_locked",
        },
        "nearest_observed": {
            "pga": "nearest_observed_log10_pga",
            "pgv": "nearest_observed_log10_pgv",
            "source": "strong_locked",
        },
        "idw_observed": {
            "pga": "idw_observed_log10_pga",
            "pgv": "idw_observed_log10_pgv",
            "source": "strong_locked",
        },
        "plum_like": {
            "pga": "plum_like_log10_pga",
            "pgv": "plum_like_log10_pgv",
            "source": "strong_locked",
        },
    }

    paired, audit = load_and_pair(
        Path(
            args.strong_predictions
        ),
        Path(
            args.caurc_predictions
        ),
    )

    paired, event_group = (
        attach_sequence_groups(
            paired,
            Path(args.manifest),
            args.split_column,
            args.test_label,
            args.sequence_column,
        )
    )

    tail_pga = robust_bool(
        paired["is_tail_pga"]
    )
    tail_pgv = robust_bool(
        paired["is_tail_pgv"]
    )

    audit.update(
        {
            "sequence_groups": int(
                event_group[
                    "sequence_group"
                ].nunique()
            ),
            "pga_tail_rows": int(
                tail_pga.sum()
            ),
            "pgv_tail_rows": int(
                tail_pgv.sum()
            ),
            "pga_tail_events": int(
                paired.loc[
                    tail_pga,
                    "event_id",
                ].nunique()
            ),
            "pgv_tail_events": int(
                paired.loc[
                    tail_pgv,
                    "event_id",
                ].nunique()
            ),
            "aggregation": (
                "targets -> repeats -> events"
            ),
            "under05_definition": (
                "prediction - truth <= -0.5"
            ),
            "tail_source": (
                "locked CA-URC is_tail_pga/is_tail_pgv flags"
            ),
            "training_performed": False,
            "model_selection_performed": False,
            "threshold_recalculation_performed": False,
        }
    )

    # Hard guards: do not silently mix another sampling realization.
    guards = {
        "paired_rows": (
            audit["paired_rows"],
            args.expected_rows,
        ),
        "paired_unique_events": (
            audit["paired_unique_events"],
            args.expected_events,
        ),
        "pga_tail_rows": (
            audit["pga_tail_rows"],
            args.expected_pga_tail_rows,
        ),
        "pgv_tail_rows": (
            audit["pgv_tail_rows"],
            args.expected_pgv_tail_rows,
        ),
        "pga_tail_events": (
            audit["pga_tail_events"],
            args.expected_pga_tail_events,
        ),
        "pgv_tail_events": (
            audit["pgv_tail_events"],
            args.expected_pgv_tail_events,
        ),
    }

    failures = []

    for name, (
        observed,
        expected,
    ) in guards.items():
        if int(observed) != int(expected):
            failures.append(
                f"{name}: observed={observed}, expected={expected}"
            )

    if (
        audit["truth_max_abs_diff_pga"]
        > 1e-5
        or audit["truth_max_abs_diff_pgv"]
        > 1e-5
    ):
        failures.append(
            "Truth mismatch exceeds 1e-5."
        )

    if failures:
        raise RuntimeError(
            "Locked-protocol audit FAILED:\n  "
            + "\n  ".join(failures)
        )

    methods = [
        "nearest_observed",
        "idw_observed",
        "plum_like",
        "ca_urc",
    ]

    event_metrics = (
        bootstrap_module
        .build_event_level_metrics(
            paired,
            methods,
        )
    )

    summary = canonical_summary(
        event_metrics,
        paired,
    )

    paper_table = make_paper_table(
        summary
    )

    comparisons = [
        (
            "ca_urc",
            "nearest_observed",
        ),
        (
            "ca_urc",
            "idw_observed",
        ),
        (
            "ca_urc",
            "plum_like",
        ),
    ]

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

    if len(group_indices) < 2:
        raise RuntimeError(
            "Sequence-aware bootstrap requires at least two groups."
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

    hierarchical_distribution = (
        bootstrap_module
        .hierarchical_sequence_event_bootstrap(
            delta_matrix,
            group_indices,
            args.bootstrap_repetitions,
            args.seed + 1,
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

    hierarchical_summary = (
        bootstrap_module
        .bootstrap_summary(
            hierarchical_distribution,
            metadata,
            "hierarchical_sequence_event",
            args.confidence_level,
        )
    )

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired[
        [
            "event_id",
            "repeat",
            "target_station_index",
            "target_slot",
            "sequence_group",
            "true_log10_pga",
            "true_log10_pgv",
            "is_tail_pga",
            "is_tail_pgv",
            "nearest_observed_log10_pga",
            "nearest_observed_log10_pgv",
            "idw_observed_log10_pga",
            "idw_observed_log10_pgv",
            "plum_like_log10_pga",
            "plum_like_log10_pgv",
            "A4_under_only_log10_pga",
            "A4_under_only_log10_pgv",
        ]
    ].to_csv(
        out_dir
        / "locked_paired_propagation_predictions.csv",
        index=False,
    )

    event_metrics.to_csv(
        out_dir
        / "event_level_canonical_metrics.csv",
        index=False,
    )

    summary.to_csv(
        out_dir
        / "final_propagation_canonical_metrics.csv",
        index=False,
    )

    paper_table.to_csv(
        out_dir
        / "paper_propagation_table.csv",
        index=False,
    )

    hierarchical_summary.to_csv(
        out_dir
        / "caurc_vs_propagation_hierarchical_bootstrap.csv",
        index=False,
    )

    cluster_summary.to_csv(
        out_dir
        / "caurc_vs_propagation_sequence_cluster_bootstrap.csv",
        index=False,
    )

    (
        out_dir
        / "pairing_and_tail_audit.json"
    ).write_text(
        json.dumps(
            {
                **audit,
                "sequence_group_names": (
                    group_names
                ),
                "bootstrap_repetitions": int(
                    args.bootstrap_repetitions
                ),
                "confidence_level": float(
                    args.confidence_level
                ),
                "bootstrap_seed": int(
                    args.seed
                ),
                "primary_inference": (
                    "hierarchical_sequence_event"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== FINAL propagation-baseline metrics "
        "under the CA-URC locked protocol ==="
    )

    print(
        f"Paired target rows        : {len(paired):,}"
    )
    print(
        f"Events                    : "
        f"{paired['event_id'].nunique()}"
    )
    print(
        f"Sequence groups           : "
        f"{event_group['sequence_group'].nunique()}"
    )
    print(
        f"PGA tail rows/events      : "
        f"{tail_pga.sum():,}/"
        f"{paired.loc[tail_pga, 'event_id'].nunique()}"
    )
    print(
        f"PGV tail rows/events      : "
        f"{tail_pgv.sum():,}/"
        f"{paired.loc[tail_pgv, 'event_id'].nunique()}"
    )
    print(
        "Truth pairing max diff    : "
        f"PGA={audit['truth_max_abs_diff_pga']:.3e}, "
        f"PGV={audit['truth_max_abs_diff_pgv']:.3e}"
    )

    show = summary.loc[
        summary["method"].isin(
            METHODS
        )
        & summary["metric"].isin(
            [
                "mae",
                "bias",
                "factor2",
                "under05",
            ]
        )
    ][
        [
            "method",
            "quantity",
            "population",
            "metric",
            "value",
            "n_events",
            "n_target_rows",
        ]
    ]

    print(
        "\n=== Canonical metrics ==="
    )
    print(
        show.to_string(
            index=False
        )
    )

    primary = hierarchical_summary.loc[
        hierarchical_summary[
            "metric"
        ].isin(
            [
                "mae",
                "factor2",
                "under05",
            ]
        )
    ]

    print(
        "\n=== PRIMARY: hierarchical "
        "sequence->event bootstrap ==="
    )
    print(
        primary.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )
    for path in [
        out_dir
        / "locked_paired_propagation_predictions.csv",
        out_dir
        / "event_level_canonical_metrics.csv",
        out_dir
        / "final_propagation_canonical_metrics.csv",
        out_dir
        / "paper_propagation_table.csv",
        out_dir
        / "caurc_vs_propagation_hierarchical_bootstrap.csv",
        out_dir
        / "caurc_vs_propagation_sequence_cluster_bootstrap.csv",
        out_dir
        / "pairing_and_tail_audit.json",
    ]:
        print(
            f"  {path.resolve()}"
        )


if __name__ == "__main__":
    main()
