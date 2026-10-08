#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig3_NC_caurc_primary_results.py

Nature Communications-style main-text figure for Results Section 2.2:
"Underprediction-risk correction selectively reduces high-motion errors
while preserving catalog-wide accuracy."

Primary input
-------------
runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv

Expected columns
----------------
event_id, repeat, target_station_index,
true_log10_pga, true_log10_pgv,
is_tail_pga, is_tail_pgv,
A2_cross_attention_base_log10_pga,
A2_cross_attention_base_log10_pgv,
A4_under_only_log10_pga,
A4_under_only_log10_pgv

Optional manifest
-----------------
data/model_manifests/scenario_t0_5s_k5_linux.csv

The manifest maps event_id to a seismic sequence group for hierarchical
sequence-to-event bootstrap confidence intervals. If no sequence mapping is
available, the script falls back to paired event-level bootstrap and records
that choice in the outputs.

Outputs
-------
Fig3_caurc_selective_tail_correction.png
Fig3_caurc_selective_tail_correction.pdf
Fig3_caurc_selective_tail_correction.svg
TableS2_caurc_primary_results.csv
TableS2_caurc_primary_results.tex
TableS2_caurc_primary_results_long.csv
Fig3_paired_effects.csv
Fig3_caption.txt
run_configuration.json

Metric aggregation
------------------
Targets -> station-resampling repeat -> earthquake event.
Primary uncertainty: paired hierarchical sequence-to-event bootstrap.

The final manuscript run should retain the locked-value audit. Use
--skip-locked-audit only for exploratory or synthetic-data tests.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
SEVERE_UNDER_THRESHOLD = -0.5
SEVERE_OVER_THRESHOLD = 0.5

MODEL_LABELS = {
    "base": "Cross-Attention\nBase",
    "final": "CA-URC",
}

QUANTITY_LABELS = {
    "pga": "PGA",
    "pgv": "PGV",
}

COLORS = {
    "pga": "#0077BB",
    "pgv": "#EE7733",
    "base": "#8B9299",
    "reference": "#555B61",
    "grid": "#E5E8EB",
    "frame": "#2B2F33",
    "text": "#202124",
}

# Final grouped-test manuscript values. These guards prevent accidentally
# drawing Fig. 3 from an older checkpoint or a differently sampled test file.
LOCKED_EXPECTED = {
    # Cross-Attention Base
    ("base", "pga", "overall", "mae"): 0.305941,
    ("base", "pga", "overall", "bias"): -0.006911,
    ("base", "pga", "overall", "factor2"): 0.576607,
    ("base", "pga", "overall", "under05"): 0.105112,
    ("base", "pga", "high_motion_tail", "mae"): 0.639199,
    ("base", "pga", "high_motion_tail", "bias"): -0.636226,
    ("base", "pga", "high_motion_tail", "factor2"): 0.145760,
    ("base", "pga", "high_motion_tail", "under05"): 0.639990,
    ("base", "pgv", "overall", "mae"): 0.265767,
    ("base", "pgv", "overall", "bias"): -0.028981,
    ("base", "pgv", "overall", "factor2"): 0.672054,
    ("base", "pgv", "overall", "under05"): 0.092388,
    ("base", "pgv", "high_motion_tail", "mae"): 0.746959,
    ("base", "pgv", "high_motion_tail", "bias"): -0.721670,
    ("base", "pgv", "high_motion_tail", "factor2"): 0.106318,
    ("base", "pgv", "high_motion_tail", "under05"): 0.714567,
    # CA-URC (A4 underprediction-risk only)
    ("final", "pga", "overall", "mae"): 0.305531,
    ("final", "pga", "overall", "bias"): 0.031684,
    ("final", "pga", "overall", "factor2"): 0.575201,
    ("final", "pga", "overall", "under05"): 0.088460,
    ("final", "pga", "high_motion_tail", "mae"): 0.566925,
    ("final", "pga", "high_motion_tail", "bias"): -0.559836,
    ("final", "pga", "high_motion_tail", "factor2"): 0.205067,
    ("final", "pga", "high_motion_tail", "under05"): 0.538049,
    ("final", "pgv", "overall", "mae"): 0.263486,
    ("final", "pgv", "overall", "bias"): 0.005872,
    ("final", "pgv", "overall", "factor2"): 0.669687,
    ("final", "pgv", "overall", "under05"): 0.077857,
    ("final", "pgv", "high_motion_tail", "mae"): 0.648410,
    ("final", "pgv", "high_motion_tail", "bias"): -0.613134,
    ("final", "pgv", "high_motion_tail", "factor2"): 0.205740,
    ("final", "pgv", "high_motion_tail", "under05"): 0.603540,
}

LOCKED_COUNTS = {
    ("pga", "overall"): (224, 22, 44_800),
    ("pga", "high_motion_tail"): (98, 21, 1_363),
    ("pgv", "overall"): (224, 22, 44_800),
    ("pgv", "high_motion_tail"): (87, 18, 1_485),
}

# Expected paired deltas are audited at point-estimate level. Bootstrap CIs
# are intentionally not hard-coded because they depend slightly on RNG and
# implementation details while the scientific conclusion is unchanged.
LOCKED_DELTAS = {
    ("pga", "overall", "mae"): -0.000410,
    ("pga", "overall", "under05"): -0.016652,
    ("pga", "high_motion_tail", "mae"): -0.072274,
    ("pga", "high_motion_tail", "factor2"): 0.059307,
    ("pga", "high_motion_tail", "under05"): -0.101942,
    ("pgv", "overall", "mae"): -0.002281,
    ("pgv", "overall", "under05"): -0.014531,
    ("pgv", "high_motion_tail", "mae"): -0.098549,
    ("pgv", "high_motion_tail", "factor2"): 0.099422,
    ("pgv", "high_motion_tail", "under05"): -0.111027,
}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def as_bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    unknown = set(normalized.unique()).difference(truthy | falsy)
    if unknown:
        raise ValueError(
            f"Cannot parse Boolean values in {series.name!r}: "
            f"{sorted(unknown)[:20]}"
        )
    return normalized.isin(truthy).to_numpy(dtype=bool)


def load_predictions(
    path: Path,
    base_prefix: str,
    final_prefix: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path.resolve())

    frame = pd.read_csv(path, dtype={"event_id": str})
    required = [
        "event_id",
        "repeat",
        "target_station_index",
        "true_log10_pga",
        "true_log10_pgv",
        "is_tail_pga",
        "is_tail_pgv",
        f"{base_prefix}_log10_pga",
        f"{base_prefix}_log10_pgv",
        f"{final_prefix}_log10_pga",
        f"{final_prefix}_log10_pgv",
    ]
    require_columns(frame, required, "Prediction table")

    frame["event_id"] = frame["event_id"].astype(str).str.strip()
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["target_station_index"] = pd.to_numeric(
        frame["target_station_index"], errors="raise"
    ).astype(int)

    keys = ["event_id", "repeat", "target_station_index"]
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, keys].head(10)
        raise RuntimeError(
            f"Prediction rows are not unique on {keys}. Examples:\n"
            f"{example.to_string(index=False)}"
        )

    for quantity in ("pga", "pgv"):
        frame[f"is_tail_{quantity}"] = as_bool_array(
            frame[f"is_tail_{quantity}"]
        )
        numeric_columns = [
            f"true_log10_{quantity}",
            f"{base_prefix}_log10_{quantity}",
            f"{final_prefix}_log10_{quantity}",
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            values = frame[column].to_numpy(float)
            if not np.isfinite(values).all():
                n_bad = int((~np.isfinite(values)).sum())
                raise ValueError(f"{column} contains {n_bad} non-finite values.")

    return frame


def detect_sequence_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested in frame.columns:
            return requested
        return None

    for candidate in (
        "sequence_group",
        "sequence_group_id",
        "sequence_id",
        "sequence",
        "cluster_group",
    ):
        if candidate in frame.columns:
            return candidate
    return None


def attach_sequence_groups(
    predictions: pd.DataFrame,
    manifest_path: Path | None,
    sequence_column: str | None,
    split_column: str | None,
    test_label: str,
) -> tuple[pd.DataFrame, str]:
    frame = predictions.copy()

    local_sequence = detect_sequence_column(frame, sequence_column)
    if local_sequence is not None:
        frame["_sequence_group"] = frame[local_sequence].astype(str)
        return frame, "hierarchical_sequence_event"

    if manifest_path is None or not manifest_path.exists():
        warnings.warn(
            "Sequence mapping is unavailable; paired event bootstrap will be "
            "used. Supply --manifest for the final manuscript figure."
        )
        frame["_sequence_group"] = frame["event_id"]
        return frame, "paired_event"

    manifest = pd.read_csv(manifest_path, dtype={"event_id": str})
    require_columns(manifest, ["event_id"], "Manifest")
    manifest["event_id"] = manifest["event_id"].astype(str).str.strip()

    detected = detect_sequence_column(manifest, sequence_column)
    if detected is None:
        warnings.warn(
            "No sequence column was found in the manifest; paired event "
            "bootstrap will be used."
        )
        frame["_sequence_group"] = frame["event_id"]
        return frame, "paired_event"

    if split_column and split_column in manifest.columns:
        selected = manifest.loc[
            manifest[split_column].astype(str).str.strip() == str(test_label)
        ].copy()
        if not selected.empty:
            manifest = selected

    mapping = manifest[["event_id", detected]].dropna().copy()
    mapping[detected] = mapping[detected].astype(str)

    inconsistent = mapping.groupby("event_id")[detected].nunique()
    inconsistent = inconsistent.loc[inconsistent > 1]
    if not inconsistent.empty:
        raise RuntimeError(
            "Some events map to more than one sequence group: "
            f"{list(inconsistent.index[:10])}"
        )

    mapping = mapping.drop_duplicates("event_id")
    frame = frame.merge(mapping, on="event_id", how="left", validate="many_to_one")

    missing = frame[detected].isna()
    if missing.any():
        events = frame.loc[missing, "event_id"].drop_duplicates().tolist()
        raise RuntimeError(
            f"Sequence mapping is missing for {len(events)} prediction events. "
            f"Examples: {events[:20]}"
        )

    frame["_sequence_group"] = frame[detected].astype(str)
    return frame, "hierarchical_sequence_event"


def element_metric(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> np.ndarray:
    residual = prediction - truth
    absolute = np.abs(residual)

    if metric == "mae":
        return absolute
    if metric == "bias":
        return residual
    if metric == "factor2":
        return (absolute <= LOG10_FACTOR_2).astype(float)
    if metric == "under05":
        return (residual <= SEVERE_UNDER_THRESHOLD).astype(float)
    if metric == "over05":
        return (residual >= SEVERE_OVER_THRESHOLD).astype(float)
    raise ValueError(f"Unknown metric: {metric}")


def event_pair_table(
    frame: pd.DataFrame,
    truth_column: str,
    base_column: str,
    final_column: str,
    metric: str,
) -> pd.DataFrame:
    """Aggregate target rows to repeats and then to paired earthquake values."""
    base_values = element_metric(
        frame[truth_column].to_numpy(float),
        frame[base_column].to_numpy(float),
        metric,
    )
    final_values = element_metric(
        frame[truth_column].to_numpy(float),
        frame[final_column].to_numpy(float),
        metric,
    )

    work = frame[["event_id", "repeat", "_sequence_group"]].copy()
    work["base"] = base_values
    work["final"] = final_values

    repeat_level = (
        work.groupby(
            ["event_id", "repeat", "_sequence_group"],
            sort=False,
            observed=True,
        )[["base", "final"]]
        .mean()
        .reset_index()
    )

    event_level = (
        repeat_level.groupby(
            ["event_id", "_sequence_group"],
            sort=False,
            observed=True,
        )[["base", "final"]]
        .mean()
        .reset_index()
    )
    event_level["delta"] = event_level["final"] - event_level["base"]
    return event_level


def paired_bootstrap(
    event_pairs: pd.DataFrame,
    mode: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    if event_pairs.empty:
        return {
            "base_ci_lower": np.nan,
            "base_ci_upper": np.nan,
            "final_ci_lower": np.nan,
            "final_ci_upper": np.nan,
            "delta_ci_lower": np.nan,
            "delta_ci_upper": np.nan,
            "probability_final_better": np.nan,
        }

    if repetitions <= 0:
        base = float(event_pairs["base"].mean())
        final = float(event_pairs["final"].mean())
        delta = final - base
        return {
            "base_ci_lower": base,
            "base_ci_upper": base,
            "final_ci_lower": final,
            "final_ci_upper": final,
            "delta_ci_lower": delta,
            "delta_ci_upper": delta,
            "probability_final_better": float(delta < 0),
        }

    rng = np.random.default_rng(seed)
    base_dist = np.empty(repetitions, dtype=float)
    final_dist = np.empty(repetitions, dtype=float)
    delta_dist = np.empty(repetitions, dtype=float)

    if mode == "hierarchical_sequence_event":
        grouped = {
            str(sequence): group[["base", "final"]].to_numpy(float)
            for sequence, group in event_pairs.groupby(
                "_sequence_group", sort=False, observed=True
            )
        }
        sequence_ids = np.array(list(grouped), dtype=object)
        n_sequences = len(sequence_ids)

        for index in range(repetitions):
            sampled_sequences = rng.choice(
                sequence_ids,
                size=n_sequences,
                replace=True,
            )
            blocks: list[np.ndarray] = []
            for sequence in sampled_sequences:
                values = grouped[str(sequence)]
                sampled_events = rng.integers(0, len(values), size=len(values))
                blocks.append(values[sampled_events])
            sample = np.concatenate(blocks, axis=0)
            base_dist[index] = float(sample[:, 0].mean())
            final_dist[index] = float(sample[:, 1].mean())
            delta_dist[index] = final_dist[index] - base_dist[index]
    else:
        values = event_pairs[["base", "final"]].to_numpy(float)
        n_events = len(values)
        for index in range(repetitions):
            sampled_events = rng.integers(0, n_events, size=n_events)
            sample = values[sampled_events]
            base_dist[index] = float(sample[:, 0].mean())
            final_dist[index] = float(sample[:, 1].mean())
            delta_dist[index] = final_dist[index] - base_dist[index]

    alpha = 1.0 - confidence_level
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0

    return {
        "base_ci_lower": float(np.quantile(base_dist, q_low)),
        "base_ci_upper": float(np.quantile(base_dist, q_high)),
        "final_ci_lower": float(np.quantile(final_dist, q_low)),
        "final_ci_upper": float(np.quantile(final_dist, q_high)),
        "delta_ci_lower": float(np.quantile(delta_dist, q_low)),
        "delta_ci_upper": float(np.quantile(delta_dist, q_high)),
        "probability_final_better": float(np.mean(delta_dist < 0.0)),
    }


def build_summary(
    frame: pd.DataFrame,
    base_prefix: str,
    final_prefix: str,
    bootstrap_mode: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    event_tables: dict[tuple[str, str, str], pd.DataFrame] = {}

    for q_index, quantity in enumerate(("pga", "pgv")):
        truth_column = f"true_log10_{quantity}"
        base_column = f"{base_prefix}_log10_{quantity}"
        final_column = f"{final_prefix}_log10_{quantity}"
        tail = frame[f"is_tail_{quantity}"].to_numpy(dtype=bool)

        populations = {
            "overall": np.ones(len(frame), dtype=bool),
            "non_tail": ~tail,
            "high_motion_tail": tail,
        }

        for p_index, (population, mask) in enumerate(populations.items()):
            subset = frame.loc[mask].copy()
            n_events = int(subset["event_id"].nunique())
            n_sequences = int(subset["_sequence_group"].nunique())

            for m_index, metric in enumerate(
                ("mae", "bias", "factor2", "under05", "over05")
            ):
                pairs = event_pair_table(
                    subset,
                    truth_column,
                    base_column,
                    final_column,
                    metric,
                )
                event_tables[(quantity, population, metric)] = pairs

                base_value = float(pairs["base"].mean())
                final_value = float(pairs["final"].mean())
                delta = final_value - base_value

                boot = paired_bootstrap(
                    pairs,
                    mode=bootstrap_mode,
                    repetitions=repetitions,
                    confidence_level=confidence_level,
                    seed=(
                        seed
                        + 100_000 * q_index
                        + 10_000 * p_index
                        + 1_000 * m_index
                    ),
                )

                relative_change = (
                    100.0 * delta / abs(base_value)
                    if metric in {"mae", "bias"} and abs(base_value) > 1e-12
                    else np.nan
                )
                delta_percentage_points = (
                    100.0 * delta
                    if metric in {"factor2", "under05", "over05"}
                    else np.nan
                )

                rows.append(
                    {
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "base_value": base_value,
                        "base_ci_lower": boot["base_ci_lower"],
                        "base_ci_upper": boot["base_ci_upper"],
                        "ca_urc_value": final_value,
                        "ca_urc_ci_lower": boot["final_ci_lower"],
                        "ca_urc_ci_upper": boot["final_ci_upper"],
                        "delta_ca_urc_minus_base": delta,
                        "delta_ci_lower": boot["delta_ci_lower"],
                        "delta_ci_upper": boot["delta_ci_upper"],
                        "relative_change_percent": relative_change,
                        "delta_percentage_points": delta_percentage_points,
                        "bootstrap_probability_ca_urc_lower": boot[
                            "probability_final_better"
                        ],
                        "confidence_level": confidence_level,
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_mode": bootstrap_mode,
                        "n_events": n_events,
                        "n_sequence_groups": n_sequences,
                        "n_target_rows": int(len(subset)),
                    }
                )

    return pd.DataFrame(rows), event_tables


def get_row(
    summary: pd.DataFrame,
    quantity: str,
    population: str,
    metric: str,
) -> pd.Series:
    row = summary.loc[
        (summary["quantity"] == quantity)
        & (summary["population"] == population)
        & (summary["metric"] == metric)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one row for {quantity}/{population}/{metric}; "
            f"found {len(row)}."
        )
    return row.iloc[0]


def audit_locked_values(summary: pd.DataFrame, tolerance: float) -> None:
    problems: list[str] = []

    for (model, quantity, population, metric), expected in LOCKED_EXPECTED.items():
        row = get_row(summary, quantity, population, metric)
        column = "base_value" if model == "base" else "ca_urc_value"
        observed = float(row[column])
        if abs(observed - expected) > tolerance:
            problems.append(
                f"{model}/{quantity}/{population}/{metric}: "
                f"observed={observed:.9f}, expected={expected:.9f}"
            )

    for (quantity, population, metric), expected in LOCKED_DELTAS.items():
        observed = float(
            get_row(summary, quantity, population, metric)[
                "delta_ca_urc_minus_base"
            ]
        )
        if abs(observed - expected) > tolerance:
            problems.append(
                f"delta/{quantity}/{population}/{metric}: "
                f"observed={observed:.9f}, expected={expected:.9f}"
            )

    for (quantity, population), (
        expected_events,
        expected_sequences,
        expected_rows,
    ) in LOCKED_COUNTS.items():
        row = get_row(summary, quantity, population, "mae")
        observed = (
            int(row["n_events"]),
            int(row["n_sequence_groups"]),
            int(row["n_target_rows"]),
        )
        expected = (expected_events, expected_sequences, expected_rows)
        if observed != expected:
            problems.append(
                f"counts/{quantity}/{population}: "
                f"observed={observed}, expected={expected}"
            )

    if problems:
        raise RuntimeError(
            "Locked manuscript audit failed. The input may be an older or "
            "differently sampled prediction file:\n  " + "\n  ".join(problems)
        )


def weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    x = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return x, cumulative


def event_balanced_row_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every earthquake the same total weight in descriptive ECDFs."""
    counts = frame.groupby("event_id", sort=False)["event_id"].transform("size")
    n_events = frame["event_id"].nunique()
    return 1.0 / (counts.to_numpy(float) * max(n_events, 1))


def configure_style() -> None:
    """Compact Nature Communications-style typography and fixed publication palette."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.titlesize": 7.6,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.60,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["frame"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Separate bold panel letter from the title for strict multi-panel alignment."""
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        pad=4.0,
        fontsize=7.6,
        color=COLORS["text"],
    )
    ax.text(
        -0.13,
        1.025,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    """Full rectangular frame, outward ticks, and restrained optional grid."""
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.60)
        ax.spines[side].set_color(COLORS["frame"])

    ax.tick_params(
        direction="out",
        top=False,
        right=False,
        width=0.55,
        length=2.6,
        pad=2.0,
    )

    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(
            axis=grid_axis,
            color=COLORS["grid"],
            linewidth=0.38,
            alpha=0.65,
            zorder=0,
        )
        ax.set_axisbelow(True)


def format_effect(row: pd.Series, metric: str) -> str:
    delta = float(row["delta_ca_urc_minus_base"])
    low = float(row["delta_ci_lower"])
    high = float(row["delta_ci_upper"])

    if metric == "mae":
        rel = float(row["relative_change_percent"])
        return (
            f"Δ={delta:+.3f} [{low:+.3f}, {high:+.3f}]\n"
            f"{rel:+.1f}%"
        )

    pp = float(row["delta_percentage_points"])
    low_pp = 100.0 * low
    high_pp = 100.0 * high
    return f"Δ={pp:+.1f} pp [{low_pp:+.1f}, {high_pp:+.1f}]"



def delta_forest_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    population: str,
    metric: str,
    letter: str,
    title: str,
) -> None:
    """Forest plot of paired CA-URC minus Base effects with 95% bootstrap CIs."""
    colors = {"pga": COLORS["pga"], "pgv": COLORS["pgv"]}
    markers = {"pga": "o", "pgv": "s"}
    y_positions = {"pga": 1.0, "pgv": 0.0}

    extrema = [0.0]
    for quantity in ("pga", "pgv"):
        row = get_row(summary, quantity, population, metric)
        delta = float(row["delta_ca_urc_minus_base"])
        low = float(row["delta_ci_lower"])
        high = float(row["delta_ci_upper"])
        y = y_positions[quantity]

        ax.errorbar(
            delta,
            y,
            xerr=np.array([[delta - low], [high - delta]]),
            fmt=markers[quantity],
            markersize=4.8,
            capsize=2.4,
            elinewidth=0.95,
            capthick=0.90,
            color=colors[quantity],
            markeredgecolor="white",
            markeredgewidth=0.50,
            zorder=3,
        )

        # One concise effect-size annotation only; absolute endpoint values are omitted
        # because they are reported in Supplementary Table 2.
        ax.text(
            0.985,
            y,
            f"Δ={delta:+.4f} [{low:+.4f}, {high:+.4f}]",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.7,
            color=colors[quantity],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.4),
            zorder=4,
        )
        extrema.extend([low, high])

    maximum = max(abs(min(extrema)), abs(max(extrema)), 1e-3)
    ax.set_xlim(-1.30 * maximum, 1.30 * maximum)
    ax.axvline(
        0.0,
        linewidth=0.75,
        linestyle=(0, (3.0, 2.0)),
        color=COLORS["reference"],
        alpha=0.85,
        zorder=1,
    )
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["PGA", "PGV"])
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel(r"Paired $\Delta$MAE, CA-URC $-$ Base ($\log_{10}$ units)")
    panel_title(ax, letter, title)
    clean_axis(ax, grid_axis=None)
    ax.tick_params(axis="y", length=0)


def paired_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    population: str,
    metric: str,
    letter: str,
    title: str,
    ylabel: str,
    percent: bool = False,
) -> None:
    """Paired model comparison with horizontally offset PGA/PGV CIs and concise effect text."""
    colors = {"pga": COLORS["pga"], "pgv": COLORS["pgv"]}
    markers = {"pga": "o", "pgv": "s"}
    offsets = {"pga": -0.025, "pgv": 0.025}
    base_x = np.array([0.0, 1.0])

    all_low: list[float] = []
    all_high: list[float] = []

    for quantity in ("pga", "pgv"):
        row = get_row(summary, quantity, population, metric)
        scale = 100.0 if percent else 1.0
        values = np.array(
            [float(row["base_value"]), float(row["ca_urc_value"])],
            dtype=float,
        ) * scale
        lower = np.array(
            [float(row["base_ci_lower"]), float(row["ca_urc_ci_lower"])],
            dtype=float,
        ) * scale
        upper = np.array(
            [float(row["base_ci_upper"]), float(row["ca_urc_ci_upper"])],
            dtype=float,
        ) * scale
        yerr = np.vstack([values - lower, upper - values])
        xx = base_x + offsets[quantity]

        ax.plot(
            xx,
            values,
            color=colors[quantity],
            linewidth=1.15,
            alpha=0.90,
            zorder=2,
        )
        ax.errorbar(
            xx,
            values,
            yerr=yerr,
            fmt=markers[quantity],
            markersize=4.5,
            capsize=2.3,
            capthick=0.85,
            elinewidth=0.85,
            color=colors[quantity],
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
            label=QUANTITY_LABELS[quantity],
        )

        # Keep only the inferential effect summary; omit duplicated endpoint labels.
        effect_text = format_effect(row, metric).replace("\n", "  ")
        vertical_position = 0.965 if quantity == "pga" else 0.825
        ax.text(
            0.03,
            vertical_position,
            f"{QUANTITY_LABELS[quantity]}: {effect_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
            color=colors[quantity],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
            zorder=4,
        )

        all_low.extend(lower.tolist())
        all_high.extend(upper.tolist())

    ax.set_xticks(base_x)
    ax.set_xticklabels([MODEL_LABELS["base"], MODEL_LABELS["final"]])
    ax.set_xlim(-0.24, 1.24)
    ax.set_ylabel(ylabel)
    panel_title(ax, letter, title)

    y_min = min(all_low)
    y_max = max(all_high)
    span = max(y_max - y_min, 1e-4)
    lower_pad = 0.18 * span
    upper_pad = 0.40 * span

    if metric in {"mae", "factor2", "under05", "over05"}:
        lower_limit = max(0.0, y_min - lower_pad)
    else:
        lower_limit = y_min - lower_pad
    upper_limit = y_max + upper_pad

    if population == "overall" and metric == "mae":
        lower_limit = max(0.0, min(all_low) - 0.025)
        upper_limit = max(all_high) + 0.035

    ax.set_ylim(lower_limit, upper_limit)
    clean_axis(ax, grid_axis="y")


def residual_ecdf_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    quantity: str,
    base_prefix: str,
    final_prefix: str,
    summary: pd.DataFrame,
    letter: str,
    title: str,
) -> None:
    tail = frame.loc[frame[f"is_tail_{quantity}"]].copy()
    truth = tail[f"true_log10_{quantity}"].to_numpy(float)
    base_residual = tail[f"{base_prefix}_log10_{quantity}"].to_numpy(float) - truth
    final_residual = tail[f"{final_prefix}_log10_{quantity}"].to_numpy(float) - truth
    weights = event_balanced_row_weights(tail)

    base_x, base_y = weighted_ecdf(base_residual, weights)
    final_x, final_y = weighted_ecdf(final_residual, weights)
    quantity_color = COLORS[quantity]

    ax.step(
        base_x,
        base_y,
        where="post",
        linewidth=1.10,
        linestyle=(0, (4.0, 2.0)),
        color=COLORS["base"],
        label="Cross-Attention Base",
        zorder=2,
    )
    ax.step(
        final_x,
        final_y,
        where="post",
        linewidth=1.65,
        color=quantity_color,
        label="CA-URC",
        zorder=3,
    )

    ax.axvline(
        -0.5,
        linewidth=0.80,
        linestyle=(0, (1.5, 1.5)),
        color=COLORS["reference"],
        zorder=1,
    )
    ax.axvline(
        0.0,
        linewidth=0.75,
        linestyle=(0, (3.0, 2.0)),
        color=COLORS["reference"],
        alpha=0.80,
        zorder=1,
    )

    combined = np.concatenate([base_residual, final_residual])
    x_low = float(np.quantile(combined, 0.005))
    x_high = float(np.quantile(combined, 0.995))
    x_low = min(x_low, -0.6)
    x_high = max(x_high, 0.12)
    span = max(x_high - x_low, 1e-3)
    ax.set_xlim(x_low - 0.03 * span, x_high + 0.03 * span)
    ax.set_ylim(0.0, 1.0)

    base_u = float(
        get_row(summary, quantity, "high_motion_tail", "under05")["base_value"]
    )
    final_u = float(
        get_row(summary, quantity, "high_motion_tail", "under05")["ca_urc_value"]
    )
    ax.text(
        0.97,
        0.055,
        f"Tail $U_{{0.5}}$: {100*base_u:.1f}% → {100*final_u:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6),
        zorder=4,
    )

    ax.set_xlabel(r"Residual ($\log_{10}$ units)")
    ax.set_ylabel("Event-balanced cumulative fraction")
    panel_title(ax, letter, title)
    clean_axis(ax, grid_axis="y")


def make_figure(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    base_prefix: str,
    final_prefix: str,
    out_dir: Path,
    dpi: int,
) -> None:
    configure_style()

    width_in = 183.0 / 25.4
    height_in = 145.0 / 25.4
    fig, axes = plt.subplots(2, 3, figsize=(width_in, height_in), facecolor="white")

    delta_forest_panel(
        axes[0, 0],
        summary,
        population="overall",
        metric="mae",
        letter="a",
        title="Catalog-wide MAE is unchanged",
    )
    paired_metric_panel(
        axes[0, 1],
        summary,
        population="high_motion_tail",
        metric="mae",
        letter="b",
        title="Tail MAE decreases",
        ylabel=r"Tail MAE ($\log_{10}$ units)",
    )
    paired_metric_panel(
        axes[0, 2],
        summary,
        population="high_motion_tail",
        metric="under05",
        letter="c",
        title="Severe underprediction decreases",
        ylabel=r"Tail $U_{0.5}$ (%)",
        percent=True,
    )
    paired_metric_panel(
        axes[1, 0],
        summary,
        population="high_motion_tail",
        metric="factor2",
        letter="d",
        title="Tail factor-two accuracy increases",
        ylabel="Tail factor-of-two accuracy (%)",
        percent=True,
    )
    residual_ecdf_panel(
        axes[1, 1],
        frame,
        quantity="pga",
        base_prefix=base_prefix,
        final_prefix=final_prefix,
        summary=summary,
        letter="e",
        title="PGA tail residual distribution",
    )
    residual_ecdf_panel(
        axes[1, 2],
        frame,
        quantity="pgv",
        base_prefix=base_prefix,
        final_prefix=final_prefix,
        summary=summary,
        letter="f",
        title="PGV tail residual distribution",
    )

    # One figure-level legend for PGA and PGV only.
    legend_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markersize=4.6,
            markerfacecolor=COLORS["pga"], markeredgecolor="white",
            markeredgewidth=0.4, label="PGA",
        ),
        Line2D(
            [], [], linestyle="none", marker="s", markersize=4.6,
            markerfacecolor=COLORS["pgv"], markeredgecolor="white",
            markeredgewidth=0.4, label="PGV",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.982),
        ncol=2,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    # Residual-panel legends explain model line styles.
    for ax in (axes[1, 1], axes[1, 2]):
        ax.legend(
            frameon=False,
            loc="upper left",
            handlelength=2.5,
            handletextpad=0.45,
            labelspacing=0.25,
            borderaxespad=0.2,
        )

    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        bottom=0.080,
        top=0.920,
        wspace=0.23,
        hspace=0.24,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Fig3_caurc_selective_tail_correction"
    # Preserve the exact 183-mm physical canvas; do not use bbox_inches='tight'.
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def make_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quantity in ("pga", "pgv"):
        for population in ("overall", "high_motion_tail"):
            record: dict[str, object] = {
                "Quantity": QUANTITY_LABELS[quantity],
                "Population": (
                    "All targets" if population == "overall" else "High-motion tail"
                ),
            }
            for metric, label in (
                ("mae", "MAE"),
                ("bias", "Bias"),
                ("factor2", "Factor2"),
                ("under05", "U0.5"),
                ("over05", "O0.5"),
            ):
                row = get_row(summary, quantity, population, metric)
                record[f"Base {label}"] = float(row["base_value"])
                record[f"CA-URC {label}"] = float(row["ca_urc_value"])
            count_row = get_row(summary, quantity, population, "mae")
            record["Events"] = int(count_row["n_events"])
            record["Sequence groups"] = int(count_row["n_sequence_groups"])
            record["Target predictions"] = int(count_row["n_target_rows"])
            rows.append(record)
    return pd.DataFrame(rows)


def compact_table_to_latex(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in display.columns:
        if column.startswith("Base ") or column.startswith("CA-URC "):
            if column.endswith(("Factor2", "U0.5", "O0.5")):
                display[column] = display[column].map(lambda value: f"{100*value:.1f}\\%")
            else:
                display[column] = display[column].map(lambda value: f"{value:.3f}")
    return display.to_latex(index=False, escape=False)


def write_caption(
    summary: pd.DataFrame,
    bootstrap_mode: str,
    out_path: Path,
) -> None:
    pga_tail = get_row(summary, "pga", "high_motion_tail", "mae")
    pgv_tail = get_row(summary, "pgv", "high_motion_tail", "mae")
    overall = get_row(summary, "pga", "overall", "mae")

    mode_text = (
        "paired hierarchical sequence-to-event bootstrap resampling"
        if bootstrap_mode == "hierarchical_sequence_event"
        else "paired event-level bootstrap resampling"
    )

    caption = f"""Fig. 3 | Underprediction-risk correction selectively reduces high-motion errors while preserving catalog-wide accuracy. a, Paired change in catalog-wide event-macro mean absolute error (MAE), defined as CA-URC minus the frozen Cross-Attention Base. Points and horizontal intervals denote paired effects and 95% confidence intervals. b, High-motion-tail MAE for the two exactly paired models. c, Severe underprediction rate, U_0.5 = Pr(prediction - observation <= -0.5), in the high-motion tail. d, Tail predictions within a factor of two of the observations. Points and vertical error bars in b-d denote event-macro estimates and 95% confidence intervals; coloured lines connect the paired base and CA-URC estimates. All confidence intervals were obtained using {mode_text}. e,f, Event-balanced empirical cumulative distributions of PGA and PGV residuals within the high-motion tail. The dotted vertical line denotes the severe-underprediction threshold of -0.5 log10 units and the dashed line denotes zero residual. The locked test set comprised {int(overall['n_events'])} earthquakes, {int(overall['n_sequence_groups'])} seismic-sequence groups and {int(overall['n_target_rows']):,} target predictions. The PGA tail contained {int(pga_tail['n_target_rows']):,} predictions from {int(pga_tail['n_events'])} earthquakes, and the PGV tail contained {int(pgv_tail['n_target_rows']):,} predictions from {int(pgv_tail['n_events'])} earthquakes. High-motion labels were determined from training data only. Exact numerical values, including bias and severe overprediction rates, are reported in Supplementary Table 2."""
    out_path.write_text(caption, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "locked_test_ablation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="model_manifests/scenario_t0_5s_k5_linux.csv",
    )
    parser.add_argument("--sequence-column", default="sequence_group")
    parser.add_argument("--split-column", default="split_grouped")
    parser.add_argument("--test-label", default="test")
    parser.add_argument("--base-prefix", default="A2_cross_attention_base")
    parser.add_argument("--final-prefix", default="A4_under_only")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--audit-tolerance", type=float, default=5e-6)
    parser.add_argument("--skip-locked-audit", action="store_true")
    parser.add_argument("--out-dir", default="figures/nc_section_2_2")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    manifest_path = Path(args.manifest) if str(args.manifest).strip() else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_predictions(
        predictions_path,
        base_prefix=args.base_prefix,
        final_prefix=args.final_prefix,
    )
    frame, bootstrap_mode = attach_sequence_groups(
        frame,
        manifest_path=manifest_path,
        sequence_column=(args.sequence_column or None),
        split_column=(args.split_column or None),
        test_label=args.test_label,
    )

    summary, event_tables = build_summary(
        frame,
        base_prefix=args.base_prefix,
        final_prefix=args.final_prefix,
        bootstrap_mode=bootstrap_mode,
        repetitions=args.bootstrap_repetitions,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )

    if not args.skip_locked_audit:
        audit_locked_values(summary, tolerance=args.audit_tolerance)
        print("Locked grouped-test audit: PASSED")
    else:
        print("Locked grouped-test audit: SKIPPED")

    summary.to_csv(
        out_dir / "TableS2_caurc_primary_results_long.csv",
        index=False,
    )

    compact = make_compact_table(summary)
    compact.to_csv(
        out_dir / "TableS2_caurc_primary_results.csv",
        index=False,
    )
    (out_dir / "TableS2_caurc_primary_results.tex").write_text(
        compact_table_to_latex(compact),
        encoding="utf-8",
    )

    primary_keys = [
        ("pga", "overall", "mae"),
        ("pgv", "overall", "mae"),
        ("pga", "overall", "under05"),
        ("pgv", "overall", "under05"),
        ("pga", "high_motion_tail", "mae"),
        ("pgv", "high_motion_tail", "mae"),
        ("pga", "high_motion_tail", "under05"),
        ("pgv", "high_motion_tail", "under05"),
        ("pga", "high_motion_tail", "factor2"),
        ("pgv", "high_motion_tail", "factor2"),
    ]
    primary_rows = [
        get_row(summary, quantity, population, metric)
        for quantity, population, metric in primary_keys
    ]
    pd.DataFrame(primary_rows).to_csv(
        out_dir / "Fig3_paired_effects.csv",
        index=False,
    )

    make_figure(
        frame,
        summary,
        base_prefix=args.base_prefix,
        final_prefix=args.final_prefix,
        out_dir=out_dir,
        dpi=args.dpi,
    )
    write_caption(
        summary,
        bootstrap_mode=bootstrap_mode,
        out_path=out_dir / "Fig3_caption.txt",
    )

    configuration = {
        **vars(args),
        "predictions_resolved": str(predictions_path.resolve()),
        "manifest_resolved": (
            str(manifest_path.resolve())
            if manifest_path is not None and manifest_path.exists()
            else None
        ),
        "bootstrap_mode_used": bootstrap_mode,
        "metric_aggregation": "targets -> repeats -> events",
        "final_model": "CA-URC = A4 underprediction-risk correction",
        "final_formula": "y_final = y_CA + p_under^gamma * Delta",
        "grouped_validation_selected_epoch": 3,
        "grouped_validation_selected_gamma": 5.0,
        "test_used_for_model_selection": False,
    }
    (out_dir / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    print("\n=== Primary paired effects ===")
    print(pd.DataFrame(primary_rows).to_string(index=False))
    print("\nSaved outputs:")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
