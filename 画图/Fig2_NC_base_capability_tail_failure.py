#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
81_plot_nc_section_2_1_base_tail_failure.py

Nature Communications-style main-text figure for Results Section 2.1:
"Sparse causal observations support regional ground-motion prediction but
mask systematic high-motion underprediction."

Primary input
-------------
runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv

Expected columns
----------------
event_id, repeat, target_station_index,
true_log10_pga, true_log10_pgv,
is_tail_pga, is_tail_pgv,
A2_cross_attention_base_log10_pga,
A2_cross_attention_base_log10_pgv

Optional manifest
-----------------
data/model_manifests/scenario_t0_5s_k5_linux.csv

The manifest is used only to map event_id to a seismic sequence group for
hierarchical sequence-to-event bootstrap confidence intervals. If sequence
mapping is unavailable, the script falls back to event-level bootstrap and
records that choice in the output table and caption.

Outputs
-------
Fig2_base_capability_and_tail_failure.png   (600 dpi)
Fig2_base_capability_and_tail_failure.pdf   (vector text; rasterized points)
Fig2_base_capability_and_tail_failure.svg
TableS1_base_overall_tail_metrics.csv
TableS1_base_overall_tail_metrics.tex
Fig2_caption.txt
run_configuration.txt

Metric aggregation
------------------
Targets -> station-resampling repeat -> earthquake event.
For the primary confidence intervals, sequence groups are sampled first and
events are then sampled within each selected sequence group.

The script audits the locked manuscript values by default. Use
--skip-locked-audit only for exploratory reruns, not for the final paper.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
SEVERE_UNDER_THRESHOLD = -0.5
SEVERE_OVER_THRESHOLD = 0.5

# Locked Cross-Attention Base values under the final grouped protocol.
# These values prevent accidental use of an older prediction file.
LOCKED_EXPECTED = {
    ("pga", "overall", "mae"): 0.305941,
    ("pga", "overall", "bias"): -0.006911,
    ("pga", "overall", "factor2"): 0.576607,
    ("pga", "overall", "under05"): 0.105112,
    ("pga", "high_motion_tail", "mae"): 0.639199,
    ("pga", "high_motion_tail", "bias"): -0.636226,
    ("pga", "high_motion_tail", "factor2"): 0.145760,
    ("pga", "high_motion_tail", "under05"): 0.639990,
    ("pgv", "overall", "mae"): 0.265767,
    ("pgv", "overall", "bias"): -0.028981,
    ("pgv", "overall", "factor2"): 0.672054,
    ("pgv", "overall", "under05"): 0.092388,
    ("pgv", "high_motion_tail", "mae"): 0.746959,
    ("pgv", "high_motion_tail", "bias"): -0.721670,
    ("pgv", "high_motion_tail", "factor2"): 0.106318,
    ("pgv", "high_motion_tail", "under05"): 0.714567,
}

LOCKED_COUNTS = {
    ("pga", "overall"): (224, 44_800),
    ("pga", "high_motion_tail"): (98, 1_363),
    ("pgv", "overall"): (224, 44_800),
    ("pgv", "high_motion_tail"): (87, 1_485),
}

QUANTITY_LABELS = {
    "pga": "PGA",
    "pgv": "PGV",
}

QUANTITY_AXIS_LABELS = {
    "pga": r"$\log_{10}[\mathrm{PGA}_{\mathrm{rem},H}/(\mathrm{m}\,\mathrm{s}^{-2})]$",
    "pgv": r"$\log_{10}[\mathrm{PGV}_{\mathrm{rem},H}/(\mathrm{m}\,\mathrm{s}^{-1})]$",
}

POPULATION_LABELS = {
    "overall": "All targets",
    "high_motion_tail": "High-motion tail",
}

METRIC_LABELS = {
    "mae": "MAE",
    "bias": "Bias",
    "factor2": "Factor-of-two accuracy",
    "under05": r"$U_{0.5}$",
    "over05": r"$O_{0.5}$",
}


def as_bool_array(series: pd.Series) -> np.ndarray:
    """Parse common textual and numeric Boolean representations safely."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    unknown = set(normalized.unique()).difference(truthy | falsy)
    if unknown:
        raise ValueError(
            "Cannot parse Boolean values in column "
            f"{series.name!r}: {sorted(unknown)[:20]}"
        )
    return normalized.isin(truthy).to_numpy(dtype=bool)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_predictions(path: Path, base_prefix: str) -> pd.DataFrame:
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
    ]
    require_columns(frame, required, "Prediction table")

    frame["event_id"] = frame["event_id"].astype(str).str.strip()
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["target_station_index"] = pd.to_numeric(
        frame["target_station_index"], errors="raise"
    ).astype(int)

    unique_keys = ["event_id", "repeat", "target_station_index"]
    duplicates = frame.duplicated(unique_keys, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, unique_keys].head(10)
        raise RuntimeError(
            "Prediction rows are not unique on "
            f"{unique_keys}. Examples:\n{example.to_string(index=False)}"
        )

    for quantity in ("pga", "pgv"):
        frame[f"is_tail_{quantity}"] = as_bool_array(
            frame[f"is_tail_{quantity}"]
        )
        for column in (
            f"true_log10_{quantity}",
            f"{base_prefix}_log10_{quantity}",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(frame[column].to_numpy(float)).all():
                bad = int((~np.isfinite(frame[column].to_numpy(float))).sum())
                raise ValueError(f"{column} contains {bad} non-finite values.")

    return frame


def detect_sequence_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(
                f"Requested sequence column {requested!r} was not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return requested

    candidates = [
        "sequence_group",
        "sequence_group_id",
        "sequence_id",
        "sequence",
        "cluster_group",
    ]
    for candidate in candidates:
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
    """
    Attach a sequence-group label to each prediction row.

    Returns
    -------
    frame, bootstrap_mode
    """
    frame = predictions.copy()

    # A requested sequence column may live in the manifest rather than in the
    # prediction table. Use it directly only when it is already present here.
    if sequence_column and sequence_column in frame.columns:
        existing_sequence = sequence_column
    elif sequence_column:
        existing_sequence = None
    else:
        existing_sequence = detect_sequence_column(frame, None)

    if existing_sequence is not None:
        frame["_sequence_group"] = frame[existing_sequence].astype(str)
        return frame, "hierarchical_sequence_event"

    if manifest_path is None or not manifest_path.exists():
        warnings.warn(
            "Sequence mapping was unavailable; confidence intervals will use "
            "event-level bootstrap. Supply --manifest for the manuscript figure."
        )
        frame["_sequence_group"] = frame["event_id"]
        return frame, "event"

    manifest = pd.read_csv(manifest_path, dtype={"event_id": str})
    require_columns(manifest, ["event_id"], "Manifest")
    manifest["event_id"] = manifest["event_id"].astype(str).str.strip()

    detected = detect_sequence_column(manifest, sequence_column)
    if detected is None:
        warnings.warn(
            "No sequence-group column was found in the manifest; confidence "
            "intervals will use event-level bootstrap."
        )
        frame["_sequence_group"] = frame["event_id"]
        return frame, "event"

    if split_column and split_column in manifest.columns:
        selected = manifest.loc[
            manifest[split_column].astype(str).str.strip() == str(test_label)
        ].copy()
        # If the filter unexpectedly removes all test events, keep the full mapping
        # rather than silently producing missing groups.
        if not selected.empty:
            manifest = selected

    mapping = manifest[["event_id", detected]].dropna().copy()
    mapping[detected] = mapping[detected].astype(str)

    inconsistent = mapping.groupby("event_id")[detected].nunique()
    inconsistent = inconsistent.loc[inconsistent > 1]
    if not inconsistent.empty:
        raise RuntimeError(
            "Manifest maps some events to multiple sequence groups: "
            f"{list(inconsistent.index[:10])}"
        )

    mapping = mapping.drop_duplicates("event_id")
    frame = frame.merge(mapping, on="event_id", how="left", validate="many_to_one")

    missing = frame[detected].isna()
    if missing.any():
        missing_events = frame.loc[missing, "event_id"].drop_duplicates().tolist()
        raise RuntimeError(
            f"Sequence mapping is missing for {len(missing_events)} prediction "
            f"events. Examples: {missing_events[:20]}"
        )

    frame["_sequence_group"] = frame[detected].astype(str)
    return frame, "hierarchical_sequence_event"


def element_values(truth: np.ndarray, prediction: np.ndarray, metric: str) -> np.ndarray:
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


def event_metric_table(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    metric: str,
) -> pd.DataFrame:
    """Aggregate target rows to repeat means and then to earthquake-event means."""
    values = element_values(
        frame[truth_column].to_numpy(float),
        frame[prediction_column].to_numpy(float),
        metric,
    )

    work = frame[["event_id", "repeat", "_sequence_group"]].copy()
    work["value"] = values

    repeat_level = (
        work.groupby(
            ["event_id", "repeat", "_sequence_group"],
            sort=False,
            observed=True,
        )["value"]
        .mean()
        .reset_index()
    )

    event_level = (
        repeat_level.groupby(
            ["event_id", "_sequence_group"],
            sort=False,
            observed=True,
        )["value"]
        .mean()
        .reset_index()
    )
    return event_level


def bootstrap_mean_ci(
    event_level: pd.DataFrame,
    mode: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a mean of event-level values."""
    values = event_level["value"].to_numpy(float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or repetitions <= 0:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)

    if mode == "hierarchical_sequence_event":
        grouped = {
            str(sequence): group["value"].to_numpy(float)
            for sequence, group in event_level.groupby(
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
            sampled_values: list[np.ndarray] = []
            for sequence in sampled_sequences:
                sequence_values = grouped[str(sequence)]
                sampled_values.append(
                    rng.choice(
                        sequence_values,
                        size=len(sequence_values),
                        replace=True,
                    )
                )
            estimates[index] = float(np.mean(np.concatenate(sampled_values)))
    else:
        n_events = len(values)
        for index in range(repetitions):
            estimates[index] = float(
                np.mean(rng.choice(values, size=n_events, replace=True))
            )

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lower, upper


def canonical_summary(
    frame: pd.DataFrame,
    base_prefix: str,
    bootstrap_mode: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("mae", "bias", "factor2", "under05", "over05")

    for quantity_index, quantity in enumerate(("pga", "pgv")):
        truth_column = f"true_log10_{quantity}"
        prediction_column = f"{base_prefix}_log10_{quantity}"
        tail_mask = frame[f"is_tail_{quantity}"].to_numpy(dtype=bool)

        populations = {
            "overall": np.ones(len(frame), dtype=bool),
            "high_motion_tail": tail_mask,
        }

        for population_index, (population, mask) in enumerate(populations.items()):
            subset = frame.loc[mask].copy()
            n_events = int(subset["event_id"].nunique())
            n_sequences = int(subset["_sequence_group"].nunique())

            for metric_index, metric in enumerate(metrics):
                event_level = event_metric_table(
                    subset,
                    truth_column,
                    prediction_column,
                    metric,
                )
                value = float(event_level["value"].mean())
                ci_lower, ci_upper = bootstrap_mean_ci(
                    event_level,
                    mode=bootstrap_mode,
                    repetitions=repetitions,
                    confidence_level=confidence_level,
                    seed=(
                        seed
                        + 10_000 * quantity_index
                        + 1_000 * population_index
                        + 100 * metric_index
                    ),
                )

                rows.append(
                    {
                        "model": "Cross-Attention Base",
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "value": value,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "confidence_level": confidence_level,
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_mode": bootstrap_mode,
                        "n_events": n_events,
                        "n_sequence_groups": n_sequences,
                        "n_target_rows": int(len(subset)),
                    }
                )

    return pd.DataFrame(rows)


def summary_value(
    summary: pd.DataFrame,
    quantity: str,
    population: str,
    metric: str,
    column: str = "value",
) -> float:
    row = summary.loc[
        (summary["quantity"] == quantity)
        & (summary["population"] == population)
        & (summary["metric"] == metric)
    ]
    if len(row) != 1:
        raise RuntimeError(
            "Expected one summary row for "
            f"{quantity}/{population}/{metric}, found {len(row)}."
        )
    return float(row.iloc[0][column])


def audit_locked_values(
    summary: pd.DataFrame,
    tolerance: float,
) -> None:
    problems: list[str] = []

    for key, expected in LOCKED_EXPECTED.items():
        quantity, population, metric = key
        actual = summary_value(summary, quantity, population, metric)
        if abs(actual - expected) > tolerance:
            problems.append(
                f"{key}: observed={actual:.9f}, expected={expected:.9f}"
            )

    for key, (expected_events, expected_rows) in LOCKED_COUNTS.items():
        quantity, population = key
        row = summary.loc[
            (summary["quantity"] == quantity)
            & (summary["population"] == population)
        ].iloc[0]
        actual_events = int(row["n_events"])
        actual_rows = int(row["n_target_rows"])
        if actual_events != expected_events or actual_rows != expected_rows:
            problems.append(
                f"{key}: counts observed=({actual_events}, {actual_rows}), "
                f"expected=({expected_events}, {expected_rows})"
            )

    if problems:
        raise RuntimeError(
            "Locked manuscript audit failed. The selected CSV may be an older "
            "or non-paired prediction file:\n  " + "\n  ".join(problems)
        )


def infer_tail_threshold(truth: np.ndarray, tail: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    tail = np.asarray(tail, dtype=bool)
    if not tail.any() or tail.all():
        return float(np.nanquantile(truth, 0.90))

    highest_non_tail = float(np.nanmax(truth[~tail]))
    lowest_tail = float(np.nanmin(truth[tail]))

    if highest_non_tail <= lowest_tail:
        return 0.5 * (highest_non_tail + lowest_tail)
    # Fallback if labels and stored floating-point values overlap at a boundary.
    return lowest_tail


def quantile_residual_profile(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    n_bins: int,
) -> pd.DataFrame:
    """
    Build an event-balanced residual profile.

    Motion bins are defined from the complete target-level truth distribution.
    Within each bin, residuals are first averaged across targets for every
    event--resampling repeat and then across repeats for every earthquake.
    The plotted line and band are the median and interquartile range across
    earthquake-level summaries, so earthquakes with many tail targets do not
    dominate the visual diagnostic.
    """
    work = frame[["event_id", "repeat", truth_column, prediction_column]].copy()
    work["truth"] = work[truth_column].to_numpy(float)
    work["residual"] = (
        work[prediction_column].to_numpy(float)
        - work[truth_column].to_numpy(float)
    )
    work["bin"] = pd.qcut(
        work["truth"],
        q=n_bins,
        duplicates="drop",
    )

    event_repeat = (
        work.groupby(
            ["event_id", "repeat", "bin"],
            observed=True,
            sort=False,
        )
        .agg(
            truth=("truth", "mean"),
            residual=("residual", "mean"),
        )
        .reset_index()
    )
    event_level = (
        event_repeat.groupby(
            ["event_id", "bin"],
            observed=True,
            sort=False,
        )
        .agg(
            truth=("truth", "mean"),
            residual=("residual", "mean"),
        )
        .reset_index()
    )

    profile = (
        event_level.groupby("bin", observed=True, sort=False)
        .agg(
            x=("truth", "median"),
            median=("residual", "median"),
            q25=("residual", lambda values: float(np.quantile(values, 0.25))),
            q75=("residual", lambda values: float(np.quantile(values, 0.75))),
            n_events=("event_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    return profile


def stratified_plot_sample(
    frame: pd.DataFrame,
    tail_column: str,
    maximum_rows: int,
    seed: int,
) -> pd.DataFrame:
    if maximum_rows <= 0 or len(frame) <= maximum_rows:
        return frame

    tail = frame.loc[frame[tail_column].astype(bool)]
    non_tail = frame.loc[~frame[tail_column].astype(bool)]

    if len(tail) >= maximum_rows:
        return tail.sample(n=maximum_rows, random_state=seed)

    remaining = maximum_rows - len(tail)
    sampled_non_tail = non_tail.sample(
        n=min(remaining, len(non_tail)),
        random_state=seed,
    )
    return pd.concat([sampled_non_tail, tail], ignore_index=True)


def robust_square_limits(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([truth, prediction]).astype(float)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    span = max(upper - lower, 1e-6)
    return lower - 0.035 * span, upper + 0.035 * span


def panel_letter_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(
        rf"$\bf{{{letter}}}$  {title}",
        loc="left",
        pad=5,
        fontsize=8.2,
    )


def remove_top_right_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.65, length=3.0)
    ax.grid(False)


def add_prediction_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    quantity: str,
    base_prefix: str,
    maximum_scatter: int,
    seed: int,
    letter: str,
) -> None:
    truth_column = f"true_log10_{quantity}"
    prediction_column = f"{base_prefix}_log10_{quantity}"
    tail_column = f"is_tail_{quantity}"

    plotting = stratified_plot_sample(
        frame,
        tail_column=tail_column,
        maximum_rows=maximum_scatter,
        seed=seed,
    )

    truth = frame[truth_column].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    threshold = infer_tail_threshold(
        truth,
        frame[tail_column].to_numpy(bool),
    )
    lower, upper = robust_square_limits(truth, prediction)
    x_line = np.linspace(lower, upper, 256)

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    non_tail_color = cycle[0]
    tail_color = cycle[1]
    reference_color = cycle[7 % len(cycle)]

    non_tail = ~plotting[tail_column].to_numpy(bool)
    tail = plotting[tail_column].to_numpy(bool)

    ax.fill_between(
        x_line,
        x_line - LOG10_FACTOR_2,
        x_line + LOG10_FACTOR_2,
        color=reference_color,
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )
    ax.scatter(
        plotting.loc[non_tail, truth_column],
        plotting.loc[non_tail, prediction_column],
        s=5.0,
        alpha=0.10,
        linewidths=0,
        rasterized=True,
        color=non_tail_color,
        label="Non-tail target",
        zorder=1,
    )
    ax.scatter(
        plotting.loc[tail, truth_column],
        plotting.loc[tail, prediction_column],
        s=8.0,
        alpha=0.32,
        linewidths=0,
        rasterized=True,
        color=tail_color,
        label="High-motion target",
        zorder=2,
    )
    ax.plot(
        x_line,
        x_line,
        linewidth=1.05,
        color=reference_color,
        zorder=3,
    )
    ax.plot(
        x_line,
        x_line - LOG10_FACTOR_2,
        linewidth=0.75,
        linestyle="--",
        color=reference_color,
        alpha=0.75,
        zorder=3,
    )
    ax.plot(
        x_line,
        x_line + LOG10_FACTOR_2,
        linewidth=0.75,
        linestyle="--",
        color=reference_color,
        alpha=0.75,
        zorder=3,
    )
    ax.axvline(
        threshold,
        linewidth=0.85,
        linestyle=":",
        color=tail_color,
        alpha=0.95,
        zorder=3,
    )

    mae = summary_value(summary, quantity, "overall", "mae")
    factor2 = summary_value(summary, quantity, "overall", "factor2")
    bias = summary_value(summary, quantity, "overall", "bias")

    ax.text(
        0.035,
        0.965,
        (
            f"MAE = {mae:.3f}\n"
            f"Bias = {bias:+.3f}\n"
            f"Factor-2 = {100.0 * factor2:.1f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )

    label = QUANTITY_LABELS[quantity]
    axis_label = QUANTITY_AXIS_LABELS[quantity]
    ax.set_xlabel(f"Observed {axis_label}")
    ax.set_ylabel(f"Predicted {axis_label}")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    panel_letter_title(
        ax,
        letter,
        f"Overall {label} prediction",
    )
    remove_top_right_spines(ax)


def add_residual_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    quantity: str,
    base_prefix: str,
    maximum_scatter: int,
    n_bins: int,
    seed: int,
    letter: str,
) -> None:
    truth_column = f"true_log10_{quantity}"
    prediction_column = f"{base_prefix}_log10_{quantity}"
    tail_column = f"is_tail_{quantity}"

    plotting = stratified_plot_sample(
        frame,
        tail_column=tail_column,
        maximum_rows=maximum_scatter,
        seed=seed,
    ).copy()
    plotting["_residual"] = (
        plotting[prediction_column].to_numpy(float)
        - plotting[truth_column].to_numpy(float)
    )

    truth = frame[truth_column].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    residual = prediction - truth
    tail_mask_full = frame[tail_column].to_numpy(bool)
    threshold = infer_tail_threshold(truth, tail_mask_full)
    profile = quantile_residual_profile(
        frame,
        truth_column=truth_column,
        prediction_column=prediction_column,
        n_bins=n_bins,
    )

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    non_tail_color = cycle[0]
    tail_color = cycle[1]
    profile_color = cycle[2]
    reference_color = cycle[7 % len(cycle)]

    non_tail = ~plotting[tail_column].to_numpy(bool)
    tail = plotting[tail_column].to_numpy(bool)

    ax.scatter(
        plotting.loc[non_tail, truth_column],
        plotting.loc[non_tail, "_residual"],
        s=4.5,
        alpha=0.075,
        linewidths=0,
        rasterized=True,
        color=non_tail_color,
        zorder=1,
    )
    ax.scatter(
        plotting.loc[tail, truth_column],
        plotting.loc[tail, "_residual"],
        s=7.5,
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        color=tail_color,
        zorder=2,
    )
    ax.fill_between(
        profile["x"].to_numpy(float),
        profile["q25"].to_numpy(float),
        profile["q75"].to_numpy(float),
        color=profile_color,
        alpha=0.16,
        linewidth=0,
        zorder=3,
        label="Interquartile range",
    )
    ax.plot(
        profile["x"].to_numpy(float),
        profile["median"].to_numpy(float),
        marker="o",
        markersize=2.8,
        linewidth=1.25,
        color=profile_color,
        zorder=4,
        label="Binned median",
    )
    ax.axhline(
        0.0,
        linewidth=0.85,
        color=reference_color,
        zorder=3,
    )
    ax.axhline(
        SEVERE_UNDER_THRESHOLD,
        linewidth=0.85,
        linestyle="--",
        color=reference_color,
        alpha=0.85,
        zorder=3,
    )
    ax.axvline(
        threshold,
        linewidth=0.85,
        linestyle=":",
        color=tail_color,
        alpha=0.95,
        zorder=3,
    )

    overall_bias = summary_value(summary, quantity, "overall", "bias")
    tail_bias = summary_value(summary, quantity, "high_motion_tail", "bias")
    tail_under = summary_value(summary, quantity, "high_motion_tail", "under05")

    ax.text(
        0.035,
        0.965,
        (
            f"Overall bias = {overall_bias:+.3f}\n"
            f"Tail bias = {tail_bias:+.3f}\n"
            rf"Tail $U_{{0.5}}$ = {100.0 * tail_under:.1f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )

    label = QUANTITY_LABELS[quantity]
    axis_label = QUANTITY_AXIS_LABELS[quantity]
    ax.set_xlabel(f"Observed {axis_label}")
    ax.set_ylabel(r"Residual, prediction $-$ observation")

    # Use full-data limits, with a small margin, so the tail is not hidden.
    x_span = max(float(np.max(truth) - np.min(truth)), 1e-6)
    y_low = float(np.min(residual))
    y_high = float(np.max(residual))
    y_span = max(y_high - y_low, 1e-6)
    ax.set_xlim(float(np.min(truth) - 0.025 * x_span), float(np.max(truth) + 0.025 * x_span))
    ax.set_ylim(y_low - 0.035 * y_span, y_high + 0.035 * y_span)

    panel_letter_title(
        ax,
        letter,
        f"{label} residual shift at high motion",
    )
    remove_top_right_spines(ax)


def metric_point(
    summary: pd.DataFrame,
    quantity: str,
    population: str,
    metric: str,
    percent: bool,
) -> tuple[float, float, float]:
    value = summary_value(summary, quantity, population, metric, "value")
    lower = summary_value(summary, quantity, population, metric, "ci_lower")
    upper = summary_value(summary, quantity, population, metric, "ci_upper")
    if percent:
        return 100.0 * value, 100.0 * lower, 100.0 * upper
    return value, lower, upper


def add_population_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    letter: str,
    title: str,
    ylabel: str,
    percent: bool,
) -> None:
    x = np.array([0.0, 1.0])
    populations = ["overall", "high_motion_tail"]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_values: list[float] = []
    for quantity_index, quantity in enumerate(("pga", "pgv")):
        points = [
            metric_point(summary, quantity, population, metric, percent)
            for population in populations
        ]
        values = np.array([point[0] for point in points], dtype=float)
        lower = np.array([point[1] for point in points], dtype=float)
        upper = np.array([point[2] for point in points], dtype=float)
        y_error = np.vstack([values - lower, upper - values])

        color = cycle[quantity_index]
        marker = "o" if quantity == "pga" else "s"

        ax.plot(
            x,
            values,
            linewidth=1.15,
            color=color,
            alpha=0.80,
            zorder=2,
        )
        ax.errorbar(
            x,
            values,
            yerr=y_error,
            fmt=marker,
            markersize=4.5,
            capsize=2.6,
            elinewidth=0.95,
            linewidth=0,
            color=color,
            label=QUANTITY_LABELS[quantity],
            zorder=3,
        )

        all_values.extend(values.tolist())
        for x_value, y_value in zip(x, values):
            text = f"{y_value:.1f}%" if percent else f"{y_value:.3f}"
            ax.annotate(
                text,
                xy=(x_value, y_value),
                xytext=(5, 5 if quantity == "pga" else -10),
                textcoords="offset points",
                fontsize=6.5,
                color=color,
                ha="left",
                va="bottom" if quantity == "pga" else "top",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([POPULATION_LABELS[population] for population in populations])
    ax.set_ylabel(ylabel)
    panel_letter_title(ax, letter, title)
    remove_top_right_spines(ax)
    ax.grid(axis="y", linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)

    low = min(all_values)
    high = max(all_values)
    span = max(high - low, 1e-6)
    lower_limit = max(0.0, low - 0.18 * span)
    upper_limit = high + 0.28 * span
    ax.set_ylim(lower_limit, upper_limit)


def build_wide_supplementary_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for quantity in ("pga", "pgv"):
        for population in ("overall", "high_motion_tail"):
            subset = summary.loc[
                (summary["quantity"] == quantity)
                & (summary["population"] == population)
            ]
            first = subset.iloc[0]
            row: dict[str, object] = {
                "Quantity": QUANTITY_LABELS[quantity],
                "Population": POPULATION_LABELS[population],
                "n events": int(first["n_events"]),
                "n sequence groups": int(first["n_sequence_groups"]),
                "n target rows": int(first["n_target_rows"]),
            }
            for metric in ("mae", "bias", "factor2", "under05", "over05"):
                metric_row = subset.loc[subset["metric"] == metric].iloc[0]
                row[METRIC_LABELS[metric]] = float(metric_row["value"])
                row[f"{METRIC_LABELS[metric]} CI lower"] = float(
                    metric_row["ci_lower"]
                )
                row[f"{METRIC_LABELS[metric]} CI upper"] = float(
                    metric_row["ci_upper"]
                )
            rows.append(row)

    return pd.DataFrame(rows)


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    """Write a compact, manuscript-ready Supplementary Table 1."""
    display_rows: list[dict[str, object]] = []

    for _, row in table.iterrows():
        display_rows.append(
            {
                "Quantity": row["Quantity"],
                "Population": row["Population"],
                "MAE (95\\% CI)": (
                    f"{row['MAE']:.3f} "
                    f"[{row['MAE CI lower']:.3f}, {row['MAE CI upper']:.3f}]"
                ),
                "Bias (95\\% CI)": (
                    f"{row['Bias']:+.3f} "
                    f"[{row['Bias CI lower']:+.3f}, {row['Bias CI upper']:+.3f}]"
                ),
                "Factor-2": f"{100.0 * row['Factor-of-two accuracy']:.1f}\\%",
                "$U_{0.5}$": f"{100.0 * row['$U_{0.5}$']:.1f}\\%",
                "$O_{0.5}$": f"{100.0 * row['$O_{0.5}$']:.1f}\\%",
                "$n_{event}$": int(row["n events"]),
                "$n_{target}$": int(row["n target rows"]),
            }
        )

    display = pd.DataFrame(display_rows)
    latex = display.to_latex(index=False, escape=False)
    note = (
        "\n% Metrics are aggregated from targets to station-resampling repeats "
        "and then to earthquake events. Confidence intervals use the bootstrap "
        "mode recorded in the accompanying CSV.\n"
    )
    path.write_text(latex + note, encoding="utf-8")


def make_caption(
    summary: pd.DataFrame,
    bootstrap_mode: str,
    maximum_scatter: int,
) -> str:
    overall_row = summary.loc[
        (summary["quantity"] == "pga")
        & (summary["population"] == "overall")
    ].iloc[0]
    pga_tail_row = summary.loc[
        (summary["quantity"] == "pga")
        & (summary["population"] == "high_motion_tail")
    ].iloc[0]
    pgv_tail_row = summary.loc[
        (summary["quantity"] == "pgv")
        & (summary["population"] == "high_motion_tail")
    ].iloc[0]

    total_events = int(overall_row["n_events"])
    total_sequences = int(overall_row["n_sequence_groups"])
    total_rows = int(overall_row["n_target_rows"])
    pga_tail_events = int(pga_tail_row["n_events"])
    pga_tail_rows = int(pga_tail_row["n_target_rows"])
    pgv_tail_events = int(pgv_tail_row["n_events"])
    pgv_tail_rows = int(pgv_tail_row["n_target_rows"])

    bootstrap_text = (
        "hierarchical sequence-to-event bootstrap"
        if bootstrap_mode == "hierarchical_sequence_event"
        else "event-level bootstrap"
    )

    shown_text = (
        f"All {total_rows:,} target predictions are shown in the scatter panels."
        if maximum_scatter <= 0 or maximum_scatter >= total_rows
        else (
            "Scatter panels show a stratified subset for legibility, retaining "
            "all high-motion targets; all metrics use the complete test set."
        )
    )

    return (
        "Fig. 2 | Overall predictive skill masks systematic "
        "underprediction in the high-motion tail. "
        "a,b, Observed versus predicted remaining horizontal peak ground "
        "acceleration (PGA_rem,H) and peak ground velocity (PGV_rem,H) "
        "for the frozen "
        "Cross-Attention Base under the locked t0=5 s, five-input-station "
        "protocol. The solid diagonal denotes perfect agreement, dashed lines "
        "denote a factor-of-two interval, and the vertical dotted line denotes "
        "the training-derived upper-decile high-motion threshold. "
        "c,d, Prediction residuals (prediction minus observation) as a function "
        "of observed motion. The solid curve and shaded band show the median and "
        "interquartile range of event-level residual summaries in equal-count "
        "motion bins after target-to-repeat-to-event aggregation; horizontal "
        "reference "
        "lines denote zero bias and severe underprediction at -0.5 log10 units. "
        "e, Event-macro mean absolute error for all targets and high-motion "
        "targets. f, Severe underprediction rate U0.5 for the same populations. "
        f"Error bars are 95% confidence intervals from {bootstrap_text}. "
        "Metrics were aggregated from target stations to station-resampling "
        "repeats and then to earthquake events. The locked test comprised "
        f"{total_events} earthquakes from {total_sequences} sequence groups "
        f"and {total_rows:,} target predictions; "
        f"the PGA tail contained {pga_tail_rows:,} targets from "
        f"{pga_tail_events} events and the PGV tail contained "
        f"{pgv_tail_rows:,} targets from {pgv_tail_events} events. "
        "High-motion labels were defined at the target-station level using "
        "training data only and therefore do not denote a high-magnitude event "
        f"subset. {shown_text} Exact values are provided in Supplementary "
        "Table 1."
    )


def make_figure(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    base_prefix: str,
    maximum_scatter: int,
    n_bins: int,
    seed: int,
    output_directory: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": dpi,
        }
    )

    # Full Nature Communications page width. A 3 x 2 arrangement keeps every
    # panel readable at final print size and prevents conclusion-style titles
    # from colliding.
    width_in = 183.0 / 25.4
    height_in = 172.0 / 25.4
    figure = plt.figure(figsize=(width_in, height_in))
    grid = figure.add_gridspec(
        3,
        2,
        left=0.085,
        right=0.992,
        bottom=0.085,
        top=0.935,
        wspace=0.46,
        hspace=0.58,
    )

    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    ]

    add_prediction_panel(
        axes[0],
        frame,
        summary,
        quantity="pga",
        base_prefix=base_prefix,
        maximum_scatter=maximum_scatter,
        seed=seed,
        letter="a",
    )
    add_prediction_panel(
        axes[1],
        frame,
        summary,
        quantity="pgv",
        base_prefix=base_prefix,
        maximum_scatter=maximum_scatter,
        seed=seed + 1,
        letter="b",
    )
    add_residual_panel(
        axes[2],
        frame,
        summary,
        quantity="pga",
        base_prefix=base_prefix,
        maximum_scatter=maximum_scatter,
        n_bins=n_bins,
        seed=seed + 2,
        letter="c",
    )
    add_residual_panel(
        axes[3],
        frame,
        summary,
        quantity="pgv",
        base_prefix=base_prefix,
        maximum_scatter=maximum_scatter,
        n_bins=n_bins,
        seed=seed + 3,
        letter="d",
    )
    add_population_metric_panel(
        axes[4],
        summary,
        metric="mae",
        letter="e",
        title="High-motion error inflation",
        ylabel=r"MAE ($\log_{10}$ units)",
        percent=False,
    )
    add_population_metric_panel(
        axes[5],
        summary,
        metric="under05",
        letter="f",
        title="Severe high-motion underprediction",
        ylabel=r"Severe underprediction $U_{0.5}$ (%)",
        percent=True,
    )

    # One compact legend for the figure. Residual-profile meaning is explained
    # in the self-contained caption to avoid visual clutter.
    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    metric_handles, metric_labels = axes[4].get_legend_handles_labels()
    for handle, label in zip(metric_handles, metric_labels):
        unique.setdefault(label, handle)

    figure.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.50, 0.993),
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.45,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "Fig2_base_capability_and_tail_failure"
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Nature Communications-style Section 2.1 figure for "
            "Cross-Attention Base capability and high-motion-tail failure."
        )
    )
    parser.add_argument(
        "--predictions",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "locked_test_ablation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="data/model_manifests/scenario_t0_5s_k5_linux.csv",
        help=(
            "Event manifest containing a sequence-group column. Use an empty "
            "string to force event-level bootstrap."
        ),
    )
    parser.add_argument(
        "--sequence-column",
        default="",
        help="Optional explicit sequence-group column name.",
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
        "--base-prefix",
        default="A2_cross_attention_base",
    )
    parser.add_argument(
        "--maximum-scatter",
        type=int,
        default=50_000,
        help=(
            "Maximum points per scatter panel. The current 44,800-row test is "
            "shown in full with the default."
        ),
    )
    parser.add_argument("--residual-bins", type=int, default=14)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--locked-audit-tolerance",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--skip-locked-audit",
        action="store_true",
    )
    parser.add_argument(
        "--out-dir",
        default="figures/nc_section_2_1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prediction_path = Path(args.predictions)
    manifest_path = Path(args.manifest) if str(args.manifest).strip() else None
    output_directory = Path(args.out_dir)

    predictions = load_predictions(
        prediction_path,
        base_prefix=args.base_prefix,
    )
    predictions, bootstrap_mode = attach_sequence_groups(
        predictions,
        manifest_path=manifest_path,
        sequence_column=(args.sequence_column.strip() or None),
        split_column=(args.split_column.strip() or None),
        test_label=args.test_label,
    )

    summary = canonical_summary(
        predictions,
        base_prefix=args.base_prefix,
        bootstrap_mode=bootstrap_mode,
        repetitions=args.bootstrap_repetitions,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )

    if not args.skip_locked_audit:
        audit_locked_values(summary, tolerance=args.locked_audit_tolerance)

    output_directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output_directory / "TableS1_base_overall_tail_metrics_long.csv",
        index=False,
    )

    wide_table = build_wide_supplementary_table(summary)
    wide_table.to_csv(
        output_directory / "TableS1_base_overall_tail_metrics.csv",
        index=False,
    )
    write_latex_table(
        wide_table,
        output_directory / "TableS1_base_overall_tail_metrics.tex",
    )

    make_figure(
        predictions,
        summary,
        base_prefix=args.base_prefix,
        maximum_scatter=args.maximum_scatter,
        n_bins=args.residual_bins,
        seed=args.seed,
        output_directory=output_directory,
        dpi=args.dpi,
    )

    caption = make_caption(
        summary,
        bootstrap_mode=bootstrap_mode,
        maximum_scatter=args.maximum_scatter,
    )
    (output_directory / "Fig2_caption.txt").write_text(
        caption + "\n",
        encoding="utf-8",
    )

    configuration = {
        "predictions": str(prediction_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "base_prefix": args.base_prefix,
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "confidence_level": args.confidence_level,
        "metric_aggregation": "targets -> repeats -> events",
        "maximum_scatter": args.maximum_scatter,
        "residual_bins": args.residual_bins,
        "seed": args.seed,
        "locked_value_audit": not args.skip_locked_audit,
        "locked_audit_tolerance": args.locked_audit_tolerance,
    }
    (output_directory / "run_configuration.txt").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Section 2.1: Base capability and tail failure ===")
    print(f"Prediction rows          : {len(predictions):,}")
    print(f"Events                   : {predictions['event_id'].nunique():,}")
    print(
        "Sequence groups          : "
        f"{predictions['_sequence_group'].nunique():,}"
    )
    print(f"Bootstrap mode           : {bootstrap_mode}")
    print(f"Locked audit             : {'SKIPPED' if args.skip_locked_audit else 'PASSED'}")
    print("\nCanonical metrics:")
    print(
        summary.loc[
            summary["metric"].isin(["mae", "bias", "factor2", "under05"]),
            [
                "quantity",
                "population",
                "metric",
                "value",
                "ci_lower",
                "ci_upper",
                "n_events",
                "n_sequence_groups",
                "n_target_rows",
            ],
        ].to_string(index=False)
    )
    print("\nSaved:")
    for path in (
        output_directory / "Fig2_base_capability_and_tail_failure.png",
        output_directory / "Fig2_base_capability_and_tail_failure.pdf",
        output_directory / "Fig2_base_capability_and_tail_failure.svg",
        output_directory / "TableS1_base_overall_tail_metrics.csv",
        output_directory / "TableS1_base_overall_tail_metrics.tex",
        output_directory / "Fig2_caption.txt",
    ):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
