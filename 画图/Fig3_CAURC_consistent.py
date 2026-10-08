#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig3_CAURC_consistent.py

Nature Communications-style figure:
"Target-specific correction of high-motion underprediction."

Primary input
-------------
runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv

Required manifest for grouped bootstrap
--------------------------------------
data/model_manifests/scenario_t0_5s_k5_linux.csv

Required columns in prediction CSV
----------------------------------
event_id, repeat, target_station_index,
true_log10_pga, true_log10_pgv,
is_tail_pga, is_tail_pgv,
A2_cross_attention_base_log10_pga,
A2_cross_attention_base_log10_pgv,
A4_under_only_log10_pga,
A4_under_only_log10_pgv

Outputs
-------
Fig3_caurc_consistent.png
Fig3_caurc_consistent.pdf
Fig3_caurc_consistent.svg
Fig3_primary_summary.csv
Fig3_aggregation_audit.csv
Fig3_ECDF_consistency_audit.csv
Fig3_input_audit.json
Fig3_results_numbers.txt
Fig3_caption.txt
run_configuration.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Constants
# ============================================================

LOG10_FACTOR_2 = math.log10(2.0)
SEVERE_UNDER_THRESHOLD = -0.5
SEVERE_OVER_THRESHOLD = 0.5

QUANTITY_LABELS = {
    "pga": "PGA",
    "pgv": "PGV",
}

MODEL_LABELS = {
    "base": "Cross-Attention\nBase",
    "final": "CA-URC",
}

COLORS = {
    "pga": "#1f77b4",       # blue
    "pgv": "#e67e22",       # orange
    "base": "#8f98a1",      # grey
    "final": "#c62828",     # red
    "ref": "#66707a",       # reference grey
    "grid": "#e3e7eb",
    "frame": "#2b2f33",
    "text": "#1f2328",
}

# ============================================================
# IO helpers
# ============================================================

def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def as_bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    unknown = set(normalized.unique()) - (truthy | falsy)
    if unknown:
        raise ValueError(
            f"Cannot parse Boolean values in {series.name!r}: {sorted(unknown)[:10]}"
        )
    return normalized.isin(truthy).to_numpy(dtype=bool)


def load_predictions(
    path: Path,
    base_prefix: str,
    final_prefix: str,
) -> pd.DataFrame:
    """Read predictions without dropping, deduplicating or changing outcomes."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {path.resolve()}")
    frame = pd.read_csv(path, dtype={"event_id": str})
    required = [
        "event_id", "repeat", "target_station_index",
        "true_log10_pga", "true_log10_pgv", "is_tail_pga", "is_tail_pgv",
        f"{base_prefix}_log10_pga", f"{base_prefix}_log10_pgv",
        f"{final_prefix}_log10_pga", f"{final_prefix}_log10_pgv",
    ]
    require_columns(frame, required, "Prediction table")
    if frame.empty:
        raise ValueError("The prediction CSV is empty.")
    if frame["event_id"].isna().any():
        raise ValueError("Missing event_id in prediction CSV.")
    frame["event_id"] = frame["event_id"].str.strip()
    if frame["event_id"].eq("").any():
        raise ValueError("Blank event_id in prediction CSV.")
    for name in ("repeat", "target_station_index", "target_slot"):
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain finite integers.")
        if np.any(values < 0):
            raise ValueError(f"{name} must be non-negative.")
        frame[name] = values.astype(np.int64)
    keys = ["event_id", "repeat", "target_station_index"]
    if frame.duplicated(keys).any():
        bad = frame.loc[frame.duplicated(keys, keep=False), keys].head(8)
        raise ValueError("Duplicate event-repeat-target keys:\n" + bad.to_string(index=False))
    if "target_slot" in frame and frame.duplicated(["event_id", "repeat", "target_slot"]).any():
        raise ValueError("Duplicate target slots within event-repeat realizations.")
    for quantity in ("pga", "pgv"):
        frame[f"is_tail_{quantity}"] = as_bool_array(frame[f"is_tail_{quantity}"])
        columns = [f"true_log10_{quantity}", f"{base_prefix}_log10_{quantity}",
                   f"{final_prefix}_log10_{quantity}"]
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(float)).all():
                raise ValueError(f"Non-finite values in {column}; no rows were silently removed.")
    return frame


def validate_protocol(frame: pd.DataFrame, args) -> dict:
    """Check sample counts and necessary (not sufficient) prediction-pairing conditions."""
    n_events = frame["event_id"].nunique()
    repeats = frame.groupby("event_id", sort=False)["repeat"].nunique()
    targets = frame.groupby(["event_id", "repeat"], sort=False).size()
    if args.expected_events and n_events != args.expected_events:
        raise ValueError(f"Expected {args.expected_events} events, found {n_events}.")
    if args.expected_repeats and not repeats.eq(args.expected_repeats).all():
        raise ValueError(f"Not every event has {args.expected_repeats} realizations.")
    if args.expected_targets and not targets.eq(args.expected_targets).all():
        raise ValueError(f"Not every realization has {args.expected_targets} targets.")
    audit = {
        "n_events": int(n_events), "n_event_repeats": int(len(targets)),
        "n_target_rows": int(len(frame)), "duplicate_target_keys": 0,
        "minimum_repeats": int(repeats.min()), "maximum_repeats": int(repeats.max()),
        "minimum_targets_per_repeat": int(targets.min()),
        "maximum_targets_per_repeat": int(targets.max()),
        "upstream_prediction_alignment": "not proven by these necessary checks",
    }
    for quantity in ("pga", "pgv"):
        correction = (frame[f"{args.final_prefix}_log10_{quantity}"].to_numpy(float)
                      - frame[f"{args.base_prefix}_log10_{quantity}"].to_numpy(float))
        tol = args.prediction_tolerance
        if np.any(correction < -tol) or np.any(correction > args.maximum_correction + tol):
            raise ValueError(
                f"{quantity.upper()}: CA-URC - Base is outside [0, {args.maximum_correction}] "
                f"(tolerance {tol:g}); observed [{correction.min():.7g}, {correction.max():.7g}]. "
                "Check the model columns, checkpoint and upstream CSV row alignment. "
                "This script will not alter predictions to make them pass."
            )
        audit[f"correction_range_{quantity}"] = [float(correction.min()), float(correction.max())]
        tail = frame.loc[frame[f"is_tail_{quantity}"]]
        audit[f"tail_{quantity}"] = {
            "n_rows": int(len(tail)), "n_events": int(tail["event_id"].nunique()),
            "n_event_repeats": int(len(tail[["event_id", "repeat"]].drop_duplicates())),
        }
    return audit


def attach_sequence_groups(
    frame: pd.DataFrame,
    manifest_path: Path | None,
    sequence_column: str | None = "sequence_group",
    split_column: str | None = "split_grouped",
    test_label: str = "test",
    bootstrap_mode: str = "hierarchical_group_event",
) -> tuple[pd.DataFrame, str]:
    """Never silently substitute event bootstrap or invented groups for a grouped analysis."""
    frame = frame.copy()
    if manifest_path is None or not manifest_path.is_file():
        if bootstrap_mode != "event":
            target = "not supplied" if manifest_path is None else str(manifest_path.resolve())
            raise FileNotFoundError(
                f"A valid manifest is required for {bootstrap_mode}: {target}. "
                "Supply --manifest. Event bootstrap requires an explicit --bootstrap-mode event."
            )
        frame["sequence_group"] = "not_available"
        return frame, bootstrap_mode
    manifest = pd.read_csv(manifest_path, dtype={"event_id": str})
    names = ["event_id"]
    if sequence_column:
        names.append(sequence_column)
    if split_column:
        names.append(split_column)
    require_columns(manifest, names, "Manifest")
    if manifest["event_id"].isna().any():
        raise ValueError("Missing event_id in manifest.")
    manifest["event_id"] = manifest["event_id"].str.strip()
    mapping = manifest[names].drop_duplicates()
    if mapping.duplicated("event_id").any():
        raise ValueError("An event has conflicting group/split assignments in the manifest.")
    selected_ids = set(frame["event_id"])
    mapping = mapping.loc[mapping["event_id"].isin(selected_ids)].copy()
    missing_ids = selected_ids - set(mapping["event_id"])
    if missing_ids:
        raise ValueError(f"Prediction events absent from manifest: {sorted(missing_ids)[:10]}")
    if split_column:
        bad_split = ~mapping[split_column].astype(str).str.strip().str.lower().eq(test_label.lower())
        if bad_split.any():
            raise ValueError("Prediction CSV contains events not assigned to the requested test split.")
    if not sequence_column:
        if bootstrap_mode != "event":
            raise ValueError("A group column is required for grouped bootstrap.")
        frame["sequence_group"] = "not_available"
        return frame, bootstrap_mode
    if mapping[sequence_column].isna().any():
        raise ValueError("Missing group assignments; refusing event-ID fallback.")
    mapping[sequence_column] = mapping[sequence_column].astype(str).str.strip()
    if mapping[sequence_column].isin(["", "nan", "None", "<NA>"]).any():
        raise ValueError("Blank/invalid group assignments in manifest.")
    group_map = mapping.set_index("event_id")[sequence_column]
    mapped = frame["event_id"].map(group_map)
    if "sequence_group" in frame and not frame["sequence_group"].astype(str).eq(mapped).all():
        raise ValueError("Prediction and manifest group assignments disagree.")
    frame["sequence_group"] = mapped.to_numpy()
    return frame, bootstrap_mode


# ============================================================
# Metric computation
# ============================================================

def compute_residual(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return pred - truth


def per_row_metric_arrays(
    truth: np.ndarray,
    pred: np.ndarray,
) -> dict[str, np.ndarray]:
    residual = pred - truth
    return {
        "residual": residual,
        "mae": np.abs(residual),
        "bias": residual,
        "factor2": (np.abs(residual) <= LOG10_FACTOR_2).astype(float),
        "under05": (residual <= SEVERE_UNDER_THRESHOLD).astype(float),
        "over05": (residual >= SEVERE_OVER_THRESHOLD).astype(float),
    }


def build_event_metric_table(
    frame: pd.DataFrame,
    quantity: str,
    model_prefix: str,
    population: str,
) -> pd.DataFrame:
    """Mask first, then mean(targets) -> mean(nonempty repeats) -> one row/event."""
    if population == "overall":
        subset = frame.copy()
    elif population == "high_motion_tail":
        subset = frame.loc[frame[f"is_tail_{quantity}"]].copy()
    elif population == "non_tail":
        subset = frame.loc[~frame[f"is_tail_{quantity}"]].copy()
    else:
        raise ValueError(f"Unknown population: {population}")
    if subset.empty:
        raise RuntimeError(f"No rows for quantity={quantity}, population={population}.")
    metrics = per_row_metric_arrays(
        subset[f"true_log10_{quantity}"].to_numpy(float),
        subset[f"{model_prefix}_log10_{quantity}"].to_numpy(float),
    )
    metric_names = ("mae", "bias", "factor2", "under05", "over05")
    for key in metric_names:
        subset[key] = metrics[key]
    aggregations = {key: (key, "mean") for key in metric_names}
    repeat_table = (
        subset.groupby(["event_id", "sequence_group", "repeat"], sort=True, dropna=False)
        .agg(**aggregations, n_targets=("event_id", "size"))
        .reset_index()
    )
    event_table = (
        repeat_table.groupby(["event_id", "sequence_group"], sort=True, dropna=False)
        .agg(**aggregations, n_repeats=("repeat", "nunique"), n_rows=("n_targets", "sum"))
        .reset_index()
    )
    return event_table


def bootstrap_event_counts(
    groups: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    repetitions: int,
) -> np.ndarray:
    """Return multiplicities of contributing events, one row per bootstrap draw.

    hierarchical_group_event: sample G groups with replacement; for each drawn
    group g, sample n_g contributing events with replacement from that group.
    Repeated draws of the same group have independent within-group resampling.
    Summing m independent Multinomial(n_g, uniform) draws is exactly equivalent
    to Multinomial(m*n_g, uniform), which is used for efficient implementation.
    group_cluster: sample G groups, retaining all their member events.
    event: sample E contributing events, without using groups.
    Realizations and targets stay together within each event in all modes.
    """
    groups = np.asarray(groups).astype(str)
    n_events = len(groups)
    if n_events == 0:
        raise ValueError("Bootstrap requires at least one contributing event.")
    if mode == "event":
        return rng.multinomial(n_events, np.full(n_events, 1.0 / n_events),
                               size=repetitions).astype(np.int32)
    if mode not in {"hierarchical_group_event", "group_cluster"}:
        raise ValueError(f"Unknown bootstrap mode: {mode}")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    sampled_group_counts = rng.multinomial(
        n_groups, np.full(n_groups, 1.0 / n_groups), size=repetitions
    )
    counts = np.zeros((repetitions, n_events), dtype=np.int32)
    for group_index in range(n_groups):
        indices = np.flatnonzero(inverse == group_index)
        multiplicities = sampled_group_counts[:, group_index]
        if mode == "group_cluster":
            counts[:, indices] = multiplicities[:, None]
            continue
        for multiplicity in np.unique(multiplicities):
            if multiplicity == 0:
                continue
            rows = np.flatnonzero(multiplicities == multiplicity)
            draws = rng.multinomial(
                int(multiplicity) * len(indices), np.full(len(indices), 1.0 / len(indices)),
                size=len(rows),
            )
            counts[np.ix_(rows, indices)] = draws
    if np.any(counts.sum(axis=1) == 0):
        raise AssertionError("An empty bootstrap replicate was generated.")
    return counts


def paired_bootstrap_intervals(
    base_values: np.ndarray,
    final_values: np.ndarray,
    groups: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    repetitions: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the SAME event resamples for both models and every metric in a subset."""
    base_values = np.asarray(base_values, dtype=float)
    final_values = np.asarray(final_values, dtype=float)
    counts = bootstrap_event_counts(groups, mode, rng, repetitions)
    denominator = counts.sum(axis=1, keepdims=True)
    base_boot = counts @ base_values / denominator
    final_boot = counts @ final_values / denominator
    difference_boot = final_boot - base_boot
    alpha = 1.0 - confidence_level
    quantiles = [alpha / 2.0, 1.0 - alpha / 2.0]
    return (
        np.quantile(base_boot, quantiles, axis=0),
        np.quantile(final_boot, quantiles, axis=0),
        np.quantile(difference_boot, quantiles, axis=0),
    )


def summarize_paired_metrics(
    frame: pd.DataFrame,
    base_prefix: str,
    final_prefix: str,
    bootstrap_mode: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    metric_names = ["mae", "bias", "factor2", "under05", "over05"]
    for quantity_index, quantity in enumerate(("pga", "pgv")):
        for population_index, population in enumerate(("overall", "high_motion_tail")):
            base_table = build_event_metric_table(frame, quantity, base_prefix, population)
            final_table = build_event_metric_table(frame, quantity, final_prefix, population)
            paired = base_table.merge(
                final_table, on=["event_id", "sequence_group"], suffixes=("_base", "_final"),
                how="outer", validate="one_to_one", indicator=True,
            )
            if not paired["_merge"].eq("both").all():
                raise AssertionError("The compared models have unmatched contributing events.")
            paired = paired.sort_values(["sequence_group", "event_id"]).reset_index(drop=True)
            base_values = paired[[f"{metric}_base" for metric in metric_names]].to_numpy(float)
            final_values = paired[[f"{metric}_final" for metric in metric_names]].to_numpy(float)
            groups = paired["sequence_group"].to_numpy(str)
            rng = np.random.default_rng(np.random.SeedSequence([seed, quantity_index, population_index]))
            base_ci, final_ci, delta_ci = paired_bootstrap_intervals(
                base_values, final_values, groups, bootstrap_mode, rng,
                repetitions, confidence_level,
            )
            for index, metric in enumerate(metric_names):
                rows.append({
                    "quantity": quantity, "population": population, "metric": metric,
                    "base_value": float(base_values[:, index].mean()),
                    "base_ci_lower": float(base_ci[0, index]),
                    "base_ci_upper": float(base_ci[1, index]),
                    "ca_urc_value": float(final_values[:, index].mean()),
                    "ca_urc_ci_lower": float(final_ci[0, index]),
                    "ca_urc_ci_upper": float(final_ci[1, index]),
                    "delta_value": float((final_values[:, index] - base_values[:, index]).mean()),
                    "delta_ci_lower": float(delta_ci[0, index]),
                    "delta_ci_upper": float(delta_ci[1, index]),
                    "n_events": int(len(paired)),
                    "n_sequence_groups": (int(paired["sequence_group"].nunique())
                                          if not paired["sequence_group"].eq("not_available").all() else 0),
                    "n_event_repeats": int(paired["n_repeats_base"].sum()),
                    "n_target_rows": int(paired["n_rows_base"].sum()),
                    "aggregation": "targets -> nonempty repeats -> contributing events",
                    "bootstrap_mode": bootstrap_mode,
                    "bootstrap_repetitions": repetitions, "confidence_level": confidence_level,
                })
    return pd.DataFrame(rows)


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
            f"Expected exactly one row for {quantity}, {population}, {metric}, got {len(row)}"
        )
    return row.iloc[0]


# ============================================================
# Residual-distribution helpers
# ============================================================

def event_balanced_row_weights(frame: pd.DataFrame) -> np.ndarray:
    """Weights on the CURRENT subset: 1 / (E * R_e * n_er).

    E = contributing events; R_e = nonempty repeats of event e;
    n_er = qualifying targets in event e, repeat r. Empty repeats do not count.
    """
    if frame.empty:
        raise ValueError("Cannot weight an empty subset.")
    n_er = frame.groupby(["event_id", "repeat"], sort=False)["event_id"].transform("size")
    r_e = frame.groupby("event_id", sort=False)["repeat"].transform("nunique")
    n_events = frame["event_id"].nunique()
    weights = 1.0 / (float(n_events) * r_e.to_numpy(float) * n_er.to_numpy(float))
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise AssertionError("Canonical row weights do not sum to one.")
    return weights


def weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape or values.size == 0:
        raise ValueError("ECDF values and weights must have the same nonempty shape.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid ECDF data or weights.")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    unique_values, starts = np.unique(sorted_values, return_index=True)
    masses = np.add.reduceat(weights[order], starts)
    cumulative = np.cumsum(masses) / weights.sum()
    cumulative[-1] = 1.0
    return unique_values, cumulative


def aggregation_audit(frame: pd.DataFrame, base_prefix: str, final_prefix: str) -> pd.DataFrame:
    """Compare the original flat-per-event estimator with the corrected estimator."""
    rows = []
    for quantity in ("pga", "pgv"):
        for population in ("overall", "high_motion_tail"):
            subset = (frame if population == "overall"
                      else frame.loc[frame[f"is_tail_{quantity}"]]).copy()
            for label, prefix in (("Cross-Attention Base", base_prefix), ("CA-URC", final_prefix)):
                arrays = per_row_metric_arrays(subset[f"true_log10_{quantity}"].to_numpy(float),
                                                subset[f"{prefix}_log10_{quantity}"].to_numpy(float))
                canonical = build_event_metric_table(frame, quantity, prefix, population)
                for metric in ("mae", "bias", "factor2", "under05", "over05"):
                    work = subset[["event_id"]].copy()
                    work["value"] = arrays[metric]
                    original_value = float(work.groupby("event_id")["value"].mean().mean())
                    corrected_value = float(canonical[metric].mean())
                    rows.append({"quantity": quantity, "population": population, "model": label,
                                 "metric": metric, "original_event_flat": original_value,
                                 "corrected_target_repeat_event": corrected_value,
                                 "corrected_minus_original": corrected_value - original_value})
    return pd.DataFrame(rows)


def ecdf_consistency_audit(
    frame: pd.DataFrame, summary: pd.DataFrame, base_prefix: str, final_prefix: str,
) -> pd.DataFrame:
    """Every ECDF-weighted metric must equal the corresponding plotted summary."""
    rows = []
    for quantity in ("pga", "pgv"):
        tail = frame.loc[frame[f"is_tail_{quantity}"]]
        weights = event_balanced_row_weights(tail)
        truth = tail[f"true_log10_{quantity}"].to_numpy(float)
        for label, prefix, column in (("Cross-Attention Base", base_prefix, "base_value"),
                                      ("CA-URC", final_prefix, "ca_urc_value")):
            arrays = per_row_metric_arrays(truth, tail[f"{prefix}_log10_{quantity}"].to_numpy(float))
            x, cumulative = weighted_ecdf(arrays["residual"], weights)
            at_threshold = np.searchsorted(x, SEVERE_UNDER_THRESHOLD, side="right") - 1
            cdf_under = 0.0 if at_threshold < 0 else float(cumulative[at_threshold])
            if not np.isclose(cdf_under, np.dot(weights, arrays["under05"]), atol=1e-10, rtol=0):
                raise AssertionError("ECDF threshold disagrees with weighted underprediction rate.")
            for metric in ("mae", "bias", "factor2", "under05", "over05"):
                summary_value = float(get_row(summary, quantity, "high_motion_tail", metric)[column])
                weighted_value = float(np.dot(weights, arrays[metric]))
                if not np.isclose(summary_value, weighted_value, atol=1e-10, rtol=0):
                    raise AssertionError(f"ECDF and summary disagree: {quantity}, {label}, {metric}.")
                rows.append({"quantity": quantity, "model": label, "metric": metric,
                             "summary_value": summary_value, "ecdf_weighted_value": weighted_value,
                             "absolute_difference": abs(summary_value - weighted_value)})
    return pd.DataFrame(rows)


# ============================================================
# Styling
# ============================================================

def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 8.6,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.9,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(COLORS["frame"])
    ax.tick_params(colors=COLORS["text"], width=0.8, length=3.2)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        pad=4.0,
        fontsize=7.2,
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


def percent_label(x: float) -> str:
    return f"{100.0 * x:.1f}%"


# ============================================================
# Plot panels
# ============================================================

def delta_overall_mae_panel(
    ax: plt.Axes, summary: pd.DataFrame, letter: str, title: str,
) -> None:
    positions = [1, 0]
    extrema = []
    for y, quantity in zip(positions, ("pga", "pgv")):
        row = get_row(summary, quantity, "overall", "mae")
        delta, low, high = map(float, (row["delta_value"], row["delta_ci_lower"], row["delta_ci_upper"]))
        ax.hlines(y, low, high, color=COLORS[quantity], linewidth=1.2, zorder=2)
        ax.scatter(delta, y, s=28, color=COLORS[quantity], edgecolor="white", linewidth=0.45, zorder=3)
        ax.text(0.97, 0.77 if quantity == "pga" else 0.31,
                f"{QUANTITY_LABELS[quantity]}  {delta:+.4f}", transform=ax.transAxes,
                fontsize=6.6, color=COLORS[quantity], ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.4))
        extrema.extend([low, high, delta])
    ax.axvline(0, color=COLORS["ref"], linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(["PGA", "PGV"])
    ax.set_xlabel(r"$\Delta$ MAE ($\log_{10}$ units)" + "\nCA-URC − Cross-Attention Base", fontsize=6.7)
    limit = max(0.012, 1.20 * max(abs(value) for value in extrema))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.6, 1.6)
    panel_title(ax, letter, title)
    clean_axis(ax, grid_axis="x")



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
    x = np.array([0.0, 1.0])
    offsets = {"pga": -0.04, "pgv": 0.04}
    markers = {"pga": "o", "pgv": "s"}

    all_low = []
    all_high = []

    for q in ("pga", "pgv"):
        row = get_row(summary, q, population, metric)

        scale = 100.0 if percent else 1.0
        values = np.array([row["base_value"], row["ca_urc_value"]], dtype=float) * scale
        lower = np.array([row["base_ci_lower"], row["ca_urc_ci_lower"]], dtype=float) * scale
        upper = np.array([row["base_ci_upper"], row["ca_urc_ci_upper"]], dtype=float) * scale
        xx = x + offsets[q]
        ax.plot(xx, values, color=COLORS[q], linewidth=1.15, alpha=0.95, zorder=2)
        ax.vlines(xx, lower, upper, color=COLORS[q], linewidth=0.8, zorder=2)
        for xi, lo, hi in zip(xx, lower, upper):
            ax.hlines([lo, hi], xi - 0.023, xi + 0.023, color=COLORS[q], linewidth=0.8, zorder=2)
        ax.scatter(xx, values, s=23, marker=markers[q], color=COLORS[q],
                   edgecolor="white", linewidth=0.45, zorder=3)

        delta = float(row["delta_value"]) * scale
        delta_low = float(row["delta_ci_lower"]) * scale
        delta_high = float(row["delta_ci_upper"]) * scale

        if percent:
            effect_text = f"{QUANTITY_LABELS[q]}: {delta:+.1f} pp [{delta_low:+.1f}, {delta_high:+.1f}]"
        else:
            effect_text = f"{QUANTITY_LABELS[q]}: {delta:+.3f} [{delta_low:+.3f}, {delta_high:+.3f}]"

        y_text = 0.965 if q == "pga" else 0.835
        ax.text(
            0.03, y_text, effect_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=6.0,
            color=COLORS[q],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.4),
            zorder=4,
        )

        all_low.extend(lower.tolist())
        all_high.extend(upper.tolist())

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS["base"], MODEL_LABELS["final"]])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel(ylabel)
    panel_title(ax, letter, title)
    clean_axis(ax, grid_axis="y")

    y_min = min(all_low)
    y_max = max(all_high)
    span = max(y_max - y_min, 1e-4)

    lower_pad = 0.18 * span
    upper_pad = 0.40 * span

    lower = max(0.0, y_min - lower_pad) if metric in {"mae", "factor2", "under05", "over05"} else y_min - lower_pad
    upper = y_max + upper_pad
    if percent:
        upper = min(100.0, upper)
    ax.set_ylim(lower, upper)


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
    base_res = tail[f"{base_prefix}_log10_{quantity}"].to_numpy(float) - truth
    final_res = tail[f"{final_prefix}_log10_{quantity}"].to_numpy(float) - truth
    weights = event_balanced_row_weights(tail)

    base_x, base_y = weighted_ecdf(base_res, weights)
    final_x, final_y = weighted_ecdf(final_res, weights)

    ax.step(
        base_x, base_y, where="post",
        linewidth=1.05,
        linestyle=(0, (4.0, 2.0)),
        color=COLORS["base"],
        label="Cross-Attention Base",
        zorder=2,
    )
    ax.step(
        final_x, final_y, where="post",
        linewidth=1.65,
        color=COLORS[quantity],
        label="CA-URC",
        zorder=3,
    )

    ax.axvline(
        SEVERE_UNDER_THRESHOLD,
        linewidth=0.80,
        linestyle=(0, (1.3, 1.8)),
        color=COLORS["ref"],
        zorder=1,
    )
    ax.axvline(
        0.0,
        linewidth=0.75,
        linestyle=(0, (3.0, 2.0)),
        color=COLORS["ref"],
        alpha=0.85,
        zorder=1,
    )

    combined = np.concatenate([base_res, final_res])
    x_low = float(np.quantile(combined, 0.005))
    x_high = float(np.quantile(combined, 0.995))
    x_low = min(x_low, -0.8)
    x_high = max(x_high, 0.18)
    span = max(x_high - x_low, 1e-3)
    ax.set_xlim(x_low - 0.04 * span, x_high + 0.04 * span)
    ax.set_ylim(0.0, 1.0)

    base_u = float(get_row(summary, quantity, "high_motion_tail", "under05")["base_value"])
    final_u = float(get_row(summary, quantity, "high_motion_tail", "under05")["ca_urc_value"])
    base_f2 = float(get_row(summary, quantity, "high_motion_tail", "factor2")["base_value"])
    final_f2 = float(get_row(summary, quantity, "high_motion_tail", "factor2")["ca_urc_value"])

    ax.text(
        0.98, 0.05,
        f"$U_{{0.5}}$: {100 * base_u:.1f}% → {100 * final_u:.1f}%\n"
        f"Factor-of-two: {100 * base_f2:.1f}% → {100 * final_f2:.1f}%",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=5.3,
        # 删除 bbox，完全透明无背景
        zorder=4,
    )

    ax.set_xlabel(r"Residual ($\log_{10}$ units)")
    ax.set_ylabel("Event-balanced cumulative fraction")
    panel_title(ax, letter, title)
    clean_axis(ax, grid_axis="y")


# ============================================================
# Figure assembly
# ============================================================

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
    height_in = 150.0 / 25.4

    fig, axes = plt.subplots(2, 3, figsize=(width_in, height_in), facecolor="white")

    delta_overall_mae_panel(
        axes[0, 0],
        summary,
        letter="a",
        title="Catalogue-wide MAE change",
    )

    paired_metric_panel(
        axes[0, 1],
        summary,
        population="high_motion_tail",
        metric="mae",
        letter="b",
        title="High-motion-tail MAE",
        ylabel=r"High-motion-tail MAE ($\log_{10}$ units)",
        percent=False,
    )

    paired_metric_panel(
        axes[0, 2],
        summary,
        population="high_motion_tail",
        metric="under05",
        letter="c",
        title="Severe-underprediction rate",
        ylabel=r"High-motion-tail $U_{0.5}$ (%)",
        percent=True,
    )

    paired_metric_panel(
        axes[1, 0],
        summary,
        population="high_motion_tail",
        metric="factor2",
        letter="d",
        title="Factor-of-two accuracy",
        ylabel="High-motion-tail factor-of-two accuracy (%)",
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
        title="PGA high-motion-tail residuals",
    )

    residual_ecdf_panel(
        axes[1, 2],
        frame,
        quantity="pgv",
        base_prefix=base_prefix,
        final_prefix=final_prefix,
        summary=summary,
        letter="f",
        title="PGV high-motion-tail residuals",
    )

    # Figure-level legend: PGA vs PGV
    fig.legend(
        handles=[
            Line2D(
                [], [], linestyle="none", marker="o", markersize=4.8,
                markerfacecolor=COLORS["pga"], markeredgecolor="white",
                markeredgewidth=0.4, label="PGA"
            ),
            Line2D(
                [], [], linestyle="none", marker="s", markersize=4.8,
                markerfacecolor=COLORS["pgv"], markeredgecolor="white",
                markeredgewidth=0.4, label="PGV"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # Residual model-style legends
    # Residual model-style legends
    for ax in (axes[1, 1], axes[1, 2]):
        ax.legend(
            frameon=False,  # 去掉白色背景
            loc="upper left",
            fontsize=5.5,  # 缩小字体，避免与曲线重叠
            handlelength=2.1,
            handletextpad=0.35,
            labelspacing=0.15,
            borderaxespad=0.20,
        )

    fig.subplots_adjust(
        left=0.075,
        right=0.987,
        bottom=0.083,
        top=0.915,
        wspace=0.30,
        hspace=0.30,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Fig3_caurc_consistent"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


# ============================================================
# Outputs
# ============================================================

def write_caption(summary: pd.DataFrame, bootstrap_mode: str, out_path: Path) -> None:
    pga_tail = get_row(summary, "pga", "high_motion_tail", "mae")
    pgv_tail = get_row(summary, "pgv", "high_motion_tail", "mae")
    overall = get_row(summary, "pga", "overall", "mae")
    mode_text = {
        "hierarchical_group_event": (
            "paired hierarchical group-to-event bootstrap resampling: groups were sampled with "
            "replacement, followed by sampling contributing events within each selected group"
        ),
        "group_cluster": "paired group-cluster bootstrap resampling, retaining all events of each sampled group",
        "event": "paired event-level bootstrap resampling",
    }[bootstrap_mode]
    n_groups = int(overall["n_sequence_groups"])
    group_text = f" in {n_groups} constructed spatiotemporal groups" if n_groups else ""
    confidence = 100 * float(overall["confidence_level"])
    repetitions = int(overall["bootstrap_repetitions"])
    caption = (
        "Fig. 3 | Target-specific correction of high-motion underprediction. "
        "a, Paired change in catalogue-wide event-balanced mean absolute error (MAE), defined as "
        "CA-URC minus the frozen Cross-Attention Base. The vertical reference denotes zero change; "
        "an interval containing zero does not establish equivalence. "
        "b, High-motion-tail MAE. c, Severe-underprediction rate in the high-motion tail, U_0.5, "
        "defined by a log10 residual at or below -0.5. d, High-motion-tail factor-of-two accuracy, "
        "defined by an absolute log10 residual at or below log10(2). Blue circles and orange squares "
        "denote PGA and PGV. Points and error bars in b-d show model estimates and their confidence "
        "intervals; annotations show paired changes and the corresponding confidence intervals. "
        "Rate changes are expressed in percentage points (pp). "
        "e,f, Event-balanced empirical cumulative distributions of high-motion-tail residuals "
        "for PGA and PGV. Residuals are prediction minus observation. Grey dashed curves denote "
        "the Cross-Attention Base and solid curves denote CA-URC. Vertical lines mark -0.5 and zero. "
        "All panels first restrict to the relevant target subset, then average targets within "
        "nonempty realizations, realizations within earthquakes and finally earthquakes. ECDF "
        "row weights follow the same hierarchy. Empty realizations and non-contributing events "
        "are excluded from subset estimates. "
        f"Intervals are {confidence:g}% percentile intervals from {repetitions:,} replicates of {mode_text}. "
        "The same resampled events were used for both models and their paired differences; "
        "within-event realizations and targets were retained together. Group resampling used "
        "the groups represented in the relevant analysis subset. "
        f"The retrospective benchmark comprised {int(overall['n_events'])} earthquakes{group_text} "
        f"and {int(overall['n_target_rows']):,} target predictions. The PGA tail contained "
        f"{int(pga_tail['n_target_rows']):,} predictions from {int(pga_tail['n_events'])} earthquakes; "
        f"the PGV tail contained {int(pgv_tail['n_target_rows']):,} predictions from "
        f"{int(pgv_tail['n_events'])} earthquakes. Tail labels were read unchanged from the prediction CSV. "
        "All numerical annotations were calculated from the same event-balanced summary."
    )
    if bootstrap_mode == "event":
        caption = caption.replace("Group resampling used the groups represented in the relevant analysis subset. ", "")
    out_path.write_text(caption + "\n", encoding="utf-8")


def write_numerical_results(summary: pd.DataFrame, path: Path) -> None:
    """One source for manuscript numbers; no previously reported values are hard-coded."""
    lines = [
        "Recomputed Fig. 3 numbers (CA-URC minus Cross-Attention Base)",
        "MAE/bias: log10 units. Rates: percent. Rate changes: percentage points.",
        "Use unrounded differences; do not subtract separately rounded printed percentages.", "",
    ]
    for quantity in ("pga", "pgv"):
        lines.append(QUANTITY_LABELS[quantity])
        for population in ("overall", "high_motion_tail"):
            for metric in ("mae", "bias", "under05", "factor2", "over05"):
                row = get_row(summary, quantity, population, metric)
                scale = 100.0 if metric in {"under05", "factor2", "over05"} else 1.0
                digits = 2 if scale == 100 else 6
                base = scale * row["base_value"]
                final = scale * row["ca_urc_value"]
                delta = scale * row["delta_value"]
                low = scale * row["delta_ci_lower"]
                high = scale * row["delta_ci_upper"]
                lines.append(f"  {population}, {metric}: {base:.{digits}f} -> {final:.{digits}f}; "
                             f"paired change {delta:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]")
                if metric == "mae" and row["base_value"] > 0:
                    improvement = 100.0 * (row["base_value"] - row["ca_urc_value"]) / row["base_value"]
                    lines.append(f"    Relative MAE reduction: {improvement:.3f}%")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "Recompute Fig. 3 using targets -> repeats -> events, matching ECDF weights "
        "and explicitly specified paired bootstrap. No paper numbers are hard-coded."
    ))
    parser.add_argument("--predictions", default="runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv")
    parser.add_argument("--manifest", default="model_manifests/scenario_t0_5s_k5_linux.csv")
    parser.add_argument("--sequence-column", default="sequence_group")
    parser.add_argument("--split-column", default="split_grouped")
    parser.add_argument("--test-label", default="test")
    parser.add_argument("--base-prefix", default="A2_cross_attention_base")
    parser.add_argument("--final-prefix", default="A4_under_only")
    parser.add_argument("--bootstrap-mode", choices=["hierarchical_group_event", "group_cluster", "event"],
                        default="hierarchical_group_event")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--expected-events", type=int, default=224, help="0 disables this count check.")
    parser.add_argument("--expected-repeats", type=int, default=20, help="0 disables this count check.")
    parser.add_argument("--expected-targets", type=int, default=10, help="0 disables this count check.")
    parser.add_argument("--expected-groups", type=int, default=22, help="0 disables this count check.")
    parser.add_argument("--maximum-correction", type=float, default=1.5)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-5)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--out-dir", default="figures/Fig3_caurc_consistent")
    args = parser.parse_args()
    if not 0 < args.confidence_level < 1:
        parser.error("--confidence-level must lie strictly between 0 and 1.")
    if args.bootstrap_repetitions < 100:
        parser.error("Use at least 100 bootstrap replicates (10,000 for the paper).")
    if args.maximum_correction <= 0 or args.prediction_tolerance < 0:
        parser.error("Invalid correction cap or numerical tolerance.")
    if args.seed < 0 or min(args.expected_events, args.expected_repeats,
                           args.expected_targets, args.expected_groups) < 0:
        parser.error("Seeds and expected counts must be non-negative.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_predictions(Path(args.predictions), args.base_prefix, args.final_prefix)
    audit = validate_protocol(frame, args)
    frame, bootstrap_mode = attach_sequence_groups(
        frame, Path(args.manifest) if args.manifest.strip() else None,
        args.sequence_column or None, args.split_column or None, args.test_label, args.bootstrap_mode,
    )
    has_groups = not frame["sequence_group"].eq("not_available").all()
    n_groups = int(frame["sequence_group"].nunique()) if has_groups else 0
    if args.expected_groups and bootstrap_mode != "event" and n_groups != args.expected_groups:
        raise ValueError(f"Expected {args.expected_groups} groups, found {n_groups}.")
    audit["n_constructed_spatiotemporal_groups"] = n_groups
    audit["bootstrap_mode"] = bootstrap_mode
    print(f"Rows={len(frame):,}; events={frame['event_id'].nunique()}; groups={n_groups}")
    print("Aggregation: targets -> nonempty repeats -> contributing events")
    print(f"Bootstrap: {bootstrap_mode}; replicates={args.bootstrap_repetitions:,}")
    summary = summarize_paired_metrics(
        frame, args.base_prefix, args.final_prefix, bootstrap_mode,
        args.bootstrap_repetitions, args.confidence_level, args.seed,
    )
    differences = aggregation_audit(frame, args.base_prefix, args.final_prefix)
    ecdf_audit = ecdf_consistency_audit(frame, summary, args.base_prefix, args.final_prefix)
    audit["maximum_ecdf_summary_difference"] = float(ecdf_audit["absolute_difference"].max())
    summary.to_csv(out_dir / "Fig3_primary_summary.csv", index=False)
    differences.to_csv(out_dir / "Fig3_aggregation_audit.csv", index=False)
    ecdf_audit.to_csv(out_dir / "Fig3_ECDF_consistency_audit.csv", index=False)
    (out_dir / "Fig3_input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_caption(summary, bootstrap_mode, out_dir / "Fig3_caption.txt")
    write_numerical_results(summary, out_dir / "Fig3_results_numbers.txt")
    make_figure(frame, summary, args.base_prefix, args.final_prefix, out_dir, args.dpi)
    config = vars(args).copy()
    config["bootstrap_mode_used"] = bootstrap_mode
    config["estimand"] = "targets -> nonempty repeats -> contributing events"
    config["input_sha256"] = {
        "predictions": file_sha256(Path(args.predictions)),
        "manifest": file_sha256(Path(args.manifest)) if args.manifest.strip() and Path(args.manifest).is_file() else None,
    }
    (out_dir / "run_configuration.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nHigh-motion-tail estimates:")
    columns = ["quantity", "metric", "base_value", "ca_urc_value", "delta_value", "delta_ci_lower", "delta_ci_upper"]
    print(summary.loc[summary["population"].eq("high_motion_tail"), columns].to_string(index=False))
    print(f"\nECDF consistency: PASS; max difference={audit['maximum_ecdf_summary_difference']:.3g}")
    print(f"Outputs: {out_dir.resolve()}")


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()