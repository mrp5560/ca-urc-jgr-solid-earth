#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig3_NC_caurc_primary_results_clean.py

Nature Communications-style figure:
"Underprediction-risk correction selectively reduces high-motion errors
while preserving catalog-wide accuracy."

Primary input
-------------
runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv

Optional manifest
-----------------
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
Fig3_caurc_selective_tail_correction_clean.png
Fig3_caurc_selective_tail_correction_clean.pdf
Fig3_caurc_selective_tail_correction_clean.svg
Fig3_primary_summary.csv
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

    frame["is_tail_pga"] = as_bool_array(frame["is_tail_pga"])
    frame["is_tail_pgv"] = as_bool_array(frame["is_tail_pgv"])

    return frame


def attach_sequence_groups(
    frame: pd.DataFrame,
    manifest_path: Path | None,
    sequence_column: str | None = "sequence_group",
    split_column: str | None = "split_grouped",
    test_label: str = "test",
) -> tuple[pd.DataFrame, str]:
    """
    If manifest exists and contains sequence group information, use
    hierarchical sequence->event bootstrap. Otherwise fall back to
    event-level paired bootstrap.
    """
    frame = frame.copy()

    if manifest_path is None or (not manifest_path.exists()):
        frame["sequence_group"] = frame["event_id"]
        return frame, "event"

    manifest = pd.read_csv(manifest_path, dtype={"event_id": str})
    manifest["event_id"] = manifest["event_id"].astype(str).str.strip()

    if sequence_column is None or sequence_column not in manifest.columns:
        frame["sequence_group"] = frame["event_id"]
        return frame, "event"

    keep = ["event_id", sequence_column]
    if split_column and split_column in manifest.columns:
        keep.append(split_column)

    manifest = manifest[keep].drop_duplicates("event_id")

    if split_column and split_column in manifest.columns:
        manifest = manifest.loc[
            manifest[split_column].astype(str).str.strip().str.lower()
            == str(test_label).strip().lower()
        ].copy()

    merged = frame.merge(manifest, on="event_id", how="left")

    if merged[sequence_column].isna().any():
        merged[sequence_column] = merged[sequence_column].fillna(merged["event_id"])

    merged["sequence_group"] = merged[sequence_column].astype(str)
    return merged, "hierarchical_sequence_event"


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
    """
    Build event-macro metric table. Each event contributes one row,
    with event-level mean of row-level metrics.
    """
    if population == "overall":
        subset = frame.copy()
    elif population == "high_motion_tail":
        subset = frame.loc[frame[f"is_tail_{quantity}"]].copy()
    else:
        raise ValueError(f"Unknown population: {population}")

    if subset.empty:
        raise RuntimeError(f"No rows found for quantity={quantity}, population={population}")

    truth = subset[f"true_log10_{quantity}"].to_numpy(float)
    pred = subset[f"{model_prefix}_log10_{quantity}"].to_numpy(float)

    metrics = per_row_metric_arrays(truth, pred)
    for key, values in metrics.items():
        subset[f"_metric_{key}"] = values
    group_cols = ["event_id", "sequence_group"]

    agg = subset.groupby(group_cols).agg(
        mae=("_metric_mae", "mean"),
        bias=("_metric_bias", "mean"),
        factor2=("_metric_factor2", "mean"),
        under05=("_metric_under05", "mean"),
        over05=("_metric_over05", "mean"),
    ).reset_index()
    repeat_table = (
        subset
        .groupby(
            [
                "event_id",
                "sequence_group",
                "repeat",
            ],
            dropna=False,
        )
        .agg(
            mae=("_metric_mae", "mean"),
            bias=("_metric_bias", "mean"),
            factor2=("_metric_factor2", "mean"),
            under05=("_metric_under05", "mean"),
            over05=("_metric_over05", "mean"),
            n_targets=("event_id", "size"),
        )
        .reset_index()
    )
    event_table = (
        repeat_table
        .groupby(
            [
                "event_id",
                "sequence_group",
            ],
            dropna=False,
        )
        .agg(
            mae=("mae", "mean"),
            bias=("bias", "mean"),
            factor2=("factor2", "mean"),
            under05=("under05", "mean"),
            over05=("over05", "mean"),
            n_repeats=("repeat", "nunique"),
            n_rows=("n_targets", "sum"),
        )
        .reset_index()
    )
    return agg


def paired_bootstrap_delta(
    paired: pd.DataFrame,
    metric: str,
    mode: str,
    rng: np.random.Generator,
    repetitions: int = 10000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap CI for mean(final - base).
    """
    alpha = 1.0 - confidence_level
    deltas = []

    if mode == "hierarchical_sequence_event":
        groups = paired["sequence_group"].astype(str).unique()
        grouped = {g: paired.loc[paired["sequence_group"].astype(str) == g] for g in groups}

        for _ in range(repetitions):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            sampled_parts = [grouped[g] for g in sampled_groups]
            boot = pd.concat(sampled_parts, axis=0, ignore_index=True)
            delta = (boot[f"{metric}_final"] - boot[f"{metric}_base"]).mean()
            deltas.append(delta)
    else:
        n = len(paired)
        idx = np.arange(n)
        for _ in range(repetitions):
            sampled = rng.choice(idx, size=n, replace=True)
            boot = paired.iloc[sampled]
            delta = (boot[f"{metric}_final"] - boot[f"{metric}_base"]).mean()
            deltas.append(delta)

    deltas = np.asarray(deltas, dtype=float)
    low = float(np.quantile(deltas, alpha / 2))
    high = float(np.quantile(deltas, 1 - alpha / 2))
    return low, high


def paired_bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    repetitions: int = 10000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap CI for mean(values), optionally resampling by sequence groups.
    """
    alpha = 1.0 - confidence_level
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)

    boot_means = []

    if mode == "hierarchical_sequence_event":
        unique_groups = np.unique(groups)
        group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

        for _ in range(repetitions):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled_idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
            boot_means.append(values[sampled_idx].mean())
    else:
        idx = np.arange(len(values))
        for _ in range(repetitions):
            sampled = rng.choice(idx, size=len(idx), replace=True)
            boot_means.append(values[sampled].mean())

    boot_means = np.asarray(boot_means, dtype=float)
    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return low, high


def summarize_paired_metrics(
    frame: pd.DataFrame,
    base_prefix: str,
    final_prefix: str,
    bootstrap_mode: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []

    for quantity in ("pga", "pgv"):
        for population in ("overall", "high_motion_tail"):
            base_table = build_event_metric_table(frame, quantity, base_prefix, population)
            final_table = build_event_metric_table(frame, quantity, final_prefix, population)

            paired = base_table.merge(
                final_table,
                on=["event_id", "sequence_group"],
                suffixes=("_base", "_final"),
                how="inner",
            )

            if paired.empty:
                raise RuntimeError(
                    f"No paired events after merge for quantity={quantity}, population={population}"
                )

            for metric in ("mae", "bias", "factor2", "under05", "over05"):
                base_values = paired[f"{metric}_base"].to_numpy(float)
                final_values = paired[f"{metric}_final"].to_numpy(float)
                groups = paired["sequence_group"].astype(str).to_numpy()

                base_mean = float(base_values.mean())
                final_mean = float(final_values.mean())
                delta = float((final_values - base_values).mean())

                base_low, base_high = paired_bootstrap_mean_ci(
                    base_values, groups, bootstrap_mode, rng,
                    repetitions=repetitions,
                    confidence_level=confidence_level,
                )
                final_low, final_high = paired_bootstrap_mean_ci(
                    final_values, groups, bootstrap_mode, rng,
                    repetitions=repetitions,
                    confidence_level=confidence_level,
                )
                delta_low, delta_high = paired_bootstrap_delta(
                    paired, metric, bootstrap_mode, rng,
                    repetitions=repetitions,
                    confidence_level=confidence_level,
                )

                rows.append({
                    "quantity": quantity,
                    "population": population,
                    "metric": metric,
                    "base_value": base_mean,
                    "base_ci_lower": base_low,
                    "base_ci_upper": base_high,
                    "ca_urc_value": final_mean,
                    "ca_urc_ci_lower": final_low,
                    "ca_urc_ci_upper": final_high,
                    "delta_value": delta,
                    "delta_ci_lower": delta_low,
                    "delta_ci_upper": delta_high,
                    "n_events": int(paired["event_id"].nunique()),
                    "n_sequence_groups": int(paired["sequence_group"].astype(str).nunique()),
                    "n_target_rows": int(
                        frame.loc[
                            frame[f"is_tail_{quantity}"] if population == "high_motion_tail"
                            else np.ones(len(frame), dtype=bool)
                        ].shape[0]
                    ),
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
    counts = frame.groupby("event_id")["event_id"].transform("size").to_numpy(float)
    return 1.0 / counts


def weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    y = np.cumsum(w)
    y = y / y[-1]
    return x, y


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
    ax: plt.Axes,
    summary: pd.DataFrame,
    letter: str,
    title: str,
) -> None:
    y_positions = [1, 0]
    quantities = ["pga", "pgv"]

    for y, q in zip(y_positions, quantities):
        row = get_row(summary, q, "overall", "mae")
        delta = float(row["delta_value"])
        low = float(row["delta_ci_lower"])
        high = float(row["delta_ci_upper"])

        ax.hlines(y, low, high, color=COLORS[q], linewidth=1.2, zorder=2)
        ax.scatter(delta, y, s=28, color=COLORS[q], edgecolor="white", linewidth=0.45, zorder=3)

        ax.text(
            high + 0.0008, y + 0.08,
            f"{QUANTITY_LABELS[q]}  {delta:+.4f}",
            fontsize=6.6,
            color=COLORS[q],
            ha="left", va="center",
        )

    ax.axvline(0.0, color=COLORS["ref"], linestyle=(0, (3.0, 2.0)), linewidth=0.9, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([QUANTITY_LABELS[q] for q in quantities])
    ax.set_xlabel(r"$\Delta$ MAE (CA-URC $-$ Base, $\log_{10}$ units)")
    ax.set_ylabel("")
    panel_title(ax, letter, title)
    clean_axis(ax, grid_axis="x")

    lim = 0.012
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, 1.6)



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
        yerr = np.vstack([values - lower, upper - values])

        xx = x + offsets[q]
        ax.plot(xx, values, color=COLORS[q], linewidth=1.15, alpha=0.95, zorder=2)
        ax.errorbar(
            xx, values, yerr=yerr,
            fmt=markers[q],
            markersize=4.5,
            capsize=2.4,
            capthick=0.8,
            elinewidth=0.8,
            color=COLORS[q],
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )

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
            fontsize=5.8,
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
        f"$U_{{0.5}}$: {100*base_u:.1f}% → {100*final_u:.1f}%\n"
        f"Factor-2: {100*base_f2:.1f}% → {100*final_f2:.1f}%",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=5.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.45),
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
    height_in = 142.0 / 25.4

    fig, axes = plt.subplots(2, 3, figsize=(width_in, height_in), facecolor="white")

    delta_overall_mae_panel(
        axes[0, 0],
        summary,
        letter="a",
        title="Overall MAE preserved",
    )

    paired_metric_panel(
        axes[0, 1],
        summary,
        population="high_motion_tail",
        metric="mae",
        letter="b",
        title="Tail MAE reduced",
        ylabel=r"High-motion-tail MAE ($\log_{10}$ units)",
        percent=False,
    )

    paired_metric_panel(
        axes[0, 2],
        summary,
        population="high_motion_tail",
        metric="under05",
        letter="c",
        title="Underprediction reduced",
        ylabel=r"High-motion-tail $U_{0.5}$ (%)",
        percent=True,
    )

    paired_metric_panel(
        axes[1, 0],
        summary,
        population="high_motion_tail",
        metric="factor2",
        letter="d",
        title="Factor-two accuracy improved",
        ylabel="High-motion-tail factor-two accuracy (%)",
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
        title="PGA tail residuals",
    )

    residual_ecdf_panel(
        axes[1, 2],
        frame,
        quantity="pgv",
        base_prefix=base_prefix,
        final_prefix=final_prefix,
        summary=summary,
        letter="f",
        title="PGV tail residuals",
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
    for ax in (axes[1, 1], axes[1, 2]):
        ax.legend(
            frameon=False,
            loc="upper left",
            handlelength=2.6,
            handletextpad=0.4,
            labelspacing=0.25,
            borderaxespad=0.15,
        )

    fig.subplots_adjust(
        left=0.075,
        right=0.987,
        bottom=0.083,
        top=0.915,
        wspace=0.24,
        hspace=0.35,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Fig3_caurc_selective_tail_correction_clean"
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

    mode_text = (
        "paired hierarchical sequence-to-event bootstrap resampling"
        if bootstrap_mode == "hierarchical_sequence_event"
        else "paired event-level bootstrap resampling"
    )

    caption = (
        "Fig. 3 | Underprediction-risk correction selectively reduces high-motion "
        "errors while preserving catalog-wide accuracy. "
        "a, Paired change in catalog-wide event-macro mean absolute error (MAE), "
        "defined as CA-URC minus the frozen Cross-Attention Base; near-zero values "
        "indicate preserved overall accuracy. "
        "b, High-motion-tail MAE for the two exactly paired models. "
        "c, Severe underprediction rate in the high-motion tail, "
        "U_0.5 = Pr(prediction - observation <= -0.5). "
        "d, Tail factor-of-two accuracy. "
        "e,f, Event-balanced empirical cumulative distributions of PGA and PGV "
        "residuals within the high-motion tail. The dotted vertical line denotes "
        "the severe-underprediction threshold of -0.5 log10 units and the dashed "
        "line denotes zero residual. All confidence intervals were obtained using "
        f"{mode_text}. The locked test set comprised {int(overall['n_events'])} "
        f"earthquakes, {int(overall['n_sequence_groups'])} seismic-sequence groups "
        f"and {int(overall['n_target_rows']):,} target predictions. The PGA tail "
        f"contained {int(pga_tail['n_target_rows']):,} predictions from "
        f"{int(pga_tail['n_events'])} earthquakes, and the PGV tail contained "
        f"{int(pgv_tail['n_target_rows']):,} predictions from "
        f"{int(pgv_tail['n_events'])} earthquakes."
    )
    out_path.write_text(caption, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv",
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
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--out-dir", default="figures/nc_section_2_2_clean")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_predictions(
        Path(args.predictions),
        base_prefix=args.base_prefix,
        final_prefix=args.final_prefix,
    )

    frame, bootstrap_mode = attach_sequence_groups(
        frame,
        manifest_path=Path(args.manifest) if str(args.manifest).strip() else None,
        sequence_column=args.sequence_column,
        split_column=args.split_column,
        test_label=args.test_label,
    )

    summary = summarize_paired_metrics(
        frame,
        base_prefix=args.base_prefix,
        final_prefix=args.final_prefix,
        bootstrap_mode=bootstrap_mode,
        repetitions=args.bootstrap_repetitions,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )

    summary.to_csv(out_dir / "Fig3_primary_summary.csv", index=False)

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

    config = vars(args).copy()
    config["bootstrap_mode_used"] = bootstrap_mode
    (out_dir / "run_configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Figure written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()