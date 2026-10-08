#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27_plot_section_4_5_tail_risk_linear_leakage.py

Section 4.5:
Tail-risk identification enables targeted high-motion correction,
but linear gating leaks corrections into non-tail predictions.

Input
-----
runs/tail_risk_gated_dual_head_t0_5s_k5/
    validation_predictions_overall.csv

The validation file is expected to contain, for PGA and PGV:
    true / true_log10
    base / base_log10
    final / final_log10
    probability / tail_probability
    tail / is_tail

The trained linear-gated model is interpreted as:
    y_linear = y_base + p_tail * Delta_tail

Therefore:
    linear_correction = y_linear - y_base

Figure
------
(a) Precision-recall curves for future high-motion-tail identification.
(b) PGA: linear correction versus tail-risk probability.
(c) PGV: linear correction versus tail-risk probability.
(d) Change in MAE caused by linear gating:
        Delta MAE = MAE_linear - MAE_base
    for Overall, Non-tail, and Tail populations.
    Negative values mean improvement.

The script recomputes all metrics directly from the target-level
validation predictions and performs event-level paired bootstrap
for panel (d).

Outputs
-------
Fig_4_5_tail_risk_linear_leakage.png
Fig_4_5_tail_risk_linear_leakage.svg
Fig_4_5_tail_risk_linear_leakage.pdf
Fig_4_5_summary_metrics.csv
Fig_4_5_bootstrap_delta_mae.csv
Fig_4_5_pr_metrics.csv

Example
-------
python 27_plot_section_4_5_tail_risk_linear_leakage.py ^
  --predictions "runs\\tail_risk_gated_dual_head_t0_5s_k5\\validation_predictions_overall.csv" ^
  --out-dir "figures\\section_4_5" ^
  --bootstrap-repetitions 2000 ^
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)


# ---------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------

def resolve_column(
    frame: pd.DataFrame,
    candidates: list[str],
    logical_name: str,
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise KeyError(
        f"Could not resolve column for {logical_name}. "
        f"Tried: {candidates}\n"
        f"Available columns:\n{list(frame.columns)}"
    )


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    columns = {
        "event_id": resolve_column(
            frame,
            ["event_id"],
            "event_id",
        ),
        "repeat": resolve_column(
            frame,
            ["repeat", "repeat_index"],
            "repeat",
        ),
    }

    for quantity in ("pga", "pgv"):
        columns[f"true_{quantity}"] = resolve_column(
            frame,
            [
                f"true_{quantity}",
                f"true_log10_{quantity}",
            ],
            f"true {quantity}",
        )

        columns[f"base_{quantity}"] = resolve_column(
            frame,
            [
                f"base_{quantity}",
                f"base_log10_{quantity}",
            ],
            f"base {quantity}",
        )

        columns[f"final_{quantity}"] = resolve_column(
            frame,
            [
                f"final_{quantity}",
                f"final_log10_{quantity}",
            ],
            f"linear-gated/final {quantity}",
        )

        columns[f"probability_{quantity}"] = resolve_column(
            frame,
            [
                f"probability_{quantity}",
                f"tail_probability_{quantity}",
            ],
            f"tail probability {quantity}",
        )

        columns[f"tail_{quantity}"] = resolve_column(
            frame,
            [
                f"tail_{quantity}",
                f"is_tail_{quantity}",
            ],
            f"tail label {quantity}",
        )

    return columns


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return text.isin(
        {"true", "1", "yes", "y", "t"}
    )


def load_predictions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        dtype={"event_id": str},
    )

    columns = resolve_columns(raw)

    frame = pd.DataFrame(
        {
            "event_id": (
                raw[columns["event_id"]]
                .astype(str)
            ),
            "repeat": pd.to_numeric(
                raw[columns["repeat"]],
                errors="coerce",
            ).fillna(-1).astype(int),
        }
    )

    for quantity in ("pga", "pgv"):
        for prefix in (
            "true",
            "base",
            "final",
            "probability",
        ):
            frame[
                f"{prefix}_{quantity}"
            ] = pd.to_numeric(
                raw[
                    columns[
                        f"{prefix}_{quantity}"
                    ]
                ],
                errors="coerce",
            )

        frame[
            f"tail_{quantity}"
        ] = as_bool(
            raw[
                columns[
                    f"tail_{quantity}"
                ]
            ]
        )

    required = [
        f"{prefix}_{quantity}"
        for quantity in ("pga", "pgv")
        for prefix in (
            "true",
            "base",
            "final",
            "probability",
        )
    ]

    bad = frame[required].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows contain "
            "missing required numeric values."
        )

    for quantity in ("pga", "pgv"):
        frame[
            f"probability_{quantity}"
        ] = np.clip(
            frame[
                f"probability_{quantity}"
            ].to_numpy(dtype=float),
            0.0,
            1.0,
        )

        correction = (
            frame[
                f"final_{quantity}"
            ].to_numpy(dtype=float)
            - frame[
                f"base_{quantity}"
            ].to_numpy(dtype=float)
        )

        # The trained correction branch is non-negative.
        # Preserve tiny numerical negatives only as zero.
        frame[
            f"correction_{quantity}"
        ] = np.maximum(
            correction,
            0.0,
        )

        frame[
            f"base_abs_error_{quantity}"
        ] = np.abs(
            frame[
                f"base_{quantity}"
            ].to_numpy(dtype=float)
            - frame[
                f"true_{quantity}"
            ].to_numpy(dtype=float)
        )

        frame[
            f"linear_abs_error_{quantity}"
        ] = np.abs(
            frame[
                f"final_{quantity}"
            ].to_numpy(dtype=float)
            - frame[
                f"true_{quantity}"
            ].to_numpy(dtype=float)
        )

    return frame


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def event_repeat_macro_mae(
    frame: pd.DataFrame,
    error_column: str,
    mask: np.ndarray | pd.Series | None = None,
) -> float:
    selected = frame

    if mask is not None:
        selected = frame.loc[
            np.asarray(mask, dtype=bool)
        ]

    if len(selected) == 0:
        return float("nan")

    unit = (
        selected.groupby(
            ["event_id", "repeat"],
            sort=False,
        )[error_column]
        .mean()
    )

    if len(unit) == 0:
        return float("nan")

    return float(unit.mean())


def compute_summary_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quantity in ("pga", "pgv"):
        tail = frame[
            f"tail_{quantity}"
        ].to_numpy(dtype=bool)

        populations = {
            "overall": np.ones(
                len(frame),
                dtype=bool,
            ),
            "non_tail": ~tail,
            "tail": tail,
        }

        for population, mask in populations.items():
            base_mae = event_repeat_macro_mae(
                frame,
                f"base_abs_error_{quantity}",
                mask,
            )
            linear_mae = event_repeat_macro_mae(
                frame,
                f"linear_abs_error_{quantity}",
                mask,
            )

            rows.append(
                {
                    "quantity": quantity,
                    "population": population,
                    "base_mae": base_mae,
                    "linear_mae": linear_mae,
                    "delta_mae_linear_minus_base": (
                        linear_mae - base_mae
                    ),
                    "n_target_rows": int(
                        np.sum(mask)
                    ),
                    "n_events": int(
                        frame.loc[
                            mask,
                            "event_id",
                        ].nunique()
                    ),
                }
            )

    return pd.DataFrame(rows)


def compute_pr_metrics(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, tuple[np.ndarray, np.ndarray]]
]:
    metric_rows = []
    curves = {}

    for quantity in ("pga", "pgv"):
        truth = frame[
            f"tail_{quantity}"
        ].to_numpy(dtype=bool)

        probability = frame[
            f"probability_{quantity}"
        ].to_numpy(dtype=float)

        prevalence = float(
            np.mean(truth)
        )

        auprc = float(
            average_precision_score(
                truth.astype(int),
                probability,
            )
        )

        precision, recall, _ = (
            precision_recall_curve(
                truth.astype(int),
                probability,
            )
        )

        curves[quantity] = (
            recall,
            precision,
        )

        metric_rows.append(
            {
                "quantity": quantity,
                "n_rows": int(len(frame)),
                "n_positive": int(
                    np.sum(truth)
                ),
                "prevalence": prevalence,
                "auprc": auprc,
                "auprc_over_random": (
                    auprc / prevalence
                    if prevalence > 0
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(metric_rows),
        curves,
    )


# ---------------------------------------------------------------------
# Event-level paired bootstrap
# ---------------------------------------------------------------------

def build_event_delta_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quantity in ("pga", "pgv"):
        tail = frame[
            f"tail_{quantity}"
        ].to_numpy(dtype=bool)

        populations = {
            "overall": np.ones(
                len(frame),
                dtype=bool,
            ),
            "non_tail": ~tail,
            "tail": tail,
        }

        for population, mask in populations.items():
            selected = frame.loc[
                mask,
                [
                    "event_id",
                    "repeat",
                    f"base_abs_error_{quantity}",
                    f"linear_abs_error_{quantity}",
                ],
            ].copy()

            if len(selected) == 0:
                continue

            selected[
                "delta"
            ] = (
                selected[
                    f"linear_abs_error_{quantity}"
                ]
                - selected[
                    f"base_abs_error_{quantity}"
                ]
            )

            unit = (
                selected.groupby(
                    ["event_id", "repeat"],
                    sort=False,
                )["delta"]
                .mean()
                .reset_index()
            )

            event = (
                unit.groupby(
                    "event_id",
                    sort=False,
                )["delta"]
                .mean()
                .reset_index()
            )

            for row in event.itertuples(
                index=False
            ):
                rows.append(
                    {
                        "event_id": str(
                            row.event_id
                        ),
                        "quantity": quantity,
                        "population": population,
                        "event_delta_mae": float(
                            row.delta
                        ),
                    }
                )

    return pd.DataFrame(rows)


def paired_bootstrap(
    event_delta: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for quantity in ("pga", "pgv"):
        for population in (
            "overall",
            "non_tail",
            "tail",
        ):
            subset = event_delta.loc[
                event_delta[
                    "quantity"
                ].eq(quantity)
                & event_delta[
                    "population"
                ].eq(population)
            ]

            values = subset[
                "event_delta_mae"
            ].to_numpy(dtype=float)

            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:
                continue

            boot = np.empty(
                repetitions,
                dtype=float,
            )

            for index in range(
                repetitions
            ):
                sampled = rng.integers(
                    0,
                    len(values),
                    size=len(values),
                )

                boot[index] = float(
                    values[sampled].mean()
                )

            rows.append(
                {
                    "quantity": quantity,
                    "population": population,
                    "n_paired_events": int(
                        len(values)
                    ),
                    "mean_delta_mae": float(
                        values.mean()
                    ),
                    "ci95_low": float(
                        np.percentile(
                            boot,
                            2.5,
                        )
                    ),
                    "ci95_high": float(
                        np.percentile(
                            boot,
                            97.5,
                        )
                    ),
                    "probability_linear_favorable": float(
                        np.mean(
                            boot < 0.0
                        )
                    ),
                    "bootstrap_repetitions": int(
                        repetitions
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Correction-versus-probability trend
# ---------------------------------------------------------------------

def binned_correction(
    probability: np.ndarray,
    correction: np.ndarray,
    labels: np.ndarray,
    target_label: bool,
    bins: int = 10,
) -> pd.DataFrame:
    probability = np.asarray(
        probability,
        dtype=float,
    )
    correction = np.asarray(
        correction,
        dtype=float,
    )
    labels = np.asarray(
        labels,
        dtype=bool,
    )

    mask = (
        np.isfinite(probability)
        & np.isfinite(correction)
        & (labels == target_label)
    )

    p = probability[mask]
    c = correction[mask]

    if len(p) == 0:
        return pd.DataFrame(
            columns=[
                "x",
                "median",
                "q25",
                "q75",
                "count",
            ]
        )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    bin_id = np.digitize(
        p,
        edges[1:-1],
        right=False,
    )

    rows = []

    for index in range(bins):
        current = (
            bin_id == index
        )

        if np.sum(current) < 5:
            continue

        rows.append(
            {
                "x": float(
                    np.median(
                        p[current]
                    )
                ),
                "median": float(
                    np.median(
                        c[current]
                    )
                ),
                "q25": float(
                    np.quantile(
                        c[current],
                        0.25,
                    )
                ),
                "q75": float(
                    np.quantile(
                        c[current],
                        0.75,
                    )
                ),
                "count": int(
                    np.sum(current)
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
        }
    )


def style_axis(
    ax: plt.Axes,
) -> None:
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.55,
        alpha=0.28,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(
        False
    )
    ax.spines["right"].set_visible(
        False
    )


def draw_pr_panel(
    ax: plt.Axes,
    pr_metrics: pd.DataFrame,
    curves: dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ],
) -> None:
    colors = {
        "pga": "#2F5597",
        "pgv": "#C55A11",
    }

    for quantity in (
        "pga",
        "pgv",
    ):
        row = pr_metrics.loc[
            pr_metrics[
                "quantity"
            ].eq(quantity)
        ].iloc[0]

        recall, precision = (
            curves[quantity]
        )

        ax.plot(
            recall,
            precision,
            linewidth=1.8,
            color=colors[quantity],
            label=(
                f"{quantity.upper()} "
                f"(AUPRC={row['auprc']:.3f})"
            ),
        )

        # Random-classifier PR baseline = prevalence.
        ax.axhline(
            float(
                row["prevalence"]
            ),
            linestyle=":",
            linewidth=1.0,
            color=colors[quantity],
            alpha=0.70,
        )

    ax.set_xlim(
        0.0,
        1.0,
    )
    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        "Recall"
    )
    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "(a) Tail-risk discrimination",
        loc="left",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    pga = pr_metrics.loc[
        pr_metrics[
            "quantity"
        ].eq("pga")
    ].iloc[0]

    pgv = pr_metrics.loc[
        pr_metrics[
            "quantity"
        ].eq("pgv")
    ].iloc[0]

    ax.text(
        0.04,
        0.06,
        (
            "Random AUPRC = tail prevalence\n"
            f"PGA: {pga['prevalence']:.3f}  "
            f"({pga['auprc_over_random']:.1f}x random)\n"
            f"PGV: {pgv['prevalence']:.3f}  "
            f"({pgv['auprc_over_random']:.1f}x random)"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    style_axis(ax)


def draw_leakage_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    quantity: str,
    panel_letter: str,
    max_background_points: int,
) -> None:
    p = frame[
        f"probability_{quantity}"
    ].to_numpy(dtype=float)

    correction = frame[
        f"correction_{quantity}"
    ].to_numpy(dtype=float)

    tail = frame[
        f"tail_{quantity}"
    ].to_numpy(dtype=bool)

    # Non-tail cloud: deterministic subsampling for visual clarity.
    non_tail_index = np.flatnonzero(
        ~tail
    )

    if (
        len(non_tail_index)
        > max_background_points
    ):
        rng = np.random.default_rng(
            20260828
            + (
                0
                if quantity == "pga"
                else 1
            )
        )

        non_tail_index = rng.choice(
            non_tail_index,
            size=max_background_points,
            replace=False,
        )

    ax.scatter(
        p[non_tail_index],
        correction[
            non_tail_index
        ],
        s=5,
        alpha=0.055,
        edgecolors="none",
        rasterized=True,
        label="Non-tail targets",
    )

    tail_index = np.flatnonzero(
        tail
    )

    ax.scatter(
        p[tail_index],
        correction[
            tail_index
        ],
        s=10,
        alpha=0.22,
        facecolors="none",
        linewidths=0.45,
        label="High-motion tail",
        rasterized=True,
    )

    # Binned median/IQR trends.
    non_tail_stats = (
        binned_correction(
            p,
            correction,
            tail,
            False,
            bins=10,
        )
    )

    tail_stats = (
        binned_correction(
            p,
            correction,
            tail,
            True,
            bins=10,
        )
    )

    if not non_tail_stats.empty:
        ax.fill_between(
            non_tail_stats[
                "x"
            ].to_numpy(),
            non_tail_stats[
                "q25"
            ].to_numpy(),
            non_tail_stats[
                "q75"
            ].to_numpy(),
            alpha=0.10,
            linewidth=0,
        )

        ax.plot(
            non_tail_stats[
                "x"
            ],
            non_tail_stats[
                "median"
            ],
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label="Non-tail median",
        )

    if not tail_stats.empty:
        ax.fill_between(
            tail_stats[
                "x"
            ].to_numpy(),
            tail_stats[
                "q25"
            ].to_numpy(),
            tail_stats[
                "q75"
            ].to_numpy(),
            alpha=0.15,
            linewidth=0,
        )

        ax.plot(
            tail_stats[
                "x"
            ],
            tail_stats[
                "median"
            ],
            linewidth=1.8,
            marker="s",
            markersize=3.2,
            label="Tail median",
        )

    ax.axvline(
        0.5,
        linestyle=":",
        linewidth=1.0,
        color="black",
        alpha=0.75,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.9,
        color="black",
        alpha=0.7,
    )

    # Focus y-axis on central 99.5% while retaining the scientific trend.
    upper = float(
        np.quantile(
            correction,
            0.995,
        )
    )

    upper = max(
        upper,
        0.02,
    )

    ax.set_ylim(
        -0.01 * upper,
        upper * 1.08,
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        r"Tail-risk probability $p_{\mathrm{tail}}$"
    )

    ax.set_ylabel(
        r"Linear correction "
        r"($y_{\mathrm{linear}}-y_{\mathrm{base}}$)"
    )

    ax.set_title(
        f"({panel_letter}) {quantity.upper()}: "
        "low-confidence correction leakage",
        loc="left",
        fontweight="bold",
    )

    # Quantify mean leakage directly in the panel.
    non_tail_mean = float(
        np.mean(
            correction[~tail]
        )
    )

    tail_mean = float(
        np.mean(
            correction[tail]
        )
    )

    ax.text(
        0.04,
        0.96,
        (
            f"Mean correction\n"
            f"non-tail = {non_tail_mean:.3f}\n"
            f"tail = {tail_mean:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    if quantity == "pga":
        ax.legend(
            frameon=False,
            loc="upper center",
            fontsize=7.4,
            ncol=2,
        )

    style_axis(ax)


def draw_delta_mae_panel(
    ax: plt.Axes,
    bootstrap: pd.DataFrame,
) -> None:
    populations = [
        "overall",
        "non_tail",
        "tail",
    ]

    labels = [
        "Overall",
        "Non-tail",
        "Tail",
    ]

    x = np.arange(
        len(populations),
        dtype=float,
    )

    width = 0.34

    colors = {
        "pga": "#2F5597",
        "pgv": "#C55A11",
    }

    for offset, quantity in (
        (-width / 2, "pga"),
        (+width / 2, "pgv"),
    ):
        means = []
        lower = []
        upper = []

        for population in populations:
            row = bootstrap.loc[
                bootstrap[
                    "quantity"
                ].eq(quantity)
                & bootstrap[
                    "population"
                ].eq(population)
            ]

            if len(row) != 1:
                raise RuntimeError(
                    "Bootstrap result missing "
                    f"for {quantity}/{population}."
                )

            row = row.iloc[0]

            mean = float(
                row["mean_delta_mae"]
            )
            low = float(
                row["ci95_low"]
            )
            high = float(
                row["ci95_high"]
            )

            means.append(
                mean
            )
            lower.append(
                max(
                    0.0,
                    mean - low,
                )
            )
            upper.append(
                max(
                    0.0,
                    high - mean,
                )
            )

        means = np.asarray(
            means,
            dtype=float,
        )

        error = np.vstack(
            [
                np.asarray(
                    lower,
                    dtype=float,
                ),
                np.asarray(
                    upper,
                    dtype=float,
                ),
            ]
        )

        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=error,
            capsize=3.0,
            linewidth=0.7,
            label=quantity.upper(),
            color=colors[quantity],
            alpha=0.82,
        )

        for bar, mean in zip(
            bars,
            means,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                mean
                + (
                    0.004
                    if mean >= 0
                    else -0.004
                ),
                f"{mean:+.3f}",
                ha="center",
                va=(
                    "bottom"
                    if mean >= 0
                    else "top"
                ),
                fontsize=7.7,
            )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        r"$\Delta$MAE = "
        r"MAE$_{\mathrm{linear}}$ "
        r"$-$ MAE$_{\mathrm{base}}$"
    )

    ax.set_title(
        "(d) Linear correction improves tail "
        "but degrades non-tail accuracy",
        loc="left",
        fontweight="bold",
    )

    ax.text(
        0.03,
        0.05,
        (
            "Negative = improvement\n"
            "Positive = degradation"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
    )

    style_axis(ax)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "runs/"
            "tail_risk_gated_dual_head_t0_5s_k5/"
            "validation_predictions_overall.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "figures/"
            "section_4_5"
        ),
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )

    parser.add_argument(
        "--max-background-points",
        type=int,
        default=12000,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    prediction_path = Path(
        args.predictions
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{prediction_path.resolve()}"
        )

    if (
        args.bootstrap_repetitions
        < 100
    ):
        raise ValueError(
            "--bootstrap-repetitions "
            "must be >= 100."
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load and recompute
    # -------------------------------------------------------------
    frame = load_predictions(
        prediction_path
    )

    summary = (
        compute_summary_metrics(
            frame
        )
    )

    (
        pr_metrics,
        curves,
    ) = compute_pr_metrics(
        frame
    )

    event_delta = (
        build_event_delta_table(
            frame
        )
    )

    bootstrap = paired_bootstrap(
        event_delta,
        repetitions=(
            args.bootstrap_repetitions
        ),
        seed=args.seed,
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------
    print(
        "\n=== Validation file ==="
    )
    print(
        prediction_path.resolve()
    )

    print(
        f"Rows   : {len(frame)}"
    )
    print(
        "Events : "
        f"{frame['event_id'].nunique()}"
    )
    print(
        "Event-repeat units : "
        f"{frame[['event_id', 'repeat']].drop_duplicates().shape[0]}"
    )

    print(
        "\n=== Tail-risk classification ==="
    )
    print(
        pr_metrics.to_string(
            index=False
        )
    )

    print(
        "\n=== Base vs linear-gate MAE ==="
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== Paired event bootstrap: Delta MAE ==="
    )
    print(
        bootstrap.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Export exact plotted numbers
    # -------------------------------------------------------------
    summary.to_csv(
        out_dir
        / "Fig_4_5_summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pr_metrics.to_csv(
        out_dir
        / "Fig_4_5_pr_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap.to_csv(
        out_dir
        / "Fig_4_5_bootstrap_delta_mae.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------
    set_style()

    fig = plt.figure(
        figsize=(10.8, 7.8)
    )

    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.27,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(
        gs[0, 0]
    )

    ax_b = fig.add_subplot(
        gs[0, 1]
    )

    ax_c = fig.add_subplot(
        gs[1, 0]
    )

    ax_d = fig.add_subplot(
        gs[1, 1]
    )

    draw_pr_panel(
        ax_a,
        pr_metrics,
        curves,
    )

    draw_leakage_panel(
        ax_b,
        frame,
        "pga",
        "b",
        max_background_points=(
            args.max_background_points
        ),
    )

    draw_leakage_panel(
        ax_c,
        frame,
        "pgv",
        "c",
        max_background_points=(
            args.max_background_points
        ),
    )

    draw_delta_mae_panel(
        ax_d,
        bootstrap,
    )

    png = (
        out_dir
        / "Fig_4_5_tail_risk_linear_leakage.png"
    )

    svg = (
        out_dir
        / "Fig_4_5_tail_risk_linear_leakage.svg"
    )

    pdf = (
        out_dir
        / "Fig_4_5_tail_risk_linear_leakage.pdf"
    )

    fig.savefig(
        png,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved:"
    )
    print(
        png.resolve()
    )
    print(
        svg.resolve()
    )
    print(
        pdf.resolve()
    )


if __name__ == "__main__":
    main()
