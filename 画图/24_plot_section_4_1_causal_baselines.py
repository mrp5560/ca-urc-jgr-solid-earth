#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TGRS-style three-panel figure for Section 4.1.
Reads metrics_summary.csv directly.

Panels:
(a) log10-domain MAE
(b) log10-domain Bias
(c) Factor-of-two accuracy

Example (PowerShell):
python 24_plot_section_4_1_causal_baselines.py `
  --csv "metrics_summary.csv" `
  --out-dir "figures\\section_4_1" `
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

METHOD_ORDER = [
    "train_median",
    "median_observed",
    "nearest_observed",
    "idw_observed",
    "model",
]

METHOD_LABELS = {
    "train_median": "Training\nmedian",
    "median_observed": "Observed\nmedian",
    "nearest_observed": "Nearest\nobserved",
    "idw_observed": "IDW\nobserved",
    "model": "Causal-SeisField\nbase",
}


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "method",
        "quantity",
        "mae_log10_event_macro",
        "bias_log10_prediction_minus_true",
        "factor2_accuracy",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.loc[
        df["method"].isin(METHOD_ORDER)
        & df["quantity"].isin(["pga", "pgv"])
    ].copy()

    expected = {
        (m, q)
        for m in METHOD_ORDER
        for q in ("pga", "pgv")
    }
    actual = set(zip(df["method"], df["quantity"]))
    missing_pairs = expected.difference(actual)
    if missing_pairs:
        raise ValueError(f"Missing method/quantity rows: {sorted(missing_pairs)}")

    return df


def pivot_metric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.pivot(index="method", columns="quantity", values=column)
        .reindex(METHOD_ORDER)
    )


def emphasize_model(bars) -> None:
    model_index = METHOD_ORDER.index("model")
    for i, bar in enumerate(bars):
        if i == model_index:
            bar.set_hatch("///")
            bar.set_linewidth(1.4)
            bar.set_edgecolor("black")
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.60)
            bar.set_linewidth(0.6)


def add_labels(ax, bars, formatter, positive_offset=0.018) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        v = bar.get_height()
        if v >= 0:
            y = v + positive_offset * span
            va = "bottom"
        else:
            y = v - positive_offset * span
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            formatter(v),
            ha="center",
            va=va,
            fontsize=7.8,
        )


def grouped_panel(
    ax,
    table,
    ylabel,
    title,
    formatter,
    zero_line=False,
    percent=False,
):
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.34

    bars_pga = ax.bar(
        x - width / 2,
        table["pga"].to_numpy(float),
        width,
        label="PGA",
    )
    bars_pgv = ax.bar(
        x + width / 2,
        table["pgv"].to_numpy(float),
        width,
        label="PGV",
    )

    emphasize_model(bars_pga)
    emphasize_model(bars_pgv)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.set_axisbelow(True)

    if zero_line:
        ax.axhline(0.0, linewidth=0.9)

    if percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    add_labels(ax, bars_pga, formatter)
    add_labels(ax, bars_pgv, formatter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="metrics_summary.csv")
    parser.add_argument("--out-dir", default='figures/section_4_1')
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(csv_path)
    mae = pivot_metric(df, "mae_log10_event_macro")
    bias = pivot_metric(df, "bias_log10_prediction_minus_true")
    factor2 = pivot_metric(df, "factor2_accuracy")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.3,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))

    # (a) MAE
    grouped_panel(
        axes[0],
        mae,
        ylabel=r"MAE ($\log_{10}$ units)",
        title="(a) Overall prediction error",
        formatter=lambda v: f"{v:.3f}",
    )
    axes[0].set_ylim(0, float(mae.max().max()) * 1.28)

    nearest = mae.loc["nearest_observed"]
    model = mae.loc["model"]
    pga_improve = (1 - model["pga"] / nearest["pga"]) * 100
    pgv_improve = (1 - model["pgv"] / nearest["pgv"]) * 100
    axes[0].text(
        0.98,
        0.97,
        "Reduction vs. nearest-observed\n"
        f"PGA: {pga_improve:.1f}%   PGV: {pgv_improve:.1f}%",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="0.65",
            alpha=0.92,
        ),
    )

    # (b) Bias
    max_abs_bias = float(np.abs(bias.to_numpy(float)).max())
    grouped_panel(
        axes[1],
        bias,
        ylabel=r"Bias ($\log_{10}$ units)",
        title="(b) Prediction bias",
        formatter=lambda v: f"{v:+.3f}",
        zero_line=True,
    )
    axes[1].set_ylim(-max_abs_bias * 1.30, max_abs_bias * 1.30)

    # (c) Factor-of-two accuracy
    grouped_panel(
        axes[2],
        factor2,
        ylabel="Factor-of-two accuracy",
        title="(c) Predictions within a factor of two",
        formatter=lambda v: f"{100*v:.1f}%",
        percent=True,
    )
    axes[2].set_ylim(0, min(1.0, float(factor2.max().max()) * 1.34))

    handles, labels = axes[2].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )

    fig.text(
        0.5,
        0.012,
        "Hatched bars denote the learned Causal-SeisField base model; "
        "all comparison methods use only causally available observations.",
        ha="center",
        va="bottom",
        fontsize=8.1,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=0.22,
        top=0.83,
        wspace=0.34,
    )

    png = out_dir / "Fig_4_1_causal_baseline_comparison.png"
    svg = out_dir / "Fig_4_1_causal_baseline_comparison.svg"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(png.resolve())
    print(svg.resolve())


if __name__ == "__main__":
    main()
