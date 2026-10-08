#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25_plot_section_4_3_tail_diagnostics_mean_base.py

TGRS-style Fig. for Section 4.3:
Overall accuracy masks systematic underprediction of high-motion targets.

This version is tailored to paired_target_predictions.csv with columns:
    event_id
    repeat
    true_log10_pga
    true_log10_pgv
    is_tail_pga
    is_tail_pgv
    mean_base_log10_pga
    mean_base_log10_pgv

Residual definition:
    residual = prediction - observation

Under05:
    residual <= -0.5

The tail labels stored in paired_target_predictions.csv are used directly.
No tail thresholds are recomputed in this script.

Outputs:
    Fig_4_3_tail_failure.png
    Fig_4_3_tail_failure.svg
    Fig_4_3_tail_failure.pdf
    Fig_4_3_metrics_check.csv

Example:
python 25_plot_section_4_3_tail_diagnostics_mean_base.py ^
  --predictions-csv "runs\\tail_checkpoint_repeated_test\\paired_target_predictions.csv" ^
  --out-dir "figures\\section_4_3" ^
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm


REQUIRED_COLUMNS = [
    "event_id",
    "repeat",
    "true_log10_pga",
    "true_log10_pgv",
    "is_tail_pga",
    "is_tail_pgv",
    "mean_base_log10_pga",
    "mean_base_log10_pgv",
]

# Validation-set reference values from the repeated evaluation.
# Used only as a consistency check; not used to draw the figure.
EXPECTED = {
    ("pga", "overall", "mae"): 0.336919,
    ("pga", "overall", "bias"): -0.024580,
    ("pga", "overall", "under05"): 0.132464,
    ("pga", "tail", "mae"): 0.739625,
    ("pga", "tail", "bias"): -0.717114,
    ("pga", "tail", "under05"): 0.697706,
    ("pgv", "overall", "mae"): 0.299954,
    ("pgv", "overall", "bias"): -0.027121,
    ("pgv", "overall", "under05"): 0.111425,
    ("pgv", "tail", "mae"): 0.798959,
    ("pgv", "tail", "bias"): -0.765453,
    ("pgv", "tail", "under05"): 0.747549,
}


def parse_bool(series: pd.Series) -> pd.Series:
    """Robustly parse TRUE/FALSE, true/false, 1/0, or bool values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "nan": False,
        "none": False,
        "": False,
    }
    bad = ~s.isin(mapping.keys())
    if bad.any():
        values = sorted(s.loc[bad].unique().tolist())[:10]
        raise ValueError(f"Unrecognized boolean values: {values}")
    return s.map(mapping).astype(bool)


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "paired_target_predictions.csv is missing required columns:\n"
            + "\n".join(missing)
        )

    keep = REQUIRED_COLUMNS.copy()
    optional = [
        "sequence_group",
        "magnitude",
        "target_station_id",
        "target_station_index",
        "target_p_offset_sec",
        "target_triggered_by_snapshot",
    ]
    keep += [c for c in optional if c in df.columns]

    df = df[keep].copy()

    numeric_cols = [
        "true_log10_pga",
        "true_log10_pgv",
        "mean_base_log10_pga",
        "mean_base_log10_pgv",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_tail_pga"] = parse_bool(df["is_tail_pga"])
    df["is_tail_pgv"] = parse_bool(df["is_tail_pgv"])

    df = df.dropna(subset=numeric_cols).copy()

    df["residual_pga"] = (
        df["mean_base_log10_pga"] - df["true_log10_pga"]
    )
    df["residual_pgv"] = (
        df["mean_base_log10_pgv"] - df["true_log10_pgv"]
    )

    return df


def metric_value(residual: np.ndarray, metric: str) -> float:
    residual = np.asarray(residual, dtype=float)
    if residual.size == 0:
        return np.nan

    if metric == "mae":
        return float(np.mean(np.abs(residual)))
    if metric == "bias":
        return float(np.mean(residual))
    if metric == "under05":
        return float(np.mean(residual <= -0.5))
    raise ValueError(metric)


def compute_event_repeat_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the repeated-evaluation logic:
    1. select population within each quantity;
    2. compute metric for each (event_id, repeat);
    3. average those event-repeat metrics.
    """
    rows = []

    for q in ("pga", "pgv"):
        residual_col = f"residual_{q}"
        tail_col = f"is_tail_{q}"

        for population in ("overall", "tail"):
            if population == "overall":
                subset = df.copy()
            else:
                subset = df.loc[df[tail_col]].copy()

            for metric in ("mae", "bias", "under05"):
                unit_values = []

                for (_, _), group in subset.groupby(
                    ["event_id", "repeat"],
                    sort=False,
                ):
                    residual = group[residual_col].to_numpy(dtype=float)
                    if residual.size:
                        unit_values.append(metric_value(residual, metric))

                value = (
                    float(np.mean(unit_values))
                    if unit_values
                    else np.nan
                )

                rows.append(
                    {
                        "quantity": q,
                        "population": population,
                        "metric": metric,
                        "event_repeat_macro": value,
                        "n_event_repeats": len(unit_values),
                        "n_events": int(subset["event_id"].nunique()),
                        "n_target_rows": int(len(subset)),
                    }
                )

    return pd.DataFrame(rows)


def get_metric(
    metrics: pd.DataFrame,
    quantity: str,
    population: str,
    metric: str,
) -> float:
    row = metrics.loc[
        metrics["quantity"].eq(quantity)
        & metrics["population"].eq(population)
        & metrics["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Metric not unique: {quantity}/{population}/{metric}"
        )
    return float(row["event_repeat_macro"].iloc[0])


def check_expected(metrics: pd.DataFrame, tolerance: float = 5e-4) -> None:
    print("\n=== Consistency check against saved validation summary ===")
    okay = True
    for key, expected in EXPECTED.items():
        q, pop, metric = key
        actual = get_metric(metrics, q, pop, metric)
        diff = actual - expected
        status = "OK" if abs(diff) <= tolerance else "CHECK"
        if status != "OK":
            okay = False
        print(
            f"{q.upper():3s} {pop:7s} {metric:7s}: "
            f"actual={actual:.6f}  expected={expected:.6f}  "
            f"diff={diff:+.6f}  [{status}]"
        )

    if okay:
        print("All key metrics match the saved repeated-validation results.")
    else:
        print(
            "WARNING: at least one metric differs from the saved summary. "
            "Check whether the CSV is the same paired_target_predictions.csv "
            "used by the repeated validation run."
        )


def binned_residual_stats(
    true: np.ndarray,
    residual: np.ndarray,
    bins: int = 16,
) -> pd.DataFrame:
    d = pd.DataFrame(
        {
            "true": np.asarray(true, dtype=float),
            "residual": np.asarray(residual, dtype=float),
        }
    ).dropna()

    # Equal-width bins are preferable here because the scientific question
    # is how residual changes as observed shaking increases.
    edges = np.linspace(
        float(d["true"].quantile(0.005)),
        float(d["true"].quantile(0.995)),
        bins + 1,
    )

    d = d.loc[
        d["true"].between(edges[0], edges[-1], inclusive="both")
    ].copy()
    d["bin"] = pd.cut(
        d["true"],
        bins=edges,
        include_lowest=True,
    )

    out = (
        d.groupby("bin", observed=True)
        .agg(
            x=("true", "median"),
            median=("residual", "median"),
            q25=("residual", lambda s: np.quantile(s, 0.25)),
            q75=("residual", lambda s: np.quantile(s, 0.75)),
            count=("residual", "size"),
        )
        .reset_index(drop=True)
    )

    # Avoid visually unstable bins.
    out = out.loc[out["count"] >= 30].copy()
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.55,
        alpha=0.28,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_observed_predicted(
    ax: plt.Axes,
    df: pd.DataFrame,
    q: str,
    threshold: float,
    panel: str,
    metrics: pd.DataFrame,
):
    true_col = f"true_log10_{q}"
    pred_col = f"mean_base_log10_{q}"

    x = df[true_col].to_numpy(dtype=float)
    y = df[pred_col].to_numpy(dtype=float)

    low = min(
        np.quantile(x, 0.003),
        np.quantile(y, 0.003),
    )
    high = max(
        np.quantile(x, 0.997),
        np.quantile(y, 0.997),
    )
    pad = 0.05 * (high - low)
    low -= pad
    high += pad

    # High-motion region.
    ax.axvspan(
        threshold,
        high,
        facecolor="0.92",
        edgecolor="none",
        zorder=0,
    )

    hb = ax.hexbin(
        x,
        y,
        gridsize=52,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
        zorder=2,
    )

    ax.plot(
        [low, high],
        [low, high],
        "--",
        linewidth=1.15,
        color="black",
        zorder=3,
    )
    ax.axvline(
        threshold,
        linestyle=":",
        linewidth=1.2,
        color="black",
        zorder=3,
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")

    quantity_label = q.upper()
    ax.set_xlabel(
        rf"Observed $\log_{{10}}({quantity_label})$"
    )
    ax.set_ylabel(
        rf"Predicted $\log_{{10}}({quantity_label})$"
    )
    ax.set_title(
        f"({panel}) {quantity_label}: observed vs. predicted",
        loc="left",
        fontweight="bold",
    )

    overall_mae = get_metric(metrics, q, "overall", "mae")
    tail_mae = get_metric(metrics, q, "tail", "mae")
    tail_bias = get_metric(metrics, q, "tail", "bias")

    ax.text(
        0.04,
        0.96,
        (
            f"Overall MAE = {overall_mae:.3f}\n"
            f"Tail MAE = {tail_mae:.3f}\n"
            f"Tail bias = {tail_bias:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.70",
            "alpha": 0.94,
        },
        zorder=5,
    )

    style_axis(ax)
    return hb


def plot_residual(
    ax: plt.Axes,
    df: pd.DataFrame,
    q: str,
    threshold: float,
    panel: str,
):
    true_col = f"true_log10_{q}"
    residual_col = f"residual_{q}"

    x = df[true_col].to_numpy(dtype=float)
    residual = df[residual_col].to_numpy(dtype=float)

    xmin = float(np.quantile(x, 0.003))
    xmax = float(np.quantile(x, 0.997))
    xpad = 0.04 * (xmax - xmin)
    xmin -= xpad
    xmax += xpad

    ax.axvspan(
        threshold,
        xmax,
        facecolor="0.92",
        edgecolor="none",
        zorder=0,
    )

    # Use a deterministic sample only for the faint cloud.
    if len(df) > 10000:
        sample = df.sample(
            n=10000,
            random_state=2026,
            replace=False,
        )
    else:
        sample = df

    ax.scatter(
        sample[true_col],
        sample[residual_col],
        s=5,
        alpha=0.06,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    stats = binned_residual_stats(
        x,
        residual,
        bins=16,
    )

    if not stats.empty:
        ax.fill_between(
            stats["x"].to_numpy(),
            stats["q25"].to_numpy(),
            stats["q75"].to_numpy(),
            alpha=0.22,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            stats["x"],
            stats["median"],
            linewidth=2.1,
            marker="o",
            markersize=3.5,
            zorder=3,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.05,
        color="black",
    )
    ax.axhline(
        -0.5,
        linestyle=":",
        linewidth=1.05,
        color="black",
    )
    ax.axvline(
        threshold,
        linestyle=":",
        linewidth=1.2,
        color="black",
    )

    ax.set_xlim(xmin, xmax)
    quantity_label = q.upper()
    ax.set_xlabel(
        rf"Observed $\log_{{10}}({quantity_label})$"
    )
    ax.set_ylabel(
        r"Residual: prediction $-$ observation"
    )
    ax.set_title(
        f"({panel}) {quantity_label}: error becomes increasingly negative",
        loc="left",
        fontweight="bold",
    )

    ax.text(
        0.04,
        0.05,
        "Line: binned median\nBand: interquartile range",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.93,
        },
    )

    style_axis(ax)


def plot_summary(
    ax: plt.Axes,
    metrics: pd.DataFrame,
):
    """
    Three metrics are normalized differently, so use a compact dot-chart:
    x position is categorical; y position is the metric value.
    Overall and tail are connected within each quantity.
    """
    metrics_to_plot = [
        ("mae", "MAE", 1.0),
        ("bias", "Bias", 1.0),
        ("under05", "Under05", 100.0),
    ]

    # We'll use three inset axes to keep units honest.
    ax.set_axis_off()

    positions = [
        [0.00, 0.68, 1.00, 0.29],
        [0.00, 0.35, 1.00, 0.29],
        [0.00, 0.02, 1.00, 0.29],
    ]

    labels = ["(e1) MAE", "(e2) Bias", "(e3) Under05"]

    for pos, (metric, metric_label, scale), title in zip(
        positions,
        metrics_to_plot,
        labels,
    ):
        sub = ax.inset_axes(pos)

        x = np.array([0.0, 1.0])
        quantities = ["pga", "pgv"]

        overall = np.array(
            [
                get_metric(metrics, q, "overall", metric) * scale
                for q in quantities
            ]
        )
        tail = np.array(
            [
                get_metric(metrics, q, "tail", metric) * scale
                for q in quantities
            ]
        )

        width = 0.32
        bars_o = sub.bar(
            x - width / 2,
            overall,
            width,
            label="Overall",
            alpha=0.72,
        )
        bars_t = sub.bar(
            x + width / 2,
            tail,
            width,
            label="High-motion tail",
            alpha=0.95,
            hatch="///",
            edgecolor="black",
            linewidth=0.8,
        )

        sub.set_xticks(x)
        sub.set_xticklabels(["PGA", "PGV"])
        sub.set_title(
            title,
            loc="left",
            fontweight="bold",
            fontsize=9.5,
        )

        if metric == "mae":
            sub.set_ylabel(r"$\log_{10}$ units")
            ymax = max(tail.max(), overall.max()) * 1.25
            sub.set_ylim(0, ymax)

        elif metric == "bias":
            ymin = min(tail.min(), overall.min()) * 1.25
            ymax = max(
                0.08,
                max(tail.max(), overall.max()) + 0.08,
            )
            sub.set_ylim(ymin, ymax)
            sub.axhline(
                0,
                linestyle="--",
                linewidth=0.9,
                color="black",
            )
            sub.set_ylabel(r"$\log_{10}$ units")

        else:
            sub.set_ylabel("%")
            sub.set_ylim(
                0,
                max(tail.max(), overall.max()) * 1.22,
            )

        for bars in (bars_o, bars_t):
            for b in bars:
                val = b.get_height()

                if metric == "bias":
                    span = sub.get_ylim()[1] - sub.get_ylim()[0]
                    offset = 0.035 * span
                    y = val + offset if val >= 0 else val - offset
                    va = "bottom" if val >= 0 else "top"
                    txt = f"{val:+.3f}"
                elif metric == "under05":
                    offset = 0.025 * sub.get_ylim()[1]
                    y = val + offset
                    va = "bottom"
                    txt = f"{val:.1f}%"
                else:
                    offset = 0.025 * sub.get_ylim()[1]
                    y = val + offset
                    va = "bottom"
                    txt = f"{val:.3f}"

                sub.text(
                    b.get_x() + b.get_width() / 2,
                    y,
                    txt,
                    ha="center",
                    va=va,
                    fontsize=7.7,
                )

        if metric == "mae":
            sub.legend(
                frameon=False,
                ncol=2,
                loc="upper left",
                fontsize=7.8,
            )

        style_axis(sub)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-csv",
        default="./runs/tail_checkpoint_repeated_test/paired_target_predictions.csv",
        help="Path to paired_target_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="figures/section_4_3",
    )
    parser.add_argument(
        "--tail-threshold-pga",
        type=float,
        default=-2.0112,
        help="Only for drawing the vertical reference line.",
    )
    parser.add_argument(
        "--tail-threshold-pgv",
        type=float,
        default=-3.3072,
        help="Only for drawing the vertical reference line.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    args = parser.parse_args()

    path = Path(args.predictions_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(path)
    metrics = compute_event_repeat_macro(df)

    print("Rows   :", len(df))
    print("Events :", df["event_id"].nunique())
    print(
        "Repeats:",
        df[["event_id", "repeat"]]
        .drop_duplicates()
        .shape[0],
    )
    print("\n=== Recomputed metrics ===")
    print(metrics.to_string(index=False))

    check_expected(metrics)

    metrics_path = out_dir / "Fig_4_3_metrics_check.csv"
    metrics.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
        }
    )

    fig = plt.figure(
        figsize=(12.6, 7.2),
    )
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.92],
        height_ratios=[1.0, 1.0],
        left=0.065,
        right=0.985,
        bottom=0.11,
        top=0.94,
        wspace=0.30,
        hspace=0.31,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[:, 2])

    hb_a = plot_observed_predicted(
        ax_a,
        df,
        "pga",
        args.tail_threshold_pga,
        "a",
        metrics,
    )
    hb_b = plot_observed_predicted(
        ax_b,
        df,
        "pgv",
        args.tail_threshold_pgv,
        "b",
        metrics,
    )

    plot_residual(
        ax_c,
        df,
        "pga",
        args.tail_threshold_pga,
        "c",
    )
    plot_residual(
        ax_d,
        df,
        "pgv",
        args.tail_threshold_pgv,
        "d",
    )

    plot_summary(
        ax_e,
        metrics,
    )

    # One compact density colorbar shared by the two hexbin panels.
    cbar_ax = fig.add_axes(
        [0.385, 0.965, 0.22, 0.015]
    )
    cb = fig.colorbar(
        hb_a,
        cax=cbar_ax,
        orientation="horizontal",
    )
    cb.set_label(
        "Target density (log-scaled count)",
        fontsize=8,
    )
    cb.ax.tick_params(labelsize=7)

    fig.text(
        0.5,
        0.025,
        (
            "Shaded regions denote training-defined high-motion tails. "
            "Residual = prediction - observation; Under05 denotes residual <= -0.5."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
    )

    png_path = out_dir / "Fig_4_3_tail_failure.png"
    svg_path = out_dir / "Fig_4_3_tail_failure.svg"
    pdf_path = out_dir / "Fig_4_3_tail_failure.pdf"

    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved:")
    print(png_path.resolve())
    print(svg_path.resolve())
    print(pdf_path.resolve())
    print(metrics_path.resolve())


if __name__ == "__main__":
    main()
