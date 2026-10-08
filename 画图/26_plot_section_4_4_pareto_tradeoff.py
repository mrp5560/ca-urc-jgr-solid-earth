#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
26_plot_section_4_4_pareto_tradeoff.py

Section 4.4:
Single-head tail optimization produces an overall-tail Pareto trade-off.

Input files
-----------
1) model_metrics_summary.csv
   Expected columns:
       model
       quantity
       population
       metric
       event_repeat_macro

2) paired_bootstrap_deltas.csv
   Expected columns:
       candidate_model
       quantity
       population
       metric
       mean_delta_candidate_minus_reference
       ci95_low
       ci95_high
       probability_candidate_favorable

Figure
------
Two panels:
    (a) PGA overall MAE vs tail MAE
    (b) PGV overall MAE vs tail MAE

Interpretation:
    lower-left = better overall and better tail performance.

Mean Base is the reference.
Dashed lines through Mean Base divide the plane into four regions.
Error bars for candidate models are the 95% CIs of paired MAE differences
relative to Mean Base, translated into absolute MAE coordinates.

Example
-------
python 26_plot_section_4_4_pareto_tradeoff.py ^
  --metrics-csv "runs\\tail_checkpoint_repeated_test\\model_metrics_summary.csv" ^
  --bootstrap-csv "runs\\tail_checkpoint_repeated_test\\paired_bootstrap_deltas.csv" ^
  --out-dir "figures\\section_4_4" ^
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REFERENCE_MODEL = "mean_base"

MODEL_ORDER = [
    "mean_base",
    "mild_overall",
    "mild_compromise",
    "mild_tail",
]

MODEL_LABELS = {
    "mean_base": "Mean Base",
    "mild_overall": "Mild Overall",
    "mild_compromise": "Mild Compromise",
    "mild_tail": "Mild Tail",
}


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"{name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    require_columns(
        df,
        {
            "model",
            "quantity",
            "population",
            "metric",
            "event_repeat_macro",
        },
        "model_metrics_summary.csv",
    )

    df["model"] = df["model"].astype(str)
    df["quantity"] = df["quantity"].astype(str).str.lower()
    df["population"] = df["population"].astype(str).str.lower()
    df["metric"] = df["metric"].astype(str).str.lower()

    return df


def load_bootstrap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    require_columns(
        df,
        {
            "candidate_model",
            "quantity",
            "population",
            "metric",
            "mean_delta_candidate_minus_reference",
            "ci95_low",
            "ci95_high",
        },
        "paired_bootstrap_deltas.csv",
    )

    df["candidate_model"] = df[
        "candidate_model"
    ].astype(str)
    df["quantity"] = df[
        "quantity"
    ].astype(str).str.lower()
    df["population"] = df[
        "population"
    ].astype(str).str.lower()
    df["metric"] = df[
        "metric"
    ].astype(str).str.lower()

    return df


def metric_value(
    metrics: pd.DataFrame,
    model: str,
    quantity: str,
    population: str,
) -> float:
    row = metrics.loc[
        metrics["model"].eq(model)
        & metrics["quantity"].eq(quantity)
        & metrics["population"].eq(population)
        & metrics["metric"].eq("mae")
    ]

    if len(row) != 1:
        raise RuntimeError(
            "Expected exactly one metric row for "
            f"{model}/{quantity}/{population}/mae, "
            f"found {len(row)}."
        )

    return float(
        row["event_repeat_macro"].iloc[0]
    )


def bootstrap_interval(
    bootstrap: pd.DataFrame,
    model: str,
    quantity: str,
    population: str,
) -> tuple[float, float, float]:
    row = bootstrap.loc[
        bootstrap["candidate_model"].eq(model)
        & bootstrap["quantity"].eq(quantity)
        & bootstrap["population"].eq(population)
        & bootstrap["metric"].eq("mae")
    ]

    if len(row) != 1:
        raise RuntimeError(
            "Expected exactly one bootstrap row for "
            f"{model}/{quantity}/{population}/mae, "
            f"found {len(row)}."
        )

    r = row.iloc[0]
    return (
        float(
            r[
                "mean_delta_candidate_minus_reference"
            ]
        ),
        float(r["ci95_low"]),
        float(r["ci95_high"]),
    )


def build_plot_table(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    quantity: str,
) -> pd.DataFrame:
    base_overall = metric_value(
        metrics,
        REFERENCE_MODEL,
        quantity,
        "overall",
    )
    base_tail = metric_value(
        metrics,
        REFERENCE_MODEL,
        quantity,
        "tail",
    )

    rows = []

    # Reference.
    rows.append(
        {
            "model": REFERENCE_MODEL,
            "label": MODEL_LABELS[REFERENCE_MODEL],
            "overall_mae": base_overall,
            "tail_mae": base_tail,
            "overall_low": base_overall,
            "overall_high": base_overall,
            "tail_low": base_tail,
            "tail_high": base_tail,
        }
    )

    # Candidate variants.
    for model in MODEL_ORDER[1:]:
        overall = metric_value(
            metrics,
            model,
            quantity,
            "overall",
        )
        tail = metric_value(
            metrics,
            model,
            quantity,
            "tail",
        )

        (
            overall_delta,
            overall_ci_low,
            overall_ci_high,
        ) = bootstrap_interval(
            bootstrap,
            model,
            quantity,
            "overall",
        )

        (
            tail_delta,
            tail_ci_low,
            tail_ci_high,
        ) = bootstrap_interval(
            bootstrap,
            model,
            quantity,
            "tail",
        )

        # Sanity check that summary and paired mean delta agree.
        expected_overall = (
            base_overall + overall_delta
        )
        expected_tail = (
            base_tail + tail_delta
        )

        if abs(
            overall - expected_overall
        ) > 2e-3:
            print(
                f"WARNING: {quantity}/{model} overall "
                "summary and paired delta differ: "
                f"summary={overall:.6f}, "
                f"base+delta={expected_overall:.6f}"
            )

        if abs(
            tail - expected_tail
        ) > 2e-3:
            print(
                f"WARNING: {quantity}/{model} tail "
                "summary and paired delta differ: "
                f"summary={tail:.6f}, "
                f"base+delta={expected_tail:.6f}"
            )

        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "overall_mae": overall,
                "tail_mae": tail,
                # Translate paired delta CI to absolute coordinates.
                "overall_low": (
                    base_overall + overall_ci_low
                ),
                "overall_high": (
                    base_overall + overall_ci_high
                ),
                "tail_low": (
                    base_tail + tail_ci_low
                ),
                "tail_high": (
                    base_tail + tail_ci_high
                ),
            }
        )

    return pd.DataFrame(rows)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.3,
            "axes.linewidth": 0.8,
        }
    )


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


def draw_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    quantity: str,
    panel_letter: str,
) -> None:
    base = table.loc[
        table["model"].eq(REFERENCE_MODEL)
    ].iloc[0]

    base_x = float(base["overall_mae"])
    base_y = float(base["tail_mae"])

    # Determine plotting limits from point estimates and translated CIs.
    xmin_data = min(
        table["overall_low"].min(),
        table["overall_mae"].min(),
    )
    xmax_data = max(
        table["overall_high"].max(),
        table["overall_mae"].max(),
    )
    ymin_data = min(
        table["tail_low"].min(),
        table["tail_mae"].min(),
    )
    ymax_data = max(
        table["tail_high"].max(),
        table["tail_mae"].max(),
    )

    xspan = max(
        xmax_data - xmin_data,
        0.02,
    )
    yspan = max(
        ymax_data - ymin_data,
        0.04,
    )

    xmin = xmin_data - 0.18 * xspan
    xmax = xmax_data + 0.30 * xspan
    ymin = ymin_data - 0.18 * yspan
    ymax = ymax_data + 0.18 * yspan

    # Reference lines.
    ax.axvline(
        base_x,
        linestyle="--",
        linewidth=0.9,
        alpha=0.65,
    )
    ax.axhline(
        base_y,
        linestyle="--",
        linewidth=0.9,
        alpha=0.65,
    )

    # The desired region relative to the base: both overall and tail MAE lower.
    ideal = Rectangle(
        (xmin, ymin),
        base_x - xmin,
        base_y - ymin,
        fill=False,
        hatch="///",
        linewidth=0.0,
        alpha=0.22,
    )
    ax.add_patch(ideal)

    # Plot reference first.
    ax.scatter(
        [base_x],
        [base_y],
        marker="*",
        s=150,
        linewidths=0.9,
        edgecolors="black",
        label="Mean Base",
        zorder=5,
    )

    # Candidate models.
    for _, row in table.loc[
        ~table["model"].eq(
            REFERENCE_MODEL
        )
    ].iterrows():
        x = float(row["overall_mae"])
        y = float(row["tail_mae"])

        xerr = np.array(
            [
                [
                    max(
                        0.0,
                        x - float(row["overall_low"]),
                    )
                ],
                [
                    max(
                        0.0,
                        float(row["overall_high"]) - x,
                    )
                ],
            ]
        )

        yerr = np.array(
            [
                [
                    max(
                        0.0,
                        y - float(row["tail_low"]),
                    )
                ],
                [
                    max(
                        0.0,
                        float(row["tail_high"]) - y,
                    )
                ],
            ]
        )

        # Arrow from base to candidate highlights the trade-off direction.
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(base_x, base_y),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 0.9,
                "alpha": 0.35,
            },
            zorder=1,
        )

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=6.2,
            capsize=3.0,
            elinewidth=0.9,
            markeredgewidth=0.7,
            label=str(row["label"]),
            zorder=4,
        )

        # Small offsets prevent labels from touching points.
        label_dx = 0.018 * (xmax - xmin)
        label_dy = 0.022 * (ymax - ymin)

        if row["model"] == "mild_tail":
            text_x = x + label_dx
            text_y = y - label_dy
            va = "top"
        elif row["model"] == "mild_compromise":
            text_x = x + label_dx
            text_y = y + label_dy
            va = "bottom"
        else:
            text_x = x + label_dx
            text_y = y + label_dy
            va = "bottom"

        ax.text(
            text_x,
            text_y,
            str(row["label"]),
            fontsize=8.0,
            ha="left",
            va=va,
        )

    # Base label.
    ax.text(
        base_x - 0.018 * (xmax - xmin),
        base_y + 0.025 * (ymax - ymin),
        "Mean Base",
        fontsize=8.0,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    # Region interpretation.
    ax.text(
        xmin + 0.035 * (xmax - xmin),
        ymin + 0.045 * (ymax - ymin),
        "Desired:\nlower overall\nand tail MAE",
        fontsize=7.8,
        ha="left",
        va="bottom",
    )

    ax.text(
        xmax - 0.035 * (xmax - xmin),
        ymin + 0.045 * (ymax - ymin),
        "Tail improves,\noverall worsens",
        fontsize=7.8,
        ha="right",
        va="bottom",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(
        r"Overall MAE ($\log_{10}$ units)"
    )
    ax.set_ylabel(
        r"High-motion-tail MAE ($\log_{10}$ units)"
    )

    ax.set_title(
        f"({panel_letter}) {quantity.upper()}",
        loc="left",
        fontweight="bold",
    )

    style_axis(ax)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics-csv",
        default=(
            "runs/tail_checkpoint_repeated_test/"
            "model_metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--bootstrap-csv",
        default=(
            "runs/tail_checkpoint_repeated_test/"
            "paired_bootstrap_deltas.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="figures/section_4_4",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    bootstrap_path = Path(
        args.bootstrap_csv
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = load_metrics(
        metrics_path
    )
    bootstrap = load_bootstrap(
        bootstrap_path
    )

    pga_table = build_plot_table(
        metrics,
        bootstrap,
        "pga",
    )
    pgv_table = build_plot_table(
        metrics,
        bootstrap,
        "pgv",
    )

    print("\n=== PGA Pareto table ===")
    print(
        pga_table.to_string(
            index=False
        )
    )
    print("\n=== PGV Pareto table ===")
    print(
        pgv_table.to_string(
            index=False
        )
    )

    export = pd.concat(
        [
            pga_table.assign(
                quantity="pga"
            ),
            pgv_table.assign(
                quantity="pgv"
            ),
        ],
        ignore_index=True,
    )

    export_path = (
        out_dir
        / "Fig_4_4_pareto_values.csv"
    )
    export.to_csv(
        export_path,
        index=False,
        encoding="utf-8-sig",
    )

    set_publication_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.15),
    )

    draw_panel(
        axes[0],
        pga_table,
        "pga",
        "a",
    )
    draw_panel(
        axes[1],
        pgv_table,
        "pgv",
        "b",
    )

    # Single legend outside axes; labels also appear next to points.
    handles, labels = (
        axes[1].get_legend_handles_labels()
    )

    # Remove duplicates while preserving order.
    unique = {}
    for handle, label in zip(
        handles,
        labels,
    ):
        if label not in unique:
            unique[label] = handle

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )

    fig.text(
        0.5,
        0.018,
        (
            "Error bars show 95% confidence intervals of paired "
            "event-bootstrap MAE differences relative to Mean Base, "
            "translated to absolute MAE coordinates."
        ),
        ha="center",
        va="bottom",
        fontsize=8.1,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.17,
        top=0.83,
        wspace=0.28,
    )

    png_path = (
        out_dir
        / "Fig_4_4_single_head_pareto_tradeoff.png"
    )
    svg_path = (
        out_dir
        / "Fig_4_4_single_head_pareto_tradeoff.svg"
    )
    pdf_path = (
        out_dir
        / "Fig_4_4_single_head_pareto_tradeoff.pdf"
    )

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
    print(export_path.resolve())


if __name__ == "__main__":
    main()
