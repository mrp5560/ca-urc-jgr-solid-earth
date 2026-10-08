#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
29_plot_section_4_8_bootstrap_forest.py

Section 4.8:
Event-level bootstrap confirms robust tail-risk reduction.

Input
-----
runs/locked_power_gated_test_gamma3/paired_bootstrap_deltas.csv

Expected columns
----------------
candidate_model
quantity
population
metric
mean_delta_candidate_minus_reference
ci95_low
ci95_high
probability_candidate_favorable
n_paired_events

The locked evaluation defines:
    delta = PowerGate - Base

For MAE and Under05:
    delta < 0  -> Power Gate is better
    delta > 0  -> Base is better

Figure
------
(a) Event-level paired Delta MAE with 95% CI
(b) Event-level paired Delta Under05 with 95% CI
    Under05 deltas are displayed in percentage points.

Populations
-----------
Overall
Non-tail
Triggered
High-motion tail
M4+

Outputs
-------
Fig_4_8_bootstrap_forest.png
Fig_4_8_bootstrap_forest.svg
Fig_4_8_bootstrap_forest.pdf
Fig_4_8_bootstrap_values.csv

Example
-------
python 29_plot_section_4_8_bootstrap_forest.py ^
  --bootstrap "runs\\locked_power_gated_test_gamma3\\paired_bootstrap_deltas.csv" ^
  --out-dir "figures\\section_4_8" ^
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POPULATION_ORDER = [
    "overall",
    "non_tail",
    "triggered",
    "tail",
    "m4plus",
]

POPULATION_LABELS = {
    "overall": "Overall",
    "non_tail": "Non-tail",
    "triggered": "Triggered",
    "tail": "High-motion tail",
    "m4plus": r"$M \geq 4$",
}

QUANTITY_LABELS = {
    "pga": "PGA",
    "pgv": "PGV",
}


def load_bootstrap(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "candidate_model",
        "quantity",
        "population",
        "metric",
        "mean_delta_candidate_minus_reference",
        "ci95_low",
        "ci95_high",
        "probability_candidate_favorable",
        "n_paired_events",
    }

    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            "paired_bootstrap_deltas.csv is missing required columns:\n"
            + "\n".join(sorted(missing))
            + "\n\nAvailable columns:\n"
            + "\n".join(frame.columns.astype(str))
        )

    frame = frame.copy()

    for column in (
        "candidate_model",
        "quantity",
        "population",
        "metric",
    ):
        frame[column] = (
            frame[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    for column in (
        "mean_delta_candidate_minus_reference",
        "ci95_low",
        "ci95_high",
        "probability_candidate_favorable",
        "n_paired_events",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.loc[
        frame["candidate_model"].eq("power_gate")
        & frame["quantity"].isin(["pga", "pgv"])
        & frame["population"].isin(POPULATION_ORDER)
        & frame["metric"].isin(["mae", "under05"])
    ].copy()

    if len(frame) == 0:
        raise ValueError(
            "No Power Gate rows for MAE/Under05 were found."
        )

    duplicate_keys = (
        frame.groupby(
            ["quantity", "population", "metric"]
        )
        .size()
    )

    bad_duplicates = duplicate_keys[
        duplicate_keys != 1
    ]

    if len(bad_duplicates) > 0:
        raise ValueError(
            "Expected exactly one row per "
            "(quantity, population, metric), but found:\n"
            f"{bad_duplicates}"
        )

    return frame


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.labelsize": 9.6,
            "axes.titlesize": 10.3,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.55,
        alpha=0.28,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_row(
    frame: pd.DataFrame,
    quantity: str,
    population: str,
    metric: str,
) -> pd.Series:
    row = frame.loc[
        frame["quantity"].eq(quantity)
        & frame["population"].eq(population)
        & frame["metric"].eq(metric)
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected one row for "
            f"{quantity}/{population}/{metric}, "
            f"found {len(row)}."
        )

    return row.iloc[0]


def collect_metric(
    frame: pd.DataFrame,
    quantity: str,
    metric: str,
    scale: float = 1.0,
) -> pd.DataFrame:
    rows = []

    for population in POPULATION_ORDER:
        row = get_row(
            frame,
            quantity,
            population,
            metric,
        )

        mean_delta = (
            float(
                row[
                    "mean_delta_candidate_minus_reference"
                ]
            )
            * scale
        )

        ci_low = (
            float(row["ci95_low"])
            * scale
        )

        ci_high = (
            float(row["ci95_high"])
            * scale
        )

        rows.append(
            {
                "quantity": quantity,
                "population": population,
                "metric": metric,
                "mean_delta": mean_delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "probability_candidate_favorable": float(
                    row[
                        "probability_candidate_favorable"
                    ]
                ),
                "n_paired_events": int(
                    row["n_paired_events"]
                ),
                "significant": bool(
                    (ci_high < 0.0)
                    or (ci_low > 0.0)
                ),
                "favors_power_gate": bool(
                    ci_high < 0.0
                ),
                "favors_base": bool(
                    ci_low > 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def format_delta(
    value: float,
    decimals: int,
) -> str:
    return f"{value:+.{decimals}f}"


def plot_forest_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    scale: float,
    xlabel: str,
    title: str,
    panel_letter: str,
    decimals: int,
) -> pd.DataFrame:
    y_base = np.arange(
        len(POPULATION_ORDER),
        dtype=float,
    )[::-1]

    offset = 0.11

    plotted_frames = []

    for quantity, marker, y_offset in (
        ("pga", "o", +offset),
        ("pgv", "s", -offset),
    ):
        values = collect_metric(
            frame,
            quantity,
            metric,
            scale=scale,
        )

        plotted_frames.append(
            values.copy()
        )

        mean = values[
            "mean_delta"
        ].to_numpy(dtype=float)

        low = values[
            "ci95_low"
        ].to_numpy(dtype=float)

        high = values[
            "ci95_high"
        ].to_numpy(dtype=float)

        xerr = np.vstack(
            [
                np.maximum(
                    mean - low,
                    0.0,
                ),
                np.maximum(
                    high - mean,
                    0.0,
                ),
            ]
        )

        y = y_base + y_offset

        ax.errorbar(
            mean,
            y,
            xerr=xerr,
            fmt=marker,
            markersize=5.0,
            capsize=2.7,
            capthick=0.8,
            elinewidth=1.05,
            linewidth=0,
            label=QUANTITY_LABELS[
                quantity
            ],
            zorder=4,
        )

        # Add exact paired deltas beside the CI.
        span = max(
            np.nanmax(high)
            - np.nanmin(low),
            1e-6,
        )

        text_offset = 0.018 * span

        for yi, m, lo, hi in zip(
            y,
            mean,
            low,
            high,
        ):
            if m <= 0:
                xpos = lo - text_offset
                ha = "right"
            else:
                xpos = hi + text_offset
                ha = "left"

            ax.text(
                xpos,
                yi,
                format_delta(
                    m,
                    decimals,
                ),
                ha=ha,
                va="center",
                fontsize=7.4,
                clip_on=False,
            )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.set_yticks(
        y_base
    )

    ax.set_yticklabels(
        [
            POPULATION_LABELS[
                population
            ]
            for population
            in POPULATION_ORDER
        ]
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_title(
        f"({panel_letter}) {title}",
        loc="left",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="best",
        ncol=2,
    )

    # Direction-of-benefit annotation.
    xlim = ax.get_xlim()
    x_min, x_max = xlim

    ax.text(
        x_min
        + 0.02 * (x_max - x_min),
        y_base[-1] - 0.67,
        "← favors Power Gate",
        ha="left",
        va="center",
        fontsize=7.5,
    )

    ax.text(
        x_max
        - 0.02 * (x_max - x_min),
        y_base[-1] - 0.67,
        "favors Base →",
        ha="right",
        va="center",
        fontsize=7.5,
    )

    ax.set_ylim(
        y_base[-1] - 0.85,
        y_base[0] + 0.65,
    )

    style_axis(ax)

    return pd.concat(
        plotted_frames,
        ignore_index=True,
    )


def build_export_table(
    mae: pd.DataFrame,
    under05: pd.DataFrame,
) -> pd.DataFrame:
    export = pd.concat(
        [
            mae.assign(
                display_metric="Delta MAE"
            ),
            under05.assign(
                display_metric=(
                    "Delta Under05 "
                    "(percentage points)"
                )
            ),
        ],
        ignore_index=True,
    )

    export["population_label"] = (
        export["population"]
        .map(POPULATION_LABELS)
    )

    export["quantity_label"] = (
        export["quantity"]
        .map(QUANTITY_LABELS)
    )

    return export[
        [
            "display_metric",
            "quantity",
            "quantity_label",
            "population",
            "population_label",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "significant",
            "favors_power_gate",
            "favors_base",
            "probability_candidate_favorable",
            "n_paired_events",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap",
        default=(
            "runs/"
            "locked_power_gated_test_gamma3/"
            "paired_bootstrap_deltas.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "figures/"
            "section_4_8"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    bootstrap_path = Path(
        args.bootstrap
    )

    if not bootstrap_path.exists():
        raise FileNotFoundError(
            f"Bootstrap file not found: "
            f"{bootstrap_path.resolve()}"
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_bootstrap(
        bootstrap_path
    )

    print(
        "\n=== Section 4.8 paired bootstrap ==="
    )

    print(
        f"Input: {bootstrap_path.resolve()}"
    )

    print(
        "\nPower Gate vs Base "
        "(negative MAE/Under05 delta is favorable):"
    )

    display = frame.loc[
        frame["metric"].isin(
            ["mae", "under05"]
        ),
        [
            "quantity",
            "population",
            "metric",
            "mean_delta_candidate_minus_reference",
            "ci95_low",
            "ci95_high",
            "probability_candidate_favorable",
            "n_paired_events",
        ],
    ].copy()

    print(
        display.to_string(
            index=False
        )
    )

    set_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.6, 4.55),
        sharey=True,
    )

    mae_values = plot_forest_panel(
        axes[0],
        frame,
        metric="mae",
        scale=1.0,
        xlabel=(
            r"$\Delta$MAE = "
            r"MAE$_{\mathrm{PowerGate}}$ "
            r"$-$ MAE$_{\mathrm{Base}}$"
        ),
        title=(
            "Paired change in MAE"
        ),
        panel_letter="a",
        decimals=3,
    )

    under_values = plot_forest_panel(
        axes[1],
        frame,
        metric="under05",
        scale=100.0,
        xlabel=(
            r"$\Delta$Under05 "
            "(percentage points)"
        ),
        title=(
            "Paired change in severe "
            "underestimation rate"
        ),
        panel_letter="b",
        decimals=2,
    )

    # Keep left panel population labels visible.
    axes[0].tick_params(
        axis="y",
        labelleft=True,
    )

    # Right panel shares the y axis; suppress duplicate labels.
    axes[1].tick_params(
        axis="y",
        labelleft=False,
    )

    fig.subplots_adjust(
        left=0.15,
        right=0.985,
        bottom=0.18,
        top=0.94,
        wspace=0.16,
    )

    png_path = (
        out_dir
        / "Fig_4_8_bootstrap_forest.png"
    )

    svg_path = (
        out_dir
        / "Fig_4_8_bootstrap_forest.svg"
    )

    pdf_path = (
        out_dir
        / "Fig_4_8_bootstrap_forest.pdf"
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

    export = build_export_table(
        mae_values,
        under_values,
    )

    export_path = (
        out_dir
        / "Fig_4_8_bootstrap_values.csv"
    )

    export.to_csv(
        export_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nSaved:"
    )

    print(
        png_path.resolve()
    )

    print(
        svg_path.resolve()
    )

    print(
        pdf_path.resolve()
    )

    print(
        export_path.resolve()
    )


if __name__ == "__main__":
    main()
