#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
80_plot_nc_generalization_distribution_shift.py

Nature Communications-style Fig. 5 for Causal-SeisField.

Panels
------
a) Chronological / sequence-OOD evaluation design
b) Chronological 2021–2024 high-motion-tail MAE
c) Ridgecrest sequence-OOD high-motion-tail MAE
d) Tail-MAE reduction across increasingly stringent distribution shifts

Inputs
------
Grouped:
  runs/caurc_sequence_aware_bootstrap/hierarchical_bootstrap_summary.csv

Chronological:
  runs/caurc_chronological_2021_2024/chronological_2021_2024_metrics.csv
  runs/caurc_chronological_2021_2024/paired_event_bootstrap_2021_2024.csv

Ridgecrest:
  runs/caurc_ridgecrest_ood/ridgecrest_ood_metrics.csv
  runs/caurc_ridgecrest_ood/paired_event_bootstrap_ridgecrest.csv

Outputs
-------
Fig5_generalization_distribution_shift.png
Fig5_generalization_distribution_shift.pdf
Fig5_generalization_distribution_shift.svg
Fig5_generalization_summary.csv

No training, model selection, or threshold recalculation is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_NAMES = {
    "cross_attention_base",
    "cross_attention",
}

CAURC_NAMES = {
    "ca_urc",
    "caurc",
    "A4_under_only",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def canonical_name(text: str) -> str:
    return str(text).strip().lower()


def get_metric_value(
    df: pd.DataFrame,
    model_aliases: set[str],
    quantity: str,
    population: str,
    metric: str,
) -> float:
    """
    Read canonical metric from a long-format metrics CSV.
    Expected columns include:
      model, quantity, population, metric, value
    """
    required = {"model", "quantity", "population", "metric", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Metrics table missing columns: {sorted(missing)}"
        )

    model_norm = df["model"].astype(str).str.strip().str.lower()

    mask = (
        model_norm.isin({x.lower() for x in model_aliases})
        & (df["quantity"].astype(str).str.lower() == quantity.lower())
        & (df["population"].astype(str).str.lower() == population.lower())
        & (df["metric"].astype(str).str.lower() == metric.lower())
    )

    sub = df.loc[mask, "value"]

    if len(sub) != 1:
        raise RuntimeError(
            "Expected exactly one metric row for "
            f"{sorted(model_aliases)}, {quantity}, {population}, {metric}; "
            f"found {len(sub)}."
        )

    return float(sub.iloc[0])


def find_bootstrap_row(
    df: pd.DataFrame,
    quantity: str,
    population: str,
    metric: str = "mae",
) -> pd.Series:
    required = {"quantity", "population", "metric"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Bootstrap table missing columns: {sorted(missing)}"
        )

    mask = (
        (df["quantity"].astype(str).str.lower() == quantity.lower())
        & (df["population"].astype(str).str.lower() == population.lower())
        & (df["metric"].astype(str).str.lower() == metric.lower())
    )

    # If comparison columns exist, force CA-URC vs Cross-Attention.
    if "candidate" in df.columns:
        cand = df["candidate"].astype(str).str.lower()
        mask &= cand.isin(
            {"ca_urc", "caurc", "a4_under_only", "s4"}
        )

    if "reference" in df.columns:
        ref = df["reference"].astype(str).str.lower()
        mask &= ref.isin(
            {
                "cross_attention_base",
                "cross_attention",
                "s3",
            }
        )

    sub = df.loc[mask]

    if len(sub) != 1:
        raise RuntimeError(
            "Expected exactly one bootstrap row for "
            f"{quantity}, {population}, {metric}; found {len(sub)}."
        )

    return sub.iloc[0]


def value_from_row(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    raise KeyError(
        f"None of these columns were found with finite values: {names}"
    )


def extract_bootstrap_effect(
    df: pd.DataFrame,
    quantity: str,
    population: str = "high_motion_tail",
) -> dict[str, float]:
    row = find_bootstrap_row(
        df,
        quantity=quantity,
        population=population,
        metric="mae",
    )

    base = value_from_row(
        row,
        [
            "reference_value",
            "base_value",
        ],
    )
    caurc = value_from_row(
        row,
        [
            "candidate_value",
            "ca_urc_value",
            "caurc_value",
        ],
    )
    delta = value_from_row(
        row,
        [
            "point_delta_candidate_minus_reference",
            "mean_delta_caurc_minus_base",
        ],
    )

    ci_low = value_from_row(
        row,
        [
            "ci_lower",
            "ci95_low",
        ],
    )
    ci_high = value_from_row(
        row,
        [
            "ci_upper",
            "ci95_high",
        ],
    )

    reduction = -100.0 * delta / abs(base)

    # Delta < 0 is improvement. Convert delta CI to positive reduction CI.
    reduction_ci_low = -100.0 * ci_high / abs(base)
    reduction_ci_high = -100.0 * ci_low / abs(base)

    return {
        "base": base,
        "caurc": caurc,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "reduction_percent": reduction,
        "reduction_ci_low": reduction_ci_low,
        "reduction_ci_high": reduction_ci_high,
    }


def add_pair_panel(
    ax,
    pga_base: float,
    pga_caurc: float,
    pgv_base: float,
    pgv_caurc: float,
    title: str,
) -> None:
    """
    Base -> CA-URC paired line panel, using Matplotlib default color cycle.
    """
    x = np.array([0.0, 1.0])

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_pga = cycle[0]
    c_pgv = cycle[1]

    ax.plot(
        x,
        [pga_base, pga_caurc],
        marker="o",
        linewidth=1.5,
        color=c_pga,
        label="PGA",
    )
    ax.plot(
        x,
        [pgv_base, pgv_caurc],
        marker="s",
        linewidth=1.5,
        color=c_pgv,
        label="PGV",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Cross-Attention\nBase", "CA-URC"]
    )

    for xx, yy in zip(
        x,
        [pga_base, pga_caurc],
    ):
        ax.text(
            xx,
            yy + 0.012,
            f"{yy:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    for xx, yy in zip(
        x,
        [pgv_base, pgv_caurc],
    ):
        ax.text(
            xx,
            yy + 0.012,
            f"{yy:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax.set_ylabel(
        r"High-motion-tail MAE ($\log_{10}$ units)"
    )

    ax.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    values = np.array(
        [
            pga_base,
            pga_caurc,
            pgv_base,
            pgv_caurc,
        ],
        dtype=float,
    )
    ymin = max(
        0.0,
        float(values.min()) - 0.10,
    )
    ymax = float(values.max()) + 0.11
    ax.set_ylim(
        ymin,
        ymax,
    )


def timeline_panel(ax) -> None:
    """
    Schematic protocol panel. Positions represent calendar year.
    Ridgecrest is shown separately because it is held out as a complete sequence.
    """
    y_train = 3.0
    y_val = 2.0
    y_test = 1.0
    y_ood = 0.0

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.plot(
        [2010, 2018],
        [y_train, y_train],
        linewidth=7,
        solid_capstyle="butt",
        color=cycle[0],
    )
    ax.plot(
        [2019, 2020],
        [y_val, y_val],
        linewidth=7,
        solid_capstyle="butt",
        color=cycle[1],
    )
    ax.plot(
        [2021, 2024],
        [y_test, y_test],
        linewidth=7,
        solid_capstyle="butt",
        color=cycle[2],
    )
    ax.scatter(
        [2019],
        [y_ood],
        s=70,
        marker="D",
        color=cycle[3],
        zorder=3,
    )

    ax.text(
        2014,
        y_train + 0.16,
        "Train\n2010–2018",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    ax.text(
        2019.5,
        y_val + 0.16,
        "Validation\n2019 non-Ridgecrest + 2020",
        ha="center",
        va="bottom",
        fontsize=6.6,
    )
    ax.text(
        2022.5,
        y_test + 0.16,
        "Locked chronological test\n2021–2024",
        ha="center",
        va="bottom",
        fontsize=6.6,
    )
    ax.text(
        2019.35,
        y_ood,
        "Ridgecrest sequence OOD\n593 events; frozen model",
        ha="left",
        va="center",
        fontsize=6.6,
    )

    ax.set_xlim(
        2009.5,
        2024.7,
    )
    ax.set_ylim(
        -0.6,
        3.7,
    )

    ax.set_yticks([])
    ax.set_xticks(
        [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    )

    ax.set_title(
        "a  Generalization protocol isolates future years and a complete unseen sequence",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax.set_xlabel("Calendar year")

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.grid(
        axis="x",
        linewidth=0.4,
        alpha=0.18,
    )
    ax.set_axisbelow(True)


def improvement_panel(
    ax,
    regimes: list[str],
    pga_effects: list[dict[str, float]],
    pgv_effects: list[dict[str, float]],
) -> None:
    y = np.arange(
        len(regimes),
        dtype=float,
    )

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_pga = cycle[0]
    c_pgv = cycle[1]

    offset = 0.12

    pga_x = np.array(
        [x["reduction_percent"] for x in pga_effects]
    )
    pgv_x = np.array(
        [x["reduction_percent"] for x in pgv_effects]
    )

    pga_low = np.array(
        [x["reduction_ci_low"] for x in pga_effects]
    )
    pga_high = np.array(
        [x["reduction_ci_high"] for x in pga_effects]
    )
    pgv_low = np.array(
        [x["reduction_ci_low"] for x in pgv_effects]
    )
    pgv_high = np.array(
        [x["reduction_ci_high"] for x in pgv_effects]
    )

    pga_xerr = np.vstack(
        [
            pga_x - pga_low,
            pga_high - pga_x,
        ]
    )
    pgv_xerr = np.vstack(
        [
            pgv_x - pgv_low,
            pgv_high - pgv_x,
        ]
    )

    ax.errorbar(
        pga_x,
        y - offset,
        xerr=pga_xerr,
        fmt="o",
        markersize=4.0,
        capsize=2.5,
        linewidth=1.0,
        color=c_pga,
        label="PGA",
    )
    ax.errorbar(
        pgv_x,
        y + offset,
        xerr=pgv_xerr,
        fmt="s",
        markersize=4.0,
        capsize=2.5,
        linewidth=1.0,
        color=c_pgv,
        label="PGV",
    )

    for i, value in enumerate(pga_x):
        ax.text(
            value + 0.3,
            y[i] - offset,
            f"{value:.1f}%",
            va="center",
            fontsize=6.4,
        )

    for i, value in enumerate(pgv_x):
        ax.text(
            value + 0.3,
            y[i] + offset,
            f"{value:.1f}%",
            va="center",
            fontsize=6.4,
        )

    ax.axvline(
        0.0,
        linewidth=0.7,
        alpha=0.55,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(regimes)
    ax.invert_yaxis()

    ax.set_title(
        "d  Tail benefit weakens as distribution shift becomes more stringent",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax.set_xlabel(
        "Relative reduction in high-motion-tail MAE (%)"
    )

    ax.grid(
        axis="x",
        linewidth=0.45,
        alpha=0.25,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(
        axis="y",
        length=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grouped-bootstrap",
        default=(
            "runs/caurc_sequence_aware_bootstrap/"
            "hierarchical_bootstrap_summary.csv"
        ),
    )
    parser.add_argument(
        "--chrono-metrics",
        default=(
            "runs/caurc_chronological_2021_2024/"
            "chronological_2021_2024_metrics.csv"
        ),
    )
    parser.add_argument(
        "--chrono-bootstrap",
        default=(
            "runs/caurc_chronological_2021_2024/"
            "paired_event_bootstrap_2021_2024.csv"
        ),
    )
    parser.add_argument(
        "--ridgecrest-metrics",
        default=(
            "runs/caurc_ridgecrest_ood/"
            "ridgecrest_ood_metrics.csv"
        ),
    )
    parser.add_argument(
        "--ridgecrest-bootstrap",
        default=(
            "runs/caurc_ridgecrest_ood/"
            "paired_event_bootstrap_ridgecrest.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="figures/nc_generalization",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    paths = {
        "grouped_bootstrap": Path(args.grouped_bootstrap),
        "chrono_metrics": Path(args.chrono_metrics),
        "chrono_bootstrap": Path(args.chrono_bootstrap),
        "ridgecrest_metrics": Path(args.ridgecrest_metrics),
        "ridgecrest_bootstrap": Path(args.ridgecrest_bootstrap),
    }

    for path in paths.values():
        require_file(path)

    grouped_boot = pd.read_csv(
        paths["grouped_bootstrap"]
    )
    chrono_metrics = pd.read_csv(
        paths["chrono_metrics"]
    )
    chrono_boot = pd.read_csv(
        paths["chrono_bootstrap"]
    )
    ridge_metrics = pd.read_csv(
        paths["ridgecrest_metrics"]
    )
    ridge_boot = pd.read_csv(
        paths["ridgecrest_bootstrap"]
    )

    # ----------------------------------------------------------
    # Locked metric extraction.
    # ----------------------------------------------------------
    chrono = {}
    ridge = {}

    for quantity in (
        "pga",
        "pgv",
    ):
        chrono[
            f"{quantity}_base"
        ] = get_metric_value(
            chrono_metrics,
            BASE_NAMES,
            quantity,
            "high_motion_tail",
            "mae",
        )
        chrono[
            f"{quantity}_caurc"
        ] = get_metric_value(
            chrono_metrics,
            CAURC_NAMES,
            quantity,
            "high_motion_tail",
            "mae",
        )

        ridge[
            f"{quantity}_base"
        ] = get_metric_value(
            ridge_metrics,
            BASE_NAMES,
            quantity,
            "high_motion_tail",
            "mae",
        )
        ridge[
            f"{quantity}_caurc"
        ] = get_metric_value(
            ridge_metrics,
            CAURC_NAMES,
            quantity,
            "high_motion_tail",
            "mae",
        )

    grouped_pga = extract_bootstrap_effect(
        grouped_boot,
        "pga",
    )
    grouped_pgv = extract_bootstrap_effect(
        grouped_boot,
        "pgv",
    )

    chrono_pga = extract_bootstrap_effect(
        chrono_boot,
        "pga",
    )
    chrono_pgv = extract_bootstrap_effect(
        chrono_boot,
        "pgv",
    )

    ridge_pga = extract_bootstrap_effect(
        ridge_boot,
        "pga",
    )
    ridge_pgv = extract_bootstrap_effect(
        ridge_boot,
        "pgv",
    )

    # ----------------------------------------------------------
    # Audits against locked manuscript values.
    # ----------------------------------------------------------
    expected = {
        "chrono_pga_base": 0.599704,
        "chrono_pga_caurc": 0.567024,
        "chrono_pgv_base": 0.706155,
        "chrono_pgv_caurc": 0.670638,
        "ridge_pga_base": 0.767550,
        "ridge_pga_caurc": 0.739054,
        "ridge_pgv_base": 0.807832,
        "ridge_pgv_caurc": 0.781284,
        "grouped_pga_reduction": 11.306992,
        "grouped_pgv_reduction": 13.193351,
        "chrono_pga_reduction": 5.449472,
        "chrono_pgv_reduction": 5.029577,
        "ridge_pga_reduction": 3.712700,
        "ridge_pgv_reduction": 3.286270,
    }

    observed = {
        "chrono_pga_base": chrono["pga_base"],
        "chrono_pga_caurc": chrono["pga_caurc"],
        "chrono_pgv_base": chrono["pgv_base"],
        "chrono_pgv_caurc": chrono["pgv_caurc"],
        "ridge_pga_base": ridge["pga_base"],
        "ridge_pga_caurc": ridge["pga_caurc"],
        "ridge_pgv_base": ridge["pgv_base"],
        "ridge_pgv_caurc": ridge["pgv_caurc"],
        "grouped_pga_reduction": grouped_pga["reduction_percent"],
        "grouped_pgv_reduction": grouped_pgv["reduction_percent"],
        "chrono_pga_reduction": chrono_pga["reduction_percent"],
        "chrono_pgv_reduction": chrono_pgv["reduction_percent"],
        "ridge_pga_reduction": ridge_pga["reduction_percent"],
        "ridge_pgv_reduction": ridge_pgv["reduction_percent"],
    }

    for key, exp in expected.items():
        obs = observed[key]
        if not np.isclose(
            obs,
            exp,
            atol=5e-4,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Locked-value audit failed for {key}: "
                f"observed={obs:.6f}, expected={exp:.6f}"
            )

    # ----------------------------------------------------------
    # Export compact summary.
    # ----------------------------------------------------------
    summary_rows = []

    regimes = [
        (
            "Sequence-grouped",
            grouped_pga,
            grouped_pgv,
            "hierarchical sequence→event bootstrap",
        ),
        (
            "Chronological 2021–2024",
            chrono_pga,
            chrono_pgv,
            "paired event bootstrap",
        ),
        (
            "Ridgecrest sequence OOD",
            ridge_pga,
            ridge_pgv,
            "event bootstrap within one held-out sequence",
        ),
    ]

    for regime, pga_eff, pgv_eff, uncertainty in regimes:
        for quantity, eff in (
            ("pga", pga_eff),
            ("pgv", pgv_eff),
        ):
            summary_rows.append(
                {
                    "regime": regime,
                    "quantity": quantity,
                    "base_tail_mae": eff["base"],
                    "caurc_tail_mae": eff["caurc"],
                    "relative_reduction_percent": eff[
                        "reduction_percent"
                    ],
                    "reduction_ci_low_percent": eff[
                        "reduction_ci_low"
                    ],
                    "reduction_ci_high_percent": eff[
                        "reduction_ci_high"
                    ],
                    "uncertainty_analysis": uncertainty,
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        out_dir
        / "Fig5_generalization_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Figure.
    # ----------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    width_in = (
        183.0
        / 25.4
    )
    height_in = (
        145.0
        / 25.4
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            width_in,
            height_in,
        ),
    )

    timeline_panel(
        axes[0, 0]
    )

    add_pair_panel(
        axes[0, 1],
        chrono["pga_base"],
        chrono["pga_caurc"],
        chrono["pgv_base"],
        chrono["pgv_caurc"],
        (
            "b  Tail-risk correction remains effective "
            "for future-year earthquakes"
        ),
    )

    add_pair_panel(
        axes[1, 0],
        ridge["pga_base"],
        ridge["pga_caurc"],
        ridge["pgv_base"],
        ridge["pgv_caurc"],
        (
            "c  Frozen CA-URC retains improvement "
            "for the held-out Ridgecrest sequence"
        ),
    )

    improvement_panel(
        axes[1, 1],
        [
            "Sequence-grouped",
            "Chronological\n2021–2024",
            "Ridgecrest\nsequence OOD",
        ],
        [
            grouped_pga,
            chrono_pga,
            ridge_pga,
        ],
        [
            grouped_pgv,
            chrono_pgv,
            ridge_pgv,
        ],
    )

    handles, labels = (
        axes[0, 1]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.74,
            0.992,
        ),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.10,
        top=0.93,
        wspace=0.40,
        hspace=0.47,
    )

    png = (
        out_dir
        / "Fig5_generalization_distribution_shift.png"
    )
    pdf = (
        out_dir
        / "Fig5_generalization_distribution_shift.pdf"
    )
    svg = (
        out_dir
        / "Fig5_generalization_distribution_shift.svg"
    )

    fig.savefig(
        png,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    print(
        "=== Fig. 5 generalization summary ==="
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )
    print(
        png.resolve()
    )
    print(
        pdf.resolve()
    )
    print(
        svg.resolve()
    )
    print(
        summary_path.resolve()
    )


if __name__ == "__main__":
    main()
