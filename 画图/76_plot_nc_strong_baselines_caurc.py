#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "median_observed": "Median observed",
    "nearest_observed": "Nearest observed",
    "idw_observed": "IDW observed",
    "plum_like": "PLUM-like",
    "graph": "Graph",
    "tail_weighted_attention": "Tail-weighted\nAttention",
    "cross_attention": "Cross-Attention\nBase",
    "ca_urc": "CA-URC",
    "source_oracle_ridge": "Source-oracle Ridge†",
}

METHOD_TYPES = {
    "median_observed": "Causal propagation",
    "nearest_observed": "Causal propagation",
    "idw_observed": "Causal propagation",
    "plum_like": "Causal propagation",
    "graph": "Learned causal",
    "tail_weighted_attention": "Learned causal",
    "cross_attention": "Learned causal",
    "ca_urc": "Proposed",
    "source_oracle_ridge": "Privileged reference",
}

TABLE_ORDER = [
    "median_observed",
    "nearest_observed",
    "idw_observed",
    "plum_like",
    "graph",
    "tail_weighted_attention",
    "cross_attention",
    "ca_urc",
    "source_oracle_ridge",
]

FIGURE_ORDER = [
    "graph",
    "tail_weighted_attention",
    "cross_attention",
    "ca_urc",
]

LOCKED_MAE_AUDIT = {
    ("median_observed", "pga", "overall"): 0.628999,
    ("median_observed", "pgv", "overall"): 0.470210,
    ("nearest_observed", "pga", "overall"): 0.606299,
    ("nearest_observed", "pgv", "overall"): 0.507124,
    ("idw_observed", "pga", "overall"): 0.818735,
    ("idw_observed", "pgv", "overall"): 0.604905,
    ("plum_like", "pga", "overall"): 0.527525,
    ("plum_like", "pgv", "overall"): 0.534475,
    ("graph", "pga", "overall"): 0.309561,
    ("graph", "pgv", "overall"): 0.262140,
    ("tail_weighted_attention", "pga", "overall"): 0.339845,
    ("tail_weighted_attention", "pgv", "overall"): 0.285829,
    ("cross_attention", "pga", "overall"): 0.305941,
    ("cross_attention", "pgv", "overall"): 0.265767,
    ("median_observed", "pga", "high_motion_tail"): 1.049199,
    ("median_observed", "pgv", "high_motion_tail"): 1.176154,
    ("nearest_observed", "pga", "high_motion_tail"): 0.956844,
    ("nearest_observed", "pgv", "high_motion_tail"): 1.070305,
    ("idw_observed", "pga", "high_motion_tail"): 0.697588,
    ("idw_observed", "pgv", "high_motion_tail"): 0.805691,
    ("plum_like", "pga", "high_motion_tail"): 0.875347,
    ("plum_like", "pgv", "high_motion_tail"): 0.813297,
    ("graph", "pga", "high_motion_tail"): 0.614605,
    ("graph", "pgv", "high_motion_tail"): 0.759396,
    ("tail_weighted_attention", "pga", "high_motion_tail"): 0.612139,
    ("tail_weighted_attention", "pgv", "high_motion_tail"): 0.740587,
    ("cross_attention", "pga", "high_motion_tail"): 0.639199,
    ("cross_attention", "pgv", "high_motion_tail"): 0.746959,
}


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_sources(
    baseline_metrics_path: Path,
    bootstrap_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(baseline_metrics_path)
    if not bootstrap_path.exists():
        raise FileNotFoundError(bootstrap_path)

    metrics = pd.read_csv(baseline_metrics_path)
    boot = pd.read_csv(bootstrap_path)

    require_columns(
        metrics,
        ["method", "quantity", "population", "metric", "value"],
        "baseline metrics",
    )
    require_columns(
        boot,
        [
            "candidate",
            "reference",
            "quantity",
            "population",
            "metric",
            "candidate_value",
        ],
        "bootstrap summary",
    )
    return metrics, boot


def extract_value(
    metrics: pd.DataFrame,
    method: str,
    quantity: str,
    population: str,
    metric: str,
) -> float:
    row = metrics.loc[
        (metrics["method"].astype(str) == method)
        & (metrics["quantity"].astype(str) == quantity)
        & (metrics["population"].astype(str) == population)
        & (metrics["metric"].astype(str) == metric)
    ]
    if row.empty:
        return np.nan
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one row for {(method, quantity, population, metric)}, "
            f"found {len(row)}."
        )
    return float(row.iloc[0]["value"])


def extract_caurc_value(
    boot: pd.DataFrame,
    quantity: str,
    population: str,
    metric: str,
) -> float:
    row = boot.loc[
        (boot["candidate"].astype(str) == "ca_urc")
        & (boot["reference"].astype(str) == "cross_attention")
        & (boot["quantity"].astype(str) == quantity)
        & (boot["population"].astype(str) == population)
        & (boot["metric"].astype(str) == metric)
    ]
    if row.empty:
        raise RuntimeError(
            f"Cannot find CA-URC value for {(quantity, population, metric)}."
        )

    vals = row["candidate_value"].to_numpy(float)
    if np.max(np.abs(vals - vals[0])) > 1e-10:
        raise RuntimeError(
            f"Inconsistent CA-URC candidate values for "
            f"{(quantity, population, metric)}: {vals}"
        )
    return float(vals[0])


def add_caurc_rows(metrics: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quantity in ("pga", "pgv"):
        for population in ("overall", "high_motion_tail"):
            for metric in ("mae", "under05", "factor2", "bias"):
                try:
                    value = extract_caurc_value(
                        boot, quantity, population, metric
                    )
                except RuntimeError:
                    continue

                rows.append(
                    {
                        "method": "ca_urc",
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "value": value,
                    }
                )

    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def audit_locked_mae(metrics: pd.DataFrame, tolerance: float = 5e-6) -> None:
    problems = []
    for (method, quantity, population), expected in LOCKED_MAE_AUDIT.items():
        actual = extract_value(metrics, method, quantity, population, "mae")
        if not np.isfinite(actual):
            problems.append(f"missing {(method, quantity, population)}")
        elif abs(actual - expected) > tolerance:
            problems.append(
                f"{(method, quantity, population)}: "
                f"actual={actual:.9f}, locked={expected:.9f}"
            )

    if problems:
        raise RuntimeError(
            "Locked benchmark audit failed:\n  " + "\n  ".join(problems)
        )


def build_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available_methods = set(metrics["method"].astype(str))

    for method in TABLE_ORDER:
        if method not in available_methods:
            continue

        rows.append(
            {
                "Model": METHOD_LABELS[method].replace("\n", " "),
                "Type": METHOD_TYPES[method],
                "PGA overall MAE": extract_value(
                    metrics, method, "pga", "overall", "mae"
                ),
                "PGA tail MAE": extract_value(
                    metrics, method, "pga", "high_motion_tail", "mae"
                ),
                "PGA tail U0.5": extract_value(
                    metrics, method, "pga", "high_motion_tail", "under05"
                ),
                "PGV overall MAE": extract_value(
                    metrics, method, "pgv", "overall", "mae"
                ),
                "PGV tail MAE": extract_value(
                    metrics, method, "pgv", "high_motion_tail", "mae"
                ),
                "PGV tail U0.5": extract_value(
                    metrics, method, "pgv", "high_motion_tail", "under05"
                ),
            }
        )

    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame, out_dir: Path) -> None:
    csv_path = out_dir / "Table1_strong_baseline_comparison.csv"
    tex_path = out_dir / "Table1_strong_baseline_comparison.tex"

    table.to_csv(csv_path, index=False)

    display = table.copy()

    for col in [
        "PGA overall MAE",
        "PGA tail MAE",
        "PGV overall MAE",
        "PGV tail MAE",
    ]:
        display[col] = display[col].map(
            lambda x: f"{x:.3f}" if np.isfinite(x) else "--"
        )

    for col in ["PGA tail U0.5", "PGV tail U0.5"]:
        display[col] = display[col].map(
            lambda x: f"{100*x:.1f}\\%" if np.isfinite(x) else "--"
        )

    latex = display.to_latex(
        index=False,
        escape=False,
        column_format="llrrrrrr",
    )

    note = (
        "\n% † Source-oracle Ridge uses final catalog magnitude and hypocenter; "
        "it is a privileged-information reference, not an operational causal baseline.\n"
    )

    tex_path.write_text(
        latex + note,
        encoding="utf-8",
    )

    print("Saved table:")
    print(csv_path.resolve())
    print(tex_path.resolve())


def panel_dotplot(
    ax,
    metrics: pd.DataFrame,
    population: str,
    metric: str,
    title: str,
    xlabel: str,
    percent: bool = False,
) -> None:
    pga_color = "#3B6FB6"
    pgv_color = "#C46A2B"
    connector_color = "#C8C8C8"
    highlight = "#F3F3F3"

    methods = FIGURE_ORDER
    y = np.arange(len(methods), dtype=float)

    ca_idx = methods.index("ca_urc")
    ax.axhspan(
        ca_idx - 0.45,
        ca_idx + 0.45,
        color=highlight,
        zorder=0,
    )

    all_values = []

    for i, method in enumerate(methods):
        pga = extract_value(
            metrics, method, "pga", population, metric
        )
        pgv = extract_value(
            metrics, method, "pgv", population, metric
        )

        if percent:
            pga *= 100.0
            pgv *= 100.0

        all_values.extend([pga, pgv])

        ax.plot(
            [pga, pgv],
            [i, i],
            color=connector_color,
            linewidth=1.0,
            zorder=1,
        )

        ax.scatter(
            pga,
            i,
            s=28,
            marker="o",
            color=pga_color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="PGA" if i == 0 else None,
        )

        ax.scatter(
            pgv,
            i,
            s=30,
            marker="s",
            color=pgv_color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="PGV" if i == 0 else None,
        )

        fmt = (
            (lambda v: f"{v:.1f}%")
            if percent
            else (lambda v: f"{v:.3f}")
        )

        span_now = max(max(all_values) - min(all_values), 1e-3)
        offset = 0.022 * span_now

        ax.text(
            pga + offset,
            i - 0.11,
            fmt(pga),
            fontsize=6.2,
            color=pga_color,
            va="center",
        )

        ax.text(
            pgv + offset,
            i + 0.13,
            fmt(pgv),
            fontsize=6.2,
            color=pgv_color,
            va="center",
        )

    labels = [METHOD_LABELS[m] for m in methods]

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    for tick, method in zip(
        ax.get_yticklabels(),
        methods,
    ):
        if method == "ca_urc":
            tick.set_fontweight("bold")

    ax.invert_yaxis()
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax.set_xlabel(xlabel)

    ax.grid(
        axis="x",
        linewidth=0.45,
        color="#E6E6E6",
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
        pad=3,
    )
    ax.tick_params(
        axis="x",
        width=0.6,
        length=3,
    )

    vmax = max(all_values)
    vmin = min(all_values)
    span = max(vmax - vmin, 1e-6)

    left = max(
        0.0,
        vmin - 0.18 * span,
    )
    right = vmax + 0.34 * span

    ax.set_xlim(left, right)


def make_figure(
    metrics: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    width_in = 183.0 / 25.4
    height_in = 74.0 / 25.4

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
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            width_in,
            height_in,
        ),
    )

    panel_dotplot(
        axes[0],
        metrics,
        population="overall",
        metric="mae",
        title="a  Catalog-wide prediction error",
        xlabel=r"MAE ($\log_{10}$ units)",
    )

    panel_dotplot(
        axes[1],
        metrics,
        population="high_motion_tail",
        metric="mae",
        title="b  High-motion-tail error",
        xlabel=r"Tail MAE ($\log_{10}$ units)",
    )

    panel_dotplot(
        axes[2],
        metrics,
        population="high_motion_tail",
        metric="under05",
        title="c  Severe underprediction in the tail",
        xlabel=r"Tail $U_{0.5}$",
        percent=True,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.19,
        top=0.83,
        wspace=0.50,
    )

    png = out_dir / "Fig3_strong_baselines_and_tail_risk.png"
    pdf = out_dir / "Fig3_strong_baselines_and_tail_risk.pdf"
    svg = out_dir / "Fig3_strong_baselines_and_tail_risk.svg"

    fig.savefig(
        png,
        dpi=dpi,
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

    plt.close(fig)

    print("Saved figure:")
    print(png.resolve())
    print(pdf.resolve())
    print(svg.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-metrics",
        default=(
            "runs/final_strong_baselines_reuse_locked/"
            "final_strong_baseline_metrics.csv"
        ),
    )

    parser.add_argument(
        "--bootstrap",
        default=(
            "runs/caurc_sequence_aware_bootstrap/"
            "hierarchical_bootstrap_summary.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default="figures/nc_main_results",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_metrics, boot = load_sources(
        Path(args.baseline_metrics),
        Path(args.bootstrap),
    )

    audit_locked_mae(
        baseline_metrics
    )

    metrics = add_caurc_rows(
        baseline_metrics,
        boot,
    )

    table = build_table(
        metrics
    )

    save_table(
        table,
        out_dir,
    )

    make_figure(
        metrics,
        out_dir,
        args.dpi,
    )

    print("\nFinal paper table:")
    print(
        table.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
