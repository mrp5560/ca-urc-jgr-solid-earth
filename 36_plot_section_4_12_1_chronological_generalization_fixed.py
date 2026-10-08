#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
36_plot_section_4_12_1_chronological_generalization.py

Publication-ready figure for Section 4.12.1:
Chronological extrapolation to 2021–2024.

The figure contains three panels:
(a) PGA event-repeat macro MAE: Frozen Base vs Power Gate (gamma=5.5)
(b) PGV event-repeat macro MAE: Frozen Base vs Power Gate (gamma=5.5)
(c) Event-level paired-bootstrap delta MAE (Power Gate - Frozen Base)

Recommended manuscript message:
- Overall MAE changes only modestly under temporal extrapolation.
- Tail and P-wave-reached targets improve consistently.
- Negative bootstrap delta indicates improvement.

Expected inputs
---------------
runs/time_clean_locked_test_2021_2024_gamma5p5/model_metrics_summary.csv
runs/time_clean_locked_test_2021_2024_gamma5p5/paired_bootstrap_deltas.csv

Outputs
-------
Fig_4_12_1_chronological_generalization.png
Fig_4_12_1_chronological_generalization.svg
Fig_4_12_1_chronological_generalization.pdf
Fig_4_12_1_chronological_generalization_values.csv

Example
-------
python 36_plot_section_4_12_1_chronological_generalization.py ^
  --metrics "runs\\time_clean_locked_test_2021_2024_gamma5p5\\model_metrics_summary.csv" ^
  --bootstrap "runs\\time_clean_locked_test_2021_2024_gamma5p5\\paired_bootstrap_deltas.csv" ^
  --out-dir "figures\\section_4_12_1" ^
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POPULATIONS = ["overall", "tail", "triggered"]
POPULATION_LABELS = {
    "overall": "Overall",
    "tail": "High-motion tail",
    "triggered": "P-wave-reached",
}
MODEL_LABELS = {
    "base": "Frozen base",
    "power_gate": r"Power gate ($\gamma=5.5$)",
}


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def extract_mae_table(metrics: pd.DataFrame, quantity: str) -> pd.DataFrame:
    subset = metrics.loc[
        metrics["quantity"].astype(str).str.lower().eq(quantity)
        & metrics["population"].astype(str).str.lower().isin(POPULATIONS)
        & metrics["metric"].astype(str).str.lower().eq("mae")
        & metrics["model"].astype(str).str.lower().isin(["base", "power_gate"])
    ].copy()

    if subset.empty:
        raise ValueError(f"No MAE rows found for quantity={quantity}")

    pivot = subset.pivot_table(
        index="population",
        columns="model",
        values="event_repeat_macro",
        aggfunc="first",
    ).reindex(POPULATIONS)

    for model in ("base", "power_gate"):
        if model not in pivot.columns:
            raise ValueError(f"Missing model={model} for quantity={quantity}")

    if pivot[["base", "power_gate"]].isna().any().any():
        raise ValueError(f"Incomplete MAE table for quantity={quantity}\n{pivot}")

    return pivot


def extract_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    subset = bootstrap.loc[
        bootstrap["candidate_model"].astype(str).str.lower().eq("power_gate")
        & bootstrap["population"].astype(str).str.lower().isin(POPULATIONS)
        & bootstrap["metric"].astype(str).str.lower().eq("mae")
        & bootstrap["quantity"].astype(str).str.lower().isin(["pga", "pgv"])
    ].copy()

    if subset.empty:
        raise ValueError("No power_gate paired-bootstrap MAE rows found.")

    required_pairs = {
        (quantity, population)
        for quantity in ("pga", "pgv")
        for population in POPULATIONS
    }
    observed_pairs = set(
        zip(
            subset["quantity"].astype(str).str.lower(),
            subset["population"].astype(str).str.lower(),
        )
    )
    missing = sorted(required_pairs - observed_pairs)
    if missing:
        raise ValueError(f"Missing bootstrap quantity/population rows: {missing}")

    return subset


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)


def plot_mae_panel(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    quantity: str,
    panel_label: str,
) -> None:
    x = np.arange(len(POPULATIONS), dtype=float)
    width = 0.34

    base = pivot["base"].to_numpy(dtype=float)
    power = pivot["power_gate"].to_numpy(dtype=float)

    bars_base = ax.bar(
        x - width / 2,
        base,
        width=width,
        label=MODEL_LABELS["base"],
        hatch="///",
        linewidth=0.8,
    )
    bars_power = ax.bar(
        x + width / 2,
        power,
        width=width,
        label=MODEL_LABELS["power_gate"],
        hatch="...",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([POPULATION_LABELS[p] for p in POPULATIONS])
    ax.set_ylabel(r"MAE in $\log_{10}$ units")
    ax.set_title(
        f"({panel_label}) {quantity.upper()}: chronological test",
        loc="left",
        fontweight="bold",
    )

    ymax = max(float(np.max(base)), float(np.max(power)))
    ax.set_ylim(0.0, ymax * 1.20)

    # Percentage change is based on event-repeat macro MAE.
    for index, population in enumerate(POPULATIONS):
        delta_pct = 100.0 * (power[index] - base[index]) / base[index]
        top = max(base[index], power[index])
        ax.text(
            x[index],
            top + 0.035 * ymax,
            rf"$\Delta$={delta_pct:+.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    # Direct values on bars make the figure self-contained.
    for bars in (bars_base, bars_power):
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                rotation=0,
            )

    style_axis(ax)


def plot_bootstrap_panel(ax: plt.Axes, bootstrap: pd.DataFrame) -> None:
    # Put Overall on top, then Tail, then P-wave-reached.
    centers = np.arange(len(POPULATIONS))[::-1].astype(float)
    offsets = {"pga": 0.11, "pgv": -0.11}
    markers = {"pga": "o", "pgv": "s"}

    for quantity in ("pga", "pgv"):
        xs = []
        low_err = []
        high_err = []
        ys = []

        for center, population in zip(centers, POPULATIONS):
            row = bootstrap.loc[
                bootstrap["quantity"].astype(str).str.lower().eq(quantity)
                & bootstrap["population"].astype(str).str.lower().eq(population)
            ].iloc[0]

            mean_delta = float(row["mean_delta_candidate_minus_reference"])
            ci_low = float(row["ci95_low"])
            ci_high = float(row["ci95_high"])

            xs.append(mean_delta)
            low_err.append(mean_delta - ci_low)
            high_err.append(ci_high - mean_delta)
            ys.append(center + offsets[quantity])

        ax.errorbar(
            xs,
            ys,
            xerr=np.vstack([low_err, high_err]),
            fmt=markers[quantity],
            capsize=3.0,
            linewidth=1.2,
            markersize=5.0,
            label=quantity.upper(),
        )

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(centers)
    ax.set_yticklabels([POPULATION_LABELS[p] for p in POPULATIONS])
    ax.set_xlabel(r"$\Delta$MAE = Power gate $-$ Frozen base")
    ax.set_title(
        "(c) Paired event-level bootstrap",
        loc="left",
        fontweight="bold",
    )

    # Interpretation cue without relying on color.
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    ax.text(
        xmin + 0.02 * span,
        centers[-1] - 0.55,
        "Improvement  ←",
        ha="left",
        va="center",
        fontsize=7.8,
    )
    ax.text(
        xmax - 0.02 * span,
        centers[-1] - 0.55,
        "→  Higher error",
        ha="right",
        va="center",
        fontsize=7.8,
    )

    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)


def build_export_table(
    pga: pd.DataFrame,
    pgv: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quantity, pivot in (("pga", pga), ("pgv", pgv)):
        for population in POPULATIONS:
            base = float(pivot.loc[population, "base"])
            power = float(pivot.loc[population, "power_gate"])
            b = bootstrap.loc[
                bootstrap["quantity"].astype(str).str.lower().eq(quantity)
                & bootstrap["population"].astype(str).str.lower().eq(population)
            ].iloc[0]

            rows.append(
                {
                    "quantity": quantity,
                    "population": population,
                    "base_event_repeat_macro_mae": base,
                    "power_event_repeat_macro_mae": power,
                    "relative_change_percent": 100.0 * (power - base) / base,
                    "bootstrap_delta_mae": float(
                        b["mean_delta_candidate_minus_reference"]
                    ),
                    "bootstrap_ci95_low": float(b["ci95_low"]),
                    "bootstrap_ci95_high": float(b["ci95_high"]),
                    "n_paired_events": int(b["n_paired_events"]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default=(
            'runs/time_clean_locked_test_2021_2024_gamma5p5/model_metrics_summary.csv'
        ),
    )
    parser.add_argument(
        "--bootstrap",
        default=(
            'runs/time_clean_locked_test_2021_2024_gamma5p5/paired_bootstrap_deltas.csv'
        ),
    )
    parser.add_argument(
        "--out-dir",
        default='figures/section_4_12_1',
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    bootstrap_path = Path(args.bootstrap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not bootstrap_path.exists():
        raise FileNotFoundError(bootstrap_path)

    metrics = pd.read_csv(metrics_path)
    bootstrap = pd.read_csv(bootstrap_path)

    require_columns(
        metrics,
        [
            "model",
            "quantity",
            "population",
            "metric",
            "event_repeat_macro",
        ],
        "metrics",
    )
    require_columns(
        bootstrap,
        [
            "candidate_model",
            "quantity",
            "population",
            "metric",
            "mean_delta_candidate_minus_reference",
            "ci95_low",
            "ci95_high",
            "n_paired_events",
        ],
        "bootstrap",
    )

    pga = extract_mae_table(metrics, "pga")
    pgv = extract_mae_table(metrics, "pgv")
    bootstrap_mae = extract_bootstrap(bootstrap)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 9.6,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.15),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]},
    )

    plot_mae_panel(axes[0], pga, "pga", "a")
    plot_mae_panel(axes[1], pgv, "pgv", "b")
    plot_bootstrap_panel(axes[2], bootstrap_mae)

    # One legend is enough for the two bar panels.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
    )

    fig.suptitle(
        "Chronological extrapolation to 2021–2024",
        fontsize=11.0,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])

    stem = "Fig_4_12_1_chronological_generalization"
    fig.savefig(out_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    export = build_export_table(pga, pgv, bootstrap_mae)
    export.to_csv(
        out_dir / f"{stem}_values.csv",
        index=False,
    )

    print("=== Section 4.12.1 chronological generalization figure ===")
    print(f"Metrics   : {metrics_path.resolve()}")
    print(f"Bootstrap : {bootstrap_path.resolve()}")
    print("\nValues used in the figure:")
    print(export.to_string(index=False))
    print(f"\nOutputs   : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
