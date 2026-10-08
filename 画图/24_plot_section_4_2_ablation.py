# -*- coding: utf-8 -*-
"""
24_plot_section_4_2_ablation.py

Plot Fig. 4.2 for the ablation study.

Panels
------
(a) Information-source ablation
    - coordinate_only
    - waveform_only
    - no_p_offset
    - full

(b) Aggregation strategy
    - full (attention pooling)
    - mean_pooling

Each panel shows:
    - PGA MAE (log10 units)
    - PGV MAE (log10 units)

Outputs
-------
PNG / SVG / PDF

Example
-------
python 24_plot_section_4_2_ablation.py ^
  --coordinate-json "results/coordinate_only.json" ^
  --waveform-json   "results/waveform_only.json" ^
  --nop-json        "results/no_p_offset.json" ^
  --full-json       "results/full.json" ^
  --mean-json       "results/mean_pooling.json" ^
  --out-dir         "figures/section_4_2" ^
  --dpi 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for key, path in paths.items():
        d = load_json(path)

        rows.append(
            {
                "variant_key": key,
                "variant": d.get("variant", key),
                "test_mae_log10_pga": float(d["test_mae_log10_pga"]),
                "test_mae_log10_pgv": float(d["test_mae_log10_pgv"]),
                "test_bias_log10_pga": float(d.get("test_bias_log10_pga", np.nan)),
                "test_bias_log10_pgv": float(d.get("test_bias_log10_pgv", np.nan)),
                "test_factor2_pga": float(d.get("test_factor2_pga", np.nan)),
                "test_factor2_pgv": float(d.get("test_factor2_pgv", np.nan)),
                "parameter_count": int(d.get("parameter_count", -1)),
                "best_epoch": int(d.get("best_epoch", -1)),
            }
        )

    df = pd.DataFrame(rows)
    return df


def add_value_labels(ax, bars, fmt="{:.3f}", dx=0.008):
    for b in bars:
        w = b.get_width()
        y = b.get_y() + b.get_height() / 2
        ax.text(
            w + dx,
            y,
            fmt.format(w),
            va="center",
            ha="left",
            fontsize=8,
        )


def style_axes(ax):
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def plot_panel_a(ax, df: pd.DataFrame):
    """
    Information-source ablation:
    coordinate_only, waveform_only, no_p_offset, full
    """
    order = ["coordinate_only", "waveform_only", "no_p_offset", "full"]

    labels = {
        "coordinate_only": "Coordinate only",
        "waveform_only": "Waveform only",
        "no_p_offset": "No P-offset",
        "full": "Full model\n(attention)",
    }

    sub = df.set_index("variant_key").loc[order].reset_index()

    y = np.arange(len(sub))
    h = 0.34

    pga = sub["test_mae_log10_pga"].to_numpy()
    pgv = sub["test_mae_log10_pgv"].to_numpy()

    bars1 = ax.barh(
        y - h / 2,
        pga,
        height=h,
        label="PGA",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.6,
    )
    bars2 = ax.barh(
        y + h / 2,
        pgv,
        height=h,
        label="PGV",
        alpha=0.52,
        edgecolor="black",
        linewidth=0.6,
    )

    # Highlight full model
    full_idx = order.index("full")
    bars1[full_idx].set_hatch("///")
    bars2[full_idx].set_hatch("///")
    bars1[full_idx].set_linewidth(1.2)
    bars2[full_idx].set_linewidth(1.2)

    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"MAE ($\log_{10}$ units)")
    ax.set_title("(a) Information-source ablation", loc="left", fontweight="bold")

    maxv = max(np.max(pga), np.max(pgv))
    ax.set_xlim(0, maxv * 1.28)

    style_axes(ax)
    add_value_labels(ax, bars1, fmt="{:.3f}")
    add_value_labels(ax, bars2, fmt="{:.3f}")

    # Add short note
    full_pga = float(sub.loc[sub["variant_key"] == "full", "test_mae_log10_pga"].iloc[0])
    coord_pga = float(sub.loc[sub["variant_key"] == "coordinate_only", "test_mae_log10_pga"].iloc[0])
    wave_pga = float(sub.loc[sub["variant_key"] == "waveform_only", "test_mae_log10_pga"].iloc[0])

    improve_coord = (coord_pga - full_pga) / coord_pga * 100.0
    improve_wave = (wave_pga - full_pga) / wave_pga * 100.0

    txt = (
        "Full model vs. single-source variants\n"
        f"PGA MAE reduction: {improve_coord:.1f}% vs coordinate-only, "
        f"{improve_wave:.1f}% vs waveform-only"
    )
    ax.text(
        0.98,
        0.03,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
    )


def plot_panel_b(ax, df: pd.DataFrame):
    """
    Aggregation strategy:
    full (attention) vs mean_pooling
    """
    order = ["full", "mean_pooling"]

    labels = {
        "full": "Attention pooling",
        "mean_pooling": "Mean pooling",
    }

    sub = df.set_index("variant_key").loc[order].reset_index()

    y = np.arange(len(sub))
    h = 0.34

    pga = sub["test_mae_log10_pga"].to_numpy()
    pgv = sub["test_mae_log10_pgv"].to_numpy()

    bars1 = ax.barh(
        y - h / 2,
        pga,
        height=h,
        label="PGA",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.6,
    )
    bars2 = ax.barh(
        y + h / 2,
        pgv,
        height=h,
        label="PGV",
        alpha=0.52,
        edgecolor="black",
        linewidth=0.6,
    )

    # Highlight mean pooling as chosen base model
    mean_idx = order.index("mean_pooling")
    bars1[mean_idx].set_hatch("///")
    bars2[mean_idx].set_hatch("///")
    bars1[mean_idx].set_linewidth(1.2)
    bars2[mean_idx].set_linewidth(1.2)

    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"MAE ($\log_{10}$ units)")
    ax.set_title("(b) Aggregation strategy", loc="left", fontweight="bold")

    maxv = max(np.max(pga), np.max(pgv))
    ax.set_xlim(0, maxv * 1.32)

    style_axes(ax)
    add_value_labels(ax, bars1, fmt="{:.3f}")
    add_value_labels(ax, bars2, fmt="{:.3f}")

    att_pga = float(sub.loc[sub["variant_key"] == "full", "test_mae_log10_pga"].iloc[0])
    mean_pga = float(sub.loc[sub["variant_key"] == "mean_pooling", "test_mae_log10_pga"].iloc[0])
    att_pgv = float(sub.loc[sub["variant_key"] == "full", "test_mae_log10_pgv"].iloc[0])
    mean_pgv = float(sub.loc[sub["variant_key"] == "mean_pooling", "test_mae_log10_pgv"].iloc[0])

    improve_pga = (att_pga - mean_pga) / att_pga * 100.0
    improve_pgv = (att_pgv - mean_pgv) / att_pgv * 100.0

    txt = (
        "Mean pooling improvement over attention\n"
        f"PGA: {improve_pga:.1f}%   |   PGV: {improve_pgv:.1f}%"
    )
    ax.text(
        0.98,
        0.06,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinate-json", type=str, default="./sparse_field_ablation_t0_5s_k5/coordinate_only.json")
    parser.add_argument("--waveform-json", type=str, default="./sparse_field_ablation_t0_5s_k5/waveform_only.json")
    parser.add_argument("--nop-json", type=str, default="./sparse_field_ablation_t0_5s_k5/no_p_offset.json")
    parser.add_argument("--full-json", type=str, default="./sparse_field_ablation_t0_5s_k5/full.json")
    parser.add_argument("--mean-json", type=str, default="./sparse_field_ablation_t0_5s_k5/mean_pooling.json")
    parser.add_argument("--out-dir", type=str, default="figures/section_4_2")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "coordinate_only": Path(args.coordinate_json),
        "waveform_only": Path(args.waveform_json),
        "no_p_offset": Path(args.nop_json),
        "full": Path(args.full_json),
        "mean_pooling": Path(args.mean_json),
    }

    df = build_dataframe(paths)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))

    plot_panel_a(axes[0], df)
    plot_panel_b(axes[1], df)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Hatched bars denote the preferred model in each comparison. "
            "All MAE values are computed in the log10 domain under the same T0=5 s, K=5 setting."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
    )

    fig.subplots_adjust(left=0.10, right=0.99, top=0.82, bottom=0.18, wspace=0.34)

    png_path = out_dir / "Fig_4_2_ablation.png"
    svg_path = out_dir / "Fig_4_2_ablation.svg"
    pdf_path = out_dir / "Fig_4_2_ablation.pdf"

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Also export a CSV summary for convenience
    csv_path = out_dir / "Fig_4_2_ablation_values.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(" ", png_path.resolve())
    print(" ", svg_path.resolve())
    print(" ", pdf_path.resolve())
    print(" ", csv_path.resolve())

    print("\nAblation summary:")
    print(
        df[
            [
                "variant_key",
                "test_mae_log10_pga",
                "test_mae_log10_pgv",
                "test_factor2_pga",
                "test_factor2_pgv",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()