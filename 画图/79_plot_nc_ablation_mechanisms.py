#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STRUCTURE_ORDER = [
    "S1_no_station_self_attention",
    "S2_no_target_cross_attention",
    "S3_cross_attention_base",
]

STRUCTURE_LABELS = {
    "S1_no_station_self_attention": "S1  w/o station\nself-attention",
    "S2_no_target_cross_attention": "S2  w/o target\ncross-attention",
    "S3_cross_attention_base": "S3  Cross-Attention\nBase",
}

GATE_ORDER = [
    "A2_cross_attention_base",
    "A3_tail_only",
    "A4_under_only",
    "A5_dual_no_leakage",
]

GATE_LABELS = {
    "A2_cross_attention_base": "A2  Base",
    "A3_tail_only": "A3  Tail-only",
    "A4_under_only": "A4  Under-only\nCA-URC",
    "A5_dual_no_leakage": "A5  Dual-risk\nno leakage",
}


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def get_row(df, key_col, key):
    row = df.loc[df[key_col].astype(str) == key]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one row for {key_col}={key!r}, found {len(row)}."
        )
    return row.iloc[0]


def paired_dot_panel(
    ax,
    labels,
    pga,
    pgv,
    title,
    xlabel,
    percent=False,
    emphasize_index=None,
):
    y = np.arange(len(labels), dtype=float)

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pga_color = cycle[0]
    pgv_color = cycle[1]

    values_all = np.concatenate([pga, pgv])
    span = max(float(np.nanmax(values_all) - np.nanmin(values_all)), 1e-6)
    dx = 0.025 * span

    for i in range(len(labels)):
        ax.plot(
            [pga[i], pgv[i]],
            [i, i],
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )

        s1 = 34 if i == emphasize_index else 26
        s2 = 36 if i == emphasize_index else 28

        ax.scatter(
            pga[i], i, s=s1, marker="o",
            color=pga_color, zorder=3,
            label="PGA" if i == 0 else None,
        )
        ax.scatter(
            pgv[i], i, s=s2, marker="s",
            color=pgv_color, zorder=3,
            label="PGV" if i == 0 else None,
        )

        pga_text = f"{pga[i]:.1f}%" if percent else f"{pga[i]:.3f}"
        pgv_text = f"{pgv[i]:.1f}%" if percent else f"{pgv[i]:.3f}"

        ax.text(
            pga[i] + dx, i - 0.12, pga_text,
            fontsize=6.2, va="center"
        )
        ax.text(
            pgv[i] + dx, i + 0.14, pgv_text,
            fontsize=6.2, va="center"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    if emphasize_index is not None:
        ax.get_yticklabels()[emphasize_index].set_fontweight("bold")

    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel(xlabel)

    ax.grid(axis="x", linewidth=0.45, alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", width=0.6, length=3)

    vmin = float(np.nanmin(values_all))
    vmax = float(np.nanmax(values_all))
    span = max(vmax - vmin, 1e-6)
    ax.set_xlim(max(0.0, vmin - 0.18 * span), vmax + 0.36 * span)


def build_combined_summary(structure, gate):
    rows = []

    for key in STRUCTURE_ORDER:
        r = get_row(structure, "model", key)
        rows.append({
            "family": "backbone_structure",
            "variant": key,
            "label": STRUCTURE_LABELS[key].replace("\n", " "),
            "pga_overall_mae": float(r["pga_overall_mae"]),
            "pgv_overall_mae": float(r["pgv_overall_mae"]),
            "pga_tail_mae": float(r["pga_tail_mae"]),
            "pgv_tail_mae": float(r["pgv_tail_mae"]),
            "pga_tail_under05": float(r["pga_tail_under05"]),
            "pgv_tail_under05": float(r["pgv_tail_under05"]),
        })

    for key in GATE_ORDER:
        r = get_row(gate, "variant", key)
        rows.append({
            "family": "risk_trigger",
            "variant": key,
            "label": GATE_LABELS[key].replace("\n", " "),
            "pga_overall_mae": float(r["overall_mae_pga"]),
            "pgv_overall_mae": float(r["overall_mae_pgv"]),
            "pga_tail_mae": float(r["tail_mae_pga"]),
            "pgv_tail_mae": float(r["tail_mae_pgv"]),
            "pga_tail_under05": float(r["tail_under05_pga"]),
            "pgv_tail_under05": float(r["tail_under05_pgv"]),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure-table",
        default=(
            "runs/network_structure_ablation_S1_S4/"
            "paper_network_structure_table.csv"
        ),
    )
    parser.add_argument(
        "--gate-table",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "paper_ablation_table.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="figures/nc_ablation",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    args = parser.parse_args()

    structure_path = Path(args.structure_table)
    gate_path = Path(args.gate_table)

    if not structure_path.exists():
        raise FileNotFoundError(structure_path)
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)

    structure = pd.read_csv(structure_path)
    gate = pd.read_csv(gate_path)

    require_columns(
        structure,
        [
            "model",
            "pga_overall_mae",
            "pga_tail_mae",
            "pga_tail_under05",
            "pga_tail_factor2",
            "pgv_overall_mae",
            "pgv_tail_mae",
            "pgv_tail_under05",
            "pgv_tail_factor2",
        ],
        "structure table",
    )

    require_columns(
        gate,
        [
            "variant",
            "overall_mae_pga",
            "tail_mae_pga",
            "tail_under05_pga",
            "tail_factor2_pga",
            "overall_mae_pgv",
            "tail_mae_pgv",
            "tail_under05_pgv",
            "tail_factor2_pgv",
        ],
        "gate table",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = build_combined_summary(structure, gate)
    combined.to_csv(
        out_dir / "combined_ablation_summary.csv",
        index=False,
    )

    width_in = 183.0 / 25.4
    height_in = 142.0 / 25.4

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(width_in, height_in),
    )

    s_rows = [
        get_row(structure, "model", key)
        for key in STRUCTURE_ORDER
    ]
    s_labels = [
        STRUCTURE_LABELS[key]
        for key in STRUCTURE_ORDER
    ]

    pga_overall = np.array(
        [float(r["pga_overall_mae"]) for r in s_rows]
    )
    pgv_overall = np.array(
        [float(r["pgv_overall_mae"]) for r in s_rows]
    )

    paired_dot_panel(
        axes[0, 0],
        s_labels,
        pga_overall,
        pgv_overall,
        "a  Target conditioning drives catalog-wide prediction",
        r"Overall MAE ($\log_{10}$ units)",
        emphasize_index=2,
    )

    pga_tail = np.array(
        [float(r["pga_tail_mae"]) for r in s_rows]
    )
    pgv_tail = np.array(
        [float(r["pgv_tail_mae"]) for r in s_rows]
    )

    paired_dot_panel(
        axes[0, 1],
        s_labels,
        pga_tail,
        pgv_tail,
        "b  Backbone changes do not consistently improve tail error",
        r"High-motion-tail MAE ($\log_{10}$ units)",
        emphasize_index=2,
    )

    g_rows = [
        get_row(gate, "variant", key)
        for key in GATE_ORDER
    ]
    g_labels = [
        GATE_LABELS[key]
        for key in GATE_ORDER
    ]

    g_pga_tail = np.array(
        [float(r["tail_mae_pga"]) for r in g_rows]
    )
    g_pgv_tail = np.array(
        [float(r["tail_mae_pgv"]) for r in g_rows]
    )

    paired_dot_panel(
        axes[1, 0],
        g_labels,
        g_pga_tail,
        g_pgv_tail,
        "c  Underprediction risk gives the strongest tail correction",
        r"High-motion-tail MAE ($\log_{10}$ units)",
        emphasize_index=2,
    )

    g_pga_u = np.array(
        [100.0 * float(r["tail_under05_pga"]) for r in g_rows]
    )
    g_pgv_u = np.array(
        [100.0 * float(r["tail_under05_pgv"]) for r in g_rows]
    )

    paired_dot_panel(
        axes[1, 1],
        g_labels,
        g_pga_u,
        g_pgv_u,
        "d  Under-only correction most consistently suppresses low bias",
        r"High-motion-tail $U_{0.5}$",
        percent=True,
        emphasize_index=2,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.3,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.99,
        bottom=0.10,
        top=0.92,
        wspace=0.58,
        hspace=0.44,
    )

    png = out_dir / "Fig4_ablation_structure_and_risk.png"
    pdf = out_dir / "Fig4_ablation_structure_and_risk.pdf"
    svg = out_dir / "Fig4_ablation_structure_and_risk.svg"

    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("=== Combined ablation summary ===")
    print(combined.to_string(index=False))
    print("\nSaved:")
    print(png.resolve())
    print(pdf.resolve())
    print(svg.resolve())
    print((out_dir / "combined_ablation_summary.csv").resolve())


if __name__ == "__main__":
    main()
