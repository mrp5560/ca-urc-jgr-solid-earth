#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature Communications-style Fig. 4 for Section 2.3.

Scientific question
-------------------
Which components explain the final system's behaviour?
1) Backbone structure: station self-attention and target-conditioned cross-attention.
2) Risk trigger: high-motion-tail membership versus predicted severe-underprediction risk.

Primary inputs
--------------
1) runs/network_structure_ablation_S1_S4/paper_network_structure_table.csv
2) runs/network_structure_ablation_S1_S4/structure_hierarchical_bootstrap_summary.csv
3) runs/cadrg_gate_ablation_A3_A5/paper_ablation_table.csv

Outputs
-------
Fig4_structure_and_risk_gate_ablations.png/.pdf/.svg
TableS3_structure_and_gate_ablations.csv/.tex
Fig4_caption.txt

Use --demo only to preview the layout from the locked aggregate values. For the
submission figure, run without --demo so the program reads and audits the
experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    "A6_full_cadrg",
]
GATE_LABELS = {
    "A2_cross_attention_base": "A2  Base",
    "A3_tail_only": "A3  Tail-only",
    "A4_under_only": "A4  Under-only\nCA-URC",
    "A5_dual_no_leakage": "A5  Dual-risk\nno leakage",
    "A6_full_cadrg": "A6  Full dual-risk",
}
GATE_SHORT = {
    "A2_cross_attention_base": "A2",
    "A3_tail_only": "A3",
    "A4_under_only": "A4",
    "A5_dual_no_leakage": "A5",
    "A6_full_cadrg": "A6",
}

# Locked aggregate values. They are used for --demo and as a drift audit.
LOCKED_STRUCTURE = {
    "S1_no_station_self_attention": {
        "pga_overall_mae": 0.296625,
        "pga_tail_mae": 0.604795,
        "pga_tail_under05": 0.596937,
        "pga_tail_factor2": 0.219941,
        "pgv_overall_mae": 0.270998,
        "pgv_tail_mae": 0.705974,
        "pgv_tail_under05": 0.646344,
        "pgv_tail_factor2": 0.240011,
    },
    "S2_no_target_cross_attention": {
        "pga_overall_mae": 0.320106,
        "pga_tail_mae": 0.672380,
        "pga_tail_under05": 0.700109,
        "pga_tail_factor2": 0.129249,
        "pgv_overall_mae": 0.280913,
        "pgv_tail_mae": 0.743247,
        "pgv_tail_under05": 0.667161,
        "pgv_tail_factor2": 0.167448,
    },
    "S3_cross_attention_base": {
        "pga_overall_mae": 0.305941,
        "pga_tail_mae": 0.639199,
        "pga_tail_under05": 0.639990,
        "pga_tail_factor2": 0.145760,
        "pgv_overall_mae": 0.265767,
        "pgv_tail_mae": 0.746959,
        "pgv_tail_under05": 0.714567,
        "pgv_tail_factor2": 0.106318,
    },
}

LOCKED_GATE = {
    "A2_cross_attention_base": {
        "overall_mae_pga": 0.305941,
        "tail_mae_pga": 0.639199,
        "tail_under05_pga": 0.639990,
        "tail_factor2_pga": 0.145760,
        "overall_mae_pgv": 0.265767,
        "tail_mae_pgv": 0.746959,
        "tail_under05_pgv": 0.714567,
        "tail_factor2_pgv": 0.106318,
        "non_tail_mae_pga": 0.294755,
        "non_tail_mae_pgv": 0.255686,
        "bias_pga": -0.006911,
        "bias_pgv": -0.028981,
    },
    "A3_tail_only": {
        "overall_mae_pga": 0.304535,
        "tail_mae_pga": 0.587103,
        "tail_under05_pga": 0.569394,
        "tail_factor2_pga": 0.198592,
        "overall_mae_pgv": 0.264718,
        "tail_mae_pgv": 0.712213,
        "tail_under05_pgv": 0.668249,
        "tail_factor2_pgv": 0.150551,
        "non_tail_mae_pga": 0.295834,
        "non_tail_mae_pgv": 0.257120,
        "bias_pga": -0.000677,
        "bias_pgv": -0.023489,
    },
    "A4_under_only": {
        "overall_mae_pga": 0.305531,
        "tail_mae_pga": 0.566925,
        "tail_under05_pga": 0.538049,
        "tail_factor2_pga": 0.205067,
        "overall_mae_pgv": 0.263486,
        "tail_mae_pgv": 0.648410,
        "tail_under05_pgv": 0.603540,
        "tail_factor2_pgv": 0.205740,
        "non_tail_mae_pga": 0.295957,
        "non_tail_mae_pgv": 0.255635,
        "bias_pga": 0.031684,
        "bias_pgv": 0.005872,
    },
    "A5_dual_no_leakage": {
        "overall_mae_pga": 0.304375,
        "tail_mae_pga": 0.582532,
        "tail_under05_pga": 0.565752,
        "tail_factor2_pga": 0.206256,
        "overall_mae_pgv": 0.265424,
        "tail_mae_pgv": 0.723101,
        "tail_under05_pgv": 0.672347,
        "tail_factor2_pgv": 0.137134,
        "non_tail_mae_pga": 0.296096,
        "non_tail_mae_pgv": 0.257873,
        "bias_pga": 0.000619,
        "bias_pgv": -0.023122,
    },
    "A6_full_cadrg": {
        "overall_mae_pga": 0.305171,
        "tail_mae_pga": 0.594620,
        "tail_under05_pga": 0.573642,
        "tail_factor2_pga": 0.197736,
        "overall_mae_pgv": 0.267100,
        "tail_mae_pgv": 0.731744,
        "tail_under05_pgv": 0.670632,
        "tail_factor2_pgv": 0.129733,
        "non_tail_mae_pga": 0.296500,
        "non_tail_mae_pgv": 0.259774,
        "bias_pga": -0.000545,
        "bias_pgv": -0.021267,
    },
}

# Primary structure comparisons, S3 minus reference, from hierarchical bootstrap.
LOCKED_STRUCTURE_EFFECTS = pd.DataFrame(
    [
        {
            "candidate": "s3", "reference": "s1", "quantity": "pga",
            "population": "overall", "metric": "mae",
            "point_delta_candidate_minus_reference": 0.009316,
            "ci_lower": 0.000708, "ci_upper": 0.017872,
        },
        {
            "candidate": "s3", "reference": "s1", "quantity": "pgv",
            "population": "overall", "metric": "mae",
            "point_delta_candidate_minus_reference": -0.005230,
            "ci_lower": -0.015180, "ci_upper": 0.005433,
        },
        {
            "candidate": "s3", "reference": "s2", "quantity": "pga",
            "population": "overall", "metric": "mae",
            "point_delta_candidate_minus_reference": -0.014165,
            "ci_lower": -0.024631, "ci_upper": -0.003299,
        },
        {
            "candidate": "s3", "reference": "s2", "quantity": "pgv",
            "population": "overall", "metric": "mae",
            "point_delta_candidate_minus_reference": -0.015146,
            "ci_lower": -0.031266, "ci_upper": -0.000598,
        },
    ]
)


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def locked_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    structure = pd.DataFrame(
        [{"model": key, **values} for key, values in LOCKED_STRUCTURE.items()]
    )
    gate = pd.DataFrame(
        [{"variant": key, **values} for key, values in LOCKED_GATE.items()]
    )
    return structure, gate, LOCKED_STRUCTURE_EFFECTS.copy()


def load_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.demo:
        return locked_frames()

    structure_path = Path(args.structure_table)
    gate_path = Path(args.gate_table)
    bootstrap_path = Path(args.structure_bootstrap)
    for path in (structure_path, gate_path, bootstrap_path):
        if not path.exists():
            raise FileNotFoundError(path)

    structure = pd.read_csv(structure_path)
    gate = pd.read_csv(gate_path)
    boot = pd.read_csv(bootstrap_path)
    return structure, gate, boot


def audit_locked(structure: pd.DataFrame, gate: pd.DataFrame, tolerance: float) -> None:
    require_columns(
        structure,
        [
            "model", "pga_overall_mae", "pga_tail_mae", "pga_tail_under05",
            "pga_tail_factor2", "pgv_overall_mae", "pgv_tail_mae",
            "pgv_tail_under05", "pgv_tail_factor2",
        ],
        "structure table",
    )
    require_columns(
        gate,
        [
            "variant", "overall_mae_pga", "tail_mae_pga", "tail_under05_pga",
            "tail_factor2_pga", "overall_mae_pgv", "tail_mae_pgv",
            "tail_under05_pgv", "tail_factor2_pgv", "non_tail_mae_pga",
            "non_tail_mae_pgv", "bias_pga", "bias_pgv",
        ],
        "gate table",
    )

    failures: list[str] = []
    for key, expected in LOCKED_STRUCTURE.items():
        row = structure.loc[structure["model"].astype(str) == key]
        if len(row) != 1:
            failures.append(f"structure row {key}: found {len(row)}")
            continue
        for col, exp in expected.items():
            obs = float(row.iloc[0][col])
            if not np.isclose(obs, exp, atol=tolerance, rtol=0.0):
                failures.append(f"{key}/{col}: observed={obs:.6f}, locked={exp:.6f}")

    for key, expected in LOCKED_GATE.items():
        row = gate.loc[gate["variant"].astype(str) == key]
        if len(row) != 1:
            failures.append(f"gate row {key}: found {len(row)}")
            continue
        for col, exp in expected.items():
            obs = float(row.iloc[0][col])
            if not np.isclose(obs, exp, atol=tolerance, rtol=0.0):
                failures.append(f"{key}/{col}: observed={obs:.6f}, locked={exp:.6f}")

    if failures:
        raise RuntimeError(
            "Locked-value audit failed. This usually indicates that a different "
            "prediction version was supplied:\n  " + "\n  ".join(failures[:30])
        )


def get_row(df: pd.DataFrame, key_column: str, key: str) -> pd.Series:
    row = df.loc[df[key_column].astype(str) == key]
    if len(row) != 1:
        raise RuntimeError(f"Expected one row for {key_column}={key!r}; found {len(row)}")
    return row.iloc[0]


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis == "x":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=3)


def paired_dot_panel(
    ax: plt.Axes,
    labels: list[str],
    pga: np.ndarray,
    pgv: np.ndarray,
    title: str,
    xlabel: str,
    highlight_index: int | None = None,
    percent: bool = False,
) -> None:
    pga_color = "#3B6FB6"
    pgv_color = "#C46A2B"
    connector = "#C8C8C8"
    highlight = "#F1F1F1"

    y = np.arange(len(labels), dtype=float)
    values = np.concatenate([pga, pgv]).astype(float)
    if percent:
        values = 100.0 * values
        pga = 100.0 * pga
        pgv = 100.0 * pgv

    if highlight_index is not None:
        ax.axhspan(
            highlight_index - 0.45,
            highlight_index + 0.45,
            color=highlight,
            zorder=0,
        )

    span = max(float(np.nanmax(values) - np.nanmin(values)), 1e-6)
    offset = 0.025 * span

    for i in range(len(labels)):
        ax.plot([pga[i], pgv[i]], [i, i], color=connector, linewidth=1.0, zorder=1)
        size = 40 if i == highlight_index else 29
        ax.scatter(
            pga[i], i, s=size, marker="o", color=pga_color,
            edgecolor="white", linewidth=0.55, zorder=3,
        )
        ax.scatter(
            pgv[i], i, s=size + 2, marker="s", color=pgv_color,
            edgecolor="white", linewidth=0.55, zorder=3,
        )
        formatter = (lambda v: f"{v:.1f}%") if percent else (lambda v: f"{v:.3f}")
        ax.text(pga[i] + offset, i - 0.12, formatter(pga[i]), color=pga_color,
                fontsize=6.4, va="center")
        ax.text(pgv[i] + offset, i + 0.14, formatter(pgv[i]), color=pgv_color,
                fontsize=6.4, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if highlight_index is not None:
        ax.get_yticklabels()[highlight_index].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel(xlabel)
    style_axis(ax, "x")

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = max(vmax - vmin, 1e-6)
    ax.set_xlim(max(0.0, vmin - 0.20 * span), vmax + 0.40 * span)


def structure_effect_inset(ax: plt.Axes, boot: pd.DataFrame) -> None:
    """Add the prespecified S3-S2 overall-MAE paired effects."""
    required = {
        "candidate", "reference", "quantity", "population", "metric",
        "point_delta_candidate_minus_reference", "ci_lower", "ci_upper",
    }
    if not required.issubset(boot.columns):
        return

    mask = (
        boot["candidate"].astype(str).str.lower().eq("s3")
        & boot["reference"].astype(str).str.lower().eq("s2")
        & boot["population"].astype(str).str.lower().eq("overall")
        & boot["metric"].astype(str).str.lower().eq("mae")
    )
    sub = boot.loc[mask].copy()
    if len(sub) != 2:
        return

    pga = sub.loc[sub["quantity"].astype(str).str.lower().eq("pga")]
    pgv = sub.loc[sub["quantity"].astype(str).str.lower().eq("pgv")]
    if len(pga) != 1 or len(pgv) != 1:
        return

    pga = pga.iloc[0]
    pgv = pgv.iloc[0]
    text = (
        "S3 − S2 paired effect\n"
        f"PGA {float(pga['point_delta_candidate_minus_reference']):+.3f} "
        f"[{float(pga['ci_lower']):+.3f}, {float(pga['ci_upper']):+.3f}]\n"
        f"PGV {float(pgv['point_delta_candidate_minus_reference']):+.3f} "
        f"[{float(pgv['ci_lower']):+.3f}, {float(pgv['ci_upper']):+.3f}]"
    )
    ax.text(
        0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top",
        fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor="#B8B8B8", linewidth=0.6, alpha=0.96),
        zorder=6,
    )


def pareto_panel(ax: plt.Axes, gate: pd.DataFrame) -> None:
    pga_color = "#3B6FB6"
    pgv_color = "#C46A2B"

    for variant in GATE_ORDER:
        row = get_row(gate, "variant", variant)
        highlight = variant == "A4_under_only"
        size = 68 if highlight else 38
        edge = "black" if highlight else "white"
        lw = 0.9 if highlight else 0.5

        x_pga = float(row["overall_mae_pga"])
        y_pga = 100.0 * float(row["tail_under05_pga"])
        x_pgv = float(row["overall_mae_pgv"])
        y_pgv = 100.0 * float(row["tail_under05_pgv"])

        ax.scatter(x_pga, y_pga, s=size, marker="o", color=pga_color,
                   edgecolor=edge, linewidth=lw, zorder=3)
        ax.scatter(x_pgv, y_pgv, s=size + 2, marker="s", color=pgv_color,
                   edgecolor=edge, linewidth=lw, zorder=3)

        # Hand-tuned offsets keep the compact Pareto panel legible.
        label = GATE_SHORT[variant]
        pga_offsets = {
            "A2_cross_attention_base": (5, 4),
            "A3_tail_only": (5, -9),
            "A4_under_only": (5, -10),
            "A5_dual_no_leakage": (5, -1),
            "A6_full_cadrg": (5, 7),
        }
        pgv_offsets = {
            "A2_cross_attention_base": (5, -10),
            "A3_tail_only": (5, -10),
            "A4_under_only": (5, -10),
            "A5_dual_no_leakage": (5, -1),
            "A6_full_cadrg": (5, 6),
        }
        ax.annotate(label, (x_pga, y_pga), xytext=pga_offsets[variant],
                    textcoords="offset points", fontsize=6.0,
                    fontweight="bold" if highlight else "normal")
        ax.annotate(label, (x_pgv, y_pgv), xytext=pgv_offsets[variant],
                    textcoords="offset points", fontsize=6.0,
                    fontweight="bold" if highlight else "normal")

    ax.set_title(
        "d  Underprediction-only gating advances the accuracy–risk frontier",
        loc="left", fontweight="bold", pad=5,
    )
    ax.set_xlabel(r"Overall MAE ($\log_{10}$ units)")
    ax.set_ylabel(r"High-motion-tail $U_{0.5}$ (%)")
    ax.text(0.02, 0.04, "Preferred direction", transform=ax.transAxes,
            fontsize=6.4, ha="left", va="bottom")
    ax.annotate(
        "", xy=(0.03, 0.06), xytext=(0.18, 0.20),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color="#555555"),
    )
    style_axis(ax, "both")


def build_supplementary_table(structure: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in STRUCTURE_ORDER:
        r = get_row(structure, "model", key)
        rows.append({
            "family": "backbone structure",
            "variant": key,
            "display_name": STRUCTURE_LABELS[key].replace("\n", " "),
            "pga_overall_mae": float(r["pga_overall_mae"]),
            "pgv_overall_mae": float(r["pgv_overall_mae"]),
            "pga_tail_mae": float(r["pga_tail_mae"]),
            "pgv_tail_mae": float(r["pgv_tail_mae"]),
            "pga_tail_under05": float(r["pga_tail_under05"]),
            "pgv_tail_under05": float(r["pgv_tail_under05"]),
            "pga_tail_factor2": float(r["pga_tail_factor2"]),
            "pgv_tail_factor2": float(r["pgv_tail_factor2"]),
            "pga_bias": np.nan,
            "pgv_bias": np.nan,
        })
    for key in GATE_ORDER:
        r = get_row(gate, "variant", key)
        rows.append({
            "family": "risk trigger",
            "variant": key,
            "display_name": GATE_LABELS[key].replace("\n", " "),
            "pga_overall_mae": float(r["overall_mae_pga"]),
            "pgv_overall_mae": float(r["overall_mae_pgv"]),
            "pga_tail_mae": float(r["tail_mae_pga"]),
            "pgv_tail_mae": float(r["tail_mae_pgv"]),
            "pga_tail_under05": float(r["tail_under05_pga"]),
            "pgv_tail_under05": float(r["tail_under05_pgv"]),
            "pga_tail_factor2": float(r["tail_factor2_pga"]),
            "pgv_tail_factor2": float(r["tail_factor2_pgv"]),
            "pga_bias": float(r["bias_pga"]),
            "pgv_bias": float(r["bias_pgv"]),
        })
    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame, out_dir: Path) -> None:
    csv_path = out_dir / "TableS3_structure_and_gate_ablations.csv"
    tex_path = out_dir / "TableS3_structure_and_gate_ablations.tex"
    table.to_csv(csv_path, index=False)

    display = table.copy()
    numeric_cols = [
        c for c in display.columns
        if c not in {"family", "variant", "display_name"}
    ]
    for col in numeric_cols:
        if "under05" in col or "factor2" in col:
            display[col] = display[col].map(
                lambda x: "--" if pd.isna(x) else f"{100.0 * float(x):.1f}\\%"
            )
        else:
            display[col] = display[col].map(
                lambda x: "--" if pd.isna(x) else f"{float(x):.3f}"
            )
    tex_path.write_text(
        display.to_latex(index=False, escape=False), encoding="utf-8"
    )


def make_figure(
    structure: pd.DataFrame,
    gate: pd.DataFrame,
    boot: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    width_in = 183.0 / 25.4
    height_in = 149.0 / 25.4

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(width_in, height_in))

    s_rows = [get_row(structure, "model", key) for key in STRUCTURE_ORDER]
    s_labels = [STRUCTURE_LABELS[key] for key in STRUCTURE_ORDER]

    paired_dot_panel(
        axes[0, 0],
        s_labels,
        np.asarray([float(r["pga_overall_mae"]) for r in s_rows]),
        np.asarray([float(r["pgv_overall_mae"]) for r in s_rows]),
        "a  Target conditioning improves overall prediction",
        r"Overall MAE ($\log_{10}$ units)",
        highlight_index=2,
    )
    structure_effect_inset(axes[0, 0], boot)

    paired_dot_panel(
        axes[0, 1],
        s_labels,
        np.asarray([float(r["pga_tail_mae"]) for r in s_rows]),
        np.asarray([float(r["pgv_tail_mae"]) for r in s_rows]),
        "b  Backbone changes alone do not resolve tail errors",
        r"High-motion-tail MAE ($\log_{10}$ units)",
        highlight_index=2,
    )

    g_rows = [get_row(gate, "variant", key) for key in GATE_ORDER]
    g_labels = [GATE_LABELS[key] for key in GATE_ORDER]
    paired_dot_panel(
        axes[1, 0],
        g_labels,
        np.asarray([float(r["tail_mae_pga"]) for r in g_rows]),
        np.asarray([float(r["tail_mae_pgv"]) for r in g_rows]),
        "c  Underprediction risk gives the lowest tail error",
        r"High-motion-tail MAE ($\log_{10}$ units)",
        highlight_index=2,
    )

    pareto_panel(axes[1, 1], gate)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B6FB6",
               markeredgecolor="white", markersize=5.3, label="PGA"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#C46A2B",
               markeredgecolor="white", markersize=5.3, label="PGV"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.4,
    )

    fig.subplots_adjust(
        left=0.13, right=0.99, bottom=0.10, top=0.92,
        wspace=0.52, hspace=0.68,
    )

    for suffix in ("png", "pdf", "svg"):
        path = out_dir / f"Fig4_structure_and_risk_gate_ablations.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
    plt.close(fig)


def caption_text() -> str:
    return (
        "Fig. 4 | Target-conditioned representation and underprediction-risk gating "
        "play complementary roles. a, Catalog-wide event-macro MAE for backbone "
        "variants without station self-attention (S1), without target-conditioned "
        "cross-attention (S2), and the full Cross-Attention Base (S3). The inset "
        "reports paired S3-minus-S2 effects with 95% confidence intervals from "
        "hierarchical sequence-to-event bootstrap resampling. b, High-motion-tail "
        "MAE for the same backbone variants. c, High-motion-tail MAE for alternative "
        "risk-trigger formulations: the uncorrected base (A2), high-motion-tail-only "
        "gating (A3), underprediction-risk-only gating (A4; CA-URC), dual-risk gating "
        "without the leakage term (A5), and the full dual-risk formulation (A6). "
        "d, Catalog-wide MAE versus severe-underprediction rate in the high-motion "
        "tail for the same risk-trigger variants; lower values on both axes are "
        "preferred. Circles and squares denote PGA and PGV, respectively. All models "
        "were evaluated on exactly matched target predictions under the locked "
        "sequence-grouped protocol. Exact numerical values are reported in "
        "Supplementary Table 3."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure-table",
        default=(
            "runs/network_structure_ablation_S1_S4/"
            "paper_network_structure_table.csv"
        ),
    )
    parser.add_argument(
        "--structure-bootstrap",
        default=(
            "runs/network_structure_ablation_S1_S4/"
            "structure_hierarchical_bootstrap_summary.csv"
        ),
    )
    parser.add_argument(
        "--gate-table",
        default="runs/cadrg_gate_ablation_A3_A5/paper_ablation_table.csv",
    )
    parser.add_argument("--out-dir", default="figures/nc_section_2_3")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--audit-tolerance", type=float, default=5e-4)
    parser.add_argument("--skip-locked-audit", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    structure, gate, boot = load_frames(args)
    if not args.skip_locked_audit:
        audit_locked(structure, gate, args.audit_tolerance)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_supplementary_table(structure, gate)
    save_table(table, out_dir)
    make_figure(structure, gate, boot, out_dir, args.dpi)
    (out_dir / "Fig4_caption.txt").write_text(caption_text(), encoding="utf-8")

    print("=== Section 2.3 figure generated ===")
    print(table.to_string(index=False))
    print("\nSaved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
