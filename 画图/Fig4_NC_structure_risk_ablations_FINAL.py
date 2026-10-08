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
import matplotlib as mpl
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

# -----------------------------------------------------------------------------
# Publication palette and layout constants
# -----------------------------------------------------------------------------
# Okabe-Ito/Wong-inspired blue-orange pairing: accessible, high contrast and
# restrained enough for a Nature-family main-text figure.
COLORS = {
    "pga": "#0072B2",
    "pgv": "#D55E00",
    "connector": "#B7BDC4",
    "connector_selected": "#6F757B",
    "reference": "#555B61",
    "frame": "#26292D",
    "text": "#202124",
    "muted": "#6D737A",
}

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 136.0
MM_TO_INCH = 1.0 / 25.4

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


def configure_style() -> None:
    """Nature-family figure typography and vector-safe export settings."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.60,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["frame"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Separate 8-pt panel letter from the concise conclusion-style title."""
    ax.set_title(
        title,
        loc="left",
        pad=4.0,
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        -0.115,
        1.025,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


def style_axis(
    ax: plt.Axes,
    *,
    hide_left_spine: bool = False,
    y_tick_marks: bool = True,
) -> None:
    """Compact full frame, no background gridlines, outward ticks."""
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.60)
        ax.spines[side].set_color(COLORS["frame"])

    if hide_left_spine:
        ax.spines["left"].set_visible(False)

    ax.grid(False)
    ax.tick_params(
        direction="out",
        top=False,
        right=False,
        width=0.55,
        length=2.5,
        pad=2.0,
    )
    if not y_tick_marks:
        ax.tick_params(axis="y", length=0, pad=3.5)


def paired_dot_panel(
    ax: plt.Axes,
    labels: list[str],
    pga: np.ndarray,
    pgv: np.ndarray,
    letter: str,
    title: str,
    xlabel: str,
    highlight_index: int | None = None,
    percent: bool = False,
    annotate_selected: bool = True,
) -> None:
    """Minimal dumbbell plot; exact labels are shown only for the selected model."""
    pga = np.asarray(pga, dtype=float).copy()
    pgv = np.asarray(pgv, dtype=float).copy()
    if percent:
        pga *= 100.0
        pgv *= 100.0

    y = np.arange(len(labels), dtype=float)
    values = np.concatenate([pga, pgv])
    span = max(float(np.nanmax(values) - np.nanmin(values)), 1e-6)

    # Draw each row as one restrained comparison. The selected row receives a
    # darker connector and slightly larger, black-edged markers rather than a
    # full-width background band.
    for i in range(len(labels)):
        selected = highlight_index is not None and i == highlight_index
        ax.plot(
            [pga[i], pgv[i]],
            [i, i],
            color=(COLORS["connector_selected"] if selected else COLORS["connector"]),
            linewidth=(0.95 if selected else 0.70),
            zorder=1,
        )
        marker_size = 34 if selected else 25
        edge = COLORS["frame"] if selected else "white"
        edge_width = 0.75 if selected else 0.45
        ax.scatter(
            pga[i], i,
            s=marker_size,
            marker="o",
            facecolor=COLORS["pga"],
            edgecolor=edge,
            linewidth=edge_width,
            zorder=3,
        )
        ax.scatter(
            pgv[i], i,
            s=marker_size + 2,
            marker="s",
            facecolor=COLORS["pgv"],
            edgecolor=edge,
            linewidth=edge_width,
            zorder=3,
        )

        if selected and annotate_selected:
            formatter = (lambda v: f"{v:.1f}%") if percent else (lambda v: f"{v:.3f}")
            # Place labels toward the free side of each marker so they do not
            # collide with the connector. Keep text black per Nature guidance.
            ax.annotate(
                formatter(pga[i]),
                (pga[i], i),
                xytext=(4, -7),
                textcoords="offset points",
                fontsize=5.2,
                color=COLORS["text"],
                ha="left",
                va="top",
                zorder=5,
            )
            ax.annotate(
                formatter(pgv[i]),
                (pgv[i], i),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=5.2,
                color=COLORS["text"],
                ha="left",
                va="bottom",
                zorder=5,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if highlight_index is not None:
        ax.get_yticklabels()[highlight_index].set_fontweight("bold")
    ax.invert_yaxis()

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    pad_left = 0.18 * span
    pad_right = 0.28 * span
    lower = vmin - pad_left
    if percent:
        lower = max(0.0, lower)
    ax.set_xlim(lower, vmax + pad_right)

    ax.set_xlabel(xlabel)
    panel_title(ax, letter, title)
    style_axis(ax, hide_left_spine=True, y_tick_marks=False)


def structure_effect_note(ax: plt.Axes, boot: pd.DataFrame) -> None:
    """Compact black-text S3-minus-S2 effect note without a boxed inset."""
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
        "S3−S2 ΔMAE (95% CI)\n"
        f"PGA {float(pga['point_delta_candidate_minus_reference']):+.3f} "
        f"[{float(pga['ci_lower']):+.3f}, {float(pga['ci_upper']):+.3f}]\n"
        f"PGV {float(pgv['point_delta_candidate_minus_reference']):+.3f} "
        f"[{float(pgv['ci_lower']):+.3f}, {float(pgv['ci_upper']):+.3f}]"
    )
    ax.text(
        0.985,
        0.965,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        linespacing=1.15,
        color=COLORS["text"],
        zorder=6,
    )


def pareto_panel(ax: plt.Axes, gate: pd.DataFrame) -> None:
    """Accuracy-risk map with one label per variant and A4 visually prioritized."""
    rows: dict[str, dict[str, float]] = {}
    for variant in GATE_ORDER:
        row = get_row(gate, "variant", variant)
        rows[variant] = {
            "x_pga": float(row["overall_mae_pga"]),
            "y_pga": 100.0 * float(row["tail_under05_pga"]),
            "x_pgv": float(row["overall_mae_pgv"]),
            "y_pgv": 100.0 * float(row["tail_under05_pgv"]),
        }

    # One faint segment per variant associates the PGA and PGV operating points.
    # A4 receives stronger emphasis; the other ablations provide context.
    for variant in GATE_ORDER:
        r = rows[variant]
        selected = variant == "A4_under_only"
        alpha = 1.0 if selected else 0.72
        connector_color = COLORS["connector_selected"] if selected else COLORS["connector"]
        ax.plot(
            [r["x_pgv"], r["x_pga"]],
            [r["y_pgv"], r["y_pga"]],
            color=connector_color,
            linewidth=0.85 if selected else 0.50,
            alpha=(1.0 if selected else 0.32),
            zorder=1,
        )

        size = 52 if selected else 27
        edge = COLORS["frame"] if selected else "white"
        ew = 0.85 if selected else 0.45
        ax.scatter(
            r["x_pga"], r["y_pga"], s=size, marker="o",
            facecolor=COLORS["pga"], edgecolor=edge, linewidth=ew,
            alpha=alpha, zorder=3,
        )
        ax.scatter(
            r["x_pgv"], r["y_pgv"], s=size + 2, marker="s",
            facecolor=COLORS["pgv"], edgecolor=edge, linewidth=ew,
            alpha=alpha, zorder=3,
        )

    # Label each variant only once, near the midpoint of its PGA/PGV pair.
    # Offsets are in points and tuned for the locked manuscript values.
    label_offsets = {
        "A2_cross_attention_base": (4, 4),
        "A3_tail_only": (-7, -7),
        "A4_under_only": (4, -8),
        "A5_dual_no_leakage": (4, 5),
        "A6_full_cadrg": (4, 7),
    }
    for variant in GATE_ORDER:
        r = rows[variant]
        xm = 0.5 * (r["x_pga"] + r["x_pgv"])
        ym = 0.5 * (r["y_pga"] + r["y_pgv"])
        dx, dy = label_offsets[variant]
        ax.annotate(
            GATE_SHORT[variant],
            (xm, ym),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.3,
            fontweight="bold" if variant == "A4_under_only" else "normal",
            color=COLORS["text"],
            ha="left",
            va="center",
            zorder=5,
        )

    panel_title(
        ax,
        "d",
        "Under-only gating advances the risk–accuracy frontier",
    )
    ax.set_xlabel(r"Overall MAE ($\log_{10}$ units)")
    ax.set_ylabel(r"High-motion-tail $U_{0.5}$ (%)")

    # Lower-left is preferable: lower overall MAE and lower severe underprediction.
    ax.annotate(
        "Preferred",
        xy=(0.055, 0.055),
        xytext=(0.23, 0.22),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=5.2,
        color=COLORS["text"],
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.65,
            color=COLORS["reference"],
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=4,
    )
    style_axis(ax)


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
    configure_style()

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, FIGURE_HEIGHT_MM * MM_TO_INCH),
        facecolor="white",
        constrained_layout=False,
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.155,
        right=0.985,
        bottom=0.095,
        top=0.925,
        wspace=0.38,
        hspace=0.32,
        width_ratios=(1.00, 1.00),
        height_ratios=(1.00, 1.00),
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ],
        dtype=object,
    )

    s_rows = [get_row(structure, "model", key) for key in STRUCTURE_ORDER]
    s_labels = [STRUCTURE_LABELS[key] for key in STRUCTURE_ORDER]

    paired_dot_panel(
        axes[0, 0],
        s_labels,
        np.asarray([float(r["pga_overall_mae"]) for r in s_rows]),
        np.asarray([float(r["pgv_overall_mae"]) for r in s_rows]),
        letter="a",
        title="Target conditioning improves overall accuracy",
        xlabel=r"Overall MAE ($\log_{10}$ units)",
        highlight_index=2,
        annotate_selected=True,
    )
    structure_effect_note(axes[0, 0], boot)

    paired_dot_panel(
        axes[0, 1],
        s_labels,
        np.asarray([float(r["pga_tail_mae"]) for r in s_rows]),
        np.asarray([float(r["pgv_tail_mae"]) for r in s_rows]),
        letter="b",
        title="Backbone changes do not resolve tail error",
        xlabel=r"High-motion-tail MAE ($\log_{10}$ units)",
        highlight_index=2,
        annotate_selected=True,
    )

    g_rows = [get_row(gate, "variant", key) for key in GATE_ORDER]
    g_labels = [GATE_LABELS[key] for key in GATE_ORDER]
    paired_dot_panel(
        axes[1, 0],
        g_labels,
        np.asarray([float(r["tail_mae_pga"]) for r in g_rows]),
        np.asarray([float(r["tail_mae_pgv"]) for r in g_rows]),
        letter="c",
        title="Underprediction-risk gating minimizes tail error",
        xlabel=r"High-motion-tail MAE ($\log_{10}$ units)",
        highlight_index=2,
        annotate_selected=True,
    )

    pareto_panel(axes[1, 1], gate)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.5,
            markerfacecolor=COLORS["pga"], markeredgecolor="white",
            markeredgewidth=0.45, label="PGA",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=4.5,
            markerfacecolor=COLORS["pgv"], markeredgecolor="white",
            markeredgewidth=0.45, label="PGV",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.992),
        ncol=2,
        frameon=False,
        handletextpad=0.38,
        columnspacing=1.10,
        borderaxespad=0.0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Fig4_structure_and_risk_gate_ablations"
    # Keep the exact double-column physical size; avoid bbox_inches='tight'.
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
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
