#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
82_plot_nc_observation_budget_robustness.py

Nature Communications-style observation-budget robustness figure for
Causal-SeisField / CA-URC.

The script reads four hierarchical bootstrap summary CSV files corresponding to:
    (t0, K) = (3 s, 3), (5 s, 5), (10 s, 5), (10 s, 10)

Panels
------
a  Observation-budget design and common-event comparison core
b  High-motion-tail MAE: Cross-Attention Base vs CA-URC
c  Paired improvement in high-motion-tail MAE with hierarchical-bootstrap 95% CI
d  Reduction in severe underprediction U0.5 with hierarchical-bootstrap 95% CI

No seaborn is used. Vector outputs are saved as PDF/SVG.

Example
-------
python 82_plot_nc_observation_budget_robustness.py \
    --b33 hierarchical_bootstrap_summary.csv \
    --b55 "hierarchical_bootstrap_summary(1).csv" \
    --b105 "hierarchical_bootstrap_summary(2).csv" \
    --b1010 "hierarchical_bootstrap_summary(3).csv" \
    --out-dir figures/observation_budget
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

# Model colours are used only where the two models are compared directly.
BASE_COLOR = "#60758A"       # muted steel blue-grey
FINAL_COLOR = "#C6524A"      # restrained vermilion

# Quantity colours are used only in paired-effect panels.
PGA_COLOR = "#2F6FA3"
PGV_COLOR = "#D9792B"

TEXT_COLOR = "#1D2733"
MUTED = "#6C7885"
GRID = "#E3E8EC"
ZERO = "#87929C"
BORDER = "#D9E0E5"
LIGHT_GRAY = "#F4F6F8"
NEGATIVE_ZONE = "#F5F8FA"
CARD_COLORS = ["#F2F6F9", "#EEF4F8", "#F5F3F1", "#F8F0EE"]
CARD_ACCENTS = ["#A9BBC9", "#7F9DB4", "#D5A68F", "#C98772"]

BUDGETS = [
    ("3s/3", 3, 3),
    ("5s/5", 5, 5),
    ("10s/5", 10, 5),
    ("10s/10", 10, 10),
]


def setup_style():
    """Compact publication styling for a ~183-mm Nature Communications figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.72,
            "xtick.major.width": 0.68,
            "ytick.major.width": 0.68,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )


def style_axis(ax, grid_axis="y"):
    """Restrained NC-style data axis: only left/bottom spines and light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.72)
    ax.spines["bottom"].set_linewidth(0.72)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.50, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", pad=2.0)


def panel_letter(ax, letter, x=-0.075, y=1.045):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
        clip_on=False,
    )


def panel_title(ax, letter: str, title: str, x_title: float = 0.0):
    panel_letter(ax, letter)
    ax.set_title(
        title,
        loc="left",
        x=x_title,
        fontsize=8.6,
        fontweight="bold",
        pad=6.0,
        color=TEXT_COLOR,
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def read_bootstrap(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "candidate",
        "reference",
        "quantity",
        "population",
        "metric",
        "candidate_value",
        "reference_value",
        "point_delta_candidate_minus_reference",
        "ci_lower",
        "ci_upper",
        "relative_change_percent",
        "delta_percentage_points",
        "n_paired_events",
        "n_sequence_groups_used",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df


def select_row(df, quantity, metric, population="high_motion_tail"):
    m = (
        df["quantity"].astype(str).eq(quantity)
        & df["metric"].astype(str).eq(metric)
        & df["population"].astype(str).eq(population)
    )
    hit = df.loc[m]
    if len(hit) != 1:
        raise RuntimeError(
            f"Expected one row: {quantity}/{population}/{metric}, found {len(hit)}"
        )
    return hit.iloc[0]


def extract_budget(path: Path) -> dict:
    df = read_bootstrap(path)

    # common-test size is taken from the overall paired evaluation
    overall = df.loc[
        df["quantity"].astype(str).eq("pga")
        & df["population"].astype(str).eq("overall")
        & df["metric"].astype(str).eq("mae")
    ]
    if len(overall) != 1:
        raise RuntimeError(f"Cannot identify common-test audit row in {path}")
    overall = overall.iloc[0]

    out = {
        "n_events": int(overall["n_paired_events"]),
        "n_sequence_groups": int(overall["n_sequence_groups_used"]),
    }

    for q in ("pga", "pgv"):
        mae = select_row(df, q, "mae")
        u05 = select_row(df, q, "under05")

        out[f"{q}_base_tail_mae"] = float(mae["reference_value"])
        out[f"{q}_final_tail_mae"] = float(mae["candidate_value"])
        out[f"{q}_delta_mae"] = float(mae["point_delta_candidate_minus_reference"])
        out[f"{q}_delta_mae_low"] = float(mae["ci_lower"])
        out[f"{q}_delta_mae_high"] = float(mae["ci_upper"])
        out[f"{q}_rel_mae_pct"] = float(mae["relative_change_percent"])

        out[f"{q}_base_tail_u05"] = float(u05["reference_value"])
        out[f"{q}_final_tail_u05"] = float(u05["candidate_value"])
        out[f"{q}_delta_u05"] = float(u05["point_delta_candidate_minus_reference"])
        out[f"{q}_delta_u05_low"] = float(u05["ci_lower"])
        out[f"{q}_delta_u05_high"] = float(u05["ci_upper"])
        out[f"{q}_delta_u05_pp"] = float(u05["delta_percentage_points"])

        # significance audit
        out[f"{q}_mae_ci_excludes_zero"] = (
            out[f"{q}_delta_mae_low"] > 0 or out[f"{q}_delta_mae_high"] < 0
        )
        out[f"{q}_u05_ci_excludes_zero"] = (
            out[f"{q}_delta_u05_low"] > 0 or out[f"{q}_delta_u05_high"] < 0
        )

    return out


# -----------------------------------------------------------------------------
# Panel a: observation-budget schematic
# -----------------------------------------------------------------------------


def synthetic_wave(x0, x1, y0, amp, cycles, ax, seed=1):
    """Draw a compact synthetic waveform glyph; it is schematic, not data."""
    rng = np.random.default_rng(seed)
    x = np.linspace(x0, x1, 240)
    t = np.linspace(0, 1, len(x))
    envelope = np.exp(-3.2 * np.abs(t - 0.40))
    signal = (
        np.sin(2 * np.pi * cycles * t)
        + 0.45 * np.sin(2 * np.pi * (cycles * 1.75) * t + 0.7)
        + 0.16 * rng.normal(size=len(t))
    )
    y = y0 + amp * envelope * signal
    ax.plot(x, y, color="#5E7B91", lw=0.72, alpha=0.95, clip_on=False)


def station_triangle(cx, cy, size, ax):
    pts = np.array(
        [
            [cx, cy + size],
            [cx - 0.82 * size, cy - 0.66 * size],
            [cx + 0.82 * size, cy - 0.66 * size],
        ]
    )
    ax.add_patch(Polygon(pts, closed=True, facecolor="#405364", edgecolor="none"))


def plot_budget_design(ax, n_events, n_groups):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_letter(ax, "a", x=-0.025, y=1.015)
    ax.text(
        0.045, 0.995,
        "Observation-budget design",
        fontsize=8.8, fontweight="bold", va="top", color=TEXT_COLOR,
    )

    # A single directional cue replaces repeated explanatory text.
    ax.text(
        0.50, 0.885,
        "Increasing observation budget ",
        fontsize=6.8, ha="center", va="center", color=MUTED,
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.09, 0.842), (0.91, 0.842),
            arrowstyle="-|>", mutation_scale=11.5,
            linewidth=0.95, color="#9CA8B2",
        )
    )

    x_centers = [0.15, 0.375, 0.625, 0.85]
    card_w = 0.185
    card_h = 0.49
    card_y = 0.255

    for i, ((label, t0, k), cx) in enumerate(zip(BUDGETS, x_centers)):
        box = FancyBboxPatch(
            (cx - card_w / 2, card_y),
            card_w,
            card_h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=CARD_COLORS[i],
            edgecolor=BORDER,
            linewidth=0.45,
        )
        ax.add_patch(box)

        # Thin accent line visually orders the budgets without implying good/bad.
        ax.plot(
            [cx - card_w / 2 + 0.012, cx + card_w / 2 - 0.012],
            [card_y + card_h - 0.012] * 2,
            lw=2.1,
            color=CARD_ACCENTS[i],
            solid_capstyle="round",
            clip_on=False,
        )

        ax.text(
            cx, 0.690,
            rf"$t_0={t0}$ s",
            ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=TEXT_COLOR,
        )

        # Longer windows are represented by a slightly longer waveform glyph.
        width = 0.085 + 0.0065 * min(t0, 10)
        synthetic_wave(
            cx - width / 2,
            cx + width / 2,
            0.565,
            0.018 + 0.0018 * min(t0, 10),
            8 + t0,
            ax,
            seed=17 + i,
        )

        # Station glyphs preserve the exact K used by each observation budget.
        if k <= 5:
            xs = np.linspace(cx - 0.052, cx + 0.052, k)
            ys = np.full(k, 0.455)
        else:
            n1 = int(np.ceil(k / 2))
            n2 = k - n1
            xs = np.concatenate(
                [
                    np.linspace(cx - 0.058, cx + 0.058, n1),
                    np.linspace(cx - 0.046, cx + 0.046, n2),
                ]
            )
            ys = np.concatenate(
                [
                    np.full(n1, 0.470),
                    np.full(n2, 0.414),
                ]
            )
        for sx, sy in zip(xs, ys):
            station_triangle(sx, sy, 0.0095, ax)

        ax.text(
            cx, 0.320,
            rf"$K={k}$",
            ha="center", va="center",
            fontsize=6.6, color="#344150",
        )

    strip = FancyBboxPatch(
        (0.10, 0.087), 0.80, 0.082,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor=LIGHT_GRAY, edgecolor="none",
    )
    ax.add_patch(strip)
    ax.text(
        0.50, 0.128,
        f"Common comparison core   {n_events} events · {n_groups} sequence groups",
        ha="center", va="center",
        fontsize=6.9, fontweight="bold", color=TEXT_COLOR,
    )
    ax.text(
        0.50, 0.035,
        "Both models are retrained and selected independently for each budget",
        ha="center", va="center", fontsize=6.2, color=MUTED,
    )


# -----------------------------------------------------------------------------
# Panel b: absolute high-motion-tail MAE
# -----------------------------------------------------------------------------


def plot_absolute_mae(ax_container, records):
    ax_container.axis("off")
    panel_letter(ax_container, "b", x=-0.025, y=1.015)
    ax_container.text(
        0.045, 0.995,
        "High-motion-tail MAE across budgets",
        fontsize=8.8, fontweight="bold", va="top", color=TEXT_COLOR,
    )

    # Two compact facets keep PGA and PGV directly comparable.
    ax_pga = ax_container.inset_axes([0.085, 0.175, 0.405, 0.645])
    ax_pgv = ax_container.inset_axes([0.565, 0.175, 0.405, 0.645])

    x = np.arange(len(BUDGETS))
    labels = [b[0] for b in BUDGETS]

    for ax, q, title in ((ax_pga, "pga", "PGA"), (ax_pgv, "pgv", "PGV")):
        base = np.array([r[f"{q}_base_tail_mae"] for r in records])
        final = np.array([r[f"{q}_final_tail_mae"] for r in records])

        # The narrow shaded gap makes the gain legible without another metric layer.
        ax.fill_between(
            x, final, base,
            color=FINAL_COLOR, alpha=0.075, linewidth=0, zorder=1,
        )
        for xx, y0, y1 in zip(x, base, final):
            ax.plot([xx, xx], [y0, y1], color="#C9D0D6", lw=0.65, zorder=1.5)

        ax.plot(
            x, base,
            marker="o", ms=4.1, lw=1.2,
            color=BASE_COLOR, linestyle=(0, (3.2, 2.2)),
            markeredgecolor="white", markeredgewidth=0.35,
            zorder=3,
        )
        ax.plot(
            x, final,
            marker="o", ms=4.2, lw=1.45,
            color=FINAL_COLOR,
            markeredgecolor="white", markeredgewidth=0.35,
            zorder=4,
        )

        # Numbers remain available but are kept small and away from the line segments.
        for xx, yy in zip(x, base):
            ax.text(
                xx, yy + 0.020, f"{yy:.3f}",
                ha="center", va="bottom", fontsize=5.6, color=BASE_COLOR,
            )
        for xx, yy in zip(x, final):
            ax.text(
                xx, yy - 0.027, f"{yy:.3f}",
                ha="center", va="top", fontsize=5.6, color=FINAL_COLOR,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(
            title,
            fontsize=7.8,
            fontweight="bold",
            y=0.95,  # 下移
            pad=0)
        ax.set_ylim(0.36, 0.96)
        style_axis(ax, "y")

    ax_pga.set_ylabel(r"High-motion-tail MAE (log$_{10}$ units)")
    ax_container.text(
        0.525, 0.075, "Observation budget",
        transform=ax_container.transAxes,
        ha="center", va="center", fontsize=7.2, color=TEXT_COLOR,
    )

    legend_handles = [
        Line2D([], [], color=BASE_COLOR, lw=1.2, ls=(0, (3.2, 2.2)), marker="o",
               ms=4.0, markeredgecolor="white", markeredgewidth=0.3,
               label="Cross-Attention"),
        Line2D([], [], color=FINAL_COLOR, lw=1.45, marker="o",
               ms=4.0, markeredgecolor="white", markeredgewidth=0.3,
               label="CA-URC"),
    ]
    ax_container.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.900),
        ncol=2,
        frameon=False,
        handlelength=1.9,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


# -----------------------------------------------------------------------------
# Panels c / d: horizontal paired forest plots
# -----------------------------------------------------------------------------


def forest_plot(
    ax,
    records,
    metric,
    ylabel,
    title,
    panel,
    percent_points=False,
):
    """Horizontal forest plot; the zero line becomes the visual decision boundary."""
    panel_title(ax, panel, title)

    # Keep the smallest budget at the top so information increases downward.
    y_center = np.arange(len(BUDGETS), dtype=float)[::-1]
    offset = 0.13
    qspec = [
        ("pga", PGA_COLOR, "o", +offset),
        ("pgv", PGV_COLOR, "s", -offset),
    ]

    if metric == "mae":
        all_low = np.array([
            r[f"{q}_delta_mae_low"] for r in records for q in ("pga", "pgv")
        ])
        xmin = min(-0.085, float(np.min(all_low)) - 0.006)
        xmax = 0.008
        x_label = r"$\Delta$ high-motion-tail MAE (CA-URC $-$ Base; log$_{10}$ units)"
        helper = "Lower error  ←"
    else:
        all_low = 100.0 * np.array([
            r[f"{q}_delta_u05_low"] for r in records for q in ("pga", "pgv")
        ])
        xmin = min(-18.0, float(np.min(all_low)) - 0.8)
        xmax = 0.8
        x_label = r"$\Delta$ high-motion-tail $U_{0.5}$ (percentage points)"
        helper = "Fewer severe underpredictions  ←"

    ax.axvspan(xmin, 0.0, color=NEGATIVE_ZONE, zorder=0)
    ax.axvline(0.0, color=ZERO, lw=0.9, linestyle=(0, (4, 3)), zorder=1)

    for q, color, marker, dy in qspec:
        if metric == "mae":
            xval = np.array([r[f"{q}_delta_mae"] for r in records])
            low = np.array([r[f"{q}_delta_mae_low"] for r in records])
            high = np.array([r[f"{q}_delta_mae_high"] for r in records])
        else:
            xval = 100.0 * np.array([r[f"{q}_delta_u05"] for r in records])
            low = 100.0 * np.array([r[f"{q}_delta_u05_low"] for r in records])
            high = 100.0 * np.array([r[f"{q}_delta_u05_high"] for r in records])

        xerr = np.vstack([xval - low, high - xval])
        yy = y_center + dy

        ax.errorbar(
            xval, yy,
            xerr=xerr,
            fmt=marker,
            ms=4.7,
            capsize=2.5,
            elinewidth=1.0,
            capthick=0.9,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            lw=0,
            label=q.upper(),
            zorder=4,
        )

        for i, (xx, yy_i) in enumerate(zip(xval, yy)):
            if metric == "mae":
                rel = abs(records[i][f"{q}_rel_mae_pct"])
                txt = f"−{rel:.1f}%"
            else:
                pp = abs(records[i][f"{q}_delta_u05_pp"])
                txt = f"−{pp:.1f} pp"

            # Offset above/below the marker rather than along the CI direction.
            yoff = 5.5 if q == "pga" else -6.0
            va = "bottom" if q == "pga" else "top"
            ax.annotate(
                txt,
                (xx, yy_i),
                xytext=(0, yoff),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=5.7, color=color,
                annotation_clip=False,
                zorder=5,
            )

    ax.set_yticks(y_center)
    ax.set_yticklabels([b[0] for b in BUDGETS])
    ax.set_ylabel("Observation budget")
    ax.set_xlabel(x_label)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(BUDGETS) - 0.45)
    style_axis(ax, "x")

    ax.text(
        0.985, 0.965, helper,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=6.0, color=MUTED,
    )

    # One compact quantity legend per panel keeps the meaning local after export/cropping.
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0.15,
        fontsize=6.2,
    )


# -----------------------------------------------------------------------------
# Build summary table
# -----------------------------------------------------------------------------

def build_summary(records):
    rows = []
    for (label, t0, k), r in zip(BUDGETS, records):
        rows.append(
            {
                "budget": label,
                "t0_sec": t0,
                "K": k,
                "n_common_events": r["n_events"],
                "n_sequence_groups": r["n_sequence_groups"],
                "pga_base_tail_mae": r["pga_base_tail_mae"],
                "pga_caurc_tail_mae": r["pga_final_tail_mae"],
                "pga_tail_mae_relative_change_percent": r["pga_rel_mae_pct"],
                "pga_tail_mae_delta": r["pga_delta_mae"],
                "pga_tail_mae_ci_low": r["pga_delta_mae_low"],
                "pga_tail_mae_ci_high": r["pga_delta_mae_high"],
                "pga_tail_u05_delta_pp": r["pga_delta_u05_pp"],
                "pga_tail_u05_ci_low_pp": 100 * r["pga_delta_u05_low"],
                "pga_tail_u05_ci_high_pp": 100 * r["pga_delta_u05_high"],
                "pgv_base_tail_mae": r["pgv_base_tail_mae"],
                "pgv_caurc_tail_mae": r["pgv_final_tail_mae"],
                "pgv_tail_mae_relative_change_percent": r["pgv_rel_mae_pct"],
                "pgv_tail_mae_delta": r["pgv_delta_mae"],
                "pgv_tail_mae_ci_low": r["pgv_delta_mae_low"],
                "pgv_tail_mae_ci_high": r["pgv_delta_mae_high"],
                "pgv_tail_u05_delta_pp": r["pgv_delta_u05_pp"],
                "pgv_tail_u05_ci_low_pp": 100 * r["pgv_delta_u05_low"],
                "pgv_tail_u05_ci_high_pp": 100 * r["pgv_delta_u05_high"],
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b33", default="caurc_observation_budget_minimal/t0_3s_k3/hierarchical_bootstrap_summary.csv", help="Bootstrap CSV for (3 s, 3)")
    p.add_argument("--b55", default="caurc_observation_budget_minimal/t0_5s_k5/hierarchical_bootstrap_summary.csv", help="Bootstrap CSV for (5 s, 5)")
    p.add_argument("--b105", default="caurc_observation_budget_minimal/t0_10s_k5/hierarchical_bootstrap_summary.csv", help="Bootstrap CSV for (10 s, 5)")
    p.add_argument("--b1010", default="caurc_observation_budget_minimal/t0_10s_k10/hierarchical_bootstrap_summary.csv", help="Bootstrap CSV for (10 s, 10)")
    p.add_argument("--out-dir", default="figures/nc_observation_budget")
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    setup_style()

    paths = [
        Path(args.b33),
        Path(args.b55),
        Path(args.b105),
        Path(args.b1010),
    ]
    records = [extract_budget(path) for path in paths]

    n_events = {r["n_events"] for r in records}
    n_groups = {r["n_sequence_groups"] for r in records}
    if len(n_events) != 1 or len(n_groups) != 1:
        raise RuntimeError(
            "The four files do not share the same common-event/common-sequence core: "
            f"events={sorted(n_events)}, groups={sorted(n_groups)}"
        )

    # manuscript safety: all plotted tail CI values must be finite
    for i, r in enumerate(records):
        for q in ("pga", "pgv"):
            for key in (
                f"{q}_delta_mae",
                f"{q}_delta_mae_low",
                f"{q}_delta_mae_high",
                f"{q}_delta_u05",
                f"{q}_delta_u05_low",
                f"{q}_delta_u05_high",
            ):
                if not np.isfinite(r[key]):
                    raise RuntimeError(f"Non-finite value in budget {BUDGETS[i][0]}: {key}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(records)
    summary.to_csv(out_dir / "Fig_observation_budget_summary.csv", index=False)

    # Full-width NC figure: 183 mm.  The slightly shorter canvas and tighter
    # inter-panel spacing produce a denser main-text figure without crowding labels.
    fig = plt.figure(figsize=(7.20, 6.55), facecolor="white")
    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[0.86, 1.0],
        width_ratios=[1.02, 1.08],
        left=0.082,
        right=0.985,
        bottom=0.092,
        top=0.970,
        wspace=0.22,
        hspace=0.11,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b_container = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_budget_design(ax_a, records[0]["n_events"], records[0]["n_sequence_groups"])
    plot_absolute_mae(ax_b_container, records)
    forest_plot(
        ax_c,
        records,
        metric="mae",
        ylabel=r"$\Delta$ Tail MAE (CA-URC − Base; log$_{10}$ units)",
        title="Tail-error reduction persists across budgets",
        panel="c",
    )
    forest_plot(
        ax_d,
        records,
        metric="u05",
        ylabel=r"$\Delta$ Tail $U_{0.5}$ (percentage points)",
        title="Severe underprediction falls across budgets",
        panel="d",
        percent_points=True,
    )

    # Compact figure-level protocol note.
    fig.text(
        0.53, 0.020,
        (
            "Paired 95% CIs from hierarchical sequence→event bootstrap; "
            f"all comparisons use the same {records[0]['n_events']}-event core."
        ),
        ha="center", va="bottom", fontsize=6.2, color=MUTED
    )

    outputs = {
        "png": out_dir / "Fig_observation_budget_robustness.png",
        "pdf": out_dir / "Fig_observation_budget_robustness.pdf",
        "svg": out_dir / "Fig_observation_budget_robustness.svg",
    }

    fig.savefig(outputs["png"], dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("=== Observation-budget figure generated ===")
    print(summary.to_string(index=False))
    print("\nOutputs:")
    for pth in outputs.values():
        print(pth.resolve())
    print((out_dir / "Fig_observation_budget_summary.csv").resolve())


if __name__ == "__main__":
    main()
