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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle


# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

# Bright but publication-safe palette
BASE_COLOR = "#2F80ED"      # Cross-Attention: cobalt blue
FINAL_COLOR = "#EB5A46"     # CA-URC: warm coral
PGA_COLOR = "#2F80ED"       # PGA: blue
PGV_COLOR = "#F2994A"       # PGV: amber-orange
TEXT_COLOR = "#17212B"
MUTED = "#667482"
GRID = "#E5EAF0"
ZERO = "#7D8996"
FRAME = "#3D4854"
LIGHT_BLUE = "#EAF4FF"
LIGHT_RED = "#FFF0EC"
LIGHT_GRAY = "#F5F7FA"

BUDGETS = [
    ("3s/3", 3, 3),
    ("5s/5", 5, 5),
    ("10s/5", 10, 5),
    ("10s/10", 10, 10),
]


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.7,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.82,
            "xtick.major.width": 0.72,
            "ytick.major.width": 0.72,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
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
    """Publication-style full box with light grid."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.82)
        spine.set_color(FRAME)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.78, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", top=False, right=False, pad=2.2)


def panel_letter(ax, letter, x=-0.09, y=1.045):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )




def add_panel_frame(ax, x0=0.025, y0=0.125, width=0.95, height=0.71):
    """Draw a crisp full frame around a schematic panel body."""
    frame = Rectangle(
        (x0, y0), width, height,
        transform=ax.transAxes,
        facecolor="white", edgecolor=FRAME, linewidth=0.82,
        zorder=-10, clip_on=False,
    )
    ax.add_patch(frame)


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
# Panel a: information-budget schematic
# -----------------------------------------------------------------------------

def synthetic_wave(x0, x1, y0, amp, cycles, ax, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(x0, x1, 240)
    t = np.linspace(0, 1, len(x))
    envelope = np.exp(-3.0 * np.abs(t - 0.38))
    signal = (
        np.sin(2 * np.pi * cycles * t)
        + 0.55 * np.sin(2 * np.pi * (cycles * 1.8) * t + 0.8)
        + 0.22 * rng.normal(size=len(t))
    )
    y = y0 + amp * envelope * signal
    ax.plot(x, y, color="#3E78A8", lw=0.85, alpha=0.95, clip_on=False)


def station_triangle(cx, cy, size, ax):
    pts = np.array(
        [
            [cx, cy + size],
            [cx - 0.80 * size, cy - 0.65 * size],
            [cx + 0.80 * size, cy - 0.65 * size],
        ]
    )
    ax.add_patch(Polygon(pts, closed=True, facecolor="#38556D", edgecolor="white", linewidth=0.25))


def plot_budget_design(ax, n_events, n_groups):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Match the vertical plotting body of panel b.
    add_panel_frame(ax, x0=0.025, y0=0.125, width=0.95, height=0.71)

    panel_letter(ax, "a", x=-0.015, y=1.01)
    ax.text(
        0.052, 0.995, "Observation-budget design",
        fontsize=9.0, fontweight="bold", va="top"
    )

    # Direction of increasing information budget.
    ax.text(
        0.50, 0.790, "Increasing observation budget",
        fontsize=6.9, ha="center", va="center", color=MUTED
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.11, 0.755), (0.89, 0.755),
            arrowstyle="-|>", mutation_scale=11.5,
            linewidth=1.0, color="#8A98A6"
        )
    )

    x_centers = [0.145, 0.380, 0.620, 0.855]
    card_w = 0.192
    card_h = 0.36
    card_y = 0.285

    for i, ((label, t0, k), cx) in enumerate(zip(BUDGETS, x_centers)):
        face = LIGHT_BLUE if i < 2 else LIGHT_RED
        accent = BASE_COLOR if i < 2 else FINAL_COLOR

        box = FancyBboxPatch(
            (cx - card_w / 2, card_y),
            card_w, card_h,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            facecolor=face, edgecolor="#CDD7E1", linewidth=0.75,
        )
        ax.add_patch(box)
        ax.plot(
            [cx - card_w * 0.37, cx + card_w * 0.37],
            [card_y + card_h - 0.017, card_y + card_h - 0.017],
            color=accent, lw=2.0, solid_capstyle="round", clip_on=False
        )

        ax.text(
            cx, card_y + card_h - 0.045, f"({t0} s, {k})",
            ha="center", va="center", fontsize=7.1, fontweight="bold"
        )

        width = 0.090 + 0.010 * min(t0, 10)
        synthetic_wave(
            cx - width / 2, cx + width / 2,
            card_y + 0.220,
            0.016 + 0.0017 * min(t0, 10),
            8 + t0, ax, seed=17 + i,
        )

        if k <= 5:
            xs = np.linspace(cx - 0.052, cx + 0.052, k)
            ys = np.full(k, card_y + 0.148)
        else:
            n1 = int(np.ceil(k / 2))
            n2 = k - n1
            xs = np.concatenate([
                np.linspace(cx - 0.058, cx + 0.058, n1),
                np.linspace(cx - 0.047, cx + 0.047, n2),
            ])
            ys = np.concatenate([
                np.full(n1, card_y + 0.157),
                np.full(n2, card_y + 0.108),
            ])

        for sx, sy in zip(xs, ys):
            station_triangle(sx, sy, 0.0095, ax)

        ax.text(
            cx, card_y + 0.056,
            rf"$t_0$ = {t0} s" + "\n" + rf"$K$ = {k}",
            ha="center", va="center", fontsize=6.4, color="#344150",
            linespacing=1.28
        )

    # Common comparison core inside the panel frame.
    strip = FancyBboxPatch(
        (0.11, 0.145), 0.78, 0.105,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=LIGHT_GRAY, edgecolor="#DCE3E9", linewidth=0.65
    )
    ax.add_patch(strip)
    ax.text(
        0.50, 0.205, "Common comparison core",
        ha="center", va="center", fontsize=6.7, fontweight="bold"
    )
    ax.text(
        0.50, 0.171, f"{n_events} events  ·  {n_groups} sequence groups",
        ha="center", va="center", fontsize=6.1, color="#344150"
    )

    # Put the methodological note below the framed body, mirroring panel b's x label.
    ax.text(
        0.50, 0.040,
        "Both models retrained and selected independently for each budget",
        ha="center", va="center", fontsize=5.9, color=MUTED
    )


# -----------------------------------------------------------------------------
# Panel b
# -----------------------------------------------------------------------------

def plot_absolute_mae(ax_container, records):
    ax_container.set_xticks([])
    ax_container.set_yticks([])
    for spine in ax_container.spines.values():
        spine.set_visible(False)

    panel_letter(ax_container, "b", x=-0.015, y=1.01)
    ax_container.text(
        0.050, 0.995, "High-motion-tail MAE across budgets",
        fontsize=9.0, fontweight="bold", va="top"
    )

    # Same vertical body as panel a: y = 0.12 ... 0.84.
    ax_pga = ax_container.inset_axes([0.080, 0.14, 0.405, 0.66])
    ax_pgv = ax_container.inset_axes([0.565, 0.14, 0.405, 0.66])

    x = np.arange(len(BUDGETS))
    labels = [b[0] for b in BUDGETS]

    for ax, q, title in ((ax_pga, "pga", "PGA"), (ax_pgv, "pgv", "PGV")):
        base = np.array([r[f"{q}_base_tail_mae"] for r in records])
        final = np.array([r[f"{q}_final_tail_mae"] for r in records])

        ax.plot(
            x, base, marker="o", ms=4.9, lw=1.45,
            color=BASE_COLOR, linestyle=(0, (3, 2)),
            label="Cross-Attention", zorder=3
        )
        ax.plot(
            x, final, marker="o", ms=4.9, lw=1.65,
            color=FINAL_COLOR, label="CA-URC", zorder=3
        )

        for xx, yy in zip(x, base):
            ax.annotate(
                f"{yy:.3f}", (xx, yy), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=5.7, color=BASE_COLOR
            )
        for xx, yy in zip(x, final):
            ax.annotate(
                f"{yy:.3f}", (xx, yy), xytext=(0, -7),
                textcoords="offset points", ha="center", va="top",
                fontsize=5.7, color=FINAL_COLOR
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=8.0, fontweight="bold", pad=4)
        ax.set_ylim(0.35, 0.98)
        ax.set_xlabel("Observation budget", labelpad=2)
        style_axis(ax, "y")

    ax_pga.set_ylabel(r"High-motion-tail MAE (log$_{10}$ units)")

    # Shared y scale: suppress duplicate PGV y labels to avoid crowding in the center.
    ax_pgv.tick_params(axis="y", labelleft=False)

    handles, labels_legend = ax_pgv.get_legend_handles_labels()
    ax_container.legend(
        handles, labels_legend,
        frameon=False, loc="upper left", bbox_to_anchor=(0.50, 0.925),
        ncol=2, handlelength=1.8, columnspacing=0.85,
        handletextpad=0.45, borderpad=0.1, fontsize=6.5
    )


# -----------------------------------------------------------------------------
# Panels c / d: forest-style effect plots
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
    """Horizontal forest plot: budgets on y, paired effects on x."""
    panel_letter(ax, panel, x=-0.075, y=1.045)

    y = np.arange(len(BUDGETS), dtype=float)
    offset = 0.12
    qspec = [
        ("pga", PGA_COLOR, "o", -offset),
        ("pgv", PGV_COLOR, "s", +offset),
    ]

    for q, color, marker, dy in qspec:
        if metric == "mae":
            val = np.array([r[f"{q}_delta_mae"] for r in records])
            low = np.array([r[f"{q}_delta_mae_low"] for r in records])
            high = np.array([r[f"{q}_delta_mae_high"] for r in records])
        else:
            val = 100.0 * np.array([r[f"{q}_delta_u05"] for r in records])
            low = 100.0 * np.array([r[f"{q}_delta_u05_low"] for r in records])
            high = 100.0 * np.array([r[f"{q}_delta_u05_high"] for r in records])

        xerr = np.vstack([val - low, high - val])
        yy = y + dy

        ax.errorbar(
            val, yy, xerr=xerr,
            fmt=marker, ms=5.0, capsize=2.7,
            elinewidth=1.15, capthick=1.05,
            color=color, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.55,
            lw=0, label=q.upper(), zorder=4,
        )

        for i, (xx, yi) in enumerate(zip(val, yy)):
            if metric == "mae":
                txt = f"−{abs(records[i][f'{q}_rel_mae_pct']):.1f}%"
            else:
                txt = f"−{abs(records[i][f'{q}_delta_u05_pp']):.1f} pp"

            # Separate PGA/PGV labels vertically; extra top margin prevents title collision.
            yoff = 8 if q == "pga" else -9
            ax.annotate(
                txt, (xx, yi), xytext=(0, yoff),
                textcoords="offset points", ha="center",
                va="bottom" if q == "pga" else "top",
                fontsize=5.7, color=color, zorder=5,
            )

    ax.axvline(0, color=ZERO, lw=0.95, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in BUDGETS])
    ax.set_ylim(len(BUDGETS) - 0.55, -0.68)  # extra breathing room above first row
    ax.set_xlabel(ylabel, labelpad=3)
    ax.set_ylabel("Observation budget", labelpad=4)
    ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold", pad=7)
    style_axis(ax, "x")

    # Compact directional cue in the empty upper-right region of each forest plot.
    cue = r"Lower error  $\leftarrow$" if metric == "mae" else r"Fewer severe cases  $\leftarrow$"
    ax.text(
        0.975, 0.958, cue, transform=ax.transAxes,
        ha="right", va="top", fontsize=5.5, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.88)
    )

    ax.legend(
        frameon=False, loc="lower left", ncol=2,
        handletextpad=0.35, columnspacing=0.85, borderaxespad=0.35
    )

    if metric == "mae":
        xmin = min(
            r[f"{q}_delta_mae_low"]
            for r in records for q in ("pga", "pgv")
        )
        ax.set_xlim(min(-0.085, xmin - 0.006), 0.008)
    else:
        xmin = 100.0 * min(
            r[f"{q}_delta_u05_low"]
            for r in records for q in ("pga", "pgv")
        )
        ax.set_xlim(min(-18.0, xmin - 1.2), 0.8)


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

    # Full-width NC figure: ~183 mm.
    fig = plt.figure(figsize=(7.20, 6.45), facecolor="white")
    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[0.86, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.072,
        right=0.987,
        bottom=0.095,
        top=0.970,
        wspace=0.16,
        hspace=0.18,
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
        ylabel=r"$\Delta$ high-motion-tail MAE (CA-URC − Base; log$_{10}$ units)",
        title="Tail-error reduction persists across budgets",
        panel="c",
    )
    forest_plot(
        ax_d,
        records,
        metric="u05",
        ylabel=r"$\Delta$ high-motion-tail $U_{0.5}$ (percentage points)",
        title="Severe underprediction falls across budgets",
        panel="d",
        percent_points=True,
    )

    # compact figure-level note, manuscript-like rather than a large banner
    fig.text(
        0.53, 0.024,
        "Paired 95% CIs from hierarchical sequence→event bootstrap; all comparisons use the same 102-event core.",
        ha="center", va="bottom", fontsize=6.0, color=MUTED
    )

    outputs = {
        "png": out_dir / "Fig_observation_budget_robustness.png",
        "pdf": out_dir / "Fig_observation_budget_robustness.pdf",
        "svg": out_dir / "Fig_observation_budget_robustness.svg",
    }

    fig.savefig(outputs["png"], dpi=args.dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    print("=== Observation-budget figure generated ===")
    print(summary.to_string(index=False))
    print("\nOutputs:")
    for pth in outputs.values():
        print(pth.resolve())
    print((out_dir / "Fig_observation_budget_summary.csv").resolve())


if __name__ == "__main__":
    main()
