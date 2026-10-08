#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation-budget figure with exactly aligned panel borders.

Reads the same four bootstrap-summary CSV files as the original script.
All numerical results and CI endpoints come from the CSV files, not an image.

Alignment:
    a.left == c.left; a.right == c.right
    b.PGA.left == d.left; b.PGV.right == d.right
    The two columns have equal total width; each row has aligned top/bottom edges.

Dependencies: numpy, pandas, matplotlib.
Relative input paths are checked against both the working and script directories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.ticker import MultipleLocator


# Original palette retained.
BASE_COLOR = "#2F80ED"
FINAL_COLOR = "#EB5A46"
PGA_COLOR = BASE_COLOR
PGV_COLOR = "#F2994A"
TEXT_COLOR = "#17212B"
MUTED = "#667482"
GRID = "#E5EAF0"
ZERO = "#7D8996"
FRAME = "#3D4854"
LIGHT_BLUE = "#EAF4FF"
LIGHT_RED = "#FFF0EC"
LIGHT_GRAY = "#F5F7FA"

BUDGETS = [("3s/3", 3, 3), ("5s/5", 5, 5),
           ("10s/5", 10, 5), ("10s/10", 10, 10)]
ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = "caurc_observation_budget_minimal"
STEM = "Fig_observation_budget_robustness_aligned"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": FRAME,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        # Do not let a global automatic-layout setting move the panel borders.
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
    })


def style_axis(ax, grid_axis: str | None = None) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.8)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.85)
    ax.tick_params(direction="out", top=False, right=False, pad=2.3)


def resolve_input(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n".join(f"  {item}" for item in dict.fromkeys(candidates))
    raise FileNotFoundError(
        f"Bootstrap CSV not found. Checked:\n{checked}\n"
        "Use --b33 / --b55 / --b105 / --b1010 to provide the actual paths."
    )


def read_bootstrap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "candidate", "reference", "quantity", "population", "metric",
        "candidate_value", "reference_value",
        "point_delta_candidate_minus_reference", "ci_lower", "ci_upper",
        "relative_change_percent", "delta_percentage_points",
        "n_paired_events", "n_sequence_groups_used",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df


def select_row(df: pd.DataFrame, quantity: str, metric: str,
               population: str = "high_motion_tail") -> pd.Series:
    mask = (
        df["quantity"].astype(str).eq(quantity)
        & df["metric"].astype(str).eq(metric)
        & df["population"].astype(str).eq(population)
    )
    selected = df.loc[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one {quantity}/{population}/{metric} row; "
            f"found {len(selected)}. Check the candidate/reference comparison."
        )
    return selected.iloc[0]


def extract_budget(path: Path) -> dict:
    df = read_bootstrap(path)
    overall = select_row(df, "pga", "mae", population="overall")
    out = {
        "n_events": int(overall["n_paired_events"]),
        "n_sequence_groups": int(overall["n_sequence_groups_used"]),
        "candidate": str(overall["candidate"]),
        "reference": str(overall["reference"]),
    }
    for q in ("pga", "pgv"):
        mae = select_row(df, q, "mae")
        u05 = select_row(df, q, "under05")
        for row in (mae, u05):
            if (str(row["candidate"]), str(row["reference"])) != (
                out["candidate"], out["reference"]
            ):
                raise ValueError(f"{path}: inconsistent candidate/reference rows.")
        for short, row in (("mae", mae), ("u05", u05)):
            out[f"{q}_base_tail_{short}"] = float(row["reference_value"])
            out[f"{q}_final_tail_{short}"] = float(row["candidate_value"])
            out[f"{q}_delta_{short}"] = float(row["point_delta_candidate_minus_reference"])
            out[f"{q}_delta_{short}_low"] = float(row["ci_lower"])
            out[f"{q}_delta_{short}_high"] = float(row["ci_upper"])
        out[f"{q}_rel_mae_pct"] = float(mae["relative_change_percent"])
        out[f"{q}_delta_u05_pp"] = float(u05["delta_percentage_points"])
    return out


def validate_records(records: list[dict]) -> None:
    if len(records) != len(BUDGETS):
        raise ValueError("Exactly four observation-budget summaries are required.")
    cores = {(r["n_events"], r["n_sequence_groups"]) for r in records}
    if len(cores) != 1:
        raise ValueError(f"Common-core counts differ across budgets: {sorted(cores)}")
    # Equal counts do not independently verify identical event/group membership.
    # The locked evaluation audit remains the authority for that assertion.
    for (label, _, _), record in zip(BUDGETS, records):
        if record["n_events"] <= 0 or record["n_sequence_groups"] <= 0:
            raise ValueError(f"{label}: common-core counts must be positive.")
        for key, value in record.items():
            if key.startswith(("pga_", "pgv_")) and not np.isfinite(value):
                raise ValueError(f"{label}: non-finite {key}={value}")
        for q in ("pga", "pgv"):
            for metric in ("mae", "u05"):
                lo = record[f"{q}_delta_{metric}_low"]
                hi = record[f"{q}_delta_{metric}_high"]
                if lo > hi:
                    raise ValueError(f"{label}/{q}/{metric}: CI lower > upper.")
            if min(record[f"{q}_base_tail_mae"], record[f"{q}_final_tail_mae"]) < 0:
                raise ValueError(f"{label}/{q}: MAE must be nonnegative.")
            for model in ("base", "final"):
                value = record[f"{q}_{model}_tail_u05"]
                if not 0 <= value <= 1:
                    raise ValueError(
                        f"{label}/{q}: under05 must be a fraction in [0, 1], "
                        "not an already percentage-scaled value."
                    )
            if not np.isclose(record[f"{q}_delta_u05"] * 100,
                              record[f"{q}_delta_u05_pp"], atol=0.051, rtol=1e-5):
                raise ValueError(f"{label}/{q}: under05 delta and percentage points disagree.")


def draw_wave(ax, left: float, right: float, center_y: float,
              height: float, cycles: float, seed: int) -> None:
    """Decorative waveform only; not an experimental observation."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 240)
    signal = np.exp(-3.8 * np.abs(t - 0.40)) * (
        np.sin(2 * np.pi * cycles * t)
        + 0.40 * np.sin(2 * np.pi * cycles * 1.8 * t + 0.8)
        + 0.08 * rng.normal(size=t.size)
    )
    signal /= np.max(np.abs(signal))
    ax.plot(np.linspace(left, right, t.size), center_y + height * signal,
            color="#3E78A8", lw=0.85, solid_capstyle="round", clip_on=True)


def station_triangle(ax, cx: float, cy: float, size: float = 0.012) -> None:
    vertices = [(cx, cy + size), (cx - 0.75 * size, cy - 0.65 * size),
                (cx + 0.75 * size, cy - 0.65 * size)]
    ax.add_patch(Polygon(vertices, closed=True, facecolor="#38556D",
                         edgecolor="white", linewidth=0.25))


def plot_budget_design(ax, n_events: int, n_groups: int) -> None:
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    style_axis(ax)
    # The actual axes spines are the schematic frame: no inset rectangle.
    ax.text(0.50, 0.934, "Increasing observation budget", ha="center", va="center",
            fontsize=6.8, color=MUTED)
    ax.add_patch(FancyArrowPatch((0.10, 0.883), (0.90, 0.883),
                                arrowstyle="-|>", mutation_scale=11,
                                color="#8A98A6", linewidth=1.0))

    margin, gap = 0.026, 0.030
    card_w = (1 - 2 * margin - 3 * gap) / 4
    card_y, card_h = 0.263, 0.534
    for i, (_, t0, k) in enumerate(BUDGETS):
        left = margin + i * (card_w + gap)
        cx = left + card_w / 2
        ax.add_patch(FancyBboxPatch(
            (left, card_y), card_w, card_h,
            boxstyle="round,pad=0.003,rounding_size=0.015",
            facecolor=LIGHT_BLUE if i < 2 else LIGHT_RED,
            edgecolor="#CDD7E1", linewidth=0.75,
        ))
        ax.plot([left + 0.030, left + card_w - 0.030], [0.771, 0.771],
                color=BASE_COLOR if i < 2 else FINAL_COLOR,
                lw=2.0, solid_capstyle="round")
        ax.text(cx, 0.718, f"({t0} s, {k})", ha="center", va="center",
                fontsize=7.0, fontweight="bold")
        wave_w = card_w * (0.62 + 0.018 * t0)
        draw_wave(ax, cx - wave_w / 2, cx + wave_w / 2, 0.591,
                  0.049 + 0.0011 * t0, 8 + t0, seed=17 + t0)
        if k <= 5:
            station_rows = [(k, 0.449)]
        else:
            station_rows = [(5, 0.468), (5, 0.409)]
        for count, cy in station_rows:
            for sx in np.linspace(cx - card_w * 0.29, cx + card_w * 0.29, count):
                station_triangle(ax, sx, cy)
        ax.text(cx, 0.325, rf"$t_0$ = {t0} s" + "\n" + rf"$K$ = {k}",
                ha="center", va="center", fontsize=6.4,
                color="#344150", linespacing=1.18)

    ax.add_patch(FancyBboxPatch(
        (0.06, 0.043), 0.88, 0.151,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=LIGHT_GRAY, edgecolor="#DCE3E9", linewidth=0.65,
    ))
    ax.text(0.50, 0.143, "Common comparison core", ha="center", va="center",
            fontsize=7.0, fontweight="bold")
    ax.text(0.50, 0.084, f"{n_events} events  ·  {n_groups} sequence groups",
            ha="center", va="center", fontsize=6.3, color="#344150")


def plot_absolute_mae(ax_pga, ax_pgv, records: list[dict]) -> None:
    x = np.arange(len(BUDGETS))
    all_values = [r[f"{q}_{model}_tail_mae"] for r in records
                  for q in ("pga", "pgv") for model in ("base", "final")]
    limits = (max(0.0, min(0.35, min(all_values) - 0.07)),
              max(0.98, max(all_values) + 0.07))
    for ax, q in ((ax_pga, "pga"), (ax_pgv, "pgv")):
        for model, color, linestyle, label, offset in (
            ("base", BASE_COLOR, (0, (3, 2)), "Cross-Attention", 6),
            ("final", FINAL_COLOR, "-", "CA-URC", -6),
        ):
            values = np.array([r[f"{q}_{model}_tail_mae"] for r in records])
            ax.plot(x, values, marker="o", markersize=4.8, lw=1.5,
                    color=color, linestyle=linestyle, label=label,
                    markeredgecolor="white", markeredgewidth=0.35, zorder=3)
            for xx, value in zip(x, values):
                ax.annotate(f"{value:.3f}", (xx, value), xytext=(0, offset),
                            textcoords="offset points", ha="center",
                            va="bottom" if offset > 0 else "top",
                            color=color, fontsize=6.0, zorder=4,
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.9, pad=0.22))
        ax.set_xlim(-0.43, len(BUDGETS) - 0.57)
        ax.set_ylim(*limits)
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in BUDGETS])
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_title(q.upper(), fontsize=8.0, fontweight="bold", pad=5)
        ax.set_xlabel("Observation budget", labelpad=3)
        style_axis(ax, "y")
    ax_pga.set_ylabel(r"High-motion-tail MAE (log$_{10}$ units)", labelpad=4)
    ax_pgv.tick_params(axis="y", labelleft=False)


def signed_label(value: float, suffix: str) -> str:
    # Preserve the real sign; never force a positive effect to look beneficial.
    if abs(value) < 0.05:
        return "0.0" + suffix
    return f"{value:+.1f}".replace("-", "−") + suffix


def forest_plot(ax, records: list[dict], metric: str, row_bands: bool) -> None:
    y = np.arange(len(BUDGETS), dtype=float)
    scale = 1.0 if metric == "mae" else 100.0
    if row_bands:
        for i in (0, 2):
            ax.axhspan(i - 0.39, i + 0.39, facecolor=LIGHT_GRAY,
                       edgecolor="none", alpha=0.72, zorder=0)
    extrema = [0.0]
    for q, color, marker, offset in (
        ("pga", PGA_COLOR, "o", -0.12),
        ("pgv", PGV_COLOR, "s", +0.12),
    ):
        values = scale * np.array([r[f"{q}_delta_{metric}"] for r in records])
        lower = scale * np.array([r[f"{q}_delta_{metric}_low"] for r in records])
        upper = scale * np.array([r[f"{q}_delta_{metric}_high"] for r in records])
        yy = y + offset
        extrema.extend(np.concatenate([values, lower, upper]).tolist())
        # Draw the actual CI endpoints without clipping or modifying their extent.
        # This also works when a percentile interval does not contain its estimate.
        ax.hlines(yy, lower, upper, color=color, linewidth=1.1, zorder=3)
        for endpoint in (lower, upper):
            ax.plot(endpoint, yy, linestyle="none", marker="|", markersize=5.4,
                    markeredgewidth=0.9, color=color, zorder=3)
        ax.plot(values, yy, linestyle="none", marker=marker, markersize=4.9,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.55, zorder=4)
        for i, (value, yvalue) in enumerate(zip(values, yy)):
            number = (records[i][f"{q}_rel_mae_pct"] if metric == "mae"
                      else records[i][f"{q}_delta_u05_pp"])
            text = signed_label(number, "%" if metric == "mae" else " pp")
            ax.annotate(text, (value, yvalue), xytext=(0, 6 if q == "pga" else -7),
                        textcoords="offset points", ha="center",
                        va="bottom" if q == "pga" else "top",
                        fontsize=6.0, color=color, zorder=5)

    if metric == "mae":
        ax.set_xlim(min(-0.085, min(extrema) - 0.006),
                    max(0.008, max(extrema) + 0.006))
        ax.xaxis.set_major_locator(MultipleLocator(0.02))
        xlabel = r"$\Delta$ high-motion-tail MAE (CA-URC − Base; log$_{10}$ units)"
        cue = "Lower error  ←"
    else:
        ax.set_xlim(min(-18.0, min(extrema) - 1.2),
                    max(0.8, max(extrema) + 0.6))
        ax.xaxis.set_major_locator(MultipleLocator(2.5))
        xlabel = r"$\Delta$ high-motion-tail $U_{0.5}$ (percentage points)"
        cue = "Fewer severe cases  ←"
    ax.axvline(0, color=ZERO, lw=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in BUDGETS])
    ax.set_ylim(3.60, -0.88)
    ax.set_xlabel(xlabel, labelpad=4, fontsize=7.0)
    ax.set_ylabel("Observation budget", labelpad=4)
    style_axis(ax, "x")
    ax.text(0.974, 0.976, cue, transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.88))
    handles = [Line2D([], [], color=color, marker=marker, lw=1.0,
                      markersize=4.5, markeredgecolor="white", markeredgewidth=0.45,
                      label=q) for q, color, marker in
               (("PGA", PGA_COLOR, "o"), ("PGV", PGV_COLOR, "s"))]
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=False,
              handlelength=1.3, handletextpad=0.4, columnspacing=0.9,
              borderaxespad=0.5, fontsize=6.3)


def build_summary(records: list[dict]) -> pd.DataFrame:
    rows = []
    for (label, t0, k), r in zip(BUDGETS, records):
        row = {"budget": label, "t0_sec": t0, "K": k,
               "n_common_events": r["n_events"],
               "n_sequence_groups": r["n_sequence_groups"]}
        for q in ("pga", "pgv"):
            row.update({
                f"{q}_base_tail_mae": r[f"{q}_base_tail_mae"],
                f"{q}_caurc_tail_mae": r[f"{q}_final_tail_mae"],
                f"{q}_tail_mae_relative_change_percent": r[f"{q}_rel_mae_pct"],
                f"{q}_tail_mae_delta": r[f"{q}_delta_mae"],
                f"{q}_tail_mae_ci_low": r[f"{q}_delta_mae_low"],
                f"{q}_tail_mae_ci_high": r[f"{q}_delta_mae_high"],
                f"{q}_tail_u05_delta_pp": r[f"{q}_delta_u05_pp"],
                f"{q}_tail_u05_ci_low_pp": 100 * r[f"{q}_delta_u05_low"],
                f"{q}_tail_u05_ci_high_pp": 100 * r[f"{q}_delta_u05_high"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def panel_header(fig, ax, letter: str, title: str, baseline: float) -> None:
    left = ax.get_position().x0
    fig.text(left - 0.034, baseline, letter, fontsize=10.8, fontweight="bold",
             ha="left", va="baseline", color="black")
    fig.text(left, baseline, title, fontsize=8.2, fontweight="bold",
             ha="left", va="baseline")


def check_alignment(fig, axes: dict) -> pd.DataFrame:
    fig.canvas.draw()
    bounds = {key: ax.get_position().frozen() for key, ax in axes.items()}
    pairs = [
        (bounds["a"].x0, bounds["c"].x0, "a/c left"),
        (bounds["a"].x1, bounds["c"].x1, "a/c right"),
        (bounds["b_PGA"].x0, bounds["d"].x0, "b/d left"),
        (bounds["b_PGV"].x1, bounds["d"].x1, "b/d right"),
        (bounds["a"].width, bounds["d"].width, "column widths"),
    ]
    for key in ("b_PGA", "b_PGV"):
        pairs.extend([(bounds["a"].y0, bounds[key].y0, f"a/{key} bottom"),
                      (bounds["a"].y1, bounds[key].y1, f"a/{key} top")])
    pairs.extend([(bounds["c"].y0, bounds["d"].y0, "c/d bottom"),
                  (bounds["c"].y1, bounds["d"].y1, "c/d top")])
    for first, second, name in pairs:
        if not np.isclose(first, second, atol=1e-10, rtol=0):
            raise RuntimeError(f"Alignment check failed: {name}: {first} != {second}")
    return pd.DataFrame([
        {"panel": key, "left": box.x0, "right": box.x1,
         "bottom": box.y0, "top": box.y1, "width": box.width, "height": box.height}
        for key, box in bounds.items()
    ])


def make_figure(records: list[dict], row_bands: bool = True):
    fig = plt.figure(figsize=(7.20, 6.30), facecolor="white")
    # One shared grid fixes the true panel borders. The middle row is whitespace.
    grid = GridSpec(
        3, 2, figure=fig, left=0.086, right=0.987, bottom=0.112, top=0.872,
        width_ratios=[1, 1], height_ratios=[0.290, 0.118, 0.352],
        wspace=0.28, hspace=0,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[2, 0])
    ax_d = fig.add_subplot(grid[2, 1])
    # b spans the complete right-hand column, with no outer inset padding.
    b_grid = grid[0, 1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.23)
    ax_pga = fig.add_subplot(b_grid[0, 0])
    ax_pgv = fig.add_subplot(b_grid[0, 1], sharey=ax_pga)

    plot_budget_design(ax_a, records[0]["n_events"], records[0]["n_sequence_groups"])
    plot_absolute_mae(ax_pga, ax_pgv, records)
    forest_plot(ax_c, records, "mae", row_bands)
    forest_plot(ax_d, records, "u05", row_bands)

    top_baseline = 0.952
    lower_baseline = ax_c.get_position().y1 + 0.028
    panel_header(fig, ax_a, "a", "Observation-budget design", top_baseline)
    panel_header(fig, ax_pga, "b", "High-motion-tail MAE across budgets", top_baseline)
    panel_header(fig, ax_c, "c", "Tail-error reduction persists across budgets", lower_baseline)
    panel_header(fig, ax_d, "d", "Severe underprediction falls across budgets", lower_baseline)

    handles, labels = ax_pga.get_legend_handles_labels()
    right_box = ax_d.get_position()
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(right_box.x1, 0.918), bbox_transform=fig.transFigure,
               ncol=2, frameon=False, handlelength=1.9, handletextpad=0.45,
               columnspacing=1.0, borderaxespad=0, fontsize=6.5)
    a_box = ax_a.get_position()
    fig.text((a_box.x0 + a_box.x1) / 2, a_box.y0 - 0.049,
             "Both models retrained and selected independently for each budget",
             ha="center", va="center", fontsize=6.0, color=MUTED)
    fig.text(0.53, 0.023,
             "Paired 95% CIs from hierarchical sequence→event bootstrap; "
             f"all comparisons use the same {records[0]['n_events']}-event core.",
             ha="center", va="bottom", fontsize=6.0, color=MUTED)
    axes = {"a": ax_a, "b_PGA": ax_pga, "b_PGV": ax_pgv, "c": ax_c, "d": ax_d}
    return fig, check_alignment(fig, axes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, folder in (("b33", "t0_3s_k3"), ("b55", "t0_5s_k5"),
                         ("b105", "t0_10s_k5"), ("b1010", "t0_10s_k10")):
        parser.add_argument(f"--{flag}", default=f"{DEFAULT_ROOT}/{folder}/hierarchical_bootstrap_summary.csv")
    parser.add_argument("--out-dir", default="figures/nc_observation_budget_aligned")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-row-bands", action="store_true",
                        help="Use white forest-plot backgrounds without alternating row bands.")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    paths = [resolve_input(getattr(args, flag)) for flag in ("b33", "b55", "b105", "b1010")]
    records = [extract_budget(path) for path in paths]
    validate_records(records)
    setup_style()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, alignment = make_figure(records, row_bands=not args.no_row_bands)
    summary = build_summary(records)
    summary_path = out_dir / "Fig_observation_budget_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    audit_path = out_dir / "Fig_observation_budget_alignment.csv"
    alignment.to_csv(audit_path, index=False, encoding="utf-8-sig")

    outputs = []
    try:
        for suffix in ("png", "pdf", "svg"):
            path = out_dir / f"{STEM}.{suffix}"
            fig.savefig(path, dpi=args.dpi, facecolor="white",
                        bbox_inches="tight", pad_inches=0.04)
            outputs.append(path)
    finally:
        plt.close(fig)
    print("=== Observation-budget figure generated; border alignment PASS ===")
    print(summary.to_string(index=False))
    print("\nPanel coordinates (figure fractions):")
    print(alignment.to_string(index=False))
    print("\nOutputs:")
    for path in outputs + [summary_path, audit_path]:
        print(path.resolve())


if __name__ == "__main__":
    main()
