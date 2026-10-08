from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent
SUMMARY = pd.read_csv(HERE / "protocol_tail_summary.csv")
EVENTS = pd.read_csv(HERE / "event_level_tail_deltas_all_protocols.csv")
OUT_PREFIX = HERE / "Fig7_main4_transportability_final"

# ============================================================
# Global settings
# ============================================================
ORDER = ["Grouped", "2021–2024", "Unseen stations", "Ridgecrest"]

# Restrained, Nature Communications-like palette
PGA_COLOR = "#2C7FB8"
PGV_COLOR = "#E8762D"
RED = "#C83E3E"
DARK = "#2A2F33"
MUTED = "#727A82"
GRID = "#DCE2E7"
ROW = "#F5F7F9"

SCENARIO_COLORS = {
    "Grouped": "#3E6FB0",
    "2021–2024": "#7293B8",
    "Unseen stations": "#9A6D91",
    "Ridgecrest": "#A86C3F",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.4,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.7,
    "ytick.labelsize": 6.7,
    "axes.linewidth": 0.72,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ============================================================
# Utilities
# ============================================================
def style_axis(ax, grid="both"):
    """Full-border Nature Communications-like axes."""
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.72)
        ax.spines[side].set_color("#34393E")

    ax.tick_params(
        top=False,
        right=False,
        direction="out",
        width=0.65,
        length=3,
    )

    ax.set_axisbelow(True)

    if grid == "x":
        ax.grid(axis="x", color=GRID, lw=0.45, alpha=0.58)
    elif grid == "y":
        ax.grid(axis="y", color=GRID, lw=0.45, alpha=0.58)
    elif grid == "both":
        ax.grid(color=GRID, lw=0.45, alpha=0.42)


def panel_header(ax, letter, title):
    """Panel letter + aligned title."""
    y = 1.03
    ax.text(
        -0.12, y, letter,
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        0.00, y, title,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def half_violin(ax, values, position, color, width=0.70):
    """Half violin for event-level distribution."""
    vp = ax.violinplot(
        [values],
        positions=[position],
        vert=False,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=220,
        bw_method=0.28,
    )
    body = vp["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.18)

    verts = body.get_paths()[0].vertices
    verts[:, 1] = np.maximum(verts[:, 1], position)


# ============================================================
# Panel a
# ============================================================
def panel_a_identity(ax):
    """
    Identity-line transportability map.
    NOTE:
    - Do NOT use ax.set_aspect("equal"), otherwise panel size
      will differ visually from the other panels.
    - We instead rely on identical GridSpec cells + box_aspect.
    """
    lo, hi = 0.52, 1.08

    xx = np.linspace(lo, hi, 200)
    ax.fill_between(xx, lo, xx, color=RED, alpha=0.035, zorder=-3)
    ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=0.85, color=MUTED, zorder=1)

    # Fine-tuned label placement for minimal overlap
    label_offsets = {
        "Grouped": (6, -8),
        "2021–2024": (8, 8),
        "Unseen stations": (-10, 9),
        "Ridgecrest": (6, -11),
    }

    for scenario in ORDER:
        d = SUMMARY[SUMMARY["scenario_label"].eq(scenario)].set_index("quantity")
        pts = []

        for q, marker in [("PGA", "o"), ("PGV", "s")]:
            r = d.loc[q]
            c = SCENARIO_COLORS[scenario]

            ax.scatter(
                r["tail_mae_base"],
                r["tail_mae_final"],
                s=36,
                marker=marker,
                color=c,
                edgecolor="white",
                lw=0.6,
                zorder=4,
            )
            pts.append((r["tail_mae_base"], r["tail_mae_final"]))

        # Subtle connector between PGA and PGV for same protocol
        ax.plot(
            [pts[0][0], pts[1][0]],
            [pts[0][1], pts[1][1]],
            color=SCENARIO_COLORS[scenario],
            lw=0.65,
            alpha=0.30,
            zorder=2,
        )

        mx = (pts[0][0] + pts[1][0]) / 2.0
        my = (pts[0][1] + pts[1][1]) / 2.0
        dx, dy = label_offsets[scenario]

        ax.annotate(
            scenario,
            (mx, my),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.55,
            color=DARK,
            ha="right" if dx < 0 else "left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.16",
                fc="white",
                ec="none",
                alpha=0.78,
            ),
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Cross-Attention Base tail MAE")
    ax.set_ylabel("CA-URC tail MAE")

    ax.text(
        0.03, 0.94,
        "below identity = CA-URC better",
        transform=ax.transAxes,
        fontsize=5.5,
        color=RED,
        va="top",
    )

    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                marker="o",
                ls="None",
                color=DARK,
                markerfacecolor="white",
                markersize=5,
                label="PGA",
            ),
            Line2D(
                [0], [0],
                marker="s",
                ls="None",
                color=DARK,
                markerfacecolor="white",
                markersize=5,
                label="PGV",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="lower right",
        fontsize=5.9,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    panel_header(ax, "a", "All distribution shifts remain below the identity line")
    style_axis(ax)


# ============================================================
# Panels b / c
# ============================================================
def panel_event_distribution(ax, quantity, letter):
    """
    Event-level tail-MAE distribution.
    quantity: 'PGA' or 'PGV'
    """
    rng = np.random.default_rng(20260906 + (0 if quantity == "PGA" else 101))
    positions = np.arange(len(ORDER))

    for i, scenario in enumerate(ORDER):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color=ROW, zorder=-4)

        vals = EVENTS[
            (EVENTS["quantity"] == quantity) &
            (EVENTS["scenario_label"] == scenario)
        ]["delta_mae"].to_numpy(float)

        c = SCENARIO_COLORS[scenario]

        half_violin(ax, vals, i, c)

        jitter = rng.uniform(-0.27, -0.08, len(vals))
        ax.scatter(
            vals,
            i + jitter,
            s=7.5,
            color=c,
            alpha=0.18,
            edgecolor="none",
            zorder=2,
        )

        # Protocol-level saved CI from summary
        r = SUMMARY[
            (SUMMARY["quantity"] == quantity) &
            (SUMMARY["scenario_label"] == scenario)
        ].iloc[0]

        ax.plot(
            [r["delta_mae_lo"], r["delta_mae_hi"]],
            [i, i],
            color=c,
            lw=2.45,
            solid_capstyle="round",
            zorder=5,
        )

        ax.scatter(
            r["delta_mae"],
            i,
            s=36,
            color=c,
            edgecolor="white",
            lw=0.6,
            zorder=6,
        )

        improved = 100.0 * np.mean(vals < 0)
        ax.text(
            0.985, i,
            f"{improved:.1f}%  (n={len(vals)})",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.7,
            color=c,
            fontweight="bold" if improved >= 95 else "normal",
        )

    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(positions, ORDER)
    ax.invert_yaxis()
    ax.set_xlabel(r"Event-level $\Delta$ tail MAE (CA-URC − Base)")

    ax.text(
        0.02, 0.04,
        "← improvement",
        transform=ax.transAxes,
        fontsize=5.5,
        color=MUTED,
    )

    title = f"{quantity} gains persist across individual earthquakes"
    panel_header(ax, letter, title)
    style_axis(ax, grid="x")


# ============================================================
# Panel d
# ============================================================
def panel_d_joint_risk(ax):
    """
    Joint risk plane:
    x = delta tail MAE
    y = delta U0.5 percentage points
    """
    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))

    for scenario in ORDER:
        d = SUMMARY[SUMMARY["scenario_label"].eq(scenario)].set_index("quantity")
        pts = []

        for q, marker in [("PGA", "o"), ("PGV", "s")]:
            r = d.loc[q]
            x = r["delta_mae"]
            y = 100.0 * r["delta_u05"]   # percentage points
            c = SCENARIO_COLORS[scenario]

            ax.scatter(
                x, y,
                s=36,
                marker=marker,
                color=c,
                edgecolor="white",
                lw=0.6,
                zorder=4,
            )
            pts.append((x, y))

        ax.plot(
            [pts[0][0], pts[1][0]],
            [pts[0][1], pts[1][1]],
            color=SCENARIO_COLORS[scenario],
            lw=0.65,
            alpha=0.30,
            zorder=2,
        )

        mx = (pts[0][0] + pts[1][0]) / 2.0
        my = (pts[0][1] + pts[1][1]) / 2.0

        ax.annotate(
            scenario,
            (mx, my),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=5.55,
            color=DARK,
            va="center",
        )

    ax.set_xlim(-0.12, 0.005)
    ax.set_ylim(-23, 1)
    ax.set_xlabel(r"$\Delta$ tail MAE")
    ax.set_ylabel(r"$\Delta U_{0.5}$ (percentage points)")

    ax.text(
        0.02, 0.04,
        "lower-left = joint risk reduction",
        transform=ax.transAxes,
        fontsize=5.45,
        color=RED,
    )

    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                marker="o",
                ls="None",
                color=DARK,
                markerfacecolor="white",
                markersize=5,
                label="PGA",
            ),
            Line2D(
                [0], [0],
                marker="s",
                ls="None",
                color=DARK,
                markerfacecolor="white",
                markersize=5,
                label="PGV",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=5.9,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    panel_header(ax, "d", "MAE and severe-underprediction risks improve together")
    style_axis(ax)


# ============================================================
# Main 2 × 2 figure
# Row 1: a, d
# Row 2: b, c
# Critical:
# - All panels use the same GridSpec slots
# - All panels use the same box_aspect
# - No bbox_inches='tight' when saving
# ============================================================
def main():
    width = 183 / 25.4
    height = 150 / 25.4

    fig = plt.figure(figsize=(width, height))

    gs = fig.add_gridspec(
        2, 2,
        left=0.09,
        right=0.985,
        bottom=0.09,
        top=0.96,
        wspace=0.28,
        hspace=0.34,
    )

    axs = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ])

    # Ensure all panel outer boxes are visually identical
    for ax in axs.flat:
        ax.set_box_aspect(0.82)

    # First row
    panel_a_identity(axs[0, 0])
    panel_d_joint_risk(axs[0, 1])

    # Second row
    panel_event_distribution(axs[1, 0], "PGA", "b")
    panel_event_distribution(axs[1, 1], "PGV", "c")

    # Save WITHOUT bbox_inches='tight' to preserve panel-size consistency
    for ext in ("png", "pdf", "svg"):
        kwargs = {"facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(f"{OUT_PREFIX}.{ext}", **kwargs)

    plt.close(fig)
    print("Saved:", OUT_PREFIX)


if __name__ == "__main__":
    main()