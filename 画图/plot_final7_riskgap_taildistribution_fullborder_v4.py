from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SUMMARY_CSV = HERE / "final7_summary.csv"
EVENT_CSV = HERE / "final7_tail_event_mae.csv"

OUT_PREFIX = HERE / "Fig_final7_riskgap_taildistribution_final"

N_BOOT = 10000
RANDOM_SEED = 20260906
BOOTSTRAP_MODE = "sequence"

summary = pd.read_csv(SUMMARY_CSV)
event = pd.read_csv(EVENT_CSV)

METHOD_ORDER = [
    "Nearest observed",
    "IDW observed",
    "PLUM-like",
    "Graph",
    "Tail-weighted Attention",
    "Cross-Attention Base",
    "CA-URC",
]

COLORS = {
    "Nearest observed": "#8A8D91",
    "IDW observed": "#A1A4A8",
    "PLUM-like": "#6F7378",
    "Graph": "#6BAED6",
    "Tail-weighted Attention": "#3182BD",
    "Cross-Attention Base": "#08519C",
    "CA-URC": "#C51B1D",
}

Y_POS = {
    "Nearest observed": 0,
    "IDW observed": 1,
    "PLUM-like": 2,
    "Graph": 4,
    "Tail-weighted Attention": 5,
    "Cross-Attention Base": 6,
    "CA-URC": 8,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 9.0,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# General style
# ============================================================

def apply_full_box(ax):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("#2A2F34")


# ============================================================
# Panel letter + title
#
# 修改点：
# a/b/c/d 与标题使用相同 y 坐标和相同 vertical alignment，
# 保证四个 panel 的字母与标题完全在同一水平线上。
# ============================================================

def add_panel_header(ax, letter, title):
    header_y = 1.025

    ax.text(
        -0.115,
        header_y,
        letter,
        transform=ax.transAxes,
        fontsize=14.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )

    ax.text(
        0.0,
        header_y,
        title,
        transform=ax.transAxes,
        fontsize=13.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
        clip_on=False,
    )


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_mean_ci(
    values,
    groups=None,
    mode=BOOTSTRAP_MODE,
    n_boot=N_BOOT,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)

    if np.allclose(values, values[0]):
        return (
            float(values.mean()),
            float(values.mean()),
            float(values.mean()),
        )

    if mode == "event" or groups is None:

        idx = rng.integers(
            0,
            len(values),
            size=(n_boot, len(values)),
        )

        boot = values[idx].mean(axis=1)

    elif mode == "sequence":

        groups = np.asarray(groups)
        unique_groups = pd.unique(groups)

        grouped = {
            g: values[groups == g]
            for g in unique_groups
        }

        boot = np.empty(n_boot, dtype=float)

        for b in range(n_boot):

            sampled = rng.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )

            sample = np.concatenate([
                grouped[g]
                for g in sampled
            ])

            boot[b] = sample.mean()

    else:
        raise ValueError(
            "mode must be 'event' or 'sequence'"
        )

    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


# ============================================================
# Group guide lines
# ============================================================

def draw_group_guides(
    ax,
    include_proposed_band=True,
):

    ax.axhline(
        3.0,
        color="#D5DEE7",
        lw=0.75,
        zorder=0,
    )

    ax.axhline(
        7.0,
        color="#D5DEE7",
        lw=0.75,
        zorder=0,
    )

    if include_proposed_band:

        y = Y_POS["CA-URC"]

        ax.axhspan(
            y - 0.46,
            y + 0.46,
            color="#C51B1D",
            alpha=0.055,
            zorder=-2,
        )


# ============================================================
# Group labels
#
# 修改点：
# 灰色分组文字进一步向左移动；
# 但 x 仍然由 xmin + 1.2% axis width 得到，
# 因此始终位于图框内部，不会超出左边界。
# ============================================================

def add_group_headers_inside(ax):

    xmin, xmax = ax.get_xlim()

    x = xmin + 0.012 * (xmax - xmin)

    ax.text(
        x,
        0.08,
        "CAUSAL PROPAGATION",
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color="#6A7480",
        clip_on=True,
    )

    ax.text(
        x,
        3.35,
        "LEARNED CAUSAL",
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color="#6A7480",
        clip_on=True,
    )

    ax.text(
        x,
        7.35,
        "PROPOSED",
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color="#6A7480",
        clip_on=True,
    )


# ============================================================
# a / b : Risk-gap panels
# ============================================================

def risk_gap_panel(
    ax,
    quantity,
    letter,
):

    s = (
        summary
        .set_index("Method")
        .loc[METHOD_ORDER]
    )

    overall = s[
        f"{quantity}_overall_MAE"
    ]

    tail = s[
        f"{quantity}_tail_MAE"
    ]

    penalty = tail - overall

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    for method in METHOD_ORDER:

        y = Y_POS[method]
        c = COLORS[method]

        proposed = (
            method == "CA-URC"
        )

        # connector
        ax.plot(
            [
                overall[method],
                tail[method],
            ],
            [y, y],
            color=c,
            lw=(
                2.5
                if proposed
                else 1.75
            ),
            alpha=(
                1.0
                if proposed
                else 0.82
            ),
            solid_capstyle="round",
            zorder=2,
        )

        # overall
        ax.scatter(
            overall[method],
            y,
            s=(
                57
                if proposed
                else 43
            ),
            marker="o",
            facecolor="white",
            edgecolor=c,
            linewidth=(
                1.65
                if proposed
                else 1.35
            ),
            zorder=4,
        )

        # tail
        ax.scatter(
            tail[method],
            y,
            s=(
                56
                if proposed
                else 43
            ),
            marker="D",
            facecolor=c,
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )

        # Tail penalty values
        ax.text(
            0.985,
            y,
            f"{penalty[method]:+.3f}",
            transform=(
                ax.get_yaxis_transform()
            ),
            ha="right",
            va="center",
            fontsize=8.3,
            fontweight=(
                "bold"
                if proposed
                else "normal"
            ),
            color=(
                c
                if proposed
                else "#3F454B"
            ),
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ticks = [
        Y_POS[m]
        for m in METHOD_ORDER
    ]

    ax.set_yticks(
        ticks,
        METHOD_ORDER,
    )

    ax.invert_yaxis()

    ax.tick_params(
        axis="y",
        length=0,
    )

    draw_group_guides(ax)

    ax.grid(
        axis="x",
        color="#BFC8D2",
        lw=0.55,
        alpha=0.22,
    )

    ax.set_axisbelow(True)

    ax.set_xlabel(
        "Mean absolute error (dex)"
    )

    # --------------------------------------------------------
    # X range
    # --------------------------------------------------------

    vals = np.r_[
        overall.to_numpy(),
        tail.to_numpy(),
    ]

    span = (
        vals.max()
        - vals.min()
    )

    ax.set_xlim(
        vals.min()
        - 0.08 * span,
        vals.max()
        + 0.18 * span,
    )

    # --------------------------------------------------------
    # Group labels
    # --------------------------------------------------------

    add_group_headers_inside(ax)

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    add_panel_header(
        ax,
        letter,
        f"{quantity} risk gap",
    )

    # --------------------------------------------------------
    # Tail penalty title
    #
    # 修改点：
    # 放到图框外、右上角。
    # x=1.0 与右边框对齐，
    # y=1.012 略高于上边框。
    # --------------------------------------------------------

    ax.text(
        1.0,
        1.012,
        "Tail penalty",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#686868",
        clip_on=False,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = [

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#3C4650",
            markeredgewidth=1.25,
            markersize=6.5,
            label="Overall",
        ),

        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="#3C4650",
            markeredgecolor="white",
            markersize=6.2,
            label="High-motion tail",
        ),
    ]

    ax.legend(
        handles=legend,
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.005),
        handletextpad=0.45,
        columnspacing=1.2,
        fontsize=8.2,
    )

    apply_full_box(ax)


# ============================================================
# Half violin
# ============================================================

def half_violin(
    ax,
    values,
    position,
    color,
    width=0.68,
):

    if (
        len(values) < 2
        or np.allclose(
            values,
            values[0],
        )
    ):

        ax.plot(
            [values[0], values[0]],
            [
                position,
                position + width / 2.0,
            ],
            color=color,
            lw=4.0,
            alpha=0.20,
            solid_capstyle="round",
            zorder=1,
        )

        return

    vp = ax.violinplot(
        [values],
        positions=[position],
        vert=False,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=250,
        bw_method=0.30,
    )

    body = vp["bodies"][0]

    body.set_facecolor(
        color
    )

    body.set_edgecolor(
        "none"
    )

    body.set_alpha(
        0.20
    )

    path = (
        body
        .get_paths()[0]
    )

    vertices = path.vertices

    # keep lower half
    vertices[:, 1] = np.maximum(
        vertices[:, 1],
        position,
    )


# ============================================================
# c / d : Tail-event distributions
# ============================================================

def tail_distribution_panel(
    ax,
    quantity,
    letter,
):

    models = [
        "Graph",
        "Tail-weighted Attention",
        "Cross-Attention Base",
        "CA-URC",
    ]

    positions = np.arange(
        len(models)
    )

    rng = np.random.default_rng(
        RANDOM_SEED
        + (
            0
            if quantity == "PGA"
            else 1
        )
    )

    stat_rows = []
    all_values = []

    q = event[
        event["Quantity"]
        == quantity
    ].copy()

    for y, method in zip(
        positions,
        models,
    ):

        sub = q[
            q["Method"] == method
        ].copy()

        values = (
            sub["Tail_event_MAE"]
            .to_numpy(float)
        )

        groups = (
            sub["sequence_group"]
            .to_numpy()
        )

        all_values.append(
            values
        )

        c = COLORS[method]

        half_violin(
            ax,
            values,
            y,
            c,
        )

        # event points
        jitter = rng.uniform(
            -0.28,
            -0.08,
            size=len(values),
        )

        ax.scatter(
            values,
            y + jitter,
            s=11,
            facecolor=c,
            edgecolor="none",
            alpha=0.27,
            zorder=2,
        )

        # mean + 95% CI
        mean, lo, hi = (
            bootstrap_mean_ci(
                values,
                groups=groups,
                seed=(
                    RANDOM_SEED
                    + y
                    + (
                        0
                        if quantity == "PGA"
                        else 100
                    )
                ),
            )
        )

        ax.plot(
            [lo, hi],
            [y, y],
            color=c,
            lw=2.7,
            solid_capstyle="round",
            zorder=5,
        )

        ax.scatter(
            mean,
            y,
            s=(
                60
                if method == "CA-URC"
                else 52
            ),
            facecolor=c,
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )

        # median
        median = float(
            np.median(values)
        )

        ax.text(
            0.985,
            y,
            f"{median:.3f}",
            transform=(
                ax.get_yaxis_transform()
            ),
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight=(
                "bold"
                if method == "CA-URC"
                else "normal"
            ),
            color=(
                c
                if method == "CA-URC"
                else "#3F454B"
            ),
        )

        stat_rows.append({
            "Quantity": quantity,
            "Method": method,
            "n_events": len(values),
            "mean_tail_event_MAE": mean,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "median_tail_event_MAE": median,
            "bootstrap_mode": BOOTSTRAP_MODE,
        })

    # --------------------------------------------------------
    # X limits
    # --------------------------------------------------------

    all_values = np.concatenate(
        all_values
    )

    span = (
        all_values.max()
        - all_values.min()
    )

    if span == 0:
        span = 0.1

    ax.set_xlim(
        all_values.min()
        - 0.09 * span,
        all_values.max()
        + 0.18 * span,
    )

    # --------------------------------------------------------
    # Appearance
    # --------------------------------------------------------

    ax.grid(
        axis="x",
        color="#BFC8D2",
        lw=0.55,
        alpha=0.20,
    )

    ax.set_axisbelow(True)

    ax.set_yticks(
        positions,
        models,
    )

    ax.invert_yaxis()

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.set_xlabel(
        "Tail-event MAE (dex)"
    )

    # --------------------------------------------------------
    # panel letter + title aligned
    # --------------------------------------------------------

    add_panel_header(
        ax,
        letter,
        f"{quantity} tail-event distribution",
    )

    # Median title outside upper-right,
    # keeping same style as Tail penalty.
    ax.text(
        1.0,
        1.012,
        "Median",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color="#686868",
        clip_on=False,
    )

    apply_full_box(ax)

    return stat_rows


# ============================================================
# Main 2 × 2 figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.9, 9.8),
    gridspec_kw={
        "height_ratios": [
            1.18,
            0.92,
        ]
    },
)

fig.subplots_adjust(
    left=0.14,
    right=0.985,
    top=0.94,
    bottom=0.10,
    wspace=0.30,
    hspace=0.22,
)

# a
risk_gap_panel(
    axes[0, 0],
    "PGA",
    "a",
)

# b
risk_gap_panel(
    axes[0, 1],
    "PGV",
    "b",
)

stats = []

# c
stats += tail_distribution_panel(
    axes[1, 0],
    "PGA",
    "c",
)

# d
stats += tail_distribution_panel(
    axes[1, 1],
    "PGV",
    "d",
)


# ============================================================
# Save statistics
# ============================================================

pd.DataFrame(
    stats
).to_csv(
    HERE
    / "tail_distribution_statistics_final.csv",
    index=False,
)


# ============================================================
# Save figure
# ============================================================

for ext in [
    "pdf",
    "svg",
    "png",
]:

    kwargs = {
        "bbox_inches": "tight",
    }

    if ext == "png":
        kwargs["dpi"] = 600

    fig.savefig(
        f"{OUT_PREFIX}.{ext}",
        **kwargs,
    )

plt.close(fig)

print(
    f"Saved: {OUT_PREFIX}.png"
)