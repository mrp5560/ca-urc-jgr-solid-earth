import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ==========================================
# Nature Communications–style plotting code
# ==========================================
# Input:
#   caurc_fair_comparison_tidy.csv
# Output:
#   Fig_main_CAURC_NComms_style.(png/pdf/svg)
#
# Design principles used here:
#   1. clean white background
#   2. restrained palette
#   3. no heavy chart junk
#   4. conclusion-driven panels
#   5. consistent ordering across all panels

DATA_PATH = Path("caurc_fair_comparison_tidy.csv")
OUT_STEM = "Fig_main_CAURC_NComms_style"

df = pd.read_csv(DATA_PATH)

order = [
    "Nearest observed",
    "IDW observed",
    "PLUM-like",
    "Graph",
    "Tail-weighted Attention",
    "Cross-Attention Base",
    "CA-URC",
]

type_order = {
    "Causal propagation": 0,
    "Learned causal": 1,
    "Proposed": 2,
}

df["Method"] = pd.Categorical(df["Method"], categories=order, ordered=True)
df = df.sort_values(["Method", "Quantity"]).reset_index(drop=True)

# Minimalist publication defaults
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Restrained palette
palette = {
    "Causal propagation": "#B8BCC2",
    "Learned causal": "#5A7DA0",
    "Proposed": "#C23B3B",
}

y = np.arange(len(order))

def draw_dumbbell(ax, quantity):
    sub = df[df["Quantity"] == quantity].set_index("Method").loc[order].reset_index()

    for i, row in enumerate(sub.itertuples(index=False)):
        method = row.Method
        typ = row.Type
        overall = row.Overall_MAE
        tail = row.Tail_MAE

        c = palette[typ]
        alpha = 1.0 if method == "CA-URC" else 0.85

        # subtle reference band for the proposed model row
        if method == "CA-URC":
            ax.axhspan(i - 0.38, i + 0.38, color=c, alpha=0.06, zorder=0)

        # overall -> tail connection
        ax.plot([overall, tail], [i, i],
                color=c, lw=2.5 if method == "CA-URC" else 1.8,
                alpha=alpha, solid_capstyle="round", zorder=1)

        # overall point: open circle
        ax.scatter(overall, i, s=56 if method == "CA-URC" else 44,
                   facecolor="white", edgecolor=c,
                   linewidth=1.6 if method == "CA-URC" else 1.2,
                   marker="o", zorder=3)

        # tail point: filled diamond
        ax.scatter(tail, i, s=58 if method == "CA-URC" else 46,
                   facecolor=c, edgecolor=c,
                   linewidth=0.8, marker="D", zorder=4)

        # tail penalty annotation
        penalty = tail - overall
        ax.text(max(overall, tail) + 0.02, i, f"+{penalty:.3f}",
                va="center", ha="left",
                fontsize=8.0,
                color=c if method == "CA-URC" else "#555555",
                fontweight="bold" if method == "CA-URC" else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="-", linewidth=0.55, alpha=0.14)
    ax.set_xlabel("MAE")
    ax.set_title(f"{quantity}: overall accuracy versus tail error",
                 loc="left", fontweight="bold")

    if quantity == "PGA":
        ax.text(0.00, 1.02,
                "Open circles: overall MAE; filled diamonds: tail MAE; labels: tail penalty",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.2, color="#444444")

def draw_underestimation(ax, quantity):
    sub = df[df["Quantity"] == quantity].set_index("Method").loc[order].reset_index()

    vals = sub["Tail_U0.5_pct"].to_numpy()
    types = sub["Type"].tolist()
    methods = sub["Method"].tolist()
    colors = [palette[t] for t in types]

    bars = ax.barh(y, vals, color=colors, edgecolor="none", height=0.58, alpha=0.92)

    for i, (bar, v, method, c) in enumerate(zip(bars, vals, methods, colors)):
        if method == "CA-URC":
            bar.set_edgecolor(c)
            bar.set_linewidth(1.6)
            ax.axhspan(i - 0.38, i + 0.38, color=c, alpha=0.06, zorder=0)
        ax.text(v + 1.1, i, f"{v:.2f}%",
                va="center", ha="left",
                fontsize=8.1,
                color=c if method == "CA-URC" else "#222222",
                fontweight="bold" if method == "CA-URC" else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="-", linewidth=0.55, alpha=0.14)
    ax.set_xlabel("Underestimation rate (%)")
    ax.set_title(f"{quantity}: tail underestimation rate ($U_{{0.5}}$)",
                 loc="left", fontweight="bold")

fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.10,
                    wspace=0.26, hspace=0.30)

draw_dumbbell(axes[0, 0], "PGA")
draw_dumbbell(axes[0, 1], "PGV")
draw_underestimation(axes[1, 0], "PGA")
draw_underestimation(axes[1, 1], "PGV")

# panel labels
panel_labels = ["a", "b", "c", "d"]
for ax, lab in zip(axes.flatten(), panel_labels):
    ax.text(-0.14, 1.03, lab,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            va="bottom", ha="left")

# concise, publication-style legend
legend_elements = [
    Line2D([0], [0], color=palette["Causal propagation"], lw=2.4, label="Causal propagation"),
    Line2D([0], [0], color=palette["Learned causal"], lw=2.4, label="Learned causal"),
    Line2D([0], [0], color=palette["Proposed"], lw=2.4, label="Proposed (CA-URC)"),
    Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
           markersize=6.2, lw=0, label="Overall MAE"),
    Line2D([0], [0], marker="D", color="black", markerfacecolor="black",
           markersize=5.9, lw=0, label="Tail MAE"),
]

fig.legend(handles=legend_elements,
           loc="upper center", bbox_to_anchor=(0.5, 0.985),
           frameon=False, ncol=5, handlelength=1.7, columnspacing=1.35)

fig.suptitle(
    "Fair comparison of causal propagation, learned causal, and proposed CA-URC models",
    fontsize=14, fontweight="bold", y=0.975
)

for ext in ["png", "pdf", "svg"]:
    save_kwargs = {"bbox_inches": "tight"}
    if ext == "png":
        save_kwargs["dpi"] = 600
    fig.savefig(f"{OUT_STEM}.{ext}", **save_kwargs)

plt.show()