#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
28_plot_section_4_6_power_calibration_scan.py

Section 4.6:
Power calibration suppresses low-confidence correction leakage.

Primary inputs
--------------
runs/gate_power_scan_validation/gate_power_scan_summary.csv
runs/gate_power_scan_validation/best_gate_power.json

Optional input
--------------
runs/gate_power_scan_validation/paired_bootstrap_all_powers.csv

Figure
------
(a) Effective gate mapping p_tail^gamma
(b) Overall Delta MAE versus gamma
(c) Non-tail Delta MAE versus gamma
(d) High-motion-tail MAE versus gamma

Definitions
-----------
y_gamma = y_base + p_tail^(gamma-1) * (y_linear - y_base)

Equivalently, because y_linear - y_base = p_tail * Delta_tail:

y_gamma = y_base + p_tail^gamma * Delta_tail

The validation scan imposes:
    Overall MAE increase <= 0.015
    Non-tail MAE increase <= 0.005
    |Bias| <= 0.05

The selected gamma is read from best_gate_power.json. If that file is
not available, the script reproduces the selection rule used by the scan:
among constraint-feasible powers, choose the lowest tail score, then
overall score, then gamma.

Outputs
-------
Fig_4_6_power_calibration_scan.png
Fig_4_6_power_calibration_scan.svg
Fig_4_6_power_calibration_scan.pdf
Fig_4_6_plot_values.csv

Example
-------
python 28_plot_section_4_6_power_calibration_scan.py ^
  --summary "runs\\gate_power_scan_validation\\gate_power_scan_summary.csv" ^
  --selection-json "runs\\gate_power_scan_validation\\best_gate_power.json" ^
  --bootstrap "runs\\gate_power_scan_validation\\paired_bootstrap_all_powers.csv" ^
  --out-dir "figures\\section_4_6" ^
  --dpi 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_SUMMARY_COLUMNS = {
    "gate_power",
    "constraint_feasible",
    "macro_mae_pga",
    "macro_mae_pgv",
    "macro_non_tail_mae_pga",
    "macro_non_tail_mae_pgv",
    "macro_tail_mae_pga",
    "macro_tail_mae_pgv",
    "constraint_overall_delta_pga",
    "constraint_overall_delta_pgv",
    "constraint_non_tail_delta_pga",
    "constraint_non_tail_delta_pgv",
    "constraint_tail_delta_pga",
    "constraint_tail_delta_pgv",
    "bias_pga",
    "bias_pgv",
}

DEFAULT_OVERALL_BUDGET = 0.015
DEFAULT_NON_TAIL_BUDGET = 0.005
DEFAULT_BIAS_LIMIT = 0.05


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    missing = REQUIRED_SUMMARY_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(
            "gate_power_scan_summary.csv is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    frame = frame.copy()

    frame["gate_power"] = pd.to_numeric(
        frame["gate_power"],
        errors="coerce",
    )

    frame["constraint_feasible"] = as_bool(
        frame["constraint_feasible"]
    )

    numeric_columns = [
        column
        for column in frame.columns
        if (
            column.startswith("macro_")
            or column.startswith("constraint_")
            or column.startswith("bias_")
            or column.startswith("tail_under05_")
            or column.startswith("non_tail_mean_correction_")
        )
        and column != "constraint_feasible"
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame.dropna(subset=["gate_power"])
        .sort_values("gate_power")
        .reset_index(drop=True)
    )

    if len(frame) == 0:
        raise ValueError("Gate-power summary is empty.")

    return frame


def infer_selected_power(summary: pd.DataFrame) -> float:
    feasible = summary.loc[
        summary["constraint_feasible"]
    ].copy()

    if len(feasible) > 0:
        sort_columns = []

        if "constraint_tail_score" in feasible.columns:
            sort_columns.append("constraint_tail_score")
        else:
            feasible["_tail_score"] = 0.5 * (
                feasible["macro_tail_mae_pga"]
                + feasible["macro_tail_mae_pgv"]
            )
            sort_columns.append("_tail_score")

        if "constraint_overall_score" in feasible.columns:
            sort_columns.append("constraint_overall_score")
        else:
            feasible["_overall_score"] = 0.5 * (
                feasible["macro_mae_pga"]
                + feasible["macro_mae_pgv"]
            )
            sort_columns.append("_overall_score")

        sort_columns.append("gate_power")

        return float(
            feasible.sort_values(sort_columns)
            .iloc[0]["gate_power"]
        )

    if "constraint_normalized_violation" in summary.columns:
        fallback = summary.sort_values(
            [
                "constraint_normalized_violation",
                "gate_power",
            ]
        )
        return float(fallback.iloc[0]["gate_power"])

    return float(summary.iloc[0]["gate_power"])


def load_selection(
    path: Path | None,
    summary: pd.DataFrame,
) -> tuple[float, float, float, float]:
    selected_power = infer_selected_power(summary)

    overall_budget = DEFAULT_OVERALL_BUDGET
    non_tail_budget = DEFAULT_NON_TAIL_BUDGET
    bias_limit = DEFAULT_BIAS_LIMIT

    if path is None or not path.exists():
        return (
            selected_power,
            overall_budget,
            non_tail_budget,
            bias_limit,
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    selected_power = float(
        data.get(
            "selected_gate_power",
            selected_power,
        )
    )

    overall_budget = float(
        data.get(
            "overall_budget",
            overall_budget,
        )
    )

    non_tail_budget = float(
        data.get(
            "non_tail_budget",
            non_tail_budget,
        )
    )

    bias_limit = float(
        data.get(
            "bias_limit",
            bias_limit,
        )
    )

    return (
        selected_power,
        overall_budget,
        non_tail_budget,
        bias_limit,
    )


def derive_base_metrics(
    summary: pd.DataFrame,
) -> dict[str, float]:
    """
    Recover frozen-base metrics from:
        candidate_metric - delta_candidate_minus_base
    using all gamma rows, then take the median for numerical robustness.
    """

    result: dict[str, float] = {}

    for quantity in ("pga", "pgv"):
        result[f"overall_{quantity}"] = float(
            np.nanmedian(
                summary[f"macro_mae_{quantity}"]
                - summary[
                    f"constraint_overall_delta_{quantity}"
                ]
            )
        )

        result[f"non_tail_{quantity}"] = float(
            np.nanmedian(
                summary[
                    f"macro_non_tail_mae_{quantity}"
                ]
                - summary[
                    f"constraint_non_tail_delta_{quantity}"
                ]
            )
        )

        result[f"tail_{quantity}"] = float(
            np.nanmedian(
                summary[
                    f"macro_tail_mae_{quantity}"
                ]
                - summary[
                    f"constraint_tail_delta_{quantity}"
                ]
            )
        )

    return result


def load_bootstrap(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    frame = pd.read_csv(path)

    required = {
        "candidate_model",
        "quantity",
        "population",
        "metric",
        "ci95_low",
        "ci95_high",
    }

    if not required.issubset(frame.columns):
        print(
            "WARNING: bootstrap CSV does not contain the expected schema; "
            "confidence intervals will be omitted."
        )
        return None

    mean_candidates = [
        "mean_delta_candidate_minus_base",
        "mean_delta_candidate_minus_reference",
    ]

    mean_column = next(
        (
            column
            for column in mean_candidates
            if column in frame.columns
        ),
        None,
    )

    if mean_column is None:
        print(
            "WARNING: bootstrap CSV has no mean-delta column; "
            "confidence intervals will be omitted."
        )
        return None

    frame = frame.copy()

    frame["quantity"] = (
        frame["quantity"]
        .astype(str)
        .str.lower()
    )

    frame["population"] = (
        frame["population"]
        .astype(str)
        .str.lower()
    )

    frame["metric"] = (
        frame["metric"]
        .astype(str)
        .str.lower()
    )

    frame["_mean_delta"] = pd.to_numeric(
        frame[mean_column],
        errors="coerce",
    )

    frame["ci95_low"] = pd.to_numeric(
        frame["ci95_low"],
        errors="coerce",
    )

    frame["ci95_high"] = pd.to_numeric(
        frame["ci95_high"],
        errors="coerce",
    )

    return frame


def model_name_for_gamma(gamma: float) -> str:
    return "gamma_" + str(float(gamma)).replace(".", "p")


def bootstrap_ci(
    bootstrap: pd.DataFrame | None,
    gamma: float,
    quantity: str,
    population: str,
) -> tuple[float, float] | None:
    if bootstrap is None:
        return None

    model_name = model_name_for_gamma(gamma)

    row = bootstrap.loc[
        bootstrap["candidate_model"].astype(str).eq(model_name)
        & bootstrap["quantity"].eq(quantity)
        & bootstrap["population"].eq(population)
        & bootstrap["metric"].eq("mae")
    ]

    if len(row) != 1:
        return None

    record = row.iloc[0]

    low = float(record["ci95_low"])
    high = float(record["ci95_high"])

    if not (
        np.isfinite(low)
        and np.isfinite(high)
    ):
        return None

    return low, high


def selected_row(
    summary: pd.DataFrame,
    gamma: float,
) -> pd.Series:
    distance = np.abs(
        summary["gate_power"].to_numpy(dtype=float)
        - float(gamma)
    )

    index = int(np.argmin(distance))

    if distance[index] > 1e-8:
        raise ValueError(
            f"Selected gamma={gamma} is not present in the scan summary."
        )

    return summary.iloc[index]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.55,
        alpha=0.28,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_selected_gamma(
    ax: plt.Axes,
    gamma: float,
) -> None:
    ax.axvline(
        gamma,
        linestyle="--",
        linewidth=1.05,
        alpha=0.75,
    )


def draw_feasibility_markers(
    ax: plt.Axes,
    summary: pd.DataFrame,
    y_values: np.ndarray,
    selected_gamma: float,
) -> None:
    x = summary["gate_power"].to_numpy(dtype=float)
    feasible = summary[
        "constraint_feasible"
    ].to_numpy(dtype=bool)

    ax.scatter(
        x[feasible],
        y_values[feasible],
        marker="o",
        s=34,
        zorder=5,
        label="Constraint-feasible",
    )

    ax.scatter(
        x[~feasible],
        y_values[~feasible],
        marker="x",
        s=38,
        zorder=5,
        label="Constraint-violating",
    )

    chosen = np.isclose(
        x,
        selected_gamma,
        atol=1e-8,
    )

    if chosen.any():
        ax.scatter(
            x[chosen],
            y_values[chosen],
            marker="*",
            s=150,
            linewidths=0.9,
            edgecolors="black",
            zorder=7,
            label=rf"Selected $\gamma={selected_gamma:g}$",
        )


def panel_a_gate_mapping(
    ax: plt.Axes,
    selected_gamma: float,
) -> None:
    p = np.linspace(
        0.0,
        1.0,
        301,
    )

    gammas = [
        1.0,
        2.0,
        selected_gamma,
        4.0,
    ]

    gammas = sorted(
        set(float(x) for x in gammas)
    )

    line_styles = [
        "-",
        "--",
        "-.",
        ":",
    ]

    for index, gamma in enumerate(gammas):
        linewidth = (
            2.2
            if np.isclose(
                gamma,
                selected_gamma,
            )
            else 1.35
        )

        ax.plot(
            p,
            np.power(p, gamma),
            linestyle=line_styles[
                index % len(line_styles)
            ],
            linewidth=linewidth,
            label=rf"$\gamma={gamma:g}$",
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linewidth=0.8,
        alpha=0.35,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(
        r"Tail-risk probability $p_{\mathrm{tail}}$"
    )

    ax.set_ylabel(
        r"Effective gate weight $p_{\mathrm{tail}}^{\gamma}$"
    )

    ax.set_title(
        "(a) Power calibration suppresses low-confidence gates",
        loc="left",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    p_low = 0.2

    original_retention = (
        p_low ** (selected_gamma - 1.0)
    )

    ax.text(
        0.04,
        0.05,
        (
            rf"At $p_{{\mathrm{{tail}}}}=0.2$, "
            rf"$\gamma={selected_gamma:g}$ retains "
            f"{100.0 * original_retention:.1f}%\n"
            "of the original linear correction."
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    style_axis(ax)


def delta_errorbars(
    bootstrap: pd.DataFrame | None,
    summary: pd.DataFrame,
    quantity: str,
    population: str,
    delta_values: np.ndarray,
) -> np.ndarray | None:
    if bootstrap is None:
        return None

    lower = []
    upper = []
    any_valid = False

    for gamma, mean_value in zip(
        summary["gate_power"].to_numpy(dtype=float),
        delta_values,
    ):
        interval = bootstrap_ci(
            bootstrap,
            gamma,
            quantity,
            population,
        )

        if interval is None:
            lower.append(0.0)
            upper.append(0.0)
            continue

        low, high = interval

        lower.append(
            max(
                0.0,
                mean_value - low,
            )
        )

        upper.append(
            max(
                0.0,
                high - mean_value,
            )
        )

        any_valid = True

    if not any_valid:
        return None

    return np.vstack(
        [
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ]
    )


def panel_delta_mae(
    ax: plt.Axes,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    population: str,
    budget: float,
    selected_gamma: float,
    panel_letter: str,
    title: str,
) -> None:
    x = summary[
        "gate_power"
    ].to_numpy(dtype=float)

    for quantity, marker, linestyle in (
        ("pga", "o", "-"),
        ("pgv", "s", "--"),
    ):
        column = (
            f"constraint_{population}_delta_{quantity}"
        )

        delta = summary[
            column
        ].to_numpy(dtype=float)

        error = delta_errorbars(
            bootstrap,
            summary,
            quantity,
            population,
            delta,
        )

        ax.errorbar(
            x,
            delta,
            yerr=error,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4.3,
            capsize=2.6 if error is not None else 0.0,
            label=quantity.upper(),
            zorder=4,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.9,
        alpha=0.65,
    )

    ax.axhline(
        budget,
        linestyle=":",
        linewidth=1.2,
        alpha=0.85,
        label=rf"Budget = +{budget:.3f}",
    )

    annotate_selected_gamma(
        ax,
        selected_gamma,
    )

    ax.set_xlabel(
        r"Gate power $\gamma$"
    )

    ax.set_ylabel(
        r"$\Delta$MAE relative to frozen base"
    )

    ax.set_title(
        f"({panel_letter}) {title}",
        loc="left",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    style_axis(ax)


def absolute_tail_errorbars(
    bootstrap: pd.DataFrame | None,
    summary: pd.DataFrame,
    base_tail: float,
    quantity: str,
    absolute_values: np.ndarray,
) -> np.ndarray | None:
    if bootstrap is None:
        return None

    lower = []
    upper = []
    any_valid = False

    for gamma, absolute in zip(
        summary["gate_power"].to_numpy(dtype=float),
        absolute_values,
    ):
        interval = bootstrap_ci(
            bootstrap,
            gamma,
            quantity,
            "tail",
        )

        if interval is None:
            lower.append(0.0)
            upper.append(0.0)
            continue

        low_delta, high_delta = interval

        low_absolute = (
            base_tail + low_delta
        )

        high_absolute = (
            base_tail + high_delta
        )

        lower.append(
            max(
                0.0,
                absolute - low_absolute,
            )
        )

        upper.append(
            max(
                0.0,
                high_absolute - absolute,
            )
        )

        any_valid = True

    if not any_valid:
        return None

    return np.vstack(
        [
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ]
    )


def panel_tail_mae(
    ax: plt.Axes,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    base_metrics: dict[str, float],
    selected_gamma: float,
) -> None:
    x = summary[
        "gate_power"
    ].to_numpy(dtype=float)

    plotted_values = []

    for quantity, marker, linestyle in (
        ("pga", "o", "-"),
        ("pgv", "s", "--"),
    ):
        y = summary[
            f"macro_tail_mae_{quantity}"
        ].to_numpy(dtype=float)

        plotted_values.append(y)

        base_tail = float(
            base_metrics[
                f"tail_{quantity}"
            ]
        )

        error = absolute_tail_errorbars(
            bootstrap,
            summary,
            base_tail,
            quantity,
            y,
        )

        ax.errorbar(
            x,
            y,
            yerr=error,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.55,
            markersize=4.3,
            capsize=2.6 if error is not None else 0.0,
            label=f"{quantity.upper()} tail MAE",
            zorder=4,
        )

        ax.axhline(
            base_tail,
            linestyle=":",
            linewidth=0.9,
            alpha=0.55,
        )

    annotate_selected_gamma(
        ax,
        selected_gamma,
    )

    # Show constraint feasibility using the mean of PGA/PGV tail MAE.
    mean_tail = 0.5 * (
        summary["macro_tail_mae_pga"].to_numpy(dtype=float)
        + summary["macro_tail_mae_pgv"].to_numpy(dtype=float)
    )

    draw_feasibility_markers(
        ax,
        summary,
        mean_tail,
        selected_gamma,
    )

    chosen = selected_row(
        summary,
        selected_gamma,
    )

    ax.text(
        0.04,
        0.05,
        (
            rf"Selected $\gamma={selected_gamma:g}$"
            "\n"
            f"PGA tail MAE = "
            f"{float(chosen['macro_tail_mae_pga']):.3f}"
            "\n"
            f"PGV tail MAE = "
            f"{float(chosen['macro_tail_mae_pgv']):.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    ax.set_xlabel(
        r"Gate power $\gamma$"
    )

    ax.set_ylabel(
        r"High-motion-tail MAE ($\log_{10}$ units)"
    )

    ax.set_title(
        "(d) Calibration trades tail gain for correction selectivity",
        loc="left",
        fontweight="bold",
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    unique = {}
    for handle, label in zip(
        handles,
        labels,
    ):
        if label not in unique:
            unique[label] = handle

    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="best",
        fontsize=7.2,
    )

    style_axis(ax)


def export_plot_values(
    summary: pd.DataFrame,
    base_metrics: dict[str, float],
    selected_gamma: float,
) -> pd.DataFrame:
    export = summary.copy()

    export["selected_gate_power"] = np.isclose(
        export["gate_power"].to_numpy(dtype=float),
        selected_gamma,
        atol=1e-8,
    )

    for quantity in ("pga", "pgv"):
        export[
            f"base_overall_mae_{quantity}"
        ] = base_metrics[
            f"overall_{quantity}"
        ]

        export[
            f"base_non_tail_mae_{quantity}"
        ] = base_metrics[
            f"non_tail_{quantity}"
        ]

        export[
            f"base_tail_mae_{quantity}"
        ] = base_metrics[
            f"tail_{quantity}"
        ]

    return export


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary",
        default=(
            "runs/gate_power_scan_validation/"
            "gate_power_scan_summary.csv"
        ),
    )

    parser.add_argument(
        "--selection-json",
        default=(
            "runs/gate_power_scan_validation/"
            "best_gate_power.json"
        ),
    )

    parser.add_argument(
        "--bootstrap",
        default=(
            "runs/gate_power_scan_validation/"
            "paired_bootstrap_all_powers.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "figures/section_4_6"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    summary_path = Path(
        args.summary
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: "
            f"{summary_path.resolve()}"
        )

    selection_path = (
        Path(args.selection_json)
        if args.selection_json
        else None
    )

    bootstrap_path = (
        Path(args.bootstrap)
        if args.bootstrap
        else None
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = load_summary(
        summary_path
    )

    (
        selected_gamma,
        overall_budget,
        non_tail_budget,
        bias_limit,
    ) = load_selection(
        selection_path,
        summary,
    )

    base_metrics = derive_base_metrics(
        summary
    )

    bootstrap = load_bootstrap(
        bootstrap_path
    )

    chosen = selected_row(
        summary,
        selected_gamma,
    )

    print(
        "\n=== Section 4.6 power-calibration scan ==="
    )

    print(
        f"Summary          : {summary_path.resolve()}"
    )

    print(
        f"Selected gamma   : {selected_gamma:g}"
    )

    print(
        "Constraints      : "
        f"overall <= +{overall_budget:.4f}, "
        f"non-tail <= +{non_tail_budget:.4f}, "
        f"|bias| <= {bias_limit:.4f}"
    )

    print(
        "Selected feasible: "
        f"{bool(chosen['constraint_feasible'])}"
    )

    print(
        "Overall MAE      : "
        f"PGA={float(chosen['macro_mae_pga']):.6f}, "
        f"PGV={float(chosen['macro_mae_pgv']):.6f}"
    )

    print(
        "Non-tail MAE     : "
        f"PGA={float(chosen['macro_non_tail_mae_pga']):.6f}, "
        f"PGV={float(chosen['macro_non_tail_mae_pgv']):.6f}"
    )

    print(
        "Tail MAE         : "
        f"PGA={float(chosen['macro_tail_mae_pga']):.6f}, "
        f"PGV={float(chosen['macro_tail_mae_pgv']):.6f}"
    )

    print(
        "Bias             : "
        f"PGA={float(chosen['bias_pga']):+.6f}, "
        f"PGV={float(chosen['bias_pgv']):+.6f}"
    )

    print(
        "\n=== Frozen-base metrics recovered from scan deltas ==="
    )

    for quantity in ("pga", "pgv"):
        print(
            f"{quantity.upper()}: "
            f"overall={base_metrics[f'overall_{quantity}']:.6f}, "
            f"non-tail={base_metrics[f'non_tail_{quantity}']:.6f}, "
            f"tail={base_metrics[f'tail_{quantity}']:.6f}"
        )

    export = export_plot_values(
        summary,
        base_metrics,
        selected_gamma,
    )

    export_path = (
        out_dir
        / "Fig_4_6_plot_values.csv"
    )

    export.to_csv(
        export_path,
        index=False,
        encoding="utf-8-sig",
    )

    set_style()

    fig = plt.figure(
        figsize=(10.8, 7.7)
    )

    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.27,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(
        gs[0, 0]
    )

    ax_b = fig.add_subplot(
        gs[0, 1]
    )

    ax_c = fig.add_subplot(
        gs[1, 0]
    )

    ax_d = fig.add_subplot(
        gs[1, 1]
    )

    panel_a_gate_mapping(
        ax_a,
        selected_gamma,
    )

    panel_delta_mae(
        ax_b,
        summary,
        bootstrap,
        population="overall",
        budget=overall_budget,
        selected_gamma=selected_gamma,
        panel_letter="b",
        title="Catalogue-wide performance returns toward the base",
    )

    panel_delta_mae(
        ax_c,
        summary,
        bootstrap,
        population="non_tail",
        budget=non_tail_budget,
        selected_gamma=selected_gamma,
        panel_letter="c",
        title="Non-tail leakage is progressively suppressed",
    )

    panel_tail_mae(
        ax_d,
        summary,
        bootstrap,
        base_metrics,
        selected_gamma,
    )

    png_path = (
        out_dir
        / "Fig_4_6_power_calibration_scan.png"
    )

    svg_path = (
        out_dir
        / "Fig_4_6_power_calibration_scan.svg"
    )

    pdf_path = (
        out_dir
        / "Fig_4_6_power_calibration_scan.pdf"
    )

    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved:"
    )

    print(
        png_path.resolve()
    )

    print(
        svg_path.resolve()
    )

    print(
        pdf_path.resolve()
    )

    print(
        export_path.resolve()
    )


if __name__ == "__main__":
    main()
