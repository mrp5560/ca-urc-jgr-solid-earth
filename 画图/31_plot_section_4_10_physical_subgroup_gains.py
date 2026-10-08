#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
31_plot_section_4_10_physical_subgroup_gains.py

Section 4.10:
Power-gate gains concentrate on P-wave-reached targets rather than on
event magnitude alone.

Primary input
-------------
runs/locked_power_gated_test_gamma3/locked_test_predictions.csv

Required columns
----------------
event_id
repeat
magnitude
target_triggered_by_snapshot
true_log10_pga
true_log10_pgv
is_tail_pga
is_tail_pgv
base_log10_pga
base_log10_pgv
power_gate_log10_pga
power_gate_log10_pgv

Physical subgroups
------------------
1. High-motion tail
2. Triggered (P wave reached the held-out target by t0 = 5 s)
3. M >= 4
4. Not triggered

Important
---------
These subgroups are NOT mutually exclusive. Each subgroup is compared
independently using exactly the same Base and Power Gate predictions.

For each quantity and subgroup:
    delta MAE     = MAE_power - MAE_base
    delta Under05 = Under05_power - Under05_base

Negative values favor Power Gate for both metrics.

Statistical aggregation
-----------------------
1. Compute metric difference inside each event-repeat unit.
2. Average the paired differences across repeats within each event.
3. Bootstrap independent events with replacement.
4. Report mean paired delta and 95% event-level bootstrap CI.

Figure
------
(a) Paired Delta MAE by physical subgroup.
(b) Paired Delta Under05 by physical subgroup, in percentage points.

The main purpose is physical interpretation:
    high-motion tail  -> largest gain
    triggered         -> clear gain
    M>=4              -> weaker / mixed gain
    not triggered     -> little change

Outputs
-------
Fig_4_10_physical_subgroup_gains.png
Fig_4_10_physical_subgroup_gains.svg
Fig_4_10_physical_subgroup_gains.pdf
Fig_4_10_subgroup_bootstrap_values.csv
Fig_4_10_consistency_check.csv

Example
-------
python 31_plot_section_4_10_physical_subgroup_gains.py ^
  --predictions "runs\\locked_power_gated_test_gamma3\\locked_test_predictions.csv" ^
  --out-dir "figures\\section_4_10" ^
  --bootstrap-repetitions 2000 ^
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "event_id",
    "repeat",
    "magnitude",
    "target_triggered_by_snapshot",
    "true_log10_pga",
    "true_log10_pgv",
    "is_tail_pga",
    "is_tail_pgv",
    "base_log10_pga",
    "base_log10_pgv",
    "power_gate_log10_pga",
    "power_gate_log10_pgv",
]

GROUP_ORDER = [
    "tail",
    "triggered",
    "m4plus",
    "not_triggered",
]

GROUP_LABELS = {
    "tail": "High-motion tail",
    "triggered": "P-wave reached by snapshot",
    "m4plus": r"$M \geq 4$ events",
    "not_triggered": "P wave not reached",
}

# Formal locked-test paired-bootstrap values already reported for the first
# three groups. These are ONLY used as a consistency check.
# Not-triggered was not printed in the compact formal log and is therefore
# recomputed directly without a hard-coded reference.
EXPECTED = {
    ("pga", "tail", "mae"): -0.080118,
    ("pgv", "tail", "mae"): -0.043814,
    ("pga", "tail", "under05"): -0.144451,
    ("pgv", "tail", "under05"): -0.062750,

    ("pga", "triggered", "mae"): -0.015370,
    ("pgv", "triggered", "mae"): -0.013391,
    ("pga", "triggered", "under05"): -0.052968,
    ("pgv", "triggered", "under05"): -0.026487,

    ("pga", "m4plus", "mae"): -0.005656,
    ("pgv", "m4plus", "mae"): -0.000359,
    ("pga", "m4plus", "under05"): -0.013500,
    ("pgv", "m4plus", "under05"): -0.010833,
}


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"event_id": str},
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "locked_test_predictions.csv is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(frame.columns.astype(str))
        )

    frame = frame[
        REQUIRED_COLUMNS
    ].copy()

    frame["event_id"] = (
        frame["event_id"]
        .astype(str)
    )

    frame["repeat"] = pd.to_numeric(
        frame["repeat"],
        errors="coerce",
    ).fillna(-1).astype(int)

    frame["magnitude"] = pd.to_numeric(
        frame["magnitude"],
        errors="coerce",
    )

    frame[
        "target_triggered_by_snapshot"
    ] = parse_bool(
        frame[
            "target_triggered_by_snapshot"
        ]
    )

    for quantity in ("pga", "pgv"):
        frame[
            f"is_tail_{quantity}"
        ] = parse_bool(
            frame[
                f"is_tail_{quantity}"
            ]
        )

        for prefix in (
            "true_log10",
            "base_log10",
            "power_gate_log10",
        ):
            column = (
                f"{prefix}_{quantity}"
            )

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    numeric_required = [
        f"{prefix}_{quantity}"
        for quantity in ("pga", "pgv")
        for prefix in (
            "true_log10",
            "base_log10",
            "power_gate_log10",
        )
    ]

    bad = (
        frame[
            numeric_required
        ]
        .isna()
        .any(axis=1)
    )

    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows contain missing required "
            "prediction/target values."
        )

    return frame


def subgroup_mask(
    frame: pd.DataFrame,
    quantity: str,
    subgroup: str,
) -> np.ndarray:
    if subgroup == "tail":
        return (
            frame[
                f"is_tail_{quantity}"
            ]
            .to_numpy(dtype=bool)
        )

    if subgroup == "triggered":
        return (
            frame[
                "target_triggered_by_snapshot"
            ]
            .to_numpy(dtype=bool)
        )

    if subgroup == "not_triggered":
        return ~(
            frame[
                "target_triggered_by_snapshot"
            ]
            .to_numpy(dtype=bool)
        )

    if subgroup == "m4plus":
        magnitude = (
            frame["magnitude"]
            .to_numpy(dtype=float)
        )

        return (
            np.isfinite(magnitude)
            & (magnitude >= 4.0)
        )

    raise ValueError(
        f"Unknown subgroup: {subgroup}"
    )


def build_event_repeat_deltas(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns one row per:
        event_id / repeat / quantity / subgroup / metric

    with:
        value = PowerGate - Base
    """
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        truth = (
            frame[
                f"true_log10_{quantity}"
            ]
            .to_numpy(dtype=float)
        )

        base = (
            frame[
                f"base_log10_{quantity}"
            ]
            .to_numpy(dtype=float)
        )

        power = (
            frame[
                f"power_gate_log10_{quantity}"
            ]
            .to_numpy(dtype=float)
        )

        base_residual = (
            base - truth
        )

        power_residual = (
            power - truth
        )

        working = pd.DataFrame(
            {
                "event_id": (
                    frame[
                        "event_id"
                    ]
                    .astype(str)
                    .to_numpy()
                ),
                "repeat": (
                    frame[
                        "repeat"
                    ]
                    .to_numpy(dtype=int)
                ),
                "base_abs_error": np.abs(
                    base_residual
                ),
                "power_abs_error": np.abs(
                    power_residual
                ),
                "base_under05": (
                    base_residual <= -0.5
                ).astype(float),
                "power_under05": (
                    power_residual <= -0.5
                ).astype(float),
            }
        )

        for subgroup in GROUP_ORDER:
            mask = subgroup_mask(
                frame,
                quantity,
                subgroup,
            )

            subset = (
                working.loc[
                    mask
                ]
                .copy()
            )

            if len(subset) == 0:
                continue

            grouped = (
                subset.groupby(
                    [
                        "event_id",
                        "repeat",
                    ],
                    sort=False,
                )
                .agg(
                    base_mae=(
                        "base_abs_error",
                        "mean",
                    ),
                    power_mae=(
                        "power_abs_error",
                        "mean",
                    ),
                    base_under05=(
                        "base_under05",
                        "mean",
                    ),
                    power_under05=(
                        "power_under05",
                        "mean",
                    ),
                    n_rows=(
                        "base_abs_error",
                        "size",
                    ),
                )
                .reset_index()
            )

            grouped[
                "delta_mae"
            ] = (
                grouped[
                    "power_mae"
                ]
                - grouped[
                    "base_mae"
                ]
            )

            grouped[
                "delta_under05"
            ] = (
                grouped[
                    "power_under05"
                ]
                - grouped[
                    "base_under05"
                ]
            )

            for record in (
                grouped.itertuples(
                    index=False
                )
            ):
                rows.append(
                    {
                        "event_id": str(
                            record.event_id
                        ),
                        "repeat": int(
                            record.repeat
                        ),
                        "quantity": quantity,
                        "subgroup": subgroup,
                        "metric": "mae",
                        "value": float(
                            record.delta_mae
                        ),
                        "n_rows": int(
                            record.n_rows
                        ),
                    }
                )

                rows.append(
                    {
                        "event_id": str(
                            record.event_id
                        ),
                        "repeat": int(
                            record.repeat
                        ),
                        "quantity": quantity,
                        "subgroup": subgroup,
                        "metric": "under05",
                        "value": float(
                            record.delta_under05
                        ),
                        "n_rows": int(
                            record.n_rows
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def bootstrap_event_deltas(
    event_repeat: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """
    1. Average repeat-level paired deltas within event.
    2. Bootstrap independent events.
    """
    rng = np.random.default_rng(
        seed
    )

    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        for subgroup in GROUP_ORDER:
            for metric in (
                "mae",
                "under05",
            ):
                subset = (
                    event_repeat.loc[
                        event_repeat[
                            "quantity"
                        ].eq(quantity)
                        & event_repeat[
                            "subgroup"
                        ].eq(subgroup)
                        & event_repeat[
                            "metric"
                        ].eq(metric)
                    ]
                )

                if len(subset) == 0:
                    continue

                event_values = (
                    subset.groupby(
                        "event_id",
                        sort=False,
                    )["value"]
                    .mean()
                )

                values = (
                    event_values
                    .to_numpy(
                        dtype=float
                    )
                )

                if len(values) == 0:
                    continue

                bootstrap_means = (
                    np.empty(
                        repetitions,
                        dtype=float,
                    )
                )

                for index in range(
                    repetitions
                ):
                    selected = rng.integers(
                        0,
                        len(values),
                        size=len(values),
                    )

                    bootstrap_means[
                        index
                    ] = float(
                        values[
                            selected
                        ].mean()
                    )

                row_mask = subgroup_mask(
                    original_frame,
                    quantity,
                    subgroup,
                )

                rows.append(
                    {
                        "quantity": quantity,
                        "subgroup": subgroup,
                        "metric": metric,
                        "mean_delta": float(
                            values.mean()
                        ),
                        "ci95_low": float(
                            np.percentile(
                                bootstrap_means,
                                2.5,
                            )
                        ),
                        "ci95_high": float(
                            np.percentile(
                                bootstrap_means,
                                97.5,
                            )
                        ),
                        "probability_power_gate_favorable": float(
                            np.mean(
                                bootstrap_means
                                < 0.0
                            )
                        ),
                        "n_events": int(
                            len(values)
                        ),
                        "n_event_repeats": int(
                            subset[
                                [
                                    "event_id",
                                    "repeat",
                                ]
                            ]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_target_rows": int(
                            np.sum(
                                row_mask
                            )
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def build_consistency_check(
    bootstrap: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    rows = []

    for (
        quantity,
        subgroup,
        metric,
    ), expected in EXPECTED.items():
        row = bootstrap.loc[
            bootstrap[
                "quantity"
            ].eq(quantity)
            & bootstrap[
                "subgroup"
            ].eq(subgroup)
            & bootstrap[
                "metric"
            ].eq(metric)
        ]

        if len(row) != 1:
            rows.append(
                {
                    "quantity": quantity,
                    "subgroup": subgroup,
                    "metric": metric,
                    "observed": np.nan,
                    "expected_reference": expected,
                    "difference": np.nan,
                    "status": "MISSING",
                }
            )
            continue

        observed = float(
            row.iloc[0][
                "mean_delta"
            ]
        )

        difference = (
            observed - expected
        )

        rows.append(
            {
                "quantity": quantity,
                "subgroup": subgroup,
                "metric": metric,
                "observed": observed,
                "expected_reference": expected,
                "difference": difference,
                "status": (
                    "OK"
                    if abs(
                        difference
                    ) <= tolerance
                    else "CHECK"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


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
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
        }
    )


def style_axis(
    ax: plt.Axes,
) -> None:
    ax.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.55,
        alpha=0.28,
    )

    ax.set_axisbelow(
        True
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


def get_plot_row(
    bootstrap: pd.DataFrame,
    quantity: str,
    subgroup: str,
    metric: str,
) -> pd.Series:
    row = bootstrap.loc[
        bootstrap[
            "quantity"
        ].eq(quantity)
        & bootstrap[
            "subgroup"
        ].eq(subgroup)
        & bootstrap[
            "metric"
        ].eq(metric)
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected one row for "
            f"{quantity}/{subgroup}/{metric}; "
            f"found {len(row)}."
        )

    return row.iloc[0]


def plot_panel(
    ax: plt.Axes,
    bootstrap: pd.DataFrame,
    metric: str,
    scale: float,
    panel_letter: str,
    title: str,
    xlabel: str,
    decimals: int,
) -> None:
    y_base = np.arange(
        len(GROUP_ORDER),
        dtype=float,
    )[::-1]

    offset = 0.11

    for (
        quantity,
        marker,
        y_offset,
    ) in (
        (
            "pga",
            "o",
            +offset,
        ),
        (
            "pgv",
            "s",
            -offset,
        ),
    ):
        means = []
        lows = []
        highs = []

        for subgroup in GROUP_ORDER:
            row = get_plot_row(
                bootstrap,
                quantity,
                subgroup,
                metric,
            )

            means.append(
                float(
                    row[
                        "mean_delta"
                    ]
                )
                * scale
            )

            lows.append(
                float(
                    row[
                        "ci95_low"
                    ]
                )
                * scale
            )

            highs.append(
                float(
                    row[
                        "ci95_high"
                    ]
                )
                * scale
            )

        means = np.asarray(
            means,
            dtype=float,
        )

        lows = np.asarray(
            lows,
            dtype=float,
        )

        highs = np.asarray(
            highs,
            dtype=float,
        )

        xerr = np.vstack(
            [
                np.maximum(
                    means - lows,
                    0.0,
                ),
                np.maximum(
                    highs - means,
                    0.0,
                ),
            ]
        )

        y = (
            y_base
            + y_offset
        )

        ax.errorbar(
            means,
            y,
            xerr=xerr,
            fmt=marker,
            markersize=5.0,
            capsize=2.5,
            capthick=0.8,
            elinewidth=1.0,
            linestyle="none",
            label=quantity.upper(),
            zorder=4,
        )

        x_span = max(
            float(
                np.nanmax(
                    highs
                )
                - np.nanmin(
                    lows
                )
            ),
            1e-6,
        )

        text_offset = (
            0.018
            * x_span
        )

        for (
            yi,
            mean,
            low,
            high,
        ) in zip(
            y,
            means,
            lows,
            highs,
        ):
            if mean <= 0.0:
                x_text = (
                    low
                    - text_offset
                )
                horizontal_alignment = (
                    "right"
                )
            else:
                x_text = (
                    high
                    + text_offset
                )
                horizontal_alignment = (
                    "left"
                )

            ax.text(
                x_text,
                yi,
                f"{mean:+.{decimals}f}",
                ha=horizontal_alignment,
                va="center",
                fontsize=7.4,
                clip_on=False,
            )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.set_yticks(
        y_base
    )

    ax.set_yticklabels(
        [
            GROUP_LABELS[
                subgroup
            ]
            for subgroup
            in GROUP_ORDER
        ]
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_title(
        f"({panel_letter}) {title}",
        loc="left",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="best",
        ncol=2,
    )

    x_min, x_max = (
        ax.get_xlim()
    )

    ax.text(
        x_min
        + 0.02
        * (
            x_max - x_min
        ),
        y_base[-1]
        - 0.66,
        "← favors Power Gate",
        ha="left",
        va="center",
        fontsize=7.5,
    )

    ax.text(
        x_max
        - 0.02
        * (
            x_max - x_min
        ),
        y_base[-1]
        - 0.66,
        "favors Base →",
        ha="right",
        va="center",
        fontsize=7.5,
    )

    ax.set_ylim(
        y_base[-1] - 0.82,
        y_base[0] + 0.62,
    )

    style_axis(
        ax
    )


def main() -> None:
    global original_frame

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "runs/"
            "locked_power_gated_test_gamma3/"
            "locked_test_predictions.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "figures/"
            "section_4_10"
        ),
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )

    parser.add_argument(
        "--check-tolerance",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    if (
        args.bootstrap_repetitions
        < 100
    ):
        raise ValueError(
            "--bootstrap-repetitions must be >= 100."
        )

    prediction_path = Path(
        args.predictions
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{prediction_path.resolve()}"
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_frame = (
        load_predictions(
            prediction_path
        )
    )

    event_repeat = (
        build_event_repeat_deltas(
            original_frame
        )
    )

    bootstrap = (
        bootstrap_event_deltas(
            event_repeat,
            repetitions=(
                args.bootstrap_repetitions
            ),
            seed=args.seed,
        )
    )

    check = (
        build_consistency_check(
            bootstrap,
            tolerance=(
                args.check_tolerance
            ),
        )
    )

    print(
        "\n=== Section 4.10 physical subgroup analysis ==="
    )

    print(
        f"Predictions : "
        f"{prediction_path.resolve()}"
    )

    print(
        f"Rows        : "
        f"{len(original_frame)}"
    )

    print(
        "Events      : "
        f"{original_frame['event_id'].nunique()}"
    )

    print(
        "\n=== Paired event-level subgroup deltas ==="
    )

    display = (
        bootstrap[
            [
                "quantity",
                "subgroup",
                "metric",
                "mean_delta",
                "ci95_low",
                "ci95_high",
                "probability_power_gate_favorable",
                "n_events",
                "n_target_rows",
            ]
        ]
        .copy()
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n=== Consistency check against formal locked-test values ==="
    )

    print(
        check.to_string(
            index=False
        )
    )

    n_check = int(
        check[
            "status"
        ].eq("CHECK").sum()
    )

    if n_check > 0:
        print(
            "\nWARNING: "
            f"{n_check} expected locked-test deltas differ "
            f"by more than {args.check_tolerance:g}. "
            "Check that the correct locked_test_predictions.csv "
            "was supplied."
        )

    value_path = (
        out_dir
        / "Fig_4_10_subgroup_bootstrap_values.csv"
    )

    check_path = (
        out_dir
        / "Fig_4_10_consistency_check.csv"
    )

    bootstrap.to_csv(
        value_path,
        index=False,
        encoding="utf-8-sig",
    )

    check.to_csv(
        check_path,
        index=False,
        encoding="utf-8-sig",
    )

    set_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.7, 4.35),
        sharey=True,
    )

    plot_panel(
        axes[0],
        bootstrap,
        metric="mae",
        scale=1.0,
        panel_letter="a",
        title=(
            "MAE gains concentrate on "
            "high-motion and P-wave-reached targets"
        ),
        xlabel=(
            r"$\Delta$MAE = "
            r"MAE$_{\mathrm{PowerGate}}$ "
            r"$-$ MAE$_{\mathrm{Base}}$"
        ),
        decimals=3,
    )

    plot_panel(
        axes[1],
        bootstrap,
        metric="under05",
        scale=100.0,
        panel_letter="b",
        title=(
            "Severe-underestimation reductions "
            "show the same concentration"
        ),
        xlabel=(
            r"$\Delta$Under05 "
            "(percentage points)"
        ),
        decimals=2,
    )

    axes[0].tick_params(
        axis="y",
        labelleft=True,
    )

    axes[1].tick_params(
        axis="y",
        labelleft=False,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Subgroups overlap and are evaluated independently; "
            "error bars denote event-level paired-bootstrap 95% CIs."
        ),
        ha="center",
        va="bottom",
        fontsize=7.7,
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.985,
        bottom=0.20,
        top=0.93,
        wspace=0.16,
    )

    png_path = (
        out_dir
        / "Fig_4_10_physical_subgroup_gains.png"
    )

    svg_path = (
        out_dir
        / "Fig_4_10_physical_subgroup_gains.svg"
    )

    pdf_path = (
        out_dir
        / "Fig_4_10_physical_subgroup_gains.pdf"
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

    plt.close(
        fig
    )

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
        value_path.resolve()
    )

    print(
        check_path.resolve()
    )


if __name__ == "__main__":
    main()
