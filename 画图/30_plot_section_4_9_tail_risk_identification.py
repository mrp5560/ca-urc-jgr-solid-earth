#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30_plot_section_4_9_tail_risk_identification.py

Section 4.9:
Tail-risk scores identify rare future high-motion targets.

Input
-----
runs/locked_power_gated_test_gamma3/locked_test_predictions.csv

Required columns
----------------
event_id
repeat
is_tail_pga
is_tail_pgv
tail_probability_pga
tail_probability_pgv

The locked-test evaluation writes one row per held-out target station and
stores both the true high-motion-tail label and the predicted tail probability.

Figure
------
(a) Precision-recall curves
    - PGA and PGV
    - random-ranking baselines equal to empirical prevalence

(b) Reliability diagram
    - 10 fixed probability bins, matching ECE10
    - ideal y=x line
    - PGA/PGV ECE and Brier score reported

(c) Risk enrichment by predicted-risk decile
    - targets ranked by p_tail
    - ten equal-count risk groups
    - observed high-motion-tail prevalence in each decile
    - dashed empirical-prevalence baselines
    - reports top-decile enrichment (lift)

Outputs
-------
Fig_4_9_tail_risk_identification.png
Fig_4_9_tail_risk_identification.svg
Fig_4_9_tail_risk_identification.pdf
Fig_4_9_gate_metrics_check.csv
Fig_4_9_reliability_bins.csv
Fig_4_9_risk_deciles.csv

Example
-------
python 30_plot_section_4_9_tail_risk_identification.py ^
  --predictions "runs\\locked_power_gated_test_gamma3\\locked_test_predictions.csv" ^
  --out-dir "figures\\section_4_9" ^
  --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)


REQUIRED_COLUMNS = [
    "event_id",
    "repeat",
    "is_tail_pga",
    "is_tail_pgv",
    "tail_probability_pga",
    "tail_probability_pgv",
]

# Locked-test references from the formal evaluation log.
# Used ONLY as consistency checks, never to draw the figure.
EXPECTED = {
    "pga": {
        "prevalence": 0.030424,
        "auprc": 0.532520,
        "auroc": 0.964137,
        "brier": 0.028313,
        "ece10": 0.039144,
        "precision_at_05": 0.422857,
        "recall_at_05": 0.651504,
        "f1_at_05": 0.512850,
    },
    "pgv": {
        "prevalence": 0.033147,
        "auprc": 0.638998,
        "auroc": 0.976701,
        "brier": 0.021445,
        "ece10": 0.024985,
        "precision_at_05": 0.561804,
        "recall_at_05": 0.679461,
        "f1_at_05": 0.615056,
    },
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

    result = frame[
        REQUIRED_COLUMNS
    ].copy()

    result["event_id"] = (
        result["event_id"]
        .astype(str)
    )

    result["repeat"] = pd.to_numeric(
        result["repeat"],
        errors="coerce",
    ).fillna(-1).astype(int)

    for quantity in ("pga", "pgv"):
        result[
            f"is_tail_{quantity}"
        ] = parse_bool(
            result[
                f"is_tail_{quantity}"
            ]
        )

        result[
            f"tail_probability_{quantity}"
        ] = pd.to_numeric(
            result[
                f"tail_probability_{quantity}"
            ],
            errors="coerce",
        )

    numeric = [
        "tail_probability_pga",
        "tail_probability_pgv",
    ]

    bad = result[numeric].isna().any(axis=1)

    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows contain missing tail probabilities."
        )

    for quantity in ("pga", "pgv"):
        p = result[
            f"tail_probability_{quantity}"
        ].to_numpy(dtype=float)

        if (
            np.nanmin(p) < -1e-8
            or np.nanmax(p) > 1.0 + 1e-8
        ):
            raise ValueError(
                f"{quantity.upper()} tail probabilities fall outside [0, 1]."
            )

        result[
            f"tail_probability_{quantity}"
        ] = np.clip(
            p,
            0.0,
            1.0,
        )

    return result


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = 10,
) -> float:
    """
    Same equal-width ECE definition used by the locked evaluation:
        sum_b (n_b / N) * |mean(p_b) - mean(y_b)|
    """
    labels = np.asarray(
        labels,
        dtype=float,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_id = np.digitize(
        probabilities,
        edges[1:-1],
        right=False,
    )

    result = 0.0

    for index in range(number_of_bins):
        mask = (
            bin_id == index
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        result += (
            count
            / len(labels)
            * abs(
                probabilities[mask].mean()
                - labels[mask].mean()
            )
        )

    return float(result)


def compute_gate_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        labels = (
            frame[
                f"is_tail_{quantity}"
            ]
            .to_numpy(dtype=int)
        )

        probabilities = (
            frame[
                f"tail_probability_{quantity}"
            ]
            .to_numpy(dtype=float)
        )

        prevalence = float(
            labels.mean()
        )

        auprc = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

        auroc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

        brier = float(
            brier_score_loss(
                labels,
                probabilities,
            )
        )

        ece10 = (
            expected_calibration_error(
                labels,
                probabilities,
                number_of_bins=10,
            )
        )

        predicted = (
            probabilities >= 0.5
        ).astype(int)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                labels,
                predicted,
                average="binary",
                zero_division=0,
            )
        )

        rows.append(
            {
                "quantity": quantity,
                "n_rows": int(
                    len(labels)
                ),
                "n_positive": int(
                    labels.sum()
                ),
                "prevalence": prevalence,
                "auprc": auprc,
                "auroc": auroc,
                "brier": brier,
                "ece10": ece10,
                "precision_at_05": float(
                    precision
                ),
                "recall_at_05": float(
                    recall
                ),
                "f1_at_05": float(
                    f1
                ),
                "auprc_over_random": (
                    auprc / prevalence
                    if prevalence > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def check_against_expected(
    metrics: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        current = metrics.loc[
            metrics[
                "quantity"
            ].eq(quantity)
        ].iloc[0]

        for metric_name, expected in (
            EXPECTED[
                quantity
            ].items()
        ):
            observed = float(
                current[
                    metric_name
                ]
            )

            difference = (
                observed - expected
            )

            rows.append(
                {
                    "quantity": quantity,
                    "metric": metric_name,
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


def reliability_table(
    frame: pd.DataFrame,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    rows = []

    edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    for quantity in (
        "pga",
        "pgv",
    ):
        labels = (
            frame[
                f"is_tail_{quantity}"
            ]
            .to_numpy(dtype=int)
        )

        probabilities = (
            frame[
                f"tail_probability_{quantity}"
            ]
            .to_numpy(dtype=float)
        )

        bin_id = np.digitize(
            probabilities,
            edges[1:-1],
            right=False,
        )

        for index in range(
            number_of_bins
        ):
            mask = (
                bin_id == index
            )

            count = int(
                mask.sum()
            )

            if count == 0:
                continue

            rows.append(
                {
                    "quantity": quantity,
                    "bin_index": (
                        index + 1
                    ),
                    "bin_low": float(
                        edges[index]
                    ),
                    "bin_high": float(
                        edges[index + 1]
                    ),
                    "n": count,
                    "mean_predicted_probability": float(
                        probabilities[
                            mask
                        ].mean()
                    ),
                    "observed_tail_frequency": float(
                        labels[
                            mask
                        ].mean()
                    ),
                    "absolute_calibration_error": float(
                        abs(
                            probabilities[
                                mask
                            ].mean()
                            - labels[
                                mask
                            ].mean()
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def risk_decile_table(
    frame: pd.DataFrame,
    number_of_groups: int = 10,
) -> pd.DataFrame:
    """
    Construct exactly equal-count risk groups using rank(method='first')
    before qcut. This avoids dropped bins if probabilities contain ties.

    Decile 1 = lowest predicted risk
    Decile 10 = highest predicted risk.
    """
    rows = []

    for quantity in (
        "pga",
        "pgv",
    ):
        probability = (
            frame[
                f"tail_probability_{quantity}"
            ]
            .to_numpy(dtype=float)
        )

        label = (
            frame[
                f"is_tail_{quantity}"
            ]
            .to_numpy(dtype=int)
        )

        working = pd.DataFrame(
            {
                "probability": probability,
                "label": label,
            }
        )

        working["rank"] = (
            working[
                "probability"
            ]
            .rank(
                method="first",
            )
        )

        working[
            "risk_decile"
        ] = pd.qcut(
            working["rank"],
            q=number_of_groups,
            labels=False,
        ) + 1

        prevalence = float(
            working[
                "label"
            ].mean()
        )

        grouped = (
            working.groupby(
                "risk_decile",
                sort=True,
            )
            .agg(
                n=(
                    "label",
                    "size",
                ),
                n_tail=(
                    "label",
                    "sum",
                ),
                mean_probability=(
                    "probability",
                    "mean",
                ),
                median_probability=(
                    "probability",
                    "median",
                ),
                observed_tail_frequency=(
                    "label",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped[
            "quantity"
        ] = quantity

        grouped[
            "overall_prevalence"
        ] = prevalence

        grouped[
            "lift_over_prevalence"
        ] = (
            grouped[
                "observed_tail_frequency"
            ]
            / prevalence
            if prevalence > 0
            else np.nan
        )

        rows.append(
            grouped
        )

    return pd.concat(
        rows,
        ignore_index=True,
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
            "axes.titlesize": 10.1,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
        }
    )


def style_axis(
    ax: plt.Axes,
) -> None:
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.55,
        alpha=0.27,
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


def plot_pr_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    colors = {
        "pga": "#2F5597",
        "pgv": "#C55A11",
    }

    for quantity in (
        "pga",
        "pgv",
    ):
        labels = (
            frame[
                f"is_tail_{quantity}"
            ]
            .to_numpy(dtype=int)
        )

        probability = (
            frame[
                f"tail_probability_{quantity}"
            ]
            .to_numpy(dtype=float)
        )

        precision, recall, _ = (
            precision_recall_curve(
                labels,
                probability,
            )
        )

        row = metrics.loc[
            metrics[
                "quantity"
            ].eq(quantity)
        ].iloc[0]

        ax.plot(
            recall,
            precision,
            linewidth=1.8,
            color=colors[
                quantity
            ],
            label=(
                f"{quantity.upper()} "
                f"(AUPRC={float(row['auprc']):.3f})"
            ),
        )

        ax.axhline(
            float(
                row[
                    "prevalence"
                ]
            ),
            linestyle=":",
            linewidth=1.0,
            color=colors[
                quantity
            ],
            alpha=0.65,
        )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "(a) Rare high-motion targets are identifiable",
        loc="left",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    pga = metrics.loc[
        metrics[
            "quantity"
        ].eq("pga")
    ].iloc[0]

    pgv = metrics.loc[
        metrics[
            "quantity"
        ].eq("pgv")
    ].iloc[0]

    ax.text(
        0.04,
        0.05,
        (
            "Random AUPRC = prevalence\n"
            f"PGA: {float(pga['prevalence']):.3f} "
            f"→ {float(pga['auprc_over_random']):.1f}× random\n"
            f"PGV: {float(pgv['prevalence']):.3f} "
            f"→ {float(pgv['auprc_over_random']):.1f}× random"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    style_axis(
        ax
    )


def plot_reliability_panel(
    ax: plt.Axes,
    reliability: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    colors = {
        "pga": "#2F5597",
        "pgv": "#C55A11",
    }

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        color="0.35",
        label="Ideal calibration",
    )

    for quantity, marker in (
        ("pga", "o"),
        ("pgv", "s"),
    ):
        subset = (
            reliability.loc[
                reliability[
                    "quantity"
                ].eq(quantity)
            ]
            .sort_values(
                "mean_predicted_probability"
            )
        )

        ax.plot(
            subset[
                "mean_predicted_probability"
            ],
            subset[
                "observed_tail_frequency"
            ],
            marker=marker,
            markersize=4.5,
            linewidth=1.35,
            color=colors[
                quantity
            ],
            label=(
                quantity.upper()
            ),
        )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        r"Mean predicted $p_{\mathrm{tail}}$"
    )

    ax.set_ylabel(
        "Observed tail frequency"
    )

    ax.set_title(
        "(b) Tail-risk probabilities remain informative",
        loc="left",
        fontweight="bold",
    )

    pga = metrics.loc[
        metrics[
            "quantity"
        ].eq("pga")
    ].iloc[0]

    pgv = metrics.loc[
        metrics[
            "quantity"
        ].eq("pgv")
    ].iloc[0]

    ax.text(
        0.04,
        0.95,
        (
            f"PGA: ECE={float(pga['ece10']):.3f}, "
            f"Brier={float(pga['brier']):.3f}\n"
            f"PGV: ECE={float(pgv['ece10']):.3f}, "
            f"Brier={float(pgv['brier']):.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    style_axis(
        ax
    )


def plot_decile_panel(
    ax: plt.Axes,
    deciles: pd.DataFrame,
) -> None:
    colors = {
        "pga": "#2F5597",
        "pgv": "#C55A11",
    }

    for quantity, marker in (
        ("pga", "o"),
        ("pgv", "s"),
    ):
        subset = (
            deciles.loc[
                deciles[
                    "quantity"
                ].eq(quantity)
            ]
            .sort_values(
                "risk_decile"
            )
        )

        ax.plot(
            subset[
                "risk_decile"
            ],
            100.0
            * subset[
                "observed_tail_frequency"
            ],
            marker=marker,
            markersize=4.8,
            linewidth=1.6,
            color=colors[
                quantity
            ],
            label=quantity.upper(),
        )

        prevalence = float(
            subset[
                "overall_prevalence"
            ].iloc[0]
        )

        ax.axhline(
            100.0
            * prevalence,
            linestyle=":",
            linewidth=1.0,
            color=colors[
                quantity
            ],
            alpha=0.65,
        )

    ax.set_xticks(
        np.arange(
            1,
            11,
        )
    )

    ax.set_xlabel(
        "Predicted-risk decile\n"
        "(1 = lowest, 10 = highest)"
    )

    ax.set_ylabel(
        "Observed high-motion-tail frequency (%)"
    )

    ax.set_title(
        "(c) Observed risk rises with predicted risk",
        loc="left",
        fontweight="bold",
    )

    annotation_lines = []

    for quantity in (
        "pga",
        "pgv",
    ):
        subset = (
            deciles.loc[
                deciles[
                    "quantity"
                ].eq(quantity)
            ]
            .sort_values(
                "risk_decile"
            )
        )

        top = subset.iloc[-1]

        annotation_lines.append(
            f"{quantity.upper()} Q10: "
            f"{100.0 * float(top['observed_tail_frequency']):.1f}% "
            f"({float(top['lift_over_prevalence']):.1f}× baseline)"
        )

    ax.text(
        0.04,
        0.95,
        "\n".join(
            annotation_lines
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(
            0.02,
            0.76,
        ),
    )

    style_axis(
        ax
    )


def main() -> None:
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
            "section_4_9"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--check-tolerance",
        type=float,
        default=5e-4,
    )

    args = parser.parse_args()

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

    frame = load_predictions(
        prediction_path
    )

    metrics = compute_gate_metrics(
        frame
    )

    check = check_against_expected(
        metrics,
        tolerance=args.check_tolerance,
    )

    reliability = reliability_table(
        frame,
        number_of_bins=10,
    )

    deciles = risk_decile_table(
        frame,
        number_of_groups=10,
    )

    print(
        "\n=== Section 4.9 tail-risk identification ==="
    )

    print(
        f"Predictions: "
        f"{prediction_path.resolve()}"
    )

    print(
        f"Rows       : {len(frame)}"
    )

    print(
        "Events     : "
        f"{frame['event_id'].nunique()}"
    )

    print(
        "Event-repeat units: "
        f"{frame[['event_id', 'repeat']].drop_duplicates().shape[0]}"
    )

    print(
        "\n=== Gate metrics recomputed from locked predictions ==="
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\n=== Consistency check against formal locked-test log ==="
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
            f"{n_check} metric checks differ from the expected "
            f"reference by more than {args.check_tolerance:g}."
        )

    metrics_path = (
        out_dir
        / "Fig_4_9_gate_metrics_check.csv"
    )

    reliability_path = (
        out_dir
        / "Fig_4_9_reliability_bins.csv"
    )

    decile_path = (
        out_dir
        / "Fig_4_9_risk_deciles.csv"
    )

    metrics.merge(
        check[
            [
                "quantity",
                "metric",
                "expected_reference",
                "difference",
                "status",
            ]
        ].pivot(
            index="quantity",
            columns="metric",
            values="status",
        ).reset_index(),
        on="quantity",
        how="left",
    ).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    reliability.to_csv(
        reliability_path,
        index=False,
        encoding="utf-8-sig",
    )

    deciles.to_csv(
        decile_path,
        index=False,
        encoding="utf-8-sig",
    )

    set_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.3, 3.85),
    )

    plot_pr_panel(
        axes[0],
        frame,
        metrics,
    )

    plot_reliability_panel(
        axes[1],
        reliability,
        metrics,
    )

    plot_decile_panel(
        axes[2],
        deciles,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        bottom=0.19,
        top=0.94,
        wspace=0.29,
    )

    png_path = (
        out_dir
        / "Fig_4_9_tail_risk_identification.png"
    )

    svg_path = (
        out_dir
        / "Fig_4_9_tail_risk_identification.svg"
    )

    pdf_path = (
        out_dir
        / "Fig_4_9_tail_risk_identification.pdf"
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
        metrics_path.resolve()
    )

    print(
        reliability_path.resolve()
    )

    print(
        decile_path.resolve()
    )


if __name__ == "__main__":
    main()
