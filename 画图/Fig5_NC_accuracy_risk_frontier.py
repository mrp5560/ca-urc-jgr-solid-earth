#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig5_NC_accuracy_risk_frontier.py

Nature Communications-style comparison of CA-URC with causal propagation
and learned baselines under the locked sequence-grouped protocol.

Main figure
-----------
a) PGA: overall MAE versus high-motion-tail MAE
b) PGV: overall MAE versus high-motion-tail MAE
c) PGA: overall MAE versus high-motion-tail severe-underprediction rate
 d) PGV: overall MAE versus high-motion-tail severe-underprediction rate
 e) Paired CA-URC-minus-reference tail-MAE effects for PGA
 f) Paired CA-URC-minus-reference tail-MAE effects for PGV

Inputs
------
1) Learned/strong-baseline metrics:
   runs/final_strong_baselines_reuse_locked/final_strong_baseline_metrics.csv
2) Propagation metrics recomputed under the final CA-URC protocol:
   runs/final_propagation_metrics_caurc_protocol/final_propagation_canonical_metrics.csv
3) Learned-baseline hierarchical bootstrap:
   runs/caurc_sequence_aware_bootstrap/hierarchical_bootstrap_summary.csv
4) Propagation-baseline hierarchical bootstrap:
   runs/final_propagation_metrics_caurc_protocol/caurc_vs_propagation_hierarchical_bootstrap.csv

The script performs no training, model selection, station resampling, or tail
threshold recalculation. It only visualizes locked results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PLOT_METHODS = [
    "nearest_observed",
    "idw_observed",
    "plum_like",
    "graph",
    "tail_weighted_attention",
    "cross_attention",
    "ca_urc",
]

TABLE_METHODS = [
    "median_observed",
    "nearest_observed",
    "idw_observed",
    "plum_like",
    "graph",
    "tail_weighted_attention",
    "cross_attention",
    "ca_urc",
    "source_oracle_ridge",
]

METHOD_LABELS = {
    "median_observed": "Median observed",
    "nearest_observed": "Nearest observed",
    "idw_observed": "IDW observed",
    "plum_like": "PLUM-like",
    "graph": "Graph",
    "tail_weighted_attention": "Tail-weighted Attention",
    "cross_attention": "Cross-Attention Base",
    "ca_urc": "CA-URC",
    "source_oracle_ridge": "Source-oracle Ridge†",
}


SHORT_LABELS = {
    "nearest_observed": "Nearest",
    "idw_observed": "IDW",
    "plum_like": "PLUM-like",
    "graph": "Graph",
    "tail_weighted_attention": "Tail-weighted",
    "cross_attention": "Cross-Attn.",
    "ca_urc": "CA-URC",
}

METHOD_FAMILY = {
    "median_observed": "Causal propagation",
    "nearest_observed": "Causal propagation",
    "idw_observed": "Causal propagation",
    "plum_like": "Causal propagation",
    "graph": "Learned causal",
    "tail_weighted_attention": "Learned causal",
    "cross_attention": "Learned causal",
    "ca_urc": "Proposed",
    "source_oracle_ridge": "Privileged reference",
}

FAMILY_MARKERS = {
    "Causal propagation": "o",
    "Learned causal": "s",
    "Proposed": "*",
    "Privileged reference": "D",
}

FOREST_ORDER = [
    "nearest_observed",
    "idw_observed",
    "plum_like",
    "graph",
    "tail_weighted_attention",
    "cross_attention",
]

LOCKED = {
    ("nearest_observed", "pga", "overall", "mae"): 0.606299,
    ("nearest_observed", "pga", "high_motion_tail", "mae"): 0.956844,
    ("nearest_observed", "pgv", "overall", "mae"): 0.507124,
    ("nearest_observed", "pgv", "high_motion_tail", "mae"): 1.070305,
    ("idw_observed", "pga", "overall", "mae"): 0.818735,
    ("idw_observed", "pga", "high_motion_tail", "mae"): 0.697588,
    ("idw_observed", "pgv", "overall", "mae"): 0.604905,
    ("idw_observed", "pgv", "high_motion_tail", "mae"): 0.805691,
    ("plum_like", "pga", "overall", "mae"): 0.527525,
    ("plum_like", "pga", "high_motion_tail", "mae"): 0.875347,
    ("plum_like", "pgv", "overall", "mae"): 0.534475,
    ("plum_like", "pgv", "high_motion_tail", "mae"): 0.813297,
    ("graph", "pga", "overall", "mae"): 0.309561,
    ("graph", "pga", "high_motion_tail", "mae"): 0.614605,
    ("graph", "pgv", "overall", "mae"): 0.262140,
    ("graph", "pgv", "high_motion_tail", "mae"): 0.759396,
    ("tail_weighted_attention", "pga", "overall", "mae"): 0.339845,
    ("tail_weighted_attention", "pga", "high_motion_tail", "mae"): 0.612139,
    ("tail_weighted_attention", "pgv", "overall", "mae"): 0.285829,
    ("tail_weighted_attention", "pgv", "high_motion_tail", "mae"): 0.740587,
    ("cross_attention", "pga", "overall", "mae"): 0.305941,
    ("cross_attention", "pga", "high_motion_tail", "mae"): 0.639199,
    ("cross_attention", "pgv", "overall", "mae"): 0.265767,
    ("cross_attention", "pgv", "high_motion_tail", "mae"): 0.746959,
    ("ca_urc", "pga", "overall", "mae"): 0.305531,
    ("ca_urc", "pga", "high_motion_tail", "mae"): 0.566925,
    ("ca_urc", "pgv", "overall", "mae"): 0.263486,
    ("ca_urc", "pgv", "high_motion_tail", "mae"): 0.648410,
}


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def canonical_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def normalize_metric_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    require_columns(
        df,
        ["method", "quantity", "population", "metric", "value"],
        name,
    )
    out = df.copy()
    for col in ["method", "quantity", "population", "metric"]:
        out[col] = canonical_text(out[col])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def normalize_bootstrap_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    require_columns(df, ["candidate", "reference", "quantity", "population", "metric"], name)
    out = df.copy()
    for col in ["candidate", "reference", "quantity", "population", "metric"]:
        out[col] = canonical_text(out[col])
    return out


def metric_value(
    metrics: pd.DataFrame,
    method: str,
    quantity: str,
    population: str,
    metric: str,
    allow_missing: bool = False,
) -> float:
    mask = (
        metrics["method"].eq(method)
        & metrics["quantity"].eq(quantity)
        & metrics["population"].eq(population)
        & metrics["metric"].eq(metric)
    )
    sub = metrics.loc[mask, "value"]
    if len(sub) == 0 and allow_missing:
        return float("nan")
    if len(sub) != 1:
        raise RuntimeError(
            f"Expected one metric row for {(method, quantity, population, metric)}, found {len(sub)}"
        )
    return float(sub.iloc[0])


def bootstrap_row(
    boot: pd.DataFrame,
    reference: str,
    quantity: str,
    population: str = "high_motion_tail",
    metric: str = "mae",
) -> pd.Series:
    mask = (
        boot["candidate"].isin({"ca_urc", "a4_under_only"})
        & boot["reference"].eq(reference)
        & boot["quantity"].eq(quantity)
        & boot["population"].eq(population)
        & boot["metric"].eq(metric)
    )
    sub = boot.loc[mask]
    if len(sub) != 1:
        raise RuntimeError(
            f"Expected one bootstrap row for CA-URC vs {reference}, {quantity}/{population}/{metric}; found {len(sub)}"
        )
    return sub.iloc[0]


def first_finite(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    raise KeyError(f"None of the candidate columns were available: {list(names)}")


def caurc_rows_from_bootstrap(boot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quantity in ["pga", "pgv"]:
        for population in ["overall", "high_motion_tail"]:
            for metric in ["mae", "bias", "factor2", "under05"]:
                mask = (
                    boot["candidate"].isin({"ca_urc", "a4_under_only"})
                    & boot["reference"].isin({"cross_attention", "cross_attention_base"})
                    & boot["quantity"].eq(quantity)
                    & boot["population"].eq(population)
                    & boot["metric"].eq(metric)
                )
                sub = boot.loc[mask]
                if len(sub) == 0:
                    continue
                values = []
                for _, row in sub.iterrows():
                    values.append(
                        first_finite(row, ["candidate_value", "ca_urc_value", "caurc_value"])
                    )
                if np.ptp(np.asarray(values, dtype=float)) > 1e-8:
                    raise RuntimeError(
                        f"Inconsistent CA-URC values for {quantity}/{population}/{metric}: {values}"
                    )
                rows.append(
                    {
                        "method": "ca_urc",
                        "quantity": quantity,
                        "population": population,
                        "metric": metric,
                        "value": float(values[0]),
                    }
                )
    return pd.DataFrame(rows)


def load_locked_results(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    learned_metrics = normalize_metric_frame(
        pd.read_csv(args.learned_metrics), "learned metrics"
    )
    propagation_metrics = normalize_metric_frame(
        pd.read_csv(args.propagation_metrics), "propagation metrics"
    )
    learned_boot = normalize_bootstrap_frame(
        pd.read_csv(args.learned_bootstrap), "learned bootstrap"
    )
    propagation_boot = normalize_bootstrap_frame(
        pd.read_csv(args.propagation_bootstrap), "propagation bootstrap"
    )

    propagation_methods = {"median_observed", "nearest_observed", "idw_observed", "plum_like"}
    learned_methods = {
        "median_observed",
        "graph",
        "tail_weighted_attention",
        "cross_attention",
        "source_oracle_ridge",
    }

    pieces = [
        propagation_metrics.loc[propagation_metrics["method"].isin(propagation_methods)].copy(),
        learned_metrics.loc[learned_metrics["method"].isin(learned_methods)].copy(),
        caurc_rows_from_bootstrap(learned_boot),
    ]
    metrics = pd.concat(pieces, ignore_index=True)

    # Remove exact duplicate rows if the same physical methods occur in both files.
    key = ["method", "quantity", "population", "metric"]
    metrics = metrics.drop_duplicates(key, keep="last").reset_index(drop=True)

    boot = pd.concat([propagation_boot, learned_boot], ignore_index=True)
    return metrics, boot


def demo_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    values = {
        # method, quantity: overall mae, tail mae, overall bias, tail bias,
        # overall factor2, tail factor2, overall under05, tail under05
        ("median_observed", "pga"): (0.628999, 1.049199, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("median_observed", "pgv"): (0.470210, 1.176154, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("nearest_observed", "pga"): (0.606299, 0.956844, 0.091082, -0.861053, 0.323929, 0.149218, 0.218415, 0.697828),
        ("nearest_observed", "pgv"): (0.507124, 1.070305, -0.052980, -0.991867, 0.392746, 0.130353, 0.224509, 0.724514),
        ("idw_observed", "pga"): (0.818735, 0.697588, 0.702121, -0.542059, 0.217656, 0.227370, 0.040647, 0.545584),
        ("idw_observed", "pgv"): (0.604905, 0.805691, 0.427055, -0.678445, 0.356451, 0.201787, 0.062991, 0.588901),
        ("plum_like", "pga"): (0.527525, 0.875347, -0.161881, -0.799292, 0.362321, 0.174867, 0.298214, 0.667468),
        ("plum_like", "pgv"): (0.534475, 0.813297, -0.190272, -0.689004, 0.314754, 0.257431, 0.330223, 0.556159),
        ("graph", "pga"): (0.309561, 0.614605, np.nan, np.nan, np.nan, 0.127377, np.nan, 0.621257),
        ("graph", "pgv"): (0.262140, 0.759396, np.nan, np.nan, np.nan, 0.100010, np.nan, 0.765901),
        ("tail_weighted_attention", "pga"): (0.339845, 0.612139, np.nan, np.nan, np.nan, 0.198007, np.nan, 0.605064),
        ("tail_weighted_attention", "pgv"): (0.285829, 0.740587, np.nan, np.nan, np.nan, 0.155702, np.nan, 0.700635),
        ("cross_attention", "pga"): (0.305941, 0.639199, -0.006911, -0.636226, 0.576607, 0.145760, 0.105112, 0.639990),
        ("cross_attention", "pgv"): (0.265767, 0.746959, -0.028981, -0.721670, 0.672054, 0.106318, 0.092388, 0.714567),
        ("ca_urc", "pga"): (0.305531, 0.566925, 0.031684, -0.559836, 0.575201, 0.205067, 0.088460, 0.538049),
        ("ca_urc", "pgv"): (0.263486, 0.648410, 0.005872, -0.613134, 0.669687, 0.205740, 0.077857, 0.603540),
        ("source_oracle_ridge", "pga"): (0.400205, 0.364205, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("source_oracle_ridge", "pgv"): (0.356669, 0.450897, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    }

    for (method, quantity), vals in values.items():
        overall_mae, tail_mae, overall_bias, tail_bias, overall_f2, tail_f2, overall_u, tail_u = vals
        for population, entries in {
            "overall": {"mae": overall_mae, "bias": overall_bias, "factor2": overall_f2, "under05": overall_u},
            "high_motion_tail": {"mae": tail_mae, "bias": tail_bias, "factor2": tail_f2, "under05": tail_u},
        }.items():
            for metric, value in entries.items():
                if np.isfinite(value):
                    rows.append(
                        {
                            "method": method,
                            "quantity": quantity,
                            "population": population,
                            "metric": metric,
                            "value": float(value),
                        }
                    )

    ci = {
        ("nearest_observed", "pga"): (-0.389919, -0.551338, -0.285567),
        ("nearest_observed", "pgv"): (-0.421895, -0.644106, -0.261879),
        ("idw_observed", "pga"): (-0.130663, -0.344466, -0.017570),
        ("idw_observed", "pgv"): (-0.157281, -0.370458, -0.028817),
        ("plum_like", "pga"): (-0.308422, -0.590293, -0.149032),
        ("plum_like", "pgv"): (-0.164887, -0.417971, -0.038253),
        ("graph", "pga"): (-0.047680, -0.102908, 0.016329),
        ("graph", "pgv"): (-0.110986, -0.172269, -0.031820),
        ("tail_weighted_attention", "pga"): (-0.045214, -0.101752, 0.051098),
        ("tail_weighted_attention", "pgv"): (-0.092178, -0.151314, 0.014001),
        ("cross_attention", "pga"): (-0.072274, -0.078435, -0.061038),
        ("cross_attention", "pgv"): (-0.098549, -0.113612, -0.083568),
    }

    boot_rows: list[dict[str, object]] = []
    metrics = pd.DataFrame(rows)
    for (reference, quantity), (delta, lo, hi) in ci.items():
        reference_value = metric_value(
            metrics, reference, quantity, "high_motion_tail", "mae"
        )
        candidate_value = metric_value(
            metrics, "ca_urc", quantity, "high_motion_tail", "mae"
        )
        boot_rows.append(
            {
                "candidate": "ca_urc",
                "reference": reference,
                "quantity": quantity,
                "population": "high_motion_tail",
                "metric": "mae",
                "candidate_value": candidate_value,
                "reference_value": reference_value,
                "point_delta_candidate_minus_reference": delta,
                "ci_lower": lo,
                "ci_upper": hi,
                "relative_change_percent": 100.0 * delta / abs(reference_value),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
        )
    return metrics, pd.DataFrame(boot_rows)


def audit_locked(metrics: pd.DataFrame, tolerance: float = 5e-4) -> None:
    failures = []
    for key, expected in LOCKED.items():
        method, quantity, population, metric = key
        try:
            actual = metric_value(metrics, method, quantity, population, metric)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
            failures.append(
                f"{key}: observed={actual:.6f}, locked={expected:.6f}"
            )
    if failures:
        raise RuntimeError(
            "Locked-value audit failed. Do not use this figure until the input files are reconciled:\n  "
            + "\n  ".join(failures)
        )


def family_colors() -> dict[str, object]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        "Causal propagation": cycle[0],
        "Learned causal": cycle[1],
        "Proposed": cycle[3] if len(cycle) > 3 else cycle[2],
        "Privileged reference": cycle[2],
    }


def pareto_methods(metrics: pd.DataFrame, quantity: str, y_metric: str) -> list[str]:
    pts = []
    for method in PLOT_METHODS:
        x = metric_value(metrics, method, quantity, "overall", "mae")
        y = metric_value(metrics, method, quantity, "high_motion_tail", y_metric)
        pts.append((method, x, y))

    frontier = []
    for method, x, y in pts:
        dominated = False
        for other, xo, yo in pts:
            if other == method:
                continue
            if xo <= x and yo <= y and (xo < x or yo < y):
                dominated = True
                break
        if not dominated:
            frontier.append((method, x, y))
    frontier.sort(key=lambda item: item[1])
    return [item[0] for item in frontier]


LABEL_OFFSETS = {
    ("pga", "mae"): {
        "nearest_observed": (6, 6),
        "idw_observed": (6, -10),
        "plum_like": (6, 4),
        "graph": (7, -11),
        "tail_weighted_attention": (7, 8),
        "cross_attention": (7, 8),
        "ca_urc": (8, -13),
    },
    ("pgv", "mae"): {
        "nearest_observed": (6, 5),
        "idw_observed": (6, -10),
        "plum_like": (6, 6),
        "graph": (8, 12),
        "tail_weighted_attention": (8, -12),
        "cross_attention": (8, 7),
        "ca_urc": (8, -13),
    },
    ("pga", "under05"): {
        "nearest_observed": (6, 5),
        "idw_observed": (6, -10),
        "plum_like": (6, 5),
        "graph": (7, 6),
        "tail_weighted_attention": (7, -12),
        "cross_attention": (7, 7),
        "ca_urc": (8, -13),
    },
    ("pgv", "under05"): {
        "nearest_observed": (6, 5),
        "idw_observed": (6, 5),
        "plum_like": (6, 8),
        "graph": (7, 7),
        "tail_weighted_attention": (7, -12),
        "cross_attention": (7, 7),
        "ca_urc": (8, -13),
    },
}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=3)


def frontier_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    quantity: str,
    y_metric: str,
    title: str,
) -> None:
    colors = family_colors()
    scale = 100.0 if y_metric == "under05" else 1.0

    frontier = pareto_methods(metrics, quantity, y_metric)
    fx = [metric_value(metrics, m, quantity, "overall", "mae") for m in frontier]
    fy = [
        scale * metric_value(metrics, m, quantity, "high_motion_tail", y_metric)
        for m in frontier
    ]
    if len(frontier) >= 2:
        ax.plot(fx, fy, linewidth=1.0, alpha=0.55, zorder=1)

    for method in PLOT_METHODS:
        family = METHOD_FAMILY[method]
        x = metric_value(metrics, method, quantity, "overall", "mae")
        y = scale * metric_value(
            metrics, method, quantity, "high_motion_tail", y_metric
        )
        marker = FAMILY_MARKERS[family]
        size = 125 if method == "ca_urc" else 46
        edge_width = 1.0 if method == "ca_urc" else 0.65
        ax.scatter(
            x,
            y,
            s=size,
            marker=marker,
            facecolor=colors[family],
            edgecolor="white",
            linewidth=edge_width,
            zorder=4 if method == "ca_urc" else 3,
        )
        dx, dy = LABEL_OFFSETS[(quantity, y_metric)][method]
        weight = "bold" if method == "ca_urc" else "normal"
        ax.annotate(
            SHORT_LABELS[method],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.2,
            fontweight=weight,
            va="center",
            zorder=5,
        )

    ax.set_xlabel(r"Catalog-wide MAE ($\log_{10}$ units)")
    if y_metric == "mae":
        ax.set_ylabel(r"High-motion-tail MAE ($\log_{10}$ units)")
    else:
        ax.set_ylabel(r"High-motion-tail $U_{0.5}$ (%)")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    x_values = [
        metric_value(metrics, m, quantity, "overall", "mae")
        for m in PLOT_METHODS
    ]
    y_values = [
        scale * metric_value(metrics, m, quantity, "high_motion_tail", y_metric)
        for m in PLOT_METHODS
    ]
    x_span = max(max(x_values) - min(x_values), 1e-3)
    y_span = max(max(y_values) - min(y_values), 1e-3)
    ax.set_xlim(min(x_values) - 0.05 * x_span, max(x_values) + 0.09 * x_span)
    ax.set_ylim(min(y_values) - 0.07 * y_span, max(y_values) + 0.08 * y_span)
    style_axis(ax)


def combined_bootstrap_row(
    boot: pd.DataFrame, reference: str, quantity: str
) -> pd.Series:
    return bootstrap_row(
        boot,
        reference=reference,
        quantity=quantity,
        population="high_motion_tail",
        metric="mae",
    )


def forest_panel(
    ax: plt.Axes,
    boot: pd.DataFrame,
    quantity: str,
    title: str,
) -> None:
    colors = family_colors()
    y = np.arange(len(FOREST_ORDER), dtype=float)

    for i, reference in enumerate(FOREST_ORDER):
        row = combined_bootstrap_row(boot, reference, quantity)
        delta = first_finite(
            row,
            [
                "point_delta_candidate_minus_reference",
                "mean_delta_caurc_minus_base",
            ],
        )
        lo = first_finite(row, ["ci_lower", "ci95_low"])
        hi = first_finite(row, ["ci_upper", "ci95_high"])
        family = METHOD_FAMILY[reference]
        ax.errorbar(
            delta,
            i,
            xerr=np.asarray([[delta - lo], [hi - delta]]),
            fmt=FAMILY_MARKERS[family],
            markersize=4.2,
            capsize=2.4,
            linewidth=0.9,
            color=colors[family],
            zorder=3,
        )
        ref_value = first_finite(row, ["reference_value", "base_value"])
        reduction = -100.0 * delta / abs(ref_value)
        ax.text(
            hi + 0.015,
            i,
            f"{reduction:.1f}%",
            fontsize=6.1,
            va="center",
        )

    ax.axvline(0.0, linewidth=0.8, linestyle="--", alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in FOREST_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(
        r"CA-URC $-$ reference tail MAE ($\log_{10}$ units)"
    )
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.grid(axis="x", linewidth=0.45, alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", width=0.6, length=3)


def build_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in TABLE_METHODS:
        if method not in set(metrics["method"]):
            continue
        row: dict[str, object] = {
            "Method": METHOD_LABELS[method],
            "Family": METHOD_FAMILY[method],
        }
        for quantity in ["pga", "pgv"]:
            prefix = quantity.upper()
            row[f"{prefix} overall MAE"] = metric_value(
                metrics, method, quantity, "overall", "mae", allow_missing=True
            )
            row[f"{prefix} overall bias"] = metric_value(
                metrics, method, quantity, "overall", "bias", allow_missing=True
            )
            row[f"{prefix} tail MAE"] = metric_value(
                metrics, method, quantity, "high_motion_tail", "mae", allow_missing=True
            )
            row[f"{prefix} tail U0.5"] = metric_value(
                metrics, method, quantity, "high_motion_tail", "under05", allow_missing=True
            )
            row[f"{prefix} tail F2"] = metric_value(
                metrics, method, quantity, "high_motion_tail", "factor2", allow_missing=True
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_contrast_table(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reference in FOREST_ORDER:
        for quantity in ["pga", "pgv"]:
            row = combined_bootstrap_row(boot, reference, quantity)
            delta = first_finite(
                row,
                ["point_delta_candidate_minus_reference", "mean_delta_caurc_minus_base"],
            )
            ref = first_finite(row, ["reference_value", "base_value"])
            cand = first_finite(row, ["candidate_value", "ca_urc_value", "caurc_value"])
            lo = first_finite(row, ["ci_lower", "ci95_low"])
            hi = first_finite(row, ["ci_upper", "ci95_high"])
            p = first_finite(
                row,
                ["two_sided_bootstrap_sign_p"],
            ) if "two_sided_bootstrap_sign_p" in row.index and pd.notna(row["two_sided_bootstrap_sign_p"]) else np.nan
            rows.append(
                {
                    "Reference": METHOD_LABELS[reference],
                    "Family": METHOD_FAMILY[reference],
                    "Quantity": quantity.upper(),
                    "Reference tail MAE": ref,
                    "CA-URC tail MAE": cand,
                    "Difference (CA-URC - reference)": delta,
                    "CI lower": lo,
                    "CI upper": hi,
                    "Relative reduction (%)": -100.0 * delta / abs(ref),
                    "Bootstrap P": p,
                    "CI excludes zero": bool(lo > 0 or hi < 0),
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_latex(table: pd.DataFrame, path: Path, percent_cols: set[str]) -> None:
    display = table.copy()
    for col in display.columns:
        if col in {"Method", "Family", "Reference", "Quantity", "CI excludes zero"}:
            continue
        if pd.api.types.is_numeric_dtype(display[col]):
            if col in percent_cols:
                display[col] = display[col].map(
                    lambda x: "--" if pd.isna(x) else f"{100*x:.1f}\\%"
                )
            elif "P" in col and "PGA" not in col and "PGV" not in col:
                display[col] = display[col].map(
                    lambda x: "--" if pd.isna(x) else f"{x:.4f}"
                )
            elif "reduction" in col.lower():
                display[col] = display[col].map(
                    lambda x: "--" if pd.isna(x) else f"{x:.1f}\\%"
                )
            else:
                display[col] = display[col].map(
                    lambda x: "--" if pd.isna(x) else f"{x:.3f}"
                )
    latex = display.to_latex(index=False, escape=False, longtable=False)
    path.write_text(latex, encoding="utf-8")


def caption_text() -> str:
    return (
        "Fig. 5 | CA-URC occupies a favourable catalog-wide accuracy–tail-risk frontier. "
        "a,b, Catalog-wide mean absolute error (MAE) versus high-motion-tail MAE for PGA and PGV. "
        "c,d, Catalog-wide MAE versus the severe-underprediction rate in the high-motion tail, "
        "U0.5 = Pr(prediction − observation <= −0.5). Lower values on both axes are preferred. "
        "Circles denote strictly causal observation-propagation baselines, squares denote learned causal baselines, "
        "and stars denote CA-URC. Thin lines connect non-dominated operational methods and are included only to guide the eye. "
        "All methods were evaluated on exactly matched target predictions from 224 earthquakes, 22 seismic-sequence groups "
        "and 44,800 target rows. The PGA and PGV tails contained 1,363 predictions from 98 earthquakes and 1,485 predictions "
        "from 87 earthquakes, respectively. PLUM-like denotes an adapted strictly causal propagation reference rather than "
        "an exact implementation of PLUM. Paired effect estimates and complete numerical metrics are reported in Supplementary "
        "Fig. X and Supplementary Table 4, respectively."
    )


def supplementary_caption_text() -> str:
    return (
        "Supplementary Fig. X | Paired high-motion-tail MAE differences between CA-URC and causal propagation or learned "
        "reference methods. a,b, PGA and PGV effects, respectively. Differences are defined as CA-URC minus reference, so "
        "negative values favour CA-URC. Points and horizontal intervals denote event-macro effects and 95% confidence intervals "
        "from 10,000 hierarchical sequence-to-event bootstrap replicates. Percentages give the relative reduction in tail MAE. "
        "All comparisons use exactly matched target-station predictions and the same training-derived high-motion labels."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learned-metrics",
        default=(
            "runs/final_strong_baselines_reuse_locked/"
            "final_strong_baseline_metrics.csv"
        ),
    )
    parser.add_argument(
        "--propagation-metrics",
        default=(
            "runs/final_propagation_metrics_caurc_protocol/"
            "final_propagation_canonical_metrics.csv"
        ),
    )
    parser.add_argument(
        "--learned-bootstrap",
        default=(
            "runs/caurc_sequence_aware_bootstrap/"
            "hierarchical_bootstrap_summary.csv"
        ),
    )
    parser.add_argument(
        "--propagation-bootstrap",
        default=(
            "runs/final_propagation_metrics_caurc_protocol/"
            "caurc_vs_propagation_hierarchical_bootstrap.csv"
        ),
    )
    parser.add_argument("--out-dir", default="figures/nc_section_2_4")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--skip-locked-audit", action="store_true")
    args = parser.parse_args()

    if args.demo:
        metrics, boot = demo_results()
    else:
        for file_name in [
            args.learned_metrics,
            args.propagation_metrics,
            args.learned_bootstrap,
            args.propagation_bootstrap,
        ]:
            if not Path(file_name).exists():
                raise FileNotFoundError(file_name)
        metrics, boot = load_locked_results(args)

    if not args.skip_locked_audit:
        audit_locked(metrics)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Main 2 x 2 frontier figure.
    width = 183.0 / 25.4
    height = 145.0 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width, height))
    frontier_panel(
        axes[0, 0], metrics, "pga", "mae",
        "a  PGA: accuracy versus tail error",
    )
    frontier_panel(
        axes[0, 1], metrics, "pgv", "mae",
        "b  PGV: accuracy versus tail error",
    )
    frontier_panel(
        axes[1, 0], metrics, "pga", "under05",
        "c  PGA: accuracy versus severe underprediction",
    )
    frontier_panel(
        axes[1, 1], metrics, "pgv", "under05",
        "d  PGV: accuracy versus severe underprediction",
    )

    colors = family_colors()
    handles = []
    labels = []
    for family in ["Causal propagation", "Learned causal", "Proposed"]:
        handles.append(
            plt.Line2D(
                [], [], linestyle="none", marker=FAMILY_MARKERS[family],
                markersize=6.2 if family != "Proposed" else 8.5,
                markerfacecolor=colors[family], markeredgecolor="white",
            )
        )
        labels.append(family)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, columnspacing=1.4, handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.09, top=0.925,
        wspace=0.33, hspace=0.38,
    )

    main_png = out / "Fig5_accuracy_risk_frontier.png"
    main_pdf = out / "Fig5_accuracy_risk_frontier.pdf"
    main_svg = out / "Fig5_accuracy_risk_frontier.svg"
    fig.savefig(main_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(main_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(main_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Supplementary two-panel paired-effect figure.
    supp_height = 80.0 / 25.4
    supp, supp_axes = plt.subplots(1, 2, figsize=(width, supp_height))
    forest_panel(
        supp_axes[0], boot, "pga",
        "a  PGA high-motion-tail MAE",
    )
    forest_panel(
        supp_axes[1], boot, "pgv",
        "b  PGV high-motion-tail MAE",
    )
    supp.subplots_adjust(
        left=0.15, right=0.99, bottom=0.22, top=0.88, wspace=0.58,
    )
    supp_png = out / "FigS_paired_tail_mae_effects.png"
    supp_pdf = out / "FigS_paired_tail_mae_effects.pdf"
    supp_svg = out / "FigS_paired_tail_mae_effects.svg"
    supp.savefig(supp_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    supp.savefig(supp_pdf, bbox_inches="tight", facecolor="white")
    supp.savefig(supp_svg, bbox_inches="tight", facecolor="white")
    plt.close(supp)

    table = build_table(metrics)
    contrasts = build_contrast_table(boot)
    table.to_csv(out / "TableS4_baseline_comparison.csv", index=False)
    contrasts.to_csv(out / "TableS4_paired_tail_mae_contrasts.csv", index=False)
    dataframe_to_latex(
        table, out / "TableS4_baseline_comparison.tex",
        percent_cols={"PGA tail U0.5", "PGA tail F2", "PGV tail U0.5", "PGV tail F2"},
    )
    dataframe_to_latex(
        contrasts, out / "TableS4_paired_tail_mae_contrasts.tex",
        percent_cols=set(),
    )

    (out / "Fig5_caption.txt").write_text(caption_text(), encoding="utf-8")
    (out / "FigS_caption.txt").write_text(
        supplementary_caption_text(), encoding="utf-8"
    )
    (out / "run_configuration.json").write_text(
        json.dumps(
            {
                **vars(args),
                "figure_protocol": "locked sequence-grouped",
                "metric_hierarchy": "targets -> repeats -> events",
                "uncertainty": "hierarchical sequence -> event bootstrap",
                "tail_definition_source": "training-derived, locked target-level labels",
                "figure_methods": PLOT_METHODS,
                "table_methods": TABLE_METHODS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved:")
    for output in [
        main_png, main_pdf, main_svg,
        supp_png, supp_pdf, supp_svg,
        out / "Fig5_caption.txt", out / "FigS_caption.txt",
        out / "TableS4_baseline_comparison.csv",
        out / "TableS4_baseline_comparison.tex",
        out / "TableS4_paired_tail_mae_contrasts.csv",
        out / "TableS4_paired_tail_mae_contrasts.tex",
        out / "run_configuration.json",
    ]:
        print(f"  {output.resolve()}")


if __name__ == "__main__":
    main()
