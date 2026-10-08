#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
55_diagnose_time_clean_gate_feasibility.py

Diagnose why no chronological epoch-gamma candidate is feasible.

NO retraining.
NO test access.
NO constraint changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True)
    p.add_argument("--overall-budget", type=float, default=0.015)
    p.add_argument("--non-tail-budget", type=float, default=0.005)
    p.add_argument("--bias-limit", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--out-dir",
        default="runs/time_clean_attention_tail_joint_selection/diagnostics",
    )
    args = p.parse_args()

    scan_path = Path(args.scan)
    if not scan_path.exists():
        raise FileNotFoundError(scan_path)

    df = pd.read_csv(scan_path)

    required = {
        "epoch", "gamma",
        "overall_delta_pga", "overall_delta_pgv",
        "non_tail_delta_pga", "non_tail_delta_pgv",
        "bias_pga", "bias_pgv",
        "tail_mae_pga", "tail_mae_pgv",
        "tail_delta_pga", "tail_delta_pgv",
        "mean_tail_mae", "mean_overall_mae",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df["margin_overall_pga"] = (
        args.overall_budget - df["overall_delta_pga"]
    )
    df["margin_overall_pgv"] = (
        args.overall_budget - df["overall_delta_pgv"]
    )
    df["margin_non_tail_pga"] = (
        args.non_tail_budget - df["non_tail_delta_pga"]
    )
    df["margin_non_tail_pgv"] = (
        args.non_tail_budget - df["non_tail_delta_pgv"]
    )
    df["margin_bias_pga"] = (
        args.bias_limit - np.abs(df["bias_pga"])
    )
    df["margin_bias_pgv"] = (
        args.bias_limit - np.abs(df["bias_pgv"])
    )

    constraint_columns = [
        "margin_overall_pga",
        "margin_overall_pgv",
        "margin_non_tail_pga",
        "margin_non_tail_pgv",
        "margin_bias_pga",
        "margin_bias_pgv",
    ]

    df["n_constraints_passed"] = (
        df[constraint_columns] >= 0
    ).sum(axis=1)

    violation = np.maximum(
        -df[constraint_columns].to_numpy(dtype=float),
        0.0,
    )
    df["total_constraint_violation"] = violation.sum(axis=1)
    df["max_constraint_violation"] = violation.max(axis=1)
    df["feasible_recomputed"] = (
        df[constraint_columns] >= 0
    ).all(axis=1)

    print("=== Chronological gate feasibility diagnosis ===")
    print(f"Candidates              : {len(df)}")
    print(
        "Feasible (recomputed)   : "
        f"{int(df['feasible_recomputed'].sum())}"
    )
    print(
        f"Budgets                 : overall={args.overall_budget:.4f}, "
        f"non-tail={args.non_tail_budget:.4f}, "
        f"|bias|<={args.bias_limit:.4f}"
    )

    print("\n=== Pass count by individual constraint ===")
    for col in constraint_columns:
        n = int((df[col] >= 0).sum())
        print(f"{col:24s}: {n:4d}/{len(df)}")

    closest = df.sort_values(
        [
            "n_constraints_passed",
            "total_constraint_violation",
            "mean_tail_mae",
            "mean_overall_mae",
        ],
        ascending=[False, True, True, True],
    ).head(args.top_k)

    closest_path = out_dir / "closest_to_feasible.csv"
    closest.to_csv(closest_path, index=False)

    display_cols = [
        "epoch",
        "gamma",
        "n_constraints_passed",
        "total_constraint_violation",
        "max_constraint_violation",
        "overall_delta_pga",
        "overall_delta_pgv",
        "non_tail_delta_pga",
        "non_tail_delta_pgv",
        "bias_pga",
        "bias_pgv",
        "tail_mae_pga",
        "tail_mae_pgv",
        "tail_delta_pga",
        "tail_delta_pgv",
    ]

    print("\n=== Closest candidates to feasibility ===")
    print(closest[display_cols].to_string(index=False))

    best_tail = df.sort_values(
        ["mean_tail_mae", "mean_overall_mae", "gamma", "epoch"]
    ).head(args.top_k)

    best_tail_path = out_dir / "best_tail_unconstrained.csv"
    best_tail.to_csv(best_tail_path, index=False)

    print("\n=== Best tail candidates regardless of constraints ===")
    print(best_tail[display_cols].to_string(index=False))

    failure_rows = []
    for _, row in df.iterrows():
        failed = []
        for col in constraint_columns:
            if float(row[col]) < 0:
                failed.append(col.replace("margin_", ""))

        failure_rows.append({
            "epoch": int(row["epoch"]),
            "gamma": float(row["gamma"]),
            "failed_constraints": "|".join(failed),
            "n_failed": len(failed),
            "n_constraints_passed": int(row["n_constraints_passed"]),
            "total_constraint_violation": float(
                row["total_constraint_violation"]
            ),
            "mean_tail_mae": float(row["mean_tail_mae"]),
            "mean_overall_mae": float(row["mean_overall_mae"]),
        })

    failures = pd.DataFrame(failure_rows)

    failure_summary = (
        failures.groupby("failed_constraints", dropna=False)
        .size()
        .reset_index(name="n_candidates")
        .sort_values("n_candidates", ascending=False)
    )

    failure_summary_path = out_dir / "failure_pattern_summary.csv"
    failure_summary.to_csv(failure_summary_path, index=False)

    print("\n=== Most common failure patterns ===")
    print(failure_summary.head(20).to_string(index=False))

    print("\nOutputs:")
    print(f"  {closest_path.resolve()}")
    print(f"  {best_tail_path.resolve()}")
    print(f"  {failure_summary_path.resolve()}")
    print("\n2021-2024 test: NOT ACCESSED")
    print("Ridgecrest OOD  : NOT ACCESSED")


if __name__ == "__main__":
    main()
