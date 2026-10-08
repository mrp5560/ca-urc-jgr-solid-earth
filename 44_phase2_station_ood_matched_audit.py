#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 / Step 2C
Post-hoc matched audit for the strict station-level OOD experiment.

No model is retrained.

For every event-repeat, the script forms a one-to-one matching between the
three seen targets and the three unseen-station targets using ONLY variables
available independently of the prediction error / ground-motion label:

    1) prioritize matching P-wave-reached state;
    2) among those assignments, minimize total absolute difference in
       nearest-input distance.

Because each event-repeat has the SAME event and SAME input stations for the
seen and unseen target groups, this matching further controls the main
remaining geometric imbalance.

The script deliberately does NOT match on PGA/PGV, residual, magnitude,
or any outcome-derived quantity.

Primary manuscript aggregation:
matched targets -> repeats -> events.

Outputs
-------
matched_target_pairs.csv
matched_geometry_summary.csv
matched_station_ood_metrics.csv
matched_station_ood_paired_event_deltas.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
LOG10_FACTOR_3 = math.log10(3.0)


def as_bool_scalar(value) -> bool:
    return str(value).strip().lower() in {
        "true", "1", "yes", "y", "t"
    }


def best_assignment(
    seen: pd.DataFrame,
    unseen: pd.DataFrame,
    state_mismatch_penalty_km: float,
) -> list[tuple[int, int]]:
    """
    Return row-index pairs (seen_pos, unseen_pos).

    Lexicographic-like objective:
      - state mismatch receives a very large penalty;
      - distance difference breaks ties / controls geometry.
    """
    if len(seen) != len(unseen):
        raise ValueError("Seen/unseen group sizes must match.")
    n = len(seen)
    if n == 0:
        return []

    seen_d = seen["nearest_input_distance_km"].to_numpy(dtype=float)
    unseen_d = unseen["nearest_input_distance_km"].to_numpy(dtype=float)

    seen_state = (
        seen["target_p_wave_reached"]
        .map(as_bool_scalar)
        .to_numpy(dtype=bool)
    )
    unseen_state = (
        unseen["target_p_wave_reached"]
        .map(as_bool_scalar)
        .to_numpy(dtype=bool)
    )

    best_perm = None
    best_key = None

    for perm in itertools.permutations(range(n)):
        mismatch_count = 0
        distance_cost = 0.0
        penalized_cost = 0.0

        for i, j in enumerate(perm):
            mismatch = int(seen_state[i] != unseen_state[j])
            mismatch_count += mismatch
            dd = abs(seen_d[i] - unseen_d[j])
            distance_cost += dd
            penalized_cost += (
                dd
                + mismatch * float(state_mismatch_penalty_km)
            )

        key = (
            mismatch_count,
            penalized_cost,
            distance_cost,
            perm,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm

    return [
        (i, int(best_perm[i]))
        for i in range(n)
    ]


def element_metric(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> np.ndarray:
    residual = prediction - truth
    absolute = np.abs(residual)

    if metric == "mae":
        return absolute
    if metric == "bias":
        return residual
    if metric == "factor2":
        return (
            absolute <= LOG10_FACTOR_2
        ).astype(float)
    if metric == "factor3":
        return (
            absolute <= LOG10_FACTOR_3
        ).astype(float)
    if metric == "under05":
        return (
            residual <= -0.5
        ).astype(float)

    raise ValueError(metric)


def canonical_metric(
    frame: pd.DataFrame,
    quantity: str,
    metric: str,
) -> float:
    truth = frame[
        f"true_log10_{quantity}"
    ].to_numpy(dtype=float)
    prediction = frame[
        f"pred_log10_{quantity}"
    ].to_numpy(dtype=float)

    values = element_metric(
        truth,
        prediction,
        metric,
    )

    work = frame[
        ["event_id", "repeat"]
    ].copy()
    work["value"] = values

    event_repeat = (
        work.groupby(
            ["event_id", "repeat"],
            sort=False,
        )["value"]
        .mean()
        .reset_index()
    )
    event = (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )["value"]
        .mean()
    )
    return float(event.mean())


def event_values(
    frame: pd.DataFrame,
    quantity: str,
    metric: str,
) -> pd.Series:
    truth = frame[
        f"true_log10_{quantity}"
    ].to_numpy(dtype=float)
    prediction = frame[
        f"pred_log10_{quantity}"
    ].to_numpy(dtype=float)

    values = element_metric(
        truth,
        prediction,
        metric,
    )

    work = frame[
        ["event_id", "repeat"]
    ].copy()
    work["value"] = values

    er = (
        work.groupby(
            ["event_id", "repeat"],
            sort=False,
        )["value"]
        .mean()
        .reset_index()
    )
    return (
        er.groupby(
            "event_id",
            sort=False,
        )["value"]
        .mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=(
            r"runs/phase2_station_ood_base/"
            r"repeated_station_ood_predictions.csv"
        ),
    )
    parser.add_argument(
        "--state-mismatch-penalty-km",
        type=float,
        default=10000.0,
        help=(
            "Large penalty used only to prioritize matching the "
            "P-wave-reached state before distance."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            r"runs/phase2_station_ood_base/"
            r"matched_audit"
        ),
    )
    args = parser.parse_args()

    path = Path(args.predictions)
    if not path.exists():
        raise FileNotFoundError(path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        path,
        dtype={
            "event_id": str,
            "target_station_id": str,
        },
    )

    required = {
        "event_id",
        "repeat",
        "target_role",
        "target_station_id",
        "target_p_wave_reached",
        "nearest_input_distance_km",
        "true_log10_pga",
        "true_log10_pgv",
        "pred_log10_pga",
        "pred_log10_pgv",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Predictions missing columns: {sorted(missing)}"
        )

    expected_roles = {
        "seen_target",
        "unseen_station_target",
    }
    observed_roles = set(
        df["target_role"].astype(str).unique()
    )
    if not expected_roles.issubset(observed_roles):
        raise ValueError(
            f"Expected roles {expected_roles}; got {observed_roles}"
        )

    pair_rows = []
    unmatched_groups = 0

    grouped = df.groupby(
        ["event_id", "repeat"],
        sort=False,
    )

    for (
        event_id,
        repeat,
    ), group in grouped:
        seen = (
            group.loc[
                group["target_role"].eq(
                    "seen_target"
                )
            ]
            .reset_index(drop=True)
        )
        unseen = (
            group.loc[
                group["target_role"].eq(
                    "unseen_station_target"
                )
            ]
            .reset_index(drop=True)
        )

        if (
            len(seen) == 0
            or len(unseen) == 0
            or len(seen) != len(unseen)
        ):
            unmatched_groups += 1
            continue

        assignments = best_assignment(
            seen,
            unseen,
            args.state_mismatch_penalty_km,
        )

        for pair_index, (
            seen_pos,
            unseen_pos,
        ) in enumerate(assignments):
            s = seen.iloc[seen_pos]
            u = unseen.iloc[unseen_pos]

            s_state = as_bool_scalar(
                s["target_p_wave_reached"]
            )
            u_state = as_bool_scalar(
                u["target_p_wave_reached"]
            )

            pair_rows.append(
                {
                    "event_id": str(event_id),
                    "repeat": int(repeat),
                    "pair_index": int(pair_index),
                    "seen_station_id": str(
                        s["target_station_id"]
                    ),
                    "unseen_station_id": str(
                        u["target_station_id"]
                    ),
                    "seen_distance_km": float(
                        s["nearest_input_distance_km"]
                    ),
                    "unseen_distance_km": float(
                        u["nearest_input_distance_km"]
                    ),
                    "absolute_distance_difference_km": abs(
                        float(s["nearest_input_distance_km"])
                        - float(u["nearest_input_distance_km"])
                    ),
                    "seen_p_wave_reached": bool(
                        s_state
                    ),
                    "unseen_p_wave_reached": bool(
                        u_state
                    ),
                    "p_wave_state_match": bool(
                        s_state == u_state
                    ),
                    "seen_true_log10_pga": float(
                        s["true_log10_pga"]
                    ),
                    "unseen_true_log10_pga": float(
                        u["true_log10_pga"]
                    ),
                    "seen_pred_log10_pga": float(
                        s["pred_log10_pga"]
                    ),
                    "unseen_pred_log10_pga": float(
                        u["pred_log10_pga"]
                    ),
                    "seen_true_log10_pgv": float(
                        s["true_log10_pgv"]
                    ),
                    "unseen_true_log10_pgv": float(
                        u["true_log10_pgv"]
                    ),
                    "seen_pred_log10_pgv": float(
                        s["pred_log10_pgv"]
                    ),
                    "unseen_pred_log10_pgv": float(
                        u["pred_log10_pgv"]
                    ),
                }
            )

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise RuntimeError("No matched pairs were produced.")

    pair_path = (
        out_dir / "matched_target_pairs.csv"
    )
    pairs.to_csv(
        pair_path,
        index=False,
    )

    # Long format for exactly the matched target population.
    long_rows = []
    for row in pairs.itertuples(index=False):
        for role, prefix in (
            ("seen_target", "seen"),
            ("unseen_station_target", "unseen"),
        ):
            long_rows.append(
                {
                    "event_id": str(row.event_id),
                    "repeat": int(row.repeat),
                    "pair_index": int(row.pair_index),
                    "target_role": role,
                    "target_station_id": str(
                        getattr(
                            row,
                            f"{prefix}_station_id",
                        )
                    ),
                    "nearest_input_distance_km": float(
                        getattr(
                            row,
                            f"{prefix}_distance_km",
                        )
                    ),
                    "target_p_wave_reached": bool(
                        getattr(
                            row,
                            f"{prefix}_p_wave_reached",
                        )
                    ),
                    "true_log10_pga": float(
                        getattr(
                            row,
                            f"{prefix}_true_log10_pga",
                        )
                    ),
                    "pred_log10_pga": float(
                        getattr(
                            row,
                            f"{prefix}_pred_log10_pga",
                        )
                    ),
                    "true_log10_pgv": float(
                        getattr(
                            row,
                            f"{prefix}_true_log10_pgv",
                        )
                    ),
                    "pred_log10_pgv": float(
                        getattr(
                            row,
                            f"{prefix}_pred_log10_pgv",
                        )
                    ),
                }
            )

    matched = pd.DataFrame(long_rows)

    metrics_rows = []
    for role, group in matched.groupby(
        "target_role",
        sort=False,
    ):
        for quantity in ("pga", "pgv"):
            for metric in (
                "mae",
                "bias",
                "factor2",
                "factor3",
                "under05",
            ):
                metrics_rows.append(
                    {
                        "target_role": role,
                        "quantity": quantity,
                        "metric": metric,
                        "value": canonical_metric(
                            group,
                            quantity,
                            metric,
                        ),
                        "n_events": int(
                            group["event_id"].nunique()
                        ),
                        "n_event_repeats": int(
                            group[
                                ["event_id", "repeat"]
                            ].drop_duplicates().shape[0]
                        ),
                        "n_rows": int(len(group)),
                        "n_unique_stations": int(
                            group[
                                "target_station_id"
                            ].nunique()
                        ),
                    }
                )

    metrics = pd.DataFrame(
        metrics_rows
    )
    metrics_path = (
        out_dir / "matched_station_ood_metrics.csv"
    )
    metrics.to_csv(
        metrics_path,
        index=False,
    )

    delta_rows = []
    for quantity in ("pga", "pgv"):
        for metric in (
            "mae",
            "bias",
            "factor2",
            "factor3",
            "under05",
        ):
            seen = event_values(
                matched.loc[
                    matched["target_role"].eq(
                        "seen_target"
                    )
                ],
                quantity,
                metric,
            )
            unseen = event_values(
                matched.loc[
                    matched["target_role"].eq(
                        "unseen_station_target"
                    )
                ],
                quantity,
                metric,
            )

            common = seen.index.intersection(
                unseen.index
            )
            delta = (
                unseen.loc[common]
                - seen.loc[common]
            )

            delta_rows.append(
                {
                    "quantity": quantity,
                    "metric": metric,
                    "n_paired_events": int(
                        len(common)
                    ),
                    "seen_value": float(
                        seen.loc[common].mean()
                    ),
                    "unseen_value": float(
                        unseen.loc[common].mean()
                    ),
                    "mean_delta_unseen_minus_seen": float(
                        delta.mean()
                    ),
                    "median_delta_unseen_minus_seen": float(
                        delta.median()
                    ),
                }
            )

    deltas = pd.DataFrame(
        delta_rows
    )
    delta_path = (
        out_dir
        / "matched_station_ood_paired_event_deltas.csv"
    )
    deltas.to_csv(
        delta_path,
        index=False,
    )

    geometry = pd.DataFrame(
        [
            {
                "metric": "n_matched_pairs",
                "value": int(len(pairs)),
            },
            {
                "metric": "n_event_repeats",
                "value": int(
                    pairs[
                        ["event_id", "repeat"]
                    ].drop_duplicates().shape[0]
                ),
            },
            {
                "metric": "n_events",
                "value": int(
                    pairs["event_id"].nunique()
                ),
            },
            {
                "metric": "fraction_p_wave_state_matched",
                "value": float(
                    pairs[
                        "p_wave_state_match"
                    ].mean()
                ),
            },
            {
                "metric": "median_seen_distance_km",
                "value": float(
                    pairs[
                        "seen_distance_km"
                    ].median()
                ),
            },
            {
                "metric": "median_unseen_distance_km",
                "value": float(
                    pairs[
                        "unseen_distance_km"
                    ].median()
                ),
            },
            {
                "metric": "median_abs_distance_difference_km",
                "value": float(
                    pairs[
                        "absolute_distance_difference_km"
                    ].median()
                ),
            },
            {
                "metric": "p75_abs_distance_difference_km",
                "value": float(
                    pairs[
                        "absolute_distance_difference_km"
                    ].quantile(0.75)
                ),
            },
            {
                "metric": "mean_abs_distance_difference_km",
                "value": float(
                    pairs[
                        "absolute_distance_difference_km"
                    ].mean()
                ),
            },
        ]
    )
    geometry_path = (
        out_dir / "matched_geometry_summary.csv"
    )
    geometry.to_csv(
        geometry_path,
        index=False,
    )

    print("=== Phase 2 / Step 2C: matched station-OOD audit ===")
    print(
        f"Input rows          : {len(df):,}"
    )
    print(
        f"Matched pairs       : {len(pairs):,}"
    )
    print(
        f"Events              : {pairs['event_id'].nunique():,}"
    )
    print(
        f"Event-repeats       : "
        f"{pairs[['event_id','repeat']].drop_duplicates().shape[0]:,}"
    )
    print(
        f"Unmatched groups    : {unmatched_groups:,}"
    )

    print("\n=== Matching quality ===")
    print(geometry.to_string(index=False))

    print("\n=== Matched canonical metrics ===")
    print(
        metrics.loc[
            metrics["metric"].isin(
                ["mae", "factor2", "under05"]
            )
        ][
            [
                "target_role",
                "quantity",
                "metric",
                "value",
                "n_events",
                "n_rows",
                "n_unique_stations",
            ]
        ].to_string(index=False)
    )

    print("\n=== Matched unseen minus seen deltas ===")
    print(
        deltas.loc[
            deltas["metric"].isin(
                ["mae", "factor2", "under05"]
            )
        ].to_string(index=False)
    )

    print("\nOutputs:")
    for p in (
        pair_path,
        geometry_path,
        metrics_path,
        delta_path,
    ):
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
