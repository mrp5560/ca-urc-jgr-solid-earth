#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_make_spatiotemporal_event_splits_v4.py

Hard-constrained, leakage-resistant event splitting for Causal-SeisField.

Differences from v3
-------------------
v3 used a soft dominance penalty. It could still finish with one sequence
occupying more than the requested fraction of validation or test.

v4 uses hard construction:
  1. Build fixed-seed, non-transitive earthquake-sequence groups.
  2. Force groups larger than the evaluation cap into training.
  3. Explicitly fill validation and test to their target event counts and
     minimum group counts using only groups below the cap.
  4. Put every remaining group into training.
  5. Verify all hard constraints before writing outputs.

Because:
    eval_group_cap <= max_eval_group_fraction * eval_target_events
and each evaluation split is filled to at least eval_target_events,
the final largest-group fraction is guaranteed not to exceed the threshold.

Outputs
-------
event_splits_v4.csv
sequence_groups_v4.csv
split_summary_v4.csv
magnitude_split_summary_v4.csv
year_split_summary_v4.csv
major_sequence_holdouts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_vector_km(
    lat0: float,
    lon0: float,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    lat0r = np.radians(lat0)
    lon0r = np.radians(lon0)
    latr = np.radians(lat)
    lonr = np.radians(lon)

    dlat = latr - lat0r
    dlon = lonr - lon0r
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2.0) ** 2
    )
    value = np.clip(value, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value))


def magnitude_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 3.3, 3.6, 4.0, 5.0, 6.0, np.inf],
        labels=[
            "M3.0-3.2",
            "M3.3-3.5",
            "M3.6-3.9",
            "M4.0-4.9",
            "M5.0-5.9",
            "M6+",
        ],
        right=False,
    ).astype(str)


def build_seeded_groups(
    events: pd.DataFrame,
    sequence_days: float,
    sequence_km: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build fixed-seed, non-transitive space-time groups."""
    n_events = len(events)
    assigned = np.full(n_events, -1, dtype=np.int64)

    times_ns = events["origin_time_dt"].astype("int64").to_numpy()
    latitudes = events["latitude"].to_numpy(dtype=float)
    longitudes = events["longitude"].to_numpy(dtype=float)
    magnitudes = events["magnitude"].to_numpy(dtype=float)

    maximum_time_ns = int(sequence_days * 86400.0 * 1e9)
    seed_order = np.lexsort((times_ns, -magnitudes))

    group_rows: list[dict] = []
    group_id = 0

    for seed_index in seed_order:
        if assigned[seed_index] >= 0:
            continue

        candidate_mask = (
            (assigned < 0)
            & (
                np.abs(times_ns - times_ns[seed_index])
                <= maximum_time_ns
            )
        )
        candidate_indices = np.flatnonzero(candidate_mask)

        distances = haversine_vector_km(
            latitudes[seed_index],
            longitudes[seed_index],
            latitudes[candidate_indices],
            longitudes[candidate_indices],
        )
        member_indices = candidate_indices[distances <= sequence_km]

        if not np.any(member_indices == seed_index):
            member_indices = np.append(member_indices, seed_index)

        assigned[member_indices] = group_id
        members = events.iloc[member_indices]

        group_rows.append({
            "sequence_group": group_id,
            "seed_event_id": str(events.iloc[seed_index]["event_id"]),
            "seed_time": events.iloc[seed_index]["origin_time_dt"],
            "seed_latitude": float(latitudes[seed_index]),
            "seed_longitude": float(longitudes[seed_index]),
            "seed_magnitude": float(magnitudes[seed_index]),
            "n_events": int(len(member_indices)),
            "start_time": members["origin_time_dt"].min(),
            "end_time": members["origin_time_dt"].max(),
            "min_magnitude": float(members["magnitude"].min()),
            "median_magnitude": float(members["magnitude"].median()),
            "max_magnitude": float(members["magnitude"].max()),
        })
        group_id += 1

    if (assigned < 0).any():
        raise RuntimeError("Some events were not assigned to a sequence group.")

    return assigned, pd.DataFrame(group_rows)


def build_group_features(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    work = events.copy()
    work["magnitude_bin"] = magnitude_bin(work["magnitude"])

    magnitude_bins = [
        "M3.0-3.2",
        "M3.3-3.5",
        "M3.6-3.9",
        "M4.0-4.9",
        "M5.0-5.9",
        "M6+",
    ]

    rows: list[dict] = []
    for group_id, group in work.groupby("sequence_group", sort=False):
        counts = group["magnitude_bin"].value_counts()
        row = {
            "sequence_group": int(group_id),
            "n_events": int(len(group)),
            "start_time": group["origin_time_dt"].min(),
            "end_time": group["origin_time_dt"].max(),
            "mean_latitude": float(group["latitude"].mean()),
            "mean_longitude": float(group["longitude"].mean()),
            "min_magnitude": float(group["magnitude"].min()),
            "median_magnitude": float(group["magnitude"].median()),
            "max_magnitude": float(group["magnitude"].max()),
        }
        for bin_name in magnitude_bins:
            row[f"count_{bin_name}"] = int(counts.get(bin_name, 0))
        rows.append(row)

    return pd.DataFrame(rows), magnitude_bins


def choose_best_group(
    candidates: pd.DataFrame,
    split_total: int,
    split_bin_counts: np.ndarray,
    target_events: int,
    target_bin_proportions: np.ndarray,
    count_columns: list[str],
    groups_still_needed: int,
    rng: np.random.Generator,
) -> int:
    """
    Choose the next group for one evaluation split.

    The score favors:
      - reaching the target without large overshoot;
      - matching the full-dataset magnitude distribution;
      - including stronger independent sequences;
      - using smaller groups while the minimum group count is unmet.
    """
    best_group_id: int | None = None
    best_score = np.inf

    random_jitter = {
        int(group_id): float(value)
        for group_id, value in zip(
            candidates["sequence_group"],
            rng.random(len(candidates)),
        )
    }

    for _, row in candidates.iterrows():
        group_id = int(row["sequence_group"])
        group_events = int(row["n_events"])
        new_total = split_total + group_events

        event_gap = abs(target_events - new_total) / max(target_events, 1)
        overshoot = max(0, new_total - target_events) / max(target_events, 1)

        new_bins = (
            split_bin_counts
            + row[count_columns].to_numpy(dtype=float)
        )
        new_proportions = new_bins / max(float(new_total), 1.0)
        magnitude_error = float(
            np.mean(
                np.abs(
                    new_proportions
                    - target_bin_proportions
                )
            )
        )

        # Give evaluation sets at least one reasonably strong sequence.
        magnitude_reward = -0.020 * max(
            0.0,
            float(row["max_magnitude"]) - 4.0,
        )

        # While group diversity is insufficient, favor smaller groups.
        diversity_term = 0.0
        if groups_still_needed > 0:
            diversity_term = (
                0.15
                * group_events
                / max(target_events, 1)
            )

        score = (
            event_gap
            + 0.45 * magnitude_error
            + 0.50 * overshoot
            + magnitude_reward
            + diversity_term
            + 1e-6 * random_jitter[group_id]
        )

        if score < best_score:
            best_score = score
            best_group_id = group_id

    if best_group_id is None:
        raise RuntimeError("No candidate sequence group is available.")

    return best_group_id


def assign_hard_constrained_splits(
    events: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    max_eval_group_fraction: float,
    max_eval_group_events: int,
    min_eval_groups: int,
    seed: int,
) -> tuple[dict[int, str], set[int], int, dict[str, int]]:
    groups, magnitude_bins = build_group_features(events)
    count_columns = [f"count_{name}" for name in magnitude_bins]

    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction],
        dtype=float,
    )
    fractions /= fractions.sum()

    target_train = int(round(fractions[0] * len(events)))
    target_validation = int(np.ceil(fractions[1] * len(events)))
    target_test = int(np.ceil(fractions[2] * len(events)))

    minimum_eval_target = min(target_validation, target_test)
    fraction_cap = int(
        np.floor(
            max_eval_group_fraction
            * minimum_eval_target
        )
    )
    if max_eval_group_events > 0:
        evaluation_group_cap = min(
            fraction_cap,
            max_eval_group_events,
        )
    else:
        evaluation_group_cap = fraction_cap
    evaluation_group_cap = max(1, evaluation_group_cap)

    strongest_group = int(
        groups.sort_values(
            ["max_magnitude", "n_events"],
            ascending=[False, False],
        ).iloc[0]["sequence_group"]
    )

    forced_train_groups = set(
        groups.loc[
            groups["n_events"] > evaluation_group_cap,
            "sequence_group",
        ].astype(int)
    )
    forced_train_groups.add(strongest_group)

    eligible = groups.loc[
        ~groups["sequence_group"].isin(forced_train_groups)
    ].copy()

    if len(eligible) < 2 * min_eval_groups:
        raise RuntimeError(
            "Too few eligible independent sequence groups remain for "
            "validation and test. Reduce --min-eval-groups, decrease the "
            "grouping window, or increase the evaluation group cap."
        )

    full_bin_counts = (
        groups[count_columns]
        .sum(axis=0)
        .to_numpy(dtype=float)
    )
    target_bin_proportions = (
        full_bin_counts / max(full_bin_counts.sum(), 1.0)
    )

    rng = np.random.default_rng(seed)
    assignment: dict[int, str] = {
        group_id: "train"
        for group_id in forced_train_groups
    }

    split_state = {
        "validation": {
            "target": target_validation,
            "total": 0,
            "group_ids": [],
            "bin_counts": np.zeros(len(count_columns), dtype=float),
        },
        "test": {
            "target": target_test,
            "total": 0,
            "group_ids": [],
            "bin_counts": np.zeros(len(count_columns), dtype=float),
        },
    }

    available = eligible.copy()

    # Alternate between the split with the larger normalized deficit.
    while True:
        unfinished = [
            split_name
            for split_name, state in split_state.items()
            if (
                state["total"] < state["target"]
                or len(state["group_ids"]) < min_eval_groups
            )
        ]
        if not unfinished:
            break
        if len(available) == 0:
            raise RuntimeError(
                "Ran out of eligible sequence groups before validation/test "
                "met their hard size and diversity constraints."
            )

        split_name = max(
            unfinished,
            key=lambda name: max(
                (
                    split_state[name]["target"]
                    - split_state[name]["total"]
                )
                / max(split_state[name]["target"], 1),
                (
                    min_eval_groups
                    - len(split_state[name]["group_ids"])
                )
                / max(min_eval_groups, 1),
            ),
        )
        state = split_state[split_name]

        selected_group = choose_best_group(
            candidates=available,
            split_total=int(state["total"]),
            split_bin_counts=state["bin_counts"],
            target_events=int(state["target"]),
            target_bin_proportions=target_bin_proportions,
            count_columns=count_columns,
            groups_still_needed=max(
                0,
                min_eval_groups - len(state["group_ids"]),
            ),
            rng=rng,
        )

        row = available.loc[
            available["sequence_group"] == selected_group
        ].iloc[0]

        assignment[selected_group] = split_name
        state["total"] += int(row["n_events"])
        state["group_ids"].append(selected_group)
        state["bin_counts"] += row[
            count_columns
        ].to_numpy(dtype=float)

        available = available.loc[
            available["sequence_group"] != selected_group
        ].copy()

    # Every unselected eligible group goes to training.
    for group_id in available["sequence_group"].astype(int):
        assignment[int(group_id)] = "train"

    final_targets = {
        "train": target_train,
        "validation": target_validation,
        "test": target_test,
    }

    return (
        assignment,
        forced_train_groups,
        evaluation_group_cap,
        final_targets,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        default="data/scedc/full_preprocessing_events.csv",
    )
    parser.add_argument(
        "--validation-events",
        default=(
            "data/scedc/processed_full_v4/validation/"
            "full_h5_event_audit.csv"
        ),
    )
    parser.add_argument(
        "--processed-index",
        default=(
            "data/scedc/processed_full_v4/"
            "processed_event_index.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/scedc/splits_v4",
    )
    parser.add_argument("--sequence-days", type=float, default=30.0)
    parser.add_argument("--sequence-km", type=float, default=30.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-eval-group-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--max-eval-group-events",
        type=int,
        default=150,
    )
    parser.add_argument("--min-eval-groups", type=int, default=20)
    parser.add_argument("--n-major-holdouts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    events = pd.read_csv(args.events, dtype={"event_id": str})
    audit = pd.read_csv(
        args.validation_events,
        dtype={"event_id": str},
    )

    required_columns = {
        "event_id",
        "origin_time",
        "latitude",
        "longitude",
        "magnitude",
    }
    missing_columns = required_columns.difference(events.columns)
    if missing_columns:
        raise ValueError(
            f"Event metadata missing columns: {sorted(missing_columns)}"
        )
    if not {"event_id", "status"}.issubset(audit.columns):
        raise ValueError(
            "Validation audit must contain event_id and status."
        )

    valid_event_ids = set(
        audit.loc[
            audit["status"]
            .astype(str)
            .str.lower()
            .isin(["ok", "reused"]),
            "event_id",
        ].astype(str)
    )
    clean = events.loc[
        events["event_id"].astype(str).isin(valid_event_ids)
    ].copy()

    processed_index_path = Path(args.processed_index)
    if processed_index_path.exists():
        processed = pd.read_csv(
            processed_index_path,
            dtype={"event_id": str},
        )
        keep_columns = [
            column
            for column in [
                "event_id",
                "n_processed",
                "h5_path",
                "preprocessing_version",
            ]
            if column in processed.columns
        ]
        clean = clean.merge(
            processed[keep_columns].drop_duplicates("event_id"),
            on="event_id",
            how="left",
        )

    clean["origin_time_dt"] = pd.to_datetime(
        clean["origin_time"],
        utc=True,
        errors="raise",
    )
    clean["latitude"] = pd.to_numeric(
        clean["latitude"],
        errors="raise",
    )
    clean["longitude"] = pd.to_numeric(
        clean["longitude"],
        errors="raise",
    )
    clean["magnitude"] = pd.to_numeric(
        clean["magnitude"],
        errors="raise",
    )
    clean["year"] = clean["origin_time_dt"].dt.year.astype(int)
    clean = clean.sort_values(
        ["origin_time_dt", "event_id"]
    ).reset_index(drop=True)

    sequence_ids, seed_metadata = build_seeded_groups(
        clean,
        sequence_days=args.sequence_days,
        sequence_km=args.sequence_km,
    )
    clean["sequence_group"] = sequence_ids

    (
        assignment,
        forced_train_groups,
        evaluation_group_cap,
        target_counts,
    ) = assign_hard_constrained_splits(
        clean,
        train_fraction=args.train_fraction,
        validation_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_eval_group_fraction=args.max_eval_group_fraction,
        max_eval_group_events=args.max_eval_group_events,
        min_eval_groups=args.min_eval_groups,
        seed=args.seed,
    )

    clean["split_grouped"] = clean["sequence_group"].map(assignment)
    clean["forced_train_large_sequence"] = clean[
        "sequence_group"
    ].isin(forced_train_groups)

    clean["split_time_extrapolation"] = np.where(
        clean["year"] <= 2018,
        "train",
        np.where(
            clean["year"] <= 2020,
            "validation",
            "test",
        ),
    )

    if clean["split_grouped"].isna().any():
        raise RuntimeError("At least one sequence group has no split assignment.")
    if (
        clean.groupby("sequence_group")["split_grouped"]
        .nunique()
        .max()
        != 1
    ):
        raise RuntimeError("Sequence leakage detected across grouped splits.")

    groups, _ = build_group_features(clean)
    groups = groups.merge(
        seed_metadata[
            [
                "sequence_group",
                "seed_event_id",
                "seed_time",
                "seed_latitude",
                "seed_longitude",
                "seed_magnitude",
            ]
        ],
        on="sequence_group",
        how="left",
        validate="one_to_one",
    )
    groups["split_grouped"] = groups["sequence_group"].map(assignment)
    groups["forced_train_large_sequence"] = groups[
        "sequence_group"
    ].isin(forced_train_groups)
    groups = groups.sort_values(
        ["n_events", "max_magnitude"],
        ascending=[False, False],
    ).reset_index(drop=True)

    groups["major_sequence_rank"] = np.arange(1, len(groups) + 1)
    major_holdouts = groups.head(args.n_major_holdouts).copy()

    split_summary = (
        clean.groupby("split_grouped")
        .agg(
            n_events=("event_id", "size"),
            n_sequence_groups=("sequence_group", "nunique"),
            min_time=("origin_time_dt", "min"),
            max_time=("origin_time_dt", "max"),
            min_magnitude=("magnitude", "min"),
            median_magnitude=("magnitude", "median"),
            max_magnitude=("magnitude", "max"),
        )
        .reset_index()
    )
    split_summary["fraction"] = (
        split_summary["n_events"] / len(clean)
    )
    split_summary["target_events"] = split_summary[
        "split_grouped"
    ].map(target_counts)

    largest_group_sizes = (
        clean.groupby(["split_grouped", "sequence_group"])
        .size()
        .groupby(level=0)
        .max()
    )
    split_summary["largest_group_events"] = split_summary[
        "split_grouped"
    ].map(largest_group_sizes)
    split_summary["largest_group_fraction"] = (
        split_summary["largest_group_events"]
        / split_summary["n_events"]
    )

    evaluation_summary = split_summary.loc[
        split_summary["split_grouped"].isin(
            ["validation", "test"]
        )
    ]
    if (
        evaluation_summary["n_events"]
        < evaluation_summary["target_events"]
    ).any():
        raise RuntimeError(
            "An evaluation split did not reach its target event count."
        )
    if (
        evaluation_summary["largest_group_fraction"]
        > args.max_eval_group_fraction + 1e-12
    ).any():
        raise RuntimeError(
            "Hard dominance constraint failed unexpectedly."
        )
    if (
        evaluation_summary["n_sequence_groups"]
        < args.min_eval_groups
    ).any():
        raise RuntimeError(
            "Hard minimum independent-group constraint failed unexpectedly."
        )

    clean["magnitude_bin"] = magnitude_bin(clean["magnitude"])
    magnitude_summary = (
        clean.groupby(["split_grouped", "magnitude_bin"])
        .size()
        .rename("n_events")
        .reset_index()
    )
    year_summary = (
        clean.groupby(["split_grouped", "year"])
        .size()
        .rename("n_events")
        .reset_index()
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean["origin_time_dt"] = clean["origin_time_dt"].astype(str)
    for column in ["start_time", "end_time", "seed_time"]:
        groups[column] = groups[column].astype(str)
        major_holdouts[column] = major_holdouts[column].astype(str)
    split_summary["min_time"] = split_summary["min_time"].astype(str)
    split_summary["max_time"] = split_summary["max_time"].astype(str)

    clean.to_csv(
        out_dir / "event_splits_v4.csv",
        index=False,
    )
    groups.to_csv(
        out_dir / "sequence_groups_v4.csv",
        index=False,
    )
    split_summary.to_csv(
        out_dir / "split_summary_v4.csv",
        index=False,
    )
    magnitude_summary.to_csv(
        out_dir / "magnitude_split_summary_v4.csv",
        index=False,
    )
    year_summary.to_csv(
        out_dir / "year_split_summary_v4.csv",
        index=False,
    )
    major_holdouts.to_csv(
        out_dir / "major_sequence_holdouts.csv",
        index=False,
    )

    print("=== Hard-constrained leakage-resistant splitting v4 ===")
    print(f"Valid processed events : {len(clean)}")
    print(
        f"Sequence rule         : fixed seed, +/- "
        f"{args.sequence_days:g} days, <= {args.sequence_km:g} km"
    )
    print(f"Sequence groups       : {clean['sequence_group'].nunique()}")
    print(f"Evaluation group cap  : {evaluation_group_cap} events")
    print(f"Forced-train groups   : {len(forced_train_groups)}")

    print("\nGrouped split summary:")
    print(split_summary.to_string(index=False))

    print("\nLargest sequence groups:")
    print(
        groups[
            [
                "sequence_group",
                "seed_event_id",
                "n_events",
                "start_time",
                "end_time",
                "max_magnitude",
                "split_grouped",
                "forced_train_large_sequence",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nMajor sequence holdouts:")
    print(
        major_holdouts[
            [
                "major_sequence_rank",
                "sequence_group",
                "seed_event_id",
                "n_events",
                "max_magnitude",
            ]
        ].to_string(index=False)
    )

    print("\nTime-extrapolation split counts:")
    print(
        clean["split_time_extrapolation"]
        .value_counts()
        .reindex(["train", "validation", "test"])
        .to_string()
    )

    print(f"\nOutputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
