#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "data/scedc/model_manifests/"
            "scenario_t0_5s_k5.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/scedc/model_manifests/"
            "scenario_t0_5s_k5_time_clean.csv"
        ),
    )
    parser.add_argument(
        "--output-column",
        default="split_time_clean",
    )
    parser.add_argument(
        "--ridgecrest-sequence-group",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    source = Path(args.manifest)
    destination = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(source)

    frame = pd.read_csv(
        source,
        dtype={"event_id": str},
    )

    required = {
        "event_id",
        "year",
        "sequence_group",
        "magnitude",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    year = pd.to_numeric(
        frame["year"],
        errors="coerce",
    )
    sequence_group = pd.to_numeric(
        frame["sequence_group"],
        errors="coerce",
    )

    split = np.full(
        len(frame),
        "unassigned",
        dtype=object,
    )

    train = year.between(2010, 2018)
    ridgecrest = (
        year.eq(2019)
        & sequence_group.eq(
            args.ridgecrest_sequence_group
        )
    )
    validation = (
        (
            year.eq(2019)
            & ~ridgecrest
        )
        | year.eq(2020)
    )
    test = year.between(2021, 2024)

    split[train.to_numpy()] = "train"
    split[validation.to_numpy()] = "validation"
    split[test.to_numpy()] = "test"
    split[ridgecrest.to_numpy()] = "ridgecrest_ood"

    frame[args.output_column] = split

    unassigned = frame.loc[
        frame[args.output_column].eq("unassigned")
    ]
    if len(unassigned) > 0:
        raise ValueError(
            f"{len(unassigned)} rows remain unassigned."
        )

    ridgecrest_frame = frame.loc[
        frame[args.output_column]
        .eq("ridgecrest_ood")
    ]
    if len(ridgecrest_frame) == 0:
        raise ValueError(
            "Ridgecrest OOD split is empty."
        )

    ridgecrest_mmax = pd.to_numeric(
        ridgecrest_frame["magnitude"],
        errors="coerce",
    ).max()

    if ridgecrest_mmax < 7.0:
        raise ValueError(
            "The selected sequence group does not "
            f"contain the M7.1 event; Mmax={ridgecrest_mmax}."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame.to_csv(
        destination,
        index=False,
    )

    summary = (
        frame.groupby(args.output_column)
        .agg(
            n=("event_id", "size"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            magnitude_min=("magnitude", "min"),
            magnitude_mean=("magnitude", "mean"),
            magnitude_max=("magnitude", "max"),
        )
        .sort_index()
    )

    years = pd.crosstab(
        frame["year"],
        frame[args.output_column],
    )

    sequence_audit = (
        frame.groupby(
            [
                args.output_column,
                "sequence_group",
            ]
        )
        .agg(
            n=("event_id", "size"),
            mmax=("magnitude", "max"),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .reset_index()
        .sort_values(
            [args.output_column, "n"],
            ascending=[True, False],
        )
    )

    summary.to_csv(
        destination.with_name(
            destination.stem
            + "_split_summary.csv"
        )
    )
    years.to_csv(
        destination.with_name(
            destination.stem
            + "_year_counts.csv"
        )
    )
    sequence_audit.to_csv(
        destination.with_name(
            destination.stem
            + "_sequence_audit.csv"
        ),
        index=False,
    )

    labels = [
        "train",
        "validation",
        "test",
        "ridgecrest_ood",
    ]
    groups = {
        label: set(
            subset["sequence_group"].dropna()
        )
        for label, subset
        in frame.groupby(args.output_column)
    }

    overlap_rows = []
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            overlap_rows.append({
                "split_a": first,
                "split_b": second,
                "n_overlapping_sequence_groups": len(
                    groups.get(first, set())
                    & groups.get(second, set())
                ),
            })

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(
        destination.with_name(
            destination.stem
            + "_sequence_overlap.csv"
        ),
        index=False,
    )

    if (
        overlap[
            "n_overlapping_sequence_groups"
        ].max()
        > 0
    ):
        raise ValueError(
            "Sequence overlap detected."
        )

    print(
        "=== Clean chronological + Ridgecrest OOD split ==="
    )
    print(f"Output manifest: {destination.resolve()}")
    print(f"Split column   : {args.output_column}")
    print("\nSplit summary:")
    print(summary.to_string())
    print("\nEvents per year:")
    print(years.to_string())
    print("\nSequence overlap:")
    print(overlap.to_string(index=False))
    print("\nAll overlap checks passed.")


if __name__ == "__main__":
    main()
