#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select all phase-ready events for full physical preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the full event list used by physical preprocessing."
    )
    p.add_argument(
        "--events",
        default="data/scedc/phase_ready/phase_ready_events.csv",
    )
    p.add_argument(
        "--out",
        default="data/scedc/full_preprocessing_events.csv",
    )
    p.add_argument("--min-manual-p", type=int, default=1)
    p.add_argument("--min-valid-stations", type=int, default=30)
    p.add_argument("--min-expected-selected", type=int, default=1000)
    args = p.parse_args()

    source = Path(args.events)
    if not source.exists():
        raise FileNotFoundError(
            f"Phase-ready event file not found: {source.resolve()}"
        )

    df = pd.read_csv(source, dtype={"event_id": str})
    required = {
        "event_id",
        "origin_time",
        "first_p_time",
        "n_manual_p",
        "n_valid_download_stations",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["n_manual_p"] = (
        pd.to_numeric(
            df["n_manual_p"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    df["n_valid_download_stations"] = (
        pd.to_numeric(
            df["n_valid_download_stations"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    selected = df.loc[
        df["first_p_time"].notna()
        & (df["n_manual_p"] >= args.min_manual_p)
        & (
            df["n_valid_download_stations"]
            >= args.min_valid_stations
        )
    ].copy()

    selected = (
        selected.sort_values(
            ["origin_time", "event_id"]
        )
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )

    if len(selected) < args.min_expected_selected:
        raise RuntimeError(
            f"Only {len(selected)} events were selected. "
            f"A full list expects at least {args.min_expected_selected}. "
            "Check the input file and filtering columns."
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)

    print("=== Full preprocessing selection ===")
    print(f"Source events            : {len(df)}")
    print(f"Selected full events     : {len(selected)}")
    print(f"Minimum manual P picks   : {args.min_manual_p}")
    print(f"Minimum valid stations   : {args.min_valid_stations}")
    print(f"Output                   : {output.resolve()}")

    if "magnitude" in selected.columns:
        print("\nMagnitude distribution:")
        print(
            selected["magnitude"]
            .describe()
            .to_string()
        )

    if "year" in selected.columns:
        print("\nEvents by year:")
        print(
            selected.groupby("year")
            .size()
            .to_string()
        )

    print(
        "\nNext step:\n"
        "  python 07_preprocess_ground_motion_full_safe.py"
    )


if __name__ == "__main__":
    main()
