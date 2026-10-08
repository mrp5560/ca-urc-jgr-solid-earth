#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 / Step 2A
Prepare and audit a deterministic station-level OOD protocol.

Purpose
-------
Create a station holdout that is independent of ground-motion labels and
quantify whether the existing grouped train/validation/test event split can
support a strict unseen-station experiment.

Protocol
--------
1. Enumerate station IDs from all events in the supplied scenario manifest.
2. Assign an exact fraction of station IDs to the OOD set by a deterministic
   SHA256 hash ranking. No waveform amplitude, PGA/PGV, magnitude, or model
   result is used in the assignment.
3. Training / validation eligibility:
      - at least K P-wave-reached SEEN stations are available as inputs;
      - after choosing K seen inputs, at least Q SEEN target stations remain.
   OOD stations are completely excluded from both inputs and targets.
4. Test station-OOD eligibility:
      - at least K P-wave-reached SEEN stations are available as inputs;
      - at least min_ood_targets held-out station locations are available;
      - at least min_seen_targets seen target stations remain for a paired
        seen-vs-unseen comparison.
5. This script does NOT train a model and does NOT inspect ground-motion labels.

Outputs
-------
station_holdout_assignment.csv
station_occurrence_summary.csv
event_station_ood_feasibility.csv
scenario_station_ood.csv
station_ood_summary.csv
station_ood_protocol.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def stable_hash(station_id: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{station_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def resolve_h5_path(
    event_id: str,
    row: pd.Series,
    h5_root: Path | None,
) -> Path:
    raw = str(row.get("h5_path", "")).strip()
    if raw:
        p = Path(raw)
        if p.exists():
            return p

    if h5_root is not None:
        p = h5_root / f"{event_id}.h5"
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Cannot resolve HDF5 for event {event_id}. "
        f"manifest h5_path={raw!r}, h5_root={h5_root}"
    )


def read_station_ids(
    h5: h5py.File,
    n_stations: int,
) -> np.ndarray:
    if "station_id" not in h5:
        raise KeyError(
            "HDF5 has no station_id dataset; "
            "strict station-level OOD cannot be constructed."
        )
    try:
        station_ids = h5["station_id"].asstr()[:]
    except Exception:
        station_ids = np.asarray(
            [str(x) for x in h5["station_id"][:]],
            dtype=object,
        )

    if len(station_ids) != n_stations:
        raise RuntimeError(
            "station_id length does not match p_offset_sec length."
        )

    station_ids = np.asarray(
        [str(x).strip() for x in station_ids],
        dtype=object,
    )
    if np.any(station_ids == ""):
        raise RuntimeError("Empty station_id found.")
    return station_ids


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            r"data/scedc/model_manifests/"
            r"scenario_t0_5s_k5.csv"
        ),
    )
    parser.add_argument(
        "--h5-root",
        default=r"data/scedc/processed_full_v4/events",
        help=(
            "Fallback event-HDF5 directory used when manifest h5_path "
            "is a Windows path or otherwise unavailable."
        ),
    )
    parser.add_argument(
        "--split-column",
        default="split_grouped",
    )
    parser.add_argument(
        "--train-label",
        default="train",
    )
    parser.add_argument(
        "--validation-label",
        default="validation",
    )
    parser.add_argument(
        "--test-label",
        default="test",
    )
    parser.add_argument(
        "--t0-sec",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--input-stations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--train-target-stations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--min-ood-targets",
        type=int,
        default=3,
        help=(
            "Minimum held-out stations required for a test event to be "
            "included in the station-OOD evaluation."
        ),
    )
    parser.add_argument(
        "--min-seen-targets",
        type=int,
        default=10,
        help=(
            "Minimum seen non-input targets required for paired "
            "seen-vs-unseen test evaluation."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
    )
    parser.add_argument(
        "--out-dir",
        default=r"data/model_manifests/station_ood",
    )

    args = parser.parse_args()

    if not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError(
            "--holdout-fraction must be between 0 and 0.5."
        )
    if args.input_stations < 1:
        raise ValueError("--input-stations must be >=1.")
    if args.train_target_stations < 1:
        raise ValueError("--train-target-stations must be >=1.")
    if args.min_ood_targets < 1:
        raise ValueError("--min-ood-targets must be >=1.")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    h5_root = (
        Path(args.h5_root)
        if str(args.h5_root).strip()
        else None
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        manifest_path,
        dtype={"event_id": str},
    )

    required = {
        "event_id",
        args.split_column,
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    formal_labels = {
        str(args.train_label),
        str(args.validation_label),
        str(args.test_label),
    }
    manifest = manifest.loc[
        manifest[args.split_column]
        .astype(str)
        .isin(formal_labels)
    ].copy()
    manifest = (
        manifest.drop_duplicates("event_id")
        .reset_index(drop=True)
    )

    print("=== Phase 2 / Step 2A: station-level OOD feasibility ===")
    print(f"Events in manifest : {len(manifest):,}")
    print(
        "Split counts       : "
        + ", ".join(
            f"{k}={v}"
            for k, v in manifest[
                args.split_column
            ].astype(str).value_counts().to_dict().items()
        )
    )

    event_records: list[dict[str, Any]] = []
    occurrence_records: list[dict[str, Any]] = []
    resolved_paths: dict[str, str] = {}

    # First pass: enumerate event-station occurrences and P-reach status.
    for i, row in manifest.iterrows():
        event_id = str(row["event_id"])
        split_name = str(row[args.split_column])

        h5_path = resolve_h5_path(
            event_id,
            row,
            h5_root,
        )
        resolved_paths[event_id] = str(
            h5_path.resolve()
        )

        with h5py.File(h5_path, "r") as h5:
            p_offset = np.asarray(
                h5["p_offset_sec"][:],
                dtype=np.float64,
            )
            station_ids = read_station_ids(
                h5,
                len(p_offset),
            )

        reached = (
            np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (p_offset <= float(args.t0_sec))
        )

        for station_id, is_reached in zip(
            station_ids,
            reached,
        ):
            occurrence_records.append(
                {
                    "event_id": event_id,
                    "split": split_name,
                    "station_id": str(station_id),
                    "p_wave_reached_by_snapshot": bool(
                        is_reached
                    ),
                }
            )

        event_records.append(
            {
                "event_id": event_id,
                "split": split_name,
                "h5_path_resolved": str(
                    h5_path.resolve()
                ),
                "n_stations_total": int(
                    len(station_ids)
                ),
                "n_p_reached_total": int(
                    reached.sum()
                ),
            }
        )

        if (
            (i + 1) % 100 == 0
            or (i + 1) == len(manifest)
        ):
            print(
                f"Enumerated {i+1}/{len(manifest)} events | "
                f"station occurrences={len(occurrence_records):,}"
            )

    occurrence = pd.DataFrame(
        occurrence_records
    )
    event_base = pd.DataFrame(
        event_records
    )

    station_ids = sorted(
        occurrence["station_id"]
        .astype(str)
        .unique()
    )
    n_stations = len(station_ids)
    n_holdout = max(
        1,
        int(
            round(
                n_stations
                * float(args.holdout_fraction)
            )
        ),
    )

    # Exact holdout fraction by deterministic hash ranking.
    ranked = sorted(
        station_ids,
        key=lambda sid: (
            stable_hash(sid, args.seed),
            sid,
        ),
    )
    holdout_ids = set(
        ranked[:n_holdout]
    )

    assignment = pd.DataFrame(
        {
            "station_id": station_ids,
        }
    )
    assignment["station_role"] = np.where(
        assignment["station_id"].isin(
            holdout_ids
        ),
        "ood_holdout",
        "seen",
    )
    assignment["hash_rank"] = (
        assignment["station_id"]
        .map(
            {
                sid: rank
                for rank, sid in enumerate(
                    ranked,
                    start=1,
                )
            }
        )
        .astype(int)
    )
    assignment["hash_value"] = assignment[
        "station_id"
    ].map(
        lambda sid: stable_hash(
            sid,
            args.seed,
        )
    )

    assignment_path = (
        out_dir / "station_holdout_assignment.csv"
    )
    assignment.to_csv(
        assignment_path,
        index=False,
    )

    occurrence = occurrence.merge(
        assignment[
            ["station_id", "station_role"]
        ],
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    # Station occurrence audit by split.
    occurrence_summary = (
        occurrence.groupby(
            ["station_id", "station_role", "split"],
            dropna=False,
        )
        .agg(
            n_event_occurrences=(
                "event_id",
                "nunique",
            ),
            n_p_reached_occurrences=(
                "p_wave_reached_by_snapshot",
                "sum",
            ),
        )
        .reset_index()
    )

    station_occurrence_path = (
        out_dir / "station_occurrence_summary.csv"
    )
    occurrence_summary.to_csv(
        station_occurrence_path,
        index=False,
    )

    # Event-level station counts.
    event_role_counts = (
        occurrence.groupby(
            ["event_id", "split", "station_role"],
            dropna=False,
        )
        .agg(
            n_stations=(
                "station_id",
                "size",
            ),
            n_p_reached=(
                "p_wave_reached_by_snapshot",
                "sum",
            ),
        )
        .reset_index()
    )

    total_pivot = (
        event_role_counts.pivot_table(
            index=["event_id", "split"],
            columns="station_role",
            values="n_stations",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(
            columns={
                "seen": "n_seen_stations",
                "ood_holdout": "n_ood_stations",
            }
        )
        .reset_index()
    )

    reached_pivot = (
        event_role_counts.pivot_table(
            index=["event_id", "split"],
            columns="station_role",
            values="n_p_reached",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(
            columns={
                "seen": "n_p_reached_seen",
                "ood_holdout": "n_p_reached_ood",
            }
        )
        .reset_index()
    )

    event_feasibility = (
        event_base.merge(
            total_pivot,
            on=["event_id", "split"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            reached_pivot,
            on=["event_id", "split"],
            how="left",
            validate="one_to_one",
        )
    )

    for col in (
        "n_seen_stations",
        "n_ood_stations",
        "n_p_reached_seen",
        "n_p_reached_ood",
    ):
        if col not in event_feasibility.columns:
            event_feasibility[col] = 0
        event_feasibility[col] = (
            event_feasibility[col]
            .fillna(0)
            .astype(int)
        )

    event_feasibility[
        "n_seen_target_pool_after_k"
    ] = np.maximum(
        0,
        event_feasibility[
            "n_seen_stations"
        ]
        - int(args.input_stations),
    )

    event_feasibility[
        "trainval_seen_only_eligible"
    ] = (
        (
            event_feasibility[
                "n_p_reached_seen"
            ]
            >= int(args.input_stations)
        )
        & (
            event_feasibility[
                "n_seen_target_pool_after_k"
            ]
            >= int(args.train_target_stations)
        )
    )

    event_feasibility[
        "test_ood_eligible"
    ] = (
        (
            event_feasibility[
                "n_p_reached_seen"
            ]
            >= int(args.input_stations)
        )
        & (
            event_feasibility[
                "n_ood_stations"
            ]
            >= int(args.min_ood_targets)
        )
    )

    event_feasibility[
        "test_paired_seen_ood_eligible"
    ] = (
        event_feasibility[
            "test_ood_eligible"
        ]
        & (
            event_feasibility[
                "n_seen_target_pool_after_k"
            ]
            >= int(args.min_seen_targets)
        )
    )

    event_feasibility_path = (
        out_dir / "event_station_ood_feasibility.csv"
    )
    event_feasibility.to_csv(
        event_feasibility_path,
        index=False,
    )

    # Merge flags and server-safe HDF5 path back into the scenario manifest.
    scenario = manifest.copy()
    scenario["h5_path_original"] = (
        scenario["h5_path"]
        if "h5_path" in scenario.columns
        else ""
    )
    scenario["h5_path"] = scenario[
        "event_id"
    ].map(resolved_paths)

    flags = event_feasibility[
        [
            "event_id",
            "n_seen_stations",
            "n_ood_stations",
            "n_p_reached_seen",
            "n_p_reached_ood",
            "n_seen_target_pool_after_k",
            "trainval_seen_only_eligible",
            "test_ood_eligible",
            "test_paired_seen_ood_eligible",
        ]
    ]
    scenario = scenario.merge(
        flags,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    scenario_path = (
        out_dir / "scenario_station_ood.csv"
    )
    scenario.to_csv(
        scenario_path,
        index=False,
    )

    # Summary table.
    summary_rows = []

    summary_rows.append(
        {
            "section": "station_assignment",
            "split": "all",
            "metric": "n_unique_station_ids",
            "value": int(n_stations),
        }
    )
    summary_rows.append(
        {
            "section": "station_assignment",
            "split": "all",
            "metric": "n_seen_station_ids",
            "value": int(
                n_stations - n_holdout
            ),
        }
    )
    summary_rows.append(
        {
            "section": "station_assignment",
            "split": "all",
            "metric": "n_ood_station_ids",
            "value": int(n_holdout),
        }
    )
    summary_rows.append(
        {
            "section": "station_assignment",
            "split": "all",
            "metric": "actual_ood_fraction",
            "value": float(
                n_holdout / n_stations
            ),
        }
    )

    split_mapping = {
        "train": args.train_label,
        "validation": args.validation_label,
        "test": args.test_label,
    }

    for alias, label in split_mapping.items():
        subset = event_feasibility.loc[
            event_feasibility["split"]
            .eq(str(label))
        ].copy()

        summary_rows.extend(
            [
                {
                    "section": "event_feasibility",
                    "split": alias,
                    "metric": "n_events_original",
                    "value": int(
                        len(subset)
                    ),
                },
                {
                    "section": "event_feasibility",
                    "split": alias,
                    "metric": (
                        "n_trainval_seen_only_eligible"
                    ),
                    "value": int(
                        subset[
                            "trainval_seen_only_eligible"
                        ].sum()
                    ),
                },
                {
                    "section": "event_feasibility",
                    "split": alias,
                    "metric": "n_test_ood_eligible",
                    "value": int(
                        subset[
                            "test_ood_eligible"
                        ].sum()
                    ),
                },
                {
                    "section": "event_feasibility",
                    "split": alias,
                    "metric": (
                        "n_test_paired_seen_ood_eligible"
                    ),
                    "value": int(
                        subset[
                            "test_paired_seen_ood_eligible"
                        ].sum()
                    ),
                },
                {
                    "section": "event_feasibility",
                    "split": alias,
                    "metric": "median_seen_stations",
                    "value": float(
                        subset[
                            "n_seen_stations"
                        ].median()
                    )
                    if len(subset)
                    else np.nan,
                },
                {
                    "section": "event_feasibility",
                    "split": alias,
                    "metric": "median_ood_stations",
                    "value": float(
                        subset[
                            "n_ood_stations"
                        ].median()
                    )
                    if len(subset)
                    else np.nan,
                },
                {
                    "section": "event_feasibility",
                    "split": alias,
                    "metric": "median_p_reached_seen",
                    "value": float(
                        subset[
                            "n_p_reached_seen"
                        ].median()
                    )
                    if len(subset)
                    else np.nan,
                },
            ]
        )

    summary = pd.DataFrame(
        summary_rows
    )
    summary_path = (
        out_dir / "station_ood_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    protocol = {
        "manifest": str(
            manifest_path.resolve()
        ),
        "split_column": args.split_column,
        "train_label": args.train_label,
        "validation_label": args.validation_label,
        "test_label": args.test_label,
        "t0_sec": args.t0_sec,
        "input_stations": args.input_stations,
        "train_target_stations": (
            args.train_target_stations
        ),
        "requested_holdout_fraction": (
            args.holdout_fraction
        ),
        "actual_holdout_fraction": float(
            n_holdout / n_stations
        ),
        "n_station_ids": n_stations,
        "n_seen_station_ids": (
            n_stations - n_holdout
        ),
        "n_ood_station_ids": n_holdout,
        "station_assignment_rule": (
            "exact fraction by deterministic SHA256 hash ranking; "
            "assignment uses station_id and seed only"
        ),
        "seed": args.seed,
        "min_ood_targets_test": (
            args.min_ood_targets
        ),
        "min_seen_targets_test": (
            args.min_seen_targets
        ),
        "ground_motion_labels_used_for_split": False,
    }

    protocol_path = (
        out_dir / "station_ood_protocol.json"
    )
    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Station assignment ===")
    print(
        f"Unique station IDs : {n_stations:,}"
    )
    print(
        f"Seen station IDs   : "
        f"{n_stations - n_holdout:,}"
    )
    print(
        f"OOD station IDs    : {n_holdout:,} "
        f"({100*n_holdout/n_stations:.1f}%)"
    )

    print("\n=== Event feasibility ===")
    display = (
        summary.loc[
            summary["section"].eq(
                "event_feasibility"
            )
        ]
        .pivot_table(
            index="metric",
            columns="split",
            values="value",
            aggfunc="first",
        )
    )
    print(display.to_string())

    test_subset = event_feasibility.loc[
        event_feasibility["split"].eq(
            str(args.test_label)
        )
    ]

    paired_count = int(
        test_subset[
            "test_paired_seen_ood_eligible"
        ].sum()
    )
    original_test = int(
        len(test_subset)
    )

    print("\n=== Decision aid ===")
    print(
        f"Paired seen/OOD test events: "
        f"{paired_count}/{original_test} "
        f"({100*paired_count/max(original_test,1):.1f}%)"
    )
    if paired_count >= 100:
        print(
            "Feasibility status: GOOD — sufficient for the next "
            "seen-vs-unseen base-model experiment."
        )
    elif paired_count >= 50:
        print(
            "Feasibility status: USABLE BUT LIMITED — inspect "
            "station/event geography before training."
        )
    else:
        print(
            "Feasibility status: TOO SMALL — do not train yet; "
            "adjust holdout fraction/minimum target requirement first."
        )

    print("\nOutputs:")
    for path in (
        assignment_path,
        station_occurrence_path,
        event_feasibility_path,
        scenario_path,
        summary_path,
        protocol_path,
    ):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
