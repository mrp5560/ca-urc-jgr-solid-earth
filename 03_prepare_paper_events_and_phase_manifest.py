#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_prepare_paper_events_and_phase_manifest.py

1. Merge the SCEDC event catalog with download QC.
2. Keep paper-quality events with at least N valid stations.
3. Build an s5cmd manifest for downloading SCEDC event phase files.

Example:
python 03_prepare_paper_events_and_phase_manifest.py ^
  --event-csv data/scedc/events_m3_2010_2024.csv ^
  --qc-events data/scedc/raw_m3_2010_2025/qc_download_events.csv ^
  --min-valid-stations 30 ^
  --out-events data/scedc/paper_events_m3_min30.csv ^
  --phase-root data/scedc/event_phases ^
  --out-manifest data/scedc/download_event_phases.s5cmd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--event-csv",
        default="data/scedc/events_m3_2010_2024.csv",
    )
    p.add_argument(
        "--qc-events",
        default="data/scedc/raw_m3_2010_2025/qc_download_events.csv",
    )
    p.add_argument("--min-valid-stations", type=int, default=30)
    p.add_argument(
        "--out-events",
        default="data/scedc/paper_events_m3_min30.csv",
    )
    p.add_argument(
        "--phase-root",
        default="data/scedc/event_phases",
    )
    p.add_argument(
        "--out-manifest",
        default="data/scedc/download_event_phases.s5cmd",
    )
    args = p.parse_args()

    catalog = pd.read_csv(args.event_csv, dtype={"event_id": str})
    qc = pd.read_csv(args.qc_events, dtype={"event_id": str})

    required_catalog = {
        "event_id", "origin_time", "latitude", "longitude",
        "depth_km", "magnitude",
    }
    missing = required_catalog.difference(catalog.columns)
    if missing:
        raise ValueError(f"Event catalog is missing columns: {sorted(missing)}")
    if "n_valid" not in qc.columns:
        raise ValueError("QC file must contain column: n_valid")

    qc["n_valid"] = pd.to_numeric(qc["n_valid"], errors="coerce").fillna(0).astype(int)
    selected_qc = qc.loc[
        qc["n_valid"] >= args.min_valid_stations
    ].copy()

    keep_qc = [
        c for c in [
            "event_id", "n_valid", "n_raw", "n_candidates",
            "n_downloaded", "status", "eligible",
        ]
        if c in selected_qc.columns
    ]

    events = catalog.merge(
        selected_qc[keep_qc].drop_duplicates("event_id"),
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    events["origin_time_dt"] = pd.to_datetime(
        events["origin_time"], utc=True, errors="raise"
    )
    events["year"] = events["origin_time_dt"].dt.year.astype(int)
    events["doy"] = events["origin_time_dt"].dt.dayofyear.astype(int)

    phase_root = Path(args.phase_root)
    phase_root.mkdir(parents=True, exist_ok=True)

    phase_paths = []
    manifest_lines = []

    for row in events.itertuples(index=False):
        event_id = str(row.event_id)
        year = int(row.year)
        doy = int(row.doy)

        local_dir = phase_root / str(year) / f"{year}_{doy:03d}"
        local_path = local_dir / f"{event_id}.phase"

        s3_path = (
            f"s3://scedc-pds/event_phases/"
            f"{year}/{year}_{doy:03d}/{event_id}.phase"
        )

        phase_paths.append(str(local_path))
        # s5cmd creates parent directories only if they already exist,
        # so a shell command is also written below to create them first.
        manifest_lines.append(f'cp "{s3_path}" "{local_path}"\n')

    events["phase_path"] = phase_paths
    events = events.drop(columns=["origin_time_dt"])
    events = events.sort_values("origin_time").reset_index(drop=True)

    out_events = Path(args.out_events)
    out_manifest = Path(args.out_manifest)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    events.to_csv(out_events, index=False)
    out_manifest.write_text("".join(manifest_lines), encoding="utf-8")

    # Create all local year/day directories now.
    for pth in phase_paths:
        Path(pth).parent.mkdir(parents=True, exist_ok=True)

    print(f"Catalog events             : {len(catalog)}")
    print(f"QC events                  : {len(qc)}")
    print(f"Minimum valid stations     : {args.min_valid_stations}")
    print(f"Paper-quality events       : {len(events)}")
    print(f"Mean valid stations        : {events['n_valid'].mean():.2f}")
    print(f"Median valid stations      : {events['n_valid'].median():.1f}")
    print(f"Wrote event manifest       : {out_events}")
    print(f"Wrote s5cmd phase manifest : {out_manifest}")

    print("\nMagnitude distribution:")
    print(events["magnitude"].describe().to_string())

    print("\nEvents by year:")
    print(events.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
