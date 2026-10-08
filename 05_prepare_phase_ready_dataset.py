#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_prepare_phase_ready_dataset.py

Convert parsed SCEDC phase results into causal sparse-to-field experiment
manifests. The script uses manual P picks only at this stage.

For each event:
  first_p_time = earliest matched manual P pick
  snapshot_time(T0) = first_p_time + T0
  n_triggered_T0 = number of stations with a manual P pick by snapshot_time

It reports how many events can support each combination of:
  observation window T0 = 3, 5, 10 s
  input station count = 3, 5, 10, 20
  minimum target pool = 10 stations

Inputs:
  data/scedc/paper_events_m3_min30.csv
  data/scedc/phases/station_phase_table.csv
  data/scedc/phases/phase_event_qc.csv

Outputs:
  data/scedc/phase_ready/
    phase_ready_events.csv
    phase_ready_stations.csv
    phase_feasibility_counts.csv
    phase_ready_events_main.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_list(text: str) -> list[int]:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def normalized_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--events",
        default="data/scedc/paper_events_m3_min30.csv",
    )
    p.add_argument(
        "--station-phases",
        default="data/scedc/phases/station_phase_table.csv",
    )
    p.add_argument(
        "--event-qc",
        default="data/scedc/phases/phase_event_qc.csv",
    )
    p.add_argument(
        "--out-dir",
        default="data/scedc/phase_ready",
    )
    p.add_argument("--t0-list", default="3,5,10")
    p.add_argument("--input-counts", default="3,5,10,20")
    p.add_argument("--min-target-stations", type=int, default=10)

    # Default main baseline subset.
    p.add_argument("--main-t0-sec", type=int, default=10)
    p.add_argument("--main-input-stations", type=int, default=10)
    args = p.parse_args()

    events = pd.read_csv(args.events, dtype={"event_id": str})
    stations = pd.read_csv(
        args.station_phases,
        dtype={
            "event_id": str,
            "network": str,
            "station": str,
            "location": str,
            "station_id": str,
        },
    )
    event_qc = pd.read_csv(args.event_qc, dtype={"event_id": str})

    if "arrival_time_p" not in stations.columns:
        raise ValueError(
            "station_phase_table.csv does not contain arrival_time_p. "
            "Run the successful phase parser first."
        )

    for col in ["has_p_pick", "has_s_pick"]:
        if col not in stations.columns:
            stations[col] = False
        else:
            stations[col] = normalized_bool(stations[col])

    stations["arrival_time_p_dt"] = pd.to_datetime(
        stations["arrival_time_p"], utc=True, errors="coerce"
    )
    stations["arrival_time_s_dt"] = pd.to_datetime(
        stations.get("arrival_time_s", pd.Series(index=stations.index, dtype=object)),
        utc=True,
        errors="coerce",
    )

    stations["has_p_pick"] &= stations["arrival_time_p_dt"].notna()
    stations["has_s_pick"] &= stations["arrival_time_s_dt"].notna()

    # One row per downloaded valid station is expected.
    stations = stations.drop_duplicates(
        ["event_id", "network", "station"], keep="first"
    ).copy()

    # Earliest matched manual P pick per event.
    first_p = (
        stations.loc[stations["has_p_pick"]]
        .groupby("event_id", as_index=False)["arrival_time_p_dt"]
        .min()
        .rename(columns={"arrival_time_p_dt": "first_p_time"})
    )

    stations = stations.merge(first_p, on="event_id", how="left")
    stations["p_offset_from_first_sec"] = (
        stations["arrival_time_p_dt"] - stations["first_p_time"]
    ).dt.total_seconds()
    stations["s_offset_from_first_sec"] = (
        stations["arrival_time_s_dt"] - stations["first_p_time"]
    ).dt.total_seconds()

    # Basic event-level counts from the station table.
    counts = stations.groupby("event_id").agg(
        n_station_rows=("station", "size"),
        n_manual_p=("has_p_pick", "sum"),
        n_manual_s=("has_s_pick", "sum"),
    ).reset_index()

    event_ready = events.merge(counts, on="event_id", how="left")
    event_ready = event_ready.merge(first_p, on="event_id", how="left")

    keep_qc = [
        c for c in [
            "event_id", "phase_file_ok", "n_nonempty_lines",
            "n_parsed_phase_lines", "n_valid_download_stations",
            "n_matched_p_stations", "n_matched_s_stations",
            "parse_fraction",
        ]
        if c in event_qc.columns
    ]
    event_ready = event_ready.merge(
        event_qc[keep_qc].drop_duplicates("event_id"),
        on="event_id",
        how="left",
    )

    for c in ["n_station_rows", "n_manual_p", "n_manual_s"]:
        event_ready[c] = (
            pd.to_numeric(event_ready[c], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    if "n_valid_download_stations" not in event_ready.columns:
        event_ready["n_valid_download_stations"] = event_ready["n_station_rows"]
    else:
        event_ready["n_valid_download_stations"] = (
            pd.to_numeric(
                event_ready["n_valid_download_stations"], errors="coerce"
            )
            .fillna(event_ready["n_station_rows"])
            .astype(int)
        )

    t0_values = parse_int_list(args.t0_list)
    input_counts = parse_int_list(args.input_counts)

    feasibility_rows = []

    for t0 in t0_values:
        col = f"n_triggered_manual_t0_{t0}s"

        triggered = (
            stations.loc[
                stations["has_p_pick"]
                & stations["p_offset_from_first_sec"].notna()
                & (stations["p_offset_from_first_sec"] <= float(t0))
                & (stations["p_offset_from_first_sec"] >= -0.001)
            ]
            .groupby("event_id")
            .size()
            .rename(col)
        )

        event_ready = event_ready.merge(
            triggered, on="event_id", how="left"
        )
        event_ready[col] = event_ready[col].fillna(0).astype(int)

        target_col = f"n_target_pool_t0_{t0}s"
        event_ready[target_col] = (
            event_ready["n_valid_download_stations"] - event_ready[col]
        ).clip(lower=0)

        for n_input in input_counts:
            eligible_col = f"eligible_t0_{t0}s_input_{n_input}"
            event_ready[eligible_col] = (
                event_ready["first_p_time"].notna()
                & (event_ready[col] >= n_input)
                & (
                    event_ready[target_col]
                    >= args.min_target_stations
                )
            )

            subset = event_ready.loc[event_ready[eligible_col]]
            feasibility_rows.append({
                "t0_sec": t0,
                "input_stations": n_input,
                "min_target_stations": args.min_target_stations,
                "eligible_events": len(subset),
                "fraction_of_all_events": len(subset) / len(event_ready),
                "median_valid_stations": (
                    subset["n_valid_download_stations"].median()
                    if len(subset) else np.nan
                ),
                "median_manual_p_total": (
                    subset["n_manual_p"].median()
                    if len(subset) else np.nan
                ),
                "median_triggered_by_snapshot": (
                    subset[col].median() if len(subset) else np.nan
                ),
                "min_magnitude": (
                    subset["magnitude"].min() if len(subset) else np.nan
                ),
                "max_magnitude": (
                    subset["magnitude"].max() if len(subset) else np.nan
                ),
            })

    event_ready["first_p_time"] = (
        pd.to_datetime(event_ready["first_p_time"], utc=True, errors="coerce")
        .astype("string")
    )
    stations["first_p_time"] = (
        pd.to_datetime(stations["first_p_time"], utc=True, errors="coerce")
        .astype("string")
    )
    stations["arrival_time_p_dt"] = stations["arrival_time_p_dt"].astype("string")
    stations["arrival_time_s_dt"] = stations["arrival_time_s_dt"].astype("string")

    feasibility = pd.DataFrame(feasibility_rows)

    main_col = (
        f"eligible_t0_{args.main_t0_sec}s_"
        f"input_{args.main_input_stations}"
    )
    if main_col not in event_ready.columns:
        raise ValueError(
            f"Main setting {main_col} is not in the requested t0/input grid."
        )
    main_events = event_ready.loc[event_ready[main_col]].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_ready.to_csv(out_dir / "phase_ready_events.csv", index=False)
    stations.to_csv(out_dir / "phase_ready_stations.csv", index=False)
    feasibility.to_csv(
        out_dir / "phase_feasibility_counts.csv", index=False
    )
    main_events.to_csv(
        out_dir / "phase_ready_events_main.csv", index=False
    )

    print("\n=== Manual phase feasibility ===")
    print(f"All paper events             : {len(event_ready)}")
    print(
        "Events with >=1 matched P    : "
        f"{int((event_ready['n_manual_p'] >= 1).sum())}"
    )
    print(
        "Events with >=3 matched P    : "
        f"{int((event_ready['n_manual_p'] >= 3).sum())}"
    )
    print(
        "Events with >=5 matched P    : "
        f"{int((event_ready['n_manual_p'] >= 5).sum())}"
    )
    print(
        "Events with >=10 matched P   : "
        f"{int((event_ready['n_manual_p'] >= 10).sum())}"
    )
    print(
        "Events with >=20 matched P   : "
        f"{int((event_ready['n_manual_p'] >= 20).sum())}"
    )

    print("\nCausal snapshot feasibility:")
    show = feasibility[
        [
            "t0_sec", "input_stations", "eligible_events",
            "fraction_of_all_events",
            "median_triggered_by_snapshot",
        ]
    ].copy()
    show["fraction_of_all_events"] = (
        100.0 * show["fraction_of_all_events"]
    )
    print(show.to_string(index=False, formatters={
        "fraction_of_all_events": lambda x: f"{x:.1f}%",
    }))

    print(
        f"\nMain baseline setting        : T0={args.main_t0_sec}s, "
        f"input={args.main_input_stations}, "
        f"targets>={args.min_target_stations}"
    )
    print(f"Main baseline events         : {len(main_events)}")
    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
