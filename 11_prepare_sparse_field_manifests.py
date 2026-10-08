#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_prepare_sparse_field_manifests.py

Build training-ready scenario manifests from event_splits_v4.csv and full HDF5.

For a scenario T0/K:
  - an input station must have a finite P offset in [0, T0];
  - at least K such triggered stations must exist;
  - after selecting K inputs, at least min_target_stations remain.

The script does not enumerate station combinations. Combinations are sampled
dynamically by the Dataset during training.

Outputs:
  all_events_with_scenario_flags.csv
  scenario_summary.csv
  scenario_t0_<T0>s_k<K>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_scenarios(text: str) -> list[tuple[int, int]]:
    scenarios = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        t0_text, k_text = item.split(":")
        scenarios.append((int(t0_text), int(k_text)))
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    return scenarios


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--splits",
        default="data/scedc/splits_v4/event_splits_v4.csv",
    )
    p.add_argument(
        "--processed-root",
        default="data/scedc/processed_full_v4",
    )
    p.add_argument(
        "--out-dir",
        default="data/scedc/model_manifests_sensitivity",
    )
    p.add_argument(
        "--scenarios",
        default="3:5,5:5,10:3,10:5,10:10",
        help="Comma-separated T0:K pairs.",
    )
    p.add_argument("--min-target-stations", type=int, default=10)
    p.add_argument("--require-version", default="v4_dead_channel_qc")
    args = p.parse_args()

    splits_path = Path(args.splits)
    processed_root = Path(args.processed_root)
    event_root = processed_root / "events"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(splits_path, dtype={"event_id": str})
    scenarios = parse_scenarios(args.scenarios)

    required = {
        "event_id",
        "split_grouped",
        "split_time_extrapolation",
        "sequence_group",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")

    rows = []
    for event in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="inspect scenario eligibility",
    ):
        event_id = str(event.event_id)
        h5_path = event_root / f"{event_id}.h5"

        row = event._asdict()
        row["h5_path"] = str(h5_path.resolve())
        row["h5_exists"] = h5_path.exists()
        row["h5_status"] = "ok"
        row["n_stations_h5"] = 0
        row["preprocessing_version_h5"] = ""

        for t0, k in scenarios:
            row[f"n_triggered_t0_{t0}s"] = 0
            row[f"n_target_pool_t0_{t0}s_k{k}"] = 0
            row[f"eligible_t0_{t0}s_k{k}"] = False

        if not h5_path.exists():
            row["h5_status"] = "missing"
            rows.append(row)
            continue

        try:
            with h5py.File(h5_path, "r") as h5:
                n_stations = int(h5.attrs["n_stations"])
                version = str(
                    h5.attrs.get(
                        "preprocessing_version",
                        "legacy_or_unknown",
                    )
                )
                p_offset = np.asarray(
                    h5["p_offset_sec"][:],
                    dtype=np.float64,
                )

            row["n_stations_h5"] = n_stations
            row["preprocessing_version_h5"] = version

            if args.require_version and version != args.require_version:
                row["h5_status"] = f"wrong_version:{version}"
                rows.append(row)
                continue

            for t0, k in scenarios:
                triggered = int(
                    np.sum(
                        np.isfinite(p_offset)
                        & (p_offset >= -1e-3)
                        & (p_offset <= float(t0))
                    )
                )
                target_pool = max(0, n_stations - k)
                eligible = (
                    triggered >= k
                    and target_pool >= args.min_target_stations
                )

                row[f"n_triggered_t0_{t0}s"] = triggered
                row[f"n_target_pool_t0_{t0}s_k{k}"] = target_pool
                row[f"eligible_t0_{t0}s_k{k}"] = bool(eligible)

        except Exception as exc:
            row["h5_status"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(
        out_dir / "all_events_with_scenario_flags.csv",
        index=False,
    )

    summary_rows = []

    for t0, k in scenarios:
        eligible_column = f"eligible_t0_{t0}s_k{k}"
        scenario = result.loc[
            result["h5_status"].eq("ok")
            & result[eligible_column].astype(bool)
        ].copy()

        scenario_path = out_dir / f"scenario_t0_{t0}s_k{k}.csv"
        scenario.to_csv(scenario_path, index=False)

        for split_column in [
            "split_grouped",
            "split_time_extrapolation",
        ]:
            for split_name, count in (
                scenario[split_column]
                .value_counts()
                .to_dict()
                .items()
            ):
                summary_rows.append({
                    "t0_sec": t0,
                    "input_stations": k,
                    "min_target_stations": args.min_target_stations,
                    "split_column": split_column,
                    "split_name": split_name,
                    "n_events": int(count),
                    "manifest_path": str(scenario_path.resolve()),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        out_dir / "scenario_summary.csv",
        index=False,
    )

    print("=== Sparse-to-field scenario manifests ===")
    print(f"All split events       : {len(df)}")
    print(f"Usable HDF5 events     : {(result['h5_status'] == 'ok').sum()}")
    print(f"Output directory       : {out_dir.resolve()}")

    print("\nScenario counts using split_grouped:")
    grouped = summary.loc[
        summary["split_column"] == "split_grouped"
    ]
    if len(grouped):
        table = grouped.pivot_table(
            index=["t0_sec", "input_stations"],
            columns="split_name",
            values="n_events",
            aggfunc="sum",
            fill_value=0,
        )
        print(table.to_string())

    bad = result.loc[~result["h5_status"].eq("ok")]
    if len(bad):
        print("\nHDF5 status counts:")
        print(bad["h5_status"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
