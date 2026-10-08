#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select a diverse pilot subset for physical waveform preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def weighted_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy()

    year_count = df.groupby("year")["event_id"].transform("count").astype(float)
    year_weight = 1.0 / np.sqrt(year_count.clip(lower=1.0))
    mag = pd.to_numeric(df["magnitude"], errors="coerce").fillna(3.0)
    mag_weight = np.exp(0.7 * (mag - 3.0))
    return df.sample(n=n, weights=year_weight * mag_weight, random_state=seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--events",
        default="data/scedc/phase_ready/phase_ready_events.csv",
    )
    p.add_argument(
        "--eligible-column",
        default="eligible_t0_5s_input_5",
    )
    p.add_argument("--n-events", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--out", default="data/scedc/pilot_events_100.csv")
    args = p.parse_args()

    df = pd.read_csv(args.events, dtype={"event_id": str})
    required = {
        "event_id", "origin_time", "magnitude", "year", "first_p_time",
        args.eligible_column,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    eligible = df.loc[
        as_bool(df[args.eligible_column]) & df["first_p_time"].notna()
    ].copy()
    eligible["magnitude"] = pd.to_numeric(eligible["magnitude"], errors="coerce")
    eligible["year"] = pd.to_numeric(eligible["year"], errors="coerce")
    eligible = eligible.dropna(subset=["magnitude", "year"]).copy()
    eligible["year"] = eligible["year"].astype(int)

    selected_parts = []
    used: set[str] = set()
    strata = [
        (5.0, np.inf, min(15, args.n_events)),
        (4.0, 5.0, min(20, args.n_events)),
        (3.6, 4.0, min(25, args.n_events)),
    ]

    seed = args.seed
    for lo, hi, quota in strata:
        pool = eligible.loc[
            (eligible["magnitude"] >= lo)
            & (eligible["magnitude"] < hi)
            & (~eligible["event_id"].isin(used))
        ]
        take = min(quota, args.n_events - len(used), len(pool))
        part = weighted_sample(pool, take, seed)
        seed += 1
        selected_parts.append(part)
        used.update(part["event_id"].astype(str))

    remaining = args.n_events - len(used)
    pool = eligible.loc[~eligible["event_id"].isin(used)]
    selected_parts.append(weighted_sample(pool, remaining, seed))

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop_duplicates("event_id").head(args.n_events).copy()
    selected = selected.sort_values(
        ["magnitude", "origin_time"], ascending=[False, True]
    ).reset_index(drop=True)
    selected.insert(0, "pilot_order", np.arange(len(selected), dtype=int))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False)

    print("=== Pilot selection ===")
    print(f"Eligible pool   : {len(eligible)}")
    print(f"Selected events : {len(selected)}")
    print(f"Output          : {out.resolve()}")
    print("\nMagnitude distribution:")
    print(selected["magnitude"].describe().to_string())
    print("\nEvents by year:")
    print(selected.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
