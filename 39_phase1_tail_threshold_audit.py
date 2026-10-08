#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 1 / Step 2
Audit the statistical unit used to define high-motion tail thresholds.

The script enumerates each unique event-station pair once from the
T0=5 s, K=5 model-ready manifest and compares three training reference
populations:

1) all_station_unique:
   every station in each training event.

2) possible_target_unique:
   every unique event-station pair that can be a held-out target under
   the K-station input rule.

3) target_sampling_weighted:
   the same possible target pairs weighted by their exact probability
   of being selected as a target in one random K-input/Q-target draw.

Thresholds are computed at Q0.85/Q0.90/Q0.95 by default and then
audited on train/validation/test.

No model is trained and no test labels are used to choose a threshold.
The recommended manuscript definition is exported separately but should
only replace the current definition after inspecting the audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


G_MPS2 = 9.80665


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def horizontal_peak(data: np.ndarray) -> np.ndarray:
    if data.shape[-1] == 0:
        raise ValueError("Empty future waveform window.")
    return np.max(
        np.sqrt(data[:, 0, :] ** 2 + data[:, 1, :] ** 2),
        axis=1,
    )


def weighted_quantile(
    values: np.ndarray,
    quantile: float,
    weights: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[ok]
    weights = weights[ok]
    if values.size == 0:
        return float("nan")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(quantile) * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(values) - 1)
    return float(values[index])


def station_ids_from_h5(h5: h5py.File, n: int) -> np.ndarray:
    if "station_id" in h5:
        try:
            return h5["station_id"].asstr()[:]
        except Exception:
            return np.asarray(
                [str(x) for x in h5["station_id"][:]],
                dtype=object,
            )
    return np.asarray([f"index_{i}" for i in range(n)], dtype=object)


def build_pair_table(
    manifest: pd.DataFrame,
    split_column: str,
    t0_sec: float,
    input_stations: int,
    target_stations: int,
) -> pd.DataFrame:
    rows = []
    total = len(manifest)

    for counter, event in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        event_id = str(event.event_id)
        split_name = str(getattr(event, split_column))
        h5_path = Path(str(event.h5_path))

        if not h5_path.exists():
            raise FileNotFoundError(
                f"Missing HDF5 for event {event_id}: {h5_path}"
            )

        with h5py.File(h5_path, "r") as h5:
            acceleration = np.asarray(
                h5["acceleration"][:],
                dtype=np.float32,
            )
            velocity = np.asarray(
                h5["velocity"][:],
                dtype=np.float32,
            )
            p_offset = np.asarray(
                h5["p_offset_sec"][:],
                dtype=np.float64,
            )
            sampling_rate = float(h5.attrs["sampling_rate_hz"])
            time_zero_index = int(h5.attrs["time_zero_index"])
            ids = station_ids_from_h5(h5, len(p_offset))

        n_stations = int(len(p_offset))
        snapshot = min(
            time_zero_index + int(round(t0_sec * sampling_rate)),
            acceleration.shape[-1] - 1,
        )

        pga = horizontal_peak(acceleration[:, :, snapshot:])
        pgv = horizontal_peak(velocity[:, :, snapshot:])

        triggered = (
            np.isfinite(p_offset)
            & (p_offset >= -1e-3)
            & (p_offset <= float(t0_sec))
        )
        n_triggered = int(triggered.sum())

        if n_triggered < input_stations:
            raise RuntimeError(
                f"Manifest inconsistency: event {event_id} has only "
                f"{n_triggered} P-reached stations, K={input_stations}."
            )

        remaining_after_inputs = n_stations - input_stations
        if remaining_after_inputs < target_stations:
            raise RuntimeError(
                f"Manifest inconsistency: event {event_id} has "
                f"{remaining_after_inputs} stations after K inputs, "
                f"but Q={target_stations}."
            )

        # A station can be held out if:
        # - it is not P-reached (therefore cannot be selected as an input), or
        # - it is P-reached but there are >K P-reached stations, so an input
        #   combination exists that leaves this station out.
        possible_target = (~triggered) | (
            triggered & (n_triggered > input_stations)
        )

        # Exact probability that station j becomes a target in a single
        # random draw: P(j not selected as input) * Q/(N-K).
        q_over_pool = (
            float(target_stations)
            / float(remaining_after_inputs)
        )
        probability_not_input = np.ones(n_stations, dtype=float)
        probability_not_input[triggered] = (
            1.0
            - float(input_stations) / float(n_triggered)
        )
        target_selection_probability = (
            probability_not_input * q_over_pool
        )
        target_selection_probability[~possible_target] = 0.0

        for station_index in range(n_stations):
            rows.append(
                {
                    "event_id": event_id,
                    "split": split_name,
                    "station_index": int(station_index),
                    "station_id": str(ids[station_index]),
                    "n_stations_event": n_stations,
                    "n_p_reached_by_snapshot": n_triggered,
                    "p_offset_sec": (
                        float(p_offset[station_index])
                        if np.isfinite(p_offset[station_index])
                        else np.nan
                    ),
                    "p_wave_reached": bool(triggered[station_index]),
                    "possible_heldout_target": bool(
                        possible_target[station_index]
                    ),
                    "target_selection_probability": float(
                        target_selection_probability[station_index]
                    ),
                    "future_pga_mps2": float(pga[station_index]),
                    "future_pgv_mps": float(pgv[station_index]),
                    "log10_pga": float(
                        np.log10(max(float(pga[station_index]), 1e-10))
                    ),
                    "log10_pgv": float(
                        np.log10(max(float(pgv[station_index]), 1e-12))
                    ),
                }
            )

        if counter % 100 == 0 or counter == total:
            print(
                f"Enumerated {counter}/{total} events | "
                f"pairs={len(rows):,}"
            )

    frame = pd.DataFrame(rows)
    if frame.duplicated(["event_id", "station_index"]).any():
        raise RuntimeError("Duplicate event-station pairs detected.")
    return frame


def threshold_from_reference(
    train: pd.DataFrame,
    source: str,
    quantity: str,
    quantile: float,
) -> float:
    value_col = f"log10_{quantity}"

    if source == "all_station_unique":
        values = train[value_col].to_numpy(dtype=float)
        return float(np.quantile(values, quantile))

    if source == "possible_target_unique":
        subset = train.loc[
            train["possible_heldout_target"].astype(bool)
        ]
        return float(
            np.quantile(
                subset[value_col].to_numpy(dtype=float),
                quantile,
            )
        )

    if source == "target_sampling_weighted":
        subset = train.loc[
            train["possible_heldout_target"].astype(bool)
        ]
        return weighted_quantile(
            subset[value_col].to_numpy(dtype=float),
            quantile,
            subset["target_selection_probability"].to_numpy(dtype=float),
        )

    raise ValueError(source)


def prevalence(
    frame: pd.DataFrame,
    quantity: str,
    threshold: float,
    unit: str,
) -> tuple[float, float, int]:
    value_col = f"log10_{quantity}"
    tail = frame[value_col].to_numpy(dtype=float) >= float(threshold)

    if unit == "all_station_unique":
        weights = np.ones(len(frame), dtype=float)

    elif unit == "possible_target_unique":
        keep = frame["possible_heldout_target"].to_numpy(dtype=bool)
        frame = frame.loc[keep]
        tail = tail[keep]
        weights = np.ones(len(frame), dtype=float)

    elif unit == "target_sampling_weighted":
        keep = frame["possible_heldout_target"].to_numpy(dtype=bool)
        frame = frame.loc[keep]
        tail = tail[keep]
        weights = frame[
            "target_selection_probability"
        ].to_numpy(dtype=float)

    else:
        raise ValueError(unit)

    if len(frame) == 0 or np.sum(weights) <= 0:
        return float("nan"), float("nan"), 0

    weighted_prev = float(
        np.sum(weights * tail.astype(float))
        / np.sum(weights)
    )
    unweighted_prev = float(np.mean(tail))
    return weighted_prev, unweighted_prev, int(len(frame))


def physical_values(quantity: str, log10_threshold: float) -> dict[str, float]:
    value = 10.0 ** float(log10_threshold)
    if quantity == "pga":
        return {
            "threshold_si": value,
            "threshold_cm_s2": value * 100.0,
            "threshold_g": value / G_MPS2,
        }
    return {
        "threshold_si": value,
        "threshold_cm_s": value * 100.0,
        "threshold_g": np.nan,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default=(
            'data/scedc/model_manifests/scenario_t0_5s_k5.csv'
        ),
    )
    p.add_argument("--split-column", default="split_grouped")
    p.add_argument("--train-label", default="train")
    p.add_argument("--validation-label", default="validation")
    p.add_argument("--test-label", default="test")
    p.add_argument("--t0-sec", type=float, default=5.0)
    p.add_argument("--input-stations", type=int, default=5)
    p.add_argument("--target-stations", type=int, default=10)
    p.add_argument(
        "--quantiles",
        default="0.85,0.90,0.95",
    )
    p.add_argument(
        "--current-threshold-json",
        default=(
            'runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json'
        ),
    )
    p.add_argument(
        "--out-dir",
        default='runs/phase1_tail_threshold_audit',
    )
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quantiles = [
        float(x.strip())
        for x in args.quantiles.split(",")
        if x.strip()
    ]
    if not quantiles:
        raise ValueError("No quantiles.")
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"Invalid quantile: {q}")

    manifest = pd.read_csv(
        manifest_path,
        dtype={"event_id": str},
    )
    required = {"event_id", "h5_path", args.split_column}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing: {sorted(missing)}")

    # Explicitly retain the three formal main splits only.
    labels = {
        args.train_label,
        args.validation_label,
        args.test_label,
    }
    manifest = manifest.loc[
        manifest[args.split_column].astype(str).isin(labels)
    ].copy()

    pair_path = out_dir / "unique_event_station_pairs.csv"
    pairs = build_pair_table(
        manifest=manifest,
        split_column=args.split_column,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
    )
    pairs.to_csv(pair_path, index=False)

    # Normalize split name to a single column used below.
    pairs["split"] = pairs["split"].astype(str)

    train = pairs.loc[
        pairs["split"].eq(args.train_label)
    ].copy()
    if train.empty:
        raise RuntimeError("Training split is empty.")

    reference_sources = [
        "all_station_unique",
        "possible_target_unique",
        "target_sampling_weighted",
    ]
    quantities = ["pga", "pgv"]

    threshold_rows = []
    thresholds = {}
    for source in reference_sources:
        thresholds[source] = {}
        for q in quantiles:
            thresholds[source][q] = {}
            for quantity in quantities:
                t = threshold_from_reference(
                    train, source, quantity, q
                )
                thresholds[source][q][quantity] = t
                threshold_rows.append(
                    {
                        "reference_source": source,
                        "quantile": q,
                        "quantity": quantity,
                        "log10_threshold": t,
                        **physical_values(quantity, t),
                    }
                )

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_path = out_dir / "threshold_candidates.csv"
    threshold_df.to_csv(threshold_path, index=False)

    prevalence_rows = []
    evaluation_units = [
        "all_station_unique",
        "possible_target_unique",
        "target_sampling_weighted",
    ]

    split_map = {
        "train": args.train_label,
        "validation": args.validation_label,
        "test": args.test_label,
    }

    for source in reference_sources:
        for q in quantiles:
            for quantity in quantities:
                t = thresholds[source][q][quantity]
                for split_alias, split_label in split_map.items():
                    split_frame = pairs.loc[
                        pairs["split"].eq(split_label)
                    ]
                    for unit in evaluation_units:
                        weighted_prev, raw_prev, n_pairs = prevalence(
                            split_frame,
                            quantity,
                            t,
                            unit,
                        )
                        prevalence_rows.append(
                            {
                                "threshold_reference_source": source,
                                "quantile": q,
                                "quantity": quantity,
                                "split": split_alias,
                                "evaluation_unit": unit,
                                "log10_threshold": t,
                                "prevalence": weighted_prev,
                                "unweighted_prevalence": raw_prev,
                                "n_unique_pairs": n_pairs,
                            }
                        )

    prevalence_df = pd.DataFrame(prevalence_rows)

    # Audit the currently used threshold JSON when available.
    current_path = Path(args.current_threshold_json)
    current_rows = []
    if current_path.exists():
        current = json.loads(
            current_path.read_text(encoding="utf-8")
        )
        current_thresholds = {
            "pga": float(current["log10_pga_threshold"]),
            "pgv": float(current["log10_pgv_threshold"]),
        }
        for quantity, t in current_thresholds.items():
            for split_alias, split_label in split_map.items():
                split_frame = pairs.loc[
                    pairs["split"].eq(split_label)
                ]
                for unit in evaluation_units:
                    wp, up, n_pairs = prevalence(
                        split_frame, quantity, t, unit
                    )
                    current_rows.append(
                        {
                            "threshold_reference_source": "current_json",
                            "quantile": float(current.get("quantile", 0.90)),
                            "quantity": quantity,
                            "split": split_alias,
                            "evaluation_unit": unit,
                            "log10_threshold": t,
                            "prevalence": wp,
                            "unweighted_prevalence": up,
                            "n_unique_pairs": n_pairs,
                        }
                    )

    if current_rows:
        prevalence_df = pd.concat(
            [prevalence_df, pd.DataFrame(current_rows)],
            ignore_index=True,
        )

    prevalence_path = out_dir / "prevalence_audit.csv"
    prevalence_df.to_csv(prevalence_path, index=False)

    # Recommended *candidate* definition for manuscript consistency:
    # Q90 over unique possible held-out event-station pairs in training.
    q90 = min(quantiles, key=lambda x: abs(x - 0.90))
    recommended = {
        "status": "audit_candidate_not_automatically_adopted",
        "definition": (
            "training-only Q0.90 over unique event-station pairs that can "
            "serve as held-out targets under the T0/K sampling rule"
        ),
        "split_column": args.split_column,
        "train_label": args.train_label,
        "t0_sec": args.t0_sec,
        "input_stations": args.input_stations,
        "target_stations": args.target_stations,
        "quantile": q90,
        "log10_pga_threshold": thresholds[
            "possible_target_unique"
        ][q90]["pga"],
        "log10_pgv_threshold": thresholds[
            "possible_target_unique"
        ][q90]["pgv"],
    }
    recommended.update(
        {
            "pga_threshold_mps2": (
                10.0 ** recommended["log10_pga_threshold"]
            ),
            "pga_threshold_g": (
                10.0 ** recommended["log10_pga_threshold"]
            ) / G_MPS2,
            "pgv_threshold_mps": (
                10.0 ** recommended["log10_pgv_threshold"]
            ),
            "pgv_threshold_cm_s": (
                100.0
                * 10.0 ** recommended["log10_pgv_threshold"]
            ),
        }
    )

    recommended_path = (
        out_dir / "candidate_thresholds_unique_target_q0.90.json"
    )
    recommended_path.write_text(
        json.dumps(recommended, indent=2),
        encoding="utf-8",
    )

    # Compact table for immediate inspection.
    compact = prevalence_df.loc[
        prevalence_df["quantile"].sub(q90).abs() < 1e-9
    ].copy()
    compact = compact.loc[
        compact["evaluation_unit"].eq("possible_target_unique")
    ]
    compact_path = out_dir / "q90_unique_target_prevalence_compact.csv"
    compact.to_csv(compact_path, index=False)

    print("=== Phase 1 / Step 2: tail-threshold audit ===")
    print(f"Manifest        : {manifest_path.resolve()}")
    print(f"Events          : {manifest['event_id'].nunique():,}")
    print(f"Unique pairs    : {len(pairs):,}")
    print(
        "Possible targets: "
        f"{int(pairs['possible_heldout_target'].sum()):,}"
    )
    print("\nThreshold candidates:")
    print(
        threshold_df[
            [
                "reference_source", "quantile", "quantity",
                "log10_threshold", "threshold_si",
                "threshold_cm_s2", "threshold_cm_s", "threshold_g"
            ]
        ].to_string(index=False)
    )
    print("\nQ90 prevalence on unique possible held-out targets:")
    print(
        compact[
            [
                "threshold_reference_source", "quantity",
                "split", "log10_threshold", "prevalence",
                "n_unique_pairs"
            ]
        ].to_string(index=False)
    )
    print(f"\nPairs out       : {pair_path.resolve()}")
    print(f"Thresholds out  : {threshold_path.resolve()}")
    print(f"Prevalence out  : {prevalence_path.resolve()}")
    print(f"Candidate JSON  : {recommended_path.resolve()}")


if __name__ == "__main__":
    main()
