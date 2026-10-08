#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validate the full HDF5 dataset produced by full physical preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


REQUIRED = [
    "velocity",
    "acceleration",
    "station_coords",
    "p_offset_sec",
    "pga_h_mps2",
    "pgv_h_mps",
    "fas_h_acc",
    "raw_clip_fraction",
    "station_id",
    "channel_family",
]


def horizontal_peak(x: np.ndarray) -> np.ndarray:
    return np.max(
        np.sqrt(x[:, 0, :] ** 2 + x[:, 1, :] ** 2),
        axis=1,
    )


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10:
        return np.nan
    a = a[good]
    b = b[good]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def write_partial(
    out_dir: Path,
    event_rows: list[dict],
    station_rows: list[dict],
) -> None:
    event_tmp = out_dir / "full_h5_event_audit_partial.csv.tmp"
    station_tmp = out_dir / "full_h5_station_audit_partial.csv.tmp"
    pd.DataFrame(event_rows).to_csv(event_tmp, index=False)
    pd.DataFrame(station_rows).to_csv(station_tmp, index=False)
    event_tmp.replace(out_dir / "full_h5_event_audit_partial.csv")
    station_tmp.replace(out_dir / "full_h5_station_audit_partial.csv")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate all full-preprocessing event HDF5 files."
    )
    p.add_argument(
        "--processed-root",
        default="data/scedc/processed_full_v4",
    )
    p.add_argument(
        "--out-dir",
        default="data/scedc/processed_full_v4/validation",
    )
    p.add_argument("--derivative-stations-per-event", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--min-expected-events", type=int, default=1000)
    p.add_argument(
        "--allow-small-run",
        action="store_true",
        help="Allow validation of fewer than --min-expected-events HDF5 files.",
    )
    p.add_argument("--seed", type=int, default=20260713)
    args = p.parse_args()

    processed_root = Path(args.processed_root)
    event_dir = processed_root / "events"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(event_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found under {event_dir.resolve()}"
        )

    if len(files) < args.min_expected_events and not args.allow_small_run:
        raise RuntimeError(
            f"Only {len(files)} HDF5 files were found under "
            f"{event_dir.resolve()}. A full validation expects at least "
            f"{args.min_expected_events}. This usually means the pilot "
            "directory was supplied by mistake. Use --allow-small-run only "
            "for an intentional pilot validation."
        )

    print("=== Full HDF5 validation configuration ===")
    print(f"Processed root : {processed_root.resolve()}")
    print(f"HDF5 files     : {len(files)}")
    print(f"Output dir     : {out_dir.resolve()}")

    rng = np.random.default_rng(args.seed)
    event_rows: list[dict] = []
    station_rows: list[dict] = []

    for file_number, path in enumerate(
        tqdm(files, desc="validate full HDF5"),
        start=1,
    ):
        event_row = {
            "event_id": path.stem,
            "h5_path": str(path.resolve()),
            "status": "ok",
            "preprocessing_version": "",
            "error": "",
        }

        try:
            with h5py.File(path, "r") as h5:
                missing = [name for name in REQUIRED if name not in h5]
                if missing:
                    raise ValueError(
                        f"missing datasets: {missing}"
                    )

                velocity = np.asarray(
                    h5["velocity"][:],
                    dtype=np.float64,
                )
                acceleration = np.asarray(
                    h5["acceleration"][:],
                    dtype=np.float64,
                )
                pga = np.asarray(
                    h5["pga_h_mps2"][:],
                    dtype=np.float64,
                )
                pgv = np.asarray(
                    h5["pgv_h_mps"][:],
                    dtype=np.float64,
                )
                coordinates = np.asarray(
                    h5["station_coords"][:],
                    dtype=np.float64,
                )
                p_offset = np.asarray(
                    h5["p_offset_sec"][:],
                    dtype=np.float64,
                )
                clipping = np.asarray(
                    h5["raw_clip_fraction"][:],
                    dtype=np.float64,
                )
                fas_h = np.asarray(
                    h5["fas_h_acc"][:],
                    dtype=np.float64,
                )
                station_ids = h5["station_id"].asstr()[:]
                channel_families = h5["channel_family"].asstr()[:]

                sampling_rate = float(
                    h5.attrs["sampling_rate_hz"]
                )
                n_stations = int(h5.attrs["n_stations"])
                version = str(
                    h5.attrs.get(
                        "preprocessing_version",
                        "legacy_or_unknown",
                    )
                )

                if (
                    velocity.shape != acceleration.shape
                    or velocity.ndim != 3
                    or velocity.shape[0] != n_stations
                    or velocity.shape[1] != 3
                ):
                    raise ValueError(
                        "invalid waveform shapes: "
                        f"velocity={velocity.shape}, "
                        f"acceleration={acceleration.shape}, "
                        f"n_stations={n_stations}"
                    )

                for name, array in [
                    ("velocity", velocity),
                    ("acceleration", acceleration),
                    ("PGA", pga),
                    ("PGV", pgv),
                    ("coordinates", coordinates),
                    ("FAS", fas_h),
                ]:
                    if not np.isfinite(array).all():
                        raise ValueError(
                            f"{name} contains NaN/Inf"
                        )

                if (
                    (pga <= 0).any()
                    or (pgv <= 0).any()
                    or (fas_h < 0).any()
                ):
                    raise ValueError(
                        "non-positive PGA/PGV or negative FAS"
                    )

                if (clipping >= 0.20).any():
                    raise ValueError(
                        "raw clipping/dead-channel fraction >= 0.20"
                    )

                pga_calculated = horizontal_peak(acceleration)
                pgv_calculated = horizontal_peak(velocity)

                pga_relative_error = (
                    np.abs(pga_calculated - pga)
                    / np.maximum(np.abs(pga), 1e-12)
                )
                pgv_relative_error = (
                    np.abs(pgv_calculated - pgv)
                    / np.maximum(np.abs(pgv), 1e-12)
                )

                sample_count = min(
                    args.derivative_stations_per_event,
                    n_stations,
                )
                sample_indices = rng.choice(
                    n_stations,
                    size=sample_count,
                    replace=False,
                )

                correlations = []
                amplitude_ratios = []

                for index in sample_indices:
                    derivative = np.gradient(
                        velocity[index],
                        1.0 / sampling_rate,
                        axis=-1,
                    )
                    current_acceleration = acceleration[index]

                    if current_acceleration.shape[-1] > 100:
                        derivative = derivative[:, 50:-50]
                        current_acceleration = (
                            current_acceleration[:, 50:-50]
                        )

                    correlations.append(
                        safe_corr(
                            derivative,
                            current_acceleration,
                        )
                    )
                    amplitude_ratios.append(
                        np.sqrt(np.mean(derivative ** 2))
                        / max(
                            np.sqrt(
                                np.mean(
                                    current_acceleration ** 2
                                )
                            ),
                            1e-20,
                        )
                    )

                event_row.update({
                    "magnitude": float(
                        h5.attrs.get("magnitude", np.nan)
                    ),
                    "n_stations": n_stations,
                    "n_samples": int(velocity.shape[-1]),
                    "sampling_rate_hz": sampling_rate,
                    "preprocessing_version": version,
                    "max_pga_peak_relative_error": float(
                        np.max(pga_relative_error)
                    ),
                    "max_pgv_peak_relative_error": float(
                        np.max(pgv_relative_error)
                    ),
                    "median_acc_dvel_corr": float(
                        np.nanmedian(correlations)
                    ),
                    "median_dvel_acc_rms_ratio": float(
                        np.nanmedian(amplitude_ratios)
                    ),
                    "max_clip_fraction": float(
                        np.nanmax(clipping)
                    ),
                })

                for index in range(n_stations):
                    station_rows.append({
                        "event_id": path.stem,
                        "station_id": station_ids[index],
                        "channel_family": channel_families[index],
                        "pga_h_mps2": pga[index],
                        "pga_h_g": pga[index] / 9.80665,
                        "pgv_h_mps": pgv[index],
                        "pgv_h_cms": pgv[index] * 100.0,
                        "raw_clip_fraction": clipping[index],
                        "p_offset_sec": p_offset[index],
                        "latitude": coordinates[index, 0],
                        "longitude": coordinates[index, 1],
                        "elevation_m": coordinates[index, 2],
                    })

        except Exception as exc:
            event_row["status"] = "failed"
            event_row["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        event_rows.append(event_row)

        if (
            file_number % max(1, args.checkpoint_every) == 0
            or file_number == len(files)
        ):
            write_partial(
                out_dir,
                event_rows,
                station_rows,
            )

    events = pd.DataFrame(event_rows)
    stations = pd.DataFrame(station_rows)

    event_output = out_dir / "full_h5_event_audit.csv"
    station_output = out_dir / "full_h5_station_audit.csv"

    events.to_csv(event_output, index=False)
    stations.to_csv(station_output, index=False)

    valid_events = events.loc[events["status"] == "ok"]
    failed_events = events.loc[events["status"] == "failed"]

    print("\n=== Processed full validation ===")
    print(f"HDF5 files checked      : {len(events)}")
    print(f"Valid event files       : {len(valid_events)}")
    print(f"Failed event files      : {len(failed_events)}")
    print(f"Station records audited : {len(stations)}")

    if len(stations):
        print("\nPGA/PGV summary:")
        print(
            stations[
                [
                    "pga_h_g",
                    "pgv_h_cms",
                    "raw_clip_fraction",
                ]
            ]
            .describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.95,
                    0.99,
                    0.999,
                ]
            )
            .to_string()
        )

        print("\nChannel-family counts:")
        print(
            stations["channel_family"]
            .value_counts()
            .to_string()
        )

    if len(valid_events):
        print("\nWaveform consistency:")
        print(
            valid_events[
                [
                    "max_pga_peak_relative_error",
                    "max_pgv_peak_relative_error",
                    "median_acc_dvel_corr",
                    "median_dvel_acc_rms_ratio",
                ]
            ]
            .describe(
                percentiles=[0.5, 0.9, 0.95, 0.99]
            )
            .to_string()
        )

        print("\nPreprocessing-version counts:")
        print(
            valid_events["preprocessing_version"]
            .value_counts(dropna=False)
            .to_string()
        )

    if len(failed_events):
        print("\nTop validation failures:")
        print(
            failed_events["error"]
            .value_counts()
            .head(20)
            .to_string()
        )

    print(f"\nEvent audit   : {event_output.resolve()}")
    print(f"Station audit : {station_output.resolve()}")


if __name__ == "__main__":
    main()
