#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physical preprocessing pilot for Causal-SeisField.

For each station:
  miniSEED + StationXML -> velocity/acceleration in SI units
  -> ZNE rotation -> 0.1-20 Hz -> 50 Hz -> PGA/PGV/FAS.

One HDF5 file is written per event. All stations use the common event time axis
[first_P - 5 s, first_P + 60 s).
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime, read, read_inventory
from tqdm import tqdm

COMPONENT_ORDER = ("E", "N", "Z")



def parse_utc_timestamp(value, field_name="time"):
    """
    Robustly convert pandas/CSV time values to ObsPy UTCDateTime.

    Handles strings such as:
      2010-01-01 00:00:08.470000+00:00
      2010-01-01T00:00:08.470000Z
    and avoids ObsPy version-dependent string parsing.
    """
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid or missing {field_name}: {value!r}")
    return UTCDateTime(float(ts.timestamp()))


def normalize_location(value: Any) -> str:
    if pd.isna(value):
        return "--"
    text = str(value).strip()
    return "--" if text in {"", "__", "--", "nan", "None"} else text


def resolve_file(path_text: Any, event_id: str, raw_root: Path, subdir: str) -> Path:
    text = "" if pd.isna(path_text) else str(path_text).strip()
    candidates = []
    if text:
        candidates.extend([
            Path(text),
            Path.cwd() / text,
            raw_root / event_id / subdir / Path(text).name,
        ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return Path(text) if text else Path()



def validate_raw_stream(st: Stream) -> float:
    """
    Reject clearly unusable raw components and return the maximum
    extreme-value occupancy used as the clipping diagnostic.

    This is deliberately conservative. It does not reject records because
    their physical PGA/PGV is large.
    """
    fractions = []

    for tr in st:
        x = np.asarray(tr.data)

        if x.size == 0:
            raise ValueError(f"Empty raw trace: {tr.id}")
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/Inf in raw trace: {tr.id}")

        peak_to_peak = float(np.ptp(x))
        std = float(np.std(x))
        unique_count = int(np.unique(x).size)

        xmin = np.min(x)
        xmax = np.max(x)
        extreme_fraction = float(
            max(np.mean(x == xmin), np.mean(x == xmax))
        )

        if peak_to_peak <= 0.0 or std <= 0.0:
            raise ValueError(f"Constant/dead raw trace: {tr.id}")

        if unique_count < 8:
            raise ValueError(
                f"Too few unique raw values: {tr.id}, "
                f"unique_count={unique_count}"
            )

        if extreme_fraction >= 0.20:
            raise ValueError(
                f"Excessive raw extreme-value occupancy: {tr.id}, "
                f"fraction={extreme_fraction:.6f}"
            )

        fractions.append(extreme_fraction)

    if not fractions:
        raise ValueError("No usable raw traces")

    return float(max(fractions))


def raw_clip_fraction(st: Stream) -> float:
    vals = []
    for tr in st:
        x = np.asarray(tr.data)
        if x.size:
            vals.append(max(np.mean(x == x.min()), np.mean(x == x.max())))
    return float(max(vals)) if vals else np.nan


def prepare_raw(st: Stream) -> Stream:
    st = st.copy()
    st.sort()
    st.merge(method=1, fill_value="interpolate")
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05, max_length=2.0, type="cosine")
    return st


def remove_response_stream(
    st: Stream,
    inventory,
    output: str,
    max_frequency_hz: float,
) -> Stream:
    out = Stream()
    for source in st:
        tr = source.copy()
        fs = float(tr.stats.sampling_rate)
        nyq = 0.5 * fs
        f4 = min(0.90 * nyq, 1.25 * max_frequency_hz)
        f3 = min(max_frequency_hz, 0.80 * f4)
        f2 = min(0.10, f3 / 8.0)
        f1 = min(0.05, f2 / 2.0)
        if not (0 < f1 < f2 < f3 < f4 < nyq):
            raise ValueError(f"Invalid pre-filter for {tr.id}, fs={fs}")
        tr.remove_response(
            inventory=inventory,
            output=output,
            pre_filt=(f1, f2, f3, f4),
            water_level=60,
            zero_mean=False,
            taper=False,
        )
        out += tr
    return out


def rotate_to_zne(st: Stream, inventory) -> Stream:
    comps = {tr.stats.channel[-1].upper() for tr in st if tr.stats.channel}
    if {"E", "N", "Z"}.issubset(comps):
        return st
    rotated = st.copy()
    rotated.rotate(method="->ZNE", inventory=inventory)
    comps = {tr.stats.channel[-1].upper() for tr in rotated if tr.stats.channel}
    if not {"E", "N", "Z"}.issubset(comps):
        raise ValueError(f"Cannot obtain E/N/Z components: {sorted(comps)}")
    return rotated


def choose_components(st: Stream) -> dict[str, Trace]:
    out = {}
    for comp in COMPONENT_ORDER:
        candidates = [
            tr for tr in st
            if tr.stats.channel and tr.stats.channel[-1].upper() == comp
        ]
        if not candidates:
            raise ValueError(f"Missing component {comp}")
        candidates.sort(
            key=lambda tr: (float(tr.stats.endtime - tr.stats.starttime), tr.stats.npts),
            reverse=True,
        )
        out[comp] = candidates[0].copy()
    return out


def exact_array(
    traces: dict[str, Trace],
    start: UTCDateTime,
    end: UTCDateTime,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    npts = int(round(float(end - start) * sampling_rate))
    arrays = []
    for comp in COMPONENT_ORDER:
        tr = traces[comp].copy()
        fs = float(tr.stats.sampling_rate)
        if fs < 2.5 * high_hz:
            raise ValueError(f"Sampling rate too low: {tr.id}, fs={fs}")
        tolerance = 1.5 / fs
        if tr.stats.starttime > start + tolerance:
            raise ValueError(f"Trace starts too late: {tr.id}")
        if tr.stats.endtime < end - tolerance:
            raise ValueError(f"Trace ends too early: {tr.id}")
        tr.filter(
            "bandpass", freqmin=low_hz, freqmax=high_hz,
            corners=4, zerophase=True,
        )
        tr.interpolate(
            sampling_rate=sampling_rate,
            method="lanczos", a=12,
            starttime=start, npts=npts,
        )
        x = np.asarray(tr.data, dtype=np.float32)
        if x.shape != (npts,):
            raise ValueError(f"Unexpected npts: {tr.id}, {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/Inf after preprocessing: {tr.id}")
        arrays.append(x)
    return np.stack(arrays, axis=0)


def station_coordinates(inventory, network: str, station: str, location: str, time):
    loc = "" if location == "--" else location
    inv = inventory.select(
        network=network or "*", station=station or "*",
        location=loc if loc else "*", time=time,
    )
    for net in inv:
        for sta in net:
            return float(sta.latitude), float(sta.longitude), float(sta.elevation)
    inv = inventory.select(network=network or "*", station=station or "*")
    for net in inv:
        for sta in net:
            return float(sta.latitude), float(sta.longitude), float(sta.elevation)
    raise ValueError(f"No coordinates for {network}.{station}")


def fas(data: np.ndarray, fs: float, target_freqs: np.ndarray) -> np.ndarray:
    n = data.shape[-1]
    window = np.hanning(n)
    scale = 2.0 / max(window.sum(), 1.0)
    source_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(data * window[None, :], axis=-1)) * scale
    keep = source_freqs > 0
    lf = np.log(source_freqs[keep])
    rows = []
    for i in range(data.shape[0]):
        la = np.log(np.maximum(spec[i, keep], 1e-20))
        rows.append(np.exp(np.interp(np.log(target_freqs), lf, la)).astype(np.float32))
    return np.stack(rows)


def process_station(
    row: pd.Series,
    event_id: str,
    first_p: UTCDateTime,
    start: UTCDateTime,
    end: UTCDateTime,
    raw_root: Path,
    fs: float,
    low_hz: float,
    high_hz: float,
    target_freqs: np.ndarray,
):
    mseed = resolve_file(row.get("mseed_path", ""), event_id, raw_root, "waveforms")
    xml = resolve_file(row.get("stationxml_path", ""), event_id, raw_root, "stationxml")
    if not mseed.exists():
        raise FileNotFoundError(f"miniSEED not found: {mseed}")
    if not xml.exists():
        raise FileNotFoundError(f"StationXML not found: {xml}")

    raw = read(str(mseed))
    inventory = read_inventory(str(xml))
    clip = validate_raw_stream(raw)
    prepared = prepare_raw(raw)
    family = next((tr.stats.channel[:2] for tr in prepared if tr.stats.channel), "")

    vel = rotate_to_zne(
        remove_response_stream(prepared, inventory, "VEL", high_hz), inventory
    )
    acc = rotate_to_zne(
        remove_response_stream(prepared, inventory, "ACC", high_hz), inventory
    )
    vel_array = exact_array(choose_components(vel), start, end, fs, low_hz, high_hz)
    acc_array = exact_array(choose_components(acc), start, end, fs, low_hz, high_hz)

    pga_comp = np.max(np.abs(acc_array), axis=-1).astype(np.float32)
    pgv_comp = np.max(np.abs(vel_array), axis=-1).astype(np.float32)
    pga_h = float(np.max(np.sqrt(acc_array[0] ** 2 + acc_array[1] ** 2)))
    pgv_h = float(np.max(np.sqrt(vel_array[0] ** 2 + vel_array[1] ** 2)))
    if pga_h > 150.0:
        raise ValueError(f"Implausible PGA: {pga_h} m/s^2")
    if pgv_h > 25.0:
        raise ValueError(f"Implausible PGV: {pgv_h} m/s")

    fas_acc = fas(acc_array, fs, target_freqs)
    fas_h = np.sqrt(
        np.maximum(fas_acc[0], 1e-20) * np.maximum(fas_acc[1], 1e-20)
    ).astype(np.float32)

    network = str(row.get("network", "")).strip()
    station = str(row.get("station", "")).strip()
    location = normalize_location(row.get("location", "--"))
    coords = np.asarray(
        station_coordinates(inventory, network, station, location, first_p),
        dtype=np.float32,
    )

    def numeric(name):
        value = pd.to_numeric(row.get(name, np.nan), errors="coerce")
        return float(value) if np.isfinite(value) else np.nan

    station_id = str(row.get("station_id", "")).strip() or f"{network}.{station}.{location}"
    return {
        "station_id": station_id,
        "network": network,
        "station": station,
        "location": location,
        "channel_family": family,
        "mseed_path": str(mseed),
        "stationxml_path": str(xml),
        "station_coords": coords,
        "velocity": vel_array,
        "acceleration": acc_array,
        "p_offset_sec": numeric("p_offset_from_first_sec"),
        "s_offset_sec": numeric("s_offset_from_first_sec"),
        "pga_comp_mps2": pga_comp,
        "pgv_comp_mps": pgv_comp,
        "pga_h_mps2": np.float32(pga_h),
        "pgv_h_mps": np.float32(pgv_h),
        "fas_acc": fas_acc,
        "fas_h_acc": fas_h,
        "raw_clip_fraction": np.float32(clip),
    }


def write_h5(path: Path, event: pd.Series, rows: list[dict], args, freqs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.part")
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(tmp, "w") as h5:
        for key in ["event_id", "origin_time", "first_p_time"]:
            h5.attrs[key] = str(event.get(key, ""))
        for key in ["magnitude", "latitude", "longitude", "depth_km"]:
            h5.attrs[key] = float(event.get(key, np.nan))
        h5.attrs["sampling_rate_hz"] = float(args.sampling_rate)
        h5.attrs["component_order"] = "E,N,Z"
        h5.attrs["velocity_unit"] = "m/s"
        h5.attrs["acceleration_unit"] = "m/s^2"
        h5.attrs["pre_first_p_sec"] = float(args.pre_first_p_sec)
        h5.attrs["post_first_p_sec"] = float(args.post_first_p_sec)
        h5.attrs["time_zero_index"] = int(round(args.pre_first_p_sec * args.sampling_rate))
        h5.attrs["n_stations"] = len(rows)
        h5.attrs["preprocessing_version"] = "v4_dead_channel_qc"

        for name in ["velocity", "acceleration", "fas_acc", "fas_h_acc"]:
            h5.create_dataset(
                name, data=np.stack([r[name] for r in rows]),
                dtype="float32", compression="gzip", compression_opts=4, shuffle=True,
            )
        for name in ["station_coords", "pga_comp_mps2", "pgv_comp_mps"]:
            h5.create_dataset(name, data=np.stack([r[name] for r in rows]), dtype="float32")
        for name in ["p_offset_sec", "s_offset_sec", "pga_h_mps2", "pgv_h_mps", "raw_clip_fraction"]:
            h5.create_dataset(name, data=np.asarray([r[name] for r in rows], dtype=np.float32))
        h5.create_dataset("fas_frequencies_hz", data=freqs.astype(np.float32))

        for name in ["station_id", "network", "station", "location", "channel_family", "mseed_path", "stationxml_path"]:
            h5.create_dataset(
                name,
                data=np.asarray([str(r[name]) for r in rows], dtype=object),
                dtype=string_dtype,
            )
    tmp.replace(path)



def write_qc_checkpoints(
    out_root: Path,
    station_qc: list[dict],
    event_qc: list[dict],
) -> None:
    """Atomically write progress tables so an interrupted full run can be inspected."""
    station_tmp = out_root / "station_preprocess_qc_partial.csv.tmp"
    event_tmp = out_root / "event_preprocess_qc_partial.csv.tmp"

    pd.DataFrame(station_qc).to_csv(station_tmp, index=False)
    pd.DataFrame(event_qc).to_csv(event_tmp, index=False)

    station_tmp.replace(out_root / "station_preprocess_qc_partial.csv")
    event_tmp.replace(out_root / "event_preprocess_qc_partial.csv")


def main():
    p = argparse.ArgumentParser(
        description="Full SCEDC physical preprocessing for Causal-SeisField."
    )
    p.add_argument(
        "--events",
        default="data/scedc/full_preprocessing_events.csv",
        help="Full event list produced by 09_select_full_preprocessing_events_full.py.",
    )
    p.add_argument(
        "--stations",
        default="data/scedc/phase_ready/phase_ready_stations.csv",
    )
    p.add_argument(
        "--raw-root",
        default="data/scedc/raw_m3_2010_2025",
    )
    p.add_argument(
        "--out-root",
        default="data/scedc/processed_full_v4",
        help="Full-output directory. It is intentionally different from the pilot directory.",
    )
    p.add_argument("--limit-events", type=int, default=None)
    p.add_argument("--sampling-rate", type=float, default=50.0)
    p.add_argument("--low-hz", type=float, default=0.1)
    p.add_argument("--high-hz", type=float, default=20.0)
    p.add_argument("--pre-first-p-sec", type=float, default=5.0)
    p.add_argument("--post-first-p-sec", type=float, default=60.0)
    p.add_argument("--fas-bins", type=int, default=64)
    p.add_argument("--min-processed-stations", type=int, default=20)
    p.add_argument("--max-stations-per-event", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--min-expected-events", type=int, default=1000)
    p.add_argument(
        "--allow-small-run",
        action="store_true",
        help="Allow fewer than --min-expected-events events, for an intentional test only.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    events_path = Path(args.events)
    stations_path = Path(args.stations)
    if not events_path.exists():
        raise FileNotFoundError(
            f"Full event list not found: {events_path.resolve()}\n"
            "Run 09_select_full_preprocessing_events_full.py first."
        )
    if not stations_path.exists():
        raise FileNotFoundError(
            f"Station-phase table not found: {stations_path.resolve()}"
        )

    events = pd.read_csv(events_path, dtype={"event_id": str})
    stations = pd.read_csv(
        stations_path,
        dtype={
            "event_id": str,
            "network": str,
            "station": str,
            "location": str,
            "station_id": str,
        },
    )

    if args.limit_events is not None:
        if not args.allow_small_run:
            raise ValueError(
                "--limit-events is a test option. Add --allow-small-run "
                "when a small run is intentional."
            )
        events = events.iloc[:args.limit_events].copy()

    if len(events) < args.min_expected_events and not args.allow_small_run:
        raise RuntimeError(
            f"Only {len(events)} events were loaded from {events_path.resolve()}. "
            f"A full run expects at least {args.min_expected_events}. "
            "This usually means a pilot CSV was supplied by mistake. "
            "Use --allow-small-run only for an intentional test."
        )

    required_event_cols = {"event_id", "first_p_time"}
    missing_event_cols = required_event_cols.difference(events.columns)
    if missing_event_cols:
        raise ValueError(
            f"Event CSV is missing columns: {sorted(missing_event_cols)}"
        )

    required_station_cols = {"event_id", "mseed_path", "stationxml_path"}
    missing_station_cols = required_station_cols.difference(stations.columns)
    if missing_station_cols:
        raise ValueError(
            f"Station CSV is missing columns: {sorted(missing_station_cols)}"
        )

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    event_root = out_root / "events"
    event_root.mkdir(parents=True, exist_ok=True)

    # Avoid scanning the full station table once for every event.
    station_groups = {
        str(event_id): group.copy()
        for event_id, group in stations.groupby("event_id", sort=False)
    }

    freqs = np.geomspace(
        args.low_hz,
        args.high_hz,
        args.fas_bins,
    ).astype(np.float32)

    print("=== Full preprocessing configuration ===")
    print(f"Events file         : {events_path.resolve()}")
    print(f"Events loaded       : {len(events)}")
    print(f"Station rows loaded : {len(stations)}")
    print(f"Raw root            : {raw_root.resolve()}")
    print(f"Output root         : {out_root.resolve()}")
    print(f"Overwrite           : {args.overwrite}")
    print(f"Checkpoint interval : {args.checkpoint_every} events")

    station_qc: list[dict] = []
    event_qc: list[dict] = []

    iterator = tqdm(
        events.iterrows(),
        total=len(events),
        desc="preprocess full events",
    )

    for event_number, (_, event) in enumerate(iterator, start=1):
        event_id = str(event["event_id"])
        output = event_root / f"{event_id}.h5"

        if output.exists() and not args.overwrite:
            try:
                with h5py.File(output, "r") as h5:
                    n = int(h5.attrs["n_stations"])
                    version = str(
                        h5.attrs.get(
                            "preprocessing_version",
                            "legacy_or_unknown",
                        )
                    )
                    station_ids = (
                        h5["station_id"].asstr()[:]
                        if "station_id" in h5
                        else np.asarray(
                            [f"{event_id}.{i}" for i in range(n)]
                        )
                    )
                    families = (
                        h5["channel_family"].asstr()[:]
                        if "channel_family" in h5
                        else np.asarray([""] * n)
                    )
                    pga = (
                        np.asarray(h5["pga_h_mps2"][:], dtype=float)
                        if "pga_h_mps2" in h5
                        else np.full(n, np.nan)
                    )
                    pgv = (
                        np.asarray(h5["pgv_h_mps"][:], dtype=float)
                        if "pgv_h_mps" in h5
                        else np.full(n, np.nan)
                    )
                    clip = (
                        np.asarray(
                            h5["raw_clip_fraction"][:],
                            dtype=float,
                        )
                        if "raw_clip_fraction" in h5
                        else np.full(n, np.nan)
                    )

                for i in range(n):
                    station_qc.append({
                        "event_id": event_id,
                        "station_id": str(station_ids[i]),
                        "status": "reused",
                        "channel_family": str(families[i]),
                        "pga_h_mps2": float(pga[i]),
                        "pgv_h_mps": float(pgv[i]),
                        "raw_clip_fraction": float(clip[i]),
                        "mseed_path": "",
                        "stationxml_path": "",
                        "error": "",
                    })

                event_qc.append({
                    "event_id": event_id,
                    "status": "reused",
                    "n_candidates": np.nan,
                    "n_processed": n,
                    "h5_path": str(output.resolve()),
                    "preprocessing_version": version,
                    "error": "",
                })

            except Exception as exc:
                event_qc.append({
                    "event_id": event_id,
                    "status": "reuse_failed",
                    "n_candidates": np.nan,
                    "n_processed": 0,
                    "h5_path": str(output.resolve()),
                    "preprocessing_version": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        else:
            try:
                first_p = parse_utc_timestamp(
                    event.get("first_p_time", None),
                    field_name="first_p_time",
                )
            except Exception as exc:
                event_qc.append({
                    "event_id": event_id,
                    "status": "invalid_first_p_time",
                    "n_candidates": 0,
                    "n_processed": 0,
                    "h5_path": "",
                    "preprocessing_version": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if args.debug:
                    print(
                        f"\n[Failed event time] event={event_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                start = first_p - args.pre_first_p_sec
                end = first_p + args.post_first_p_sec
                candidates = station_groups.get(
                    event_id,
                    stations.iloc[0:0].copy(),
                )

                if args.max_stations_per_event > 0:
                    candidates = candidates.iloc[
                        :args.max_stations_per_event
                    ].copy()

                results = []

                for _, row in candidates.iterrows():
                    station_id = str(row.get("station_id", ""))
                    try:
                        result = process_station(
                            row,
                            event_id,
                            first_p,
                            start,
                            end,
                            raw_root,
                            args.sampling_rate,
                            args.low_hz,
                            args.high_hz,
                            freqs,
                        )
                        results.append(result)
                        station_qc.append({
                            "event_id": event_id,
                            "station_id": result["station_id"],
                            "status": "ok",
                            "channel_family": result["channel_family"],
                            "pga_h_mps2": float(result["pga_h_mps2"]),
                            "pgv_h_mps": float(result["pgv_h_mps"]),
                            "raw_clip_fraction": float(
                                result["raw_clip_fraction"]
                            ),
                            "mseed_path": result["mseed_path"],
                            "stationxml_path": result["stationxml_path"],
                            "error": "",
                        })
                    except Exception as exc:
                        station_qc.append({
                            "event_id": event_id,
                            "station_id": station_id,
                            "status": "failed",
                            "channel_family": "",
                            "pga_h_mps2": np.nan,
                            "pgv_h_mps": np.nan,
                            "raw_clip_fraction": np.nan,
                            "mseed_path": str(
                                row.get("mseed_path", "")
                            ),
                            "stationxml_path": str(
                                row.get("stationxml_path", "")
                            ),
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        if args.debug:
                            print(
                                f"\n[Failed station] event={event_id}, "
                                f"station={station_id}: "
                                f"{type(exc).__name__}: {exc}"
                            )

                if len(results) < args.min_processed_stations:
                    event_qc.append({
                        "event_id": event_id,
                        "status": "too_few_processed_stations",
                        "n_candidates": len(candidates),
                        "n_processed": len(results),
                        "h5_path": "",
                        "preprocessing_version": "",
                        "error": "",
                    })
                else:
                    try:
                        write_h5(output, event, results, args, freqs)
                        event_qc.append({
                            "event_id": event_id,
                            "status": "ok",
                            "n_candidates": len(candidates),
                            "n_processed": len(results),
                            "h5_path": str(output.resolve()),
                            "preprocessing_version": (
                                "v4_dead_channel_qc"
                            ),
                            "error": "",
                        })
                    except Exception as exc:
                        event_qc.append({
                            "event_id": event_id,
                            "status": "write_failed",
                            "n_candidates": len(candidates),
                            "n_processed": len(results),
                            "h5_path": "",
                            "preprocessing_version": "",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        if args.debug:
                            traceback.print_exc()

        if (
            event_number % max(1, args.checkpoint_every) == 0
            or event_number == len(events)
        ):
            write_qc_checkpoints(
                out_root,
                station_qc,
                event_qc,
            )

    station_df = pd.DataFrame(station_qc)
    event_df = pd.DataFrame(event_qc)

    station_df.to_csv(
        out_root / "station_preprocess_qc.csv",
        index=False,
    )
    event_df.to_csv(
        out_root / "event_preprocess_qc.csv",
        index=False,
    )

    ok = event_df.loc[
        event_df["status"].isin(["ok", "reused"])
    ].copy()
    ok.to_csv(
        out_root / "processed_event_index.csv",
        index=False,
    )

    print("\n=== Full physical preprocessing summary ===")
    print(f"Events requested      : {len(events)}")
    print(f"Events written/reused : {len(ok)}")

    print("\nEvent status counts:")
    print(
        event_df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nStation status counts:")
    if len(station_df) and "status" in station_df.columns:
        print(
            station_df["status"]
            .value_counts(dropna=False)
            .to_string()
        )
        failed = station_df.loc[
            station_df["status"] == "failed"
        ]
        if len(failed):
            print("\nTop station failure reasons:")
            print(
                failed["error"]
                .value_counts()
                .head(20)
                .to_string()
            )
    else:
        print("(no station rows were processed or loaded)")

    if "preprocessing_version" in event_df.columns:
        print("\nPreprocessing-version counts:")
        print(
            event_df["preprocessing_version"]
            .replace("", np.nan)
            .fillna("unknown")
            .value_counts(dropna=False)
            .to_string()
        )

    print(f"\nOutputs: {out_root.resolve()}")


if __name__ == "__main__":
    main()
