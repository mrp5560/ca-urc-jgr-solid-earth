#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build snapshot-available acceleration inputs without altering the label archive.

Based on 07_preprocess_ground_motion_full_safe(2).py supplied by the user.
The seven original processing/helper functions below are copied verbatim from that
script; cropping RAW traces before these functions is the experimental change.

This is a retrospective, fixed-snapshot, prefix-only preprocessing experiment.
It is NOT a sample-by-sample causal filter or a validated real-time EEW system.
Archived P picks, reference times, station cohorts, and offline labels remain.

Outputs: one sidecar HDF5 per event, aligned to the ORIGINAL station indices.
No original HDF5, label, station pool, split, or locked target draw is changed.
A separate reader must be used by the training and inference data loaders.
Failure is NEVER replaced by old full-record input or by zeros.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import traceback
from typing import Any

import h5py
import numpy as np
import pandas as pd

try:
    import obspy
    from obspy import Stream, Trace, UTCDateTime, read, read_inventory
except ImportError:
    obspy = None
    Stream = Trace = UTCDateTime = read = read_inventory = None

COMPONENT_ORDER = ("E", "N", "Z")
SCHEMA_VERSION = "snapshot_available_acceleration_v1"

# Verbatim scientific-processing functions from the supplied original script.
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

# -----------------------------------------------------------------------------
# Prefix boundary and reproducibility utilities (new experiment only)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    t0_sec: float = 5.0
    input_pre_sec: float = 2.0
    sampling_rate: float = 50.0
    low_hz: float = 0.1
    high_hz: float = 20.0
    audit_stations_per_event: int = 2


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def json_text(value: Any) -> str:
    return json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json_text(value), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stored_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sample_count_before(start_ns: int, cutoff_ns: int,
                        sampling_rate: float, npts: int) -> int:
    """Number of samples with sample time STRICTLY smaller than cutoff.

    Fraction arithmetic avoids rounding a sample at the snapshot into the prefix.
    No samples, amplitudes, or QC information from the future enter this decision.
    """
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Invalid raw sampling rate")
    if npts < 0:
        raise ValueError("Negative npts")
    if cutoff_ns <= start_ns:
        return 0
    pos = Fraction(int(cutoff_ns) - int(start_ns), 1_000_000_000)
    pos *= Fraction(str(float(sampling_rate)))
    exclusive_stop = -(-pos.numerator // pos.denominator)
    return min(int(npts), max(0, exclusive_stop))


def strict_raw_prefix(raw: Stream, cutoff: UTCDateTime) -> tuple[Stream, dict]:
    """Crop EVERY raw trace before QC, merge/interpolation, or response removal.

    All recorded past context in the same raw file is retained. Future-only
    segments are discarded BEFORE merging. No waveform padding is performed.
    """
    prefix = Stream()
    kept_n = 0
    future_n = 0
    for source in raw:
        k = sample_count_before(source.stats.starttime.ns, cutoff.ns,
                                float(source.stats.sampling_rate),
                                int(source.stats.npts))
        future_n += int(source.stats.npts) - k
        if k <= 0:
            continue
        tr = source.copy()
        tr.data = source.data[:k].copy()
        # Guard the nanosecond rounding used by ObsPy's endtime as well.
        while tr.stats.npts and tr.stats.endtime.ns >= cutoff.ns:
            tr.data = tr.data[:-1].copy()
        if tr.stats.npts:
            prefix += tr
            kept_n += int(tr.stats.npts)
    if not prefix:
        raise ValueError("No raw samples strictly before prediction snapshot")
    last_ns = max(tr.stats.endtime.ns for tr in prefix)
    if last_ns >= cutoff.ns:
        raise AssertionError("Raw prefix contains a sample at/after the snapshot")
    audit = {
        "raw_prefix_samples": kept_n,
        "raw_samples_at_or_after_snapshot": future_n,
        "raw_prefix_start": str(min(tr.stats.starttime for tr in prefix)),
        "raw_prefix_last_sample": str(max(tr.stats.endtime for tr in prefix)),
        "raw_last_sample_minus_snapshot_sec": (last_ns - cutoff.ns) / 1e9,
        "raw_prefix_fingerprint": prefix_fingerprint(prefix),
    }
    return prefix, audit


def prefix_fingerprint(prefix: Stream) -> str:
    h = hashlib.sha256()
    # Sorting is based on retained metadata only, not original full durations.
    traces = sorted(prefix, key=lambda tr: (tr.id, tr.stats.starttime.ns,
                                          float(tr.stats.sampling_rate),
                                          int(tr.stats.npts)))
    for tr in traces:
        header = [tr.id, tr.stats.starttime.ns,
                  float(tr.stats.sampling_rate), int(tr.stats.npts)]
        h.update(json.dumps(header, sort_keys=True).encode())
        a = np.ma.asarray(tr.data)
        h.update(np.asarray(np.ma.getdata(a), dtype="<f8").tobytes())
        h.update(np.asarray(np.ma.getmaskarray(a), dtype=np.uint8).tobytes())
    return h.hexdigest()


def input_transform(acceleration: np.ndarray) -> np.ndarray:
    # Same amplitude transform as the supplied downstream prediction script.
    a = np.asarray(acceleration, dtype=np.float32)
    return (np.sign(a) * np.log1p(np.abs(a) / 1e-3)).astype(np.float32)


def preprocess_prefix(raw: Stream, inventory, first_p: UTCDateTime,
                      cfg: Settings) -> tuple[np.ndarray, dict]:
    snapshot = first_p + cfg.t0_sec
    input_start = first_p - cfg.input_pre_sec
    prefix, audit = strict_raw_prefix(raw, snapshot)
    # A masked gap inside the prefix is a genuine missing observation. Do not
    # allow a masked or non-finite value to be silently treated as valid data.
    for tr in prefix:
        if np.ma.isMaskedArray(tr.data) and np.ma.getmaskarray(tr.data).any():
            raise ValueError(f"Masked raw prefix samples: {tr.id}")
    clip = validate_raw_stream(prefix)
    prepared = prepare_raw(prefix)
    if any(tr.stats.endtime.ns >= snapshot.ns for tr in prepared):
        raise AssertionError("Merge/prepare extended data beyond the snapshot")
    acc = rotate_to_zne(
        remove_response_stream(prepared, inventory, "ACC", cfg.high_hz), inventory
    )
    components = choose_components(acc)
    count = int(round((cfg.input_pre_sec + cfg.t0_sec) * cfg.sampling_rate))
    last_output = input_start + (count - 1) / cfg.sampling_rate
    if last_output.ns >= snapshot.ns:
        raise AssertionError("Requested resampling grid reaches the snapshot")
    # ObsPy interpolation does not extrapolate. Fail explicitly rather than
    # reading a future raw sample to fill the last output point.
    for name, tr in components.items():
        if tr.stats.starttime.ns > input_start.ns:
            raise ValueError(f"Insufficient prefix at input start: {tr.id}")
        if tr.stats.endtime.ns < last_output.ns:
            raise ValueError(f"Insufficient prefix at last output sample: {tr.id}; "
                             "no future-sample padding is permitted")
        if tr.stats.endtime.ns >= snapshot.ns:
            raise AssertionError(f"Response/rotation exceeded snapshot: {tr.id}")
    result = exact_array(components, input_start, snapshot, cfg.sampling_rate,
                         cfg.low_hz, cfg.high_hz)
    if result.shape != (3, count) or not np.isfinite(result).all():
        raise RuntimeError("Invalid prefix input tensor")
    audit.update({
        "prefix_raw_clip_fraction": clip,
        "input_start": str(input_start),
        "snapshot": str(snapshot),
        "last_output_sample": str(last_output),
        "input_npts": count,
        "selected_component_ids": [components[k].id for k in COMPONENT_ORDER],
        "prefix_context_sec": [float(snapshot - tr.stats.starttime) for tr in prefix],
    })
    return result, audit


def mutate_future(raw: Stream, snapshot: UTCDateTime,
                  variant: str) -> tuple[Stream, int]:
    """Audit-only future perturbation; never used for output inputs or labels."""
    altered = raw.copy()
    changed = 0
    for tr in altered:
        k = sample_count_before(tr.stats.starttime.ns, snapshot.ns,
                                float(tr.stats.sampling_rate), int(tr.stats.npts))
        if k >= tr.stats.npts:
            continue
        x = np.asarray(tr.data, dtype=np.float64).copy()
        previous = x[k:].copy()
        if variant == "scale2":
            x[k:] *= 2.0
        elif variant == "deterministic_noise":
            past = x[:k]
            scale = float(np.std(past)) if len(past) else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            seed = int.from_bytes(hashlib.sha256(tr.id.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            x[k:] = rng.normal(0.0, scale, len(x) - k)
        else:
            raise ValueError(f"Unknown perturbation: {variant}")
        changed += int(np.count_nonzero(previous != x[k:]))
        tr.data = x
    return altered, changed


def legacy_input_from_raw(raw: Stream, inventory, meta: dict,
                          first_p: UTCDateTime, cfg: Settings) -> np.ndarray:
    """Audit-only reconstruction of the ORIGINAL full-record ACC path."""
    validate_raw_stream(raw)
    prepared = prepare_raw(raw)
    acc = rotate_to_zne(
        remove_response_stream(prepared, inventory, "ACC", cfg.high_hz), inventory
    )
    start = first_p - meta["pre_first_p_sec"]
    end = start + meta["full_npts"] / cfg.sampling_rate
    full = exact_array(choose_components(acc), start, end, cfg.sampling_rate,
                       cfg.low_hz, cfg.high_hz)
    return full[:, meta["input_start_index"]:meta["snapshot_index"]]


def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def boundary_audit(raw: Stream, inventory, meta: dict, first_p: UTCDateTime,
                   cfg: Settings, prefix_array: np.ndarray,
                   prefix_audit: dict, archived_input: np.ndarray) -> list[dict]:
    """Compare two futures for prefix pipeline and original pipeline separately."""
    old = None
    old_error = ""
    try:
        old = legacy_input_from_raw(raw, inventory, meta, first_p, cfg)
    except Exception as exc:
        old_error = f"{type(exc).__name__}: {exc}"
    rows = []
    for variant in ("scale2", "deterministic_noise"):
        changed_raw, changed_count = mutate_future(raw, first_p + cfg.t0_sec, variant)
        row = {"variant": variant, "changed_future_sample_count": changed_count,
               "legacy_original_status": "ok" if old is not None else "failed",
               "legacy_original_error": old_error}
        if old is not None:
            row["legacy_reconstruction_vs_archive_max_abs"] = maxdiff(old, archived_input)
        try:
            new_array, new_audit = preprocess_prefix(changed_raw, inventory, first_p, cfg)
            same_raw = (new_audit["raw_prefix_fingerprint"] ==
                        prefix_audit["raw_prefix_fingerprint"])
            same_tensor = bool(np.array_equal(new_array, prefix_array))
            row.update({
                "prefix_raw_fingerprint_equal": same_raw,
                "prefix_input_bitwise_equal": same_tensor,
                "prefix_input_max_abs_diff": maxdiff(new_array, prefix_array),
                "prefix_transformed_max_abs_diff": maxdiff(
                    input_transform(new_array), input_transform(prefix_array)),
                "prefix_audit_status": (
                    "fail" if not (same_raw and same_tensor) else
                    "pass" if changed_count > 0 else "inconclusive_no_changed_future"),
            })
        except Exception as exc:
            row.update(prefix_audit_status="fail", prefix_audit_error=f"{type(exc).__name__}: {exc}")
        if old is not None:
            try:
                altered_old = legacy_input_from_raw(changed_raw, inventory, meta, first_p, cfg)
                row["legacy_perturbed_status"] = "ok"
                row["legacy_input_max_abs_diff"] = maxdiff(altered_old, old)
                row["legacy_transformed_max_abs_diff"] = maxdiff(
                    input_transform(altered_old), input_transform(old))
            except Exception as exc:
                row["legacy_perturbed_status"] = "failed"
                row["legacy_perturbed_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def portable_basename(path_text: str) -> str:
    return PureWindowsPath(path_text).name if "\\" in path_text else Path(path_text).name


def resolve_raw(path_text: str, event_id: str, raw_root: Path, subdir: str) -> Path:
    text = path_text.strip()
    if not text or text.lower() in {"nan", "none"}:
        raise FileNotFoundError(f"Missing archived {subdir} path for event {event_id}")
    name = portable_basename(text)
    candidates = [Path(text), Path.cwd() / text, raw_root / event_id / subdir / name]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError("File not found. Tried: " + "; ".join(map(str, candidates)))


def resolve_legacy(row: pd.Series, root: Path) -> Path:
    event_id = str(row["event_id"]).strip()
    direct = root / f"{event_id}.h5"
    if direct.is_file():
        return direct.resolve()
    text = str(row.get("h5_path", "")).strip()
    if text and text.lower() not in {"nan", "none"} and Path(text).is_file():
        return Path(text).resolve()
    raise FileNotFoundError(f"Original label HDF5 not found for {event_id}: {direct}")


def read_metadata(h5: h5py.File, event_id: str, cfg: Settings) -> dict:
    needed = ["acceleration", "velocity", "station_coords", "station_id",
              "p_offset_sec", "mseed_path", "stationxml_path"]
    missing = [k for k in needed if k not in h5]
    if missing:
        raise ValueError(f"Original HDF5 is missing {missing}")
    if stored_text(h5.attrs.get("event_id", "")) != event_id:
        raise ValueError("Manifest event_id and original HDF5 event_id differ")
    if stored_text(h5.attrs.get("component_order", "")) != "E,N,Z":
        raise ValueError("Original component order is not E,N,Z")
    if not np.isclose(float(h5.attrs["sampling_rate_hz"]), cfg.sampling_rate):
        raise ValueError("New and archived sampling rates differ; label task must stay fixed")
    n, comps, full_npts = h5["acceleration"].shape
    if comps != 3 or h5["velocity"].shape != (n, 3, full_npts):
        raise ValueError("Invalid archived acceleration/velocity dimensions")
    start = int(h5.attrs["time_zero_index"]) - int(round(cfg.input_pre_sec * cfg.sampling_rate))
    stop = int(h5.attrs["time_zero_index"]) + int(round(cfg.t0_sec * cfg.sampling_rate))
    if not (0 <= start < stop < full_npts):
        raise ValueError("Input/target intervals do not fit the original archive")
    ids = np.asarray(h5["station_id"].asstr()[:], dtype=str)
    if len(ids) != n or len(set(ids.tolist())) != n:
        raise ValueError("Archived station IDs are not unique/aligned")
    p = np.asarray(h5["p_offset_sec"][:], dtype=np.float32)
    if p.shape != (n,):
        raise ValueError("P offset length differs from station count")
    metadata = {
        "n_stations": n, "full_npts": full_npts,
        "first_p_time": stored_text(h5.attrs["first_p_time"]),
        "pre_first_p_sec": float(h5.attrs["pre_first_p_sec"]),
        "time_zero_index": int(h5.attrs["time_zero_index"]),
        "input_start_index": start, "snapshot_index": stop,
        "station_id": ids,
        "station_coords": np.asarray(h5["station_coords"][:], dtype=np.float32),
        "p_offset_sec": p,
        "snapshot_eligible": np.isfinite(p) & (p >= -1e-3) & (p <= cfg.t0_sec),
        "mseed_path": np.asarray(h5["mseed_path"].asstr()[:], dtype=str),
        "stationxml_path": np.asarray(h5["stationxml_path"].asstr()[:], dtype=str),
    }
    if metadata["station_coords"].shape != (n, 3):
        raise ValueError("Unexpected archived station coordinate shape")
    for key in ("s_offset_sec",):
        if key in h5:
            metadata[key] = np.asarray(h5[key][:], dtype=np.float32)
    for key in ("network", "station", "location", "channel_family"):
        if key in h5:
            metadata[key] = np.asarray(h5[key].asstr()[:], dtype=str)
    for key in ("mseed_path", "stationxml_path"):
        if len(metadata[key]) != n:
            raise ValueError(f"Archived {key} is not station-aligned")
    expected_zero = int(round(metadata["pre_first_p_sec"] * cfg.sampling_rate))
    if expected_zero != metadata["time_zero_index"]:
        raise ValueError("Archived time-axis attributes are inconsistent")
    return metadata


def validate_raw_identity(raw: Stream, meta: dict, index: int) -> None:
    # Never pick a different station merely because a similarly named file exists.
    if "network" not in meta or "station" not in meta:
        return
    net, sta = str(meta["network"][index]), str(meta["station"][index])
    if net and sta and any((tr.stats.network != net or tr.stats.station != sta) for tr in raw):
        raise ValueError(f"Raw station identity differs from archived {net}.{sta}")


def write_sidecar(path: Path, source: Path, source_hash: str, meta: dict,
                  cfg: Settings, config_hash: str, arrays: np.ndarray,
                  valid: np.ndarray, station_rows: list[dict], audit_rows: list[dict],
                  event_report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite original label file")
    tmp = path.with_name(path.name + ".part")
    strings = h5py.string_dtype("utf-8")
    try:
        with h5py.File(tmp, "w") as h:
            h.attrs["schema_version"] = SCHEMA_VERSION
            h.attrs["event_id"] = event_report["event_id"]
            h.attrs["config_sha256"] = config_hash
            h.attrs["config_json"] = json_text(asdict(cfg))
            h.attrs["source_h5_path"] = str(source.resolve())
            h.attrs["source_h5_sha256"] = source_hash
            h.attrs["first_p_time"] = meta["first_p_time"]
            h.attrs["sampling_rate_hz"] = cfg.sampling_rate
            h.attrs["component_order"] = "E,N,Z"
            h.attrs["acceleration_unit"] = "m/s^2"
            h.attrs["t0_sec"] = cfg.t0_sec
            h.attrs["input_pre_sec"] = cfg.input_pre_sec
            h.attrs["input_start_index_in_source"] = meta["input_start_index"]
            h.attrs["snapshot_index_in_source"] = meta["snapshot_index"]
            h.attrs["input_npts"] = arrays.shape[-1]
            h.attrs["n_stations"] = meta["n_stations"]
            h.attrs["all_eligible_inputs_valid"] = event_report["all_eligible_inputs_valid"]
            h.attrs["ready_for_paired_use"] = event_report["ready_for_paired_use"]
            h.attrs["is_streaming_causal_filter"] = False
            h.attrs["archived_arrival_picks_used"] = True
            h.attrs["label_policy"] = "Unchanged original HDF5; sidecar contains no prediction labels"
            h.attrs["history_policy"] = "all available raw samples strictly before snapshot"
            h.attrs["ineligible_rows_policy"] = "NaN placeholders, never valid model inputs"
            h.create_dataset("input_acceleration", data=arrays, dtype="float32",
                             compression="gzip", compression_opts=4, shuffle=True)
            h.create_dataset("input_valid", data=valid, dtype="bool")
            h.create_dataset("station_index", data=np.arange(meta["n_stations"]), dtype="int64")
            for key in ("station_coords", "p_offset_sec", "s_offset_sec", "snapshot_eligible"):
                if key in meta:
                    h.create_dataset(key, data=meta[key])
            for key in ("station_id", "mseed_path", "stationxml_path", "network",
                        "station", "location", "channel_family"):
                if key in meta:
                    h.create_dataset(key, data=np.asarray(meta[key], dtype=object), dtype=strings)
            for key, rows in (("station_report_json", station_rows),
                              ("boundary_audit_json", audit_rows),
                              ("event_report_json", event_report)):
                h.create_dataset(key, data=json_text(rows), dtype=strings)
            h.attrs["write_complete"] = True
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def read_reusable(path: Path, source: Path, config_hash: str,
                  raw_root: Path) -> tuple[dict, list[dict], list[dict]]:
    with h5py.File(path, "r") as h:
        if not bool(h.attrs.get("write_complete", False)):
            raise RuntimeError("Incomplete sidecar; rerun with --overwrite")
        if stored_text(h.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise RuntimeError("Sidecar schema mismatch; rerun with --overwrite")
        if stored_text(h.attrs.get("config_sha256", "")) != config_hash:
            raise RuntimeError("Sidecar settings/code/dependencies changed; use --overwrite")
        if stored_text(h.attrs.get("source_h5_sha256", "")) != sha256_file(source):
            raise RuntimeError("Original label archive changed; reconciliation is required")
        if not bool(h.attrs.get("ready_for_paired_use", False)):
            raise RuntimeError("Previous event was incomplete; inspect QC and use --overwrite")
        event = json.loads(h["event_report_json"].asstr()[()])
        stations = json.loads(h["station_report_json"].asstr()[()])
        audits = json.loads(h["boundary_audit_json"].asstr()[()])
        for row in stations:
            if row["status"] != "ok":
                continue
            for key, subdir in (("mseed", "waveforms"), ("stationxml", "stationxml")):
                resolved = resolve_raw(row[f"{key}_source_text"], event["event_id"], raw_root, subdir)
                if sha256_file(resolved) != row[f"{key}_sha256"]:
                    raise RuntimeError(f"Raw/inventory input changed for {row['station_id']}; "
                                       "use --overwrite after checking the provenance")
    event["status"] = "reused"
    event["source_h5_path"] = str(source.resolve())
    event["snapshot_input_h5_path"] = str(path.resolve())
    return event, stations, audits


def process_event(row: pd.Series, cfg: Settings, legacy_root: Path,
                  raw_root: Path, out_root: Path, config_hash: str,
                  overwrite: bool, debug: bool) -> tuple[dict, list[dict], list[dict]]:
    event_id = str(row["event_id"]).strip()
    source = resolve_legacy(row, legacy_root)
    output = out_root / "events" / f"{event_id}.h5"
    if output.resolve() == source.resolve():
        raise ValueError("Output must not be the source label HDF5")
    if output.exists() and not overwrite:
        return read_reusable(output, source, config_hash, raw_root)
    source_hash = sha256_file(source)
    with h5py.File(source, "r") as h:
        meta = read_metadata(h, event_id, cfg)
        # Used only to report the difference from archived inputs, never to fill
        # a failed new input and never to compute the new input itself.
        archived_inputs = np.asarray(h["acceleration"][:, :,
            meta["input_start_index"]:meta["snapshot_index"]], dtype=np.float32)
    first_p = parse_utc_timestamp(meta["first_p_time"], "archived first_p_time")
    n = meta["n_stations"]
    count = int(round((cfg.t0_sec + cfg.input_pre_sec) * cfg.sampling_rate))
    arrays = np.full((n, 3, count), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    station_rows, audit_rows = [], []
    attempted_audits = 0
    for index in range(n):
        eligible = bool(meta["snapshot_eligible"][index])
        report = {"event_id": event_id, "station_index": index,
                  "station_id": str(meta["station_id"][index]),
                  "snapshot_eligible": eligible,
                  "status": "not_snapshot_eligible", "error": ""}
        if not eligible:
            station_rows.append(report)
            continue
        try:
            rawpath = resolve_raw(meta["mseed_path"][index], event_id, raw_root, "waveforms")
            xmlpath = resolve_raw(meta["stationxml_path"][index], event_id, raw_root, "stationxml")
            report.update({"mseed_source_text": str(meta["mseed_path"][index]),
                           "stationxml_source_text": str(meta["stationxml_path"][index]),
                           "mseed_path_resolved": str(rawpath),
                           "stationxml_path_resolved": str(xmlpath),
                           "mseed_sha256": sha256_file(rawpath),
                           "stationxml_sha256": sha256_file(xmlpath)})
            raw = read(str(rawpath))
            inventory = read_inventory(str(xmlpath))
            validate_raw_identity(raw, meta, index)
            value, prefix_audit = preprocess_prefix(raw, inventory, first_p, cfg)
            report.update(prefix_audit)
            report["prefix_vs_archived_input_max_abs"] = maxdiff(value, archived_inputs[index])
            report["prefix_vs_archived_transformed_max_abs"] = maxdiff(
                input_transform(value), input_transform(archived_inputs[index]))
            report["status"] = "ok"
            arrays[index] = value
            valid[index] = True
            if attempted_audits < cfg.audit_stations_per_event:
                attempted_audits += 1
                checks = boundary_audit(raw, inventory, meta, first_p, cfg,
                                        value, prefix_audit, archived_inputs[index])
                for check in checks:
                    check.update(event_id=event_id, station_id=report["station_id"],
                                 station_index=index)
                audit_rows.extend(checks)
        except Exception as exc:
            arrays[index] = np.nan
            valid[index] = False
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            if debug:
                traceback.print_exc()
        station_rows.append(report)
    required = meta["snapshot_eligible"]
    all_valid = bool(required.any() and valid[required].all())
    n_audit_fail = sum(r.get("prefix_audit_status") == "fail" for r in audit_rows)
    n_audit_pass = sum(r.get("prefix_audit_status") == "pass" for r in audit_rows)
    event = {"event_id": event_id, "status": "ok" if all_valid and n_audit_fail == 0 else "incomplete",
             "n_original_stations": n, "n_snapshot_eligible": int(required.sum()),
             "n_valid_snapshot_inputs": int(valid.sum()),
             "n_failed_snapshot_inputs": int((required & ~valid).sum()),
             "all_eligible_inputs_valid": all_valid,
             "prefix_audit_variant_pass_count": n_audit_pass,
             "prefix_audit_variant_fail_count": n_audit_fail,
             "prefix_audit_variant_inconclusive_count": sum(
                 r.get("prefix_audit_status", "").startswith("inconclusive") for r in audit_rows),
             "ready_for_paired_use": all_valid and n_audit_fail == 0,
             "source_h5_path": str(source),
             "snapshot_input_h5_path": str(output.resolve()),
             "input_shape": list(arrays.shape), "error": ""}
    write_sidecar(output, source, source_hash, meta, cfg, config_hash, arrays,
                  valid, station_rows, audit_rows, event)
    return event, station_rows, audit_rows


def write_progress(out_root: Path, event_reports: list[dict],
                   station_reports: list[dict], audits: list[dict]) -> None:
    for name, rows in (("event_prefix_qc.csv", event_reports),
                       ("station_prefix_qc.csv", station_reports),
                       ("prefix_invariance_audit.csv", audits)):
        tmp = out_root / (name + ".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(out_root / name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", default="data/scedc/model_manifests/scenario_t0_5s_k5.csv",
                   help="Existing manifest; preserve event IDs and train/validation/test columns.")
    p.add_argument("--legacy-h5-root", default="data/scedc/processed_full_v4/events")
    p.add_argument("--raw-root", default="data/scedc/raw_m3_2010_2025")
    p.add_argument("--out-root", default="data/scedc/snapshot_available_t0_5s_k5")
    p.add_argument("--t0-sec", type=float, default=5.0)
    p.add_argument("--input-pre-sec", type=float, default=2.0)
    p.add_argument("--sampling-rate", type=float, default=50.0)
    p.add_argument("--low-hz", type=float, default=0.1)
    p.add_argument("--high-hz", type=float, default=20.0)
    p.add_argument("--audit-stations-per-event", type=int, default=2,
                   help="First N successfully processed eligible stations, deterministic order; 0 disables.")
    p.add_argument("--split-column", default="split_grouped")
    p.add_argument("--splits", default="", help="Optional comma-separated pilot subset, e.g. train,validation.")
    p.add_argument("--limit-events", type=int, default=None)
    p.add_argument("--expected-events", type=int, default=1620,
                   help="Optional exact count guard, e.g. 1620 for the complete primary manifest.")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--overwrite", action="store_true", help="Replace derived sidecars ONLY, never original files.")
    p.add_argument("--debug", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if obspy is None:
        raise ImportError("ObsPy is required for waveform processing. Use the same environment "
                          "as the original 07 script; --help works without ObsPy.")
    cfg = Settings(args.t0_sec, args.input_pre_sec, args.sampling_rate,
                   args.low_hz, args.high_hz, args.audit_stations_per_event)
    if not (cfg.t0_sec > 0 and cfg.input_pre_sec >= 0 and cfg.sampling_rate > 0
            and 0 < cfg.low_hz < cfg.high_hz < cfg.sampling_rate / 2):
        raise ValueError("Invalid time/frequency/sampling settings")
    for seconds in (cfg.t0_sec, cfg.input_pre_sec):
        if not np.isclose(seconds * cfg.sampling_rate,
                          round(seconds * cfg.sampling_rate), rtol=0, atol=1e-9):
            raise ValueError("Snapshot and input-start offsets must align to the archived sample grid")
    if cfg.audit_stations_per_event < 0 or args.checkpoint_every < 1:
        raise ValueError("Invalid audit/checkpoint count")
    if args.limit_events is not None and args.limit_events < 1:
        raise ValueError("--limit-events must be positive")
    events_path = Path(args.events).resolve()
    if not events_path.is_file():
        raise FileNotFoundError(f"Existing scenario manifest not found: {events_path}")
    legacy_root = Path(args.legacy_h5_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root == legacy_root or legacy_root in out_root.parents or (out_root / "events") == legacy_root:
        raise ValueError("Choose a NEW output tree outside processed_full_v4/events")
    out_root.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(events_path, dtype={"event_id": str})
    if "event_id" not in events:
        raise ValueError("Manifest needs event_id; no event identifiers will be guessed")
    events["event_id"] = events["event_id"].str.strip()
    if events["event_id"].isna().any() or events["event_id"].eq("").any() or events["event_id"].duplicated().any():
        raise ValueError("Manifest event IDs must be present and unique")
    # Avoid directory traversal in file names.
    if events["event_id"].str.contains(r"[\\/]", regex=True).any() or events["event_id"].isin([".", ".."]).any():
        raise ValueError("Invalid event ID for file naming")
    if args.splits:
        if args.split_column not in events:
            raise ValueError(f"Manifest has no {args.split_column!r} column")
        requested = [x.strip() for x in args.splits.split(",") if x.strip()]
        missing = set(requested) - set(events[args.split_column].astype(str))
        if missing:
            raise ValueError(f"Requested split labels absent: {sorted(missing)}")
        events = events.loc[events[args.split_column].astype(str).isin(requested)].copy()
    if args.limit_events is not None:
        events = events.iloc[:args.limit_events].copy()
    if events.empty:
        raise ValueError("No events remain after explicit pilot selection")
    if args.expected_events is not None and len(events) != args.expected_events:
        raise ValueError(f"Expected {args.expected_events} events, found {len(events)}")
    import scipy
    environment = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "obspy": obspy.__version__, "h5py": h5py.__version__}
    config = {"settings": asdict(cfg), "schema_version": SCHEMA_VERSION,
              "script_sha256": sha256_file(Path(__file__).resolve()),
              "environment": environment}
    config_hash = hashlib.sha256(json_text(config).encode()).hexdigest()
    write_json(out_root / "run_configuration.json", {
        **config, "config_sha256": config_hash,
        "events_manifest": str(events_path), "events_manifest_sha256": sha256_file(events_path),
        "selected_events": len(events), "splits_filter": args.splits,
        "legacy_h5_root": str(legacy_root), "raw_root": str(raw_root),
        "scope": "prefix-only fixed-snapshot input preprocessing; not end-to-end real-time validation"})
    event_reports, station_reports, audits, paired_rows = [], [], [], []
    print(f"Preparing {len(events)} events. Original labels and station indices are read-only.")
    print(f"Input shape per original station: (3, {int(round((cfg.t0_sec+cfg.input_pre_sec)*cfg.sampling_rate))})")
    for position, (_, row) in enumerate(events.iterrows(), 1):
        event_id = str(row["event_id"])
        try:
            report, station_rows, audit_rows = process_event(
                row, cfg, legacy_root, raw_root, out_root, config_hash, args.overwrite, args.debug)
        except Exception as exc:
            report = {"event_id": event_id, "status": "failed",
                      "ready_for_paired_use": False, "error": f"{type(exc).__name__}: {exc}"}
            station_rows, audit_rows = [], []
            if args.debug:
                traceback.print_exc()
        event_reports.append(report)
        station_reports.extend(station_rows)
        audits.extend(audit_rows)
        paired = row.to_dict()
        paired["snapshot_input_h5_path"] = report.get("snapshot_input_h5_path", "")
        paired["snapshot_input_status"] = report["status"]
        paired["ready_for_paired_use"] = report["ready_for_paired_use"]
        paired["snapshot_input_error"] = report.get("error", "")
        # Existing h5_path and split labels remain untouched in the new manifest.
        paired["source_h5_path_resolved"] = report.get("source_h5_path", "")
        paired_rows.append(paired)
        print(f"[{position}/{len(events)}] {event_id}: {report['status']} "
              f"{report.get('n_valid_snapshot_inputs', 0)}/{report.get('n_snapshot_eligible', 0)} eligible inputs")
        if position % args.checkpoint_every == 0 or position == len(events):
            write_progress(out_root, event_reports, station_reports, audits)
    pd.DataFrame(paired_rows).to_csv(out_root / "paired_manifest.csv", index=False)
    incomplete = [r["event_id"] for r in event_reports if not r["ready_for_paired_use"]]
    audit_pass = sum(r.get("prefix_audit_status") == "pass" for r in audits)
    audit_fail = sum(r.get("prefix_audit_status") == "fail" for r in audits)
    summary = {"requested_events": len(events), "ready_events": len(events) - len(incomplete),
               "incomplete_events": incomplete, "prefix_audit_variants_passed": audit_pass,
               "prefix_audit_variants_failed": audit_fail,
               "prefix_audit_variants_inconclusive": sum(
                   r.get("prefix_audit_status", "").startswith("inconclusive") for r in audits),
               "audit_scope": "Only the configured eligible stations per event; no prediction-level audit included",
               "original_labels_modified": False,
               "station_rows_removed_or_reordered": False,
               "instructions": "Do not silently discard incomplete events or resample stations. "
                   "Inspect QC; reconcile failures or declare a common eligible subset for all comparisons."}
    write_json(out_root / "run_summary.json", summary)
    print(json_text(summary))
    print(f"Output: {out_root}")
    if incomplete or audit_fail:
        print("INCOMPLETE: inspect QC; paired full-cohort training is not ready.", file=sys.stderr)
        return 2
    print("Preprocessing finished. Training/inference MUST use the supplied prefix-input reader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
