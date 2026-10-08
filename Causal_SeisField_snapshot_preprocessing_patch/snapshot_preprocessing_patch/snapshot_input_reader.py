# -*- coding: utf-8 -*-
"""Strict input reader for the snapshot-available preprocessing sidecars.

This module NEVER reads the archived acceleration as a fallback input.
Continue using the ORIGINAL acceleration/velocity datasets for target labels.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = "snapshot_available_acceleration_v1"


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@lru_cache(maxsize=256)
def _checked_hash(filename: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with Path(filename).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_snapshot_input(
    original_h5: h5py.File,
    sidecar_path: str | Path,
    input_indices: np.ndarray,
    *,
    t0_sec: float = 5.0,
    input_pre_sec: float = 2.0,
    apply_log_transform: bool = True,
    verify_source_hash: bool = True,
) -> np.ndarray:
    """Return (K, 3, T) prefix-derived input, preserving input-index order.

    Parameters
    ----------
    original_h5:
        Read-only original full-waveform HDF5, still used for targets/metadata.
    sidecar_path:
        New sidecar produced by 08_preprocess_snapshot_available_inputs.py.
    input_indices:
        EXACT original station indices from the existing station-draw protocol.
    apply_log_transform:
        If True (default), return sign(a)*log1p(abs(a)/1e-3), float32.
        Do not apply this transform a second time in the model loader.
    verify_source_hash:
        Default True. Checks that the old label archive is the file used during
        preprocessing. Hashes are cached by path, file size and modification time.

    Refuses incomplete events to prevent unnoticed station resampling or fallback.
    The original event station order, time reference and P picks must match.
    """
    indices = np.asarray(input_indices)
    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
        raise ValueError("input_indices must be a non-empty 1-D integer array")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate input station indices")
    filename = Path(sidecar_path)
    if not filename.is_file():
        raise FileNotFoundError(f"Missing prefix sidecar (no fallback permitted): {filename}")
    with h5py.File(filename, "r") as side:
        if _text(side.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError("Not a supported snapshot-input sidecar")
        if not bool(side.attrs.get("write_complete", False)):
            raise ValueError("Sidecar was not fully written")
        if not bool(side.attrs.get("ready_for_paired_use", False)):
            raise ValueError("Event has failed prefix inputs/audits; do not silently "
                             "drop stations or replace them with other stations")
        if not bool(side.attrs.get("all_eligible_inputs_valid", False)):
            raise ValueError("The archived snapshot-eligible input pool is not fully reproduced")
        if _text(side.attrs["event_id"]) != _text(original_h5.attrs["event_id"]):
            raise ValueError("Event ID mismatch")
        if _text(side.attrs["first_p_time"]) != _text(original_h5.attrs["first_p_time"]):
            raise ValueError("Earliest network P reference changed")
        if _text(side.attrs["component_order"]) != "E,N,Z":
            raise ValueError("Wrong component order")
        for key, value in (("t0_sec", t0_sec), ("input_pre_sec", input_pre_sec)):
            if not np.isclose(float(side.attrs[key]), value, rtol=0, atol=1e-9):
                raise ValueError(f"Unexpected {key}; sidecar belongs to another task")
        fs = float(original_h5.attrs["sampling_rate_hz"])
        if not np.isclose(float(side.attrs["sampling_rate_hz"]), fs, rtol=0, atol=1e-9):
            raise ValueError("Sampling rate mismatch")
        n = original_h5["acceleration"].shape[0]
        if np.any(indices < 0) or np.any(indices >= n):
            raise IndexError("Input station index is outside the original archive")
        original_ids = original_h5["station_id"].asstr()[:]
        side_ids = side["station_id"].asstr()[:]
        if not np.array_equal(original_ids, side_ids):
            raise ValueError("Station order/identity changed")
        if not np.array_equal(side["station_index"][:], np.arange(n)):
            raise ValueError("Sidecar station indices were reindexed")
        for key in ("station_coords", "p_offset_sec"):
            if not np.array_equal(original_h5[key][:], side[key][:], equal_nan=True):
                raise ValueError(f"Original metadata changed: {key}")
        start = int(original_h5.attrs["time_zero_index"]) - int(round(input_pre_sec * fs))
        stop = int(original_h5.attrs["time_zero_index"]) + int(round(t0_sec * fs))
        if int(side.attrs["input_start_index_in_source"]) != start or int(side.attrs["snapshot_index_in_source"]) != stop:
            raise ValueError("Sidecar time grid and original input window differ")
        if side["input_acceleration"].shape != (n, 3, stop - start):
            raise ValueError("Unexpected prefix array shape")
        p = np.asarray(original_h5["p_offset_sec"][:], dtype=float)
        eligible = np.isfinite(p) & (p >= -1e-3) & (p <= t0_sec)
        if not np.array_equal(side["snapshot_eligible"][:], eligible):
            raise ValueError("Snapshot eligibility changed")
        valid = side["input_valid"][:]
        if not np.all(eligible[indices]) or not np.all(valid[indices]):
            raise ValueError("A selected input station is ineligible or failed preprocessing")
        if verify_source_hash:
            source = Path(original_h5.filename).resolve()
            stat = source.stat()
            actual = _checked_hash(str(source), stat.st_size, stat.st_mtime_ns)
            if actual != _text(side.attrs["source_h5_sha256"]):
                raise ValueError("Original label HDF5 differs from preprocessing source")
        # Individual indexing retains the original, possibly unsorted input order.
        data = np.stack([side["input_acceleration"][int(i)] for i in indices]).astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Selected prefix inputs contain NaN/Inf; no fallback permitted")
    if apply_log_transform:
        data = (np.sign(data) * np.log1p(np.abs(data) / 1e-3)).astype(np.float32)
    return data
