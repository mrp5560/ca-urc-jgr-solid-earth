"""Unit tests for index bounds, sidecar integrity and strict reader behavior.

These tests do NOT validate real miniSEED response removal. Tests that require
ObsPy are skipped if it is absent. No synthetic waveform is a paper result.
Run: python -m pytest -q test_snapshot_preprocessing.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("snapshot_preprocess", ROOT / "08_preprocess_snapshot_available_inputs.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from snapshot_input_reader import load_snapshot_input


@pytest.mark.parametrize("start,cut,fs,n,expected", [
    (0, 1_000_000_000, 50.0, 500, 50),
    (0, 1_000_000_001, 50.0, 500, 51),
    (0, 999_999_999, 50.0, 500, 50),
    (0, 0, 50.0, 500, 0),
    (100, 99, 50.0, 500, 0),
    (0, 1_000_000_000, 50.0, 8, 8),
    (0, 1_000_000_000, 62.5, 500, 63),
    (1_600_000_000_000_000_000, 1_600_000_001_000_000_000, 100.0, 500, 100),
])
def test_strict_sample_boundary(start, cut, fs, n, expected):
    assert module.sample_count_before(start, cut, fs, n) == expected


def test_invalid_rate():
    with pytest.raises(ValueError):
        module.sample_count_before(0, 10, 0, 4)


def test_portable_raw_basename():
    assert module.portable_basename(r"F:\data\event\waveforms\CI.AAA.mseed") == "CI.AAA.mseed"
    assert module.portable_basename("/data/CI.AAA.mseed") == "CI.AAA.mseed"


@pytest.fixture
def fixture_pair(tmp_path):
    source = tmp_path / "original.h5"
    cfg = module.Settings(t0_sec=1.0, input_pre_sec=0.0, audit_stations_per_event=0)
    n = 3
    strings = h5py.string_dtype("utf-8")
    with h5py.File(source, "w") as h:
        h.attrs.update(event_id="TEST", first_p_time="2020-01-01T00:00:00Z",
                       sampling_rate_hz=50.0, component_order="E,N,Z",
                       pre_first_p_sec=0.0, time_zero_index=0, n_stations=n)
        h.create_dataset("acceleration", data=np.ones((n, 3, 100), dtype=np.float32))
        h.create_dataset("velocity", data=np.ones((n, 3, 100), dtype=np.float32))
        h.create_dataset("station_coords", data=np.arange(n*3, dtype=np.float32).reshape(n, 3))
        h.create_dataset("p_offset_sec", data=np.array([0.0, 0.5, 2.0], np.float32))
        h.create_dataset("s_offset_sec", data=np.array([0.5, np.nan, 4.0], np.float32))
        for key, values in (("station_id", ["A", "B", "C"]),
                            ("mseed_path", ["A.mseed", "B.mseed", "C.mseed"]),
                            ("stationxml_path", ["A.xml", "B.xml", "C.xml"])):
            h.create_dataset(key, data=np.asarray(values, dtype=object), dtype=strings)
    original_hash = module.sha256_file(source)
    with h5py.File(source, "r") as h:
        meta = module.read_metadata(h, "TEST", cfg)
    a = np.full((n, 3, 50), np.nan, dtype=np.float32)
    a[0] = 0.001
    a[1] = 0.003
    sidecar = tmp_path / "prefix.h5"
    report = {"event_id": "TEST", "all_eligible_inputs_valid": True,
              "ready_for_paired_use": True}
    module.write_sidecar(sidecar, source, original_hash, meta, cfg, "test-config",
                         a, np.array([True, True, False]), [], [], report)
    return source, sidecar, a, original_hash


def test_source_archive_unchanged(fixture_pair):
    source, sidecar, _, digest = fixture_pair
    assert module.sha256_file(source) == digest
    with h5py.File(sidecar, "r") as h:
        assert "acceleration" not in h and "velocity" not in h
        assert "input_acceleration" in h
        assert np.isnan(h["input_acceleration"][2]).all()
        assert np.isnan(h["s_offset_sec"][1])


def test_reader_retains_unsorted_indices_and_transform(fixture_pair):
    source, sidecar, a, _ = fixture_pair
    with h5py.File(source, "r") as h:
        x = load_snapshot_input(h, sidecar, np.array([1, 0]), t0_sec=1, input_pre_sec=0)
    np.testing.assert_array_equal(x, module.input_transform(a[[1, 0]]))
    assert x.shape == (2, 3, 50)


def test_reader_untransformed(fixture_pair):
    source, sidecar, a, _ = fixture_pair
    with h5py.File(source, "r") as h:
        x = load_snapshot_input(h, sidecar, np.array([0]), t0_sec=1, input_pre_sec=0,
                                apply_log_transform=False)
    np.testing.assert_array_equal(x, a[[0]])


def test_no_fallback_for_ineligible_input(fixture_pair):
    source, sidecar, _, _ = fixture_pair
    with h5py.File(source, "r") as h, pytest.raises(ValueError, match="ineligible"):
        load_snapshot_input(h, sidecar, np.array([2]), t0_sec=1, input_pre_sec=0)


def test_no_fallback_for_missing_sidecar(fixture_pair):
    source, sidecar, _, _ = fixture_pair
    with h5py.File(source, "r") as h, pytest.raises(FileNotFoundError):
        load_snapshot_input(h, sidecar.with_name("missing.h5"), np.array([0]), t0_sec=1, input_pre_sec=0)


def test_failed_event_refused(fixture_pair):
    source, sidecar, _, _ = fixture_pair
    with h5py.File(sidecar, "r+") as h:
        h.attrs["ready_for_paired_use"] = False
    with h5py.File(source, "r") as h, pytest.raises(ValueError, match="failed prefix"):
        load_snapshot_input(h, sidecar, np.array([0]), t0_sec=1, input_pre_sec=0)


def test_station_reordering_refused(fixture_pair):
    source, sidecar, _, _ = fixture_pair
    with h5py.File(sidecar, "r+") as h:
        h["station_id"][0] = "B"
        h["station_id"][1] = "A"
    with h5py.File(source, "r") as h, pytest.raises(ValueError, match="Station order"):
        load_snapshot_input(h, sidecar, np.array([0]), t0_sec=1, input_pre_sec=0)


def test_modified_label_archive_refused(fixture_pair):
    source, sidecar, _, _ = fixture_pair
    with h5py.File(source, "r+") as h:
        h["velocity"][0, 0, 80] = 200.0
    with h5py.File(source, "r") as h, pytest.raises(ValueError, match="label HDF5 differs"):
        load_snapshot_input(h, sidecar, np.array([0]), t0_sec=1, input_pre_sec=0)


def test_wrong_snapshot_refused(fixture_pair):
    source, sidecar, _, _ = fixture_pair
    with h5py.File(source, "r") as h, pytest.raises(ValueError, match="t0_sec"):
        load_snapshot_input(h, sidecar, np.array([0]), t0_sec=5, input_pre_sec=0)


def test_no_float_indices(fixture_pair):
    source, sidecar, _, _ = fixture_pair
    with h5py.File(source, "r") as h, pytest.raises(ValueError, match="integer"):
        load_snapshot_input(h, sidecar, np.array([0.3]), t0_sec=1, input_pre_sec=0)


def test_strict_obspy_prefix_and_future_segments():
    obspy = pytest.importorskip("obspy")
    t = obspy.UTCDateTime("2020-01-01")
    a = obspy.Trace(np.arange(300, dtype=np.int32),
                    header={"starttime": t, "sampling_rate": 100., "channel": "HHE"})
    b = obspy.Trace(np.arange(50, dtype=np.int32),
                    header={"starttime": t + 4, "sampling_rate": 100., "channel": "HHE"})
    raw = obspy.Stream([a, b])
    prefix, audit = module.strict_raw_prefix(raw, t + 1)
    assert len(prefix) == 1 and len(prefix[0]) == 100
    assert prefix[0].stats.endtime.ns < (t + 1).ns
    altered, changed = module.mutate_future(raw, t + 1, "deterministic_noise")
    prefix2, audit2 = module.strict_raw_prefix(altered, t + 1)
    assert changed > 0
    assert audit["raw_prefix_fingerprint"] == audit2["raw_prefix_fingerprint"]
    np.testing.assert_array_equal(prefix[0].data, prefix2[0].data)
