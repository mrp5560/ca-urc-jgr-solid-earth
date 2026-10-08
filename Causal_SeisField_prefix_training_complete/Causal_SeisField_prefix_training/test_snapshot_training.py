"""Synthetic tests only. None of the numbers are scientific experiment results."""
import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

import source45_baselines as b45
import source63_risk as h63
import snapshot_experiment_data as data
import train_snapshot_experiment as run
from snapshot_input_reader import SCHEMA_VERSION


def build_fixture(root):
    labels, prefix = root / "labels", root / "prefix"
    labels.mkdir(parents=True)
    prefix.mkdir()
    rows = []
    specifications = [("tr1", "train", .04, "g1"), ("tr2", "train", 3., "g2"),
                      ("tr3", "train", 4., "g3"), ("va1", "validation", .04, "g4"),
                      ("va2", "validation", 3., "g5"), ("te1", "test", .04, "g6"),
                      ("te2", "test", 3., "g7")]
    rng = np.random.default_rng(19)
    n, samples = 18, 900
    strings = h5py.string_dtype("utf-8")
    for eid, split, scale, group in specifications:
        old = labels / f"{eid}.h5"
        coords = np.column_stack([34. + rng.normal(0, .1, n), -117. + rng.normal(0, .1, n), rng.uniform(0, 500, n)]).astype(np.float32)
        offsets = np.r_[np.linspace(0, 4.9, 10), np.linspace(5.2, 8, 8)].astype(np.float32)
        acc = (rng.normal(size=(n, 3, samples)) * scale).astype(np.float32)
        vel = (rng.normal(size=(n, 3, samples)) * scale * .3).astype(np.float32)
        ids = np.array([f"X.S{i:02d}.--" for i in range(n)], dtype=object)
        with h5py.File(old, "w") as h:
            h.attrs.update({"event_id": eid, "first_p_time": "2020-01-01T00:00:10Z",
                            "sampling_rate_hz": 50., "pre_first_p_sec": 5., "time_zero_index": 250})
            for name, arr in [("acceleration", acc), ("velocity", vel), ("station_coords", coords), ("p_offset_sec", offsets)]:
                h.create_dataset(name, data=arr)
            h.create_dataset("station_id", data=ids, dtype=strings)
        eligible = offsets <= 5
        array = np.full((n, 3, 350), np.nan, np.float32)
        array[eligible] = rng.normal(size=(int(eligible.sum()), 3, 350)).astype(np.float32) * .001
        with h5py.File(prefix / f"{eid}.h5", "w") as h:
            h.attrs.update({"schema_version": SCHEMA_VERSION, "write_complete": True,
                            "ready_for_paired_use": True, "all_eligible_inputs_valid": True,
                            "event_id": eid, "first_p_time": "2020-01-01T00:00:10Z",
                            "component_order": "E,N,Z", "sampling_rate_hz": 50.,
                            "t0_sec": 5., "input_pre_sec": 2., "input_start_index_in_source": 150,
                            "snapshot_index_in_source": 500, "source_h5_sha256": data.sha256_file(old),
                            "config_sha256": "synthetic_test_only"})
            for name, arr in [("input_acceleration", array), ("station_coords", coords),
                              ("p_offset_sec", offsets), ("input_valid", eligible),
                              ("snapshot_eligible", eligible), ("station_index", np.arange(n))]:
                h.create_dataset(name, data=arr)
            h.create_dataset("station_id", data=ids, dtype=strings)
            h.create_dataset("event_report_json", data=json.dumps({"n_snapshot_eligible": int(eligible.sum()),
                             "n_valid_snapshot_inputs": int(eligible.sum()), "n_failed_snapshot_inputs": 0}), dtype=strings)
            h.create_dataset("station_report_json", data="[]", dtype=strings)
            h.create_dataset("boundary_audit_json", data=json.dumps([{"prefix_audit_status": "pass"}]), dtype=strings)
        rows.append({"event_id": eid, "h5_path": str(old), "split_grouped": split, "test_group": group})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    thresholds = root / "thresholds.json"
    thresholds.write_text(json.dumps({"log10_pga_threshold": .5, "log10_pgv_threshold": 0.}), encoding="utf-8")
    args = run.build_parser().parse_args([])
    args.manifest, args.h5_root, args.prefix_root = str(manifest), str(labels), str(prefix)
    args.out_dir, args.expected_events = str(root / "results"), 7
    args.threshold_json, args.test_repeats = str(thresholds), 2
    args.group_column, args.head_seed = "test_group", args.seed + 2000
    args.device_resolved = "cpu"
    return args


def common_for(args, cohort):
    return {"manifest": cohort["manifest"], "h5_root": args.h5_root,
            "split_column": args.split_column, "t0_sec": args.t0_sec,
            "input_stations": 5, "target_stations": 10, "input_pre_sec": 2.,
            "seed": args.seed, "prefix_root": args.prefix_root,
            "event_fingerprints": cohort["fingerprints"]}


def make_reference(args, common):
    ds = data.StrongBaselineDataset(split_name="test", training=False, repeats=2, **common)
    rows = []
    for i in range(len(ds)):
        batch = ds[i]
        for slot, idx in enumerate(batch["target_station_index"].tolist()):
            a, v = batch["target_log"][slot].tolist()
            rows.append({"event_id": batch["event_id"], "repeat": int(batch["repeat"]),
                         "target_slot": slot, "target_station_index": idx,
                         "true_log10_pga": a, "true_log10_pgv": v,
                         "is_tail_pga": a >= .5, "is_tail_pgv": v >= 0.,
                         "A2_cross_attention_base_log10_pga": a-.4,
                         "A2_cross_attention_base_log10_pgv": v-.4,
                         "A4_under_only_log10_pga": a-.3,
                         "A4_under_only_log10_pgv": v-.3})
    frame = pd.DataFrame(rows).sample(frac=1, random_state=123).reset_index(drop=True)
    path = Path(args.out_dir).parent / "reference.csv"
    frame.to_csv(path, index=False)
    return frame, path


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("suffix", ["epoch:3", "repeat:1"])
def test_seed_equals_source63(split, suffix):
    namespace = SimpleNamespace()
    if split == "test":
        h63.patch_locked_test_seed_protocol(namespace)
    else:
        h63.patch_train_validation_seed_protocol(namespace)
    string = f"strong:{split}:12345:{suffix}"
    assert data.paired_draw_seed(string, 20260713) == namespace.stable_seed(string, 20260713)


def test_raw45_seed_is_not_locked_seed():
    assert b45.stable_seed("strong:test:12345:repeat:0", 20260713) != data.paired_draw_seed("strong:test:12345:repeat:0", 20260713)


def test_prefix_replacement_preserves_every_other_model_tensor(tmp_path, monkeypatch):
    args = build_fixture(tmp_path)
    cohort = data.preflight_cohort(args)
    common = common_for(args, cohort)
    prefix = data.StrongBaselineDataset(split_name="test", training=False, repeats=2, **common)
    old_common = {k: v for k, v in common.items() if k not in ("prefix_root", "event_fingerprints")}
    monkeypatch.setattr(b45, "stable_seed", data.paired_draw_seed)
    original = b45.StrongBaselineDataset(split_name="test", training=False, repeats=2, **old_common)
    for i in range(len(prefix)):
        new, old = prefix[i], original[i]
        for key in ("target_station_index", "target_slot", "repeat", "input_features", "target_features", "target_log", "distances_km"):
            assert torch.equal(new[key], old[key]), key
        assert not torch.equal(new["input_waveforms"], old["input_waveforms"])
        assert "observed_log10_pgv" not in new
        draw = prefix.draw(i)
        with h5py.File(draw["prefix"], "r") as h:
            raw = np.stack([h["input_acceleration"][j] for j in draw["input_indices"]])
        expected = (np.sign(raw) * np.log1p(np.abs(raw) / 1e-3)).astype(np.float32)
        np.testing.assert_array_equal(new["input_waveforms"].numpy(), expected)


def test_epoch_draws_change_and_repeat_draws_are_stable(tmp_path):
    args = build_fixture(tmp_path)
    common = common_for(args, data.preflight_cohort(args))
    train = data.StrongBaselineDataset(split_name="train", training=True, repeats=1, **common)
    train.set_epoch(1); a = train.draw(0)
    train.set_epoch(2); b = train.draw(0)
    assert not np.array_equal(a["input_indices"], b["input_indices"])
    test = data.StrongBaselineDataset(split_name="test", training=False, repeats=2, **common)
    a = test.draw(1)
    h63.patch_train_validation_seed_protocol(data)  # must not alter Dataset's test convention
    b = test.draw(1)
    np.testing.assert_array_equal(a["target_indices"], b["target_indices"])


def test_incomplete_event_blocks_and_reports(tmp_path):
    args = build_fixture(tmp_path)
    p = Path(args.prefix_root) / "te1.h5"
    with h5py.File(p, "a") as h:
        h.attrs["ready_for_paired_use"] = False
        h.attrs["all_eligible_inputs_valid"] = False
        h["input_valid"][0] = False
    with pytest.raises(RuntimeError, match="Unresolved events=1"):
        data.preflight_cohort(args)
    report = pd.read_csv(Path(args.out_dir) / "audit" / "incomplete_events.csv")
    assert report.event_id.tolist() == ["te1"]
    assert not (Path(args.out_dir) / "base").exists()


def test_reviewed_exclusion_preserves_splits(tmp_path):
    args = build_fixture(tmp_path)
    with h5py.File(Path(args.prefix_root)/"tr1.h5", "a") as h:
        h.attrs["ready_for_paired_use"] = False
    exclusion = tmp_path / "exclusions.csv"
    pd.DataFrame([{"event_id": "tr1", "reason": "Synthetic test: reviewed corrupt input"}]).to_csv(exclusion, index=False)
    args.excluded_events_csv = str(exclusion)
    cohort = data.preflight_cohort(args)
    assert len(cohort["frame"]) == 6
    assert cohort["frame"].set_index("event_id").loc["te1", "split_grouped"] == "test"


def test_empty_exclusion_reason_rejected(tmp_path):
    args = build_fixture(tmp_path)
    exclusion = tmp_path / "exclusions.csv"
    pd.DataFrame([{"event_id": "tr1", "reason": ""}]).to_csv(exclusion, index=False)
    args.excluded_events_csv = str(exclusion)
    with pytest.raises(ValueError, match="reason"):
        data.preflight_cohort(args)


def test_source_changed_after_preflight_rejected(tmp_path):
    args = build_fixture(tmp_path)
    common = common_for(args, data.preflight_cohort(args))
    ds = data.StrongBaselineDataset(split_name="test", training=False, repeats=2, **common)
    with h5py.File(Path(args.h5_root)/"te1.h5", "a") as h:
        h["acceleration"][0,0,0] += 1
    with pytest.raises(RuntimeError, match="changed since"):
        ds[0]


def test_prefix_changed_after_preflight_rejected(tmp_path):
    args = build_fixture(tmp_path)
    common = common_for(args, data.preflight_cohort(args))
    ds = data.StrongBaselineDataset(split_name="test", training=False, repeats=2, **common)
    with h5py.File(Path(args.prefix_root)/"te1.h5", "a") as h:
        h["input_acceleration"][0,0,0] += 1
    with pytest.raises(RuntimeError, match="changed since"):
        ds[0]


def test_reference_reordering_is_exact(tmp_path):
    args = build_fixture(tmp_path)
    cohort = data.preflight_cohort(args)
    common = common_for(args, cohort)
    reference, _ = make_reference(args, common)
    current = reference[run.KEYS + run.TRUTH].sample(frac=1, random_state=321).reset_index(drop=True)
    aligned, audit = run.align_reference(current, reference, {"te1", "te2"}, np.array([.5,0],np.float32))
    assert audit["exact_key_pairing"]
    assert aligned[run.KEYS].equals(current[run.KEYS])
    np.testing.assert_allclose(aligned["A4_under_only_log10_pga"], current["true_log10_pga"]-.3)


def test_reference_mismatch_fails(tmp_path):
    args = build_fixture(tmp_path)
    common = common_for(args, data.preflight_cohort(args))
    reference, _ = make_reference(args, common)
    current = reference[run.KEYS + run.TRUTH].copy()
    current.loc[0, "target_station_index"] = 99
    with pytest.raises(RuntimeError, match="station-draw mismatch"):
        run.align_reference(current, reference, {"te1", "te2"}, np.array([.5,0]))


def test_no_old_checkpoint_fallback():
    with pytest.raises(RuntimeError, match="Checkpoint does not belong"):
        run.verify_protocol({"variant": "cross_attention"}, {"protocol_id": "new"})


def test_hierarchical_bootstrap_keeps_nan_subset_empty():
    f = pd.DataFrame({"event_id": ["a","b","c"], "g": ["G1","G1","G2"]})
    w, groups = run.bootstrap_weights(f, ["a","b","c"], "g", 1000, 42)
    assert w.shape == (1000,3)
    d = np.array([np.nan, -.1, -.1])
    den = w @ np.isfinite(d).astype(float)
    num = w @ np.nan_to_num(d)
    np.testing.assert_allclose(num[den>0]/den[den>0], -.1)


def test_missing_group_column_does_not_fake_ci():
    f = pd.DataFrame({"event_id": ["a", "b"]})
    assert run.bootstrap_weights(f, ["a", "b"], None, 100, 42) == (None, None)


def test_architecture_context_matches_original():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    model = b45.make_model("cross_attention", 16, 4, 50.)
    model.eval()
    waves, fi, ft = torch.randn(2,5,3,350), torch.randn(2,5,4), torch.randn(2,10,3)
    with torch.no_grad():
        out = model(waves,fi,ft)
        frozen_out, features = h63.cross_attention_base_context(model, waves,fi,ft)
    assert torch.equal(out, frozen_out)
    assert features.shape == (2,10,37)


def test_synthetic_end_to_end(tmp_path):
    args = build_fixture(tmp_path)
    common = common_for(args, data.preflight_cohort(args))
    _, reference = make_reference(args, common)
    command = [sys.executable, str(Path(run.__file__).resolve()), "--stage", "all",
               "--manifest", args.manifest, "--h5-root", args.h5_root,
               "--prefix-root", args.prefix_root, "--threshold-json", args.threshold_json,
               "--reference-predictions", str(reference), "--out-dir", args.out_dir,
               "--expected-events", "7", "--test-repeats", "2", "--validation-repeats", "1",
               "--selection-validation-repeats", "2", "--training-validation-repeats", "1",
               "--prevalence-repeats", "1", "--base-epochs", "1", "--head-epochs", "1",
               "--hidden-dim", "16", "--risk-hidden", "8", "--powers", "1,2,20",
               "--base-learning-rate", "0.00000001", "--head-learning-rate", "0.00000001",
               "--overall-budget", "10", "--non-tail-budget", "10", "--bias-limit", "10",
               "--bootstrap-replicates", "100", "--group-column", "test_group",
               "--device", "cpu", "--torch-threads", "1"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (tmp_path / "SYNTHETIC_TEST_ONLY_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (tmp_path / "SYNTHETIC_TEST_ONLY_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evaluation = Path(args.out_dir) / "evaluation"
    output = pd.read_csv(evaluation/"locked_prefix_test_predictions.csv")
    assert len(output) == 40
    assert output[run.KEYS].duplicated().sum() == 0
    for q in ("pga","pgv"):
        np.testing.assert_allclose(output[f"FullRecord_CAURC_log10_{q}"], output[f"true_log10_{q}"]-.3)
        assert (output[f"Prefix_CAURC_log10_{q}"] >= output[f"Prefix_Base_log10_{q}"]-1e-10).all()
    audit = json.loads((evaluation/"evaluation_audit.json").read_text())
    assert audit["frozen_base_unchanged"]
    changes = pd.read_csv(evaluation/"paired_changes.csv")
    assert changes["ci95_low"].notna().any()
