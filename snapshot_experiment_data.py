# -*- coding: utf-8 -*-
"""Prefix-input adapter for the uploaded 45/63 code. No new model architecture.

Only prefix acceleration enters the neural input. All targets and station metadata
remain in the original full-waveform archive. Failed eligible stations are NEVER
removed from the draw pool, replaced, or filled with archived inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

import source45_baselines as original45
from snapshot_input_reader import load_snapshot_input

# These interfaces are used by the original source63.run_variant().
make_model = original45.make_model


def text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False,
                              indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def stable_seed(text_value: str, base_seed: int) -> int:
    """Compatibility symbol; Dataset uses paired_draw_seed directly.

    source63 may patch this symbol while building its train/validation/test
    loaders. The Dataset uses a per-split rule instead, so its sampling cannot
    change when an unrelated loader is constructed or when Windows spawns.
    """
    return paired_draw_seed(text_value, base_seed)


def paired_draw_seed(text_value: str, base_seed: int) -> int:
    """Exactly source63's combined protocol: text:seed, locked:test prefix.

    Original source45 alone uses seed:text and strong:*. It is NOT the draw
    convention in the later locked CA-URC archive; source63 explicitly patches it.
    """
    value = str(text_value)
    if value.startswith("strong:test:"):
        value = "locked:" + value[len("strong:"):]
    elif value.startswith("strong:"):
        value = value[len("strong:"):]
    digest = hashlib.sha256(f"{value}:{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 32)


def stat_token(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    s = p.stat()
    return {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def assert_file_unchanged(path: str | Path, checked: dict[str, Any]) -> None:
    if stat_token(path) != {k: checked[k] for k in ("path", "size", "mtime_ns")}:
        raise RuntimeError(f"Input file changed since this run's full preflight: {path}. "
                           "Stop and rerun the audit; no fallback is allowed.")


def _read_json_dataset(h5: h5py.File, name: str, default: Any) -> Any:
    if name not in h5:
        return default
    return json.loads(text(h5[name][()]))


def _read_exclusions(path: str | None, ids: set[str]) -> dict[str, str]:
    if not path:
        return {}
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"event_id", "reason"}.issubset(f.columns):
        raise ValueError("Reviewed exclusions CSV must have event_id,reason columns.")
    f["event_id"] = f["event_id"].str.strip()
    if f["event_id"].duplicated().any() or not set(f["event_id"]).issubset(ids):
        raise ValueError("Duplicate or unknown event IDs in the reviewed exclusions CSV.")
    forbidden = {"", "todo", "review required", "pending"}
    if f["reason"].str.strip().str.lower().isin(forbidden).any():
        raise ValueError("Every exclusion needs an explicit, reviewed non-placeholder reason.")
    return dict(zip(f["event_id"], f["reason"].str.strip()))


def preflight_cohort(args) -> dict[str, Any]:
    """Read-only check of EVERY requested event; no automatic cohort filtering.

    Writes reports even if some events are incomplete. A reviewed exclusions file
    is the only way to form a smaller cohort. Original split assignments and row
    order are retained. An event is required to reproduce its entire eligible
    input pool, which is needed by epoch-varying training station draws.
    """
    out = Path(args.out_dir) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    source_manifest = Path(args.manifest).resolve()
    frame = pd.read_csv(source_manifest, dtype={"event_id": str})
    needed = {"event_id", "h5_path", args.split_column}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Manifest missing: {sorted(needed - set(frame.columns))}")
    frame["event_id"] = frame["event_id"].str.strip()
    if frame["event_id"].isna().any() or frame["event_id"].duplicated().any():
        raise ValueError("Manifest event IDs must be nonmissing and unique.")
    splits = frame[args.split_column].astype(str)
    if not splits.isin(["train", "validation", "test"]).all():
        raise ValueError("Supply the primary scenario manifest with train/validation/test "
                         "labels. Unknown labels are not silently discarded.")
    if args.expected_events and len(frame) != args.expected_events:
        raise ValueError(f"Requested {len(frame)} events; expected {args.expected_events}. "
                         "For a declared different cohort, set --expected-events explicitly.")
    eligible_col = f"eligible_t0_{int(args.t0_sec)}s_k{args.input_stations}"
    if eligible_col in frame and not original45.as_bool(frame[eligible_col]).all():
        raise ValueError("Manifest includes events outside the original scenario. "
                         "Supply the original scenario manifest, not a broad master list.")
    exclusions = _read_exclusions(args.excluded_events_csv, set(frame["event_id"]))
    prefix_root = Path(args.prefix_root).resolve()
    label_root = Path(args.h5_root).resolve()
    checks, station_failures, content_records = [], [], []
    fingerprints: dict[str, dict[str, Any]] = {}
    runtime_frame = frame.copy()
    runtime_frame["snapshot_input_h5_path"] = ""
    for row_index, row in frame.iterrows():
        eid = str(row["event_id"])
        prefix = prefix_root / f"{eid}.h5"
        rec: dict[str, Any] = {
            "event_id": eid, "split": str(row[args.split_column]),
            "ready": False, "declared_excluded": eid in exclusions,
            "exclusion_reason": exclusions.get(eid, ""), "error": "",
            "prefix_path": str(prefix), "source_path": "",
        }
        try:
            source = original45.resolve_h5_path(eid, str(row["h5_path"]), label_root).resolve()
            rec["source_path"] = str(source)
            runtime_frame.loc[row_index, "h5_path"] = str(source)
            runtime_frame.loc[row_index, "snapshot_input_h5_path"] = str(prefix)
            if prefix.exists():
                with h5py.File(prefix, "r") as side:
                    for name in ("write_complete", "ready_for_paired_use", "all_eligible_inputs_valid"):
                        rec[name] = bool(side.attrs.get(name, False))
                    event_report = _read_json_dataset(side, "event_report_json", {})
                    for k in ("n_snapshot_eligible", "n_valid_snapshot_inputs",
                              "n_failed_snapshot_inputs", "prefix_audit_variant_pass_count",
                              "prefix_audit_variant_fail_count", "prefix_audit_variant_inconclusive_count"):
                        rec[k] = event_report.get(k)
                    for station in _read_json_dataset(side, "station_report_json", []):
                        if station.get("status") == "failed":
                            station_failures.append({"event_id": eid, **station})
                    audits = _read_json_dataset(side, "boundary_audit_json", [])
                    statuses = [str(x.get("prefix_audit_status", "")) for x in audits]
                    rec["audited_variants_passed"] = sum(s == "pass" for s in statuses)
                    rec["audited_variants_failed"] = sum(s == "fail" for s in statuses)
                    rec["audited_variants_inconclusive"] = sum(s.startswith("inconclusive") for s in statuses)
                    cfg_hash = text(side.attrs.get("config_sha256", ""))
            else:
                raise FileNotFoundError(f"Missing prefix file: {prefix}")
            before_source, before_prefix = stat_token(source), stat_token(prefix)
            with h5py.File(source, "r") as old:
                p = np.asarray(old["p_offset_sec"][:], dtype=float)
                eligible = np.flatnonzero(np.isfinite(p) & (p >= -1e-3) & (p <= args.t0_sec))
                rec["eligible_pool_size"] = len(eligible)
                if len(eligible) < args.input_stations:
                    raise ValueError("Original eligible input pool has fewer than K stations.")
                if len(p) - args.input_stations < args.target_stations:
                    raise ValueError("Original target pool has fewer than Q stations.")
                # Reads and validates ALL eligible prefix arrays; verifies old HDF5 SHA256.
                load_snapshot_input(old, prefix, eligible, t0_sec=args.t0_sec,
                                    input_pre_sec=args.input_pre_sec, apply_log_transform=False,
                                    verify_source_hash=True)
                with h5py.File(prefix, "r") as side:
                    old_hash = text(side.attrs["source_h5_sha256"])
            if rec["audited_variants_failed"] or rec["audited_variants_inconclusive"]:
                raise ValueError("Prefix boundary audits contain failures or inconclusive variants.")
            if not rec["audited_variants_passed"]:
                raise ValueError("No passed boundary-audit variants stored for this event.")
            prefix_hash = sha256_file(prefix)
            assert_file_unchanged(source, before_source)
            assert_file_unchanged(prefix, before_prefix)
            rec["ready"] = True
            if eid not in exclusions:
                fingerprints[eid] = {"source": before_source, "prefix": before_prefix}
                content_records.append({"event_id": eid, "split": str(row[args.split_column]),
                                        "source_sha256": old_hash, "prefix_sha256": prefix_hash,
                                        "preprocessing_config_sha256": cfg_hash})
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        checks.append(rec)
        if (len(checks) % 100 == 0) or len(checks) == len(frame):
            print(f"Preflight {len(checks)}/{len(frame)}", flush=True)
            pd.DataFrame(checks).to_csv(out / "event_input_checks.csv", index=False)
    check_frame = pd.DataFrame(checks)
    failed = check_frame.loc[~check_frame["ready"]].copy()
    failed.to_csv(out / "incomplete_events.csv", index=False)
    pd.DataFrame(station_failures).to_csv(out / "failed_station_details.csv", index=False)
    review = failed[["event_id", "split", "error"]].copy()
    review["reason"] = ""  # Must be reviewed; passing this untouched will fail.
    review.to_csv(out / "exclusions_REVIEW_REQUIRED.csv", index=False)
    accepted = runtime_frame.loc[~runtime_frame["event_id"].isin(exclusions)].copy()
    counts = []
    for split in ("train", "validation", "test"):
        g = check_frame.loc[check_frame["split"].eq(split)]
        counts.append({"split": split, "requested_events": len(g),
                       "prefix_ready_events": int(g["ready"].sum()),
                       "declared_excluded_events": int(g["declared_excluded"].sum()),
                       "retained_events": int((~g["declared_excluded"]).sum()),
                       "retained_unready_events": int((~g["declared_excluded"] & ~g["ready"]).sum())})
    pd.DataFrame(counts).to_csv(out / "cohort_counts_by_split.csv", index=False)
    accepted_path = out / "cohort_manifest.csv"
    accepted.to_csv(accepted_path, index=False)  # Failed retained rows remain visible; no filtering!
    unresolved = failed.loc[~failed["declared_excluded"], "event_id"].tolist()
    config_hashes = sorted({r["preprocessing_config_sha256"] for r in content_records})
    nonempty = all(c["retained_events"] > 0 for c in counts)
    uniform_config = len(config_hashes) == 1 and bool(config_hashes[0])
    ok = not unresolved and nonempty and uniform_config
    summary = {"ready_for_training": ok, "requested_events": len(frame),
               "retained_events": len(accepted), "unresolved_events": unresolved,
               "reviewed_exclusions": exclusions, "counts_by_split": counts,
               "preprocessing_config_hashes": config_hashes,
               "manifest_columns": list(frame.columns),
               "audit_scope": "All retained eligible prefix inputs checked; perturbation coverage is only the stored audited variants",
               "input_policy": "prefix acceleration only; no prefix velocity, no propagation baseline",
               "cohort_policy": "No resampling/repartitioning/automatic exclusion",
               "test_labels_read_for_pairing_only": True,
               "source_manifest": str(source_manifest)}
    write_json(out / "cohort_audit.json", summary)
    if not ok:
        reason = (f"Unresolved events={len(unresolved)}. " if unresolved else "")
        if not uniform_config:
            reason += "Retained preprocessing configurations are missing or nonuniform. "
        if not nonempty:
            reason += "A train/validation/test subset is empty. "
        raise RuntimeError(reason + f"Inspect {out}; training has NOT started. "
                           "Repair QC or supply a reviewed event_id,reason exclusions CSV.")
    return {"manifest": str(accepted_path.resolve()), "frame": accepted,
            "fingerprints": fingerprints, "content_records": content_records,
            "summary": summary}


class SnapshotDataset(original45.StrongBaselineDataset):
    """Same sampling/metadata/labels as source45 with source63's locked seeds.

    Neural input is read exclusively from the separately validated prefix sidecar.
    The observed PGA/PGV propagation-baseline fields are deliberately absent, so
    the old propagation suite cannot accidentally use full-record input velocity.
    """
    def __init__(self, *args, prefix_root: str | Path,
                 event_fingerprints: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_root = Path(prefix_root)
        self.event_fingerprints = event_fingerprints
        missing = set(self.frame["event_id"]) - set(event_fingerprints)
        if missing:
            raise ValueError(f"Events did not pass preflight: {sorted(missing)[:10]}")

    def draw(self, index: int) -> dict[str, Any]:
        if self.training:
            event_index, repeat_index = int(index), 0
            suffix = f"epoch:{self.epoch}"
        else:
            event_index, repeat_index = divmod(int(index), self.repeats)
            suffix = f"repeat:{repeat_index}"
        row = self.frame.iloc[event_index]
        eid = str(row["event_id"])
        path = original45.resolve_h5_path(eid, str(row["h5_path"]), self.h5_root).resolve()
        side = (self.prefix_root / f"{eid}.h5").resolve()
        checked = self.event_fingerprints[eid]
        assert_file_unchanged(path, checked["source"])
        assert_file_unchanged(side, checked["prefix"])
        with h5py.File(path, "r") as h:
            coordinates = np.asarray(h["station_coords"][:], dtype=np.float32)
            p = np.asarray(h["p_offset_sec"][:], dtype=np.float32)
            fs = float(h.attrs["sampling_rate_hz"])
            pre = float(h.attrs["pre_first_p_sec"])
            zero = int(h.attrs["time_zero_index"])
            total_samples = h["acceleration"].shape[-1]
        candidates = np.flatnonzero(np.isfinite(p) & (p >= -1e-3) & (p <= self.t0_sec))
        rng = np.random.default_rng(paired_draw_seed(
            f"strong:{self.split_name}:{eid}:{suffix}", self.seed))
        inputs = rng.choice(candidates, size=self.input_stations, replace=False)
        pool = np.setdiff1d(np.arange(len(coordinates)), inputs, assume_unique=False)
        targets = rng.choice(pool, size=self.target_stations, replace=False)
        start = max(0, int(round((pre - self.input_pre_sec) * fs)))
        stop = min(zero + int(round(self.t0_sec * fs)), total_samples - 1)
        return {"event_id": eid, "repeat": repeat_index, "source": path, "prefix": side,
                "input_indices": inputs, "target_indices": targets, "coords": coordinates,
                "p_offset": p, "fs": fs, "start": start, "stop": stop}

    def __getitem__(self, index: int) -> dict[str, Any]:
        d = self.draw(index)
        inputs, targets = d["input_indices"], d["target_indices"]
        with h5py.File(d["source"], "r") as old:
            # SHA256 was verified for ALL retained sources in this run's preflight;
            # stat guards above reject modification during training/evaluation.
            waves = load_snapshot_input(old, d["prefix"], inputs, t0_sec=self.t0_sec,
                                        input_pre_sec=self.input_pre_sec,
                                        apply_log_transform=True, verify_source_hash=False)
            # No full-record signal goes to neural input. Future slices are labels only.
            acc = np.stack([old["acceleration"][int(j), :, d["stop"]:] for j in targets]).astype(np.float32)
            vel = np.stack([old["velocity"][int(j), :, d["stop"]:] for j in targets]).astype(np.float32)
        if waves.shape != (self.input_stations, 3, d["stop"] - d["start"]):
            raise ValueError("Prefix input length differs from the original input window.")
        coords_in, coords_tar = d["coords"][inputs], d["coords"][targets]
        lat, lon = float(coords_in[:, 0].mean()), float(coords_in[:, 1].mean())
        xy_in = original45.local_xy_km(coords_in, lat, lon)
        xy_tar = original45.local_xy_km(coords_tar, lat, lon)
        inputs_f = np.concatenate([xy_in / 100.0, coords_in[:, 2:3] / 2000.0,
                                   d["p_offset"][inputs, None] / max(float(self.t0_sec), 1.0)], axis=1).astype(np.float32)
        targets_f = np.concatenate([xy_tar / 100.0, coords_tar[:, 2:3] / 2000.0], axis=1).astype(np.float32)
        target_log = np.stack([np.log10(np.maximum(original45.horizontal_peak(acc), 1e-10)),
                               np.log10(np.maximum(original45.horizontal_peak(vel), 1e-12))], axis=-1).astype(np.float32)
        if not np.isfinite(target_log).all():
            raise ValueError(f"Nonfinite archived target labels: {d['event_id']}")
        return {"event_id": d["event_id"], "repeat": torch.tensor(d["repeat"], dtype=torch.int64),
                "target_slot": torch.arange(self.target_stations, dtype=torch.int64),
                "target_station_index": torch.from_numpy(targets.astype(np.int64)),
                "input_station_index": torch.from_numpy(inputs.astype(np.int64)),
                "input_waveforms": torch.from_numpy(waves),
                "input_features": torch.from_numpy(inputs_f),
                "target_features": torch.from_numpy(targets_f),
                "target_log": torch.from_numpy(target_log),
                "distances_km": torch.from_numpy(original45.pairwise_distance_km(xy_tar, xy_in).astype(np.float32))}


StrongBaselineDataset = SnapshotDataset
