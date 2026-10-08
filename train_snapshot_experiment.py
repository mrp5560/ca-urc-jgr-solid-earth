#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One minimal supplementary experiment: prefix Base -> frozen Base + CA-URC.

Uses the uploaded 45/63 model, losses, and training functions unchanged.
Default stage is audit: it never starts training. The old archive supplies test
keys/labels for pairing, not the new Base predictions. Source scripts are
included in this folder and imported as libraries, NOT run through their mains.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import source45_baselines as base_code
import source63_risk as head_code
import snapshot_experiment_data as data_code
from snapshot_experiment_data import preflight_cohort, write_json, sha256_file

KEYS = ["event_id", "repeat", "target_slot", "target_station_index"]
TRUTH = ["true_log10_pga", "true_log10_pgv"]


def read_reference(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"event_id": str})
    missing = set(KEYS + TRUTH) - set(frame.columns)
    if missing:
        raise ValueError(f"Reference CSV missing key/truth columns: {sorted(missing)}")
    frame["event_id"] = frame["event_id"].str.strip()
    for col in KEYS[1:]:
        v = pd.to_numeric(frame[col], errors="raise")
        if v.isna().any() or not np.equal(v, np.floor(v)).all():
            raise ValueError(f"Invalid integer reference key: {col}")
        frame[col] = v.astype(np.int64)
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Missing/duplicate reference keys.")
    if frame.duplicated(["event_id", "repeat", "target_station_index"]).any():
        raise ValueError("Reference contains repeated target stations in the same draw.")
    if not np.isfinite(frame[TRUTH].to_numpy(float)).all():
        raise ValueError("Nonfinite reference truth values.")
    return frame


def align_reference(current: pd.DataFrame, reference: pd.DataFrame,
                    test_events: set[str], thresholds: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Align by keys, NOT CSV row position. Audit labels and optional tail flags."""
    reference = reference.loc[reference["event_id"].isin(test_events)].copy()
    if set(reference["event_id"]) != test_events:
        raise ValueError("Reference does not cover every retained test event.")
    if current.duplicated(KEYS).any():
        raise ValueError("New predictions contain duplicate test keys.")
    current_index = pd.MultiIndex.from_frame(current[KEYS])
    ref_index = pd.MultiIndex.from_frame(reference[KEYS])
    missing = current_index.difference(ref_index)
    extra = ref_index.difference(current_index)
    if len(missing) or len(extra) or len(current) != len(reference):
        raise RuntimeError(f"Locked station-draw mismatch: {len(missing)} missing, "
                           f"{len(extra)} extra reference keys. Do NOT change the seed "
                           "or resample to bypass this check.")
    aligned = reference.set_index(KEYS).reindex(current_index).reset_index()
    delta = np.abs(current[TRUTH].to_numpy(float) - aligned[TRUTH].to_numpy(float))
    max_diff = float(delta.max()) if delta.size else float("nan")
    if not np.isfinite(max_diff) or max_diff > 1e-5:
        raise RuntimeError(f"Archived-label mismatch: max difference {max_diff}.")
    tail_checks = {}
    for j, q in enumerate(("pga", "pgv")):
        if f"is_tail_{q}" in aligned:
            observed = head_code.bool_array(aligned[f"is_tail_{q}"])
            computed = current[TRUTH[j]].to_numpy(float) >= float(thresholds[j])
            tail_checks[q] = bool(np.array_equal(observed, computed))
            if not tail_checks[q]:
                raise RuntimeError(f"Tail flags and original threshold JSON disagree for {q}.")
    return aligned, {"current_rows": len(current), "reference_rows_same_test_cohort": len(aligned),
                     "test_events": len(test_events), "exact_key_pairing": True,
                     "max_abs_truth_difference": max_diff, "tail_flags_match": tail_checks,
                     "reference_rows_reordered_by_keys": True,
                     "input_indices_verified_against_archive": "not stored in this reference; canonical seed plus exact target draw are checked"}


def test_pairing_preflight(dataset, reference, thresholds, out: Path):
    """No model inference or performance calculation; test reads are label/key QA."""
    rows, draws = [], []
    for i in range(len(dataset)):
        batch = dataset[i]
        eid, repeat = batch["event_id"], int(batch["repeat"])
        truth = batch["target_log"].numpy()
        ids = batch["target_station_index"].numpy()
        draws.append({"event_id": eid, "repeat": repeat,
                      "input_station_indices": "|".join(map(str, batch["input_station_index"].tolist())),
                      "target_station_indices": "|".join(map(str, ids.tolist()))})
        for slot, station in enumerate(ids):
            rows.append({"event_id": eid, "repeat": repeat, "target_slot": slot,
                         "target_station_index": int(station),
                         "true_log10_pga": float(truth[slot, 0]),
                         "true_log10_pgv": float(truth[slot, 1])})
    frame = pd.DataFrame(rows)
    aligned, audit = align_reference(frame, reference, set(dataset.frame["event_id"]), thresholds)
    frame.to_csv(out / "checked_test_keys_and_truth.csv", index=False)
    pd.DataFrame(draws).to_csv(out / "checked_test_station_draws.csv", index=False)
    write_json(out / "locked_draw_audit.json", audit)
    return aligned, audit


def make_loader(dataset, args, *, shuffle=False, seed=None):
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(args.seed if seed is None else seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0,
                      pin_memory=str(args.device_resolved).startswith("cuda"))


def source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: sha256_file(here / name) for name in
            ("source45_baselines.py", "source63_risk.py", "snapshot_input_reader.py",
             "snapshot_experiment_data.py", "train_snapshot_experiment.py")}


def make_protocol(args, cohort, thresholds):
    keys = ("split_column", "t0_sec", "input_stations", "target_stations", "input_pre_sec",
            "seed", "head_seed", "hidden_dim", "attention_heads", "risk_hidden",
            "maximum_correction", "batch_size", "base_epochs", "base_learning_rate",
            "base_weight_decay", "lr_patience", "minimum_learning_rate",
            "early_stopping_patience", "minimum_delta", "validation_repeats",
            "test_repeats", "head_epochs", "head_learning_rate", "head_weight_decay",
            "training_validation_repeats", "selection_validation_repeats", "prevalence_repeats",
            "lambda_tail_classification", "lambda_under_classification", "lambda_regression",
            "risk_regression_weight", "lambda_directional", "lambda_leakage",
            "max_positive_weight", "overall_budget", "non_tail_budget", "bias_limit", "powers")
    body = {"input_policy": "prefix-only acceleration; original labels",
            "draw_seed_policy": "source63 canonical text:seed; train/validation strip strong:, test uses locked:test:",
            "settings": {k: getattr(args, k) for k in keys},
            "event_order_and_content": cohort["content_records"],
            "thresholds": thresholds.tolist(),
            "source_hashes": source_hashes(),
            "exclusions": cohort["summary"]["reviewed_exclusions"]}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"protocol_id": hashlib.sha256(encoded).hexdigest(), **body}


def verify_protocol(checkpoint: dict, protocol: dict) -> None:
    if checkpoint.get("snapshot_protocol_id") != protocol["protocol_id"]:
        raise RuntimeError("Checkpoint does not belong to this prefix-input/data/seed/configuration "
                           "protocol. Old checkpoints are not silently accepted. Use a separate output "
                           "directory or restore the exact saved settings.")


def run_base(args, common, protocol, device):
    root = Path(args.out_dir) / "base"
    checkpoint_path = root / "cross_attention" / "best_model.pt"
    marker_path = root / "base_complete.json"
    if marker_path.exists():
        checkpoint = head_code.load_checkpoint(checkpoint_path, device)
        verify_protocol(checkpoint, protocol)
        print(f"Base already completed under matching protocol: {checkpoint_path}")
        return checkpoint_path
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Partial/unverified base outputs exist in {root}. "
                           "Use a new --out-dir; existing results will not be overwritten.")
    train = data_code.StrongBaselineDataset(split_name="train", training=True, repeats=1, **common)
    val = data_code.StrongBaselineDataset(split_name="validation", training=False,
                                         repeats=args.validation_repeats, **common)
    b = copy.copy(args)
    b.epochs, b.learning_rate, b.weight_decay = args.base_epochs, args.base_learning_rate, args.base_weight_decay
    b.out_dir = str(root)
    b.graph_initial_length_scale_km = 50.0  # unused by cross_attention; original interface
    b.snapshot_protocol_id = protocol["protocol_id"]
    # Exact original 45 model/optimizer/loss/checkpoint-selection function.
    model, summary = base_code.train_learned_variant(
        "cross_attention", b, make_loader(train, args, shuffle=True), train,
        make_loader(val, args), device)
    checkpoint = head_code.load_checkpoint(checkpoint_path, device)
    checkpoint["snapshot_protocol_id"] = protocol["protocol_id"]
    checkpoint["input_policy"] = protocol["input_policy"]
    checkpoint["draw_seed_policy"] = protocol["draw_seed_policy"]
    torch.save(checkpoint, checkpoint_path)
    write_json(marker_path, {"snapshot_protocol_id": protocol["protocol_id"],
                            "checkpoint": str(checkpoint_path.resolve()), **summary})
    del model
    return checkpoint_path


def load_new_base(args, protocol, device):
    path = Path(args.out_dir) / "base" / "cross_attention" / "best_model.pt"
    if not (path.parent.parent / "base_complete.json").is_file():
        raise RuntimeError("Run --stage base (or all) first; no completed prefix Base is available.")
    checkpoint = head_code.load_checkpoint(path, device)
    verify_protocol(checkpoint, protocol)
    if checkpoint.get("variant") != "cross_attention":
        raise ValueError("Expected Cross-Attention Base, not mean-pooling 'base'.")
    return checkpoint


def compute_prevalence(args, common, checkpoint, thresholds, device):
    base = head_code.make_frozen_base(data_code, checkpoint, args.hidden_dim,
                                     args.attention_heads, device)
    ds = data_code.StrongBaselineDataset(split_name="train", training=False,
                                        repeats=args.prevalence_repeats, **common)
    prevalence = head_code.estimate_training_prevalence(base, make_loader(ds, args), thresholds, device)
    return base, prevalence


def run_head(args, common, protocol, thresholds, device):
    checkpoint = load_new_base(args, protocol, device)
    root = Path(args.out_dir) / "head"
    result_path = root / "head_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("snapshot_protocol_id") != protocol["protocol_id"]:
            raise RuntimeError("Head result protocol mismatch.")
        print(f"Head selection already completed: {result_path}")
        return result
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Partial head outputs exist in {root}; use a new output directory.")
    base, prevalence = compute_prevalence(args, common, checkpoint, thresholds, device)
    val = data_code.StrongBaselineDataset(split_name="validation", training=False,
                                         repeats=args.selection_validation_repeats, **common)
    selection_cache = head_code.build_eval_cache(base, make_loader(val, args), thresholds, device)
    base_metrics = head_code.metric_bundle(selection_cache, selection_cache["base"])
    weights = {k: torch.as_tensor(np.clip((1.0 - prevalence[k]) / np.maximum(prevalence[k], 1e-6),
                                         1.0, args.max_positive_weight),
                                 dtype=torch.float32, device=device) for k in ("tail", "under")}
    h = copy.copy(args)
    h.epochs, h.learning_rate, h.weight_decay = args.head_epochs, args.head_learning_rate, args.head_weight_decay
    h.train_label, h.validation_label, h.test_label = "train", "validation", "test"
    h.num_workers = 0
    h.out_dir = str(root)
    h.snapshot_protocol_id = protocol["protocol_id"]
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "training_prevalence.json", prevalence)
    # Exact original 63 model/loss/training/joint validation selection.
    result = head_code.run_variant(
        "A4_under_only", h, data_code, checkpoint, thresholds,
        torch.as_tensor(thresholds, dtype=torch.float32, device=device),
        prevalence, weights["tail"], weights["under"], common, selection_cache,
        base_metrics, head_code.parse_float_list(args.powers), device, args.head_seed, root)
    result["snapshot_protocol_id"] = protocol["protocol_id"]
    result["test_performance_used_for_selection"] = False
    write_json(result_path, result)
    return result


def metric_event_series(frame, pred_prefix, quantity, population, metric):
    truth = frame[f"true_log10_{quantity}"].to_numpy(float)
    pred = frame[f"{pred_prefix}_log10_{quantity}"].to_numpy(float)
    tail = frame[f"is_tail_{quantity}"].to_numpy(bool)
    mask = np.ones(len(frame), dtype=bool) if population == "overall" else (
        tail if population == "high_motion_tail" else ~tail)
    part = frame.loc[mask, ["event_id", "repeat"]].copy()
    if part.empty:
        return pd.Series(dtype=float), 0, 0
    part["value"] = base_code.element_metric(truth[mask], pred[mask], metric)
    er = part.groupby(["event_id", "repeat"], sort=False)["value"].mean()
    ev = er.groupby(level=0, sort=False).mean()
    return ev, len(part), len(er)


def bootstrap_weights(frame, event_order, group_column, draws, seed):
    """Hierarchical group -> event resampling; average events, not group means.

    Bootstrap population is the complete retained test-event set. Missing subset
    metrics stay NaN and are omitted within a replicate, never replaced by zero.
    """
    if not group_column:
        return None, None
    if group_column not in frame:
        raise ValueError(f"--group-column {group_column!r} absent. Available columns: {list(frame.columns)}")
    lookup = frame.set_index("event_id").reindex(event_order)[group_column]
    if lookup.isna().any() or lookup.astype(str).str.strip().eq("").any():
        raise ValueError("Missing group IDs in retained test cohort.")
    labels = lookup.astype(str).to_numpy()
    groups = list(dict.fromkeys(labels))
    if len(groups) < 2:
        raise ValueError("At least two actual grouping units needed for grouped uncertainty.")
    rng = np.random.default_rng(seed)
    ng = len(groups)
    group_counts = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=draws)
    weights = np.zeros((draws, len(event_order)), dtype=np.float64)
    for gi, group in enumerate(groups):
        indices = np.flatnonzero(labels == group)
        n = len(indices)
        for multiplicity in np.unique(group_counts[:, gi]):
            if not multiplicity:
                continue
            row_ids = np.flatnonzero(group_counts[:, gi] == multiplicity)
            counts = rng.multinomial(n * int(multiplicity), np.full(n, 1.0 / n), size=len(row_ids))
            weights[np.ix_(row_ids, indices)] = counts
    return weights, dict(zip(event_order, labels))


def summarize(frame, methods, cohort_frame, args, out):
    events = list(cohort_frame.loc[cohort_frame[args.split_column].eq("test"), "event_id"])
    weights, event_groups = bootstrap_weights(cohort_frame, events, args.group_column,
                                             args.bootstrap_replicates, args.seed + 77)
    rows, series = [], {}
    for q in ("pga", "pgv"):
        for pop in ("overall", "non_tail", "high_motion_tail"):
            for metric in ("mae", "bias", "factor2", "under05"):
                for method in methods:
                    ev, nrows, nreps = metric_event_series(frame, method, q, pop, metric)
                    series[(method, q, pop, metric)] = ev.reindex(events)
                    rows.append({"method": method, "quantity": q, "population": pop,
                                 "metric": metric, "value": float(ev.mean()) if len(ev) else np.nan,
                                 "rate_unit": "fraction" if metric in ("factor2", "under05") else "log10",
                                 "n_events": len(ev), "n_target_rows": nrows, "n_event_repeats": nreps,
                                 "n_groups": len({event_groups[e] for e in ev.index}) if event_groups else np.nan})
    pd.DataFrame(rows).to_csv(out / "paired_metrics.csv", index=False)
    contrasts = [("Prefix_Base", "Prefix_CAURC")]
    if "FullRecord_Base" in methods and "FullRecord_CAURC" in methods:
        contrasts.append(("FullRecord_Base", "FullRecord_CAURC"))
    delta_rows, event_rows = [], []
    for base_name, final_name in contrasts:
        for q in ("pga", "pgv"):
            for pop in ("overall", "non_tail", "high_motion_tail"):
                for metric in ("mae", "bias", "factor2", "under05"):
                    delta = (series[(final_name, q, pop, metric)] - series[(base_name, q, pop, metric)])
                    arr = delta.to_numpy(float)
                    valid = np.isfinite(arr)
                    lo = hi = np.nan
                    valid_draws = 0
                    if weights is not None and valid.any():
                        denom = weights @ valid.astype(float)
                        nums = weights @ np.nan_to_num(arr, nan=0.0)
                        samples = nums[denom > 0] / denom[denom > 0]
                        valid_draws = len(samples)
                        if valid_draws:
                            lo, hi = np.quantile(samples, [0.025, 0.975])
                    delta_rows.append({"reference": base_name, "candidate": final_name,
                                       "quantity": q, "population": pop, "metric": metric,
                                       "mean_delta": float(delta.mean()), "ci95_low": lo, "ci95_high": hi,
                                       "n_events": int(valid.sum()), "bootstrap_valid_replicates": valid_draws,
                                       "bootstrap_scope": "hierarchical_group_to_event" if weights is not None else "not_computed_no_group_column",
                                       "rate_unit": "fraction" if metric in ("factor2", "under05") else "log10"})
                    for eid, value in delta.dropna().items():
                        event_rows.append({"event_id": eid, "reference": base_name, "candidate": final_name,
                                           "quantity": q, "population": pop, "metric": metric,
                                           "delta_candidate_minus_reference": value})
    result = pd.DataFrame(delta_rows)
    result.to_csv(out / "paired_changes.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out / "event_level_paired_changes.csv", index=False)
    print("\nPaired tail MAE changes (CA-URC minus corresponding Base):")
    print(result.loc[result.population.eq("high_motion_tail") & result.metric.eq("mae")].to_string(index=False))
    if weights is None:
        print("NOTE: No grouping field supplied; no group-bootstrap confidence intervals were computed. "
              "Use --group-column with the ACTUAL original grouping column and a new analysis output.")
    return {"methods": methods, "test_events": len(events), "target_rows": len(frame),
            "group_column": args.group_column, "bootstrap_replicates": args.bootstrap_replicates if weights is not None else 0,
            "rates_are_fractions": True,
            "old_models_status": "Archived full-record model predictions reaggregated on retained test keys; they were not retrained on a reduced development cohort",
            "primary_comparison": "Prefix_CAURC - Prefix_Base; NOT old Base versus new CA-URC"}


def evaluate(args, common, cohort, protocol, thresholds, reference, device):
    checkpoint = load_new_base(args, protocol, device)
    result_path = Path(args.out_dir) / "head" / "head_result.json"
    if not result_path.is_file():
        raise RuntimeError("Run --stage head (or all) before evaluation.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("snapshot_protocol_id") != protocol["protocol_id"]:
        raise RuntimeError("Head selection protocol mismatch.")
    if not result.get("feasible"):
        raise RuntimeError("No feasible validation candidate. Do not relax guards using test results.")
    if result.get("selected_at_upper_gamma_boundary"):
        raise RuntimeError("Validation selected the upper gamma boundary. Following source63's policy, "
                           "test performance has not been evaluated. Review validation scan first; "
                           "do not adjust the search based on test results.")
    out = Path(args.out_dir) / args.analysis_subdir
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Evaluation output exists: {out}. Use a different --analysis-subdir "
                           "for a documented statistics-only rerun; old results are not overwritten.")
    base = head_code.make_frozen_base(data_code, checkpoint, args.hidden_dim, args.attention_heads, device)
    ds = data_code.StrongBaselineDataset(split_name="test", training=False,
                                        repeats=args.test_repeats, **common)
    cache = head_code.build_eval_cache(base, make_loader(ds, args), thresholds, device)
    frame = pd.DataFrame({k: cache[k] for k in KEYS})
    for j, q in enumerate(("pga", "pgv")):
        frame[f"true_log10_{q}"] = cache["truth"][:, j]
        frame[f"is_tail_{q}"] = cache["tail"][:, j]
    old_aligned, audit = align_reference(frame, reference, set(ds.frame["event_id"]), thresholds)
    prevalence = json.loads((Path(args.out_dir) / "head" / "training_prevalence.json").read_text(encoding="utf-8"))
    model = head_code.AblationRiskGate(base, "A4_under_only", args.hidden_dim,
                                     args.risk_hidden, args.maximum_correction,
                                     np.asarray(prevalence["tail"]), np.asarray(prevalence["under"])) .to(device)
    selected_path = Path(args.out_dir) / "head" / "A4_under_only" / "selected_head.pt"
    selected = head_code.load_checkpoint(selected_path, device)
    if selected.get("args", {}).get("snapshot_protocol_id") != protocol["protocol_id"]:
        raise RuntimeError("Selected head checkpoint provenance mismatch.")
    # Check that stored frozen Base equals the newly trained Base, not an old model.
    for k, v in checkpoint["model_state"].items():
        if not torch.equal(v, selected["model_state"]["base." + k]):
            raise RuntimeError(f"Frozen Base changed in correction training: {k}")
    model.load_state_dict(selected["model_state"], strict=True)
    head = head_code.predict_head_from_cache(model, cache["risk_feature"], device, args.head_eval_batch_size)
    prediction = head_code.candidate_prediction_numpy("A4_under_only", cache, head, float(result["selected_gamma"]))
    for j, q in enumerate(("pga", "pgv")):
        frame[f"Prefix_Base_log10_{q}"] = cache["base"][:, j]
        frame[f"Prefix_CAURC_log10_{q}"] = prediction[:, j]
        frame[f"underprediction_risk_score_{q}"] = head["under_probability"][:, j]
        frame[f"raw_correction_amplitude_{q}"] = head["residual"][:, j]
        frame[f"applied_correction_{q}"] = prediction[:, j] - cache["base"][:, j]
    methods = ["Prefix_Base", "Prefix_CAURC"]
    # Exact prefixes are required: 'final' in the old dual-risk file is not CA-URC!
    for old_prefix, name in ((args.old_base_prefix, "FullRecord_Base"),
                              (args.old_caurc_prefix, "FullRecord_CAURC")):
        if old_prefix and all(f"{old_prefix}_log10_{q}" in old_aligned for q in ("pga", "pgv")):
            for q in ("pga", "pgv"):
                frame[f"{name}_log10_{q}"] = old_aligned[f"{old_prefix}_log10_{q}"].to_numpy(float)
            methods.append(name)
        elif old_prefix:
            print(f"NOTE: No explicit {old_prefix!r} prediction columns in reference; "
                  f"{name} is omitted, not guessed from a generic 'final' column.")
    for method in methods:
        if not np.isfinite(frame[[f"{method}_log10_pga", f"{method}_log10_pgv"]].to_numpy(float)).all():
            raise ValueError(f"Nonfinite predictions in {method}; no row dropping is allowed.")
    draws = pd.read_csv(Path(args.out_dir) / "audit" / "checked_test_station_draws.csv", dtype={"event_id": str})
    frame = frame.merge(draws[["event_id", "repeat", "input_station_indices"]],
                        on=["event_id", "repeat"], how="left", validate="many_to_one")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "locked_prefix_test_predictions.csv", index=False)
    summary = summarize(frame, methods, cohort["frame"], args, out)
    summary.update({"snapshot_protocol_id": protocol["protocol_id"],
                    "selected_epoch": result["selected_epoch"], "selected_gamma": result["selected_gamma"],
                    "pairing_audit": audit, "frozen_base_unchanged": True,
                    "reference_csv_sha256": sha256_file(args.reference_predictions),
                    "old_base_column_prefix": args.old_base_prefix,
                    "old_caurc_column_prefix": args.old_caurc_prefix,
                    "scope": "Retrospective prefix-input supplementary experiment; archived P picks and offline labels retained"})
    write_json(out / "evaluation_audit.json", summary)
    print(f"\nEvaluation written to {out.resolve()}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["audit", "base", "head", "evaluate", "all"], default="audit")
    p.add_argument("--manifest", default="data/scedc/model_manifests/scenario_t0_5s_k5.csv")
    p.add_argument("--h5-root", default="data/scedc/processed_full_v4/events", help="ORIGINAL label HDF5 root")
    p.add_argument("--prefix-root", default="data/scedc/snapshot_available_t0_5s_k5/events")
    p.add_argument("--reference-predictions", default="runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv")
    p.add_argument("--threshold-json", default="runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json")
    p.add_argument("--out-dir", default="runs/snapshot_available_validation")
    p.add_argument("--split-column", default="split_grouped")
    p.add_argument("--group-column", default=None, help="ACTUAL original spatiotemporal group column; omit for point estimates only")
    p.add_argument("--excluded-events-csv", default=None, help="Reviewed explicit event_id,reason CSV; no automatic exclusions")
    p.add_argument("--expected-events", type=int, default=1620, help="Requested manifest count BEFORE any reviewed exclusions")
    p.add_argument("--old-base-prefix", default="A2_cross_attention_base")
    p.add_argument("--old-caurc-prefix", default="A4_under_only")
    p.add_argument("--analysis-subdir", default="evaluation")
    for name, default in (("t0-sec", 5), ("input-stations", 5), ("target-stations", 10),
                          ("validation-repeats", 3), ("test-repeats", 20),
                          ("hidden-dim", 128), ("attention-heads", 4), ("risk-hidden", 64),
                          ("batch-size", 8), ("base-epochs", 50), ("head-epochs", 40),
                          ("lr-patience", 4), ("early-stopping-patience", 12),
                          ("prevalence-repeats", 3), ("training-validation-repeats", 3),
                          ("selection-validation-repeats", 20), ("head-eval-batch-size", 8192),
                          ("seed", 20260713), ("num-workers", 0), ("bootstrap-replicates", 10000),
                          ("torch-threads", 0)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--head-seed", type=int, default=None,
                   help="Default seed+2000 matches A4's position in original full A3,A4,A5 run; verify against historical run args")
    for name, default in (("input-pre-sec", 2.0), ("maximum-correction", 1.5),
                          ("base-learning-rate", 1e-3), ("base-weight-decay", 1e-4),
                          ("head-learning-rate", 1e-3), ("head-weight-decay", 1e-4),
                          ("minimum-learning-rate", 1e-5), ("minimum-delta", 1e-4),
                          ("lambda-tail-classification", .25), ("lambda-under-classification", .25),
                          ("lambda-regression", 1.), ("risk-regression-weight", 2.),
                          ("lambda-directional", .50), ("lambda-leakage", .05),
                          ("max-positive-weight", 9.), ("overall-budget", .005),
                          ("non-tail-budget", .003), ("bias-limit", .05)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--powers", default="1,1.5,2,2.5,3,4,5,6,7,8,10,12,15,20")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def main():
    args = build_parser().parse_args()
    if args.num_workers != 0:
        raise ValueError("This minimal adapter requires --num-workers 0. This preserves epoch-updated "
                         "station draws and avoids patched seed state in persistent/spawned workers.")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.head_seed is None:
        args.head_seed = args.seed + 2000
    if args.hidden_dim % args.attention_heads:
        raise ValueError("hidden-dim must be divisible by attention-heads")
    if min(args.t0_sec, args.input_stations, args.target_stations, args.base_epochs,
           args.head_epochs, args.test_repeats, args.bootstrap_replicates) <= 0:
        raise ValueError("Time, station counts, epochs, repeats and bootstrap count must be positive")
    if Path(args.analysis_subdir).is_absolute() or ".." in Path(args.analysis_subdir).parts:
        raise ValueError("analysis-subdir must be a relative folder under out-dir")
    args.device_resolved = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device_resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device_resolved)
    threshold_path = Path(args.threshold_json)
    threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    thresholds = np.asarray([threshold_data["log10_pga_threshold"], threshold_data["log10_pgv_threshold"]], dtype=np.float32)
    if not np.isfinite(thresholds).all():
        raise ValueError("Nonfinite original thresholds")
    reference = read_reference(args.reference_predictions)
    cohort = preflight_cohort(args)  # raises BEFORE training on any unreviewed incompleteness
    if args.group_column:
        if args.group_column == args.split_column:
            raise ValueError("The train/validation/test split column is NOT a spatiotemporal group ID.")
        if args.group_column not in cohort["frame"]:
            raise ValueError(f"Group column {args.group_column!r} is absent. Available columns: "
                             f"{list(cohort['frame'].columns)}")
        test_groups = cohort["frame"].loc[
            cohort["frame"][args.split_column].eq("test"), args.group_column]
        if test_groups.isna().any() or test_groups.astype(str).str.strip().eq("").any():
            raise ValueError("Retained test events have missing original group identifiers.")
        if test_groups.nunique() < 2:
            raise ValueError("Fewer than two original test groups; grouped confidence intervals cannot be computed.")
    common = {"manifest": cohort["manifest"], "h5_root": args.h5_root,
              "split_column": args.split_column, "t0_sec": args.t0_sec,
              "input_stations": args.input_stations, "target_stations": args.target_stations,
              "input_pre_sec": args.input_pre_sec, "seed": args.seed,
              "prefix_root": str(Path(args.prefix_root).resolve()),
              "event_fingerprints": cohort["fingerprints"]}
    test_ds = data_code.StrongBaselineDataset(split_name="test", training=False,
                                             repeats=args.test_repeats, **common)
    _, draw_audit = test_pairing_preflight(test_ds, reference, thresholds, Path(args.out_dir) / "audit")
    protocol = make_protocol(args, cohort, thresholds)
    protocol_path = Path(args.out_dir) / "experiment_protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old["protocol_id"] != protocol["protocol_id"]:
            raise RuntimeError("Existing experiment protocol differs. Use a new --out-dir rather "
                               "than mixing cohorts, source files, seeds or model settings.")
    else:
        write_json(protocol_path, protocol)
    print("\nPrefix input and locked-target pairing preflight PASS.")
    print(pd.DataFrame(cohort["summary"]["counts_by_split"]).to_string(index=False))
    actual_shape = tuple(test_ds[0]["input_waveforms"].shape)
    print(f"Locked paired rows: {draw_audit['current_rows']}; actual input shape: {actual_shape}")
    print("Model: cross_attention (NOT the mean-pooling variant named base).")
    if not args.group_column:
        print("No group-bootstrap field supplied. Manifest columns:", ", ".join(cohort["frame"].columns))
    if args.stage == "audit":
        print("Audit only. No model training or performance evaluation was started.")
        return
    if args.stage in ("base", "all"):
        run_base(args, common, protocol, device)
    if args.stage in ("head", "all"):
        result = run_head(args, common, protocol, thresholds, device)
        if not result.get("feasible"):
            raise RuntimeError("Head has no feasible validation candidate; test performance was not evaluated.")
        if result.get("selected_at_upper_gamma_boundary"):
            raise RuntimeError("Head gamma is at the validation-grid boundary; test performance was not evaluated.")
    if args.stage in ("evaluate", "all"):
        evaluate(args, common, cohort, protocol, thresholds, reference, device)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        sys.exit(2)
