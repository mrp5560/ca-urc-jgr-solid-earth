#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
81_train_evaluate_caurc_observation_budget_grid.py

Observation-budget robustness experiment for Causal-SeisField / CA-URC.

Scientific question
-------------------
Does the final architecture remain useful when both the causal observation
snapshot t0 and the number K of P-wave-reached input stations change?

Minimal main-text grid:
    (t0, K) = (3 s, 3), (5 s, 5), (10 s, 5), (10 s, 10)

Full supplementary grid:
    t0 in {3, 5, 7, 10} s
    K  in {3, 5, 10}

Protocol
--------
For EACH budget separately:
  1. Recompute scenario eligibility from HDF5 using only information available
     at the snapshot:
         finite p_offset, p_offset >= -1e-3, p_offset <= t0,
         at least K input candidates,
         at least Q=10 remaining target stations.
  2. Keep the pre-existing grouped event split; do NOT repartition events.
  3. Recompute the TRAIN-only Q90 high-motion thresholds for that budget.
  4. Retrain a fresh Cross-Attention Base using TRAIN/VALIDATION only.
  5. Freeze the Base and retrain ONLY the A4 underprediction-risk correction
     head using the grouped CA-URC protocol.
  6. Select head epoch + gamma on VALIDATION only with the original grouped
     preservation guards; TEST is not touched during selection.
  7. After locking the candidate, evaluate on:
       a) the native eligible test set for that budget;
       b) a common test-event core eligible under ALL requested budgets.
  8. On the common core, compute the same canonical metrics:
         targets -> repeats -> events
     and hierarchical sequence -> event paired bootstrap for CA-URC vs Base.

Important
---------
The common test EVENT set is fixed across requested budgets. Target stations do
not need to be identical across budgets because changing K/t0 changes the input
set and therefore the admissible held-out pool. Cross-budget scientific claims
should use the common-event metrics, not the native-event metrics.

The full 12-cell grid includes the very restrictive (3 s, 10) budget. It may
leave only a small common test core. Run --dry-run first. By default this script
stops before expensive training when the common test core has fewer than
--min-common-test-events events.

Required local modules
----------------------
45_phase2_strong_baseline_suite.py
63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py
62_sequence_aware_paired_bootstrap_cadrg.py

Outputs
-------
<out-dir>/
  budget_eligibility_summary.csv
  all_events_with_budget_flags.csv
  common_test_event_ids.csv
  observation_budget_summary.csv
  observation_budget_long_metrics.csv
  protocol.json
  t0_<T>s_k<K>/
      scenario_native.csv
      scenario_common_test.csv
      tail_thresholds_q0.90.json
      cross_attention/best_model.pt
      A4_under_only/selected_head.pt
      A4_under_only/selected_epoch_gamma.json
      native_test_metrics.csv
      common_test_metrics.csv
      common_test_predictions.csv
      event_level_canonical_metrics.csv
      hierarchical_bootstrap_summary.csv
      sequence_cluster_bootstrap_summary.csv
      scenario_summary.json

This script intentionally does not reuse the locked t0=5,K=5 model: the
observation-budget experiment asks whether a model retrained under each
information budget retains the same qualitative benefit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


MINIMAL_BUDGETS = [(3, 3), (5, 5), (10, 5), (10, 10)]
FULL_BUDGETS = [
    (t0, k)
    for t0 in (3, 5, 7, 10)
    for k in (3, 5, 10)
]

EXPECTED_MAIN_55 = {
    "train": 1189,
    "validation": 207,
    "test": 224,
}
EXPECTED_MAIN_55_THRESHOLDS = {
    "pga": -2.011172,
    "pgv": -3.307250,
}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def load_module(path: str | Path, name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"}).to_numpy(dtype=bool)


def parse_budgets(text: str) -> list[tuple[int, int]]:
    text = str(text).strip().lower()
    if text == "minimal":
        return list(MINIMAL_BUDGETS)
    if text == "full":
        return list(FULL_BUDGETS)

    output: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Custom --budgets must be comma-separated t0:K pairs, "
                f"got {token!r}."
            )
        t0_text, k_text = token.split(":", 1)
        t0 = int(t0_text)
        k = int(k_text)
        if t0 <= 0 or k <= 0:
            raise ValueError("t0 and K must be positive.")
        output.append((t0, k))

    if not output:
        raise ValueError("No observation budgets were supplied.")

    # Stable unique order.
    seen = set()
    unique = []
    for item in output:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def budget_name(t0: int, k: int) -> str:
    return f"t0_{int(t0)}s_k{int(k)}"


def eligible_column(t0: int, k: int) -> str:
    return f"eligible_t0_{int(t0)}s_k{int(k)}"


def resolve_auto_path(explicit: str, candidates: list[str], label: str) -> Path:
    if explicit and str(explicit).strip().lower() != "auto":
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not auto-resolve {label}. Tried: {candidates}. "
        f"Supply it explicitly."
    )


def resolve_event_h5(
    event_id: str,
    row: pd.Series,
    h5_root: Path,
) -> Path:
    raw = str(row.get("h5_path", "")).strip()
    if raw and raw.lower() not in {"nan", "none"}:
        candidate = Path(raw)
        if candidate.exists():
            return candidate

    for candidate in (
        h5_root / f"{event_id}.h5",
        h5_root / str(event_id) / f"{event_id}.h5",
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve HDF5 for event {event_id}; "
        f"manifest h5_path={raw!r}; h5_root={h5_root}"
    )


def choose_device(text: str) -> torch.device:
    text = str(text).lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(text)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


# -----------------------------------------------------------------------------
# Eligibility and common-test construction
# -----------------------------------------------------------------------------


def inspect_budget_eligibility(
    master: pd.DataFrame,
    h5_root: Path,
    budgets: list[tuple[int, int]],
    target_stations: int,
) -> pd.DataFrame:
    frame = master.copy()

    unique_t0 = sorted({int(t0) for t0, _ in budgets})

    for t0 in unique_t0:
        frame[f"n_triggered_t0_{t0}s"] = 0
    for t0, k in budgets:
        frame[f"n_target_pool_t0_{t0}s_k{k}"] = 0
        frame[eligible_column(t0, k)] = False

    frame["h5_resolved"] = ""
    frame["h5_ok_budget_audit"] = False
    frame["budget_audit_error"] = ""
    frame["n_stations_h5_budget_audit"] = 0

    total = len(frame)
    for idx, (row_index, row) in enumerate(frame.iterrows(), start=1):
        event_id = str(row["event_id"]).strip()
        try:
            h5_path = resolve_event_h5(event_id, row, h5_root)
            with h5py.File(h5_path, "r") as h5:
                p_offset = np.asarray(h5["p_offset_sec"][:], dtype=np.float32)
                n_stations = int(len(p_offset))

            frame.at[row_index, "h5_resolved"] = str(h5_path.resolve())
            frame.at[row_index, "h5_ok_budget_audit"] = True
            frame.at[row_index, "n_stations_h5_budget_audit"] = n_stations

            triggered_counts = {}
            for t0 in unique_t0:
                n_triggered = int(
                    np.sum(
                        np.isfinite(p_offset)
                        & (p_offset >= -1e-3)
                        & (p_offset <= float(t0))
                    )
                )
                triggered_counts[t0] = n_triggered
                frame.at[row_index, f"n_triggered_t0_{t0}s"] = n_triggered

            for t0, k in budgets:
                n_target_pool = max(0, n_stations - int(k))
                ok = (
                    triggered_counts[int(t0)] >= int(k)
                    and n_target_pool >= int(target_stations)
                )
                frame.at[
                    row_index,
                    f"n_target_pool_t0_{t0}s_k{k}",
                ] = n_target_pool
                frame.at[row_index, eligible_column(t0, k)] = bool(ok)

        except Exception as exc:
            frame.at[row_index, "budget_audit_error"] = repr(exc)

        if idx % 250 == 0 or idx == total:
            print(f"Eligibility audit: {idx:,}/{total:,}")

    return frame


def eligibility_summary(
    frame: pd.DataFrame,
    budgets: list[tuple[int, int]],
    split_column: str,
    train_label: str,
    validation_label: str,
    test_label: str,
) -> pd.DataFrame:
    rows = []
    for t0, k in budgets:
        col = eligible_column(t0, k)
        mask = parse_bool(frame[col])
        for split_name, split_label in (
            ("train", train_label),
            ("validation", validation_label),
            ("test", test_label),
        ):
            split_mask = frame[split_column].astype(str).eq(str(split_label)).to_numpy()
            sub = frame.loc[mask & split_mask]
            rows.append(
                {
                    "t0_sec": int(t0),
                    "input_stations": int(k),
                    "budget": budget_name(t0, k),
                    "split": split_name,
                    "n_eligible_events": int(sub["event_id"].nunique()),
                    "n_sequence_groups": int(
                        sub["sequence_group"].astype(str).nunique()
                        if "sequence_group" in sub.columns
                        else 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def common_test_ids(
    frame: pd.DataFrame,
    budgets: list[tuple[int, int]],
    split_column: str,
    test_label: str,
) -> list[str]:
    test = frame.loc[
        frame[split_column].astype(str).eq(str(test_label))
    ].copy()

    if test.empty:
        raise RuntimeError("Test split is empty in the master manifest.")

    mask = np.ones(len(test), dtype=bool)
    for t0, k in budgets:
        mask &= parse_bool(test[eligible_column(t0, k)])

    ids = (
        test.loc[mask, "event_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return ids


def write_budget_manifests(
    audited: pd.DataFrame,
    t0: int,
    k: int,
    common_ids: set[str],
    split_column: str,
    train_label: str,
    validation_label: str,
    test_label: str,
    scenario_dir: Path,
) -> tuple[Path, Path, dict[str, int]]:
    col = eligible_column(t0, k)
    eligible = parse_bool(audited[col])
    native = audited.loc[eligible].copy()
    native["h5_path_original_budget_audit"] = native.get("h5_path", "")
    native["h5_path"] = native["h5_resolved"]

    common = native.loc[
        ~native[split_column].astype(str).eq(str(test_label))
        | native["event_id"].astype(str).isin(common_ids)
    ].copy()

    native_path = scenario_dir / "scenario_native.csv"
    common_path = scenario_dir / "scenario_common_test.csv"
    native.to_csv(native_path, index=False)
    common.to_csv(common_path, index=False)

    counts = {
        "native_train": int(
            native.loc[native[split_column].astype(str).eq(str(train_label)), "event_id"].nunique()
        ),
        "native_validation": int(
            native.loc[native[split_column].astype(str).eq(str(validation_label)), "event_id"].nunique()
        ),
        "native_test": int(
            native.loc[native[split_column].astype(str).eq(str(test_label)), "event_id"].nunique()
        ),
        "common_test": int(
            common.loc[common[split_column].astype(str).eq(str(test_label)), "event_id"].nunique()
        ),
    }
    return native_path, common_path, counts


# -----------------------------------------------------------------------------
# TRAIN-only high-motion thresholds
# -----------------------------------------------------------------------------


def horizontal_peak(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return np.max(
        np.sqrt(array[:, 0, :] ** 2 + array[:, 1, :] ** 2),
        axis=1,
    )


def compute_train_only_thresholds(
    manifest_path: Path,
    split_column: str,
    train_label: str,
    t0_sec: int,
    quantile: float,
    output_path: Path,
) -> dict[str, Any]:
    """
    Match the established grouped threshold definition:
    TRAIN events only, all stations in every eligible TRAIN event, and
    remaining horizontal PGA/PGV after the causal snapshot.
    """
    if output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            int(cached.get("t0_sec", -1)) == int(t0_sec)
            and math.isclose(
                float(cached.get("quantile", -1.0)),
                float(quantile),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and str(cached.get("train_label", "")) == str(train_label)
        ):
            print(f"Reuse TRAIN-only thresholds: {output_path.resolve()}")
            return cached

    frame = pd.read_csv(manifest_path, dtype={"event_id": str})
    train = frame.loc[
        frame[split_column].astype(str).eq(str(train_label))
    ].copy()
    if train.empty:
        raise RuntimeError("Training split is empty for threshold computation.")

    pga_parts = []
    pgv_parts = []
    for idx, row in enumerate(train.itertuples(index=False), start=1):
        with h5py.File(str(row.h5_path), "r") as h5:
            acceleration = np.asarray(h5["acceleration"][:], dtype=np.float32)
            velocity = np.asarray(h5["velocity"][:], dtype=np.float32)
            sampling_rate = float(h5.attrs["sampling_rate_hz"])
            time_zero_index = int(h5.attrs["time_zero_index"])

        snapshot = min(
            time_zero_index + int(round(float(t0_sec) * sampling_rate)),
            acceleration.shape[-1] - 1,
        )
        pga = horizontal_peak(acceleration[:, :, snapshot:])
        pgv = horizontal_peak(velocity[:, :, snapshot:])
        pga_parts.append(np.log10(np.maximum(pga, 1e-10)))
        pgv_parts.append(np.log10(np.maximum(pgv, 1e-12)))

        if idx % 250 == 0 or idx == len(train):
            print(f"Threshold audit: {idx:,}/{len(train):,} TRAIN events")

    pga_values = np.concatenate(pga_parts)
    pgv_values = np.concatenate(pgv_parts)
    result = {
        "split_column": str(split_column),
        "train_label": str(train_label),
        "t0_sec": int(t0_sec),
        "quantile": float(quantile),
        "definition": (
            "TRAIN-only quantile over all stations in budget-eligible training "
            "events; remaining horizontal peak after snapshot"
        ),
        "log10_pga_threshold": float(np.quantile(pga_values, quantile)),
        "log10_pgv_threshold": float(np.quantile(pgv_values, quantile)),
        "n_training_station_targets": int(len(pga_values)),
        "n_training_events": int(train["event_id"].astype(str).nunique()),
        "test_accessed": False,
    }
    output_path.write_text(
        json.dumps(json_safe(result), indent=2),
        encoding="utf-8",
    )
    return result


# -----------------------------------------------------------------------------
# Base and CA-URC training
# -----------------------------------------------------------------------------


def loader_kwargs(args, device: torch.device) -> dict[str, Any]:
    return {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(device.type == "cuda"),
        "persistent_workers": bool(args.num_workers > 0),
    }


def train_cross_attention_base(
    baseline_module,
    ablation_module,
    manifest_path: Path,
    scenario_dir: Path,
    t0: int,
    k: int,
    args,
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    # Match the final grouped train/validation draw convention used by script 59/63.
    ablation_module.patch_train_validation_seed_protocol(baseline_module)

    common_dataset = dict(
        manifest=str(manifest_path),
        h5_root=str(args.h5_root_resolved),
        split_column=args.split_column,
        t0_sec=int(t0),
        input_stations=int(k),
        target_stations=int(args.target_stations),
        input_pre_sec=float(args.input_pre_sec),
        seed=int(args.seed),
    )

    train_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.train_label,
        training=True,
        repeats=1,
        **common_dataset,
    )
    validation_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.validation_label,
        training=False,
        repeats=int(args.base_validation_repeats),
        **common_dataset,
    )

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))

    kw = loader_kwargs(args, device)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **kw,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **kw,
    )

    base_args = Namespace(
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        attention_heads=int(args.attention_heads),
        graph_initial_length_scale_km=float(args.graph_initial_length_scale_km),
        learning_rate=float(args.base_learning_rate),
        weight_decay=float(args.base_weight_decay),
        lr_patience=int(args.base_lr_patience),
        minimum_learning_rate=float(args.base_minimum_learning_rate),
        epochs=int(args.base_epochs),
        minimum_delta=float(args.base_minimum_delta),
        early_stopping_patience=int(args.base_early_stopping_patience),
        out_dir=str(scenario_dir),
    )

    _, summary = baseline_module.train_learned_variant(
        "cross_attention",
        base_args,
        train_loader,
        train_dataset,
        validation_loader,
        device,
    )

    checkpoint_path = scenario_dir / "cross_attention" / "best_model.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Base checkpoint was not created: {checkpoint_path}")

    return checkpoint_path, summary


def prepare_head_training(
    baseline_module,
    ablation_module,
    base_checkpoint_path: Path,
    native_manifest: Path,
    thresholds_np: np.ndarray,
    t0: int,
    k: int,
    args,
    device: torch.device,
):
    ablation_module.patch_train_validation_seed_protocol(baseline_module)

    base_checkpoint = ablation_module.load_checkpoint(
        base_checkpoint_path,
        device,
    )

    common_dataset = {
        "manifest": str(native_manifest),
        "h5_root": str(args.h5_root_resolved),
        "split_column": args.split_column,
        "t0_sec": int(t0),
        "input_stations": int(k),
        "target_stations": int(args.target_stations),
        "input_pre_sec": float(args.input_pre_sec),
        "seed": int(args.seed),
    }

    head_args = Namespace(
        train_label=args.train_label,
        validation_label=args.validation_label,
        test_label=args.test_label,
        hidden_dim=int(args.hidden_dim),
        attention_heads=int(args.attention_heads),
        risk_hidden=int(args.risk_hidden),
        maximum_correction=float(args.maximum_correction),
        training_validation_repeats=int(args.head_training_validation_repeats),
        selection_validation_repeats=int(args.head_selection_validation_repeats),
        prevalence_repeats=int(args.prevalence_repeats),
        test_repeats=int(args.test_repeats),
        batch_size=int(args.batch_size),
        head_eval_batch_size=int(args.head_eval_batch_size),
        num_workers=int(args.num_workers),
        learning_rate=float(args.head_learning_rate),
        weight_decay=float(args.head_weight_decay),
        epochs=int(args.head_epochs),
        lambda_tail_classification=float(args.lambda_tail_classification),
        lambda_under_classification=float(args.lambda_under_classification),
        lambda_regression=float(args.lambda_regression),
        risk_regression_weight=float(args.risk_regression_weight),
        lambda_directional=float(args.lambda_directional),
        lambda_leakage=float(args.lambda_leakage),
        max_positive_weight=float(args.max_positive_weight),
        overall_budget=float(args.overall_budget),
        non_tail_budget=float(args.non_tail_budget),
        bias_limit=float(args.bias_limit),
        seed=int(args.seed),
    )

    prevalence_base = ablation_module.make_frozen_base(
        baseline_module,
        base_checkpoint,
        int(args.hidden_dim),
        int(args.attention_heads),
        device,
    )

    prevalence_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.train_label,
        training=False,
        repeats=int(args.prevalence_repeats),
        **common_dataset,
    )
    prevalence_loader = DataLoader(
        prevalence_dataset,
        shuffle=False,
        **loader_kwargs(args, device),
    )

    prevalence = ablation_module.estimate_training_prevalence(
        prevalence_base,
        prevalence_loader,
        thresholds_np,
        device,
    )

    tail_pos_weight_np = np.clip(
        (1.0 - prevalence["tail"])
        / np.maximum(prevalence["tail"], 1e-6),
        1.0,
        float(args.max_positive_weight),
    )
    under_pos_weight_np = np.clip(
        (1.0 - prevalence["under"])
        / np.maximum(prevalence["under"], 1e-6),
        1.0,
        float(args.max_positive_weight),
    )

    tail_pos_weight = torch.as_tensor(
        tail_pos_weight_np,
        dtype=torch.float32,
        device=device,
    )
    under_pos_weight = torch.as_tensor(
        under_pos_weight_np,
        dtype=torch.float32,
        device=device,
    )

    selection_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.validation_label,
        training=False,
        repeats=int(args.head_selection_validation_repeats),
        **common_dataset,
    )
    selection_loader = DataLoader(
        selection_dataset,
        shuffle=False,
        **loader_kwargs(args, device),
    )

    selection_cache = ablation_module.build_eval_cache(
        prevalence_base,
        selection_loader,
        thresholds_np,
        device,
    )
    base_validation_metrics = ablation_module.metric_bundle(
        selection_cache,
        selection_cache["base"],
    )

    thresholds_tensor = torch.as_tensor(
        thresholds_np,
        dtype=torch.float32,
        device=device,
    )

    return (
        head_args,
        base_checkpoint,
        common_dataset,
        prevalence,
        tail_pos_weight,
        under_pos_weight,
        selection_cache,
        base_validation_metrics,
        thresholds_tensor,
    )


def train_caurc_a4(
    baseline_module,
    ablation_module,
    base_checkpoint_path: Path,
    native_manifest: Path,
    thresholds_np: np.ndarray,
    scenario_dir: Path,
    t0: int,
    k: int,
    args,
    device: torch.device,
):
    (
        head_args,
        base_checkpoint,
        common_dataset,
        prevalence,
        tail_pos_weight,
        under_pos_weight,
        selection_cache,
        base_validation_metrics,
        thresholds_tensor,
    ) = prepare_head_training(
        baseline_module,
        ablation_module,
        base_checkpoint_path,
        native_manifest,
        thresholds_np,
        t0,
        k,
        args,
        device,
    )

    powers = ablation_module.parse_float_list(args.powers)

    # A4 was the second variant in the original grouped A3/A4/A5 study,
    # therefore preserve the original independent initialization stream.
    variant_seed = int(args.seed) + 2000

    result = ablation_module.run_variant(
        variant="A4_under_only",
        args=head_args,
        baseline_module=baseline_module,
        base_checkpoint=base_checkpoint,
        thresholds_np=thresholds_np,
        thresholds_tensor=thresholds_tensor,
        prevalence=prevalence,
        tail_pos_weight=tail_pos_weight,
        under_pos_weight=under_pos_weight,
        common_dataset=common_dataset,
        selection_cache=selection_cache,
        base_validation_metrics=base_validation_metrics,
        powers=powers,
        device=device,
        variant_seed=variant_seed,
        out_dir=scenario_dir,
    )

    return result, head_args, base_checkpoint, prevalence


# -----------------------------------------------------------------------------
# Test evaluation and bootstrap
# -----------------------------------------------------------------------------


def build_test_cache(
    baseline_module,
    ablation_module,
    base_checkpoint: dict[str, Any],
    manifest_path: Path,
    thresholds_np: np.ndarray,
    t0: int,
    k: int,
    args,
    device: torch.device,
) -> dict[str, Any]:
    common_dataset = {
        "manifest": str(manifest_path),
        "h5_root": str(args.h5_root_resolved),
        "split_column": args.split_column,
        "t0_sec": int(t0),
        "input_stations": int(k),
        "target_stations": int(args.target_stations),
        "input_pre_sec": float(args.input_pre_sec),
        "seed": int(args.seed),
    }

    test_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.test_label,
        training=False,
        repeats=int(args.test_repeats),
        **common_dataset,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs(args, device),
    )

    base = ablation_module.make_frozen_base(
        baseline_module,
        base_checkpoint,
        int(args.hidden_dim),
        int(args.attention_heads),
        device,
    )

    return ablation_module.build_eval_cache(
        base,
        test_loader,
        thresholds_np,
        device,
    )


def prediction_frame_from_cache(
    cache: dict[str, Any],
    final_prediction: np.ndarray,
    manifest_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"event_id": str})
    manifest["event_id"] = manifest["event_id"].astype(str).str.strip()

    if "sequence_group" not in manifest.columns:
        raise ValueError("Manifest lacks sequence_group required for bootstrap.")

    sequence_map = (
        manifest[["event_id", "sequence_group"]]
        .drop_duplicates("event_id")
        .set_index("event_id")["sequence_group"]
        .astype(str)
        .to_dict()
    )

    frame = pd.DataFrame(
        {
            "event_id": cache["event_id"].astype(str),
            "repeat": cache["repeat"].astype(int),
            "target_slot": cache["target_slot"].astype(int),
            "target_station_index": cache["target_station_index"].astype(int),
            "true_log10_pga": cache["truth"][:, 0],
            "true_log10_pgv": cache["truth"][:, 1],
            "is_tail_pga": cache["tail"][:, 0].astype(bool),
            "is_tail_pgv": cache["tail"][:, 1].astype(bool),
            "base_log10_pga": cache["base"][:, 0],
            "base_log10_pgv": cache["base"][:, 1],
            "final_log10_pga": final_prediction[:, 0],
            "final_log10_pgv": final_prediction[:, 1],
        }
    )
    frame["sequence_group"] = frame["event_id"].map(sequence_map)
    if frame["sequence_group"].isna().any():
        missing = frame.loc[frame["sequence_group"].isna(), "event_id"].unique()
        raise RuntimeError(
            f"Missing sequence_group for test events: {missing[:20].tolist()}"
        )
    return frame


def long_metrics_table(
    base_metrics: dict[str, float],
    final_metrics: dict[str, float],
    t0: int,
    k: int,
    test_population: str,
    n_events: int,
    n_rows: int,
) -> pd.DataFrame:
    rows = []
    for model, metrics in (
        ("cross_attention_base", base_metrics),
        ("ca_urc", final_metrics),
    ):
        for key, value in metrics.items():
            if key.startswith("mean_"):
                continue
            rows.append(
                {
                    "t0_sec": int(t0),
                    "input_stations": int(k),
                    "budget": budget_name(t0, k),
                    "test_population": test_population,
                    "model": model,
                    "metric_key": key,
                    "value": float(value),
                    "n_events_total": int(n_events),
                    "n_target_rows_total": int(n_rows),
                }
            )
    return pd.DataFrame(rows)


def run_sequence_bootstrap(
    bootstrap_module,
    predictions: pd.DataFrame,
    repetitions: int,
    confidence_level: float,
    seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    bootstrap_module.METHOD_COLUMNS = {
        "cross_attention": {
            "pga": "base_log10_pga",
            "pgv": "base_log10_pgv",
            "source": "budget_grid",
        },
        "ca_urc": {
            "pga": "final_log10_pga",
            "pgv": "final_log10_pgv",
            "source": "budget_grid",
        },
    }

    event_metrics = bootstrap_module.build_event_level_metrics(
        predictions,
        ["cross_attention", "ca_urc"],
    )
    event_metrics.to_csv(
        out_dir / "event_level_canonical_metrics.csv",
        index=False,
    )

    event_group = (
        predictions[["event_id", "sequence_group"]]
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )

    delta_matrix, metadata = bootstrap_module.build_delta_matrix(
        event_metrics,
        event_group,
        [("ca_urc", "cross_attention")],
    )

    group_names, group_indices = bootstrap_module.group_index_arrays(event_group)

    if len(group_indices) < 2:
        warning = pd.DataFrame(
            [
                {
                    "status": "insufficient_sequence_groups",
                    "n_sequence_groups": int(len(group_indices)),
                }
            ]
        )
        warning.to_csv(
            out_dir / "hierarchical_bootstrap_summary.csv",
            index=False,
        )
        return warning

    cluster_distribution = bootstrap_module.sequence_cluster_bootstrap(
        delta_matrix,
        group_indices,
        int(repetitions),
        int(seed),
    )
    hierarchical_distribution = (
        bootstrap_module.hierarchical_sequence_event_bootstrap(
            delta_matrix,
            group_indices,
            int(repetitions),
            int(seed) + 1,
        )
    )

    cluster_summary = bootstrap_module.bootstrap_summary(
        cluster_distribution,
        metadata,
        "sequence_cluster",
        float(confidence_level),
    )
    hierarchical_summary = bootstrap_module.bootstrap_summary(
        hierarchical_distribution,
        metadata,
        "hierarchical_sequence_event",
        float(confidence_level),
    )

    cluster_summary.to_csv(
        out_dir / "sequence_cluster_bootstrap_summary.csv",
        index=False,
    )
    hierarchical_summary.to_csv(
        out_dir / "hierarchical_bootstrap_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "sequence_group": group_names,
            "n_events": [int(len(x)) for x in group_indices],
        }
    ).to_csv(
        out_dir / "common_test_sequence_groups.csv",
        index=False,
    )

    return hierarchical_summary


def bootstrap_tail_mae_record(
    bootstrap: pd.DataFrame,
    quantity: str,
) -> dict[str, float]:
    required = {
        "quantity",
        "population",
        "metric",
        "reference_value",
        "candidate_value",
        "point_delta_candidate_minus_reference",
    }
    if not required.issubset(bootstrap.columns):
        return {}

    row = bootstrap.loc[
        bootstrap["quantity"].astype(str).eq(quantity)
        & bootstrap["population"].astype(str).eq("high_motion_tail")
        & bootstrap["metric"].astype(str).eq("mae")
    ]
    if len(row) != 1:
        return {}
    r = row.iloc[0]

    result = {
        f"{quantity}_bootstrap_base_tail_mae": float(r["reference_value"]),
        f"{quantity}_bootstrap_caurc_tail_mae": float(r["candidate_value"]),
        f"{quantity}_bootstrap_tail_delta": float(
            r["point_delta_candidate_minus_reference"]
        ),
    }
    if "relative_change_percent" in r.index and pd.notna(r["relative_change_percent"]):
        result[f"{quantity}_tail_relative_change_percent"] = float(
            r["relative_change_percent"]
        )
    else:
        result[f"{quantity}_tail_relative_change_percent"] = (
            100.0
            * float(r["point_delta_candidate_minus_reference"])
            / max(abs(float(r["reference_value"])), 1e-12)
        )
    for source, target in (
        ("ci_lower", "ci_low"),
        ("ci_upper", "ci_high"),
        ("two_sided_bootstrap_sign_p", "p_two_sided"),
    ):
        if source in r.index and pd.notna(r[source]):
            result[f"{quantity}_{target}"] = float(r[source])
    return result


# -----------------------------------------------------------------------------
# Per-budget pipeline
# -----------------------------------------------------------------------------


def run_one_budget(
    t0: int,
    k: int,
    audited: pd.DataFrame,
    common_ids: set[str],
    ablation_module,
    bootstrap_module,
    args,
    device: torch.device,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    name = budget_name(t0, k)
    scenario_dir = Path(args.out_dir) / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    complete_path = scenario_dir / "scenario_summary.json"

    if args.resume and complete_path.exists():
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"\n[{name}] resume: already complete; skipping training.")
            metrics_frames = []
            for filename in ("native_test_metrics.csv", "common_test_metrics.csv"):
                path = scenario_dir / filename
                if path.exists():
                    metrics_frames.append(pd.read_csv(path))
            return existing, metrics_frames

    print("\n" + "=" * 100)
    print(f"OBSERVATION BUDGET: t0={t0}s, K={k}")
    print("=" * 100)

    native_manifest, common_manifest, counts = write_budget_manifests(
        audited,
        t0,
        k,
        common_ids,
        args.split_column,
        args.train_label,
        args.validation_label,
        args.test_label,
        scenario_dir,
    )

    # TRAIN-only Q90, budget-specific because the eligible training-event set
    # may change with K/t0.
    threshold_path = scenario_dir / "tail_thresholds_q0.90.json"
    threshold_data = compute_train_only_thresholds(
        manifest_path=native_manifest,
        split_column=args.split_column,
        train_label=args.train_label,
        t0_sec=int(t0),
        quantile=float(args.tail_quantile),
        output_path=threshold_path,
    )
    thresholds_np = np.asarray(
        [
            threshold_data["log10_pga_threshold"],
            threshold_data["log10_pgv_threshold"],
        ],
        dtype=np.float32,
    )

    if (int(t0), int(k)) == (5, 5):
        pga_diff = abs(float(thresholds_np[0]) - EXPECTED_MAIN_55_THRESHOLDS["pga"])
        pgv_diff = abs(float(thresholds_np[1]) - EXPECTED_MAIN_55_THRESHOLDS["pgv"])
        if max(pga_diff, pgv_diff) > 5e-4:
            print(
                "WARNING: (5s,5) TRAIN-only Q90 thresholds differ from the "
                "established grouped anchor: "
                f"observed={thresholds_np.tolist()}, "
                f"expected={[EXPECTED_MAIN_55_THRESHOLDS['pga'], EXPECTED_MAIN_55_THRESHOLDS['pgv']]}."
            )
        else:
            print(
                "(5s,5) threshold anchor: PASS "
                f"({thresholds_np[0]:.6f}/{thresholds_np[1]:.6f})."
            )

    # Load a fresh baseline module for every budget so seed-protocol monkey
    # patches from the previous budget cannot leak into the next one.
    baseline_module = load_module(
        args.baseline_module,
        f"budget_baseline_{t0}_{k}_{int(time.time_ns())}",
    )

    set_seed(int(args.seed))

    base_checkpoint_path, base_training_summary = train_cross_attention_base(
        baseline_module,
        ablation_module,
        native_manifest,
        scenario_dir,
        t0,
        k,
        args,
        device,
    )

    (
        variant_result,
        head_args,
        base_checkpoint,
        prevalence,
    ) = train_caurc_a4(
        baseline_module,
        ablation_module,
        base_checkpoint_path,
        native_manifest,
        thresholds_np,
        scenario_dir,
        t0,
        k,
        args,
        device,
    )

    base_row: dict[str, Any] = {
        "status": "selection_failed",
        "budget": name,
        "t0_sec": int(t0),
        "input_stations": int(k),
        **counts,
        "threshold_pga": float(thresholds_np[0]),
        "threshold_pgv": float(thresholds_np[1]),
        "base_best_epoch": int(base_training_summary.get("best_epoch", -1)),
        "base_validation_score": float(
            base_training_summary.get("best_validation_score", np.nan)
        ),
        "tail_prevalence_train_pga": float(prevalence["tail"][0]),
        "tail_prevalence_train_pgv": float(prevalence["tail"][1]),
        "under05_prevalence_train_pga": float(prevalence["under"][0]),
        "under05_prevalence_train_pgv": float(prevalence["under"][1]),
    }

    if not variant_result.get("feasible", False):
        complete_path.write_text(
            json.dumps(json_safe(base_row), indent=2),
            encoding="utf-8",
        )
        return base_row, []

    base_row.update(
        {
            "selected_head_epoch": int(variant_result["selected_epoch"]),
            "selected_gamma": float(variant_result["selected_gamma"]),
            "selected_at_upper_gamma_boundary": bool(
                variant_result["selected_at_upper_gamma_boundary"]
            ),
        }
    )

    if variant_result["selected_at_upper_gamma_boundary"]:
        base_row["status"] = "gamma_boundary_validation_only"
        complete_path.write_text(
            json.dumps(json_safe(base_row), indent=2),
            encoding="utf-8",
        )
        print(
            f"[{name}] gamma selected at upper validation boundary. "
            "TEST remains untouched. Extend --powers and rerun this budget."
        )
        return base_row, []

    # Only now is TEST accessed.
    ablation_module.patch_locked_test_seed_protocol(baseline_module)

    all_metric_frames: list[pd.DataFrame] = []
    test_outputs = {}

    for population_name, manifest_path in (
        ("native", native_manifest),
        ("common", common_manifest),
    ):
        cache = build_test_cache(
            baseline_module,
            ablation_module,
            base_checkpoint,
            manifest_path,
            thresholds_np,
            t0,
            k,
            args,
            device,
        )

        base_metrics = ablation_module.metric_bundle(cache, cache["base"])
        final_metrics, final_prediction = (
            ablation_module.evaluate_variant_on_locked_test(
                variant_result,
                head_args,
                baseline_module,
                base_checkpoint,
                prevalence,
                cache,
                device,
            )
        )

        n_events = int(pd.Series(cache["event_id"].astype(str)).nunique())
        n_rows = int(len(cache["event_id"]))

        metrics_long = long_metrics_table(
            base_metrics,
            final_metrics,
            t0,
            k,
            population_name,
            n_events,
            n_rows,
        )
        metrics_path = scenario_dir / f"{population_name}_test_metrics.csv"
        metrics_long.to_csv(metrics_path, index=False)
        all_metric_frames.append(metrics_long)

        test_outputs[population_name] = {
            "cache": cache,
            "base_metrics": base_metrics,
            "final_metrics": final_metrics,
            "final_prediction": final_prediction,
            "manifest": manifest_path,
        }

    common_output = test_outputs["common"]
    common_predictions = prediction_frame_from_cache(
        common_output["cache"],
        common_output["final_prediction"],
        common_manifest,
    )
    common_predictions.to_csv(
        scenario_dir / "common_test_predictions.csv",
        index=False,
    )

    hierarchical = run_sequence_bootstrap(
        bootstrap_module,
        common_predictions,
        repetitions=int(args.bootstrap_repetitions),
        confidence_level=float(args.confidence_level),
        seed=int(args.seed),
        out_dir=scenario_dir,
    )

    common_base = common_output["base_metrics"]
    common_final = common_output["final_metrics"]
    native_base = test_outputs["native"]["base_metrics"]
    native_final = test_outputs["native"]["final_metrics"]

    common_tail_pga = parse_bool(common_predictions["is_tail_pga"])
    common_tail_pgv = parse_bool(common_predictions["is_tail_pgv"])

    summary = dict(base_row)
    summary.update(
        {
            "status": "complete",
            "common_test_rows": int(len(common_predictions)),
            "common_test_sequence_groups": int(
                common_predictions["sequence_group"].astype(str).nunique()
            ),
            "common_pga_tail_rows": int(common_tail_pga.sum()),
            "common_pgv_tail_rows": int(common_tail_pgv.sum()),
            "common_pga_tail_events": int(
                common_predictions.loc[common_tail_pga, "event_id"].nunique()
            ),
            "common_pgv_tail_events": int(
                common_predictions.loc[common_tail_pgv, "event_id"].nunique()
            ),
        }
    )

    for prefix, metrics in (
        ("common_base", common_base),
        ("common_caurc", common_final),
        ("native_base", native_base),
        ("native_caurc", native_final),
    ):
        for key, value in metrics.items():
            summary[f"{prefix}_{key}"] = float(value)

    # Relative point-estimate improvements on the common test core.
    for quantity in ("pga", "pgv"):
        base_tail = float(common_base[f"tail_mae_{quantity}"])
        final_tail = float(common_final[f"tail_mae_{quantity}"])
        summary[f"common_{quantity}_tail_mae_reduction_percent"] = (
            100.0 * (base_tail - final_tail) / max(abs(base_tail), 1e-12)
        )
        summary.update(bootstrap_tail_mae_record(hierarchical, quantity))

    complete_path.write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    # Explicit cleanup before next budget.
    del baseline_module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, all_metric_frames


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--master-manifest",
        default="auto",
        help=(
            "Master grouped-split manifest BEFORE any single T0/K eligibility "
            "filter. 'auto' tries all_events_with_scenario_flags.csv then "
            "event_splits_v4.csv."
        ),
    )
    p.add_argument(
        "--h5-root",
        default="auto",
        help=(
            "Event HDF5 directory. 'auto' tries data/processed_full_v4/events "
            "and data/scedc/processed_full_v4/events."
        ),
    )

    p.add_argument("--baseline-module", default="45_phase2_strong_baseline_suite.py")
    p.add_argument("--ablation-module", default="63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py")
    p.add_argument("--bootstrap-module", default="62_sequence_aware_paired_bootstrap_cadrg.py")

    p.add_argument(
        "--budgets",
        default="minimal",
        help=(
            "minimal | full | custom comma-separated t0:K pairs. "
            "Examples: minimal; full; 3:3,5:5,7:5,10:10"
        ),
    )

    p.add_argument("--split-column", default="split_grouped")
    p.add_argument("--train-label", default="train")
    p.add_argument("--validation-label", default="validation")
    p.add_argument("--test-label", default="test")
    p.add_argument("--target-stations", type=int, default=10)
    p.add_argument("--input-pre-sec", type=float, default=2.0)
    p.add_argument("--tail-quantile", type=float, default=0.90)

    # Base training = final grouped Cross-Attention protocol.
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--graph-initial-length-scale-km", type=float, default=50.0)
    p.add_argument("--base-epochs", type=int, default=50)
    p.add_argument("--base-validation-repeats", type=int, default=3)
    p.add_argument("--base-learning-rate", type=float, default=1e-3)
    p.add_argument("--base-weight-decay", type=float, default=1e-4)
    p.add_argument("--base-lr-patience", type=int, default=4)
    p.add_argument("--base-minimum-learning-rate", type=float, default=1e-5)
    p.add_argument("--base-minimum-delta", type=float, default=1e-4)
    p.add_argument("--base-early-stopping-patience", type=int, default=12)

    # CA-URC A4 head = original grouped A4 protocol.
    p.add_argument("--risk-hidden", type=int, default=64)
    p.add_argument("--maximum-correction", type=float, default=1.5)
    p.add_argument("--head-epochs", type=int, default=40)
    p.add_argument("--head-learning-rate", type=float, default=1e-3)
    p.add_argument("--head-weight-decay", type=float, default=1e-4)
    p.add_argument("--prevalence-repeats", type=int, default=3)
    p.add_argument("--head-training-validation-repeats", type=int, default=3)
    p.add_argument("--head-selection-validation-repeats", type=int, default=20)
    p.add_argument("--head-eval-batch-size", type=int, default=8192)
    p.add_argument("--lambda-tail-classification", type=float, default=0.25)
    p.add_argument("--lambda-under-classification", type=float, default=0.25)
    p.add_argument("--lambda-regression", type=float, default=1.0)
    p.add_argument("--risk-regression-weight", type=float, default=2.0)
    p.add_argument("--lambda-directional", type=float, default=0.50)
    p.add_argument("--lambda-leakage", type=float, default=0.05)
    p.add_argument("--max-positive-weight", type=float, default=9.0)
    p.add_argument(
        "--powers",
        default="1,1.5,2,2.5,3,4,5,6,7,8,10,12,15,20",
    )
    p.add_argument("--overall-budget", type=float, default=0.005)
    p.add_argument("--non-tail-budget", type=float, default=0.003)
    p.add_argument("--bias-limit", type=float, default=0.05)

    p.add_argument("--test-repeats", type=int, default=20)
    p.add_argument("--bootstrap-repetitions", type=int, default=10000)
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument(
        "--min-common-test-events",
        type=int,
        default=50,
        help=(
            "Stop before training if the all-budget common test core is smaller "
            "than this. For the 12-cell grid, inspect --dry-run first."
        ),
    )
    p.add_argument(
        "--allow-small-common-test",
        action="store_true",
        help="Allow training even when the common test core is below the minimum.",
    )
    p.add_argument(
        "--strict-main-anchor",
        action="store_true",
        help=(
            "Require recomputed (t0=5,K=5) eligibility counts to equal the "
            "known 1189/207/224 grouped counts."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only audit eligibility/common-test size; do not train models.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip budgets whose scenario_summary.json has status=complete.",
    )
    p.add_argument(
        "--out-dir",
        default="runs/caurc_observation_budget_grid",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    budgets = parse_budgets(args.budgets)

    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    master_path = resolve_auto_path(
        args.master_manifest,
        [
            "data/scedc/model_manifests/all_events_with_scenario_flags.csv",
            "data/model_manifests/all_events_with_scenario_flags.csv",
            "data/scedc/splits_v4/event_splits_v4.csv",
            "data/splits_v4/event_splits_v4.csv",
        ],
        "master manifest",
    )
    h5_root = resolve_auto_path(
        args.h5_root,
        [
            "data/processed_full_v4/events",
            "data/scedc/processed_full_v4/events",
        ],
        "HDF5 root",
    )
    args.h5_root_resolved = str(h5_root.resolve())

    master = pd.read_csv(master_path, dtype={"event_id": str})
    required = {"event_id", args.split_column, "sequence_group"}
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(
            f"Master manifest lacks required columns: {sorted(missing)}"
        )

    master["event_id"] = master["event_id"].astype(str).str.strip()
    if master["event_id"].duplicated().any():
        duplicate = master.loc[
            master["event_id"].duplicated(keep=False),
            ["event_id", args.split_column, "sequence_group"],
        ].head(20)
        raise RuntimeError(
            "Master manifest has duplicated event_id rows. Resolve this before "
            f"budget training:\n{duplicate}"
        )

    print("=== CA-URC observation-budget robustness ===")
    print(f"Master manifest : {master_path.resolve()}")
    print(f"HDF5 root       : {h5_root.resolve()}")
    print(f"Budgets         : {budgets}")
    print(f"Target Q        : {args.target_stations}")
    print(f"Dry run         : {args.dry_run}")

    audited_path = root_out / "all_events_with_budget_flags.csv"
    if args.resume and audited_path.exists():
        audited = pd.read_csv(audited_path, dtype={"event_id": str})
        needed = {eligible_column(t0, k) for t0, k in budgets}
        if not needed.issubset(audited.columns):
            audited = inspect_budget_eligibility(
                master,
                h5_root,
                budgets,
                int(args.target_stations),
            )
            audited.to_csv(audited_path, index=False)
        else:
            print(f"Reuse eligibility audit: {audited_path.resolve()}")
    else:
        audited = inspect_budget_eligibility(
            master,
            h5_root,
            budgets,
            int(args.target_stations),
        )
        audited.to_csv(audited_path, index=False)

    summary = eligibility_summary(
        audited,
        budgets,
        args.split_column,
        args.train_label,
        args.validation_label,
        args.test_label,
    )
    summary.to_csv(
        root_out / "budget_eligibility_summary.csv",
        index=False,
    )

    print("\n=== Native eligible event counts ===")
    print(
        summary.pivot_table(
            index=["t0_sec", "input_stations"],
            columns="split",
            values="n_eligible_events",
            aggfunc="first",
        ).to_string()
    )

    # Anchor audit for the established 5 s / 5-station grouped scenario.
    if (5, 5) in budgets:
        anchor = summary.loc[
            summary["t0_sec"].eq(5)
            & summary["input_stations"].eq(5)
        ].set_index("split")["n_eligible_events"].to_dict()
        failures = []
        for split, expected in EXPECTED_MAIN_55.items():
            observed = int(anchor.get(split, -1))
            if observed != int(expected):
                failures.append(
                    f"{split}: observed={observed}, expected={expected}"
                )
        if failures:
            message = (
                "(5s,5) eligibility does not reproduce the established grouped "
                "scenario counts: " + "; ".join(failures)
            )
            if args.strict_main_anchor:
                raise RuntimeError(message)
            print("WARNING:", message)
        else:
            print("\n(5s,5) eligibility anchor: PASS (1189/207/224).")

    common_ids_list = common_test_ids(
        audited,
        budgets,
        args.split_column,
        args.test_label,
    )
    common_ids = set(common_ids_list)

    common_frame = audited.loc[
        audited["event_id"].astype(str).isin(common_ids)
        & audited[args.split_column].astype(str).eq(str(args.test_label))
    ][["event_id", "sequence_group"]].drop_duplicates("event_id")

    common_frame.to_csv(
        root_out / "common_test_event_ids.csv",
        index=False,
    )

    print("\n=== Common test core ===")
    print(f"Events          : {len(common_ids):,}")
    print(
        "Sequence groups : "
        f"{common_frame['sequence_group'].astype(str).nunique():,}"
    )

    if len(common_ids) < int(args.min_common_test_events):
        message = (
            f"Common test core has only {len(common_ids)} events, below "
            f"--min-common-test-events={args.min_common_test_events}."
        )
        if not args.allow_small_common_test:
            print("\nSTOP:", message)
            print(
                "This is expected to be possible for the full 12-cell grid "
                "because (3s,10) is very restrictive. Use the minimal grid for "
                "the primary paired robustness test, or inspect the counts and "
                "rerun with --allow-small-common-test only for descriptive "
                "supplementary analysis."
            )
            protocol = {
                "status": "stopped_small_common_test",
                "budgets": budgets,
                "common_test_events": len(common_ids),
                "min_common_test_events": int(args.min_common_test_events),
                "master_manifest": str(master_path.resolve()),
                "h5_root": str(h5_root.resolve()),
            }
            (root_out / "protocol.json").write_text(
                json.dumps(json_safe(protocol), indent=2),
                encoding="utf-8",
            )
            return

    protocol = {
        "status": "eligibility_complete" if args.dry_run else "training_started",
        "budgets": budgets,
        "common_test_event_definition": (
            "intersection of grouped-test events satisfying availability-only "
            "eligibility for every requested (t0,K) budget"
        ),
        "common_test_events": int(len(common_ids)),
        "common_test_sequence_groups": int(
            common_frame["sequence_group"].astype(str).nunique()
        ),
        "target_stations": int(args.target_stations),
        "metric_hierarchy": "targets -> repeats -> events",
        "bootstrap_hierarchy": "sequence groups -> events",
        "tail_definition": "budget-specific TRAIN-only Q0.90 remaining PGA/PGV",
        "under05_definition": "prediction - truth <= -0.5 log10 units",
        "test_used_for_selection": False,
        "master_manifest": str(master_path.resolve()),
        "h5_root": str(h5_root.resolve()),
        "args": vars(args),
    }
    (root_out / "protocol.json").write_text(
        json.dumps(json_safe(protocol), indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        print("\nDry run complete. No model was trained.")
        return

    device = choose_device(args.device)
    print(f"\nDevice: {device}")

    ablation_module = load_module(
        args.ablation_module,
        "budget_ablation_module",
    )
    bootstrap_module = load_module(
        args.bootstrap_module,
        "budget_bootstrap_module",
    )

    scenario_rows: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []

    for t0, k in budgets:
        try:
            row, frames = run_one_budget(
                int(t0),
                int(k),
                audited,
                common_ids,
                ablation_module,
                bootstrap_module,
                args,
                device,
            )
        except Exception as exc:
            row = {
                "status": "failed",
                "budget": budget_name(t0, k),
                "t0_sec": int(t0),
                "input_stations": int(k),
                "error": repr(exc),
            }
            failure_path = root_out / budget_name(t0, k) / "FAILED.txt"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(repr(exc), encoding="utf-8")
            print(f"\n[{budget_name(t0, k)}] FAILED: {exc}")
            frames = []

        scenario_rows.append(row)
        metric_frames.extend(frames)

        pd.DataFrame(scenario_rows).to_csv(
            root_out / "observation_budget_summary.csv",
            index=False,
        )
        if metric_frames:
            pd.concat(metric_frames, ignore_index=True).to_csv(
                root_out / "observation_budget_long_metrics.csv",
                index=False,
            )

    final_summary = pd.DataFrame(scenario_rows)
    final_summary.to_csv(
        root_out / "observation_budget_summary.csv",
        index=False,
    )

    protocol["status"] = "finished"
    protocol["completed_budgets"] = int(
        final_summary["status"].eq("complete").sum()
        if "status" in final_summary.columns
        else 0
    )
    protocol["failed_or_unusable_budgets"] = int(
        len(final_summary) - protocol["completed_budgets"]
    )
    (root_out / "protocol.json").write_text(
        json.dumps(json_safe(protocol), indent=2),
        encoding="utf-8",
    )

    print("\n=== Observation-budget experiment finished ===")
    display_cols = [
        c
        for c in [
            "budget",
            "status",
            "native_train",
            "native_validation",
            "native_test",
            "common_test",
            "selected_head_epoch",
            "selected_gamma",
            "common_base_tail_mae_pga",
            "common_caurc_tail_mae_pga",
            "common_pga_tail_mae_reduction_percent",
            "common_base_tail_mae_pgv",
            "common_caurc_tail_mae_pgv",
            "common_pgv_tail_mae_reduction_percent",
        ]
        if c in final_summary.columns
    ]
    print(final_summary[display_cols].to_string(index=False))
    print(f"\nOutputs: {root_out.resolve()}")


if __name__ == "__main__":
    main()
