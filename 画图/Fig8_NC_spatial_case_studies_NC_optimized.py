#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig8_NC_spatial_case_studies.py

Nature Communications-style spatial case-study analysis for the final
Cross-Attention Underprediction-Risk Correction model (CA-URC):

    y_final = y_CA + p_under**gamma * Delta

Default locked configuration:
    A4_under_only, selected epoch=3, gamma=5.

Scientific design
-----------------
1. Cases are selected algorithmically from the locked 44,800 target rows,
   before any spatial map is inspected.
2. The selected input-station set and ten locked target stations are
   reconstructed using the original deterministic test-draw convention.
3. The frozen model is then queried at every non-input station of each
   selected event solely to visualize spatial behaviour.
4. Quantitative aggregate conclusions remain based on the locked target
   draws; all-station maps are descriptive case studies.
5. Station values are plotted at their actual locations. No interpolated
   surface is presented as a dense model output or continuous ground truth.

Main cases
----------
a) Representative preservation:
   one (preferred) or at most one high-motion target and base overall error
   closest to the median, with minimal paired overall change as tie-breaker.

b) Successful tail correction:
   at least two high-motion targets; largest tail-MAE reduction subject to a
   bounded non-tail degradation (default <= 0.02 log10 units).

c) Residual failure:
   at least two high-motion targets and at least one severe underprediction
   after CA-URC; largest remaining CA-URC tail MAE.

Supplementary overcorrection case
---------------------------------
The event-repeat unit with the largest non-tail MAE increase, excluding the
three main cases when possible.

Primary outputs
---------------
Fig8_spatial_cases.png/.pdf/.svg
FigS_selected_case_risk_and_correction.png/.pdf/.svg
TableS7_selected_cases.csv/.tex
TableS7_failure_mode_summary.csv/.tex
Section2_7_manuscript_draft.txt
Fig8_caption.txt
FigS_caption.txt
case_selection_statistics.csv
case_selection_audit.json

The --demo mode creates a layout preview from synthetic data and does not
require project files or PyTorch checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOG10_FACTOR_2 = math.log10(2.0)
SEVERE_UNDER = -0.5
EXPECTED_ROWS = 44_800
EXPECTED_EVENTS = 224
EXPECTED_PGA_TAIL_ROWS = 1_363
EXPECTED_PGV_TAIL_ROWS = 1_485

BASE_COLUMNS = {
    "pga": "A2_cross_attention_base_log10_pga",
    "pgv": "A2_cross_attention_base_log10_pgv",
}
FINAL_COLUMNS = {
    "pga": "A4_under_only_log10_pga",
    "pgv": "A4_under_only_log10_pgv",
}
TRUTH_COLUMNS = {
    "pga": "true_log10_pga",
    "pgv": "true_log10_pgv",
}
TAIL_COLUMNS = {
    "pga": "is_tail_pga",
    "pgv": "is_tail_pgv",
}

CASE_ORDER = ["representative", "successful_correction", "residual_failure"]
CASE_LABELS = {
    "representative": "Representative preservation",
    "successful_correction": "Successful tail correction",
    "residual_failure": "Residual failure",
    "overcorrection": "Non-tail overcorrection",
}


@dataclass
class CaseSelection:
    name: str
    row: pd.Series


# -----------------------------------------------------------------------------
# General utilities
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


def load_checkpoint(path: str | Path, device):
    import torch

    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)



def configure_torch_determinism() -> dict[str, Any]:
    """
    Make repeated frozen-model inference as deterministic as practical.

    The archived locked CSV may still differ slightly because it could have been
    produced with another CUDA/PyTorch kernel choice.  This helper stabilizes the
    *current* run and disables TF32 so that repeated inference does not wander by
    O(1e-4) across executions.
    """
    import torch

    audit: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        audit["deterministic_algorithms"] = True
    except Exception as exc:
        audit["deterministic_algorithms"] = f"unavailable: {exc}"

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        audit["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        audit["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)

        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
            audit["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
            audit["cuda_matmul_allow_tf32"] = bool(
                torch.backends.cuda.matmul.allow_tf32
            )

    # Prefer the deterministic/math SDPA implementation when the API exists.
    # Cross-attention output is mathematically independent across target queries,
    # but different fused kernels can introduce small batch-shape-dependent drift.
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            audit["flash_sdp"] = False
            audit["mem_efficient_sdp"] = False
            audit["math_sdp"] = True
    except Exception as exc:
        audit["sdp_configuration"] = f"unavailable: {exc}"

    return audit


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    unknown = sorted(set(text.unique()).difference(mapping))
    if unknown:
        raise ValueError(f"Cannot parse boolean values: {unknown[:20]}")
    return text.map(mapping).to_numpy(dtype=bool)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def detect_value(row: pd.Series, candidates: list[str]) -> float | None:
    for column in candidates:
        if column not in row.index:
            continue
        value = pd.to_numeric(row[column], errors="coerce")
        if np.isfinite(value):
            return float(value)
    return None


def detect_magnitude(row: pd.Series) -> float:
    value = detect_value(
        row,
        ["magnitude", "mag", "event_magnitude", "catalog_magnitude", "magnitude_value"],
    )
    return float(value) if value is not None else float("nan")


def detect_epicenter(row: pd.Series) -> tuple[float | None, float | None]:
    lat = detect_value(
        row,
        ["event_latitude", "latitude", "lat", "event_lat", "hypocenter_latitude"],
    )
    lon = detect_value(
        row,
        ["event_longitude", "longitude", "lon", "event_lon", "hypocenter_longitude"],
    )
    return lat, lon


def local_xy_km(
    coordinates: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
) -> np.ndarray:
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    x = (
        (longitude - origin_longitude)
        * 111.32
        * math.cos(math.radians(origin_latitude))
    )
    y = (latitude - origin_latitude) * 110.57
    return np.column_stack([x, y])


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def safe_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else float("nan")


def output_tex(frame: pd.DataFrame, path: Path, float_format: str = "%.4f") -> None:
    try:
        text = frame.to_latex(index=False, escape=False, float_format=float_format)
    except Exception:
        text = "% LaTeX export failed; use the accompanying CSV.\n"
    path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Locked prediction loading and case statistics
# -----------------------------------------------------------------------------


def load_locked_predictions(path: Path, skip_locked_audit: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"event_id": str})
    required = [
        "event_id",
        "repeat",
        "target_slot",
        "target_station_index",
        *TRUTH_COLUMNS.values(),
        *TAIL_COLUMNS.values(),
        *BASE_COLUMNS.values(),
        *FINAL_COLUMNS.values(),
    ]
    require_columns(frame, required, "locked CA-URC prediction file")

    frame["event_id"] = frame["event_id"].astype(str).str.strip()
    frame["is_tail_pga"] = parse_bool(frame["is_tail_pga"])
    frame["is_tail_pgv"] = parse_bool(frame["is_tail_pgv"])

    keys = ["event_id", "repeat", "target_station_index"]
    if frame.duplicated(keys).any():
        duplicated = frame.loc[frame.duplicated(keys, keep=False), keys].head(20)
        raise RuntimeError(f"Prediction rows are not unique on {keys}:\n{duplicated}")

    audit = {
        "prediction_rows": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "event_repeat_groups": int(frame[["event_id", "repeat"]].drop_duplicates().shape[0]),
        "pga_tail_rows": int(frame["is_tail_pga"].sum()),
        "pgv_tail_rows": int(frame["is_tail_pgv"].sum()),
        "pga_tail_events": int(frame.loc[frame["is_tail_pga"], "event_id"].nunique()),
        "pgv_tail_events": int(frame.loc[frame["is_tail_pgv"], "event_id"].nunique()),
    }

    if not skip_locked_audit:
        expected = {
            "prediction_rows": EXPECTED_ROWS,
            "events": EXPECTED_EVENTS,
            "pga_tail_rows": EXPECTED_PGA_TAIL_ROWS,
            "pgv_tail_rows": EXPECTED_PGV_TAIL_ROWS,
        }
        failures = [
            f"{name}: observed={audit[name]}, expected={value}"
            for name, value in expected.items()
            if int(audit[name]) != int(value)
        ]
        if failures:
            raise RuntimeError(
                "Locked-result audit failed. Do not use this file for the paper:\n  "
                + "\n  ".join(failures)
            )

    return frame, audit


def build_case_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (event_id, repeat), group in frame.groupby(["event_id", "repeat"], sort=False):
        group = group.sort_values("target_slot", kind="stable")
        truth = np.column_stack(
            [
                group[TRUTH_COLUMNS["pga"]].to_numpy(float),
                group[TRUTH_COLUMNS["pgv"]].to_numpy(float),
            ]
        )
        base = np.column_stack(
            [
                group[BASE_COLUMNS["pga"]].to_numpy(float),
                group[BASE_COLUMNS["pgv"]].to_numpy(float),
            ]
        )
        final = np.column_stack(
            [
                group[FINAL_COLUMNS["pga"]].to_numpy(float),
                group[FINAL_COLUMNS["pgv"]].to_numpy(float),
            ]
        )
        tail = np.column_stack(
            [
                group[TAIL_COLUMNS["pga"]].to_numpy(bool),
                group[TAIL_COLUMNS["pgv"]].to_numpy(bool),
            ]
        )

        base_residual = base - truth
        final_residual = final - truth
        base_abs = np.abs(base_residual)
        final_abs = np.abs(final_residual)
        non_tail = ~tail

        base_overall = safe_mean(base_abs)
        final_overall = safe_mean(final_abs)
        base_tail = safe_mean(base_abs[tail])
        final_tail = safe_mean(final_abs[tail])
        base_non_tail = safe_mean(base_abs[non_tail])
        final_non_tail = safe_mean(final_abs[non_tail])

        correction = final - base
        final_tail_residual = final_residual[tail]
        base_tail_residual = base_residual[tail]
        final_non_tail_residual = final_residual[non_tail]
        base_non_tail_residual = base_residual[non_tail]

        rows.append(
            {
                "event_id": str(event_id),
                "repeat": int(repeat),
                "n_targets": int(len(group)),
                "n_tail_targets": int(np.any(tail, axis=1).sum()),
                "n_tail_values": int(tail.sum()),
                "n_non_tail_values": int(non_tail.sum()),
                "base_overall_mae": base_overall,
                "caurc_overall_mae": final_overall,
                "overall_mae_change": final_overall - base_overall,
                "base_tail_mae": base_tail,
                "caurc_tail_mae": final_tail,
                "tail_mae_reduction": base_tail - final_tail,
                "base_non_tail_mae": base_non_tail,
                "caurc_non_tail_mae": final_non_tail,
                "non_tail_mae_change": final_non_tail - base_non_tail,
                "base_tail_u05": safe_fraction(base_tail_residual <= SEVERE_UNDER),
                "caurc_tail_u05": safe_fraction(final_tail_residual <= SEVERE_UNDER),
                "remaining_tail_u05_count": int(
                    np.sum(final_tail_residual <= SEVERE_UNDER)
                ),
                "base_non_tail_o05": safe_fraction(base_non_tail_residual >= 0.5),
                "caurc_non_tail_o05": safe_fraction(final_non_tail_residual >= 0.5),
                "mean_applied_correction": safe_mean(correction),
                "max_applied_correction": float(np.nanmax(correction)),
                "minimum_applied_correction": float(np.nanmin(correction)),
                "correction_nonnegative_with_tolerance": bool(np.nanmin(correction) >= -1e-6),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No event-repeat case statistics were created.")
    return result


def choose_manual(stats: pd.DataFrame, event_id: str, repeat: int | None) -> pd.Series:
    subset = stats.loc[stats["event_id"].astype(str).eq(str(event_id))].copy()
    if repeat is not None:
        subset = subset.loc[subset["repeat"].eq(int(repeat))]
    if subset.empty:
        raise ValueError(f"Requested event/repeat is absent: {event_id}/{repeat}")
    if repeat is None:
        subset = subset.sort_values(
            ["n_tail_targets", "tail_mae_reduction", "base_overall_mae"],
            ascending=[False, False, True],
        )
    return subset.iloc[0]


def exclude_events(frame: pd.DataFrame, event_ids: set[str]) -> pd.DataFrame:
    if not event_ids:
        return frame
    reduced = frame.loc[~frame["event_id"].astype(str).isin(event_ids)].copy()
    return reduced if not reduced.empty else frame.copy()


def select_cases(
    stats: pd.DataFrame,
    success_non_tail_tolerance: float,
    manual: dict[str, tuple[str | None, int | None]],
) -> dict[str, CaseSelection]:
    selections: dict[str, CaseSelection] = {}
    used_events: set[str] = set()

    # Representative preservation: prefer exactly one tail target so the point
    # also appears in the accuracy-risk selection landscape.
    event_id, repeat = manual.get("representative", (None, None))
    if event_id is not None:
        representative = choose_manual(stats, event_id, repeat)
    else:
        candidates = stats.loc[stats["n_tail_targets"].eq(1)].copy()
        if candidates.empty:
            candidates = stats.loc[stats["n_tail_targets"].le(1)].copy()
        if candidates.empty:
            candidates = stats.copy()
        median_error = float(candidates["base_overall_mae"].median())
        candidates["distance_to_median_base_error"] = np.abs(
            candidates["base_overall_mae"] - median_error
        )
        candidates["absolute_overall_change"] = np.abs(candidates["overall_mae_change"])
        representative = candidates.sort_values(
            ["distance_to_median_base_error", "absolute_overall_change", "base_overall_mae"],
            ascending=[True, True, True],
        ).iloc[0]
    selections["representative"] = CaseSelection("representative", representative)
    used_events.add(str(representative["event_id"]))

    # Successful correction.
    event_id, repeat = manual.get("successful_correction", (None, None))
    if event_id is not None:
        success = choose_manual(stats, event_id, repeat)
    else:
        candidates = stats.loc[
            stats["n_tail_targets"].ge(2)
            & stats["tail_mae_reduction"].notna()
            & stats["tail_mae_reduction"].gt(0)
            & stats["non_tail_mae_change"].le(float(success_non_tail_tolerance))
        ].copy()
        candidates = exclude_events(candidates, used_events)
        if candidates.empty:
            candidates = stats.loc[
                stats["n_tail_targets"].ge(2)
                & stats["tail_mae_reduction"].notna()
                & stats["tail_mae_reduction"].gt(0)
            ].copy()
            candidates = exclude_events(candidates, used_events)
        if candidates.empty:
            raise RuntimeError("No successful high-motion correction candidate was found.")
        success = candidates.sort_values(
            ["tail_mae_reduction", "caurc_tail_u05", "n_tail_targets", "base_tail_mae"],
            ascending=[False, True, False, False],
        ).iloc[0]
    selections["successful_correction"] = CaseSelection("successful_correction", success)
    used_events.add(str(success["event_id"]))

    # Residual failure: severe underprediction remains after correction.
    event_id, repeat = manual.get("residual_failure", (None, None))
    if event_id is not None:
        failure = choose_manual(stats, event_id, repeat)
    else:
        candidates = stats.loc[
            stats["n_tail_targets"].ge(2)
            & stats["remaining_tail_u05_count"].gt(0)
            & stats["caurc_tail_mae"].notna()
        ].copy()
        candidates = exclude_events(candidates, used_events)
        if candidates.empty:
            candidates = stats.loc[
                stats["n_tail_targets"].ge(1)
                & stats["caurc_tail_mae"].notna()
            ].copy()
            candidates = exclude_events(candidates, used_events)
        if candidates.empty:
            raise RuntimeError("No residual-failure candidate was found.")
        failure = candidates.sort_values(
            ["caurc_tail_mae", "caurc_tail_u05", "remaining_tail_u05_count", "tail_mae_reduction"],
            ascending=[False, False, False, True],
        ).iloc[0]
    selections["residual_failure"] = CaseSelection("residual_failure", failure)
    used_events.add(str(failure["event_id"]))

    # Supplementary overcorrection case.
    event_id, repeat = manual.get("overcorrection", (None, None))
    if event_id is not None:
        over = choose_manual(stats, event_id, repeat)
    else:
        candidates = stats.loc[stats["non_tail_mae_change"].notna()].copy()
        candidates = exclude_events(candidates, used_events)
        over = candidates.sort_values(
            ["non_tail_mae_change", "caurc_non_tail_o05", "max_applied_correction"],
            ascending=[False, False, False],
        ).iloc[0]
    selections["overcorrection"] = CaseSelection("overcorrection", over)

    return selections


def failure_mode_summary(stats: pd.DataFrame) -> pd.DataFrame:
    tail_units = stats.loc[stats["n_tail_values"].gt(0)].copy()
    rows = [
        {
            "population": "all event-repeat units",
            "criterion": "number of units",
            "count": int(len(stats)),
            "fraction": 1.0,
        },
        {
            "population": "units with >=1 tail value",
            "criterion": "number of units",
            "count": int(len(tail_units)),
            "fraction": float(len(tail_units) / max(len(stats), 1)),
        },
        {
            "population": "units with >=1 tail value",
            "criterion": "CA-URC reduces combined tail MAE",
            "count": int(tail_units["tail_mae_reduction"].gt(0).sum()),
            "fraction": safe_fraction(tail_units["tail_mae_reduction"].gt(0).to_numpy()),
        },
        {
            "population": "units with >=1 tail value",
            "criterion": "CA-URC does not reduce combined tail MAE",
            "count": int(tail_units["tail_mae_reduction"].le(0).sum()),
            "fraction": safe_fraction(tail_units["tail_mae_reduction"].le(0).to_numpy()),
        },
        {
            "population": "units with >=1 tail value",
            "criterion": "at least one severe tail underprediction remains",
            "count": int(tail_units["remaining_tail_u05_count"].gt(0).sum()),
            "fraction": safe_fraction(tail_units["remaining_tail_u05_count"].gt(0).to_numpy()),
        },
        {
            "population": "all event-repeat units",
            "criterion": "non-tail MAE increases",
            "count": int(stats["non_tail_mae_change"].gt(0).sum()),
            "fraction": safe_fraction(stats["non_tail_mae_change"].gt(0).to_numpy()),
        },
        {
            "population": "all event-repeat units",
            "criterion": "non-tail MAE increases by >0.02 log10 units",
            "count": int(stats["non_tail_mae_change"].gt(0.02).sum()),
            "fraction": safe_fraction(stats["non_tail_mae_change"].gt(0.02).to_numpy()),
        },
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Frozen model reconstruction and all-held-out inference
# -----------------------------------------------------------------------------


def resolve_h5_path(event_id: str, manifest_row: pd.Series, h5_root: Path | None) -> Path:
    raw = str(manifest_row.get("h5_path", "")).strip()
    if raw and raw.lower() not in {"nan", "none"}:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
    if h5_root is not None:
        candidates = [
            h5_root / f"{event_id}.h5",
            h5_root / str(event_id) / f"{event_id}.h5",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Cannot resolve HDF5 for event {event_id}; manifest h5_path={raw!r}, h5_root={h5_root}"
    )


def reconstruct_final_model(
    baseline_module_path: Path,
    ablation_module_path: Path,
    base_checkpoint_path: Path,
    head_checkpoint_path: Path,
    selection_json_path: Path,
    device_name: str,
    hidden_dim: int,
    attention_heads: int,
    risk_hidden: int,
    maximum_correction: float,
):
    import torch

    baseline_module = load_module(baseline_module_path, "nc27_baseline_module")
    ablation_module = load_module(ablation_module_path, "nc27_ablation_module")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device(device_name)

    base_checkpoint = load_checkpoint(base_checkpoint_path, device)
    head_checkpoint = load_checkpoint(head_checkpoint_path, device)
    selection = read_json(selection_json_path)

    if str(selection.get("variant", "")) != "A4_under_only":
        raise RuntimeError("Selection JSON is not the final A4_under_only CA-URC model.")
    gamma = float(selection.get("selected_gamma", np.nan))
    selected_epoch = int(selection.get("selected_epoch", -1))
    if not np.isclose(gamma, 5.0):
        raise RuntimeError(f"Expected the locked CA-URC gamma=5, found {gamma}.")
    if selected_epoch != 3:
        raise RuntimeError(f"Expected the locked CA-URC epoch=3, found {selected_epoch}.")
    if str(head_checkpoint.get("variant", "")) != "A4_under_only":
        raise RuntimeError("Head checkpoint is not A4_under_only.")
    if int(head_checkpoint.get("epoch", -1)) != selected_epoch:
        raise RuntimeError("Head-checkpoint epoch does not match selection JSON.")

    head_args = head_checkpoint.get("args", {}) or {}
    hidden_dim = int(head_args.get("hidden_dim", hidden_dim))
    attention_heads = int(head_args.get("attention_heads", attention_heads))
    risk_hidden = int(head_args.get("risk_hidden", risk_hidden))
    maximum_correction = float(head_args.get("maximum_correction", maximum_correction))

    base = ablation_module.make_frozen_base(
        baseline_module,
        base_checkpoint,
        hidden_dim,
        attention_heads,
        device,
    )
    model = ablation_module.AblationRiskGate(
        base=base,
        variant="A4_under_only",
        hidden_dim=hidden_dim,
        risk_hidden=risk_hidden,
        maximum_correction=maximum_correction,
        tail_prevalence=np.asarray([0.1, 0.1]),
        under_prevalence=np.asarray([0.1, 0.1]),
    ).to(device)
    model.load_state_dict(head_checkpoint["model_state"], strict=True)
    model.eval()

    model_audit = {
        "device": str(device),
        "variant": "A4_under_only",
        "selected_epoch": selected_epoch,
        "selected_gamma": gamma,
        "hidden_dim": hidden_dim,
        "attention_heads": attention_heads,
        "risk_hidden": risk_hidden,
        "maximum_correction": maximum_correction,
        "base_checkpoint_variant": str(base_checkpoint.get("variant", "cross_attention")),
        "head_checkpoint_variant": str(head_checkpoint.get("variant", "")),
    }
    return baseline_module, ablation_module, model, gamma, device, model_audit


def prepare_model_arrays(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    coordinates: np.ndarray,
    p_offset: np.ndarray,
    input_indices: np.ndarray,
    target_indices: np.ndarray,
    sampling_rate: float,
    pre_first_p: float,
    time_zero_index: int,
    t0_sec: float,
    input_pre_sec: float,
) -> dict[str, np.ndarray | int]:
    input_start = max(
        0,
        int(round((pre_first_p - input_pre_sec) * sampling_rate)),
    )
    snapshot = min(
        time_zero_index + int(round(t0_sec * sampling_rate)),
        acceleration.shape[-1] - 1,
    )

    observed_acceleration = acceleration[input_indices, :, input_start:snapshot]
    input_waveforms = (
        np.sign(observed_acceleration)
        * np.log1p(np.abs(observed_acceleration) / 1e-3)
    ).astype(np.float32)

    input_coordinates = coordinates[input_indices]
    target_coordinates = coordinates[target_indices]
    origin_latitude = float(input_coordinates[:, 0].mean())
    origin_longitude = float(input_coordinates[:, 1].mean())
    input_xy = local_xy_km(input_coordinates, origin_latitude, origin_longitude)
    target_xy = local_xy_km(target_coordinates, origin_latitude, origin_longitude)

    input_features = np.concatenate(
        [
            input_xy / 100.0,
            input_coordinates[:, 2:3] / 2000.0,
            p_offset[input_indices, None] / max(float(t0_sec), 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    target_features = np.concatenate(
        [target_xy / 100.0, target_coordinates[:, 2:3] / 2000.0],
        axis=1,
    ).astype(np.float32)

    future_acceleration = acceleration[target_indices, :, snapshot:]
    future_velocity = velocity[target_indices, :, snapshot:]
    future_pga = np.max(
        np.sqrt(future_acceleration[:, 0, :] ** 2 + future_acceleration[:, 1, :] ** 2),
        axis=1,
    )
    future_pgv = np.max(
        np.sqrt(future_velocity[:, 0, :] ** 2 + future_velocity[:, 1, :] ** 2),
        axis=1,
    )
    truth = np.column_stack(
        [
            np.log10(np.maximum(future_pga, 1e-10)),
            np.log10(np.maximum(future_pgv, 1e-12)),
        ]
    ).astype(np.float32)

    return {
        "input_waveforms": input_waveforms,
        "input_features": input_features,
        "target_features": target_features,
        "truth": truth,
        "snapshot": snapshot,
    }


def predict_targets(model, model_arrays: dict[str, Any], gamma: float, device) -> dict[str, np.ndarray]:
    import torch

    with torch.inference_mode():
        output = model(
            torch.from_numpy(model_arrays["input_waveforms"])[None].to(device),
            torch.from_numpy(model_arrays["input_features"])[None].to(device),
            torch.from_numpy(model_arrays["target_features"])[None].to(device),
        )
        correction = model.correction(output, gamma=float(gamma))
        base = output["base"]
        final = base + correction

    return {
        "base": base[0].detach().cpu().numpy(),
        "final": final[0].detach().cpu().numpy(),
        "correction": correction[0].detach().cpu().numpy(),
        "under_probability": output["under_probability"][0].detach().cpu().numpy(),
        "raw_correction_amplitude": output["residual"][0].detach().cpu().numpy(),
    }



def predict_targets_individually(
    model,
    acceleration: np.ndarray,
    velocity: np.ndarray,
    coordinates: np.ndarray,
    p_offset: np.ndarray,
    input_indices: np.ndarray,
    target_indices: np.ndarray,
    sampling_rate: float,
    pre_first_p: float,
    time_zero_index: int,
    t0_sec: float,
    input_pre_sec: float,
    gamma: float,
    device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Query one target at a time and concatenate the outputs.

    This removes any dependence on target-query batch shape from the descriptive
    all-held-out maps.  It is slower than one large query but the selected case
    studies contain only a few hundred station queries in total.
    """
    outputs = {
        "base": [],
        "final": [],
        "correction": [],
        "under_probability": [],
        "raw_correction_amplitude": [],
    }
    truths = []

    for target_index in np.asarray(target_indices, dtype=int):
        arrays = prepare_model_arrays(
            acceleration,
            velocity,
            coordinates,
            p_offset,
            input_indices,
            np.asarray([target_index], dtype=int),
            sampling_rate,
            pre_first_p,
            time_zero_index,
            t0_sec,
            input_pre_sec,
        )
        pred = predict_targets(model, arrays, gamma, device)

        truths.append(np.asarray(arrays["truth"][0], dtype=np.float64))
        for key in outputs:
            outputs[key].append(np.asarray(pred[key][0], dtype=np.float64))

    merged = {
        key: np.stack(value, axis=0)
        for key, value in outputs.items()
    }
    truth = np.stack(truths, axis=0)
    return merged, truth


def case_metrics(truth: np.ndarray, prediction: np.ndarray, tail: np.ndarray) -> dict[str, float | int]:
    residual = prediction - truth
    absolute = np.abs(residual)
    output: dict[str, float | int] = {}
    for index, quantity in enumerate(["pga", "pgv"]):
        mask = tail[:, index]
        output[f"{quantity}_mae"] = safe_mean(absolute[:, index])
        output[f"{quantity}_bias"] = safe_mean(residual[:, index])
        output[f"{quantity}_u05"] = safe_fraction(residual[:, index] <= SEVERE_UNDER)
        output[f"{quantity}_tail_n"] = int(mask.sum())
        output[f"{quantity}_tail_mae"] = safe_mean(absolute[mask, index])
        output[f"{quantity}_tail_bias"] = safe_mean(residual[mask, index])
        output[f"{quantity}_tail_u05"] = safe_fraction(residual[mask, index] <= SEVERE_UNDER)
    return output


def infer_all_heldout_case(
    selection: CaseSelection,
    locked_predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    h5_root: Path | None,
    baseline_module,
    model,
    gamma: float,
    device,
    thresholds: np.ndarray,
    split_label: str,
    seed: int,
    t0_sec: float,
    input_stations: int,
    target_stations: int,
    input_pre_sec: float,
    audit_tolerance: float,
    query_tolerance: float,
) -> dict[str, Any]:
    import h5py

    event_id = str(selection.row["event_id"])
    repeat = int(selection.row["repeat"])
    if event_id not in manifest.index:
        raise KeyError(f"Selected event {event_id} is absent from the manifest.")
    manifest_row = manifest.loc[event_id]
    h5_path = resolve_h5_path(event_id, manifest_row, h5_root)

    with h5py.File(h5_path, "r") as h5:
        acceleration = np.asarray(h5["acceleration"][:], dtype=np.float32)
        velocity = np.asarray(h5["velocity"][:], dtype=np.float32)
        coordinates = np.asarray(h5["station_coords"][:], dtype=np.float32)
        p_offset = np.asarray(h5["p_offset_sec"][:], dtype=np.float32)
        station_ids = (
            h5["station_id"].asstr()[:]
            if "station_id" in h5
            else np.asarray([str(i) for i in range(len(coordinates))], dtype=object)
        )
        sampling_rate = float(h5.attrs["sampling_rate_hz"])
        pre_first_p = float(h5.attrs["pre_first_p_sec"])
        time_zero_index = int(h5.attrs["time_zero_index"])

    triggered = np.flatnonzero(
        np.isfinite(p_offset)
        & (p_offset >= -1e-3)
        & (p_offset <= float(t0_sec))
    )
    if len(triggered) < input_stations:
        raise RuntimeError(f"Event {event_id} has only {len(triggered)} eligible input stations.")

    random_seed = baseline_module.stable_seed(
        f"strong:{split_label}:{event_id}:repeat:{repeat}",
        int(seed),
    )
    rng = np.random.default_rng(random_seed)
    input_indices = rng.choice(triggered, size=int(input_stations), replace=False)
    target_pool = np.setdiff1d(np.arange(len(coordinates)), input_indices, assume_unique=False)
    locked_target_indices = rng.choice(target_pool, size=int(target_stations), replace=False)

    locked_group = locked_predictions.loc[
        locked_predictions["event_id"].eq(event_id)
        & locked_predictions["repeat"].eq(repeat)
    ].sort_values("target_slot", kind="stable")
    if len(locked_group) != target_stations:
        raise RuntimeError(
            f"Selected event-repeat {event_id}/{repeat} contains {len(locked_group)} locked rows; "
            f"expected {target_stations}."
        )
    stored_targets = locked_group["target_station_index"].to_numpy(dtype=int)
    if not np.array_equal(locked_target_indices.astype(int), stored_targets):
        raise RuntimeError(
            "\nLocked-test station draw reproduction failed.\n"
            f"Event       : {event_id}\n"
            f"Repeat      : {repeat}\n"
            f"Random seed : {int(random_seed)}\n"
            f"Input idx   : {input_indices.astype(int).tolist()}\n"
            f"Generated   : {locked_target_indices.astype(int).tolist()}\n"
            f"Stored      : {stored_targets.tolist()}\n"
            "The script intentionally stops here rather than plotting a "
            "case that is not exactly paired with the locked manuscript "
            "predictions."
        )

    # Exact locked batch audit.
    locked_arrays = prepare_model_arrays(
        acceleration,
        velocity,
        coordinates,
        p_offset,
        input_indices,
        locked_target_indices,
        sampling_rate,
        pre_first_p,
        time_zero_index,
        t0_sec,
        input_pre_sec,
    )
    locked_output = predict_targets(model, locked_arrays, gamma, device)
    stored_truth = np.column_stack(
        [
            locked_group[TRUTH_COLUMNS["pga"]].to_numpy(float),
            locked_group[TRUTH_COLUMNS["pgv"]].to_numpy(float),
        ]
    )
    stored_base = np.column_stack(
        [
            locked_group[BASE_COLUMNS["pga"]].to_numpy(float),
            locked_group[BASE_COLUMNS["pgv"]].to_numpy(float),
        ]
    )
    stored_final = np.column_stack(
        [
            locked_group[FINAL_COLUMNS["pga"]].to_numpy(float),
            locked_group[FINAL_COLUMNS["pgv"]].to_numpy(float),
        ]
    )
    audit = {
        "max_abs_truth_diff": float(np.max(np.abs(locked_arrays["truth"] - stored_truth))),
        "max_abs_base_diff": float(np.max(np.abs(locked_output["base"] - stored_base))),
        "max_abs_final_diff": float(np.max(np.abs(locked_output["final"] - stored_final))),
    }
    # The archived locked predictions may have been generated on a different
    # GPU / CUDA / PyTorch backend. Attention kernels can therefore differ by
    # several 1e-5 in float32 even when checkpoints, station draws, targets and
    # labels are identical.  Use a strict truth tolerance, a small cross-backend
    # reproduction tolerance for model outputs, and a separate hard ceiling.
    truth_tolerance = min(float(audit_tolerance), 2e-6)
    prediction_tolerance = float(audit_tolerance)
    hard_prediction_ceiling = 1e-3

    if audit["max_abs_truth_diff"] > truth_tolerance:
        raise RuntimeError(
            "Truth reconstruction failed for the locked targets: "
            f"{audit}; truth_tolerance={truth_tolerance}. "
            "This indicates a data/window mismatch and must not be ignored."
        )

    if (
        audit["max_abs_base_diff"] > hard_prediction_ceiling
        or audit["max_abs_final_diff"] > hard_prediction_ceiling
    ):
        raise RuntimeError(
            "Frozen-model reproduction differs too much from the locked archive: "
            f"{audit}; hard_prediction_ceiling={hard_prediction_ceiling}. "
            "This is too large to attribute safely to floating-point kernel differences."
        )

    archive_drift = max(
        audit["max_abs_base_diff"],
        audit["max_abs_final_diff"],
    )
    audit["archive_reproduction_within_soft_tolerance"] = bool(
        archive_drift <= prediction_tolerance
    )
    audit["archive_reproduction_status"] = (
        "pass"
        if archive_drift <= prediction_tolerance
        else "warning_only_within_hard_ceiling"
    )

    if archive_drift > prediction_tolerance:
        print(
            "WARNING: locked archive and freshly reconstructed frozen model differ "
            f"by {archive_drift:.3e} log10 units, above the soft tolerance "
            f"{prediction_tolerance:.3e} but below the hard ceiling "
            f"{hard_prediction_ceiling:.3e}. Exact station draws and truth labels "
            "have been reproduced; proceeding with the freshly loaded frozen model "
            "for descriptive spatial maps."
        )

    audit["truth_tolerance"] = float(truth_tolerance)
    audit["prediction_tolerance"] = float(prediction_tolerance)
    audit["hard_prediction_ceiling"] = float(hard_prediction_ceiling)

    # Query-independence audit: the same target should retain its output when
    # queried alone. If this fails, all-held-out maps would not be uniquely
    # defined and the script stops instead of presenting a misleading surface.
    single_diffs: list[float] = []
    for slot, target_index in enumerate(locked_target_indices):
        one_arrays = prepare_model_arrays(
            acceleration,
            velocity,
            coordinates,
            p_offset,
            input_indices,
            np.asarray([target_index], dtype=int),
            sampling_rate,
            pre_first_p,
            time_zero_index,
            t0_sec,
            input_pre_sec,
        )
        one_output = predict_targets(model, one_arrays, gamma, device)
        single_diffs.extend(
            [
                float(np.max(np.abs(one_output["base"][0] - locked_output["base"][slot]))),
                float(np.max(np.abs(one_output["final"][0] - locked_output["final"][slot]))),
            ]
        )
    audit["max_query_set_dependence"] = float(max(single_diffs) if single_diffs else 0.0)
    audit["query_tolerance"] = float(query_tolerance)
    query_hard_ceiling = 1e-3
    audit["query_hard_ceiling"] = float(query_hard_ceiling)

    if audit["max_query_set_dependence"] > query_hard_ceiling:
        raise RuntimeError(
            "Target predictions change too much when query batch shape changes. "
            "All-held-out visualization is not safe: "
            f"Maximum difference={audit['max_query_set_dependence']:.3e}; "
            f"hard ceiling={query_hard_ceiling:.3e}."
        )

    if audit["max_query_set_dependence"] > float(query_tolerance):
        print(
            "WARNING: small query-batch numerical drift detected: "
            f"{audit['max_query_set_dependence']:.3e} > "
            f"{float(query_tolerance):.3e}. This remains below the "
            f"{query_hard_ceiling:.3e} hard ceiling. "
            "All-held-out maps will therefore use target-by-target inference."
        )

    # Frozen all-held-out query for descriptive spatial visualization.
    # Query one target at a time so every mapped station has an unambiguous,
    # batch-composition-independent prediction.
    all_target_indices = target_pool.astype(int)
    all_output, truth = predict_targets_individually(
        model,
        acceleration,
        velocity,
        coordinates,
        p_offset,
        input_indices,
        all_target_indices,
        sampling_rate,
        pre_first_p,
        time_zero_index,
        t0_sec,
        input_pre_sec,
        gamma,
        device,
    )
    tail = truth >= thresholds[None, :]

    epicenter_lat, epicenter_lon = detect_epicenter(manifest_row)
    if epicenter_lat is None or epicenter_lon is None:
        plot_origin_lat = float(coordinates[:, 0].mean())
        plot_origin_lon = float(coordinates[:, 1].mean())
        epicenter_xy = None
    else:
        plot_origin_lat = epicenter_lat
        plot_origin_lon = epicenter_lon
        epicenter_xy = np.asarray([0.0, 0.0])
    plot_xy = local_xy_km(coordinates, plot_origin_lat, plot_origin_lon)

    base_metrics = case_metrics(truth, all_output["base"], tail)
    final_metrics = case_metrics(truth, all_output["final"], tail)

    audit.update(
        {
            "event_id": event_id,
            "repeat": repeat,
            "random_seed": int(random_seed),
            "n_stations": int(len(coordinates)),
            "n_triggered_by_t0": int(len(triggered)),
            "n_input_stations": int(len(input_indices)),
            "n_all_heldout_stations": int(len(all_target_indices)),
            "h5_path": str(h5_path.resolve()),
        }
    )

    return {
        "case_name": selection.name,
        "event_id": event_id,
        "repeat": repeat,
        "magnitude": detect_magnitude(manifest_row),
        "input_indices": input_indices,
        "target_indices": all_target_indices,
        "locked_target_indices": locked_target_indices,
        "station_ids": np.asarray(station_ids),
        "xy_input": plot_xy[input_indices],
        "xy_target": plot_xy[all_target_indices],
        "epicenter_xy": epicenter_xy,
        "truth": truth,
        "base": all_output["base"],
        "final": all_output["final"],
        "correction": all_output["correction"],
        "under_probability": all_output["under_probability"],
        "raw_correction_amplitude": all_output["raw_correction_amplitude"],
        "tail": tail,
        "p_wave_reached": np.isfinite(p_offset[all_target_indices])
        & (p_offset[all_target_indices] <= t0_sec),
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "locked_selection": selection.row.to_dict(),
        "audit": audit,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def setup_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],

            # JGR-SE final figure readability
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,

            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,

            "legend.fontsize": 7.8,

            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,

            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax, grid: bool = True) -> None:
    """Restrained full-box axis styling used across the main figure."""
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.68)
        ax.spines[side].set_color("0.38")
    if grid:
        ax.grid(color="0.90", linewidth=0.45, alpha=0.80, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", top=False, right=False, length=2.5, width=0.65)


def selected_marker_style(name: str) -> dict[str, Any]:
    return {
        "representative": {"marker": "o", "s": 60, "facecolor": "#4C78A8"},
        "successful_correction": {"marker": "*", "s": 115, "facecolor": "#009E73"},
        "residual_failure": {"marker": "X", "s": 72, "facecolor": "#D55E00"},
        "overcorrection": {"marker": "D", "s": 55, "facecolor": "#CC79A7"},
    }[name]


def plot_selection_landscape(ax, stats: pd.DataFrame, selections: dict[str, CaseSelection]) -> None:
    """Panel a: case-selection landscape with restrained annotations.

    The three main-text cases are labelled directly.  The supplementary
    non-tail-overcorrection case remains visible and in the legend, but its
    event identifier is intentionally not annotated to reduce visual clutter.
    """
    plot = stats.loc[stats["n_tail_values"].gt(0)].copy()
    ax.scatter(
        plot["non_tail_mae_change"],
        plot["tail_mae_reduction"],
        s=10,
        facecolors="#AEB4BA",
        edgecolors="none",
        alpha=0.30,
        rasterized=True,
        zorder=2,
    )
    ax.axvline(0.0, color="0.28", linestyle=(0, (4, 2.5)), linewidth=0.78, zorder=1)
    ax.axhline(0.0, color="0.28", linestyle=(0, (4, 2.5)), linewidth=0.78, zorder=1)

    main_cases = {"representative", "successful_correction", "residual_failure"}
    for name, selection in selections.items():
        row = selection.row
        if not np.isfinite(row.get("tail_mae_reduction", np.nan)):
            continue
        marker_style = selected_marker_style(name)
        ax.scatter(
            [row["non_tail_mae_change"]],
            [row["tail_mae_reduction"]],
            marker=marker_style["marker"],
            s=marker_style["s"],
            facecolor=marker_style["facecolor"],
            edgecolor="black",
            linewidth=0.70,
            zorder=6,
            label=CASE_LABELS[name],
        )

        # Keep direct event labels only for the three main-text cases.
        if name in main_cases:
            median_x = np.nanmedian(plot["non_tail_mae_change"])
            dx = 7 if row["non_tail_mae_change"] <= median_x else -7
            ha = "left" if dx > 0 else "right"
            short_label = {
                "representative": "Representative preservation",
                "successful_correction": "Successful tail correction",
                "residual_failure": "Residual failure",
            }[name]
            ax.annotate(
                f"{short_label}\n{row['event_id']} / r{int(row['repeat'])}",
                xy=(row["non_tail_mae_change"], row["tail_mae_reduction"]),
                xytext=(dx, 6),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=7.2,
                color="#20252A",
                linespacing=1.05,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.25),
                zorder=7,
            )

    style_axis(ax, grid=True)
    ax.set_xlabel(
        "Non-tail MAE change (CA-URC − Cross-Attention Base; log$_{10}$ units)",
        labelpad=2.0,
    )
    ax.set_ylabel(
        "Tail MAE reduction\n(Cross-Attention Base − CA-URC; log$_{10}$ units)"
    )
    ax.text(
        0.01,
        0.98,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        0.985,
        0.96,
        "Higher tail-MAE reduction",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="0.38",
    )
    ax.text(
        0.985,
        0.055,
        f"{len(plot):,} event–repeat units with ≥1 tail value",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="0.42",
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Preserve the scientific narrative order in the legend.
        desired = [
            "Representative preservation",
            "Successful tail correction",
            "Residual failure",
            "Non-tail overcorrection",
        ]
        lookup = dict(zip(labels, handles))
        ordered_handles = [lookup[label] for label in desired if label in lookup]
        ordered_labels = [label for label in desired if label in lookup]
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="lower left",
            ncol=2,
            frameon=False,
            handletextpad=0.45,
            columnspacing=1.0,
            borderaxespad=0.25,
            fontsize=6.2,
        )


def map_extent(case: dict[str, Any]) -> tuple[float, float, float, float]:
    xy = np.vstack([case["xy_target"], case["xy_input"]])
    if case["epicenter_xy"] is not None:
        xy = np.vstack([xy, case["epicenter_xy"][None, :]])
    x_min, y_min = np.nanmin(xy, axis=0)
    x_max, y_max = np.nanmax(xy, axis=0)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    pad = 0.08 * span
    return x_min - pad, x_max + pad, y_min - pad, y_max + pad


def metric_annotation(case: dict[str, Any], quantity: str, model_name: str) -> str:

    metrics = (
        case["base_metrics"]
        if model_name == "base"
        else case["final_metrics"]
    )

    tail_mae = metrics[f"{quantity}_tail_mae"]
    tail_u05 = metrics[f"{quantity}_tail_u05"]

    tail_text = (
        "NA"
        if not np.isfinite(tail_mae)
        else f"{tail_mae:.3f}"
    )

    u_text = (
        "NA"
        if not np.isfinite(tail_u05)
        else f"{100*tail_u05:.1f}%"
    )

    tail_n = metrics[f"{quantity}_tail_n"]

    return (
        f"MAE = {metrics[f'{quantity}_mae']:.3f}\n"
        f"Tail MAE = {tail_text}\n"
        f"Tail U$_{{0.5}}$ = {u_text}\n"
        f"n$_{{tail}}$ = {tail_n}"
    )


def plot_residual_map(
    ax,
    case: dict[str, Any],
    quantity_index: int,
    model_name: str,
    norm,
    cmap: str,
    show_legend: bool,
) -> None:
    quantity = "pga" if quantity_index == 0 else "pgv"
    prediction = case["base"] if model_name == "base" else case["final"]
    residual = prediction[:, quantity_index] - case["truth"][:, quantity_index]
    tail = case["tail"][:, quantity_index]
    xy = case["xy_target"]

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=residual,
        cmap=cmap,
        norm=norm,
        s=np.where(tail, 31.0, 16.0),
        edgecolors=np.where(tail, "black", "white"),
        linewidths=np.where(tail, 0.85, 0.35),
        alpha=0.92,
        zorder=4,
        rasterized=True,
    )
    ax.scatter(
        case["xy_input"][:, 0],
        case["xy_input"][:, 1],
        marker="^",
        s=27,
        facecolor="white",
        edgecolor="black",
        linewidth=0.7,
        zorder=7,
        label="Input station",
    )
    if case["epicenter_xy"] is not None:
        ax.scatter(
            [case["epicenter_xy"][0]],
            [case["epicenter_xy"][1]],
            marker="*",
            s=68,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
            label="Epicentre",
        )

    x0, x1, y0, y1 = map_extent(case)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.91", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("0.45")
    ax.tick_params(length=2.2, pad=1.3)
    ax.text(
        0.03,
        0.97,
        metric_annotation(case, quantity, model_name),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        zorder=10,
    )
    if show_legend:
        # Proxy legend for held-out and high-motion targets.
        ax.scatter([], [], s=16, facecolor="0.7", edgecolor="white", linewidth=0.35, label="Non-input target")
        ax.scatter([], [], s=31, facecolor="0.7", edgecolor="black", linewidth=0.85, label="High-motion-tail target")
        ax.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.88,
            borderpad=0.45,
            handletextpad=0.45,
            labelspacing=0.35,
            fontsize=7.2,
        )


def plot_main_figure(
    stats: pd.DataFrame,
    selections: dict[str, CaseSelection],
    cases: dict[str, dict[str, Any]],
    out_dir: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm

    setup_matplotlib()
    figure = plt.figure(figsize=(7.20, 9.05))
    outer_grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.10, 3.25],
        hspace=0.10,
        left=0.105,
        right=0.985,
        bottom=0.095,
        top=0.970,
    )

    selection_axis = figure.add_subplot(outer_grid[0, 0])
    plot_selection_landscape(selection_axis, stats, selections)

    map_grid = outer_grid[1, 0].subgridspec(
        6,
        4,
        height_ratios=[0.14, 0.6, 0.08, 0.6, 0.05, 0.6],
        hspace=0.10,
        wspace=0.15,
    )

    residual_values = []
    for case_name in CASE_ORDER:
        case = cases[case_name]
        residual_values.append(case["base"] - case["truth"])
        residual_values.append(case["final"] - case["truth"])
    residual_values = np.concatenate([x.ravel() for x in residual_values])
    residual_values = residual_values[np.isfinite(residual_values)]
    limit = float(np.percentile(np.abs(residual_values), 98)) if len(residual_values) else 1.0
    limit = float(np.clip(max(limit, 0.6), 0.6, 1.5))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = "RdBu_r"

    titles = [
        "PGA — Cross-Attention Base",
        "PGA — CA-URC",
        "PGV — Cross-Attention Base",
        "PGV — CA-URC",
    ]
    panel_letters = {"representative": "b", "successful_correction": "c", "residual_failure": "d"}

    for case_index, case_name in enumerate(CASE_ORDER):
        case = cases[case_name]
        header_row = 2 * case_index
        map_row = header_row + 1
        header_axis = figure.add_subplot(map_grid[header_row, :])
        header_axis.axis("off")
        axes = [figure.add_subplot(map_grid[map_row, column]) for column in range(4)]
        for column, axis in enumerate(axes):
            quantity_index = 0 if column < 2 else 1
            model_name = "base" if column % 2 == 0 else "final"
            plot_residual_map(
                axis,
                case,
                quantity_index,
                model_name,
                norm,
                cmap,
                show_legend=(case_index == 0 and column == 0),
            )
            if column == 0:
                axis.set_ylabel("North–south distance (km)", fontsize=6.5, labelpad=2.5)
            else:
                axis.set_yticklabels([])
            if case_index == len(CASE_ORDER) - 1:
                axis.set_xlabel("East–west distance (km)")
            else:
                axis.set_xticklabels([])

        magnitude = (
            f"M{case['magnitude']:.1f}" if np.isfinite(case["magnitude"]) else "M unknown"
        )
        # b 保持原来的位置；c、d 下移，使标题更靠近对应地图
        # b / c / d 标题纵向位置分别控制
        if case_index == 0:  # b
            title_y = 0.78
        elif case_index == 1:  # c
            title_y = 0.05  # c 标题下移
        else:  # d
            title_y = 0.50

        header_axis.text(
            0.0,
            title_y,
            (
                f"{panel_letters[case_name]}  {CASE_LABELS[case_name]} — "
                f"Event {case['event_id']}, {magnitude}, r{case['repeat']}"
            ),
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#20252A",
        )
        if case_index == 0:
            for column, title in enumerate(titles):
                header_axis.text(
                    (column + 0.5) / 4.0,
                    0.02,
                    title,
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                    fontweight="bold",
                )

    colorbar_axis = figure.add_axes([0.285, 0.035, 0.50, 0.016])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Residual = prediction − observation (log$_{10}$ units)", labelpad=2.0)
    colorbar.ax.tick_params(labelsize=6.8, length=2.2)
    for extension in ["png", "pdf", "svg"]:
        path = out_dir / f"Fig8_spatial_cases.{extension}"
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(figure)


def plot_risk_correction_supplement(
    cases: dict[str, dict[str, Any]],
    out_dir: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    setup_matplotlib()
    names = ["successful_correction", "residual_failure", "overcorrection"]
    figure, axes = plt.subplots(3, 4, figsize=(7.20, 6.25), squeeze=False)
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.94, wspace=0.12, hspace=0.07)

    risk_norm = Normalize(0.0, 1.0)
    correction_max = max(
        0.1,
        float(
            np.percentile(
                np.concatenate([cases[name]["correction"].ravel() for name in names]),
                99,
            )
        ),
    )
    correction_norm = Normalize(0.0, correction_max)

    column_titles = [
        "PGA underprediction-risk score",
        "PGA applied correction",
        "PGV underprediction-risk score",
        "PGV applied correction",
    ]

    for row, name in enumerate(names):
        case = cases[name]
        for column in range(4):
            ax = axes[row, column]
            quantity_index = 0 if column < 2 else 1
            is_risk = column % 2 == 0
            values = (
                case["under_probability"][:, quantity_index]
                if is_risk
                else case["correction"][:, quantity_index]
            )
            norm = risk_norm if is_risk else correction_norm
            cmap = "viridis" if is_risk else "magma"
            tail = case["tail"][:, quantity_index]
            xy = case["xy_target"]
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=values,
                cmap=cmap,
                norm=norm,
                s=np.where(tail, 30.0, 15.0),
                edgecolors=np.where(tail, "black", "white"),
                linewidths=np.where(tail, 0.8, 0.3),
                alpha=0.94,
                rasterized=True,
            )
            ax.scatter(
                case["xy_input"][:, 0],
                case["xy_input"][:, 1],
                marker="^",
                s=24,
                facecolor="white",
                edgecolor="black",
                linewidth=0.65,
                zorder=5,
            )
            if case["epicenter_xy"] is not None:
                ax.scatter(
                    [case["epicenter_xy"][0]],
                    [case["epicenter_xy"][1]],
                    marker="*",
                    s=58,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=6,
                )
            x0, x1, y0, y1 = map_extent(case)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="0.91", linewidth=0.4)
            ax.tick_params(length=2.0, pad=1.0)
            if row == 0:
                ax.set_title(
                    column_titles[column],
                    fontsize=8.5,
                    fontweight="bold",
                    pad=5
                )
            if column == 0:
                magnitude = f"M{case['magnitude']:.1f}" if np.isfinite(case["magnitude"]) else ""
                ax.set_ylabel(
                    f"{CASE_LABELS[name]}\nEvent {case['event_id']}, {magnitude}\nNorth–south distance (km)",
                    fontsize=6.8,
                )
            else:
                ax.set_yticklabels([])
            if row == len(names) - 1:
                ax.set_xlabel("East–west distance (km)", fontsize=6.8)
            else:
                ax.set_xticklabels([])

    risk_cax = figure.add_axes([0.18, 0.045, 0.28, 0.015])
    corr_cax = figure.add_axes([0.59, 0.045, 0.28, 0.015])
    risk_scalar = mpl.cm.ScalarMappable(norm=risk_norm, cmap="viridis")
    corr_scalar = mpl.cm.ScalarMappable(norm=correction_norm, cmap="magma")
    risk_scalar.set_array([])
    corr_scalar.set_array([])
    cb1 = figure.colorbar(risk_scalar, cax=risk_cax, orientation="horizontal")
    cb2 = figure.colorbar(corr_scalar, cax=corr_cax, orientation="horizontal")
    cb1.set_label("Underprediction-risk score", labelpad=1.5)
    cb2.set_label("Applied correction (log$_{10}$ units)", labelpad=1.5)
    cb1.ax.tick_params(labelsize=6.5, length=2)
    cb2.ax.tick_params(labelsize=6.5, length=2)

    for extension in ["png", "pdf", "svg"]:
        path = out_dir / f"FigS_selected_case_risk_and_correction.{extension}"
        kwargs = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(figure)


# -----------------------------------------------------------------------------
# Tables, captions and manuscript draft
# -----------------------------------------------------------------------------


def selected_case_table(
    selections: dict[str, CaseSelection],
    cases: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in [*CASE_ORDER, "overcorrection"]:
        selection = selections[name]
        case = cases[name]
        locked = selection.row
        row: dict[str, Any] = {
            "case": name,
            "case_label": CASE_LABELS[name],
            "event_id": case["event_id"],
            "repeat": case["repeat"],
            "magnitude": case["magnitude"],
            "locked_n_targets": int(locked["n_targets"]),
            "locked_n_tail_targets": int(locked["n_tail_targets"]),
            "locked_base_overall_mae": float(locked["base_overall_mae"]),
            "locked_caurc_overall_mae": float(locked["caurc_overall_mae"]),
            "locked_overall_mae_change": float(locked["overall_mae_change"]),
            "locked_base_tail_mae": float(locked["base_tail_mae"]),
            "locked_caurc_tail_mae": float(locked["caurc_tail_mae"]),
            "locked_tail_mae_reduction": float(locked["tail_mae_reduction"]),
            "locked_non_tail_mae_change": float(locked["non_tail_mae_change"]),
            "all_heldout_stations": int(len(case["target_indices"])),
        }
        for quantity in ["pga", "pgv"]:
            for model_key, metrics_key in [("base", "base_metrics"), ("caurc", "final_metrics")]:
                metrics = case[metrics_key]
                for metric in ["mae", "bias", "tail_mae", "tail_bias", "tail_u05"]:
                    row[f"all_{model_key}_{quantity}_{metric}"] = metrics[f"{quantity}_{metric}"]
            row[f"all_{quantity}_tail_n"] = int(case["final_metrics"][f"{quantity}_tail_n"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_captions(
    out_dir: Path,
    prediction_audit: dict[str, Any],
    selected_table: pd.DataFrame,
) -> None:
    selected_lookup = selected_table.set_index("case")
    case_texts = []
    for letter, name in zip(["b", "c", "d"], CASE_ORDER):
        row = selected_lookup.loc[name]
        magnitude = f"M{row['magnitude']:.1f}" if np.isfinite(row["magnitude"]) else "magnitude unavailable"
        case_texts.append(
            f"{letter}, {CASE_LABELS[name].lower()} for event {row['event_id']} ({magnitude}; repeat {int(row['repeat'])})."
        )
    caption = (
        "Fig. 8 | Spatial case studies illustrate selective correction and residual failure. "
        "a, Event–repeat selection landscape showing the change in non-tail MAE against the reduction in high-motion-tail MAE. "
        "Positive vertical values indicate lower tail error after CA-URC, whereas positive horizontal values indicate larger non-tail error. "
        "Cases were selected algorithmically from the locked target predictions without inspecting the spatial maps. "
        + " ".join(case_texts)
        + " Each case row shows residuals, defined as prediction minus observation, for PGA and PGV from the frozen Cross-Attention Base and CA-URC. "
        "Filled circles denote non-input target stations, black outlines mark high-motion-tail targets, "
        "triangles denote the five input stations and stars denote epicentres when catalog coordinates were available. "
        "After case selection, the frozen model was queried at every remaining non-input station using the same input set solely for descriptive spatial visualization; "
        "aggregate performance conclusions remain based on the locked target draws. "
        f"The selection landscape contains {prediction_audit['event_repeat_groups']:,} event–repeat units from {prediction_audit['events']} test earthquakes. "
        "Exact selected-case metrics and selection rules are provided in Supplementary Table 7."
    )
    (out_dir / "Fig8_caption.txt").write_text(caption, encoding="utf-8")

    supp_caption = (
        "Supplementary Fig. X | Spatial distribution of underprediction-risk scores and applied corrections in selected cases. "
        "Rows show the successful-correction, residual-failure and non-tail-overcorrection cases selected by the algorithmic rules in Supplementary Table 7. "
        "Columns show the underprediction-risk score and the resulting non-negative correction for PGA and PGV. "
        "Markers denote non-input target stations; black outlines mark high-motion-tail targets, triangles denote input stations and stars denote epicentres. "
        "The score is not described as a calibrated probability unless calibration is established separately."
    )
    (out_dir / "FigS_caption.txt").write_text(supp_caption, encoding="utf-8")


def format_ci_free_value(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def write_manuscript_draft(
    out_dir: Path,
    selected_table: pd.DataFrame,
    failure_summary: pd.DataFrame,
) -> None:
    lookup = selected_table.set_index("case")
    failure_lookup = failure_summary.set_index("criterion")

    rep = lookup.loc["representative"]
    success = lookup.loc["successful_correction"]
    failure = lookup.loc["residual_failure"]
    over = lookup.loc["overcorrection"]

    tail_help = failure_lookup.loc["CA-URC reduces combined tail MAE"]
    tail_no_help = failure_lookup.loc["CA-URC does not reduce combined tail MAE"]
    remaining = failure_lookup.loc["at least one severe tail underprediction remains"]
    non_tail_worse = failure_lookup.loc["non-tail MAE increases"]
    non_tail_large = failure_lookup.loc["non-tail MAE increases by >0.02 log10 units"]

    text = f"""Spatial case studies illustrate selective correction and residual failure

Aggregate metrics establish the average behaviour of CA-URC but do not show how a non-negative target-specific correction is distributed across an individual station network. We therefore defined three case-selection rules at the event–repeat level before inspecting any spatial map. A representative-preservation case was chosen near the median base-model error among units containing at most one high-motion target. A successful-correction case was selected as the largest combined tail-MAE reduction among units containing at least two high-motion targets while limiting non-tail degradation to 0.02 log10 units. A residual-failure case was selected as the unit with the largest remaining CA-URC tail MAE among those retaining at least one severe tail underprediction. An additional non-tail-overcorrection case was selected for Supplementary Fig. X. Selection used only the ten locked target predictions for each event–repeat unit; the frozen model was subsequently queried at all remaining stations using the same input set solely for descriptive spatial visualization (Fig. 8a).

In the representative-preservation case (event {rep['event_id']}, M{rep['magnitude']:.1f}, repeat {int(rep['repeat'])}), the locked overall MAE changed from {rep['locked_base_overall_mae']:.3f} to {rep['locked_caurc_overall_mae']:.3f} log10 units, a paired change of {rep['locked_overall_mae_change']:+.3f}. The all-held-out maps showed that corrections were spatially limited rather than uniformly applied across the network (Fig. 8b). This behaviour is consistent with the small catalog-wide MAE change reported in Section 2.2, although a single representative case is illustrative rather than inferential.

The successful-correction case (event {success['event_id']}, M{success['magnitude']:.1f}, repeat {int(success['repeat'])}) contained {int(success['locked_n_tail_targets'])} high-motion targets in the locked query. Its combined locked tail MAE decreased from {success['locked_base_tail_mae']:.3f} to {success['locked_caurc_tail_mae']:.3f}, a reduction of {success['locked_tail_mae_reduction']:.3f} log10 units, while non-tail MAE changed by {success['locked_non_tail_mae_change']:+.3f}. Spatially, the largest positive corrections coincided with stations at which the base residuals were strongly negative, moving both PGA and PGV predictions towards the observations (Fig. 8c and Supplementary Fig. X). The maps therefore illustrate the intended selective action of the underprediction-risk gate rather than a uniform event-level offset.

The residual-failure case (event {failure['event_id']}, M{failure['magnitude']:.1f}, repeat {int(failure['repeat'])}) demonstrates the limit of this mechanism. After correction, the locked combined tail MAE remained {failure['locked_caurc_tail_mae']:.3f} log10 units and at least one target retained a residual below -0.5 log10 units. The all-held-out maps showed spatially coherent regions of negative residual that were reduced but not eliminated by CA-URC (Fig. 8d). Because the correction is constrained by information available at the 5-s snapshot and by a maximum correction amplitude, severe underprediction can persist where the early input network does not sufficiently constrain the subsequent target motion. The maps identify this operating boundary but do not by themselves distinguish among source evolution, propagation geometry and local site response as causal explanations.

Across the {int(tail_help['count'] + tail_no_help['count'])} event–repeat units containing at least one high-motion quantity, CA-URC reduced combined tail MAE in {100*tail_help['fraction']:.1f}% and did not reduce it in {100*tail_no_help['fraction']:.1f}%. At least one severe tail underprediction remained in {100*remaining['fraction']:.1f}% of these units. Across all event–repeat units, non-tail MAE increased in {100*non_tail_worse['fraction']:.1f}%, but an increase greater than 0.02 log10 units occurred in {100*non_tail_large['fraction']:.1f}%. The supplementary overcorrection case (event {over['event_id']}, M{over['magnitude']:.1f}, repeat {int(over['repeat'])}) illustrates why directional risk reduction must be evaluated together with signed bias and overprediction. Collectively, the spatial analyses show that CA-URC acts selectively and can correct coherent high-motion underprediction, but its benefit is heterogeneous and substantial residual errors remain in the most weakly constrained cases.
"""
    (out_dir / "Section2_7_manuscript_draft.txt").write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Demo data
# -----------------------------------------------------------------------------


def demo_statistics(seed: int = 20260905) -> tuple[pd.DataFrame, dict[str, CaseSelection], dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    n_units = 700
    stats = pd.DataFrame(
        {
            "event_id": [f"D{10000+i:05d}" for i in range(n_units)],
            "repeat": rng.integers(0, 20, n_units),
            "n_targets": 10,
            "n_tail_targets": rng.choice([0, 1, 2, 3, 4], size=n_units, p=[0.34, 0.30, 0.20, 0.11, 0.05]),
        }
    )
    stats["n_tail_values"] = stats["n_tail_targets"] + rng.binomial(stats["n_tail_targets"], 0.35)
    stats["n_non_tail_values"] = 20 - stats["n_tail_values"]
    stats["base_overall_mae"] = np.clip(rng.normal(0.31, 0.09, n_units), 0.08, 0.75)
    stats["overall_mae_change"] = rng.normal(-0.001, 0.015, n_units)
    stats["caurc_overall_mae"] = stats["base_overall_mae"] + stats["overall_mae_change"]
    stats["base_tail_mae"] = np.where(
        stats["n_tail_values"].gt(0), np.clip(rng.normal(0.66, 0.20, n_units), 0.15, 1.35), np.nan
    )
    stats["tail_mae_reduction"] = np.where(
        stats["n_tail_values"].gt(0), rng.normal(0.075, 0.075, n_units), np.nan
    )
    stats["caurc_tail_mae"] = stats["base_tail_mae"] - stats["tail_mae_reduction"]
    stats["base_non_tail_mae"] = np.clip(rng.normal(0.27, 0.07, n_units), 0.06, 0.60)
    stats["non_tail_mae_change"] = rng.normal(0.001, 0.014, n_units)
    stats["caurc_non_tail_mae"] = stats["base_non_tail_mae"] + stats["non_tail_mae_change"]
    stats["base_tail_u05"] = np.where(stats["n_tail_values"].gt(0), rng.uniform(0.35, 0.9, n_units), np.nan)
    stats["caurc_tail_u05"] = np.where(stats["n_tail_values"].gt(0), np.clip(stats["base_tail_u05"] - rng.normal(0.1, 0.08, n_units), 0, 1), np.nan)
    stats["remaining_tail_u05_count"] = np.where(
        stats["n_tail_values"].gt(0),
        np.maximum(0, np.rint(stats["caurc_tail_u05"].fillna(0) * stats["n_tail_values"]).astype(int)),
        0,
    )
    stats["base_non_tail_o05"] = rng.uniform(0, 0.08, n_units)
    stats["caurc_non_tail_o05"] = np.clip(stats["base_non_tail_o05"] + rng.normal(0.01, 0.02, n_units), 0, 1)
    stats["mean_applied_correction"] = np.clip(rng.normal(0.04, 0.025, n_units), 0, None)
    stats["max_applied_correction"] = np.clip(rng.normal(0.35, 0.18, n_units), 0.02, 1.2)
    stats["minimum_applied_correction"] = 0.0
    stats["correction_nonnegative_with_tolerance"] = True

    selections = select_cases(
        stats,
        success_non_tail_tolerance=0.02,
        manual={
            "representative": (None, None),
            "successful_correction": (None, None),
            "residual_failure": (None, None),
            "overcorrection": (None, None),
        },
    )

    cases: dict[str, dict[str, Any]] = {}
    for index, name in enumerate([*CASE_ORDER, "overcorrection"]):
        row = selections[name].row
        n_target = 58
        theta = rng.uniform(0, 2 * np.pi, n_target)
        radius = np.sqrt(rng.uniform(0, 1, n_target)) * (80 + 20 * index)
        xy_target = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        xy_input = rng.normal(0, 18, size=(5, 2))
        truth = np.column_stack(
            [rng.normal(-2.5, 0.4, n_target), rng.normal(-3.8, 0.45, n_target)]
        )
        thresholds = np.asarray([-2.35, -3.65])
        tail = truth >= thresholds[None, :]
        base_residual = rng.normal(-0.15, 0.30, size=(n_target, 2))
        if name == "successful_correction":
            base_residual[tail] -= 0.45
        elif name == "residual_failure":
            base_residual[tail] -= 0.75
        elif name == "representative":
            base_residual *= 0.55
        under_probability = 1.0 / (1.0 + np.exp(4.2 * (base_residual + 0.35)))
        raw = np.clip(rng.normal(0.55, 0.18, size=(n_target, 2)), 0.05, 1.0)
        correction = under_probability**5 * raw
        if name == "residual_failure":
            correction *= 0.55
        if name == "overcorrection":
            correction += rng.uniform(0.05, 0.18, size=(n_target, 2))
        base = truth + base_residual
        final = base + correction
        cases[name] = {
            "case_name": name,
            "event_id": str(row["event_id"]),
            "repeat": int(row["repeat"]),
            "magnitude": float([3.5, 4.2, 4.8, 3.7][index]),
            "xy_input": xy_input,
            "xy_target": xy_target,
            "epicenter_xy": np.asarray([0.0, 0.0]),
            "truth": truth,
            "base": base,
            "final": final,
            "correction": correction,
            "under_probability": under_probability,
            "raw_correction_amplitude": raw,
            "tail": tail,
            "target_indices": np.arange(n_target),
            "locked_target_indices": np.arange(10),
            "station_ids": np.asarray([f"S{i:03d}" for i in range(n_target + 5)]),
            "p_wave_reached": rng.random(n_target) > 0.4,
            "base_metrics": case_metrics(truth, base, tail),
            "final_metrics": case_metrics(truth, final, tail),
            "locked_selection": row.to_dict(),
            "audit": {"demo": True},
        }
    return stats, selections, cases


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="runs/cadrg_gate_ablation_A3_A5/locked_test_ablation_predictions.csv",
    )
    parser.add_argument(
        "--manifest",
        default="model_manifests/scenario_t0_5s_k5_linux.csv",
    )
    parser.add_argument(
        "--h5-root",
        default='data/scedc/processed_full_v4/events',
    )
    parser.add_argument("--baseline-module", default="45_phase2_strong_baseline_suite.py")
    parser.add_argument(
        "--ablation-module",
        default="63_train_and_evaluate_cadrg_gate_ablations_A3_A5.py",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="runs/final_strong_baselines_reuse_locked/cross_attention/best_model.pt",
    )
    parser.add_argument(
        "--head-checkpoint",
        default="runs/cadrg_gate_ablation_A3_A5/A4_under_only/selected_head.pt",
    )
    parser.add_argument(
        "--selection-json",
        default="runs/cadrg_gate_ablation_A3_A5/A4_under_only/selected_epoch_gamma.json",
    )
    parser.add_argument(
        "--threshold-json",
        default="runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json",
    )
    parser.add_argument("--split-column", default="split_grouped")
    parser.add_argument("--test-label", default="test")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--t0-sec", type=float, default=5.0)
    parser.add_argument("--input-stations", type=int, default=5)
    parser.add_argument("--target-stations", type=int, default=10)
    parser.add_argument("--input-pre-sec", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--risk-hidden", type=int, default=64)
    parser.add_argument("--maximum-correction", type=float, default=1.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--success-non-tail-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--audit-tolerance",
        type=float,
        default=2e-4,
        help=(
            "Cross-backend tolerance for reproducing archived locked model outputs. "
            "Default 2e-4; truth is still checked at <=2e-6 and a 1e-3 hard ceiling is enforced."
        ),
    )
    parser.add_argument(
        "--query-tolerance",
        type=float,
        default=2e-4,
        help=(
            "Strict tolerance for query-set independence within the current run. "
            "This is a soft warning threshold; a 1e-3 hard ceiling is still enforced."
        ),
    )

    for name in ["representative", "success", "failure", "overcorrection"]:
        parser.add_argument(f"--event-{name}", default=None)
        parser.add_argument(f"--repeat-{name}", type=int, default=None)

    parser.add_argument("--skip-locked-audit", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--out-dir", default="figures/nc_section_2_7")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        stats, selections, cases = demo_statistics()
        prediction_audit = {
            "prediction_rows": EXPECTED_ROWS,
            "events": EXPECTED_EVENTS,
            "event_repeat_groups": int(len(stats)),
            "pga_tail_rows": EXPECTED_PGA_TAIL_ROWS,
            "pgv_tail_rows": EXPECTED_PGV_TAIL_ROWS,
            "demo": True,
        }
        model_audit = {"demo": True, "variant": "A4_under_only", "selected_epoch": 3, "selected_gamma": 5.0}
        torch_backend_audit = {"demo": True}
        file_fingerprints: dict[str, str] = {}
    else:
        torch_backend_audit = configure_torch_determinism()

        predictions_path = Path(args.predictions)
        manifest_path = Path(args.manifest)
        threshold_path = Path(args.threshold_json)
        base_checkpoint_path = Path(args.base_checkpoint)
        head_checkpoint_path = Path(args.head_checkpoint)
        selection_json_path = Path(args.selection_json)
        baseline_module_path = Path(args.baseline_module)
        ablation_module_path = Path(args.ablation_module)

        required_paths = [
            predictions_path,
            manifest_path,
            threshold_path,
            base_checkpoint_path,
            head_checkpoint_path,
            selection_json_path,
            baseline_module_path,
            ablation_module_path,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(path)

        locked_predictions, prediction_audit = load_locked_predictions(
            predictions_path,
            skip_locked_audit=args.skip_locked_audit,
        )
        stats = build_case_statistics(locked_predictions)
        manual = {
            "representative": (args.event_representative, args.repeat_representative),
            "successful_correction": (args.event_success, args.repeat_success),
            "residual_failure": (args.event_failure, args.repeat_failure),
            "overcorrection": (args.event_overcorrection, args.repeat_overcorrection),
        }
        selections = select_cases(stats, args.success_non_tail_tolerance, manual)

        manifest_frame = pd.read_csv(manifest_path, dtype={"event_id": str})
        require_columns(manifest_frame, ["event_id", "h5_path", args.split_column], "manifest")
        manifest_frame["event_id"] = manifest_frame["event_id"].astype(str).str.strip()
        manifest_frame = manifest_frame.loc[
            manifest_frame[args.split_column].astype(str).eq(str(args.test_label))
        ].drop_duplicates("event_id")
        manifest = manifest_frame.set_index("event_id", drop=False)

        threshold_data = read_json(threshold_path)
        thresholds = np.asarray(
            [
                threshold_data["log10_pga_threshold"],
                threshold_data["log10_pgv_threshold"],
            ],
            dtype=float,
        )

        (
            baseline_module,
            _ablation_module,
            model,
            gamma,
            device,
            model_audit,
        ) = reconstruct_final_model(
            baseline_module_path,
            ablation_module_path,
            base_checkpoint_path,
            head_checkpoint_path,
            selection_json_path,
            args.device,
            args.hidden_dim,
            args.attention_heads,
            args.risk_hidden,
            args.maximum_correction,
        )

        # -----------------------------------------------------------------
        # IMPORTANT: reproduce the exact deterministic station sampling used
        # by the locked 44,800-row A4/CA-URC test predictions.
        #
        # Script 63 generated its locked test only after patching:
        #     strong:test:*  ->  locked:test:*
        # before calling the baseline module's stable_seed().  Fig. 8 must
        # apply the identical patch; otherwise both the five input stations
        # and the ten target stations are drawn from a different RNG stream.
        # Keep the downstream Generated-vs-Stored and prediction audits: they
        # are deliberate manuscript reproducibility guards.
        # -----------------------------------------------------------------
        if not hasattr(_ablation_module, "patch_locked_test_seed_protocol"):
            raise AttributeError(
                f"{ablation_module_path.name} does not expose "
                "patch_locked_test_seed_protocol(), which is required to "
                "reproduce the locked test station draws."
            )
        _ablation_module.patch_locked_test_seed_protocol(baseline_module)

        h5_root = Path(args.h5_root) if str(args.h5_root).strip() else None
        cases = {}
        for name in [*CASE_ORDER, "overcorrection"]:
            cases[name] = infer_all_heldout_case(
                selections[name],
                locked_predictions,
                manifest,
                h5_root,
                baseline_module,
                model,
                gamma,
                device,
                thresholds,
                str(args.test_label),
                args.seed,
                args.t0_sec,
                args.input_stations,
                args.target_stations,
                args.input_pre_sec,
                args.audit_tolerance,
                args.query_tolerance,
            )

        file_fingerprints = {
            str(path.resolve()): sha256_file(path)
            for path in [
                predictions_path,
                manifest_path,
                threshold_path,
                base_checkpoint_path,
                head_checkpoint_path,
                selection_json_path,
            ]
        }

    stats.to_csv(out_dir / "case_selection_statistics.csv", index=False)
    failure_summary = failure_mode_summary(stats)
    failure_summary.to_csv(out_dir / "TableS7_failure_mode_summary.csv", index=False)
    output_tex(failure_summary, out_dir / "TableS7_failure_mode_summary.tex")

    selected_table = selected_case_table(selections, cases)
    selected_table.to_csv(out_dir / "TableS7_selected_cases.csv", index=False)
    output_tex(selected_table, out_dir / "TableS7_selected_cases.tex")

    plot_main_figure(stats, selections, cases, out_dir, args.dpi)
    plot_risk_correction_supplement(cases, out_dir, args.dpi)
    write_captions(out_dir, prediction_audit, selected_table)
    write_manuscript_draft(out_dir, selected_table, failure_summary)

    audit = {
        "final_model_name": "CA-URC",
        "final_model_formula": "y_CA + p_under^gamma * Delta",
        "selection_source": "locked event-repeat predictions; maps not inspected during selection",
        "all_heldout_query_purpose": "descriptive spatial visualization only",
        "aggregate_metric_source": "locked target draws",
        "prediction_audit": prediction_audit,
        "model_audit": model_audit,
        "torch_backend_audit": torch_backend_audit,
        "selected_cases": {
            name: {
                "event_id": str(selection.row["event_id"]),
                "repeat": int(selection.row["repeat"]),
            }
            for name, selection in selections.items()
        },
        "per_case_reproduction_audit": {
            name: cases[name]["audit"] for name in cases
        },
        "success_non_tail_tolerance": float(args.success_non_tail_tolerance),
        "audit_tolerance": float(args.audit_tolerance),
        "query_tolerance": float(args.query_tolerance),
        "file_sha256": file_fingerprints,
        "demo": bool(args.demo),
    }
    (out_dir / "case_selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("=== NC Section 2.7 spatial case studies ===")
    print(f"Output directory : {out_dir.resolve()}")
    print(f"Case units       : {len(stats):,}")
    print("Selected cases:")
    for name in [*CASE_ORDER, "overcorrection"]:
        selection = selections[name].row
        print(
            f"  {name:24s} event={selection['event_id']} repeat={int(selection['repeat'])} "
            f"tail_reduction={selection['tail_mae_reduction']:+.4f} "
            f"non_tail_change={selection['non_tail_mae_change']:+.4f}"
        )
    print("Generated:")
    for name in [
        "Fig8_spatial_cases.png",
        "Fig8_spatial_cases.pdf",
        "Fig8_spatial_cases.svg",
        "FigS_selected_case_risk_and_correction.png",
        "FigS_selected_case_risk_and_correction.pdf",
        "TableS7_selected_cases.csv",
        "TableS7_failure_mode_summary.csv",
        "Section2_7_manuscript_draft.txt",
        "Fig8_caption.txt",
        "FigS_caption.txt",
        "case_selection_audit.json",
    ]:
        print(f"  {(out_dir / name).resolve()}")


if __name__ == "__main__":
    main()

