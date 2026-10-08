#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig7_JGR_SE_spatial_case_studies.py

JGR-SE manuscript spatial case-study analysis for the final
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
Fig7_spatial_cases.png/.pdf/.svg
FigS2_selected_case_risk_and_correction.png/.pdf/.svg
TableS7_selected_cases.csv/.tex
TableS7_failure_mode_summary.csv/.tex
Spatial_cases_manuscript_draft.txt
Fig7_caption.txt
FigS2_caption.txt
case_selection_statistics.csv
case_selection_audit.json

Revision notes:
- Case selection, thresholds, checkpoint loading, locked station draws, and
  numerical reproduction checks are unchanged.
- Main figure: separate metric bands, tail sample counts, non-overlapping case
  labels, original map coordinates/aspect/colormaps, larger final-size fonts.
- Empty tails remain NA; raw values are exported to Fig7_map_metric_audit.csv.
- Output names match the existing manuscript; figure numbers are assigned by LaTeX.
- The --demo mode generates watermarked synthetic tests in a separate demo_preview
  directory. Demo output is never a substitute for real prediction data.
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


# Output file stems match the existing manuscript. Actual figure numbers remain
# controlled by LaTeX; a file named Fig7 may become Figure 9 in the final layout.
FIGURE_WIDTH_MM = 183.0
MAIN_HEIGHT_MM = 252.0
SUPPLEMENT_HEIGHT_MM = 208.0
MAIN_STEM = "Fig7_spatial_cases"
SUPPLEMENT_STEM = "FigS2_selected_case_risk_and_correction"
FONT_PT = {
    "body": 8.2, "axis": 8.2, "tick": 7.8, "title": 8.1,
    "header": 8.7, "annotation": 7.8, "metric": 7.7,
    "legend": 7.8, "panel": 10.5,
}


def axis_mm(figure, left: float, top: float, width: float, height: float):
    """Create an axis using millimeters measured from the canvas top-left."""
    canvas_width, canvas_height = figure.get_size_inches()*25.4
    return figure.add_axes([left/canvas_width,
                            1-(top+height)/canvas_height,
                            width/canvas_width, height/canvas_height])


def display_mae(value: float) -> str:
    """Three decimals, with a nonzero sub-rounding value distinguished from zero."""
    if not np.isfinite(value):
        return "NA"
    if 0 < float(value) < 0.0005:
        return "<0.001"
    return f"{float(value):.3f}"


def station_legend_handles() -> list:
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", markersize=4.8, label="Input station"),
        Line2D([], [], marker="*", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", markersize=7.0, label="Epicenter"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.7",
               markeredgecolor="white", markersize=4.0, label="Non-input target"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.7",
               markeredgecolor="black", markersize=5.2, label="High-motion tail"),
    ]


def cases_are_demo(cases: dict[str, dict[str, Any]]) -> bool:
    return any(bool(case.get("audit", {}).get("demo", False)) for case in cases.values())


def annotate_demo(figure, cases: dict[str, dict[str, Any]]) -> None:
    if cases_are_demo(cases):
        figure.text(0.5, 0.995, "SYNTHETIC LAYOUT TEST — NOT RESEARCH RESULTS",
                    ha="center", va="top", fontsize=7.8, fontweight="bold", color="0.30")


def colorbar_extension(values: np.ndarray, vmin: float, vmax: float) -> str:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = bool(np.any(finite < vmin))
    high = bool(np.any(finite > vmax))
    return "both" if low and high else "min" if low else "max" if high else "neither"


def save_figure_outputs(figure, out_dir: Path, stem: str, dpi: int,
                        audit: dict[str, Any]) -> None:
    """Preserve physical width; rasterized point layers use the requested DPI."""
    from matplotlib.text import Text
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    # Locators may retain text artists for ticks outside the current view.
    # Matplotlib does not draw those labels, so exclude them from the audit.
    undrawn_ticks = set()
    for ax in figure.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not lo-1e-10 <= tick.get_loc() <= hi+1e-10:
                    undrawn_ticks.update([id(tick.label1), id(tick.label2)])
    visible = [text for text in figure.findobj(Text)
               if text.get_visible() and text.get_text().strip()
               and id(text) not in undrawn_ticks
               and (text.axes is None or text.axes.get_visible())]
    outside = []
    for text in visible:
        try:
            bbox = text.get_window_extent(renderer)
        except (ValueError, RuntimeError):
            continue
        if (bbox.x0 < canvas.x0-2 or bbox.y0 < canvas.y0-2 or
                bbox.x1 > canvas.x1+2 or bbox.y1 > canvas.y1+2):
            outside.append(text.get_text())
    audit.update({
        "canvas_width_mm": float(figure.get_figwidth()*25.4),
        "canvas_height_mm": float(figure.get_figheight()*25.4),
        "font_settings_pt": FONT_PT,
        "minimum_text_artist_font_pt": min((t.get_fontsize() for t in visible), default=None),
        "text_outside_canvas": outside,
        "export_dpi": int(dpi),
        "bbox_inches": None,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(out_dir/f"{stem}.{extension}", dpi=int(dpi), bbox_inches=None,
                       facecolor="white")
    (out_dir/f"{stem}_style_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if outside:
        print(f"WARNING: {stem}: {len(outside)} text objects exceed the canvas; inspect style audit.")


def export_map_source_data(cases: dict[str, dict[str, Any]], out_dir: Path) -> None:
    """Raw expanded-map values and sample counts, separate from locked summaries."""
    metric_rows, target_rows = [], []
    for name in [*CASE_ORDER, "overcorrection"]:
        case = cases[name]
        locked_indices = set(np.asarray(case["locked_target_indices"], dtype=int).tolist())
        for q_index, quantity in enumerate(["pga", "pgv"]):
            for model, key in [("base", "base_metrics"), ("caurc", "final_metrics")]:
                metrics = case[key]
                n_tail = int(metrics[f"{quantity}_tail_n"])
                metric_rows.append({
                    "case": name, "event_id": str(case["event_id"]), "repeat": int(case["repeat"]),
                    "scope": "expanded_non_input_stations", "quantity": quantity, "model": model,
                    "N_targets": len(case["truth"]), "n_tail": n_tail,
                    "mae": metrics[f"{quantity}_mae"], "tail_mae": metrics[f"{quantity}_tail_mae"],
                    "tail_u05_fraction": metrics[f"{quantity}_tail_u05"],
                    "tail_empty": n_tail == 0,
                    "tail_mae_would_round_to_0_00": bool(n_tail and np.isfinite(metrics[f"{quantity}_tail_mae"])
                        and f"{metrics[f'{quantity}_tail_mae']:.2f}" == "0.00"),
                    "demo": bool(case.get("audit", {}).get("demo", False)),
                })
            for slot, station_index in enumerate(np.asarray(case["target_indices"], dtype=int)):
                station_ids = case.get("station_ids", [])
                station_id = str(station_ids[station_index]) if station_index < len(station_ids) else str(station_index)
                target_rows.append({
                    "case": name, "event_id": str(case["event_id"]), "repeat": int(case["repeat"]),
                    "scope": "expanded_non_input_stations", "station_index": int(station_index),
                    "station_id": station_id, "is_locked_target": station_index in locked_indices,
                    "quantity": quantity, "x_km": float(case["xy_target"][slot, 0]),
                    "y_km": float(case["xy_target"][slot, 1]),
                    "true_log10": float(case["truth"][slot, q_index]),
                    "base_log10": float(case["base"][slot, q_index]),
                    "caurc_log10": float(case["final"][slot, q_index]),
                    "is_tail": bool(case["tail"][slot, q_index]),
                    "underprediction_risk_score": float(case["under_probability"][slot, q_index]),
                    "applied_correction": float(case["correction"][slot, q_index]),
                    "demo": bool(case.get("audit", {}).get("demo", False)),
                })
    pd.DataFrame(metric_rows).to_csv(out_dir/"Fig7_map_metric_audit.csv", index=False)
    pd.DataFrame(target_rows).to_csv(out_dir/"Fig7_source_data_map_targets.csv", index=False)


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


def plot_selection_landscape(ax, stats: pd.DataFrame,
                             selections: dict[str, CaseSelection]) -> None:
    """Locked-draw selection landscape. Only annotation placement is changed."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    tail_units = stats.loc[stats["n_tail_values"].gt(0)].copy()
    finite = np.isfinite(tail_units["non_tail_mae_change"]) & np.isfinite(tail_units["tail_mae_reduction"])
    plot = tail_units.loc[finite]
    if plot.empty:
        raise ValueError("No finite tail/non-tail pairs are available for panel a.")
    ax.scatter(plot["non_tail_mae_change"], plot["tail_mae_reduction"],
               s=10, facecolors="#AEB4BA", edgecolors="none", alpha=0.30,
               rasterized=True, zorder=2)
    ax.axvline(0.0, color="0.28", linestyle=(0, (4, 2.5)), linewidth=0.78, zorder=1)
    ax.axhline(0.0, color="0.28", linestyle=(0, (4, 2.5)), linewidth=0.78, zorder=1)
    valid_selections = {}
    for name, selection in selections.items():
        row = selection.row
        if not (np.isfinite(row.get("tail_mae_reduction", np.nan)) and
                np.isfinite(row.get("non_tail_mae_change", np.nan))):
            continue
        valid_selections[name] = selection
        marker = selected_marker_style(name)
        ax.scatter([row["non_tail_mae_change"]], [row["tail_mae_reduction"]],
                   marker=marker["marker"], s=marker["s"], facecolor=marker["facecolor"],
                   edgecolor="black", linewidth=0.70, zorder=6, label=CASE_LABELS[name])

    # Show every finite original point; add space rather than crop difficult cases.
    ax.margins(x=0.08, y=0.18)
    style_axis(ax, grid=True)
    ax.set_xlabel("Non-tail MAE change (CA-URC − Base; log$_{10}$ units)", labelpad=3.0)
    ax.set_ylabel("Tail MAE reduction\n(Base − CA-URC; log$_{10}$ units)", labelpad=4.0)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.text(-0.055, 1.015, "a", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_PT["panel"], fontweight="bold")
    count_text = f"{len(plot):,} event–repeat units with ≥1 tail value"
    if len(plot) != len(tail_units):
        count_text += f" ({len(tail_units)-len(plot)} undefined pairs omitted)"
    ax.text(1.0, 1.02, count_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FONT_PT["annotation"], color="0.38")
    handles, labels = ax.get_legend_handles_labels()
    legend = None
    if handles:
        lookup = dict(zip(labels, handles))
        desired = [CASE_LABELS[name] for name in [*CASE_ORDER, "overcorrection"]]
        desired = [label for label in desired if label in lookup]
        legend = ax.legend([lookup[label] for label in desired], desired,
                           loc="lower left", ncol=2, frameon=True, framealpha=0.90,
                           facecolor="white", edgecolor="none", handletextpad=0.4,
                           columnspacing=0.9, borderpad=0.3, labelspacing=0.3,
                           borderaxespad=0.25, fontsize=FONT_PT["legend"])
        legend.set_zorder(15)

    # Deterministic candidate anchors. Labels avoid one another, selected markers,
    # the legend and the frame. Coordinates of the scientific points never move.
    anchors = {
        "successful_correction": [(0.035, 0.91, "left"), (0.33, 0.95, "left"),
                                  (0.96, 0.95, "right"), (0.035, 0.70, "left")],
        "representative": [(0.035, 0.49, "left"), (0.035, 0.68, "left"),
                           (0.50, 0.39, "left"), (0.96, 0.70, "right")],
        "residual_failure": [(0.63, 0.48, "left"), (0.96, 0.65, "right"),
                             (0.96, 0.88, "right"), (0.42, 0.75, "left")],
    }
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer).padded(-3.0)
    occupied = []
    if legend is not None:
        occupied.append(legend.get_window_extent(renderer).padded(3.0))
    for selection in valid_selections.values():
        row = selection.row
        x, y = ax.transData.transform((row["non_tail_mae_change"], row["tail_mae_reduction"]))
        occupied.append(Bbox.from_extents(x-7, y-7, x+7, y+7))
    annotation_audit = []
    for name in ["successful_correction", "representative", "residual_failure"]:
        if name not in valid_selections:
            continue
        row = valid_selections[name].row
        ann = ax.annotate(
            f"{CASE_LABELS[name]}\n{row['event_id']} / r{int(row['repeat'])}",
            xy=(row["non_tail_mae_change"], row["tail_mae_reduction"]), xycoords="data",
            xytext=anchors[name][0][:2], textcoords="axes fraction",
            ha=anchors[name][0][2], va="top", fontsize=FONT_PT["annotation"],
            color="#20252A", linespacing=1.08,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.0),
            arrowprops=dict(arrowstyle="-", color="0.38", lw=0.65, shrinkA=3, shrinkB=6),
            zorder=12, annotation_clip=False)
        px, py = ax.transAxes.inverted().transform(
            ax.transData.transform((row["non_tail_mae_change"], row["tail_mae_reduction"])))
        nearby = {
            "representative": [(max(0.025, px-0.30), py-0.075, "left")],
            "residual_failure": [(px+0.055, py+0.01, "left"),
                                 (px+0.055, min(0.94, py+0.19), "left")],
            "successful_correction": [(px+0.03, min(0.94, py+0.14), "left")],
        }
        candidates = nearby[name] + anchors[name] + [
            (x, y, ha) for x, ha in [(0.025, "left"), (0.42, "left"), (0.975, "right")]
            for y in [0.94, 0.76, 0.56, 0.39]]
        best = None
        for x, y, ha in candidates:
            ann.set_position((x, y)); ann.set_ha(ha)
            ann.update_positions(renderer)
            bbox = Text.get_window_extent(ann, renderer).padded(2.0)
            outside = max(frame.x0-bbox.x0, 0) + max(bbox.x1-frame.x1, 0)
            outside += max(frame.y0-bbox.y0, 0) + max(bbox.y1-frame.y1, 0)
            overlap = 0.0
            for previous in occupied:
                intersection = Bbox.intersection(bbox, previous)
                if intersection is not None:
                    overlap += intersection.width * intersection.height
            score = 1000.0 * outside + overlap
            if best is None or score < best[0]:
                best = (score, x, y, ha, bbox)
            if score == 0:
                break
        score, x, y, ha, bbox = best
        ann.set_position((x, y)); ann.set_ha(ha)
        ann.update_positions(renderer)
        occupied.append(bbox)
        annotation_audit.append({"case": name, "anchor": [x, y],
                                 "alignment": ha, "overlap_score": float(score)})
    ax._jgr_annotation_audit = annotation_audit



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



def plot_residual_map(ax, case: dict[str, Any], quantity_index: int,
                      model_name: str, norm, cmap: str, show_legend: bool) -> None:
    """Original station values, equal map aspect and original color mapping."""
    from matplotlib.ticker import MaxNLocator

    prediction = case["base"] if model_name == "base" else case["final"]
    residual = prediction[:, quantity_index] - case["truth"][:, quantity_index]
    tail = case["tail"][:, quantity_index]
    xy = case["xy_target"]
    ax.scatter(xy[:, 0], xy[:, 1], c=residual, cmap=cmap, norm=norm,
               s=np.where(tail, 31.0, 16.0),
               edgecolors=np.where(tail, "black", "white"),
               linewidths=np.where(tail, 0.85, 0.35), alpha=0.92,
               zorder=4, rasterized=True)
    ax.scatter(case["xy_input"][:, 0], case["xy_input"][:, 1], marker="^", s=27,
               facecolor="white", edgecolor="black", linewidth=0.7, zorder=7)
    if case["epicenter_xy"] is not None:
        ax.scatter([case["epicenter_xy"][0]], [case["epicenter_xy"][1]], marker="*",
                   s=68, facecolor="white", edgecolor="black", linewidth=0.8, zorder=8)
    x0, x1, y0, y1 = map_extent(case)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    style_axis(ax, grid=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(length=2.2, pad=1.6)
    # Metrics and station legend have their own bands in the parent figure.
    # Keep the old argument for compatibility with callers of this function.
    if show_legend:
        ax.legend(handles=station_legend_handles(), loc="lower left", framealpha=0.9,
                  fontsize=FONT_PT["legend"])



def plot_main_figure(stats: pd.DataFrame, selections: dict[str, CaseSelection],
                     cases: dict[str, dict[str, Any]], out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm

    setup_matplotlib()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, MAIN_HEIGHT_MM / 25.4))
    left, right, gap = 23.0, 5.0, 4.8
    plot_width = FIGURE_WIDTH_MM-left-right
    column_width = (plot_width-3*gap)/4
    selection_ax = axis_mm(fig, left, 9.0, plot_width, 41.0)
    plot_selection_landscape(selection_ax, stats, selections)
    station_ax = axis_mm(fig, left-1.5, 59.0, plot_width+1.5, 5.5)
    station_ax.axis("off")
    station_ax.legend(handles=station_legend_handles(), ncol=4, loc="center",
                      frameon=False, columnspacing=1.0, handletextpad=0.3,
                      fontsize=FONT_PT["legend"])

    values = np.concatenate([(cases[n][model]-cases[n]["truth"]).ravel()
                             for n in CASE_ORDER for model in ["base", "final"]])
    values = values[np.isfinite(values)]
    # Identical normalization to the original script.
    limit = float(np.percentile(np.abs(values), 98)) if len(values) else 1.0
    limit = float(np.clip(max(limit, 0.6), 0.6, 1.5))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = "RdBu_r"
    titles = ["PGA\nCross-Attention Base", "PGA\nCA-URC",
              "PGV\nCross-Attention Base", "PGV\nCA-URC"]
    # Each row has a distinct header and metric band; text no longer covers data.
    headers = [68.0, 126.0, 176.0]
    metric_tops = [83.0, 133.0, 183.0]
    map_tops = [95.0, 145.0, 195.0]
    for row, name in enumerate(CASE_ORDER):
        case = cases[name]
        magnitude = f"M{case['magnitude']:.1f}" if np.isfinite(case["magnitude"]) else "M unknown"
        header = axis_mm(fig, left, headers[row], plot_width, 5.5)
        header.axis("off")
        header.text(0, 0.6,
                    f"{'bcd'[row]}  {CASE_LABELS[name]} — Event {case['event_id']}, {magnitude}, r{case['repeat']}",
                    ha="left", va="center", fontsize=FONT_PT["header"], fontweight="bold")
        for col in range(4):
            x = left + col*(column_width+gap)
            qindex = 0 if col < 2 else 1
            model = "base" if col % 2 == 0 else "final"
            if row == 0:
                title_ax = axis_mm(fig, x, 74.5, column_width, 7.5)
                title_ax.axis("off")
                title_ax.text(0.5, 0.65, titles[col], ha="center", va="center",
                              fontsize=FONT_PT["title"], fontweight="bold", linespacing=1.12)
            text_ax = axis_mm(fig, x, metric_tops[row], column_width, 10.5)
            text_ax.axis("off")
            text_ax.text(0.01, 0.98, metric_annotation(case, "pga" if qindex == 0 else "pgv", model),
                         ha="left", va="top", fontsize=FONT_PT["metric"], linespacing=1.18)
            ax = axis_mm(fig, x, map_tops[row], column_width, 27.0)
            plot_residual_map(ax, case, qindex, model, norm, cmap, False)
            if col == 0:
                ax.set_ylabel("North–south (km)", fontsize=FONT_PT["axis"], labelpad=3)
            else:
                ax.tick_params(labelleft=False)
            if row != 2:
                ax.tick_params(labelbottom=False)

    fig.text((left+plot_width/2)/FIGURE_WIDTH_MM, 1-229.0/MAIN_HEIGHT_MM,
             "East–west distance (km)", ha="center", va="center", fontsize=FONT_PT["axis"])
    cax = axis_mm(fig, 52.0, 235.0, 102.0, 3.1)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); scalar.set_array([])
    cb = fig.colorbar(scalar, cax=cax, orientation="horizontal",
                      extend=colorbar_extension(values, -limit, limit))
    cb.set_label("Residual = prediction − observation (log$_{10}$ units)",
                 labelpad=2.8, fontsize=FONT_PT["axis"])
    cb.ax.tick_params(labelsize=FONT_PT["tick"], length=2.2, pad=1.8)
    annotate_demo(fig, cases)
    stem = "DEMO_"+MAIN_STEM if cases_are_demo(cases) else MAIN_STEM
    style_audit = {
        "annotation_layout": getattr(selection_ax, "_jgr_annotation_audit", []),
        "residual_color_limit": limit,
        "residual_values_below_scale": int((values < -limit).sum()),
        "residual_values_above_scale": int((values > limit).sum()),
        "map_metrics_scope": "all expanded non-input stations displayed in each case",
        "demo": cases_are_demo(cases),
    }
    save_figure_outputs(fig, out_dir, stem, dpi, style_audit)
    plt.close(fig)



def plot_risk_correction_supplement(cases: dict[str, dict[str, Any]],
                                    out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    setup_matplotlib()
    names = ["successful_correction", "residual_failure", "overcorrection"]
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM/25.4, SUPPLEMENT_HEIGHT_MM/25.4))
    left, right, gap = 23.0, 5.0, 4.8
    plot_width = FIGURE_WIDTH_MM-left-right
    column_width = (plot_width-3*gap)/4
    risk_norm = Normalize(0.0, 1.0)
    all_corrections = np.concatenate([cases[n]["correction"].ravel() for n in names])
    correction_max = max(0.1, float(np.percentile(all_corrections, 99)))
    correction_norm = Normalize(0.0, correction_max)
    titles = ["PGA\nUnderprediction-risk score", "PGA\nApplied correction",
              "PGV\nUnderprediction-risk score", "PGV\nApplied correction"]
    for col, title in enumerate(titles):
        title_ax = axis_mm(fig, left+col*(column_width+gap), 4.5, column_width, 10.0)
        title_ax.axis("off")
        title_ax.text(0.5, 0.5, title, ha="center", va="center",
                      fontsize=FONT_PT["title"], fontweight="bold", linespacing=1.15)
    legend_ax = axis_mm(fig, left-1.5, 16.0, plot_width+1.5, 5.5)
    legend_ax.axis("off")
    legend_ax.legend(handles=station_legend_handles(), loc="center", ncol=4, frameon=False,
                     fontsize=FONT_PT["legend"], columnspacing=1.0, handletextpad=0.3)
    for row, name in enumerate(names):
        case = cases[name]
        magnitude = f"M{case['magnitude']:.1f}" if np.isfinite(case["magnitude"]) else "M unknown"
        header_top = 24.0+52*row
        header = axis_mm(fig, left, header_top, plot_width, 5.5)
        header.axis("off")
        header.text(0, 0.6, f"{'abc'[row]}  {CASE_LABELS[name]} — Event {case['event_id']}, {magnitude}, r{case['repeat']}",
                    ha="left", va="center", fontsize=FONT_PT["header"], fontweight="bold")
        for col in range(4):
            qindex = 0 if col < 2 else 1
            is_risk = col % 2 == 0
            x = left+col*(column_width+gap)
            values = case["under_probability"][:, qindex] if is_risk else case["correction"][:, qindex]
            norm, cmap = (risk_norm, "viridis") if is_risk else (correction_norm, "magma")
            tail = case["tail"][:, qindex]
            stats_ax = axis_mm(fig, x, header_top+6, column_width, 4.3)
            stats_ax.axis("off")
            stats_ax.text(0.02, 0.8, f"N={len(tail)}; $n_{{\\mathrm{{tail}}}}$={int(tail.sum())}",
                          ha="left", va="top", fontsize=FONT_PT["metric"])
            ax = axis_mm(fig, x, header_top+11, column_width, 35.0)
            xy = case["xy_target"]
            ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, norm=norm,
                       s=np.where(tail, 30.0, 15.0),
                       edgecolors=np.where(tail, "black", "white"),
                       linewidths=np.where(tail, 0.8, 0.3), alpha=0.94,
                       rasterized=True, zorder=4)
            ax.scatter(case["xy_input"][:, 0], case["xy_input"][:, 1], marker="^", s=24,
                       facecolor="white", edgecolor="black", linewidth=0.65, zorder=5)
            if case["epicenter_xy"] is not None:
                ax.scatter([case["epicenter_xy"][0]], [case["epicenter_xy"][1]], marker="*", s=58,
                           facecolor="white", edgecolor="black", linewidth=0.7, zorder=6)
            x0, x1, y0, y1 = map_extent(case)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            style_axis(ax, grid=True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(length=2.2, pad=1.6)
            if col == 0:
                ax.set_ylabel("North–south (km)", fontsize=FONT_PT["axis"], labelpad=3)
            else:
                ax.tick_params(labelleft=False)
            if row != 2:
                ax.tick_params(labelbottom=False)
    fig.text((left+plot_width/2)/FIGURE_WIDTH_MM, 1-183.0/SUPPLEMENT_HEIGHT_MM,
             "East–west distance (km)", ha="center", va="center", fontsize=FONT_PT["axis"])
    risk_cax = axis_mm(fig, 30.0, 190.0, 60.0, 3.0)
    corr_cax = axis_mm(fig, 113.0, 190.0, 60.0, 3.0)
    risk_scalar = mpl.cm.ScalarMappable(norm=risk_norm, cmap="viridis")
    corr_scalar = mpl.cm.ScalarMappable(norm=correction_norm, cmap="magma")
    risk_scalar.set_array([]); corr_scalar.set_array([])
    cb1 = fig.colorbar(risk_scalar, cax=risk_cax, orientation="horizontal")
    cb2 = fig.colorbar(corr_scalar, cax=corr_cax, orientation="horizontal",
                       extend=colorbar_extension(all_corrections, 0.0, correction_max))
    cb1.set_label("Underprediction-risk score", fontsize=FONT_PT["axis"], labelpad=2.8)
    cb2.set_label("Applied correction (log$_{10}$ units)", fontsize=FONT_PT["axis"], labelpad=2.8)
    for cb in [cb1, cb2]:
        cb.ax.tick_params(labelsize=FONT_PT["tick"], length=2.2, pad=1.8)
    annotate_demo(fig, cases)
    stem = "DEMO_"+SUPPLEMENT_STEM if cases_are_demo(cases) else SUPPLEMENT_STEM
    save_figure_outputs(fig, out_dir, stem, dpi, {
        "risk_score_range": [0.0, 1.0],
        "correction_scale_max": correction_max,
        "correction_values_above_scale": int((all_corrections > correction_max).sum()),
        "demo": cases_are_demo(cases),
    })
    plt.close(fig)



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


def write_captions(out_dir: Path, prediction_audit: dict[str, Any],
                   selected_table: pd.DataFrame, stats: pd.DataFrame | None = None) -> None:
    selected_lookup = selected_table.set_index("case")
    case_texts = []
    for letter, name in zip(["b", "c", "d"], CASE_ORDER):
        row = selected_lookup.loc[name]
        magnitude = f"M{row['magnitude']:.1f}" if np.isfinite(row["magnitude"]) else "magnitude unavailable"
        case_texts.append(f"({letter}) {CASE_LABELS[name]}: event {row['event_id']} ({magnitude}, repeat {int(row['repeat'])}).")
    if stats is not None:
        tail_units = stats.loc[stats["n_tail_values"].gt(0)]
        finite = np.isfinite(tail_units["non_tail_mae_change"]) & np.isfinite(tail_units["tail_mae_reduction"])
        n_landscape = int(finite.sum())
        n_tail = len(tail_units)
        population = (f"The full locked population comprises {len(stats):,} event–repeat units from "
                      f"{prediction_audit['events']:,} earthquakes. Of these, {n_tail:,} contain at least "
                      f"one high-motion quantity, and {n_landscape:,} have finite paired coordinates in panel (a). ")
    else:
        population = (f"The full locked population comprises {prediction_audit['event_repeat_groups']:,} "
                      f"event–repeat units from {prediction_audit['events']:,} earthquakes; panel (a) displays "
                      "only units with at least one high-motion quantity and finite paired coordinates. ")
    manual_names = prediction_audit.get("manually_requested_cases", [])
    selection_text = ("Cases follow the locked-prediction selection rules, with explicit event/repeat overrides "
                      f"for {', '.join(manual_names)}. " if manual_names else
                      "Cases were selected algorithmically from the locked predictions before inspecting the spatial maps. ")
    caption = (
        "Spatial case studies of prediction preservation, tail correction, and residual failure. "
        "(a) Change in non-tail MAE versus reduction in combined high-motion-tail MAE. "
        "Positive vertical values indicate lower tail error after CA-URC; positive horizontal values indicate larger non-tail error. "
        + population + selection_text + " ".join(case_texts) +
        " The four columns show PGA and PGV residuals (prediction minus observation) for the frozen Cross-Attention Base and CA-URC. "
        "Filled circles denote non-input stations; black outlines identify quantity-specific high-motion-tail targets. "
        "Triangles denote input stations and stars denote catalog epicenters when available. "
        "After case selection, the frozen model was queried at the expanded set of all non-input stations using the same input set, "
        "solely for descriptive visualization. Metric annotations above the maps refer to this expanded set: "
        "N is the total number of mapped targets and n is the quantity-specific tail count. "
        "Tail U0.5 is the fraction of mapped tail targets with residuals at or below -0.5 log10 units. "
        "Empty tail subsets are labeled NA. Values smaller than the displayed precision are not forced to zero. "
        "Map annotations need not equal the locked-target summaries in the text or case-selection landscape. "
        "The color scale is shared across all residual maps; a colorbar extension marks values beyond the displayed range. "
        "Aggregate conclusions remain based on the locked draws; event–repeat units are not independent earthquakes."
    )
    supp_caption = (
        "Underprediction-risk scores and applied corrections in selected spatial cases. "
        "Rows show successful tail correction, residual failure, and non-tail overcorrection, respectively. "
        "Columns show the underprediction-risk score and resulting non-negative applied correction for PGA and PGV. "
        "Scores are not interpreted as calibrated probabilities. All mapped targets belong to the expanded non-input station set; "
        "N and n_tail give the total target count and quantity-specific high-motion-tail count. "
        "Triangles denote input stations, stars denote catalog epicenters, and black target outlines denote tail membership. "
        "Risk panels share the range 0–1. Correction panels share the original 99th-percentile-based display range, "
        "with a colorbar extension identifying values above that range. No predictions are clipped or modified in the source data. "
        "These maps describe spatial model behavior rather than establish causal explanations of correction success or failure."
    )
    if prediction_audit.get("demo", False):
        caption = "SYNTHETIC LAYOUT TEST — NOT RESEARCH RESULTS. " + caption
        supp_caption = "SYNTHETIC LAYOUT TEST — NOT RESEARCH RESULTS. " + supp_caption
    (out_dir/"Fig7_caption.txt").write_text(caption+"\n", encoding="utf-8")
    (out_dir/"FigS2_caption.txt").write_text(supp_caption+"\n", encoding="utf-8")



def format_ci_free_value(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def write_manuscript_draft(out_dir: Path, selected_table: pd.DataFrame,
                           failure_summary: pd.DataFrame) -> None:
    """Numerical summaries only; do not auto-assert physical causes from maps."""
    lookup = selected_table.set_index("case")
    fail = failure_summary.set_index("criterion")
    rep, success = lookup.loc["representative"], lookup.loc["successful_correction"]
    failure, over = lookup.loc["residual_failure"], lookup.loc["overcorrection"]
    helped = fail.loc["CA-URC reduces combined tail MAE"]
    not_helped = fail.loc["CA-URC does not reduce combined tail MAE"]
    remained = fail.loc["at least one severe tail underprediction remains"]
    worse = fail.loc["non-tail MAE increases by >0.02 log10 units"]
    text = f"""Spatial case studies: numerical draft for review

All numbers below use locked target draws, not the expanded map station sets. Case identities are taken from the actual selection output; verify any manual overrides before describing case selection as automatic.

For the representative-preservation case (event {rep['event_id']}, M{rep['magnitude']:.1f}, repeat {int(rep['repeat'])}), locked combined overall MAE changed from {rep['locked_base_overall_mae']:.3f} to {rep['locked_caurc_overall_mae']:.3f} log10 units. This selection label does not guarantee improvement in every quantity or at every mapped target.

For the successful-correction case (event {success['event_id']}, M{success['magnitude']:.1f}, repeat {int(success['repeat'])}), locked combined tail MAE decreased from {success['locked_base_tail_mae']:.3f} to {success['locked_caurc_tail_mae']:.3f}, a reduction of {success['locked_tail_mae_reduction']:.3f} log10 units, while locked non-tail MAE changed by {success['locked_non_tail_mae_change']:+.3f}.

For the residual-failure case (event {failure['event_id']}, M{failure['magnitude']:.1f}, repeat {int(failure['repeat'])}), locked combined tail MAE after correction was {failure['locked_caurc_tail_mae']:.3f} log10 units. The available spatial diagnostics do not by themselves isolate the roles of the observations, training data, model representation, or correction constraints.

Across {int(helped['count']+not_helped['count'])} locked event–repeat units containing at least one high-motion quantity, combined tail MAE decreased in {100*helped['fraction']:.1f}%; at least one severe tail underprediction remained in {100*remained['fraction']:.1f}%. Across all locked event–repeat units, non-tail MAE increased by more than 0.02 log10 units in {100*worse['fraction']:.1f}%. These frequencies concern repeated station configurations, not independent earthquakes. The supplementary overcorrection case is event {over['event_id']}, M{over['magnitude']:.1f}, repeat {int(over['repeat'])}.

Consult Fig7_map_metric_audit.csv for unrounded per-quantity expanded-map statistics, tail counts, and flags identifying values that would round to 0.00 in the original two-decimal display. Manuscript figure and table numbers should be assigned by the final LaTeX document.
"""
    (out_dir/"Spatial_cases_manuscript_draft.txt").write_text(text, encoding="utf-8")



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
    parser.add_argument("--out-dir", default="figures/jgr_se_spatial_cases")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    if args.demo:
        out_dir = out_dir / "demo_preview"  # Never overwrite real figures with synthetic output.
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        stats, selections, cases = demo_statistics()
        prediction_audit = {
            "prediction_rows": int(stats["n_targets"].sum()),
            "events": int(stats["event_id"].nunique()),
            "event_repeat_groups": int(len(stats)),
            "pga_tail_rows": None,  # Synthetic statistics do not instantiate a locked CSV.
            "pgv_tail_rows": None,
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

    prediction_audit["manually_requested_cases"] = [
        label for label in ["representative", "success", "failure", "overcorrection"]
        if getattr(args, f"event_{label}") is not None
    ]
    stats.to_csv(out_dir / "case_selection_statistics.csv", index=False)
    failure_summary = failure_mode_summary(stats)
    failure_summary.to_csv(out_dir / "TableS7_failure_mode_summary.csv", index=False)
    output_tex(failure_summary, out_dir / "TableS7_failure_mode_summary.tex")

    selected_table = selected_case_table(selections, cases)
    selected_table.to_csv(out_dir / "TableS7_selected_cases.csv", index=False)
    output_tex(selected_table, out_dir / "TableS7_selected_cases.tex")

    plot_main_figure(stats, selections, cases, out_dir, args.dpi)
    plot_risk_correction_supplement(cases, out_dir, args.dpi)
    write_captions(out_dir, prediction_audit, selected_table, stats=stats)
    export_map_source_data(cases, out_dir)
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

    print("=== JGR-SE spatial case studies ===")
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
        "Fig7_spatial_cases.png",
        "Fig7_spatial_cases.pdf",
        "Fig7_spatial_cases.svg",
        "FigS2_selected_case_risk_and_correction.png",
        "FigS2_selected_case_risk_and_correction.pdf",
        "TableS7_selected_cases.csv",
        "TableS7_failure_mode_summary.csv",
        "Spatial_cases_manuscript_draft.txt",
        "Fig7_caption.txt",
        "FigS2_caption.txt",
        "case_selection_audit.json",
        "Fig7_map_metric_audit.csv",
        "Fig7_source_data_map_targets.csv",
    ]:
        if args.demo and name.endswith((".png", ".pdf", ".svg")):
            name = "DEMO_" + name
        print(f"  {(out_dir / name).resolve()}")


if __name__ == "__main__":
    main()

