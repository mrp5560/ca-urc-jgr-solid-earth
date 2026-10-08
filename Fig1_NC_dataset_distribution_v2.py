#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig1_NC_dataset_distribution.py

Nature Communications-style four-panel dataset figure for the CA-URC study.

Panels
------
a  Southern California event and station distribution.
b  Magnitude distributions of all preprocessed and causally eligible events.
c  Monthly event counts from 2010 to 2024, highlighting the Ridgecrest sequence.
d  Final station availability versus P-wave-reached station availability at T0.

Primary protocol defaults
-------------------------
T0 = 5 s, K = 5 input stations, Q = 10 target stations.

Panel a uses Cartopy and an optional local ETOPO/GEBCO GeoTIFF to render a
publication-quality coloured shaded-relief map. Event symbols, station symbols,
labels and map linework remain vector objects in PDF/SVG, whereas the relief is
embedded as a raster layer. The script reads event metadata, station coordinates
and P-arrival offsets from the final HDF5 archive and writes source-data CSVs, a
caption file and a JSON audit trail.

Example (Windows)
-----------------
D:\\python3\\python.exe Fig1_NC_dataset_distribution.py ^
  --master-manifest auto ^
  --scenario-manifest auto ^
  --h5-root auto ^
  --strict-count-audit ^
  --out-dir figures\\nc_dataset

Example (Linux)
---------------
python3 Fig1_NC_dataset_distribution.py \
  --master-manifest auto \
  --scenario-manifest auto \
  --h5-root auto \
  --strict-count-audit \
  --out-dir figures/nc_dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
except ImportError:  # pragma: no cover - checked at runtime
    ccrs = None
    cfeature = None
    LATITUDE_FORMATTER = None
    LONGITUDE_FORMATTER = None

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
except ImportError:  # pragma: no cover - checked only when a relief raster is used
    rasterio = None
    Resampling = None
    from_bounds = None
    reproject = None


# -----------------------------------------------------------------------------
# Publication style
# -----------------------------------------------------------------------------

MM_TO_INCH = 1.0 / 25.4
FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 144.0

COLORS = {
    # Bright, colour-blind-aware palette with good RGB/CMYK separation.
    # Keep the basemap restrained and let the scientific overlays carry colour.
    "all": "#B5BCC5",          # light neutral grey
    "eligible": "#0077BB",     # vivid blue
    "ridgecrest": "#EE7733",   # vivid orange
    "station": "#232629",      # near-black
    "threshold": "#666A70",    # neutral dark grey
    "trend": "#009988",        # teal
    "grid": "#E2E6EA",
    "text": "#1F2328",
}


def configure_matplotlib() -> None:
    """Configure a compact, editable, journal-style Matplotlib theme."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # Nature-family final-size typography: body text 5--7 pt, panel labels 8 pt.
            "font.size": 6.2,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.titlepad": 3.0,
            "xtick.major.width": 0.60,
            "ytick.major.width": 0.60,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "lines.linewidth": 0.95,
            "patch.linewidth": 0.65,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
        }
    )


# -----------------------------------------------------------------------------
# Paths, schemas, and audit helpers
# -----------------------------------------------------------------------------

MASTER_CANDIDATES = [
    "data/scedc/model_manifests/all_events_with_scenario_flags.csv",
    "data/model_manifests/all_events_with_scenario_flags.csv",
    "data/scedc/splits_v4/event_splits_v4.csv",
    "data/splits_v4/event_splits_v4.csv",
]

SCENARIO_CANDIDATES = [
    "data/scedc/model_manifests/scenario_t0_5s_k5.csv",
    "data/model_manifests/scenario_t0_5s_k5.csv",
    "data/scedc/model_manifests/scenario_t0_5s_k5_linux.csv",
    "data/model_manifests/scenario_t0_5s_k5_linux.csv",
]

H5_ROOT_CANDIDATES = [
    "data/processed_full_v4/events",
    "data/scedc/processed_full_v4/events",
]

# Supply a local geographic relief raster for the publication version. ETOPO 2022
# or GEBCO in a georeferenced GeoTIFF is recommended. Only the plotted region is
# reprojected into memory, so a global Cloud-Optimized GeoTIFF can also be used.
RELIEF_RASTER_CANDIDATES = [
    "data/basemaps/ETOPO_2022_v1_15s_N90W180_surface.tif",
    "data/basemaps/ETOPO_2022_v1_30s_N90W180_surface.tif",
    "data/basemaps/ETOPO_2022_v1_60s_N90W180_surface.tif",
    "data/basemaps/GEBCO_2024.tif",
    "data/topography/ETOPO_2022.tif",
    "data/topography/GEBCO.tif",
]

RELIEF_CMAP = LinearSegmentedColormap.from_list(
    "nc_hypsometric_relief",
    [
        (0.00, "#071B39"),
        (0.15, "#0B4772"),
        (0.31, "#1478A6"),
        (0.44, "#55B6D1"),
        (0.495, "#D5EDF2"),
        (0.505, "#D7E3BE"),
        (0.60, "#8BA86A"),
        (0.69, "#63884E"),
        (0.79, "#B5A16E"),
        (0.89, "#8A6849"),
        (1.00, "#EEEAE0"),
    ],
    N=512,
)

EVENT_COLUMN_ALIASES = {
    "event_id": ["event_id", "evid", "event", "id"],
    "magnitude": ["magnitude", "mag", "catalog_magnitude", "event_magnitude", "M"],
    "origin_time": ["origin_time", "event_time", "time", "datetime", "date_time"],
    "latitude": ["latitude", "lat", "event_latitude", "epicenter_latitude"],
    "longitude": ["longitude", "lon", "lng", "event_longitude", "epicenter_longitude"],
    "depth_km": ["depth_km", "depth", "event_depth_km"],
}


@dataclass
class AuditSummary:
    master_manifest: str
    scenario_manifest: str | None
    h5_root: str
    n_master_rows: int
    n_h5_resolved: int
    n_h5_missing: int
    n_preprocessed_events: int
    n_eligible_events: int
    n_stations: int
    t0_sec: float
    input_stations: int
    target_stations: int
    expected_preprocessed: int
    expected_eligible: int
    ridgecrest_identification: str
    time_start: str | None
    time_end: str | None
    magnitude_min: float | None
    magnitude_max: float | None
    relief_raster: str | None
    map_extent: list[float]


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_auto_path(value: str, candidates: list[str], label: str, required: bool = True) -> Path | None:
    if str(value).lower() != "auto":
        path = Path(value)
        if path.exists():
            return path.resolve()
        if required:
            raise FileNotFoundError(f"{label} not found: {path}")
        return None

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path.resolve()

    if required:
        raise FileNotFoundError(
            f"Could not auto-resolve {label}. Tried:\n  " + "\n  ".join(candidates)
        )
    return None


def first_existing_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def resolve_h5_path(event_id: str, row: pd.Series, h5_root: Path, recursive_index: dict[str, Path]) -> Path | None:
    raw = str(row.get("h5_path", "")).strip()
    if raw and raw.lower() not in {"nan", "none", ""}:
        direct = Path(raw)
        for candidate in (direct, Path.cwd() / direct):
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

    direct_candidates = [
        h5_root / f"{event_id}.h5",
        h5_root / str(event_id) / f"{event_id}.h5",
    ]
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return recursive_index.get(str(event_id))


def build_recursive_h5_index(h5_root: Path) -> dict[str, Path]:
    print(f"Indexing HDF5 files under: {h5_root}")
    result: dict[str, Path] = {}
    for path in h5_root.rglob("*.h5"):
        result.setdefault(path.stem, path.resolve())
    print(f"Indexed {len(result):,} HDF5 files.")
    return result


def safe_attr(h5: h5py.File, key: str, default: Any = np.nan) -> Any:
    value = h5.attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def as_datetime(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


# -----------------------------------------------------------------------------
# Data extraction
# -----------------------------------------------------------------------------


def normalize_master_columns(master: pd.DataFrame) -> pd.DataFrame:
    frame = master.copy()
    event_col = first_existing_column(frame, EVENT_COLUMN_ALIASES["event_id"])
    if event_col is None:
        raise ValueError("Master manifest lacks an event identifier column.")
    if event_col != "event_id":
        frame = frame.rename(columns={event_col: "event_id"})
    frame["event_id"] = frame["event_id"].astype(str).str.strip()

    for canonical in ("magnitude", "origin_time", "latitude", "longitude", "depth_km"):
        source = first_existing_column(frame, EVENT_COLUMN_ALIASES[canonical])
        if source is not None and source != canonical:
            frame = frame.rename(columns={source: canonical})

    if "origin_time" in frame:
        frame["origin_time"] = pd.to_datetime(frame["origin_time"], utc=True, errors="coerce")
    for column in ("magnitude", "latitude", "longitude", "depth_km"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["event_id"].duplicated().any():
        duplicates = frame.loc[frame["event_id"].duplicated(), "event_id"].head().tolist()
        raise ValueError(f"Master manifest contains duplicate event IDs, for example: {duplicates}")
    return frame


def scenario_event_ids(
    master: pd.DataFrame,
    scenario_path: Path | None,
    t0_sec: float,
    input_stations: int,
) -> tuple[set[str] | None, str]:
    if scenario_path is not None:
        scenario = pd.read_csv(scenario_path, dtype={"event_id": str})
        event_col = first_existing_column(scenario, EVENT_COLUMN_ALIASES["event_id"])
        if event_col is None:
            raise ValueError(f"Scenario manifest lacks event_id: {scenario_path}")
        ids = set(scenario[event_col].astype(str).str.strip())
        return ids, f"scenario manifest: {scenario_path}"

    eligible_column = f"eligible_t0_{int(t0_sec)}s_k{int(input_stations)}"
    if eligible_column in master.columns:
        ids = set(master.loc[parse_bool(master[eligible_column]), "event_id"].astype(str))
        return ids, f"master column: {eligible_column}"

    return None, "computed from HDF5 station counts"


def infer_ridgecrest_mask(frame: pd.DataFrame, mode: str) -> tuple[pd.Series, str]:
    false_mask = pd.Series(False, index=frame.index)
    if mode == "none":
        return false_mask, "disabled"

    # Prefer an explicit Ridgecrest-labelled column.
    for column in frame.columns:
        if "ridgecrest" not in str(column).lower():
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or set(series.dropna().astype(str).str.lower().unique()).issubset(
            {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}
        ):
            mask = parse_bool(series)
        else:
            mask = series.astype(str).str.contains("ridgecrest", case=False, na=False)
        if mask.any():
            return mask, f"explicit column: {column}"

    # Then inspect generic split/group/name columns.
    for column in frame.columns:
        name = str(column).lower()
        if not any(token in name for token in ("split", "sequence", "cluster", "group", "region", "name")):
            continue
        mask = frame[column].astype(str).str.contains("ridgecrest", case=False, na=False)
        if mask.any():
            return mask, f"label values in column: {column}"

    if mode == "column":
        raise ValueError("--ridgecrest-mode=column was requested, but no Ridgecrest label was found.")

    # Transparent fallback used only when no explicit label exists.
    if "origin_time" not in frame or "latitude" not in frame or "longitude" not in frame:
        return false_mask, "not identified; insufficient metadata for fallback"

    start = pd.Timestamp("2019-07-01", tz="UTC")
    end = pd.Timestamp("2019-10-01", tz="UTC")
    mask = (
        frame["origin_time"].between(start, end, inclusive="left")
        & frame["latitude"].between(34.0, 36.5, inclusive="both")
        & frame["longitude"].between(-118.6, -116.0, inclusive="both")
    )
    return mask, "transparent spatiotemporal fallback: 2019-07-01 to 2019-10-01, 34.0–36.5°N, 118.6–116.0°W"


def collect_event_and_station_metadata(
    master: pd.DataFrame,
    h5_root: Path,
    eligible_ids: set[str] | None,
    t0_sec: float,
    input_stations: int,
    target_stations: int,
    station_scope: str,
    allow_missing_h5: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    recursive_index = build_recursive_h5_index(h5_root)
    event_rows: list[dict[str, Any]] = []
    missing_events: list[str] = []
    station_records: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    total = len(master)
    for position, (_, row) in enumerate(master.iterrows(), start=1):
        event_id = str(row["event_id"]).strip()
        h5_path = resolve_h5_path(event_id, row, h5_root, recursive_index)
        if h5_path is None:
            missing_events.append(event_id)
            if not allow_missing_h5:
                continue
            else:
                continue

        try:
            with h5py.File(h5_path, "r") as h5:
                p_offset = np.asarray(h5["p_offset_sec"][:], dtype=np.float64)
                n_stations = int(len(p_offset))
                n_triggered = int(
                    np.sum(
                        np.isfinite(p_offset)
                        & (p_offset >= -1e-3)
                        & (p_offset <= float(t0_sec))
                    )
                )

                magnitude = as_float(row.get("magnitude", np.nan))
                if not np.isfinite(magnitude):
                    magnitude = as_float(safe_attr(h5, "magnitude"))

                latitude = as_float(row.get("latitude", np.nan))
                if not np.isfinite(latitude):
                    latitude = as_float(safe_attr(h5, "latitude"))

                longitude = as_float(row.get("longitude", np.nan))
                if not np.isfinite(longitude):
                    longitude = as_float(safe_attr(h5, "longitude"))

                depth_km = as_float(row.get("depth_km", np.nan))
                if not np.isfinite(depth_km):
                    depth_km = as_float(safe_attr(h5, "depth_km"))

                origin_time = row.get("origin_time", pd.NaT)
                if pd.isna(origin_time):
                    origin_time = as_datetime(safe_attr(h5, "origin_time", pd.NaT))

                eligible = (
                    event_id in eligible_ids
                    if eligible_ids is not None
                    else (
                        n_triggered >= int(input_stations)
                        and n_stations - int(input_stations) >= int(target_stations)
                    )
                )

                event_rows.append(
                    {
                        "event_id": event_id,
                        "h5_path": str(h5_path),
                        "origin_time": origin_time,
                        "magnitude": magnitude,
                        "latitude": latitude,
                        "longitude": longitude,
                        "depth_km": depth_km,
                        "n_valid_stations": n_stations,
                        "n_p_reached_t0": n_triggered,
                        "eligible": bool(eligible),
                    }
                )

                include_stations = station_scope == "all" or (station_scope == "eligible" and eligible)
                if include_stations:
                    coords = np.asarray(h5["station_coords"][:], dtype=np.float64)
                    if "station_id" in h5:
                        try:
                            station_ids = h5["station_id"].asstr()[:]
                        except Exception:
                            raw_ids = h5["station_id"][:]
                            station_ids = [
                                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
                                for value in raw_ids
                            ]
                    else:
                        station_ids = [f"unknown_{event_id}_{index}" for index in range(len(coords))]

                    for station_id, coordinate in zip(station_ids, coords):
                        if len(coordinate) < 2:
                            continue
                        lat = as_float(coordinate[0])
                        lon = as_float(coordinate[1])
                        elev = as_float(coordinate[2]) if len(coordinate) >= 3 else float("nan")
                        if np.isfinite(lat) and np.isfinite(lon):
                            station_records[str(station_id)].append((lat, lon, elev))

        except Exception as exc:
            missing_events.append(event_id)
            print(f"WARNING: failed to read {event_id}: {exc}", file=sys.stderr)

        if position % 250 == 0 or position == total:
            print(f"Metadata scan: {position:,}/{total:,} events")

    events = pd.DataFrame(event_rows)
    if events.empty:
        raise RuntimeError("No readable event HDF5 files were found.")

    station_rows: list[dict[str, Any]] = []
    for station_id, coordinates in station_records.items():
        array = np.asarray(coordinates, dtype=np.float64)
        station_rows.append(
            {
                "station_id": station_id,
                "latitude": float(np.nanmedian(array[:, 0])),
                "longitude": float(np.nanmedian(array[:, 1])),
                "elevation_m": float(np.nanmedian(array[:, 2])) if array.shape[1] >= 3 else np.nan,
                "n_event_records": int(len(array)),
            }
        )

    stations = pd.DataFrame(station_rows)
    if not stations.empty:
        stations = stations.sort_values("station_id").reset_index(drop=True)

    return events, stations, sorted(set(missing_events))


# -----------------------------------------------------------------------------
# Figure source data
# -----------------------------------------------------------------------------


def build_monthly_counts(events: pd.DataFrame) -> pd.DataFrame:
    valid = events.dropna(subset=["origin_time"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["month", "all_preprocessed", "causal_eligible", "ridgecrest"])

    start = valid["origin_time"].min().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    end = valid["origin_time"].max().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    months = pd.date_range(start, end, freq="MS", tz="UTC")

    valid["month"] = (
        valid["origin_time"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    )
    all_counts = valid.groupby("month").size().reindex(months, fill_value=0)
    eligible_counts = valid.loc[valid["eligible"]].groupby("month").size().reindex(months, fill_value=0)
    ridgecrest_counts = valid.loc[valid["ridgecrest"]].groupby("month").size().reindex(months, fill_value=0)

    return pd.DataFrame(
        {
            "month": months,
            "all_preprocessed": all_counts.to_numpy(dtype=int),
            "causal_eligible": eligible_counts.to_numpy(dtype=int),
            "ridgecrest": ridgecrest_counts.to_numpy(dtype=int),
        }
    )


def magnitude_source(events: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    values = events["magnitude"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "population", "count", "percent"])

    left = math.floor(values.min() / bin_width) * bin_width
    right = math.ceil(values.max() / bin_width) * bin_width + bin_width
    bins = np.arange(left, right + 0.5 * bin_width, bin_width)

    rows: list[dict[str, Any]] = []
    populations = {
        "all_preprocessed": events,
        "causal_eligible": events.loc[events["eligible"]],
    }
    for name, frame in populations.items():
        data = frame["magnitude"].dropna().to_numpy(dtype=float)
        counts, edges = np.histogram(data, bins=bins)
        denominator = max(len(data), 1)
        for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
            rows.append(
                {
                    "bin_left": float(lo),
                    "bin_right": float(hi),
                    "bin_center": float((lo + hi) / 2.0),
                    "population": name,
                    "count": int(count),
                    "percent": float(100.0 * count / denominator),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plot utilities
# -----------------------------------------------------------------------------


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    """Nature-style full frame with outward ticks and minimal background decoration."""
    # Full rectangular frame requested by the manuscript layout.  Keep ticks only
    # on the left/bottom so the frame reads as a boundary rather than a boxed grid.
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.65)
        ax.spines[side].set_color(COLORS["text"])

    # Nature's figure guide discourages background gridlines.  Retain this switch
    # only for exceptional cases; the primary b/c/d panels call it with None.
    if grid_axis is not None:
        ax.grid(
            True,
            axis=grid_axis,
            color=COLORS["grid"],
            linewidth=0.35,
            alpha=0.55,
            zorder=0,
        )
    else:
        ax.grid(False)

    ax.tick_params(
        direction="out",
        top=False,
        right=False,
        pad=2.2,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.115,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )



def resolve_relief_raster(value: str) -> Path | None:
    """Resolve a local ETOPO/GEBCO GeoTIFF. 'none' disables relief."""
    normalized = str(value).strip().lower()
    if normalized in {"none", "off", "false", "0", ""}:
        return None
    if normalized != "auto":
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Relief raster not found: {path}")
        return path.resolve()
    for candidate in RELIEF_RASTER_CANDIDATES:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def parse_map_extent(text: str, events: pd.DataFrame, stations: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return west, east, south, north in degrees."""
    if str(text).strip().lower() != "auto":
        values = [float(item.strip()) for item in str(text).split(",")]
        if len(values) != 4:
            raise ValueError("--map-extent must be 'auto' or west,east,south,north")
        west, east, south, north = values
        if not (west < east and south < north):
            raise ValueError("Invalid --map-extent ordering.")
        return west, east, south, north

    lon_parts = [events["longitude"].to_numpy(dtype=float)]
    lat_parts = [events["latitude"].to_numpy(dtype=float)]
    if not stations.empty:
        lon_parts.append(stations["longitude"].to_numpy(dtype=float))
        lat_parts.append(stations["latitude"].to_numpy(dtype=float))
    longitude = np.concatenate(lon_parts)
    latitude = np.concatenate(lat_parts)
    longitude = longitude[np.isfinite(longitude)]
    latitude = latitude[np.isfinite(latitude)]
    if len(longitude) == 0 or len(latitude) == 0:
        raise RuntimeError("Cannot infer map extent from missing coordinates.")

    lon_span = max(float(np.ptp(longitude)), 1.0)
    lat_span = max(float(np.ptp(latitude)), 1.0)
    west = float(np.min(longitude) - max(0.18, 0.035 * lon_span))
    east = float(np.max(longitude) + max(0.18, 0.035 * lon_span))
    south = float(np.min(latitude) - max(0.15, 0.035 * lat_span))
    north = float(np.max(latitude) + max(0.15, 0.035 * lat_span))
    return west, east, south, north


def relief_rgb_from_geotiff(
    path: Path,
    extent: tuple[float, float, float, float],
    output_width: int,
    minimum_elevation: float,
    maximum_elevation: float,
    vertical_exaggeration: float,
) -> np.ndarray:
    """Read only the map region and produce a coloured hill-shaded RGB image."""
    if rasterio is None or reproject is None or from_bounds is None:
        raise ImportError(
            "A relief raster was requested, but rasterio is unavailable. "
            "Install it with: python -m pip install rasterio"
        )
    west, east, south, north = extent
    mean_latitude = 0.5 * (south + north)
    map_width_km = max((east - west) * 111.32 * math.cos(math.radians(mean_latitude)), 1.0)
    map_height_km = max((north - south) * 110.57, 1.0)
    output_width = int(np.clip(output_width, 700, 3000))
    output_height = int(np.clip(round(output_width * map_height_km / map_width_km), 500, 2600))

    destination = np.full((output_height, output_width), np.nan, dtype=np.float32)
    destination_transform = from_bounds(west, south, east, north, output_width, output_height)

    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f"Relief raster has no coordinate reference system: {path}")
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=destination_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=2,
        )

    invalid = ~np.isfinite(destination)
    finite = destination[~invalid]
    if finite.size == 0:
        raise RuntimeError(f"Relief raster does not overlap map extent {extent}: {path}")
    work = destination.copy()
    work[invalid] = float(np.nanmedian(finite))
    work = np.clip(work, minimum_elevation, maximum_elevation)

    normalization = TwoSlopeNorm(
        vmin=float(minimum_elevation),
        vcenter=0.0,
        vmax=float(maximum_elevation),
    )
    base_rgb = RELIEF_CMAP(normalization(work))[..., :3]
    dx_m = max((east - west) * 111_320.0 * math.cos(math.radians(mean_latitude)) / output_width, 1.0)
    dy_m = max((north - south) * 110_570.0 / output_height, 1.0)
    light = LightSource(azdeg=315, altdeg=42)
    rgb = light.shade_rgb(
        base_rgb,
        work,
        vert_exag=float(vertical_exaggeration),
        dx=dx_m,
        dy=dy_m,
        blend_mode="soft",
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    if np.any(invalid):
        rgb[invalid] = np.array([1.0, 1.0, 1.0])
    return rgb


def add_geographic_scale_bar(
    ax,
    extent: tuple[float, float, float, float],
    length_km: float = 100.0,
) -> None:
    """Add an approximately geodesic horizontal scale bar to a PlateCarree map."""
    west, east, south, north = extent
    latitude = south + 0.075 * (north - south)
    right = east - 0.055 * (east - west)
    degree_length = length_km / (111.32 * max(math.cos(math.radians(latitude)), 0.2))
    left = right - degree_length
    cap = 0.012 * (north - south)
    transform = ccrs.PlateCarree()
    ax.plot([left, right], [latitude, latitude], color="white", lw=2.8, transform=transform, zorder=20)
    ax.plot([left, right], [latitude, latitude], color=COLORS["text"], lw=1.15, transform=transform, zorder=21)
    for x in (left, right):
        ax.plot([x, x], [latitude - cap, latitude + cap], color="white", lw=2.5, transform=transform, zorder=20)
        ax.plot([x, x], [latitude - cap, latitude + cap], color=COLORS["text"], lw=0.85, transform=transform, zorder=21)
    ax.text(
        0.5 * (left + right),
        latitude + 1.7 * cap,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color=COLORS["text"],
        transform=transform,
        zorder=22,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.72),
    )


def magnitude_marker_sizes(magnitude: np.ndarray) -> np.ndarray:
    magnitude = np.asarray(magnitude, dtype=float)
    finite = magnitude[np.isfinite(magnitude)]
    fallback = float(np.nanmedian(finite)) if len(finite) else 3.0
    m = np.where(np.isfinite(magnitude), magnitude, fallback)
    return 4.0 + 2.5 * np.clip(m - 3.0, 0.0, 5.0) ** 1.55


def add_scale_bar(ax: plt.Axes, length_km: float = 100.0) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    latitude = y0 + 0.075 * (y1 - y0)
    longitude = x1 - 0.08 * (x1 - x0)
    degree_length = length_km / (111.32 * math.cos(math.radians(latitude)))
    longitude = longitude - degree_length
    ax.plot(
        [longitude, longitude + degree_length],
        [latitude, latitude],
        color=COLORS["text"],
        linewidth=1.3,
        solid_capstyle="butt",
        zorder=10,
    )
    cap = 0.012 * (y1 - y0)
    ax.plot([longitude, longitude], [latitude - cap, latitude + cap], color=COLORS["text"], lw=0.8)
    ax.plot(
        [longitude + degree_length, longitude + degree_length],
        [latitude - cap, latitude + cap],
        color=COLORS["text"],
        lw=0.8,
    )
    ax.text(
        longitude + 0.5 * degree_length,
        latitude + 1.8 * cap,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=6.3,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.93),
        xytext=(0.94, 0.82),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=0.85, color=COLORS["text"]),
    )


def plot_panel_a(
    ax,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    map_extent: tuple[float, float, float, float],
    relief_raster: Path | None,
    relief_pixels: int,
    relief_min_elevation: float,
    relief_max_elevation: float,
    relief_vertical_exaggeration: float,
) -> None:
    if ccrs is None or cfeature is None:
        raise ImportError(
            "The coloured geographic panel requires Cartopy. Install it with: "
            "python -m pip install cartopy"
        )

    map_events = events.dropna(subset=["longitude", "latitude"]).copy()
    if map_events.empty:
        raise RuntimeError("No finite event coordinates available for panel a.")
    eligible = map_events.loc[map_events["eligible"] & ~map_events["ridgecrest"]]
    ridgecrest = map_events.loc[map_events["eligible"] & map_events["ridgecrest"]]
    transform = ccrs.PlateCarree()
    ax.set_extent(map_extent, crs=transform)

    if relief_raster is not None:
        rgb = relief_rgb_from_geotiff(
            relief_raster,
            map_extent,
            output_width=relief_pixels,
            minimum_elevation=relief_min_elevation,
            maximum_elevation=relief_max_elevation,
            vertical_exaggeration=relief_vertical_exaggeration,
        )
        # Slightly wash out the relief so bright event overlays remain dominant.
        rgb = np.clip(0.90 * rgb + 0.10, 0.0, 1.0)
        ax.imshow(
            rgb,
            origin="upper",
            extent=map_extent,
            transform=transform,
            interpolation="bilinear",
            zorder=0,
        )
    else:
        # Reproducible fallback if no local ETOPO/GEBCO raster is supplied.
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#CDE7F0", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#D9E0C2", zorder=0)

    # Geographic context. Natural Earth linework is cached by Cartopy after the
    # first run. Keep it subtle so the scientific symbols remain dominant.
    ax.coastlines(resolution="10m", color="#2C3338", linewidth=0.52, zorder=5)
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        edgecolor="#4A5157",
        linewidth=0.38,
        linestyle=(0, (2.5, 1.8)),
        facecolor="none",
        zorder=5,
    )
    state_lines = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(state_lines, edgecolor="#61676D", linewidth=0.30, alpha=0.78, zorder=5)

    # Plot the larger, lower-priority population first.
    ax.scatter(
        map_events["longitude"],
        map_events["latitude"],
        s=magnitude_marker_sizes(map_events["magnitude"]) * 0.50,
        marker="o",
        facecolor="#D7DCE1",
        edgecolor="#59616A",
        linewidth=0.18,
        alpha=0.48,
        transform=transform,
        zorder=7,
    )
    ax.scatter(
        eligible["longitude"],
        eligible["latitude"],
        s=magnitude_marker_sizes(eligible["magnitude"]),
        marker="o",
        facecolor=COLORS["eligible"],
        edgecolor="white",
        linewidth=0.28,
        alpha=0.92,
        transform=transform,
        zorder=8,
    )
    if not ridgecrest.empty:
        ax.scatter(
            ridgecrest["longitude"],
            ridgecrest["latitude"],
            s=magnitude_marker_sizes(ridgecrest["magnitude"]) * 1.10,
            marker="o",
            facecolor=COLORS["ridgecrest"],
            edgecolor="white",
            linewidth=0.32,
            alpha=0.96,
            transform=transform,
            zorder=9,
        )
    if not stations.empty:
        ax.scatter(
            stations["longitude"],
            stations["latitude"],
            s=8.8,
            marker="^",
            facecolor=COLORS["station"],
            edgecolor="white",
            linewidth=0.30,
            alpha=0.82,
            transform=transform,
            zorder=10,
        )

    west, east, south, north = map_extent
    lon_step = 1.0 if (east - west) <= 9.0 else 2.0
    lat_step = 1.0 if (north - south) <= 8.0 else 2.0
    x_ticks = np.arange(math.ceil(west / lon_step) * lon_step, east + 1e-6, lon_step)
    y_ticks = np.arange(math.ceil(south / lat_step) * lat_step, north + 1e-6, lat_step)
    gridliner = ax.gridlines(
        crs=transform,
        draw_labels=True,
        x_inline=False,
        y_inline=False,
        linewidth=0.28,
        color="white" if relief_raster is not None else "#8C949B",
        alpha=0.42,
        linestyle=(0, (1.5, 2.4)),
        zorder=4,
    )
    gridliner.top_labels = False
    gridliner.right_labels = False
    gridliner.xlocator = mticker.FixedLocator(x_ticks)
    gridliner.ylocator = mticker.FixedLocator(y_ticks)
    gridliner.xformatter = LONGITUDE_FORMATTER
    gridliner.yformatter = LATITUDE_FORMATTER
    gridliner.xlabel_style = {"size": 5.8, "color": COLORS["text"]}
    gridliner.ylabel_style = {"size": 5.8, "color": COLORS["text"]}
    gridliner.xpadding = 2.0
    gridliner.ypadding = 2.0

    ax.set_title("Event and station coverage", loc="left", fontweight="bold")
    if "geo" in ax.spines:
        ax.spines["geo"].set_linewidth(0.70)
        ax.spines["geo"].set_edgecolor(COLORS["text"])
    add_geographic_scale_bar(ax, map_extent, length_km=100.0)
    add_north_arrow(ax)

    handles = [
        Line2D([], [], marker="^", linestyle="none", markerfacecolor=COLORS["station"], markeredgecolor="white", markersize=5.0, label=f"Stations (n={len(stations):,})"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#D7DCE1", markeredgecolor="#59616A", markeredgewidth=0.25, markersize=4.2, label=f"All preprocessed (n={len(events):,})"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["eligible"], markeredgecolor="white", markersize=4.5, label=f"Causally eligible (n={int(events['eligible'].sum()):,})"),
    ]
    if events["ridgecrest"].any():
        handles.append(
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["ridgecrest"], markeredgecolor="white", markersize=4.5, label="Ridgecrest sequence")
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#C5CAD0",
        framealpha=0.94,
        borderpad=0.28,
        borderaxespad=0.22,
        handletextpad=0.38,
        labelspacing=0.22,
    )
    panel_label(ax, "a")

def plot_panel_b(ax: plt.Axes, magnitude_table: pd.DataFrame, events: pd.DataFrame) -> None:
    styles = (
        ("all_preprocessed", COLORS["all"], (0, (3.2, 2.0)), 0.90, f"All preprocessed (n={len(events):,})"),
        ("causal_eligible", COLORS["eligible"], "-", 1.00, f"Causally eligible (n={int(events['eligible'].sum()):,})"),
    )
    for population, color, linestyle, linewidth, label in styles:
        subset = magnitude_table.loc[magnitude_table["population"] == population]
        if subset.empty:
            continue
        x = subset["bin_center"].to_numpy(dtype=float)
        y = subset["percent"].to_numpy(dtype=float)
        ax.step(
            x, y, where="mid", color=color, linestyle=linestyle,
            linewidth=linewidth, label=label, zorder=3
        )

    all_mag = events["magnitude"].dropna()
    eligible_mag = events.loc[events["eligible"], "magnitude"].dropna()
    if len(all_mag):
        ax.axvline(all_mag.median(), color=COLORS["all"], lw=0.70, linestyle=(0, (1.6, 1.6)), alpha=0.95)
    if len(eligible_mag):
        ax.axvline(eligible_mag.median(), color=COLORS["eligible"], lw=0.75, linestyle=(0, (1.6, 1.6)), alpha=0.95)

    ax.set_xlabel("Catalogue magnitude")
    ax.set_ylabel("Events per bin (%)")
    ax.set_title("Magnitude composition", loc="left", fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    clean_axis(ax)
    ax.legend(
        frameon=False, loc="upper right", handlelength=2.5,
        handletextpad=0.45, labelspacing=0.25, borderaxespad=0.20
    )

    text = f"Median: all {all_mag.median():.1f}; eligible {eligible_mag.median():.1f}"
    ax.text(
        0.975, 0.055, text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.7, color=COLORS["text"]
    )
    panel_label(ax, "b")


def plot_panel_c(ax: plt.Axes, monthly: pd.DataFrame, ridgecrest_identified: bool) -> None:
    if monthly.empty:
        raise RuntimeError("No valid origin times available for panel c.")

    dates = pd.to_datetime(monthly["month"], utc=True)
    ax.plot(dates, monthly["all_preprocessed"], color=COLORS["all"], lw=0.90, linestyle=(0, (3.2, 2.0)), label="All preprocessed", zorder=2)
    ax.plot(dates, monthly["causal_eligible"], color=COLORS["eligible"], lw=1.00, label="Causally eligible", zorder=3)

    if ridgecrest_identified and monthly["ridgecrest"].sum() > 0:
        ax.fill_between(
            dates,
            0,
            monthly["ridgecrest"].to_numpy(dtype=float),
            step="mid",
            color=COLORS["ridgecrest"],
            alpha=0.28,
            linewidth=0,
            label="Ridgecrest sequence",
            zorder=1,
        )
        peak_index = int(np.argmax(monthly["ridgecrest"].to_numpy()))
        peak_date = dates.iloc[peak_index] if isinstance(dates, pd.Series) else dates[peak_index]
        peak_value = float(monthly.iloc[peak_index]["ridgecrest"])
        if peak_value > 0:
            ax.annotate(
                "Ridgecrest",
                xy=(peak_date, peak_value),
                xytext=(16, 15),
                textcoords="offset points",
                fontsize=5.8,
                color=COLORS["text"],
                arrowprops=dict(arrowstyle="-", color=COLORS["ridgecrest"], lw=0.65),
            )
    else:
        ax.axvspan(
            pd.Timestamp("2019-07-01", tz="UTC"),
            pd.Timestamp("2019-09-01", tz="UTC"),
            color=COLORS["ridgecrest"],
            alpha=0.10,
            linewidth=0,
            label="Ridgecrest period",
        )

    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Events per month")
    ax.set_title("Temporal distribution", loc="left", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    clean_axis(ax)
    ax.legend(
        frameon=False, loc="upper left", ncol=1, handlelength=2.5,
        handletextpad=0.45, labelspacing=0.25, borderaxespad=0.20
    )
    panel_label(ax, "c")


def binned_median_trend(frame: pd.DataFrame, bins: int = 12) -> pd.DataFrame:
    data = frame[["n_valid_stations", "n_p_reached_t0"]].dropna().copy()
    if len(data) < bins:
        return pd.DataFrame(columns=["x", "median", "q25", "q75", "n"])
    quantile_bins = pd.qcut(data["n_valid_stations"], q=bins, duplicates="drop")
    grouped = data.groupby(quantile_bins, observed=True)
    return grouped.agg(
        x=("n_valid_stations", "median"),
        median=("n_p_reached_t0", "median"),
        q25=("n_p_reached_t0", lambda x: x.quantile(0.25)),
        q75=("n_p_reached_t0", lambda x: x.quantile(0.75)),
        n=("n_p_reached_t0", "size"),
    ).reset_index(drop=True)


def plot_panel_d(ax: plt.Axes, events: pd.DataFrame, input_stations: int) -> None:
    noneligible = events.loc[~events["eligible"]]
    eligible = events.loc[events["eligible"]]

    ax.scatter(
        noneligible["n_valid_stations"], noneligible["n_p_reached_t0"],
        s=7.0, facecolor=COLORS["all"], edgecolor="none", alpha=0.26,
        rasterized=True, label="Not eligible", zorder=2,
    )
    ax.scatter(
        eligible["n_valid_stations"], eligible["n_p_reached_t0"],
        s=8.0, facecolor=COLORS["eligible"], edgecolor="white", linewidth=0.12,
        alpha=0.58, rasterized=True, label="Causally eligible", zorder=3,
    )

    # Avoid forcing the y axis to the x-axis maximum.  The original equality line
    # expanded y to ~180 although the observed early-station counts are much lower,
    # visually compressing the scientifically important distribution.
    x_data_max = float(events["n_valid_stations"].max())
    y_data_max = float(events["n_p_reached_t0"].max())
    x_upper = max(10.0, math.ceil(1.04 * x_data_max / 10.0) * 10.0)
    y_pad = max(2.0, 0.10 * y_data_max)
    y_upper = max(float(input_stations + 5), math.ceil((y_data_max + y_pad) / 5.0) * 5.0)

    equality_end = min(x_upper, y_upper)
    ax.plot(
        [0, equality_end], [0, equality_end], color="#B9C0C7", lw=0.70,
        linestyle=(0, (2.5, 2.5)), zorder=1
    )
    ax.axhline(input_stations, color=COLORS["threshold"], lw=0.78, linestyle=(0, (4, 2)), zorder=4)
    ax.text(
        0.985, input_stations, f"K = {input_stations}",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        fontsize=5.7, color=COLORS["threshold"],
    )

    trend = binned_median_trend(events, bins=12)
    if not trend.empty:
        ax.plot(trend["x"], trend["median"], color=COLORS["trend"], lw=1.00, zorder=5, label="Binned median")
        ax.fill_between(
            trend["x"].to_numpy(dtype=float),
            trend["q25"].to_numpy(dtype=float),
            trend["q75"].to_numpy(dtype=float),
            color=COLORS["trend"], alpha=0.11, linewidth=0, zorder=1,
        )

    eligible_fraction = 100.0 * events["eligible"].mean()
    ax.text(
        0.035, 0.965,
        f"Eligible: {events['eligible'].sum():,}/{len(events):,} ({eligible_fraction:.1f}%)",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.7,
        bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="#CDD2D8", linewidth=0.45, alpha=0.94),
    )

    ax.set_xlim(0, x_upper)
    ax.set_ylim(0, y_upper)
    ax.set_xlabel("Valid stations after preprocessing")
    ax.set_ylabel(r"P-wave-reached stations at $T_0=5$ s")
    ax.set_title("Early station availability", loc="left", fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    clean_axis(ax)
    ax.legend(
        frameon=False, loc="upper right", handletextpad=0.35,
        labelspacing=0.25, borderaxespad=0.20
    )
    panel_label(ax, "d")


# -----------------------------------------------------------------------------
# Caption and output
# -----------------------------------------------------------------------------


def figure_caption(events: pd.DataFrame, stations: pd.DataFrame, t0_sec: float, input_stations: int) -> str:
    n_all = len(events)
    n_eligible = int(events["eligible"].sum())
    ridgecrest_text = (
        "Ridgecrest-sequence events are highlighted in vermilion."
        if events["ridgecrest"].any()
        else "The July–August 2019 Ridgecrest period is shaded because an explicit sequence label was unavailable."
    )
    return (
        "Fig. 1 | Dataset composition and causal observation protocol. "
        f"a, Spatial distribution of {n_all:,} earthquakes with fully preprocessed waveforms, "
        f"{n_eligible:,} events satisfying the primary causal setting of T0={t0_sec:g} s and K={input_stations}, "
        f"and {len(stations):,} stations represented in the causally eligible events. Shaded relief is shown for geographic context, and event-symbol area scales with catalogue magnitude. "
        f"{ridgecrest_text} "
        "b, Magnitude distributions of the fully preprocessed and causally eligible event populations. Dashed lines denote medians. "
        "c, Monthly event counts from 2010 to 2024, with the 2019 Ridgecrest sequence or period highlighted. "
        f"d, Number of valid stations per event versus the number reached by the P wave before T0={t0_sec:g} s. "
        f"The horizontal dashed line denotes the minimum K={input_stations} input stations, the diagonal denotes equality, "
        "and the green line and band show the binned median and interquartile range. Although many earthquakes have dense final station coverage, "
        "the number of stations causally available in the first seconds is substantially smaller."
    )


def write_outputs(
    out_dir: Path,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    magnitude_table: pd.DataFrame,
    monthly: pd.DataFrame,
    caption: str,
    audit: AuditSummary,
    file_hashes: dict[str, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    events[
        [
            "event_id",
            "origin_time",
            "magnitude",
            "latitude",
            "longitude",
            "depth_km",
            "n_valid_stations",
            "n_p_reached_t0",
            "eligible",
            "ridgecrest",
        ]
    ].to_csv(out_dir / "Fig1_source_data_events.csv", index=False)
    stations.to_csv(out_dir / "Fig1_source_data_stations.csv", index=False)
    magnitude_table.to_csv(out_dir / "Fig1_source_data_magnitude_distribution.csv", index=False)
    monthly.to_csv(out_dir / "Fig1_source_data_monthly_counts.csv", index=False)
    events[["event_id", "n_valid_stations", "n_p_reached_t0", "eligible"]].to_csv(
        out_dir / "Fig1_source_data_station_availability.csv", index=False
    )
    (out_dir / "Fig1_caption.txt").write_text(caption + "\n", encoding="utf-8")
    payload = asdict(audit)
    payload["input_file_sha256"] = file_hashes
    (out_dir / "Fig1_run_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def create_figure(
    events: pd.DataFrame,
    stations: pd.DataFrame,
    magnitude_table: pd.DataFrame,
    monthly: pd.DataFrame,
    out_dir: Path,
    input_stations: int,
    dpi: int,
    map_extent: tuple[float, float, float, float],
    relief_raster: Path | None,
    relief_pixels: int,
    relief_min_elevation: float,
    relief_max_elevation: float,
    relief_vertical_exaggeration: float,
) -> None:
    configure_matplotlib()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, FIGURE_HEIGHT_MM * MM_TO_INCH),
        constrained_layout=False,
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=0.072,
        right=0.992,
        bottom=0.078,
        top=0.955,
        wspace=0.225,
        hspace=0.245,
        width_ratios=(1.04, 1.00),
        height_ratios=(1.00, 1.00),
    )

    if ccrs is None:
        raise ImportError("Cartopy is required for panel a. Install with: python -m pip install cartopy")
    ax_a = fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree())
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_panel_a(
        ax_a,
        events,
        stations,
        map_extent=map_extent,
        relief_raster=relief_raster,
        relief_pixels=relief_pixels,
        relief_min_elevation=relief_min_elevation,
        relief_max_elevation=relief_max_elevation,
        relief_vertical_exaggeration=relief_vertical_exaggeration,
    )
    plot_panel_b(ax_b, magnitude_table, events)
    plot_panel_c(ax_c, monthly, ridgecrest_identified=bool(events["ridgecrest"].any()))
    plot_panel_d(ax_d, events, input_stations=input_stations)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "Fig1_dataset_composition_and_causal_protocol"
    # Preserve the exact 183-mm double-column canvas.  Avoid bbox_inches="tight",
    # which silently changes the physical output size and can break panel alignment.
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Demo data for layout testing only
# -----------------------------------------------------------------------------


def make_demo_data(seed: int = 20260713) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 3239
    dates = pd.date_range("2010-01-01", "2024-12-31", periods=n, tz="UTC")
    magnitude = np.clip(3.0 + rng.exponential(0.42, n), 3.0, 7.1)
    latitude = rng.normal(34.1, 0.75, n)
    longitude = rng.normal(-117.4, 0.90, n)
    n_valid = rng.integers(20, 181, n)
    fraction = np.clip(rng.beta(1.5, 6.0, n), 0.0, 1.0)
    n_triggered = np.minimum(n_valid, np.rint(n_valid * fraction)).astype(int)
    eligible = (n_triggered >= 5) & (n_valid >= 15)

    # Force the documented count for a faithful layout test.
    order = np.argsort(-(n_triggered + 0.01 * n_valid))
    eligible[:] = False
    eligible[order[:1620]] = True

    ridgecrest = (
        (dates >= pd.Timestamp("2019-07-01", tz="UTC"))
        & (dates < pd.Timestamp("2019-10-01", tz="UTC"))
    )
    events = pd.DataFrame(
        {
            "event_id": [f"demo_{i:05d}" for i in range(n)],
            "origin_time": dates,
            "magnitude": magnitude,
            "latitude": latitude,
            "longitude": longitude,
            "depth_km": rng.uniform(1, 20, n),
            "n_valid_stations": n_valid,
            "n_p_reached_t0": n_triggered,
            "eligible": eligible,
            "ridgecrest": ridgecrest,
            "h5_path": "",
        }
    )
    station_n = 447
    stations = pd.DataFrame(
        {
            "station_id": [f"DEMO.{i:03d}" for i in range(station_n)],
            "latitude": rng.normal(34.0, 0.95, station_n),
            "longitude": rng.normal(-117.5, 1.05, station_n),
            "elevation_m": rng.normal(600, 350, station_n),
            "n_event_records": rng.integers(1, 200, station_n),
        }
    )
    return events, stations


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-manifest", default="./data/scedc/model_manifests/all_events_with_scenario_flags.csv")
    parser.add_argument("--scenario-manifest", default="./data/scedc/model_manifests/scenario_t0_5s_k5.csv")
    parser.add_argument("--h5-root", default="./data/scedc/processed_full_v4/events")
    parser.add_argument("--t0-sec", type=float, default=5.0)
    parser.add_argument("--input-stations", type=int, default=5)
    parser.add_argument("--target-stations", type=int, default=10)
    parser.add_argument("--station-scope", choices=["eligible", "all"], default="eligible")
    parser.add_argument("--ridgecrest-mode", choices=["auto", "column", "box", "none"], default="auto")
    parser.add_argument("--magnitude-bin-width", type=float, default=0.20)
    parser.add_argument("--expected-preprocessed", type=int, default=3239)
    parser.add_argument("--expected-eligible", type=int, default=1620)
    parser.add_argument("--expected-train", type=int, default=1189)
    parser.add_argument("--expected-validation", type=int, default=207)
    parser.add_argument("--expected-test", type=int, default=224)
    parser.add_argument("--split-column", default="split_grouped")
    parser.add_argument("--strict-count-audit",action="store_true")
    parser.add_argument("--allow-missing-h5", action="store_true")
    parser.add_argument(
        "--relief-raster",
        default='auto',
        help=(
            "Local ETOPO/GEBCO GeoTIFF, 'auto' to search common project paths, "
            "or 'none' to use a flat land/ocean fallback."
        ),
    )
    parser.add_argument(
        "--map-extent",
        default="-121.5,-114.2,31.5,37.2",
        help="'auto' or west,east,south,north in degrees, e.g. -121.5,-114.2,31.5,37.2",
    )
    parser.add_argument("--relief-pixels", type=int, default=1800)
    parser.add_argument("--relief-min-elevation", type=float, default=-5000.0)
    parser.add_argument("--relief-max-elevation", type=float, default=3500.0)
    parser.add_argument("--relief-vertical-exaggeration", type=float, default=1.35)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--out-dir", default="figures/nc_dataset")
    parser.add_argument("--demo", action="store_true", help="Generate a layout preview using synthetic data only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        events, stations = make_demo_data()
        ridgecrest_source = "synthetic demo"
        master_path = Path("synthetic_master.csv")
        scenario_path = None
        h5_root = Path("synthetic_h5")
        missing_events: list[str] = []
        scenario_source = "synthetic demo"
        file_hashes: dict[str, str] = {}
    else:
        master_path = resolve_auto_path(args.master_manifest, MASTER_CANDIDATES, "master manifest", required=True)
        scenario_path = resolve_auto_path(
            args.scenario_manifest,
            SCENARIO_CANDIDATES,
            "scenario manifest",
            required=False,
        )
        h5_root = resolve_auto_path(args.h5_root, H5_ROOT_CANDIDATES, "HDF5 root", required=True)
        assert master_path is not None and h5_root is not None

        master = normalize_master_columns(pd.read_csv(master_path, dtype={"event_id": str}))
        eligible_ids, scenario_source = scenario_event_ids(
            master,
            scenario_path,
            args.t0_sec,
            args.input_stations,
        )
        events, stations, missing_events = collect_event_and_station_metadata(
            master=master,
            h5_root=h5_root,
            eligible_ids=eligible_ids,
            t0_sec=args.t0_sec,
            input_stations=args.input_stations,
            target_stations=args.target_stations,
            station_scope=args.station_scope,
            allow_missing_h5=args.allow_missing_h5,
        )

        # Preserve any explicit Ridgecrest metadata from the master manifest.
        metadata_columns = [column for column in master.columns if column not in events.columns or column == "event_id"]
        merged = events.merge(master[["event_id", *[c for c in metadata_columns if c != "event_id"]]], on="event_id", how="left")
        ridgecrest_mask, ridgecrest_source = infer_ridgecrest_mask(merged, args.ridgecrest_mode)
        events["ridgecrest"] = ridgecrest_mask.to_numpy(dtype=bool)

        file_hashes = {
            "master_manifest": sha256_file(master_path),
        }
        if scenario_path is not None:
            file_hashes["scenario_manifest"] = sha256_file(scenario_path)

    if "ridgecrest" not in events:
        events["ridgecrest"] = False

    # Sort once so all source-data outputs are deterministic.
    events = events.sort_values(["origin_time", "event_id"], na_position="last").reset_index(drop=True)

    n_preprocessed = int(len(events))
    n_eligible = int(events["eligible"].sum())

    if args.strict_count_audit:
        failures: list[str] = []
        if n_preprocessed != args.expected_preprocessed:
            failures.append(
                f"preprocessed events: observed={n_preprocessed}, expected={args.expected_preprocessed}"
            )
        if n_eligible != args.expected_eligible:
            failures.append(f"eligible events: observed={n_eligible}, expected={args.expected_eligible}")
        if missing_events and not args.allow_missing_h5:
            failures.append(f"missing/unreadable HDF5 events: {len(missing_events)}")

        if not args.demo and args.split_column in master.columns:
            eligible_master = master.loc[master["event_id"].isin(set(events.loc[events["eligible"], "event_id"]))]
            split_counts = eligible_master[args.split_column].astype(str).value_counts()
            expected = {
                "train": args.expected_train,
                "validation": args.expected_validation,
                "test": args.expected_test,
            }
            for label, expected_count in expected.items():
                observed = int(split_counts.get(label, 0))
                if observed != expected_count:
                    failures.append(
                        f"{args.split_column}={label}: observed={observed}, expected={expected_count}"
                    )

        if failures:
            raise RuntimeError("Locked dataset audit FAILED:\n  " + "\n  ".join(failures))

    magnitude_table = magnitude_source(events, args.magnitude_bin_width)
    monthly = build_monthly_counts(events)
    caption = figure_caption(events, stations, args.t0_sec, args.input_stations)

    relief_path = resolve_relief_raster(args.relief_raster)
    if relief_path is None:
        print(
            "WARNING: no local relief raster was resolved; panel a will use a flat "
            "land/ocean background. Supply --relief-raster for the publication map.",
            file=sys.stderr,
        )
    map_extent = parse_map_extent(args.map_extent, events, stations)

    create_figure(
        events=events,
        stations=stations,
        magnitude_table=magnitude_table,
        monthly=monthly,
        out_dir=out_dir,
        input_stations=args.input_stations,
        dpi=args.dpi,
        map_extent=map_extent,
        relief_raster=relief_path,
        relief_pixels=args.relief_pixels,
        relief_min_elevation=args.relief_min_elevation,
        relief_max_elevation=args.relief_max_elevation,
        relief_vertical_exaggeration=args.relief_vertical_exaggeration,
    )

    finite_times = events["origin_time"].dropna()
    finite_magnitudes = events["magnitude"].dropna()
    audit = AuditSummary(
        master_manifest=str(master_path),
        scenario_manifest=str(scenario_path) if scenario_path is not None else None,
        h5_root=str(h5_root),
        n_master_rows=int(args.expected_preprocessed if args.demo else len(master)),
        n_h5_resolved=n_preprocessed,
        n_h5_missing=len(missing_events),
        n_preprocessed_events=n_preprocessed,
        n_eligible_events=n_eligible,
        n_stations=int(len(stations)),
        t0_sec=float(args.t0_sec),
        input_stations=int(args.input_stations),
        target_stations=int(args.target_stations),
        expected_preprocessed=int(args.expected_preprocessed),
        expected_eligible=int(args.expected_eligible),
        ridgecrest_identification=ridgecrest_source,
        time_start=str(finite_times.min()) if len(finite_times) else None,
        time_end=str(finite_times.max()) if len(finite_times) else None,
        magnitude_min=float(finite_magnitudes.min()) if len(finite_magnitudes) else None,
        magnitude_max=float(finite_magnitudes.max()) if len(finite_magnitudes) else None,
        relief_raster=str(relief_path) if relief_path is not None else None,
        map_extent=[float(value) for value in map_extent],
    )
    write_outputs(
        out_dir=out_dir,
        events=events,
        stations=stations,
        magnitude_table=magnitude_table,
        monthly=monthly,
        caption=caption,
        audit=audit,
        file_hashes=file_hashes,
    )

    print("\n=== Fig. 1 dataset audit ===")
    print(f"Master manifest      : {master_path}")
    print(f"Scenario source      : {scenario_source}")
    print(f"HDF5 root            : {h5_root}")
    print(f"Relief raster        : {relief_path if relief_path is not None else 'flat fallback'}")
    print(f"Map extent           : {map_extent}")
    print(f"Preprocessed events  : {n_preprocessed:,}")
    print(f"Causally eligible    : {n_eligible:,}")
    print(f"Unique stations      : {len(stations):,}")
    print(f"Ridgecrest rule      : {ridgecrest_source}")
    print(f"Missing HDF5         : {len(missing_events):,}")
    print(f"Output directory     : {out_dir.resolve()}")
    print("\nGenerated:")
    for name in (
        "Fig1_dataset_composition_and_causal_protocol.pdf",
        "Fig1_dataset_composition_and_causal_protocol.svg",
        "Fig1_dataset_composition_and_causal_protocol.png",
        "Fig1_caption.txt",
        "Fig1_source_data_events.csv",
        "Fig1_source_data_stations.csv",
        "Fig1_source_data_magnitude_distribution.csv",
        "Fig1_source_data_monthly_counts.csv",
        "Fig1_source_data_station_availability.csv",
        "Fig1_run_audit.json",
    ):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
