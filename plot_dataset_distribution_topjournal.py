#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-style dataset overview for Section II.

Panels
------
(a) Spatial coverage: preprocessed events, model-ready events, and stations.
(b) Magnitude distribution.
(c) Temporal distribution and grouped split counts.
(d) Causal observability: total stations vs P-reached stations by T0.

Default inputs
--------------
data/scedc/full_preprocessing_events.csv
data/scedc/model_manifests/scenario_t0_5s_k5.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def num(x):
    return pd.to_numeric(x, errors="coerce")


def ensure_year(df):
    df = df.copy()
    if "year" in df.columns:
        df["year"] = num(df["year"]).astype("Int64")
    elif "origin_time" in df.columns:
        df["year"] = pd.to_datetime(
            df["origin_time"], utc=True, errors="coerce"
        ).dt.year.astype("Int64")
    else:
        raise ValueError("Need 'year' or 'origin_time' column.")
    return df


def merge_event_metadata(events, scenario):
    scenario = scenario.copy()
    wanted = ["latitude", "longitude", "depth_km",
              "magnitude", "origin_time", "year"]
    missing = [c for c in wanted if c not in scenario.columns]
    if missing:
        cols = ["event_id"] + [c for c in missing if c in events.columns]
        scenario = scenario.merge(
            events[cols].drop_duplicates("event_id"),
            on="event_id", how="left", validate="many_to_one"
        )
    return ensure_year(scenario)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(ls=":", lw=0.55, alpha=0.40)


def label_panel(ax, letter):
    ax.text(
        0.015, 0.985, f"({letter})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, fontweight="bold"
    )


def collect_stations(scenario, cache_path, max_h5=0):
    if cache_path.exists():
        x = pd.read_csv(cache_path)
        if {"station_id", "latitude", "longitude"}.issubset(x.columns):
            print(f"Reuse station cache: {cache_path.resolve()}")
            return x

    if "h5_path" not in scenario.columns:
        print("[Warning] no h5_path; stations omitted.")
        return pd.DataFrame(columns=["station_id", "latitude", "longitude"])

    paths = scenario["h5_path"].dropna().astype(str).tolist()
    if max_h5 > 0:
        paths = paths[:max_h5]

    # station key -> running coordinate sums
    store = {}
    for i, text in enumerate(paths, 1):
        p = Path(text)
        if not p.exists():
            continue
        try:
            with h5py.File(p, "r") as h5:
                if "station_coords" not in h5:
                    continue
                coords = np.asarray(h5["station_coords"][:], dtype=float)
                if "station_id" in h5:
                    ids = h5["station_id"].asstr()[:]
                else:
                    ids = np.array([""] * len(coords), dtype=object)

                for sid, xyz in zip(ids, coords):
                    if len(xyz) < 2:
                        continue
                    lat, lon = float(xyz[0]), float(xyz[1])
                    if not (np.isfinite(lat) and np.isfinite(lon)):
                        continue
                    sid = str(sid).strip()
                    key = sid if sid else f"{lat:.5f}_{lon:.5f}"
                    if key not in store:
                        store[key] = [0.0, 0.0, 0]
                    store[key][0] += lat
                    store[key][1] += lon
                    store[key][2] += 1
        except Exception as exc:
            print(f"[Warning] {p.name}: {type(exc).__name__}: {exc}")

        if i % 250 == 0 or i == len(paths):
            print(f"Station scan {i}/{len(paths)}; unique={len(store)}")

    rows = [
        {
            "station_id": key,
            "latitude": values[0] / values[2],
            "longitude": values[1] / values[2],
            "n_occurrences": values[2],
        }
        for key, values in store.items()
    ]
    stations = pd.DataFrame(rows)
    if len(stations):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stations.to_csv(cache_path, index=False)
    return stations


def plot_map(ax, events, scenario, stations):
    for df in (events, scenario):
        for c in ("latitude", "longitude", "magnitude"):
            df[c] = num(df[c])

    all_map = events.dropna(subset=["latitude", "longitude", "magnitude"])
    model = scenario.dropna(subset=["latitude", "longitude", "magnitude"])

    ax.scatter(
        all_map["longitude"], all_map["latitude"],
        s=5, alpha=0.12, linewidths=0,
        label=f"Preprocessed events (N={len(all_map):,})",
        rasterized=True
    )

    if len(stations):
        sta = stations.copy()
        sta["latitude"] = num(sta["latitude"])
        sta["longitude"] = num(sta["longitude"])
        sta = sta.dropna(subset=["latitude", "longitude"])
        ax.scatter(
            sta["longitude"], sta["latitude"],
            marker="^", s=9, alpha=0.30, linewidths=0,
            label=f"Unique stations (N={len(sta):,})",
            rasterized=True
        )

    mag = model["magnitude"].to_numpy(float)
    sizes = 8 + 8 * np.maximum(mag - 2.8, 0) ** 1.6
    ax.scatter(
        model["longitude"], model["latitude"],
        s=sizes, alpha=0.58, linewidths=0.35,
        label=f"Model-ready events (N={len(model):,})",
        rasterized=True
    )

    if len(model):
        r = model.loc[model["magnitude"].idxmax()]
        ax.annotate(
            f"M {float(r['magnitude']):.1f}",
            xy=(float(r["longitude"]), float(r["latitude"])),
            xytext=(6, 7), textcoords="offset points",
            fontsize=7.3,
            arrowprops={"arrowstyle": "-", "linewidth": 0.65}
        )

    lon = pd.concat([all_map["longitude"], model["longitude"]])
    lat = pd.concat([all_map["latitude"], model["latitude"]])
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    ax.set_xlim(lon_min - 0.25, lon_max + 0.25)
    ax.set_ylim(lat_min - 0.18, lat_max + 0.18)

    mean_lat = 0.5 * (lat_min + lat_max)
    c = np.cos(np.deg2rad(mean_lat))
    if c > 1e-6:
        ax.set_aspect(1 / c, adjustable="box")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Spatial coverage", fontweight="bold")
    ax.legend(frameon=False, loc="lower left", fontsize=6.6)
    ax.text(
        0.98, 0.02, "Event symbol size ∝ magnitude",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7
    )
    label_panel(ax, "a")
    style(ax)


def plot_magnitude(ax, events, scenario):
    a = num(events["magnitude"]).dropna().to_numpy()
    b = num(scenario["magnitude"]).dropna().to_numpy()

    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    width = 0.2
    bins = np.arange(
        np.floor(lo / width) * width,
        np.ceil(hi / width) * width + width + 1e-9,
        width
    )

    ax.hist(a, bins=bins, alpha=0.32,
            label=f"Preprocessed (N={len(a):,})")
    ax.hist(b, bins=bins, histtype="step", lw=1.35,
            label=f"Model-ready (N={len(b):,})")
    ax.set_yscale("log")

    ax.text(
        0.97, 0.95,
        (
            f"Median = {np.median(b):.1f}\n"
            f"M ≥ 4.0: {100*np.mean(b >= 4.0):.1f}%\n"
            f"Mmax = {np.max(b):.1f}"
        ),
        transform=ax.transAxes, ha="right", va="top", fontsize=7
    )
    ax.set_xlabel("Catalog magnitude")
    ax.set_ylabel("Number of events")
    ax.set_title("Magnitude distribution", fontweight="bold")
    ax.legend(frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.68), fontsize=6.8)
    label_panel(ax, "b")
    style(ax)


def plot_time(ax, events, scenario):
    events = ensure_year(events)
    scenario = ensure_year(scenario)

    ymin = int(min(events["year"].dropna().min(),
                   scenario["year"].dropna().min()))
    ymax = int(max(events["year"].dropna().max(),
                   scenario["year"].dropna().max()))
    years = np.arange(ymin, ymax + 1)

    c_all = events["year"].value_counts().reindex(years, fill_value=0)
    c_mod = scenario["year"].value_counts().reindex(years, fill_value=0)

    x = np.arange(len(years), dtype=float)
    w = 0.38
    ax.bar(x - w/2, c_all.to_numpy(float), width=w, alpha=0.38,
           label="Preprocessed")
    ax.bar(x + w/2, c_mod.to_numpy(float), width=w, alpha=0.82,
           label="Model-ready")

    keep = [
        i for i, yr in enumerate(years)
        if yr == years[0] or yr == years[-1] or yr % 2 == 0
    ]
    ax.set_xticks(keep)
    ax.set_xticklabels([str(years[i]) for i in keep],
                       rotation=35, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of events")
    ax.set_title("Temporal coverage", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    split_col = (
        "split_grouped" if "split_grouped" in scenario.columns
        else ("split" if "split" in scenario.columns else None)
    )
    if split_col:
        counts = scenario[split_col].astype(str).value_counts().to_dict()
        train = int(counts.get("train", 0))
        val = int(counts.get("validation", counts.get("val", 0)))
        test = int(counts.get("test", 0))
        ax.text(
            0.98, 0.95,
            f"Grouped split\ntrain / val / test = "
            f"{train:,} / {val:,} / {test:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7
        )

    label_panel(ax, "c")
    style(ax)


def plot_observability(ax, scenario, t0, k):
    total_col = next(
        (c for c in ["n_stations_h5",
                     "n_valid_download_stations",
                     "n_valid"] if c in scenario.columns),
        None
    )
    trig_col = f"n_triggered_t0_{t0}s"

    if total_col and trig_col in scenario.columns:
        x = num(scenario[total_col]).to_numpy(float)
        y = num(scenario[trig_col]).to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        ax.scatter(x, y, s=9, alpha=0.22, linewidths=0, rasterized=True)

        maximum = float(max(np.nanmax(x), np.nanmax(y)))
        line = np.array([0.0, maximum])
        ax.plot(line, line, ls=":", lw=0.85, label="All stations P-reached")
        ax.axhline(float(k), ls="--", lw=0.95,
                   label=f"K = {k} input requirement")

        ax.text(
            0.97, 0.05,
            (
                f"Median total = {np.median(x):.0f}\n"
                f"Median P-reached = {np.median(y):.0f}\n"
                f"Median early fraction = "
                f"{np.median(y/np.maximum(x,1)):.2f}"
            ),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7
        )
        ax.set_xlabel("Stations available in processed event")
        ax.set_ylabel(f"P-reached stations by {t0} s")
        ax.legend(frameon=False, loc="upper left", fontsize=6.7)
    elif trig_col in scenario.columns:
        y = num(scenario[trig_col]).dropna().to_numpy(float)
        bins = np.arange(np.floor(y.min()), np.ceil(y.max()) + 2, 1)
        ax.hist(y, bins=bins, alpha=0.60)
        ax.axvline(float(k), ls="--", lw=0.95, label=f"K = {k}")
        ax.set_xlabel(f"P-reached stations by {t0} s")
        ax.set_ylabel("Number of events")
        ax.legend(frameon=False, fontsize=6.8)
    else:
        ax.text(
            0.5, 0.5,
            "Causal station-count columns\nnot found in scenario manifest.",
            transform=ax.transAxes, ha="center", va="center", fontsize=8
        )
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Causal observability", fontweight="bold")
    label_panel(ax, "d")
    style(ax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--events",
        default='data/scedc/full_preprocessing_events.csv'
    )
    p.add_argument(
        "--scenario",
        default='data/scedc/model_manifests/scenario_t0_5s_k5.csv'
    )
    p.add_argument(
        "--out-dir",
        default='figures/section_2_dataset'
    )
    p.add_argument("--t0-sec", type=int, default=5)
    p.add_argument("--input-stations", type=int, default=5)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--skip-stations", action="store_true")
    p.add_argument("--max-h5-for-stations", type=int, default=0)
    args = p.parse_args()

    events_path = Path(args.events)
    scenario_path = Path(args.scenario)
    out_dir = Path(args.out_dir)

    if not events_path.exists():
        raise FileNotFoundError(events_path)
    if not scenario_path.exists():
        raise FileNotFoundError(scenario_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_path, dtype={"event_id": str})
    scenario = pd.read_csv(scenario_path, dtype={"event_id": str})

    if "event_id" not in events or "magnitude" not in events:
        raise ValueError("events CSV must include event_id and magnitude.")
    if "event_id" not in scenario:
        raise ValueError("scenario CSV must include event_id.")

    events = ensure_year(events).drop_duplicates("event_id").reset_index(drop=True)
    scenario = merge_event_metadata(events, scenario)
    scenario = scenario.drop_duplicates("event_id").reset_index(drop=True)

    station_cache = out_dir / (
        f"station_locations_t0_{args.t0_sec}s_k{args.input_stations}.csv"
    )
    stations = (
        pd.DataFrame(columns=["station_id", "latitude", "longitude"])
        if args.skip_stations
        else collect_stations(
            scenario, station_cache, args.max_h5_for_stations
        )
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.6,
        "axes.titlesize": 9.2,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.55))
    plot_map(axes[0, 0], events.copy(), scenario.copy(), stations)
    plot_magnitude(axes[0, 1], events, scenario)
    plot_time(axes[1, 0], events, scenario)
    plot_observability(
        axes[1, 1], scenario, args.t0_sec, args.input_stations
    )

    # TGRS-style: no figure-level title; message goes in caption.
    fig.subplots_adjust(
        left=0.085, right=0.985,
        bottom=0.105, top=0.965,
        wspace=0.28, hspace=0.34
    )

    stem = "Fig_2_dataset_distribution"
    fig.savefig(out_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame([
        {
            "group": "preprocessed",
            "n_events": len(events),
            "mag_min": num(events["magnitude"]).min(),
            "mag_median": num(events["magnitude"]).median(),
            "mag_mean": num(events["magnitude"]).mean(),
            "mag_max": num(events["magnitude"]).max(),
            "year_min": events["year"].min(),
            "year_max": events["year"].max(),
        },
        {
            "group": "model_ready_t0_5s_k5",
            "n_events": len(scenario),
            "mag_min": num(scenario["magnitude"]).min(),
            "mag_median": num(scenario["magnitude"]).median(),
            "mag_mean": num(scenario["magnitude"]).mean(),
            "mag_max": num(scenario["magnitude"]).max(),
            "year_min": scenario["year"].min(),
            "year_max": scenario["year"].max(),
        },
    ])
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)

    print("=== Dataset distribution figure ===")
    print(f"Preprocessed events : {len(events):,}")
    print(f"Model-ready events  : {len(scenario):,}")
    print(f"Unique stations     : {len(stations):,}")
    print(f"Outputs             : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
