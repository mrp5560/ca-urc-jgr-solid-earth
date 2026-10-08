#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download SCEDC/SCSN yearly ASCII catalog files and convert them to event_catalog.csv.

This version avoids the unstable SCEDC FDSN event endpoint.

Output columns:
    event_id, resource_id, origin_time, latitude, longitude, depth_km,
    magnitude, mag_type, event_type, client, split,
    quality, nph, rms, catalog
"""

import argparse
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


SCEDC_SCSN_URL = "https://service.scedc.caltech.edu/ftp/catalogs/SCSN/{year}.catalog"
GITHUB_SCSN_URL = "https://raw.githubusercontent.com/SCEDC/SCEDC-catalogs/refs/heads/master/SCSN/{year}.catalog"


def parse_date(s: str) -> datetime:
    s = s.strip()
    if len(s) == 4:
        return datetime(int(s), 1, 1, tzinfo=timezone.utc)
    if len(s) == 10:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    s = s.replace("Z", "")
    if "T" in s:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def download_text(url: str, out_path: Path, retries: int = 3, sleep_sec: float = 2.0) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[Cache] using existing file: {out_path}")
        return out_path.read_text(encoding="utf-8", errors="ignore")

    last_err = None
    headers = {
        "User-Agent": "Mozilla/5.0 SCEDC-catalog-downloader"
    }

    for i in range(retries):
        try:
            print(f"[Download] {url}")
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            text = r.text
            out_path.write_text(text, encoding="utf-8", errors="ignore")
            print(f"[OK] saved -> {out_path}, bytes={out_path.stat().st_size}")
            return text

        except Exception as e:
            last_err = e
            print(f"[Warn] failed {i + 1}/{retries}: {type(e).__name__}: {e}")
            time.sleep(sleep_sec * (i + 1))

    raise last_err


def dms_to_decimal(deg_str: str, min_str: str) -> float:
    """
    Convert degree-minute coordinate to decimal degrees.

    Examples:
        33, 53.11   ->  33.8851667
        -116, 15.47 -> -116.2578333
    """
    deg = float(deg_str)
    minute = float(min_str)

    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minute / 60.0)


def parse_scsn_line(line: str):
    """
    Parse one SCSN-format catalog line.

    Official example:
    2003 01 01  00 41 56.53  33 53.11-116 15.47 A 1.5      5.32 30     0.20  9875253

    Fields:
        year mon day hour min sec lat_deg lat_min lon_deg lon_min quality mag depth nph rms evid
    """

    if not line.strip():
        return None
    if line.lstrip().startswith("#"):
        return None
    if not re.match(r"^\s*\d{4}\s+\d{2}\s+\d{2}", line):
        return None

    pattern = re.compile(
        r"^\s*"
        r"(?P<year>\d{4})\s+"
        r"(?P<mon>\d{2})\s+"
        r"(?P<day>\d{2})\s+"
        r"(?P<hour>\d{2})\s+"
        r"(?P<minute>\d{2})\s+"
        r"(?P<sec>\d+(?:\.\d+)?)\s+"
        r"(?P<lat_deg>[+-]?\d+)\s+"
        r"(?P<lat_min>\d+(?:\.\d+)?)"
        r"(?P<lon_deg>[+-]\d+)\s+"
        r"(?P<lon_min>\d+(?:\.\d+)?)\s+"
        r"(?P<quality>[A-Z])\s+"
        r"(?P<mag>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<depth>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<nph>\d+)\s+"
        r"(?P<rms>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<evid>\S+)"
    )

    m = pattern.match(line)
    if not m:
        return None

    g = m.groupdict()

    sec_float = float(g["sec"])
    sec_int = int(sec_float)
    micro = int(round((sec_float - sec_int) * 1_000_000))

    # 防止 59.999999 四舍五入到 1_000_000
    if micro >= 1_000_000:
        sec_int += 1
        micro -= 1_000_000

    dt = datetime(
        int(g["year"]),
        int(g["mon"]),
        int(g["day"]),
        int(g["hour"]),
        int(g["minute"]),
        sec_int,
        micro,
        tzinfo=timezone.utc,
    )

    lat = dms_to_decimal(g["lat_deg"], g["lat_min"])
    lon = dms_to_decimal(g["lon_deg"], g["lon_min"])
    evid = str(g["evid"]).strip()

    return {
        "event_id": evid,
        "resource_id": f"scedc:{evid}",
        "origin_time": dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "latitude": lat,
        "longitude": lon,
        "depth_km": float(g["depth"]),
        "magnitude": float(g["mag"]),
        "mag_type": "",
        "event_type": "",
        "client": "SCEDC",
        "split": "",
        "quality": g["quality"],
        "nph": int(g["nph"]),
        "rms": float(g["rms"]),
        "catalog": "SCSN",
    }


def parse_scsn_catalog_text(text: str) -> pd.DataFrame:
    rows = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        row = parse_scsn_line(line)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df = df.copy()
    df["origin_time_dt"] = pd.to_datetime(df["origin_time"], utc=True)

    start_dt = pd.Timestamp(parse_date(args.start))
    end_dt = pd.Timestamp(parse_date(args.end))

    mask = (df["origin_time_dt"] >= start_dt) & (df["origin_time_dt"] < end_dt)

    if args.min_magnitude is not None:
        mask &= df["magnitude"] >= args.min_magnitude
    if args.max_magnitude is not None:
        mask &= df["magnitude"] <= args.max_magnitude

    if args.min_lat is not None:
        mask &= df["latitude"] >= args.min_lat
    if args.max_lat is not None:
        mask &= df["latitude"] <= args.max_lat

    if args.min_lon is not None:
        mask &= df["longitude"] >= args.min_lon
    if args.max_lon is not None:
        mask &= df["longitude"] <= args.max_lon

    if args.min_depth_km is not None:
        mask &= df["depth_km"] >= args.min_depth_km
    if args.max_depth_km is not None:
        mask &= df["depth_km"] <= args.max_depth_km

    df = df.loc[mask].copy()
    df = df.drop(columns=["origin_time_dt"])
    return df


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2025-01-01")

    p.add_argument("--min-magnitude", type=float, default=3.0)
    p.add_argument("--max-magnitude", type=float, default=None)

    p.add_argument("--min-lat", type=float, default=32.0)
    p.add_argument("--max-lat", type=float, default=36.5)
    p.add_argument("--min-lon", type=float, default=-121.0)
    p.add_argument("--max-lon", type=float, default=-114.0)

    p.add_argument("--min-depth-km", type=float, default=0)
    p.add_argument("--max-depth-km", type=float, default=30)

    p.add_argument("--cache-dir", default="scedc_catalog_cache")
    p.add_argument("--out", default="data/scedc/events_m3_2010_2024.csv")

    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--sleep-sec", type=float, default=2.0)

    args = p.parse_args()

    start_dt = parse_date(args.start)
    end_dt = parse_date(args.end)

    years = list(range(start_dt.year, end_dt.year + 1))
    if end_dt.month == 1 and end_dt.day == 1 and end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
        years = list(range(start_dt.year, end_dt.year))

    print(f"[Info] start = {start_dt}")
    print(f"[Info] end   = {end_dt}")
    print(f"[Info] years = {years}")

    all_dfs = []
    cache_dir = Path(args.cache_dir)

    for year in years:
        cache_path = cache_dir / f"SCSN_{year}.catalog"

        # 优先使用 SCEDC 官方静态文件；失败时自动 fallback 到 GitHub raw
        urls = [
            SCEDC_SCSN_URL.format(year=year),
            GITHUB_SCSN_URL.format(year=year),
        ]

        text = None
        last_err = None

        for url in urls:
            try:
                text = download_text(
                    url=url,
                    out_path=cache_path,
                    retries=args.retries,
                    sleep_sec=args.sleep_sec,
                )
                break
            except Exception as e:
                last_err = e
                print(f"[Warn] source failed: {url}")

        if text is None:
            raise RuntimeError(f"Failed to download year {year}") from last_err

        df_year = parse_scsn_catalog_text(text)
        print(f"[Info] parsed year={year}, raw_events={len(df_year)}")
        all_dfs.append(df_year)

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    print(f"[Info] raw combined events = {len(df)}")

    df = apply_filters(df, args)

    if len(df) > 0:
        df = df.drop_duplicates(subset=["event_id"]).copy()
        df = df.sort_values("origin_time").reset_index(drop=True)

    keep_cols = [
        "event_id",
        "resource_id",
        "origin_time",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "mag_type",
        "event_type",
        "client",
        "split",
        "quality",
        "nph",
        "rms",
        "catalog",
    ]

    if len(df) == 0:
        df = pd.DataFrame(columns=keep_cols)
    else:
        df = df[[c for c in keep_cols if c in df.columns]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"[Done] wrote filtered events = {len(df)} -> {out}")

    if len(df) > 0:
        print("[Preview]")
        print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()