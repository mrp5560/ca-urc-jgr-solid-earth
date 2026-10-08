#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast SCEDC waveform downloader.

Paper-ready small patch: fixes argparse boolean flags and defaults to keeping many stations per event.

Compared with the original version:
  1. Build/reuse a station-channel cache once.
  2. Select nearby stations locally for each event.
  3. Download waveforms only for valid 3-component channel groups.
  4. Cache StationXML by year/station/channel pattern.
  5. Optionally stop after enough stations are downloaded.

Typical outputs:
  raw_fast/
    _cache/
      station_cache.csv
    _stationxml_cache/
      2015/
        CI.ADO.__.HHX.xml
    <event_id>/
      waveforms/
        CI.ADO.__.HHX.mseed
      raw_records.csv
    download_summary.csv
"""

import argparse
import fnmatch
import re
import shutil
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics.base import gps2dist_azimuth


def parse_csv_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def fdsn_location(loc: str) -> str:
    loc = "" if pd.isna(loc) else str(loc)
    return loc if loc not in ["", "__", "--"] else "--"


def display_location(loc: str) -> str:
    loc = "" if pd.isna(loc) else str(loc)
    return loc if loc not in ["", "__", "--"] else "--"


def file_location(loc: str) -> str:
    loc = "" if pd.isna(loc) else str(loc)
    return loc if loc not in ["", "--"] else "__"


def station_key(net, sta, loc):
    return f"{net}.{sta}.{display_location(loc)}"


def channel_pattern_key(chan_pat: str) -> str:
    """
    Convert HH? -> HHX, BH? -> BHX for file names.
    """
    return safe_name(chan_pat.replace("?", "X").replace("*", "STAR"))


def has_three_components_stream(st):
    comps = set()
    for tr in st:
        ch = tr.stats.channel
        if not ch:
            continue
        c = ch[-1].upper()
        if c in ["E", "N", "Z", "1", "2", "3"]:
            comps.add(c)

    return (
        {"E", "N", "Z"}.issubset(comps)
        or {"1", "2", "Z"}.issubset(comps)
        or {"1", "2", "3"}.issubset(comps)
    )


def has_three_components_channels(channels):
    comps = set()
    for ch in channels:
        if not ch:
            continue
        c = str(ch)[-1].upper()
        if c in ["E", "N", "Z", "1", "2", "3"]:
            comps.add(c)

    return (
        {"E", "N", "Z"}.issubset(comps)
        or {"1", "2", "Z"}.issubset(comps)
        or {"1", "2", "3"}.issubset(comps)
    )


def utc_to_timestamp(t: UTCDateTime) -> pd.Timestamp:
    return pd.Timestamp(t.datetime).tz_localize("UTC")


def max_utcdatetime(a: UTCDateTime, b: UTCDateTime) -> UTCDateTime:
    return a if a >= b else b


def min_utcdatetime(a: UTCDateTime, b: UTCDateTime) -> UTCDateTime:
    return a if a <= b else b


def parse_inventory_to_rows(inv, cache_year=None):
    rows = []

    for net in inv:
        net_code = net.code

        for sta in net:
            sta_code = sta.code

            for ch in sta.channels:
                loc = ch.location_code or ""

                # Prefer channel coordinates; fall back to station coordinates.
                lat = ch.latitude if ch.latitude is not None else sta.latitude
                lon = ch.longitude if ch.longitude is not None else sta.longitude
                elev = ch.elevation if ch.elevation is not None else sta.elevation

                start_date = ch.start_date.isoformat() if ch.start_date else ""
                end_date = ch.end_date.isoformat() if ch.end_date else ""

                rows.append({
                    "network": net_code,
                    "station": sta_code,
                    "location": loc,
                    "channel": ch.code,
                    "station_latitude": float(lat),
                    "station_longitude": float(lon),
                    "station_elevation_m": float(elev) if elev is not None else float("nan"),
                    "channel_start": start_date,
                    "channel_end": end_date,
                    "cache_year": cache_year if cache_year is not None else "",
                })

    return rows


def build_or_load_station_cache(client, df_events, args, out_root: Path) -> pd.DataFrame:
    if args.station_cache:
        cache_path = Path(args.station_cache)
    else:
        cache_path = out_root / "_cache" / "station_cache.csv"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and cache_path.stat().st_size > 0 and not args.rebuild_station_cache:
        print(f"[Cache] loading station cache: {cache_path}")
        df = pd.read_csv(cache_path, dtype={
            "network": str,
            "station": str,
            "location": str,
            "channel": str,
        })
        return normalize_station_cache(df)

    print("[Info] building station cache from FDSN station service ...")

    t_min = UTCDateTime(str(df_events["origin_time"].min())) - 86400.0
    t_max = UTCDateTime(str(df_events["origin_time"].max())) + 86400.0

    y0 = t_min.datetime.year
    y1 = t_max.datetime.year

    channel_patterns = parse_csv_list(args.channel_priority)
    channel_query = ",".join(channel_patterns)

    all_rows = []

    for year in range(y0, y1 + 1):
        win1 = max_utcdatetime(UTCDateTime(year, 1, 1), t_min)
        win2 = min_utcdatetime(UTCDateTime(year + 1, 1, 1), t_max)

        if win2 <= win1:
            continue

        print(
            f"[Station] year={year}, network={args.network}, "
            f"channel={channel_query}, {win1} -> {win2}"
        )

        try:
            inv = client.get_stations(
                network=args.network,
                station="*",
                location=args.location,
                channel=channel_query,
                starttime=win1,
                endtime=win2,
                level="channel",
            )
            rows = parse_inventory_to_rows(inv, cache_year=year)
            all_rows.extend(rows)
            print(f"[Station] year={year}, channel rows={len(rows)}")

        except FDSNNoDataException:
            print(f"[Station] year={year}, no station data")
            continue

        except Exception as e:
            print(f"[Warn] station query failed for year={year}: {type(e).__name__}: {e}")
            continue

        if args.station_sleep > 0:
            time.sleep(args.station_sleep)

    if not all_rows:
        raise RuntimeError("No station-channel metadata was downloaded. Check network/channel/time range.")

    df = pd.DataFrame(all_rows)

    df = df.drop_duplicates(
        subset=[
            "network",
            "station",
            "location",
            "channel",
            "channel_start",
            "channel_end",
        ]
    ).reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    print(f"[Done] wrote station cache: {cache_path}, rows={len(df)}")

    return normalize_station_cache(df)


def normalize_station_cache(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["network", "station", "location", "channel"]:
        if col not in df.columns:
            raise ValueError(f"station cache missing column: {col}")
        df[col] = df[col].fillna("").astype(str)

    for col in ["station_latitude", "station_longitude", "station_elevation_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["channel_start_ts"] = pd.to_datetime(
        df.get("channel_start", ""), utc=True, errors="coerce"
    )
    df["channel_end_ts"] = pd.to_datetime(
        df.get("channel_end", ""), utc=True, errors="coerce"
    )

    return df


def select_event_candidates(ev, station_df: pd.DataFrame, channel_patterns, args):
    """
    Select candidate station-location-channel-pattern groups locally.
    """
    origin = UTCDateTime(ev.origin_time)
    evt_ts = utc_to_timestamp(origin)

    ev_lat = float(ev.latitude)
    ev_lon = float(ev.longitude)

    # Active channels at event time.
    active_mask = (
        (station_df["channel_start_ts"].isna() | (station_df["channel_start_ts"] <= evt_ts))
        & (station_df["channel_end_ts"].isna() | (station_df["channel_end_ts"] >= evt_ts))
    )

    df_active = station_df.loc[active_mask].copy()

    candidates = []

    group_cols = ["network", "station", "location"]

    for (net, sta, loc), g in df_active.groupby(group_cols, dropna=False):
        g = g.dropna(subset=["station_latitude", "station_longitude"])
        if len(g) == 0:
            continue

        slat = float(g.iloc[0]["station_latitude"])
        slon = float(g.iloc[0]["station_longitude"])
        elev = float(g.iloc[0]["station_elevation_m"])

        dist_m, az, baz = gps2dist_azimuth(ev_lat, ev_lon, slat, slon)
        dist_km = dist_m / 1000.0

        if dist_km < args.min_radius_km:
            continue
        if dist_km > args.max_radius_km:
            continue

        channels = sorted(set(g["channel"].astype(str).tolist()))

        selected_pat = None
        selected_channels = None

        for pat in channel_patterns:
            matched = sorted([ch for ch in channels if fnmatch.fnmatchcase(ch, pat)])
            if has_three_components_channels(matched):
                selected_pat = pat
                selected_channels = matched
                break

        if selected_pat is None:
            continue

        candidates.append({
            "network": net,
            "station": sta,
            "location": loc,
            "channel_pattern": selected_pat,
            "available_channels": ",".join(selected_channels),
            "station_latitude": slat,
            "station_longitude": slon,
            "station_elevation_m": elev,
            "dist_km": dist_km,
            "azimuth": az,
            "back_azimuth": baz,
        })

    candidates = sorted(candidates, key=lambda x: x["dist_km"])

    if args.max_candidates is not None and args.max_candidates > 0:
        candidates = candidates[:args.max_candidates]

    return candidates


def get_waveforms_retry(client, args, net, sta, loc, chan_pat, t1, t2):
    last_err = None

    for i in range(args.retries):
        try:
            return client.get_waveforms(
                network=net,
                station=sta,
                location=fdsn_location(loc),
                channel=chan_pat,
                starttime=t1,
                endtime=t2,
                attach_response=False,
            )

        except FDSNNoDataException:
            raise

        except Exception as e:
            last_err = e
            if args.debug:
                print(
                    f"[Warn] waveform retry {i + 1}/{args.retries}: "
                    f"{net}.{sta}.{display_location(loc)}.{chan_pat}, "
                    f"{type(e).__name__}: {e}"
                )
            time.sleep(args.sleep * (i + 1))

    raise last_err


def get_stationxml_cached(client, args, cand, t1, t2, event_year, ev_xml_dir: Path, xml_cache_dir: Path):
    if args.no_stationxml:
        return ""

    net = cand["network"]
    sta = cand["station"]
    loc = cand["location"]
    chan_pat = cand["channel_pattern"]

    loc_file = file_location(loc)
    pat_file = channel_pattern_key(chan_pat)
    xml_name = f"{net}.{sta}.{loc_file}.{pat_file}.xml"

    if args.stationxml_cache_mode == "event":
        xml_path = ev_xml_dir / xml_name
    else:
        xml_path = xml_cache_dir / str(event_year) / xml_name

    xml_path.parent.mkdir(parents=True, exist_ok=True)

    if xml_path.exists() and xml_path.stat().st_size > 0 and not args.rebuild_stationxml:
        return str(xml_path)

    last_err = None

    for i in range(args.retries):
        try:
            inv = client.get_stations(
                network=net,
                station=sta,
                location=fdsn_location(loc),
                channel=chan_pat,
                starttime=t1,
                endtime=t2,
                level="response",
            )

            inv.write(str(xml_path), format="STATIONXML")

            # Optional compatibility copy to each event folder.
            if args.copy_stationxml_to_event and args.stationxml_cache_mode != "event":
                event_xml_path = ev_xml_dir / xml_name
                event_xml_path.parent.mkdir(parents=True, exist_ok=True)
                if not event_xml_path.exists():
                    shutil.copy2(xml_path, event_xml_path)
                return str(event_xml_path)

            return str(xml_path)

        except Exception as e:
            last_err = e
            if args.debug:
                print(
                    f"[Warn] stationxml retry {i + 1}/{args.retries}: "
                    f"{net}.{sta}.{display_location(loc)}.{chan_pat}, "
                    f"{type(e).__name__}: {e}"
                )
            time.sleep(args.sleep * (i + 1))

    if args.allow_missing_stationxml:
        return ""

    raise last_err


def download_one_event(client, ev, station_df, channel_patterns, args, out_root: Path, xml_cache_dir: Path):
    eid = str(ev.event_id)

    ev_dir = out_root / eid
    wf_dir = ev_dir / "waveforms"
    xml_dir = ev_dir / "stationxml"

    ev_dir.mkdir(parents=True, exist_ok=True)
    wf_dir.mkdir(exist_ok=True)
    xml_dir.mkdir(exist_ok=True)

    raw_csv = ev_dir / "raw_records.csv"
    err_csv = ev_dir / "download_errors.csv"

    if args.resume and raw_csv.exists() and raw_csv.stat().st_size > 0:
        try:
            old = pd.read_csv(raw_csv)
            return {
                "event_id": eid,
                "n_candidates": -1,
                "n_downloaded": len(old),
                "status": "skipped_resume",
            }
        except Exception:
            pass

    origin = UTCDateTime(ev.origin_time)
    t1 = origin - args.pre_sec
    t2 = origin + args.post_sec
    event_year = origin.datetime.year

    candidates = select_event_candidates(ev, station_df, channel_patterns, args)

    rows = []
    err_rows = []

    if len(candidates) == 0:
        pd.DataFrame(rows).to_csv(raw_csv, index=False)
        return {
            "event_id": eid,
            "n_candidates": 0,
            "n_downloaded": 0,
            "status": "no_candidates",
        }

    for cand in candidates:
        if args.stop_after_min_stations and len(rows) >= args.min_stations:
            break

        net = cand["network"]
        sta = cand["station"]
        loc = cand["location"]
        chan_pat = cand["channel_pattern"]

        loc_file = file_location(loc)
        pat_file = channel_pattern_key(chan_pat)
        safe_key = f"{net}.{sta}.{loc_file}.{pat_file}"

        mseed_path = wf_dir / f"{safe_key}.mseed"

        try:
            if args.resume and mseed_path.exists() and mseed_path.stat().st_size > 0:
                waveform_status = "reused"
            else:
                st = get_waveforms_retry(
                    client=client,
                    args=args,
                    net=net,
                    sta=sta,
                    loc=loc,
                    chan_pat=chan_pat,
                    t1=t1,
                    t2=t2,
                )

                if len(st) < 3 or not has_three_components_stream(st):
                    err_rows.append({
                        "event_id": eid,
                        "network": net,
                        "station": sta,
                        "location": display_location(loc),
                        "channel_pattern": chan_pat,
                        "reason": "not_three_components",
                    })
                    continue

                # Merge fragmented traces and trim to a consistent event window.
                try:
                    st.merge(method=1, fill_value="interpolate")
                except Exception:
                    pass

                st.trim(t1, t2, pad=False)

                if len(st) < 3 or not has_three_components_stream(st):
                    err_rows.append({
                        "event_id": eid,
                        "network": net,
                        "station": sta,
                        "location": display_location(loc),
                        "channel_pattern": chan_pat,
                        "reason": "not_three_components_after_trim",
                    })
                    continue

                st.write(str(mseed_path), format="MSEED")
                waveform_status = "downloaded"

            stationxml_path = get_stationxml_cached(
                client=client,
                args=args,
                cand=cand,
                t1=t1,
                t2=t2,
                event_year=event_year,
                ev_xml_dir=xml_dir,
                xml_cache_dir=xml_cache_dir,
            )

            rows.append({
                "event_id": eid,
                "origin_time": str(ev.origin_time),
                "network": net,
                "station": sta,
                "location": display_location(loc),
                "channel_pattern": chan_pat,
                "available_channels": cand["available_channels"],
                "station_id": station_key(net, sta, loc),
                "station_latitude": cand["station_latitude"],
                "station_longitude": cand["station_longitude"],
                "station_elevation_m": cand["station_elevation_m"],
                "dist_km": cand["dist_km"],
                "azimuth": cand["azimuth"],
                "back_azimuth": cand["back_azimuth"],
                "mseed_path": str(mseed_path),
                "stationxml_path": stationxml_path,
                "starttime": t1.isoformat(),
                "endtime": t2.isoformat(),
                "waveform_status": waveform_status,
            })

        except FDSNNoDataException:
            err_rows.append({
                "event_id": eid,
                "network": net,
                "station": sta,
                "location": display_location(loc),
                "channel_pattern": chan_pat,
                "reason": "no_data",
            })
            continue

        except Exception as e:
            err_rows.append({
                "event_id": eid,
                "network": net,
                "station": sta,
                "location": display_location(loc),
                "channel_pattern": chan_pat,
                "reason": f"{type(e).__name__}: {e}",
            })
            if args.debug:
                print(
                    f"[Warn] failed: event={eid}, "
                    f"{net}.{sta}.{display_location(loc)}.{chan_pat}: "
                    f"{type(e).__name__}: {e}"
                )
            continue

        finally:
            if args.sleep > 0:
                time.sleep(args.sleep)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(raw_csv, index=False)

    if args.write_errors:
        pd.DataFrame(err_rows).to_csv(err_csv, index=False)

    if len(raw_df) >= args.min_stations:
          status = "ok"
    else:
        status = f"too_few_stations:{len(raw_df)}"

    return {
        "event_id": eid,
        "n_candidates": len(candidates),
        "n_downloaded": len(raw_df),
        "status": status,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--client", default="SCEDC")
    p.add_argument("--event-csv", default="./data/scedc/events_m3_2010_2024.csv")
    p.add_argument("--out-root", default="data/scedc/raw_m3_2010_2025")

    p.add_argument("--network", default="CI")
    p.add_argument("--channel-priority", default="HN?,HH?,BH?,EH?")
    p.add_argument("--location", default="*")

    p.add_argument("--pre-sec", type=float, default=10.0)
    p.add_argument("--post-sec", type=float, default=90.0)

    p.add_argument("--min-radius-km", type=float, default=0.0)
    p.add_argument("--max-radius-km", type=float, default=250.0)
    p.add_argument("--max-candidates", type=int, default=200)

    p.add_argument("--min-stations", type=int, default=25)
    p.add_argument("--stop-after-min-stations", action=argparse.BooleanOptionalAction, default=False,
                   help="For paper data, keep False so each event retains enough held-out target stations. Use --stop-after-min-stations only for a tiny pilot.")

    p.add_argument("--limit-events", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)

    p.add_argument("--station-cache", default=None)
    p.add_argument("--rebuild-station-cache", action="store_true")
    p.add_argument("--station-sleep", type=float, default=0.5)

    p.add_argument("--no-stationxml", action="store_true", default=False)
    p.add_argument("--stationxml-cache-dir", default=None)
    p.add_argument(
        "--stationxml-cache-mode",
        default="event",
        choices=["year", "event"],
        help="year: cache StationXML globally by year; event: save StationXML in each event folder.",
    )
    p.add_argument("--rebuild-stationxml", action="store_true")
    p.add_argument("--allow-missing-stationxml", action="store_true")
    p.add_argument("--copy-stationxml-to-event", action="store_true")

    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--debug", default=True)
    p.add_argument("--write-errors", action=argparse.BooleanOptionalAction, default=True)

    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.stationxml_cache_dir:
        xml_cache_dir = Path(args.stationxml_cache_dir)
    else:
        xml_cache_dir = out_root / "_stationxml_cache"

    xml_cache_dir.mkdir(parents=True, exist_ok=True)

    df_events = pd.read_csv(args.event_csv, dtype={"event_id": str})
    if "origin_time" not in df_events.columns:
        raise ValueError("event_csv must contain column: origin_time")

    df_events = df_events.sort_values("origin_time").reset_index(drop=True)

    if args.start_index > 0:
        df_events = df_events.iloc[args.start_index:].copy()

    if args.limit_events is not None:
        df_events = df_events.iloc[:args.limit_events].copy()

    print(f"[Info] events to process = {len(df_events)}")
    print(f"[Info] output root = {out_root}")

    client = Client(args.client)
    channel_patterns = parse_csv_list(args.channel_priority)

    station_df = build_or_load_station_cache(
        client=client,
        df_events=df_events,
        args=args,
        out_root=out_root,
    )

    print(f"[Info] station cache rows = {len(station_df)}")
    print(f"[Info] channel priority = {channel_patterns}")
    print(f"[Info] max_radius_km = {args.max_radius_km}")
    print(f"[Info] max_candidates = {args.max_candidates}")
    print(f"[Info] no_stationxml = {args.no_stationxml}")

    summary = []

    for _, ev in tqdm(df_events.iterrows(), total=len(df_events), desc="events"):
        result = download_one_event(
            client=client,
            ev=ev,
            station_df=station_df,
            channel_patterns=channel_patterns,
            args=args,
            out_root=out_root,
            xml_cache_dir=xml_cache_dir,
        )
        summary.append(result)

        # Incrementally save summary to avoid losing progress.
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(out_root / "download_summary_partial.csv", index=False)

    summary_df = pd.DataFrame(summary)
    summary_path = out_root / "download_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[Done] wrote summary -> {summary_path}")

    if len(summary_df) > 0:
        print("[Summary status counts]")
        print(summary_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()