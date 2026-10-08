#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_parse_scedc_phase_files_v3.py

Robust parser for SCEDC AWS event_phases files.

Improvements:
- Reconstruct network/station/location from station_id or miniSEED filename.
- Read UTF-8/UTF-8-BOM/Latin-1 and optional gzip content.
- Do not assume latitude, longitude and elevation occupy fixed split tokens.
- Supports latitude and negative longitude being joined, e.g.
      33.8284-117.7894
- Detects P/S token and reads the final three numeric values as
  weight, source distance and travel time.
- Writes rejection diagnostics and sample unparsed lines.

Outputs:
  phase_picks_all.csv
  station_phase_table.csv
  phase_event_qc.csv
  missing_or_invalid_phase_events.csv
  phase_parse_diagnostics.csv
  phase_unparsed_samples.txt
"""

from __future__ import annotations

import argparse
import gzip
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
)


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def normalize_location(value: Any) -> str:
    if pd.isna(value):
        return "--"
    text = str(value).strip()
    return "--" if text in {"", "__", "--", "nan", "None"} else text


def parse_station_identity_from_id(value: Any) -> tuple[str, str, str]:
    if pd.isna(value):
        return "", "", "--"
    parts = str(value).strip().split(".")
    if len(parts) >= 2:
        net = parts[0].strip()
        sta = parts[1].strip()
        loc = normalize_location(parts[2] if len(parts) >= 3 else "--")
        return net, sta, loc
    return "", "", "--"


def parse_station_identity_from_mseed(value: Any) -> tuple[str, str, str]:
    if pd.isna(value):
        return "", "", "--"
    name = Path(str(value)).name
    parts = name.split(".")
    # Downloader naming: NET.STA.LOC.CHANNEL.mseed
    if len(parts) >= 4:
        return parts[0], parts[1], normalize_location(parts[2])
    return "", "", "--"


def normalize_station_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["event_id", "station_id", "mseed_path", "stationxml_path"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if "network" not in df.columns:
        df["network"] = ""
    if "station" not in df.columns:
        df["station"] = ""
    if "location" not in df.columns:
        df["location"] = "--"

    df["network"] = df["network"].fillna("").astype(str).str.strip()
    df["station"] = df["station"].fillna("").astype(str).str.strip()
    df["location"] = df["location"].map(normalize_location)

    for idx in df.index:
        net = df.at[idx, "network"]
        sta = df.at[idx, "station"]
        loc = df.at[idx, "location"]

        if not net or not sta:
            n2, s2, l2 = parse_station_identity_from_id(
                df.at[idx, "station_id"]
            )
            if not net:
                net = n2
            if not sta:
                sta = s2
            if loc == "--" and l2 != "--":
                loc = l2

        if not net or not sta:
            n3, s3, l3 = parse_station_identity_from_mseed(
                df.at[idx, "mseed_path"]
            )
            if not net:
                net = n3
            if not sta:
                sta = s3
            if loc == "--" and l3 != "--":
                loc = l3

        df.at[idx, "network"] = net
        df.at[idx, "station"] = sta
        df.at[idx, "location"] = normalize_location(loc)

    return df


def decode_phase_file(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def find_coordinate_triplet(prefix_text: str):
    """
    Find plausible Southern California latitude, longitude and elevation.
    Handles both:
      33.8284 -117.7894 181.0
    and:
      33.8284-117.7894 181.0
    """
    values = []
    for match in FLOAT_RE.finditer(prefix_text):
        try:
            values.append((float(match.group()), match.group()))
        except ValueError:
            pass

    # Search every consecutive triplet. Optional numeric location codes before
    # the coordinates are therefore harmless.
    for i in range(max(0, len(values) - 2)):
        lat = values[i][0]
        lon = values[i + 1][0]
        elev = values[i + 2][0]

        if (
            20.0 <= lat <= 50.0
            and -130.0 <= lon <= -100.0
            and -2000.0 <= elev <= 10000.0
        ):
            return lat, lon, elev

    return None


def parse_phase_line(
    line: str,
    event_id: str,
    origin: pd.Timestamp,
) -> tuple[dict[str, Any] | None, str]:
    text = line.strip()

    if not text:
        return None, "blank"
    if text.startswith("#"):
        return None, "comment"

    parts = text.split()
    if len(parts) < 7:
        return None, "too_few_tokens"

    # Event header begins with event ID.
    if re.fullmatch(r"\d+", parts[0]):
        return None, "event_header"

    # Locate the P/S marker after the identity fields.
    phase_idx = None
    phase_raw = None
    for i in range(2, len(parts)):
        token = parts[i].strip().upper().rstrip(":")
        if token in {"P", "S"}:
            phase_idx = i
            phase_raw = token
            break
        # Tolerate labels such as Pg/Pn/Sg/Sn, while avoiding arbitrary words.
        if re.fullmatch(r"[PS][A-Z0-9]?", token):
            phase_idx = i
            phase_raw = token
            break

    if phase_idx is None:
        return None, "no_phase_token"

    if phase_idx < 4:
        return None, "phase_token_too_early"

    network = parts[0].strip()
    station = parts[1].strip()

    # Usually token 2 is channel. If a location code is present before the
    # channel, choose the nearest SEED-like 3-character channel before P/S.
    channel = parts[2].strip()
    for token in reversed(parts[2:phase_idx]):
        if re.fullmatch(r"[A-Za-z0-9]{3}", token):
            # Coordinates can also be 3 digits, so require at least one letter.
            if re.search(r"[A-Za-z]", token):
                channel = token
                break

    # Parse coordinates from everything between identity and P/S.
    prefix = " ".join(parts[3:phase_idx])
    coords = find_coordinate_triplet(prefix)
    if coords is None:
        # Fallback includes token 2 in case a location field shifted channel.
        coords = find_coordinate_triplet(" ".join(parts[2:phase_idx]))
    if coords is None:
        return None, "bad_coordinates"

    lat, lon, elev = coords

    # Read all numeric values after the phase token. In STP phase output, the
    # final three values are weight, distance_km and travel_time_sec.
    tail_text = " ".join(parts[phase_idx + 1:])
    tail_nums = [float(x) for x in FLOAT_RE.findall(tail_text)]

    if len(tail_nums) < 3:
        return None, "bad_numeric_tail"

    weight, distance_km, travel_time_sec = tail_nums[-3:]

    if not np.isfinite(travel_time_sec) or travel_time_sec < -5:
        return None, "invalid_travel_time"
    if not np.isfinite(distance_km) or distance_km < 0:
        return None, "invalid_distance"

    extras = " ".join(parts[phase_idx + 1:])
    phase_type = "P" if str(phase_raw).startswith("P") else "S"
    arrival = origin + pd.to_timedelta(travel_time_sec, unit="s")

    return {
        "event_id": str(event_id),
        "network": network,
        "station": station,
        "phase_channel": channel,
        "station_latitude_phase": lat,
        "station_longitude_phase": lon,
        "station_elevation_phase_m": elev,
        "phase_raw": phase_raw,
        "phase_type": phase_type,
        "phase_extras": extras,
        "phase_weight": weight,
        "phase_distance_km": distance_km,
        "travel_time_sec": travel_time_sec,
        "arrival_time": arrival.isoformat(),
        "raw_phase_line": text,
    }, "parsed"


def locate_phase_file(row: Any, phase_root: Path) -> Path:
    # Do not trust a relative phase_path unless it really exists.
    if hasattr(row, "phase_path"):
        value = getattr(row, "phase_path")
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            if candidate.exists():
                return candidate

    year = int(row.year)
    doy = int(row.doy)
    return (
        phase_root
        / str(year)
        / f"{year}_{doy:03d}"
        / f"{row.event_id}.phase"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--events",
        default="data/scedc/paper_events_m3_min30.csv",
    )
    p.add_argument(
        "--qc-stations",
        default="data/scedc/raw_m3_2010_2025/qc_download_stations.csv",
    )
    p.add_argument(
        "--phase-root",
        default="data/scedc/event_phases",
    )
    p.add_argument(
        "--out-dir",
        default="data/scedc/phases",
    )
    p.add_argument("--valid-column", default="valid")
    p.add_argument(
        "--max-unparsed-samples",
        type=int,
        default=300,
    )
    args = p.parse_args()

    events = pd.read_csv(args.events, dtype={"event_id": str})
    stations = pd.read_csv(args.qc_stations, dtype={"event_id": str})
    stations = normalize_station_table(stations)

    if args.valid_column in stations.columns:
        stations = stations.loc[
            normalize_bool(stations[args.valid_column])
        ].copy()

    recognized = (
        stations["network"].ne("")
        & stations["station"].ne("")
    )
    unrecognized_count = int((~recognized).sum())
    stations = stations.loc[recognized].copy()

    station_cols = [
        c for c in [
            "event_id", "network", "station", "location", "station_id",
            "mseed_path", "stationxml_path",
        ]
        if c in stations.columns
    ]
    station_keys = stations[station_cols].drop_duplicates(
        ["event_id", "network", "station"],
        keep="first",
    )

    if "year" not in events.columns or "doy" not in events.columns:
        t = pd.to_datetime(events["origin_time"], utc=True)
        events["year"] = t.dt.year
        events["doy"] = t.dt.dayofyear

    all_picks = []
    event_qc = []
    bad_events = []
    global_reasons = Counter()
    unparsed_samples = []

    phase_root = Path(args.phase_root)

    for row in tqdm(
        events.itertuples(index=False),
        total=len(events),
        desc="parse phases",
    ):
        event_id = str(row.event_id)
        origin = pd.to_datetime(row.origin_time, utc=True)
        phase_file = locate_phase_file(row, phase_root)

        ev_valid_stations = station_keys.loc[
            station_keys["event_id"] == event_id
        ]
        valid_pairs = set(
            zip(ev_valid_stations["network"], ev_valid_stations["station"])
        )

        if not phase_file.exists() or phase_file.stat().st_size == 0:
            bad_events.append({
                "event_id": event_id,
                "phase_path": str(phase_file),
                "reason": "missing_or_empty",
            })
            event_qc.append({
                "event_id": event_id,
                "phase_file_ok": False,
                "n_nonempty_lines": 0,
                "n_parsed_phase_lines": 0,
                "n_p_stations": 0,
                "n_s_stations": 0,
                "n_valid_download_stations": len(valid_pairs),
                "n_matched_p_stations": 0,
                "n_matched_s_stations": 0,
            })
            continue

        parsed = []
        local_reasons = Counter()
        nonempty = 0

        try:
            text = decode_phase_file(phase_file)
            lines = text.splitlines()

            for line_no, line in enumerate(lines, start=1):
                if line.strip():
                    nonempty += 1

                rec, reason = parse_phase_line(
                    line=line,
                    event_id=event_id,
                    origin=origin,
                )
                local_reasons[reason] += 1
                global_reasons[reason] += 1

                if rec is not None:
                    rec["phase_path"] = str(phase_file)
                    rec["phase_line_no"] = line_no
                    parsed.append(rec)
                    all_picks.append(rec)
                elif (
                    reason not in {"blank", "comment", "event_header"}
                    and len(unparsed_samples) < args.max_unparsed_samples
                    and line.strip()
                ):
                    unparsed_samples.append(
                        f"{event_id}\t{phase_file}\tline={line_no}"
                        f"\treason={reason}\t{line.rstrip()}"
                    )

        except Exception as exc:
            bad_events.append({
                "event_id": event_id,
                "phase_path": str(phase_file),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            event_qc.append({
                "event_id": event_id,
                "phase_file_ok": False,
                "n_nonempty_lines": 0,
                "n_parsed_phase_lines": 0,
                "n_p_stations": 0,
                "n_s_stations": 0,
                "n_valid_download_stations": len(valid_pairs),
                "n_matched_p_stations": 0,
                "n_matched_s_stations": 0,
            })
            continue

        p_stations = {
            (x["network"], x["station"])
            for x in parsed
            if x["phase_type"] == "P"
        }
        s_stations = {
            (x["network"], x["station"])
            for x in parsed
            if x["phase_type"] == "S"
        }

        event_qc.append({
            "event_id": event_id,
            "phase_file_ok": True,
            "n_nonempty_lines": nonempty,
            "n_parsed_phase_lines": len(parsed),
            "n_p_stations": len(p_stations),
            "n_s_stations": len(s_stations),
            "n_valid_download_stations": len(valid_pairs),
            "n_matched_p_stations": len(valid_pairs & p_stations),
            "n_matched_s_stations": len(valid_pairs & s_stations),
            "parse_fraction": (
                len(parsed) / nonempty if nonempty else 0.0
            ),
            "top_rejection_reason": (
                local_reasons.most_common(1)[0][0]
                if local_reasons else ""
            ),
        })

    picks = pd.DataFrame(all_picks)
    qc = pd.DataFrame(event_qc)
    bad = pd.DataFrame(bad_events)

    if len(picks):
        picks = picks.sort_values(
            [
                "event_id", "network", "station",
                "phase_type", "travel_time_sec",
            ]
        )

        best = picks.drop_duplicates(
            ["event_id", "network", "station", "phase_type"],
            keep="first",
        )

        value_cols = [
            "arrival_time",
            "travel_time_sec",
            "phase_weight",
            "phase_distance_km",
            "phase_channel",
            "phase_raw",
        ]

        wide = best.pivot(
            index=["event_id", "network", "station"],
            columns="phase_type",
            values=value_cols,
        )
        wide.columns = [
            f"{name}_{phase.lower()}"
            for name, phase in wide.columns
        ]
        wide = wide.reset_index()

        station_phase = station_keys.merge(
            wide,
            on=["event_id", "network", "station"],
            how="left",
            validate="one_to_one",
        )
        station_phase["has_p_pick"] = (
            station_phase["arrival_time_p"].notna()
            if "arrival_time_p" in station_phase.columns
            else False
        )
        station_phase["has_s_pick"] = (
            station_phase["arrival_time_s"].notna()
            if "arrival_time_s" in station_phase.columns
            else False
        )
    else:
        station_phase = station_keys.copy()
        station_phase["has_p_pick"] = False
        station_phase["has_s_pick"] = False

    diagnostics = pd.DataFrame(
        [
            {"reason": reason, "count": count}
            for reason, count in global_reasons.most_common()
        ]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    picks.to_csv(out_dir / "phase_picks_all.csv", index=False)
    station_phase.to_csv(
        out_dir / "station_phase_table.csv", index=False
    )
    qc.to_csv(out_dir / "phase_event_qc.csv", index=False)
    bad.to_csv(
        out_dir / "missing_or_invalid_phase_events.csv",
        index=False,
    )
    diagnostics.to_csv(
        out_dir / "phase_parse_diagnostics.csv", index=False
    )
    (out_dir / "phase_unparsed_samples.txt").write_text(
        "\n".join(unparsed_samples),
        encoding="utf-8",
    )

    print("\n=== Phase parsing summary ===")
    print(f"Paper events                  : {len(events)}")
    print(
        f"Phase files OK                : "
        f"{int(qc['phase_file_ok'].sum()) if len(qc) else 0}"
    )
    print(f"Missing/invalid phase events  : {len(bad)}")
    print(f"Parsed phase records          : {len(picks)}")
    print(f"Valid downloaded stations     : {len(station_keys)}")
    print(f"Unrecognized station rows     : {unrecognized_count}")
    print(
        f"Stations with P pick          : "
        f"{int(station_phase['has_p_pick'].sum())}"
    )
    print(
        f"Stations with S pick          : "
        f"{int(station_phase['has_s_pick'].sum())}"
    )

    if len(qc):
        print("\nMatched P-station distribution per event:")
        print(qc["n_matched_p_stations"].describe().to_string())

    print("\nTop line classifications:")
    for reason, count in global_reasons.most_common(12):
        print(f"{reason:24s} {count}")

    print(f"\nWrote outputs to: {out_dir}")
    print(
        "If Parsed phase records is still 0, inspect:\n"
        f"  {out_dir / 'phase_unparsed_samples.txt'}"
    )


if __name__ == "__main__":
    main()
