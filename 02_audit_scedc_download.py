#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit downloaded SCEDC waveform data with crash-safe resume support.

Main improvements over the original script:
1. Empty/header-only/malformed raw_records.csv files are recorded and skipped.
2. QC results are committed to SQLite after every event, so an interruption does
   not lose completed work.
3. Re-running the same command automatically resumes unfinished events.
4. Empty path values cannot be mistaken for the current directory.
5. Valid station counts are based on unique station IDs rather than raw rows.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from obspy import read
from tqdm import tqdm

STATION_COLUMNS = [
    "event_id",
    "row_index",
    "station_id",
    "mseed_path",
    "stationxml_path",
    "mseed_exists",
    "stationxml_exists",
    "readable",
    "three_components",
    "duration_ok",
    "gap_ok",
    "valid",
    "gap_sec",
    "error",
]

EVENT_COLUMNS = [
    "event_id",
    "n_raw",
    "n_raw_stations",
    "n_valid",
    "eligible",
    "raw_csv_status",
    "raw_csv_error",
]


def three_components(st) -> bool:
    """Return True when the stream contains a recognized three-component set."""
    comps = {
        str(tr.stats.channel)[-1].upper()
        for tr in st
        if getattr(tr.stats, "channel", "")
    }
    return (
            {"E", "N", "Z"}.issubset(comps)
            or {"1", "2", "Z"}.issubset(comps)
            or {"1", "2", "3"}.issubset(comps)
    )


def clean_text(value: Any) -> str:
    """Convert a CSV value to a stripped string without turning NaN into 'nan'."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def missing_placeholder(event_dir: Path, subdir: str, kind: str) -> Path:
    """Return a guaranteed non-existing placeholder path."""
    return event_dir / subdir / f"__missing_{kind}__"


def find_existing(
        path_value: Any,
        event_dir: Path,
        subdir: str,
        kind: str,
) -> Path:
    """
    Resolve a path stored in raw_records.csv.

    It first tries the stored path, then the expected event subdirectory using
    only the basename. Empty/NaN values return a non-existing placeholder.
    """
    text = clean_text(path_value)
    if not text:
        return missing_placeholder(event_dir, subdir, kind)

    p = Path(text)
    candidates = [p]
    if p.name:
        candidates.append(event_dir / subdir / p.name)

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue

    # Preserve the most useful expected location in QC output.
    return candidates[-1]


def safe_read_raw_csv(raw_csv: Path) -> Tuple[Optional[pd.DataFrame], str, str]:
    """Read raw_records.csv without allowing one bad file to stop the audit."""
    if not raw_csv.exists():
        return None, "missing", ""

    try:
        if not raw_csv.is_file():
            return None, "not_a_file", "Path exists but is not a regular file"
        if raw_csv.stat().st_size == 0:
            return None, "zero_bytes", ""
    except OSError as exc:
        return None, "stat_error", f"{type(exc).__name__}: {exc}"

    try:
        raw = pd.read_csv(raw_csv)
    except pd.errors.EmptyDataError as exc:
        # Typical case: file contains only blank lines, whitespace, or a BOM.
        return None, "empty_no_columns", f"{type(exc).__name__}: {exc}"
    except pd.errors.ParserError as exc:
        return None, "parse_error", f"{type(exc).__name__}: {exc}"
    except UnicodeDecodeError as exc:
        return None, "decode_error", f"{type(exc).__name__}: {exc}"
    except (OSError, PermissionError) as exc:
        return None, "read_error", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, "unexpected_read_error", f"{type(exc).__name__}: {exc}"

    if len(raw.columns) == 0:
        return None, "empty_no_columns", "CSV contains no columns"
    if raw.empty:
        return raw, "header_only", ""

    return raw, "ok", ""


def connect_progress_db(db_path: Path) -> sqlite3.Connection:
    """Open the progress database and create tables if needed."""
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS event_qc (
            event_id TEXT PRIMARY KEY,
            n_raw INTEGER NOT NULL,
            n_raw_stations INTEGER NOT NULL,
            n_valid INTEGER NOT NULL,
            eligible INTEGER NOT NULL,
            raw_csv_status TEXT NOT NULL,
            raw_csv_error TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS station_qc (
            event_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            station_id TEXT NOT NULL,
            mseed_path TEXT NOT NULL,
            stationxml_path TEXT NOT NULL,
            mseed_exists INTEGER NOT NULL,
            stationxml_exists INTEGER NOT NULL,
            readable INTEGER NOT NULL,
            three_components INTEGER NOT NULL,
            duration_ok INTEGER NOT NULL,
            gap_ok INTEGER NOT NULL,
            valid INTEGER NOT NULL,
            gap_sec REAL,
            error TEXT NOT NULL,
            PRIMARY KEY (event_id, row_index)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_station_qc_event ON station_qc(event_id)"
    )
    conn.commit()
    return conn


def save_event_transaction(
        conn: sqlite3.Connection,
        event_row: Dict[str, Any],
        station_rows: List[Dict[str, Any]],
) -> None:
    """Atomically replace all QC records for one event."""
    event_id = str(event_row["event_id"])

    station_values = []
    for row in station_rows:
        station_values.append(
            (
                str(row["event_id"]),
                int(row["row_index"]),
                clean_text(row["station_id"]),
                str(row["mseed_path"]),
                str(row["stationxml_path"]),
                int(bool(row["mseed_exists"])),
                int(bool(row["stationxml_exists"])),
                int(bool(row["readable"])),
                int(bool(row["three_components"])),
                int(bool(row["duration_ok"])),
                int(bool(row["gap_ok"])),
                int(bool(row["valid"])),
                None if row.get("gap_sec") is None else float(row["gap_sec"]),
                clean_text(row.get("error", "")),
            )
        )

    # The context manager makes deletion + insertion + event completion atomic.
    with conn:
        conn.execute("DELETE FROM station_qc WHERE event_id = ?", (event_id,))
        if station_values:
            conn.executemany(
                """
                INSERT INTO station_qc (
                    event_id, row_index, station_id,
                    mseed_path, stationxml_path,
                    mseed_exists, stationxml_exists, readable,
                    three_components, duration_ok, gap_ok, valid,
                    gap_sec, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                station_values,
            )

        conn.execute(
            """
            INSERT OR REPLACE INTO event_qc (
                event_id, n_raw, n_raw_stations, n_valid, eligible,
                raw_csv_status, raw_csv_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                int(event_row["n_raw"]),
                int(event_row["n_raw_stations"]),
                int(event_row["n_valid"]),
                int(bool(event_row["eligible"])),
                clean_text(event_row["raw_csv_status"]),
                clean_text(event_row["raw_csv_error"]),
            ),
        )


def audit_event(
        event_id: str,
        root: Path,
        min_stations: int,
        expected_sec: float,
        min_duration_ratio: float,
        max_gap_sec: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Audit one event and return one event row plus all station rows."""
    event_dir = root / event_id
    raw_csv = event_dir / "raw_records.csv"
    raw, raw_status, raw_error = safe_read_raw_csv(raw_csv)

    if raw is None or raw.empty:
        return (
            {
                "event_id": event_id,
                "n_raw": 0,
                "n_raw_stations": 0,
                "n_valid": 0,
                "eligible": False,
                "raw_csv_status": raw_status,
                "raw_csv_error": raw_error,
            },
            [],
        )

    station_rows: List[Dict[str, Any]] = []
    raw_station_ids = set()
    valid_station_ids = set()

    for row_index, (_, r) in enumerate(raw.iterrows()):
        station_id = clean_text(r.get("station_id", ""))
        mseed = find_existing(
            r.get("mseed_path", ""), event_dir, "waveforms", "mseed"
        )
        xml = find_existing(
            r.get("stationxml_path", ""), event_dir, "stationxml", "xml"
        )

        # Fall back to the waveform filename for uniqueness when station_id is absent.
        station_key = station_id or mseed.stem or f"row_{row_index}"
        raw_station_ids.add(station_key)

        try:
            mseed_exists = mseed.is_file() and mseed.stat().st_size > 1024
        except OSError:
            mseed_exists = False

        try:
            stationxml_exists = xml.is_file() and xml.stat().st_size > 0
        except OSError:
            stationxml_exists = False

        row: Dict[str, Any] = {
            "event_id": event_id,
            "row_index": row_index,
            "station_id": station_id,
            "mseed_path": str(mseed),
            "stationxml_path": str(xml),
            "mseed_exists": mseed_exists,
            "stationxml_exists": stationxml_exists,
            "readable": False,
            "three_components": False,
            "duration_ok": False,
            "gap_ok": False,
            "valid": False,
            "gap_sec": None,
            "error": "",
        }

        if mseed_exists:
            try:
                st = read(str(mseed))
                row["readable"] = True
                row["three_components"] = three_components(st)

                durations = [
                    float(tr.stats.endtime - tr.stats.starttime)
                    for tr in st
                ]
                row["duration_ok"] = bool(
                    durations
                    and min(durations)
                    >= expected_sec * min_duration_ratio
                )

                gap_sec = sum(
                    max(0.0, float(gap[6]))
                    for gap in st.get_gaps()
                )
                row["gap_sec"] = gap_sec
                row["gap_ok"] = gap_sec <= max_gap_sec

                row["valid"] = bool(
                    stationxml_exists
                    and row["readable"]
                    and row["three_components"]
                    and row["duration_ok"]
                    and row["gap_ok"]
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"

        if row["valid"]:
            valid_station_ids.add(station_key)
        station_rows.append(row)

    n_valid = len(valid_station_ids)
    event_row = {
        "event_id": event_id,
        "n_raw": int(len(raw)),
        "n_raw_stations": int(len(raw_station_ids)),
        "n_valid": int(n_valid),
        "eligible": bool(n_valid >= min_stations),
        "raw_csv_status": raw_status,
        "raw_csv_error": raw_error,
    }
    return event_row, station_rows


def export_csv_outputs(
        conn: sqlite3.Connection,
        root: Path,
        summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export current SQLite progress to the original CSV deliverables."""
    event_core = pd.read_sql_query(
        "SELECT * FROM event_qc", conn, dtype={"event_id": str}
    )
    station_df = pd.read_sql_query(
        "SELECT * FROM station_qc", conn, dtype={"event_id": str}
    )

    summary_ids = set(summary["event_id"].astype(str))
    if not event_core.empty:
        event_core = event_core[event_core["event_id"].isin(summary_ids)].copy()
        event_core["eligible"] = event_core["eligible"].astype(bool)
    else:
        event_core = pd.DataFrame(columns=EVENT_COLUMNS)

    if not station_df.empty:
        station_df = station_df[station_df["event_id"].isin(summary_ids)].copy()
        for col in [
            "mseed_exists",
            "stationxml_exists",
            "readable",
            "three_components",
            "duration_ok",
            "gap_ok",
            "valid",
        ]:
            station_df[col] = station_df[col].astype(bool)
        station_df = station_df.sort_values(["event_id", "row_index"])
    else:
        station_df = pd.DataFrame(columns=STATION_COLUMNS)

    # Avoid duplicate non-key columns if the summary already contains QC fields.
    qc_names = set(EVENT_COLUMNS) - {"event_id"}
    summary_clean = summary.drop(
        columns=[col for col in summary.columns if col in qc_names],
        errors="ignore",
    )
    event_df = event_core.merge(summary_clean, on="event_id", how="left")

    order_map = {
        event_id: i for i, event_id in enumerate(summary["event_id"].astype(str))
    }
    if not event_df.empty:
        event_df["__order"] = event_df["event_id"].map(order_map)
        event_df = event_df.sort_values("__order").drop(columns="__order")

    station_df.to_csv(root / "qc_download_stations.csv", index=False)
    event_df.to_csv(root / "qc_download_events.csv", index=False)
    event_df[event_df["eligible"]].to_csv(
        root / "eligible_events.csv", index=False
    )
    return station_df, event_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/scedc/raw_m3_2010_2025")
    parser.add_argument("--min-stations", type=int, default=15)
    parser.add_argument("--expected-sec", type=float, default=100.0)
    parser.add_argument("--min-duration-ratio", type=float, default=0.98)
    parser.add_argument("--max-gap-sec", type=float, default=1.0)
    parser.add_argument(
        "--progress-db",
        default="qc_audit_progress.sqlite",
        help="SQLite checkpoint filename stored under --root",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete previous QC progress and audit every event again",
    )
    parser.add_argument(
        "--export-every",
        type=int,
        default=100,
        help="Refresh output CSV files every N newly processed events; 0 disables",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    summary_path = root / "download_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing download summary: {summary_path}")

    summary = pd.read_csv(summary_path, dtype={"event_id": str})
    if "event_id" not in summary.columns:
        raise ValueError(f"Missing 'event_id' column in {summary_path}")

    summary["event_id"] = summary["event_id"].astype(str).str.strip()
    summary = summary[summary["event_id"].ne("")].drop_duplicates(
        subset="event_id", keep="first"
    )
    event_ids = summary["event_id"].tolist()

    db_path = root / args.progress_db
    if args.restart and db_path.exists():
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    conn = connect_progress_db(db_path)
    try:
        processed = {
            str(row[0])
            for row in conn.execute("SELECT event_id FROM event_qc").fetchall()
        }
        event_id_set = set(event_ids)
        completed_in_summary = processed & event_id_set
        pending = [event_id for event_id in event_ids if event_id not in processed]

        print(f"Root: {root}")
        print(f"Progress DB: {db_path}")
        print(f"Events in summary: {len(event_ids)}")
        print(f"Already completed: {len(completed_in_summary)}")
        print(f"Pending: {len(pending)}")

        newly_done = 0
        progress = tqdm(
            pending,
            total=len(event_ids),
            initial=len(completed_in_summary),
            desc="QC",
        )

        try:
            for event_id in progress:
                event_row, station_rows = audit_event(
                    event_id=event_id,
                    root=root,
                    min_stations=args.min_stations,
                    expected_sec=args.expected_sec,
                    min_duration_ratio=args.min_duration_ratio,
                    max_gap_sec=args.max_gap_sec,
                )
                save_event_transaction(conn, event_row, station_rows)
                newly_done += 1

                progress.set_postfix(
                    raw=event_row["n_raw"],
                    valid=event_row["n_valid"],
                    status=event_row["raw_csv_status"],
                    refresh=False,
                )

                if args.export_every > 0 and newly_done % args.export_every == 0:
                    export_csv_outputs(conn, root, summary)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Completed events are safely checkpointed.")
        finally:
            progress.close()

        _, event_df = export_csv_outputs(conn, root, summary)

        print("\nEvents checked:", len(event_df))
        print("Eligible events:", int(event_df["eligible"].sum()))
        if not event_df.empty:
            print(event_df["n_valid"].describe())
            for n in [8, 10, 15, 20, 25, 30]:
                count = int((event_df["n_valid"] >= n).sum())
                print(f"n_valid >= {n}: {count}")

            print("\nraw_records.csv status counts:")
            print(event_df["raw_csv_status"].value_counts(dropna=False))

        print("\nOutputs:")
        print(root / "qc_download_stations.csv")
        print(root / "qc_download_events.csv")
        print(root / "eligible_events.csv")
        print(db_path)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
