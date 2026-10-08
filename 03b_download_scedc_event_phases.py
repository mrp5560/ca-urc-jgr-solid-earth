#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03b_download_scedc_event_phases.py

Robust Windows-friendly downloader for SCEDC event phase files.
It does not require s5cmd or an AWS account.

The script reads paper_events_m3_min30.csv and downloads:
  event_phases/YYYY/YYYY_DOY/<event_id>.phase

It supports:
  - concurrent downloads
  - retry and resume
  - atomic .part files
  - local path diagnostics
  - download summary CSV

Example:
D:\python3\python.exe 03b_download_scedc_event_phases.py ^
  --events data\scedc\paper_events_m3_min30.csv ^
  --phase-root data\scedc\event_phases ^
  --workers 24
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


URL_TEMPLATES = [
    "https://scedc-pds.s3.us-west-2.amazonaws.com/{key}",
    "https://s3.us-west-2.amazonaws.com/scedc-pds/{key}",
    "https://scedc-pds.s3.amazonaws.com/{key}",
]


def is_valid_phase_file(path: Path, min_bytes: int = 20) -> bool:
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    try:
        head = path.read_bytes()[:300].lstrip()
    except OSError:
        return False
    # Avoid treating an S3 XML error page or HTML page as a phase file.
    lowered = head.lower()
    if lowered.startswith(b"<?xml") or b"<error>" in lowered or b"<html" in lowered:
        return False
    return True


def build_item(row: Any, phase_root: Path) -> dict[str, Any]:
    event_id = str(row.event_id).strip()

    if hasattr(row, "year") and pd.notna(row.year):
        year = int(row.year)
    else:
        year = int(pd.to_datetime(row.origin_time, utc=True).year)

    if hasattr(row, "doy") and pd.notna(row.doy):
        doy = int(row.doy)
    else:
        doy = int(pd.to_datetime(row.origin_time, utc=True).dayofyear)

    key = f"event_phases/{year}/{year}_{doy:03d}/{event_id}.phase"
    output = phase_root / str(year) / f"{year}_{doy:03d}" / f"{event_id}.phase"

    return {
        "event_id": event_id,
        "year": year,
        "doy": doy,
        "key": key,
        "output": output,
    }


def download_one(
    item: dict[str, Any],
    retries: int,
    timeout: float,
    min_bytes: int,
) -> dict[str, Any]:
    output: Path = item["output"]
    output.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "event_id": item["event_id"],
        "year": item["year"],
        "doy": item["doy"],
        "s3_key": item["key"],
        "phase_path": str(output.resolve()),
        "status": "",
        "http_status": "",
        "bytes": 0,
        "url_used": "",
        "error": "",
    }

    if is_valid_phase_file(output, min_bytes=min_bytes):
        result["status"] = "reused"
        result["bytes"] = output.stat().st_size
        return result

    part = output.with_suffix(output.suffix + ".part")
    last_error = ""

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Causal-SeisField/1.0 SCEDC-phase-downloader"
    })

    for attempt in range(1, retries + 1):
        for template in URL_TEMPLATES:
            url = template.format(key=item["key"])
            result["url_used"] = url
            try:
                response = session.get(url, timeout=timeout)
                result["http_status"] = response.status_code

                if response.status_code == 200:
                    content = response.content
                    head = content[:300].lstrip().lower()
                    if (
                        len(content) < min_bytes
                        or head.startswith(b"<?xml")
                        or b"<error>" in head
                        or b"<html" in head
                    ):
                        last_error = (
                            f"invalid response body: bytes={len(content)}"
                        )
                        continue

                    part.write_bytes(content)
                    os.replace(part, output)

                    if is_valid_phase_file(output, min_bytes=min_bytes):
                        result["status"] = "downloaded"
                        result["bytes"] = output.stat().st_size
                        return result

                    last_error = "saved file failed validation"
                    try:
                        output.unlink(missing_ok=True)
                    except Exception:
                        pass

                elif response.status_code == 404:
                    last_error = "HTTP 404: object not found"
                else:
                    last_error = f"HTTP {response.status_code}"

            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            time.sleep(min(8.0, 1.5 * attempt))

    try:
        part.unlink(missing_ok=True)
    except Exception:
        pass

    result["status"] = "failed"
    result["error"] = last_error
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        default="data/scedc/paper_events_m3_min30.csv",
    )
    parser.add_argument(
        "--phase-root",
        default="data/scedc/event_phases",
    )
    parser.add_argument(
        "--summary",
        default="data/scedc/phase_download_summary.csv",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-bytes", type=int, default=20)
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Only inspect expected local paths; do not download.",
    )
    args = parser.parse_args()

    events_path = Path(args.events).resolve()
    phase_root = Path(args.phase_root).resolve()
    summary_path = Path(args.summary).resolve()

    if not events_path.exists():
        raise FileNotFoundError(f"Event file not found: {events_path}")

    events = pd.read_csv(events_path, dtype={"event_id": str})
    if "event_id" not in events.columns:
        raise ValueError("events CSV must contain event_id")
    if not {"year", "doy"}.issubset(events.columns) and "origin_time" not in events.columns:
        raise ValueError("events CSV needs year+doy or origin_time")

    items = [
        build_item(row, phase_root)
        for row in events.itertuples(index=False)
    ]

    existing = [
        item for item in items
        if is_valid_phase_file(item["output"], min_bytes=args.min_bytes)
    ]

    print("=== Path diagnosis ===")
    print(f"Current working directory : {Path.cwd()}")
    print(f"Events CSV                : {events_path}")
    print(f"Expected phase root       : {phase_root}")
    print(f"Events in manifest        : {len(items)}")
    print(f"Valid files already found : {len(existing)}")
    if items:
        print(f"First expected phase file : {items[0]['output'].resolve()}")
        print(f"First S3 key              : {items[0]['key']}")

    if args.diagnose_only:
        return

    phase_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download_one,
                item,
                args.retries,
                args.timeout,
                args.min_bytes,
            ): item
            for item in items
        }

        with tqdm(total=len(futures), desc="download phases") as bar:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "event_id": item["event_id"],
                        "year": item["year"],
                        "doy": item["doy"],
                        "s3_key": item["key"],
                        "phase_path": str(item["output"].resolve()),
                        "status": "failed",
                        "http_status": "",
                        "bytes": 0,
                        "url_used": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append(result)
                bar.update(1)

                # Periodic checkpoint.
                if len(results) % 200 == 0:
                    pd.DataFrame(results).to_csv(
                        summary_path.with_name(
                            summary_path.stem + "_partial.csv"
                        ),
                        index=False,
                    )

    result_df = pd.DataFrame(results).sort_values(
        ["year", "doy", "event_id"]
    )
    result_df.to_csv(summary_path, index=False)

    print("\n=== Phase download summary ===")
    print(result_df["status"].value_counts(dropna=False).to_string())
    print(f"Valid local files: {sum(is_valid_phase_file(x['output'], args.min_bytes) for x in items)}")
    print(f"Summary: {summary_path}")

    failed = result_df[result_df["status"] == "failed"]
    if len(failed):
        failed_path = summary_path.with_name("phase_download_failed.csv")
        failed.to_csv(failed_path, index=False)
        print(f"Failed list: {failed_path}")
        print("\nTop failure reasons:")
        print(failed["error"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
