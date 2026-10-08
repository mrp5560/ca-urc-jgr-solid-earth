#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

EVENT_CSV = "data/scedc/events_m3_2010_2024.csv"
OUT_DIR = Path("data/scedc/aws_raw")
MANIFEST_DIR = Path("data/scedc/manifests")

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "event_waveforms").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "event_phases").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "FDSNstationXML" / "CI").mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(EVENT_CSV, dtype={"event_id": str})

cmds = []

for _, r in df.iterrows():
    eid = str(r["event_id"])
    t = pd.to_datetime(r["origin_time"], utc=True)

    year = int(t.year)
    doy = int(t.dayofyear)

    wf_src = f"s3://scedc-pds/event_waveforms/{year}/{year}_{doy:03d}/{eid}.ms"
    ph_src = f"s3://scedc-pds/event_phases/{year}/{year}_{doy:03d}/{eid}.phase"

    wf_dst = OUT_DIR / "event_waveforms" / f"{eid}.ms"
    ph_dst = OUT_DIR / "event_phases" / f"{eid}.phase"

    cmds.append(f"cp {wf_src} {wf_dst}\n")
    cmds.append(f"cp {ph_src} {ph_dst}\n")

manifest = MANIFEST_DIR / "scedc_events.s5cmd"
manifest.write_text("".join(cmds), encoding="utf-8")

print(f"events = {len(df)}")
print(f"download commands = {len(cmds)}")
print(f"wrote manifest -> {manifest}")