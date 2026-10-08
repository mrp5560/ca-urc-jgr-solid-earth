#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, importlib.util, json, math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def load_module(path, name):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def local_xy_km(coords, lat0, lon0):
    x = (coords[:, 1] - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (coords[:, 0] - lat0) * 110.57
    return np.c_[x, y]


def detect_epicenter(row):
    lat_cols = ["event_latitude", "latitude", "lat", "event_lat", "hypocenter_latitude"]
    lon_cols = ["event_longitude", "longitude", "lon", "event_lon", "hypocenter_longitude"]
    lat = lon = None
    for c in lat_cols:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if np.isfinite(v):
                lat = float(v); break
    for c in lon_cols:
        if c in row.index:
            v = pd.to_numeric(row[c], errors="coerce")
            if np.isfinite(v):
                lon = float(v); break
    return lat, lon


def build_case_stats(df):
    rows = []
    for (eid, rep), g in df.groupby(["event_id", "repeat"], sort=False):
        tpga = g.true_log10_pga.to_numpy(float)
        tpgv = g.true_log10_pgv.to_numpy(float)
        bpga = g.base_log10_pga.to_numpy(float)
        bpgv = g.base_log10_pgv.to_numpy(float)
        ppga = g.power_gate_log10_pga.to_numpy(float)
        ppgv = g.power_gate_log10_pgv.to_numpy(float)
        tail_pga = g.is_tail_pga.astype(bool).to_numpy()
        tail_pgv = g.is_tail_pgv.astype(bool).to_numpy()
        base_all = 0.5 * (np.mean(abs(bpga-tpga)) + np.mean(abs(bpgv-tpgv)))
        power_all = 0.5 * (np.mean(abs(ppga-tpga)) + np.mean(abs(ppgv-tpgv)))
        btail, ptail = [], []
        if tail_pga.any():
            btail.extend(abs(bpga[tail_pga]-tpga[tail_pga])); ptail.extend(abs(ppga[tail_pga]-tpga[tail_pga]))
        if tail_pgv.any():
            btail.extend(abs(bpgv[tail_pgv]-tpgv[tail_pgv])); ptail.extend(abs(ppgv[tail_pgv]-tpgv[tail_pgv]))
        btail_mae = float(np.mean(btail)) if btail else np.nan
        ptail_mae = float(np.mean(ptail)) if ptail else np.nan
        rows.append({
            "event_id": str(eid), "repeat": int(rep),
            "magnitude": float(pd.to_numeric(g.magnitude.iloc[0], errors="coerce")) if "magnitude" in g else np.nan,
            "n_tail": int((tail_pga | tail_pgv).sum()),
            "base_overall": float(base_all), "power_overall": float(power_all),
            "base_tail": btail_mae, "power_tail": ptail_mae,
            "tail_gain": btail_mae-ptail_mae if np.isfinite(btail_mae) else np.nan,
            "input_station_ids": str(g.input_station_ids.iloc[0])
        })
    return pd.DataFrame(rows)


def choose_pairs(stats, event_a=None, repeat_a=None, event_b=None, repeat_b=None):
    def manual(eid, rep, hazardous):
        s = stats[stats.event_id.astype(str) == str(eid)].copy()
        if rep is not None:
            s = s[s.repeat == rep]
        if len(s) == 0:
            raise ValueError(f"Requested event/repeat not found: {eid}/{rep}")
        if rep is not None:
            return s.iloc[0]
        return s.sort_values("tail_gain" if hazardous else "base_overall", ascending=not hazardous).iloc[0]

    if event_a is not None:
        a = manual(event_a, repeat_a, False)
    else:
        s = stats[stats.n_tail <= 1].copy()
        if len(s) == 0: s = stats.copy()
        med = s.base_overall.median()
        s["d"] = abs(s.base_overall-med)
        a = s.sort_values(["d", "base_overall"]).iloc[0]

    if event_b is not None:
        b = manual(event_b, repeat_b, True)
    else:
        s = stats[stats.n_tail >= 2].copy()
        if len(s) == 0: s = stats[stats.n_tail >= 1].copy()
        if len(s) == 0: s = stats.copy()
        s["tail_gain_f"] = s.tail_gain.fillna(-1e9)
        b = s.sort_values(["tail_gain_f", "n_tail", "base_tail"], ascending=[False, False, False]).iloc[0]
    return a, b


def station_indices(station_ids, text):
    lookup = {str(s): i for i, s in enumerate(station_ids)}
    ids = [x.strip() for x in str(text).split("|") if x.strip()]
    missing = [s for s in ids if s not in lookup]
    if missing:
        raise ValueError(f"Input station IDs not found in H5: {missing}")
    return np.array([lookup[s] for s in ids], dtype=int)


def infer_all_heldout(pair, manifest_row, model, evmod, device, t0, pre, gamma, thresholds):
    # pair is a pandas Series. Always use pair["column"] access because
    # names such as "repeat" may collide with pandas Series methods.
    with h5py.File(str(manifest_row.h5_path), "r") as h5:
        acc = np.asarray(h5["acceleration"][:], np.float32)
        vel = np.asarray(h5["velocity"][:], np.float32)
        coords = np.asarray(h5["station_coords"][:], np.float32)
        poff = np.asarray(h5["p_offset_sec"][:], np.float32)
        station_ids = h5["station_id"].asstr()[:]
        fs = float(h5.attrs["sampling_rate_hz"])
        pre_first_p = float(h5.attrs["pre_first_p_sec"])
        tzero = int(h5.attrs["time_zero_index"])

    inp = station_indices(station_ids, pair["input_station_ids"])
    tgt = np.setdiff1d(np.arange(len(station_ids)), inp)
    input_start = max(0, int(round((pre_first_p-pre)*fs)))
    snap = min(tzero + int(round(t0*fs)), acc.shape[-1]-1)

    iw, inf, tf = evmod.prepare_model_inputs(
        acceleration=acc, coordinates=coords, p_offset=poff,
        input_indices=inp, target_indices=tgt,
        input_start_index=input_start, snapshot_index=snap, t0_sec=t0
    )

    true_pga = np.log10(np.maximum(evmod.horizontal_peak(acc[tgt, :, snap:]), 1e-10))
    true_pgv = np.log10(np.maximum(evmod.horizontal_peak(vel[tgt, :, snap:]), 1e-12))
    truth = np.c_[true_pga, true_pgv]

    with torch.inference_mode():
        out = model(
            torch.from_numpy(iw)[None].to(device),
            torch.from_numpy(inf)[None].to(device),
            torch.from_numpy(tf)[None].to(device),
        )
        base = out["base"][0].cpu().numpy()
        prob = out["probability"][0].cpu().numpy()
        residual = out["residual"][0].cpu().numpy()
        power = base + (prob ** gamma) * residual

    lat0, lon0 = detect_epicenter(manifest_row)
    if lat0 is None or lon0 is None:
        lat0, lon0 = float(coords[:,0].mean()), float(coords[:,1].mean())
        epicenter = None
    else:
        epicenter = np.array([0.0, 0.0])
    xy = local_xy_km(coords, lat0, lon0)
    tail = truth >= thresholds[None, :]

    def metrics(pred):
        e = pred-truth; ae = abs(e); outm = {}
        for j, q in enumerate(["pga", "pgv"]):
            outm[f"{q}_mae"] = float(ae[:,j].mean())
            m = tail[:,j]
            outm[f"{q}_tail_n"] = int(m.sum())
            outm[f"{q}_tail_mae"] = float(ae[m,j].mean()) if m.any() else np.nan
        return outm

    mag = pd.to_numeric(manifest_row.get("magnitude", np.nan), errors="coerce")
    return {
        "event_id": str(pair["event_id"]), "repeat": int(pair["repeat"]),
        "magnitude": float(mag) if np.isfinite(mag) else np.nan,
        "xy_target": xy[tgt], "xy_input": xy[inp], "epicenter": epicenter,
        "truth": truth, "base": base, "power": power, "prob": prob, "tail": tail,
        "base_metrics": metrics(base), "power_metrics": metrics(power),
        "station_ids_target": [str(station_ids[i]) for i in tgt]
    }


def shifted_log(values, q):
    return values - math.log10(9.80665) if q == 0 else values + 2.0


def limits(cases, q):
    vals = np.concatenate([shifted_log(c[k][:,q], q) for c in cases for k in ["truth","base","power"]])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2, 98])
    if hi-lo < 0.2:
        m = (hi+lo)/2; lo, hi = m-0.1, m+0.1
    return float(lo), float(hi)


def tick_label(x):
    v = 10**x
    if v >= 100: return f"{v:.0f}"
    if v >= 10: return f"{v:.1f}"
    if v >= 1: return f"{v:.2f}"
    if v >= 0.1: return f"{v:.2f}"
    if v >= 0.01: return f"{v:.3f}"
    return f"{v:.1e}"


def draw(ax, case, vals, q, lo, hi, cmap, annotation=None):
    xy = case["xy_target"]; x, y = xy[:,0], xy[:,1]
    z = shifted_log(vals, q)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[good], y[good], z[good]
    levels = np.linspace(lo, hi, 80)
    try:
        tri = mtri.Triangulation(x, y)
        ax.tricontourf(tri, np.clip(z, lo, hi), levels=levels, cmap=cmap, extend="both")
    except Exception:
        ax.scatter(x, y, c=np.clip(z, lo, hi), cmap=cmap, vmin=lo, vmax=hi, s=24)
    ax.scatter(x, y, s=5, facecolors="none", edgecolors="k", linewidths=.3, alpha=.65, zorder=5)
    ii = case["xy_input"]
    ax.scatter(ii[:,0], ii[:,1], marker="^", s=42, facecolors="white", edgecolors="k", linewidths=.9, zorder=8)
    if case["epicenter"] is not None:
        ax.scatter([0],[0], marker="*", s=95, facecolors="white", edgecolors="k", linewidths=.9, zorder=9)
    if annotation:
        ax.text(.03,.04,annotation,transform=ax.transAxes,fontsize=8.2,ha="left",va="bottom",
                bbox=dict(boxstyle="round,pad=.22",fc="white",ec="0.6",alpha=.88))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8, direction="out", length=3)
    for sp in ax.spines.values(): sp.set_linewidth(.8)


def ann(metrics, q):
    s = f"MAE={metrics[q+'_mae']:.3f}"
    if metrics[q+"_tail_n"] > 0 and np.isfinite(metrics[q+"_tail_mae"]):
        s += f"\nTail MAE={metrics[q+'_tail_mae']:.3f}"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default='runs/locked_power_gated_test_gamma3/locked_test_predictions.csv')
    p.add_argument("--manifest", default='data/scedc/model_manifests/scenario_t0_5s_k5.csv')
    p.add_argument("--checkpoint", default='runs/tail_risk_gated_dual_head_t0_5s_k5/best_overall.pt')
    p.add_argument("--threshold-json", default='runs/tail_gated_compromise_t0_5s_k5/tail_thresholds_q0.90_t0_5s.json')
    p.add_argument("--training-script", default="18_train_tail_risk_gated_dual_head.py")
    p.add_argument("--evaluation-script", default="20_evaluate_locked_power_gated_test.py")
    p.add_argument("--event-a", default=None); p.add_argument("--repeat-a", type=int, default=None)
    p.add_argument("--event-b", default=None); p.add_argument("--repeat-b", type=int, default=None)
    p.add_argument("--t0-sec", type=int, default=5); p.add_argument("--input-pre-sec", type=float, default=2.0)
    p.add_argument("--gate-power", type=float, default=3.0)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--cmap", default="turbo"); p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out-dir", default='figures/wavecast_style_cases')
    a = p.parse_args()

    outdir = Path(a.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device))

    pred = pd.read_csv(a.predictions, dtype={"event_id":str})
    stats = build_case_stats(pred); stats.to_csv(outdir/"event_repeat_case_statistics.csv", index=False)
    pair_a, pair_b = choose_pairs(stats, a.event_a, a.repeat_a, a.event_b, a.repeat_b)

    man = pd.read_csv(a.manifest, dtype={"event_id":str}).drop_duplicates("event_id").set_index("event_id")
    th = json.loads(Path(a.threshold_json).read_text(encoding="utf-8"))
    thresholds = np.array([th["log10_pga_threshold"], th["log10_pgv_threshold"]], float)

    evmod = load_module(a.evaluation_script, "eval20")
    trainmod = evmod.load_python_module(a.training_script)
    model, ckpt, _ = evmod.load_locked_model(trainmod, a.checkpoint, th, device)

    case_a = infer_all_heldout(pair_a, man.loc[str(pair_a["event_id"])], model, evmod, device, a.t0_sec, a.input_pre_sec, a.gate_power, thresholds)
    case_b = infer_all_heldout(pair_b, man.loc[str(pair_b["event_id"])], model, evmod, device, a.t0_sec, a.input_pre_sec, a.gate_power, thresholds)
    cases = [case_a, case_a, case_b, case_b]

    pga_lo,pga_hi = limits([case_a,case_b],0); pgv_lo,pgv_hi = limits([case_a,case_b],1)
    fig, axes = plt.subplots(3,4, figsize=(13.4,10.0))
    rows = [("truth","Ground truth"),("base","Mean base"),("power","Power gate")]
    qs = [0,1,0,1]

    for r,(key,rowlab) in enumerate(rows):
        for c in range(4):
            case = cases[c]; q = qs[c]; qname = "pga" if q==0 else "pgv"
            lo,hi = (pga_lo,pga_hi) if q==0 else (pgv_lo,pgv_hi)
            annotation = None if key=="truth" else ann(case[key+"_metrics"] if key in ["base","power"] else {}, qname)
            draw(axes[r,c], case, case[key][:,q], q, lo, hi, a.cmap, annotation)
            if r==0: axes[r,c].set_title("PGA" if q==0 else "PGV", fontsize=12, fontweight="bold")
            if r<2: axes[r,c].set_xticklabels([])
            else: axes[r,c].set_xlabel("X (km)", fontsize=9.5)
            if c in [0,2]: axes[r,c].set_ylabel("Y (km)", fontsize=9.5)
            else: axes[r,c].set_yticklabels([])

    for r,(_,lab) in enumerate(rows):
        pos = axes[r,0].get_position(); fig.text(.022,(pos.y0+pos.y1)/2,lab,rotation=90,va="center",ha="center",fontsize=11.5,fontweight="bold")

    ma = f"M{case_a['magnitude']:.1f}" if np.isfinite(case_a['magnitude']) else ""
    mb = f"M{case_b['magnitude']:.1f}" if np.isfinite(case_b['magnitude']) else ""
    fig.text(.285,.965,f"(a) Representative event  {case_a['event_id']}  {ma}",ha="center",va="top",fontsize=12.5,fontweight="bold")
    fig.text(.745,.965,f"(b) Hazardous-tail event  {case_b['event_id']}  {mb}",ha="center",va="top",fontsize=12.5,fontweight="bold")

    for lo,hi,x0,label in [(pga_lo,pga_hi,.16,"Future PGA (g)"),(pgv_lo,pgv_hi,.58,"Future PGV (cm/s)")]:
        cax = fig.add_axes([x0,.055,.28,.020]); sm = ScalarMappable(norm=Normalize(lo,hi),cmap=a.cmap); sm.set_array([])
        cb = fig.colorbar(sm,cax=cax,orientation="horizontal",extend="both")
        ticks=np.linspace(lo,hi,5); cb.set_ticks(ticks); cb.set_ticklabels([tick_label(x) for x in ticks]); cb.set_label(label,fontsize=9.5)

    fig.text(.5,.015,"Filled surfaces are triangulated from actual held-out station queries for visualization only; quantitative metrics are computed at station locations.",ha="center",va="bottom",fontsize=8.3)
    fig.subplots_adjust(left=.075,right=.985,bottom=.115,top=.91,wspace=.055,hspace=.055)

    png = outdir/"Fig_spatial_ground_truth_base_power.png"; pdf = outdir/"Fig_spatial_ground_truth_base_power.pdf"
    fig.savefig(png,dpi=a.dpi,bbox_inches="tight"); fig.savefig(pdf,bbox_inches="tight"); plt.close(fig)

    summary = pd.DataFrame([
        {"case":"representative","event_id":case_a["event_id"],"repeat":case_a["repeat"],"magnitude":case_a["magnitude"],**{"base_"+k:v for k,v in case_a["base_metrics"].items()},**{"power_"+k:v for k,v in case_a["power_metrics"].items()}},
        {"case":"hazardous","event_id":case_b["event_id"],"repeat":case_b["repeat"],"magnitude":case_b["magnitude"],**{"base_"+k:v for k,v in case_b["base_metrics"].items()},**{"power_"+k:v for k,v in case_b["power_metrics"].items()}},
    ])
    summary.to_csv(outdir/"selected_case_summary.csv",index=False)

    print("=== Selected cases ===")
    print(summary.to_string(index=False))
    print(f"\nPNG: {png.resolve()}\nPDF: {pdf.resolve()}")

if __name__ == "__main__":
    main()
