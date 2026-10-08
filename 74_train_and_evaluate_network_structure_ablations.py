#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
74_train_and_evaluate_network_structure_ablations.py

Focused structure ablation for the final Causal-SeisField / CA-URC system.
Only two new models are trained; S3/S4 reuse existing locked predictions.

S1: w/o station self-attention
    StationEncoder -> target-conditioned cross-attention -> decoder

S2: w/o target cross-attention
    StationEncoder -> station self-attention -> global mean readout
    + explicit target query -> decoder

S3: full Cross-Attention Base (existing locked prediction)
S4: CA-URC = S3 + underprediction-risk correction (existing locked prediction)

Primary questions:
    S3 vs S1: contribution of station self-attention.
    S3 vs S2: contribution of target-conditioned cross-attention.
    S4 vs S3: contribution of CA-URC risk correction.

Fair protocol:
    - same grouped train/validation/test split
    - same T0/K/Q and input window
    - same deterministic train/validation draws as final strong-baseline suite
    - same locked test draws as script-20 protocol
    - validation-only checkpoint selection
    - test instantiated only after S1/S2 are locked
    - canonical aggregation: targets -> repeats -> events
    - hierarchical sequence->event bootstrap as primary statistics
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

LOG10_FACTOR_2 = math.log10(2.0)
SEVERE_UNDER_THRESHOLD = -0.5


def load_module(path: str | Path, name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path.resolve()}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}
    unknown = set(normalized.unique()).difference(allowed)
    if unknown:
        raise ValueError(f"Cannot parse boolean values: {sorted(unknown)[:20]}")
    return normalized.isin({"true", "1", "yes", "y", "t"}).to_numpy(dtype=bool)


# -----------------------------------------------------------------------------
# Structural variants
# -----------------------------------------------------------------------------

class WithoutStationSelfAttention(nn.Module):
    """S1: remove the full station self-attention residual block."""

    def __init__(self, baseline_module, hidden_dim: int, heads: int):
        super().__init__()
        self.encoder = baseline_module.StationEncoder(hidden_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_waveforms, input_features, target_features):
        station = self.encoder(input_waveforms, input_features)
        query = self.query_encoder(target_features)
        context, _ = self.cross_attention(
            query, station, station, need_weights=False
        )
        context = self.cross_norm(query + context)
        return self.decoder(torch.cat([context, query], dim=-1))


class WithoutTargetCrossAttention(nn.Module):
    """
    S2: retain station self-attention and explicit target query, but replace
    target-conditioned station retrieval with one global station mean.
    """

    def __init__(self, baseline_module, hidden_dim: int, heads: int):
        super().__init__()
        self.encoder = baseline_module.StationEncoder(hidden_dim)
        self.station_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.station_norm = nn.LayerNorm(hidden_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_waveforms, input_features, target_features):
        station = self.encoder(input_waveforms, input_features)
        attended, _ = self.station_attention(
            station, station, station, need_weights=False
        )
        station = self.station_norm(station + attended)
        context = station.mean(dim=1)[:, None, :].expand(
            -1, target_features.shape[1], -1
        )
        query = self.query_encoder(target_features)
        return self.decoder(torch.cat([context, query], dim=-1))


def make_structure_model(variant, baseline_module, hidden_dim, heads):
    if variant == "S1_no_station_self_attention":
        return WithoutStationSelfAttention(baseline_module, hidden_dim, heads)
    if variant == "S2_no_target_cross_attention":
        return WithoutTargetCrossAttention(baseline_module, hidden_dim, heads)
    raise ValueError(variant)


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def make_train_loader(dataset, args, device):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )


def train_structure_variant(
    variant,
    baseline_module,
    args,
    train_dataset,
    validation_loader,
    device,
    out_dir,
):
    # Identical random initialization stream / shuffle stream per variant.
    set_seed(args.seed)
    model = make_structure_model(
        variant, baseline_module, args.hidden_dim, args.attention_heads
    ).to(device)
    train_loader = make_train_loader(train_dataset, args, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.minimum_learning_rate,
    )

    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_dir / "best_model.pt"

    best_score = np.inf
    best_epoch = 0
    stale = 0
    history = []
    start = time.time()

    print("\n" + "=" * 88)
    print(f"Training structure ablation: {variant}")
    print("=" * 88)
    print(f"Parameters                : {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss = baseline_module.train_epoch(
            model, train_loader, optimizer, device
        )
        val_frame = baseline_module.predict_model(
            model, validation_loader, device
        )
        val_pga = baseline_module.event_balanced_mae(
            val_frame, "pga", "pred_log10_pga"
        )
        val_pgv = baseline_module.event_balanced_mae(
            val_frame, "pgv", "pred_log10_pgv"
        )
        score = 0.5 * (val_pga + val_pgv)
        scheduler.step(score)
        lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_mae_pga": float(val_pga),
                "validation_mae_pgv": float(val_pgv),
                "validation_score": float(score),
                "learning_rate": lr,
            }
        )
        pd.DataFrame(history).to_csv(
            variant_dir / "history.csv", index=False
        )

        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | "
            f"val PGA={val_pga:.4f} | PGV={val_pgv:.4f} | "
            f"score={score:.4f} | lr={lr:.2e}"
        )

        if score < best_score - args.minimum_delta:
            best_score = float(score)
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "epoch": epoch,
                    "validation_score": float(score),
                    "validation_mae_pga": float(val_pga),
                    "validation_mae_pgv": float(val_pgv),
                    "parameter_count": int(
                        sum(p.numel() for p in model.parameters())
                    ),
                    "test_accessed": False,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            stale += 1

        if stale >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}"
            )
            break

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    summary = {
        "variant": variant,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_score": float(checkpoint["validation_score"]),
        "validation_mae_pga": float(checkpoint["validation_mae_pga"]),
        "validation_mae_pgv": float(checkpoint["validation_mae_pgv"]),
        "parameter_count": int(checkpoint["parameter_count"]),
        "training_seconds": float(time.time() - start),
        "test_accessed_during_selection": False,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (variant_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Selected {variant}: epoch={summary['best_epoch']}, "
        f"val PGA/PGV={summary['validation_mae_pga']:.4f}/"
        f"{summary['validation_mae_pgv']:.4f}"
    )
    return model, summary


# -----------------------------------------------------------------------------
# Locked references / pairing
# -----------------------------------------------------------------------------

def prepare_reference_frame(strong_path: Path, ablation_path: Path):
    strong = pd.read_csv(strong_path, dtype={"event_id": str})
    ablation = pd.read_csv(ablation_path, dtype={"event_id": str})
    keys = ["event_id", "repeat", "target_slot", "target_station_index"]

    required_strong = set(keys + [
        "true_log10_pga", "true_log10_pgv",
        "cross_attention_log10_pga", "cross_attention_log10_pgv",
    ])
    required_ablation = set(keys + [
        "true_log10_pga", "true_log10_pgv",
        "is_tail_pga", "is_tail_pgv",
        "A4_under_only_log10_pga", "A4_under_only_log10_pgv",
    ])
    if required_strong.difference(strong.columns):
        raise ValueError(
            "Strong reference missing columns: "
            f"{sorted(required_strong.difference(strong.columns))}"
        )
    if required_ablation.difference(ablation.columns):
        raise ValueError(
            "Ablation reference missing columns: "
            f"{sorted(required_ablation.difference(ablation.columns))}"
        )
    if strong.duplicated(keys).any() or ablation.duplicated(keys).any():
        raise RuntimeError("Existing locked references contain duplicate keys.")

    ref = strong[keys + [
        "true_log10_pga", "true_log10_pgv",
        "cross_attention_log10_pga", "cross_attention_log10_pgv",
    ]].merge(
        ablation[keys + [
            "true_log10_pga", "true_log10_pgv",
            "is_tail_pga", "is_tail_pgv",
            "A4_under_only_log10_pga", "A4_under_only_log10_pgv",
        ]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_strong", "_ablation"),
    )

    audit = {
        "strong_rows": int(len(strong)),
        "ablation_rows": int(len(ablation)),
        "reference_rows": int(len(ref)),
        "exact_strong_ablation_pairing": bool(
            len(ref) == len(strong) == len(ablation)
        ),
    }
    if not audit["exact_strong_ablation_pairing"]:
        raise RuntimeError(f"S3/S4 reference pairing failed: {audit}")

    for q in ("pga", "pgv"):
        diff = np.abs(
            ref[f"true_log10_{q}_strong"].to_numpy(float)
            - ref[f"true_log10_{q}_ablation"].to_numpy(float)
        )
        audit[f"max_abs_truth_diff_existing_{q}"] = float(np.max(diff))
        if audit[f"max_abs_truth_diff_existing_{q}"] > 1e-5:
            raise RuntimeError(f"Existing S3/S4 truth mismatch for {q}.")

    ref = ref.rename(
        columns={
            "true_log10_pga_strong": "true_log10_pga",
            "true_log10_pgv_strong": "true_log10_pgv",
            "cross_attention_log10_pga": "S3_cross_attention_log10_pga",
            "cross_attention_log10_pgv": "S3_cross_attention_log10_pgv",
            "A4_under_only_log10_pga": "S4_ca_urc_log10_pga",
            "A4_under_only_log10_pgv": "S4_ca_urc_log10_pgv",
        }
    ).drop(
        columns=["true_log10_pga_ablation", "true_log10_pgv_ablation"]
    )
    return ref, audit


def merge_new_prediction(reference, prediction, variant):
    keys = ["event_id", "repeat", "target_slot", "target_station_index"]
    required = set(keys + [
        "true_log10_pga", "true_log10_pgv",
        "pred_log10_pga", "pred_log10_pgv",
    ])
    missing = required.difference(prediction.columns)
    if missing:
        raise ValueError(f"{variant} missing test columns: {sorted(missing)}")
    if prediction.duplicated(keys).any():
        raise RuntimeError(f"{variant} has duplicate locked keys.")

    compact = prediction[keys + [
        "true_log10_pga", "true_log10_pgv",
        "pred_log10_pga", "pred_log10_pgv",
    ]].rename(
        columns={
            "true_log10_pga": f"{variant}_true_log10_pga",
            "true_log10_pgv": f"{variant}_true_log10_pgv",
            "pred_log10_pga": f"{variant}_log10_pga",
            "pred_log10_pgv": f"{variant}_log10_pgv",
        }
    )
    merged = reference.merge(
        compact, on=keys, how="inner", validate="one_to_one"
    )
    audit = {
        "variant": variant,
        "reference_rows_before_merge": int(len(reference)),
        "prediction_rows": int(len(prediction)),
        "paired_rows": int(len(merged)),
        "exact_pairing": bool(
            len(merged) == len(reference) == len(prediction)
        ),
    }
    if not audit["exact_pairing"]:
        raise RuntimeError(f"Locked pairing failed for {variant}: {audit}")

    for q in ("pga", "pgv"):
        diff = np.abs(
            merged[f"true_log10_{q}"].to_numpy(float)
            - merged[f"{variant}_true_log10_{q}"].to_numpy(float)
        )
        audit[f"max_abs_truth_diff_{q}"] = float(np.max(diff))
        if audit[f"max_abs_truth_diff_{q}"] > 1e-5:
            raise RuntimeError(f"Truth audit failed for {variant}/{q}.")
        merged = merged.drop(columns=[f"{variant}_true_log10_{q}"])
    return merged, audit


# -----------------------------------------------------------------------------
# Canonical metrics
# -----------------------------------------------------------------------------

def element_metric(truth, prediction, metric):
    residual = prediction - truth
    absolute = np.abs(residual)
    if metric == "mae":
        return absolute
    if metric == "bias":
        return residual
    if metric == "factor2":
        return (absolute <= LOG10_FACTOR_2).astype(float)
    if metric == "under05":
        return (residual <= SEVERE_UNDER_THRESHOLD).astype(float)
    raise ValueError(metric)


def canonical_event_value(frame, quantity, prediction_column, metric):
    values = element_metric(
        frame[f"true_log10_{quantity}"].to_numpy(float),
        frame[prediction_column].to_numpy(float),
        metric,
    )
    work = frame[["event_id", "repeat"]].copy()
    work["value"] = values
    event_repeat = work.groupby(
        ["event_id", "repeat"], sort=False, as_index=False
    )["value"].mean()
    event = event_repeat.groupby("event_id", sort=False)["value"].mean()
    return float(event.mean()), int(len(event)), int(len(frame))


def build_metric_table(frame):
    methods = {
        "S1_no_station_self_attention": {
            "pga": "S1_no_station_self_attention_log10_pga",
            "pgv": "S1_no_station_self_attention_log10_pgv",
        },
        "S2_no_target_cross_attention": {
            "pga": "S2_no_target_cross_attention_log10_pga",
            "pgv": "S2_no_target_cross_attention_log10_pgv",
        },
        "S3_cross_attention_base": {
            "pga": "S3_cross_attention_log10_pga",
            "pgv": "S3_cross_attention_log10_pgv",
        },
        "S4_ca_urc": {
            "pga": "S4_ca_urc_log10_pga",
            "pgv": "S4_ca_urc_log10_pgv",
        },
    }
    rows = []
    for method, cols in methods.items():
        for q in ("pga", "pgv"):
            populations = {
                "overall": np.ones(len(frame), dtype=bool),
                "high_motion_tail": as_bool_array(frame[f"is_tail_{q}"]),
            }
            for population, mask in populations.items():
                subset = frame.loc[mask].copy()
                for metric in ("mae", "bias", "factor2", "under05"):
                    value, n_events, n_rows = canonical_event_value(
                        subset, q, cols[q], metric
                    )
                    rows.append(
                        {
                            "model": method,
                            "quantity": q,
                            "population": population,
                            "metric": metric,
                            "value": value,
                            "n_events": n_events,
                            "n_target_rows": n_rows,
                        }
                    )
    return pd.DataFrame(rows)


def compact_table(metrics, training_summaries, baseline_module, args):
    summary_map = {x["variant"]: x for x in training_summaries}
    s3_params = int(
        sum(
            p.numel()
            for p in baseline_module.make_model(
                "cross_attention",
                args.hidden_dim,
                args.attention_heads,
                50.0,
            ).parameters()
        )
    )

    def get(model, q, population, metric):
        row = metrics.loc[
            metrics["model"].eq(model)
            & metrics["quantity"].eq(q)
            & metrics["population"].eq(population)
            & metrics["metric"].eq(metric)
        ]
        return float(row.iloc[0]["value"])

    specs = [
        (
            "S1_no_station_self_attention",
            "S1 w/o Station Self-Attention",
            False, True, False,
        ),
        (
            "S2_no_target_cross_attention",
            "S2 w/o Target Cross-Attention",
            True, False, False,
        ),
        (
            "S3_cross_attention_base",
            "S3 Cross-Attention Base",
            True, True, False,
        ),
        (
            "S4_ca_urc",
            "S4 CA-URC",
            True, True, True,
        ),
    ]
    rows = []
    for model, label, self_attn, cross_attn, risk in specs:
        if model in summary_map:
            epoch = summary_map[model]["best_epoch"]
            params = summary_map[model]["parameter_count"]
            status = "newly_trained"
        elif model == "S3_cross_attention_base":
            epoch = np.nan
            params = s3_params
            status = "existing_locked"
        else:
            epoch = 3  # grouped A4 / CA-URC lock
            params = np.nan
            status = "existing_locked"

        rows.append(
            {
                "model": model,
                "label": label,
                "station_self_attention": self_attn,
                "target_cross_attention": cross_attn,
                "explicit_target_query": True,
                "underprediction_risk_correction": risk,
                "status": status,
                "selected_epoch": epoch,
                "parameter_count": params,
                "pga_overall_mae": get(model, "pga", "overall", "mae"),
                "pga_tail_mae": get(model, "pga", "high_motion_tail", "mae"),
                "pga_tail_under05": get(model, "pga", "high_motion_tail", "under05"),
                "pga_tail_factor2": get(model, "pga", "high_motion_tail", "factor2"),
                "pgv_overall_mae": get(model, "pgv", "overall", "mae"),
                "pgv_tail_mae": get(model, "pgv", "high_motion_tail", "mae"),
                "pgv_tail_under05": get(model, "pgv", "high_motion_tail", "under05"),
                "pgv_tail_factor2": get(model, "pgv", "high_motion_tail", "factor2"),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-module",
        default="45_phase2_strong_baseline_suite.py",
    )
    parser.add_argument(
        "--protocol-module",
        default="59_final_strong_baselines_reuse_locked.py",
    )
    parser.add_argument(
        "--bootstrap-module",
        default="62_sequence_aware_paired_bootstrap_cadrg.py",
    )
    parser.add_argument(
        "--manifest",
        default="data/model_manifests/scenario_t0_5s_k5_linux.csv",
    )
    parser.add_argument(
        "--h5-root",
        default="data/processed_full_v4/events",
    )
    parser.add_argument("--split-column", default="split_grouped")
    parser.add_argument("--train-label", default="train")
    parser.add_argument("--validation-label", default="validation")
    parser.add_argument("--test-label", default="test")
    parser.add_argument("--sequence-column", default="sequence_group")
    parser.add_argument(
        "--strong-reference",
        default=(
            "runs/final_strong_baselines_reuse_locked/"
            "final_strong_baseline_reused_locked_predictions.csv"
        ),
    )
    parser.add_argument(
        "--caurc-reference",
        default=(
            "runs/cadrg_gate_ablation_A3_A5/"
            "locked_test_ablation_predictions.csv"
        ),
    )

    parser.add_argument("--t0-sec", type=int, default=5)
    parser.add_argument("--input-stations", type=int, default=5)
    parser.add_argument("--target-stations", type=int, default=10)
    parser.add_argument("--input-pre-sec", type=float, default=2.0)
    parser.add_argument("--validation-repeats", type=int, default=3)
    parser.add_argument("--test-repeats", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--out-dir", default="runs/network_structure_ablation_S1_S4"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device(args.device)

    if args.bootstrap_repetitions < 1000:
        raise ValueError("--bootstrap-repetitions should be >= 1000.")

    baseline_module = load_module(
        args.baseline_module, "structure_baseline_module"
    )
    protocol_module = load_module(
        args.protocol_module, "structure_protocol_module"
    )
    bootstrap_module = load_module(
        args.bootstrap_module, "structure_bootstrap_module"
    )

    # Critical fair-draw patch: train/validation -> script 40 convention;
    # test -> locked script 20 convention.
    protocol_module.patch_baseline_seed_protocol(baseline_module)
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_dataset = {
        "manifest": args.manifest,
        "h5_root": args.h5_root,
        "split_column": args.split_column,
        "t0_sec": args.t0_sec,
        "input_stations": args.input_stations,
        "target_stations": args.target_stations,
        "input_pre_sec": args.input_pre_sec,
        "seed": args.seed,
    }

    # TRAIN + VALIDATION only.
    train_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.train_label,
        training=True,
        repeats=1,
        **common_dataset,
    )
    validation_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.validation_label,
        training=False,
        repeats=args.validation_repeats,
        **common_dataset,
    )
    eval_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": (args.num_workers > 0),
    }
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **eval_loader_kwargs
    )

    print("=== Focused Network-Structure Ablation ===")
    print(f"Device                    : {device}")
    print("New models                : S1, S2")
    print("Reused locked models      : S3 Cross-Attention, S4 CA-URC")
    print(
        "Train / Validation events : "
        f"{len(train_dataset.frame)}/{len(validation_dataset.frame)}"
    )
    print(f"Validation repeats        : {args.validation_repeats}")
    print("TEST during training      : NOT ACCESSED")

    variants = [
        "S1_no_station_self_attention",
        "S2_no_target_cross_attention",
    ]
    trained_models = {}
    training_summaries = []

    for variant in variants:
        model, summary = train_structure_variant(
            variant,
            baseline_module,
            args,
            train_dataset,
            validation_loader,
            device,
            out_dir,
        )
        trained_models[variant] = model
        training_summaries.append(summary)

    pd.DataFrame(training_summaries).to_csv(
        out_dir / "training_selection_summary.csv", index=False
    )

    print("\n=== ALL S1/S2 MODEL SELECTION FINISHED ===")
    print("Locked test has not been used for checkpoint selection.")

    # Only now read existing locked S3/S4 predictions.
    reference, existing_audit = prepare_reference_frame(
        Path(args.strong_reference),
        Path(args.caurc_reference),
    )

    # Only now instantiate the locked TEST split.
    test_dataset = baseline_module.StrongBaselineDataset(
        split_name=args.test_label,
        training=False,
        repeats=args.test_repeats,
        **common_dataset,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, **eval_loader_kwargs
    )

    print("\n=== LOCKED STRUCTURE-ABLATION TEST ===")
    print(f"Test events               : {len(test_dataset.frame)}")
    print(f"Test repeats              : {args.test_repeats}")
    print(f"Targets/repeat            : {args.target_stations}")

    current = reference.copy()
    pairing_audits = [
        {"variant": "S3_vs_S4_existing", **existing_audit}
    ]

    for variant in variants:
        prediction = baseline_module.predict_model(
            trained_models[variant], test_loader, device
        )
        prediction.to_csv(
            out_dir / variant / "locked_test_predictions.csv",
            index=False,
        )
        current, audit = merge_new_prediction(
            current, prediction, variant
        )
        pairing_audits.append(audit)
        print(
            f"{variant} paired rows : {audit['paired_rows']:,} | "
            f"truth max diff PGA/PGV="
            f"{audit['max_abs_truth_diff_pga']:.3e}/"
            f"{audit['max_abs_truth_diff_pgv']:.3e}"
        )

    expected_rows = (
        len(test_dataset.frame) * args.test_repeats * args.target_stations
    )
    if len(current) != expected_rows:
        raise RuntimeError(
            "Locked row-count mismatch: "
            f"{len(current):,} vs expected {expected_rows:,}"
        )

    prediction_path = out_dir / "locked_structure_ablation_predictions.csv"
    current.to_csv(prediction_path, index=False)

    (out_dir / "pairing_audit.json").write_text(
        json.dumps(
            {
                "expected_rows": int(expected_rows),
                "final_rows": int(len(current)),
                "exact_locked_row_count": bool(len(current) == expected_rows),
                "audits": pairing_audits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = build_metric_table(current)
    metrics_path = out_dir / "network_structure_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    compact = compact_table(
        metrics, training_summaries, baseline_module, args
    )
    compact_path = out_dir / "paper_network_structure_table.csv"
    compact.to_csv(compact_path, index=False)

    # Sequence-aware paired bootstrap, reusing the already-tested script 62.
    bootstrap_module.METHOD_COLUMNS = {
        "s1": {
            "pga": "S1_no_station_self_attention_log10_pga",
            "pgv": "S1_no_station_self_attention_log10_pgv",
            "source": "unified",
        },
        "s2": {
            "pga": "S2_no_target_cross_attention_log10_pga",
            "pgv": "S2_no_target_cross_attention_log10_pgv",
            "source": "unified",
        },
        "s3": {
            "pga": "S3_cross_attention_log10_pga",
            "pgv": "S3_cross_attention_log10_pgv",
            "source": "unified",
        },
        "s4": {
            "pga": "S4_ca_urc_log10_pga",
            "pgv": "S4_ca_urc_log10_pgv",
            "source": "unified",
        },
    }

    current_seq, event_group = bootstrap_module.attach_sequence_groups(
        current,
        Path(args.manifest),
        args.split_column,
        args.test_label,
        args.sequence_column,
    )
    comparisons = [("s3", "s1"), ("s3", "s2"), ("s4", "s3")]
    event_metrics = bootstrap_module.build_event_level_metrics(
        current_seq, ["s1", "s2", "s3", "s4"]
    )
    delta_matrix, metadata = bootstrap_module.build_delta_matrix(
        event_metrics, event_group, comparisons
    )
    group_names, group_indices = bootstrap_module.group_index_arrays(
        event_group
    )

    hierarchical_distribution = (
        bootstrap_module.hierarchical_sequence_event_bootstrap(
            delta_matrix,
            group_indices,
            args.bootstrap_repetitions,
            args.seed + 1,
        )
    )
    hierarchical_summary = bootstrap_module.bootstrap_summary(
        hierarchical_distribution,
        metadata,
        "hierarchical_sequence_event",
        args.confidence_level,
    )
    hierarchical_path = (
        out_dir / "structure_hierarchical_bootstrap_summary.csv"
    )
    hierarchical_summary.to_csv(hierarchical_path, index=False)

    cluster_distribution = bootstrap_module.sequence_cluster_bootstrap(
        delta_matrix,
        group_indices,
        args.bootstrap_repetitions,
        args.seed,
    )
    cluster_summary = bootstrap_module.bootstrap_summary(
        cluster_distribution,
        metadata,
        "sequence_cluster",
        args.confidence_level,
    )
    cluster_path = (
        out_dir / "structure_sequence_cluster_bootstrap_summary.csv"
    )
    cluster_summary.to_csv(cluster_path, index=False)

    primary_bootstrap = hierarchical_summary.loc[
        hierarchical_summary["metric"].isin(
            ["mae", "under05", "factor2"]
        )
    ].copy()
    primary_path = out_dir / "paper_structure_bootstrap_table.csv"
    primary_bootstrap.to_csv(primary_path, index=False)

    protocol = {
        **vars(args),
        "newly_trained_models": variants,
        "reused_models": ["S3_cross_attention_base", "S4_ca_urc"],
        "S1_definition": (
            "remove station self-attention block; retain target-conditioned "
            "cross-attention and target query"
        ),
        "S2_definition": (
            "retain station self-attention and target query; replace target "
            "cross-attention with global station mean"
        ),
        "S3_definition": "full Cross-Attention Base",
        "S4_definition": "S3 + CA-URC underprediction-risk correction",
        "checkpoint_selection": "validation mean PGA/PGV overall MAE only",
        "test_used_for_selection": False,
        "metric_aggregation": "targets -> repeats -> events",
        "primary_statistics": "hierarchical sequence -> event paired bootstrap",
        "bootstrap_comparisons": ["S3-S1", "S3-S2", "S4-S3"],
        "n_test_events": int(len(test_dataset.frame)),
        "n_sequence_groups": int(len(group_names)),
    }
    (out_dir / "run_configuration.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\n=== FINAL NETWORK-STRUCTURE ABLATION TABLE ===")
    print(compact.to_string(index=False))

    print("\n=== PRIMARY STRUCTURE BOOTSTRAP ===")
    display = primary_bootstrap[
        [
            "candidate",
            "reference",
            "quantity",
            "population",
            "metric",
            "reference_value",
            "candidate_value",
            "point_delta_candidate_minus_reference",
            "ci_lower",
            "ci_upper",
            "ci_excludes_zero",
            "two_sided_bootstrap_sign_p",
            "n_paired_events",
        ]
    ].copy()
    print(display.to_string(index=False))

    print("\nDelta interpretation:")
    print("  s3 - s1: negative MAE/U0.5 favors station self-attention.")
    print("  s3 - s2: negative MAE/U0.5 favors target cross-attention.")
    print("  s4 - s3: negative MAE/U0.5 favors CA-URC correction.")

    print("\nOutputs:")
    for path in [
        prediction_path,
        metrics_path,
        compact_path,
        hierarchical_path,
        cluster_path,
        primary_path,
        out_dir / "training_selection_summary.csv",
        out_dir / "pairing_audit.json",
        out_dir / "run_configuration.json",
    ]:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
