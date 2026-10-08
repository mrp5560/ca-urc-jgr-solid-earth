#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
53_train_time_extrapolation_attention_base.py

Chronology-safe training for the FINAL Attention Base:
waveform + geometry + P-offset + station-attention pooling.

The script imports the audited causal dataset/threshold logic from
21_train_time_extrapolation_mean_base.py, but replaces mean pooling with
the final attention pooling architecture.

Only train/validation are instantiated. Test and Ridgecrest OOD are not used.
Primary validation aggregation: targets -> repeats -> events.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "21_train_time_extrapolation_mean_base.py"


def load_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("chrono_mean_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module(BASE_SCRIPT)

SparseFieldDataset = M.SparseFieldDataset
WaveformEncoder = M.WaveformEncoder
as_bool = M.as_bool
compute_training_thresholds = M.compute_training_thresholds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class AttentionPoolingSparseFieldModel(nn.Module):
    """Final validation-selected base architecture."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.waveform_encoder = WaveformEncoder(self.hidden_dim)

        self.station_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.attention_score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.query_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim + 3, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def encode_event(self, input_waveforms, input_features):
        b, k, c, t = input_waveforms.shape

        wave = self.waveform_encoder(
            input_waveforms.reshape(b * k, c, t)
        ).reshape(b, k, -1)

        station = self.station_encoder(
            torch.cat([wave, input_features], dim=-1)
        )

        weight = torch.softmax(
            self.attention_score(station),
            dim=1,
        )

        event = torch.sum(
            weight * station,
            dim=1,
        )

        return event, station, weight

    def forward(self, input_waveforms, input_features, target_features):
        event, _, _ = self.encode_event(
            input_waveforms,
            input_features,
        )

        event = event[:, None, :].expand(
            -1,
            target_features.shape[1],
            -1,
        )

        return self.query_decoder(
            torch.cat([event, target_features], dim=-1)
        )


# Public aliases for subsequent chronology-tail scripts.
MeanBase = AttentionPoolingSparseFieldModel
MeanPoolingSparseFieldModel = AttentionPoolingSparseFieldModel


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n_batches = 0

    for batch in loader:
        pred = model(
            batch["input_waveforms"].to(device),
            batch["input_features"].to(device),
            batch["target_features"].to(device),
        )
        target = batch["target_log"].to(device)

        loss = nn.functional.smooth_l1_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total += float(loss.detach().cpu())
        n_batches += 1

    return total / max(n_batches, 1)


def canonical_mean(frame, column, mask=None):
    selected = frame
    if mask is not None:
        selected = frame.loc[np.asarray(mask, dtype=bool)]

    if len(selected) == 0:
        return float("nan")

    event_repeat = (
        selected.groupby(
            ["event_id", "repeat"],
            sort=False,
        )[column]
        .mean()
        .reset_index()
    )

    event = (
        event_repeat.groupby(
            "event_id",
            sort=False,
        )[column]
        .mean()
    )

    return float(event.mean())


def evaluate(model, loader, thresholds, device):
    model.eval()
    rows = []

    with torch.inference_mode():
        for batch in loader:
            pred = model(
                batch["input_waveforms"].to(device),
                batch["input_features"].to(device),
                batch["target_features"].to(device),
            ).cpu().numpy()

            truth = batch["target_log"].numpy()
            magnitude = batch["magnitude"].numpy()
            repeats = batch["repeat"].numpy()
            event_ids = list(batch["event_id"])

            for b in range(truth.shape[0]):
                for q in range(truth.shape[1]):
                    y_pga = float(truth[b, q, 0])
                    y_pgv = float(truth[b, q, 1])

                    rows.append({
                        "event_id": str(event_ids[b]),
                        "repeat": int(repeats[b]),
                        "target_slot": int(q),
                        "magnitude": float(magnitude[b]),
                        "true_log10_pga": y_pga,
                        "true_log10_pgv": y_pgv,
                        "pred_log10_pga": float(pred[b, q, 0]),
                        "pred_log10_pgv": float(pred[b, q, 1]),
                        "is_tail_pga": bool(y_pga >= thresholds[0]),
                        "is_tail_pgv": bool(y_pgv >= thresholds[1]),
                    })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No validation predictions.")

    metrics = {
        "n_rows": int(len(frame)),
        "n_events": int(frame["event_id"].nunique()),
        "n_event_repeats": int(
            frame[["event_id", "repeat"]].drop_duplicates().shape[0]
        ),
    }

    for quantity in ("pga", "pgv"):
        truth = frame[f"true_log10_{quantity}"].to_numpy(float)
        pred = frame[f"pred_log10_{quantity}"].to_numpy(float)
        residual = pred - truth
        absolute = np.abs(residual)
        tail = frame[f"is_tail_{quantity}"].to_numpy(bool)
        non_tail = ~tail

        frame[f"abs_{quantity}"] = absolute
        frame[f"res_{quantity}"] = residual
        frame[f"u05_{quantity}"] = (residual <= -0.5).astype(float)
        frame[f"f2_{quantity}"] = (
            absolute <= np.log10(2.0)
        ).astype(float)

        metrics[f"canonical_mae_{quantity}"] = canonical_mean(
            frame, f"abs_{quantity}"
        )
        metrics[f"canonical_bias_{quantity}"] = canonical_mean(
            frame, f"res_{quantity}"
        )
        metrics[f"canonical_factor2_{quantity}"] = canonical_mean(
            frame, f"f2_{quantity}"
        )
        metrics[f"canonical_under05_{quantity}"] = canonical_mean(
            frame, f"u05_{quantity}"
        )

        metrics[f"canonical_tail_mae_{quantity}"] = canonical_mean(
            frame, f"abs_{quantity}", tail
        )
        metrics[f"canonical_tail_under05_{quantity}"] = canonical_mean(
            frame, f"u05_{quantity}", tail
        )
        metrics[f"canonical_non_tail_mae_{quantity}"] = canonical_mean(
            frame, f"abs_{quantity}", non_tail
        )

        metrics[f"tail_prevalence_{quantity}"] = float(tail.mean())

    metrics["selection_score"] = 0.5 * (
        metrics["canonical_mae_pga"]
        + metrics["canonical_mae_pgv"]
    )

    return metrics, frame


def save_checkpoint(path, model, optimizer, epoch, args, metrics, thresholds):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "args": vars(args),
            "validation_metrics": metrics,
            "thresholds": thresholds,
            "base_architecture": "attention_pooling_full",
            "split_protocol": "chronological_time_clean",
            "test_split_evaluated": False,
            "ridgecrest_ood_evaluated": False,
        },
        path,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--manifest", required=True)
    p.add_argument("--split-column", default="split_time_clean")
    p.add_argument("--train-label", default="train")
    p.add_argument("--validation-label", default="validation")

    p.add_argument("--t0-sec", type=int, default=5)
    p.add_argument("--input-stations", type=int, default=5)
    p.add_argument("--target-stations", type=int, default=10)
    p.add_argument("--input-pre-sec", type=float, default=2.0)

    p.add_argument("--validation-repeats", type=int, default=3)
    p.add_argument("--final-validation-repeats", type=int, default=20)
    p.add_argument("--tail-quantile", type=float, default=0.90)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-patience", type=int, default=4)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--minimum-delta", type=float, default=1e-4)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--out-dir",
        default="runs/time_clean_attention_base_t0_5s_k5",
    )

    args = p.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device(args.device)

    if args.validation_repeats < 1:
        raise ValueError("--validation-repeats must be >= 1.")
    if args.final_validation_repeats < 1:
        raise ValueError("--final-validation-repeats must be >= 1.")
    if not 0.5 < args.tail_quantile < 1.0:
        raise ValueError("--tail-quantile must be between 0.5 and 1.")

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold_path = out_dir / (
        f"tail_thresholds_q{args.tail_quantile:.2f}_"
        f"t0_{args.t0_sec}s.json"
    )

    # Uses chronological training rows only and each event-station once.
    thresholds = compute_training_thresholds(
        manifest=args.manifest,
        split_column=args.split_column,
        train_label=args.train_label,
        t0_sec=args.t0_sec,
        quantile=args.tail_quantile,
        cache_path=threshold_path,
    )

    threshold_array = np.asarray(
        [
            thresholds["log10_pga_threshold"],
            thresholds["log10_pgv_threshold"],
        ],
        dtype=np.float32,
    )

    train_dataset = SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.train_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=True,
    )

    val_dataset = SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.validation_repeats,
    )

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": False,
    }

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args,
    )

    model = AttentionPoolingSparseFieldModel(
        args.hidden_dim
    ).to(device)

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

    checkpoint_path = out_dir / "best_model.pt"

    print("=== Chronological Attention-Base training ===")
    print(f"Device                  : {device}")
    print("Architecture            : attention_pooling_full")
    print(f"Split column            : {args.split_column}")
    print(f"Train events            : {len(train_dataset.frame)}")
    print(f"Validation events       : {len(val_dataset.frame)}")
    print(f"Validation repeats      : {args.validation_repeats}")
    print(f"Final validation repeats: {args.final_validation_repeats}")
    print(
        "Parameters              : "
        f"{sum(x.numel() for x in model.parameters()):,}"
    )
    print(
        "Chronological Q90       : "
        f"PGA={threshold_array[0]:.4f}, "
        f"PGV={threshold_array[1]:.4f}"
    )
    print("Test/Ridgecrest OOD     : NOT ACCESSED")

    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics, _ = evaluate(
            model,
            val_loader,
            threshold_array,
            device,
        )

        score = float(
            val_metrics["selection_score"]
        )
        scheduler.step(score)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{
                f"validation_{k}": v
                for k, v in val_metrics.items()
            },
        })
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val PGA={val_metrics['canonical_mae_pga']:.4f} | "
            f"PGV={val_metrics['canonical_mae_pgv']:.4f} | "
            f"score={score:.4f} | "
            f"tail={val_metrics['canonical_tail_mae_pga']:.4f}/"
            f"{val_metrics['canonical_tail_mae_pgv']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if score < best_score - args.minimum_delta:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                args,
                val_metrics,
                thresholds,
            )
        else:
            stale += 1

        if stale >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}."
            )
            break

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    # Stable 20-repeat validation audit, still no test access.
    final_val_dataset = SparseFieldDataset(
        manifest=args.manifest,
        split_column=args.split_column,
        split_label=args.validation_label,
        t0_sec=args.t0_sec,
        input_stations=args.input_stations,
        target_stations=args.target_stations,
        input_pre_sec=args.input_pre_sec,
        seed=args.seed,
        training=False,
        repeats=args.final_validation_repeats,
    )

    final_val_loader = DataLoader(
        final_val_dataset,
        shuffle=False,
        **loader_args,
    )

    final_metrics, final_predictions = evaluate(
        model,
        final_val_loader,
        threshold_array,
        device,
    )

    prediction_path = (
        out_dir
        / "validation_predictions_best_20repeats.csv"
    )
    final_predictions.to_csv(
        prediction_path,
        index=False,
    )

    (out_dir / "validation_metrics_best.json").write_text(
        json.dumps(
            {
                "best_epoch": int(checkpoint["epoch"]),
                "training_seconds": float(
                    time.time() - start_time
                ),
                "validation_repeats": int(
                    args.final_validation_repeats
                ),
                "base_architecture": "attention_pooling_full",
                "split_protocol": "chronological_time_clean",
                "test_split_evaluated": False,
                "ridgecrest_ood_evaluated": False,
                **final_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out_dir / "run_configuration.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(device),
                "best_epoch": int(checkpoint["epoch"]),
                "tail_thresholds": thresholds,
                "base_architecture": "attention_pooling_full",
                "metric_aggregation": "targets -> repeats -> events",
                "test_split_evaluated": False,
                "ridgecrest_ood_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Best chronological Attention checkpoint ===")
    print(f"Epoch                   : {checkpoint['epoch']}")
    print(
        "Overall PGA/PGV         : "
        f"{final_metrics['canonical_mae_pga']:.4f}/"
        f"{final_metrics['canonical_mae_pgv']:.4f}"
    )
    print(
        "Tail PGA/PGV            : "
        f"{final_metrics['canonical_tail_mae_pga']:.4f}/"
        f"{final_metrics['canonical_tail_mae_pgv']:.4f}"
    )
    print(
        "Tail U0.5 PGA/PGV       : "
        f"{final_metrics['canonical_tail_under05_pga']:.3f}/"
        f"{final_metrics['canonical_tail_under05_pgv']:.3f}"
    )
    print(f"Checkpoint              : {checkpoint_path.resolve()}")
    print(f"Threshold JSON          : {threshold_path.resolve()}")
    print(f"20-repeat predictions   : {prediction_path.resolve()}")
    print("Test/Ridgecrest OOD     : NOT ACCESSED")


if __name__ == "__main__":
    main()
