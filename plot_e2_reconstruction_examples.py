#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="runs/e2_masked_reconstruction/reconstruction_examples.npz")
    p.add_argument("--out-dir", default="runs/e2_debug100/figures")
    p.add_argument("--sampling-rate", type=float, default=100.0)
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz)

    x_true = data["x_true"]
    x_pred = data["x_pred"]
    visible_mask = data["visible_mask"]

    i = args.index
    t = np.arange(x_true.shape[-1]) / args.sampling_rate

    comp_names = ["Z", "N", "E"]

    for c in range(3):
        plt.figure(figsize=(14, 4))

        plt.plot(t, x_true[i, c], linewidth=1.0, label="True")
        plt.plot(t, x_pred[i, c], linewidth=1.0, label="Reconstructed", alpha=0.8)

        mask = visible_mask[i, 0]
        masked = mask < 0.5

        if masked.any():
            ymin = min(x_true[i, c].min(), x_pred[i, c].min())
            ymax = max(x_true[i, c].max(), x_pred[i, c].max())
            plt.fill_between(
                t,
                ymin,
                ymax,
                where=masked,
                alpha=0.15,
                label="Masked region",
            )

        plt.xlabel("Time since window start (s)")
        plt.ylabel("Normalized amplitude")
        plt.title(f"E2 reconstruction example {i}, component {comp_names[c]}")
        plt.legend()
        plt.tight_layout()

        out = out_dir / f"example_{i}_component_{comp_names[c]}.png"
        plt.savefig(out, dpi=200)
        plt.close()

        print("saved:", out)


if __name__ == "__main__":
    main()