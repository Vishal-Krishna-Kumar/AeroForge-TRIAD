"""Render the training-visualisation video (outreach deliverable; docs/outreach/training_video_plan.md).

Every number and field shown comes from the logged training run (video_frames_seed*.npz written by
scripts/train_surrogates.py) and the test-set report (h1_test_metrics.json). Nothing is staged.

Usage: python scripts/render_training_video.py --frames checkpoints/v1/fno/video_frames_seed0.npz \
           --metrics checkpoints/v1/fno/h1_test_metrics.json --out results/video/aeroforge_training.mp4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

BG, FG, ACC, ACC2, MUTED = "#0f1117", "#e8eaf0", "#5b8def", "#f2a541", "#8a90a0"
EXTENT = (-0.5, 1.5, -0.5, 0.5)


def field_img(ax, f, mask, vmin, vmax):
    a = np.ma.masked_where(mask == 0, f.astype(np.float32))
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#3a3f4b")
    ax.imshow(a, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xlim(-0.3, 1.4); ax.set_ylim(-0.42, 0.42); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a2f3a")


def text_card(fig, lines, sizes, colors=None):
    fig.clf(); fig.patch.set_facecolor(BG)
    n = len(lines)
    for i, (t, sz) in enumerate(zip(lines, sizes)):
        fig.text(0.5, 0.62 - i * 0.11, t, ha="center", va="center", color=(colors[i] if colors else FG), fontsize=sz,
                 fontweight="bold" if i == 0 else "normal")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=int, default=4, help="video frames per training snapshot")
    args = ap.parse_args()
    z = np.load(args.frames, allow_pickle=False)
    ep, P, PT, MK = z["epoch"], z["p_pred"].astype(np.float32), z["p_true"], z["mask"]
    cos, loss, val_ld, fmse = z["grad_cos"], z["loss"], z["val_rel_err_LD"], z["val_field_mse"]
    meta = json.loads(str(z["meta"]))
    n_epochs = int(ep.max()) + 1
    vals = PT[MK == 1]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    met = json.loads(Path(args.metrics).read_text()) if args.metrics and Path(args.metrics).exists() else None
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = FFMpegWriter(fps=args.fps, bitrate=6000, metadata={"title": "AeroForge-TRIAD: watching a neural operator learn airflow"})
    fps = args.fps
    with writer.saving(fig, str(out), dpi=100):
        # 1. title
        text_card(fig, ["Teaching a neural network to see airflow",
                        "AeroForge-TRIAD  ·  Fourier neural operator trained on 448 CFD simulations",
                        "NACA 4-digit airfoils · Re = 1000 · SU2 laminar Navier–Stokes"], [44, 24, 20], [FG, ACC, MUTED])
        for _ in range(int(3.5 * fps)):
            writer.grab_frame()
        # 2. ground truth
        fig.clf(); fig.patch.set_facecolor(BG)
        fig.text(0.5, 0.93, "Ground truth: pressure around four airfoils, computed by CFD", ha="center", color=FG, fontsize=28, fontweight="bold")
        fig.text(0.5, 0.88, "each simulation takes ~4 minutes of solver time", ha="center", color=MUTED, fontsize=18)
        for k in range(4):
            ax = fig.add_axes([0.04 + k * 0.24, 0.3, 0.22, 0.45]); field_img(ax, PT[k], MK[k], vmin, vmax)
            ax.set_title(meta["field_cases"][k], color=MUTED, fontsize=14)
        for _ in range(int(4 * fps)):
            writer.grab_frame()
        # 3. training timelapse
        for i in range(len(ep)):
            fig.clf(); fig.patch.set_facecolor(BG)
            e = int(ep[i])
            fig.text(0.03, 0.95, "Training timelapse", color=FG, fontsize=30, fontweight="bold", va="top")
            fig.text(0.03, 0.90, "pressure around four airfoils the network has never trained on", color=MUTED, fontsize=17, va="top")
            fig.text(0.97, 0.95, f"epoch {max(e, 0):>3d} / {n_epochs}" if e >= 0 else "untrained network", color=ACC2,
                     fontsize=30, fontweight="bold", va="top", ha="right")
            for k in range(4):
                a1 = fig.add_axes([0.03 + k * 0.155, 0.53, 0.145, 0.16]); field_img(a1, PT[k], MK[k], vmin, vmax)
                a2 = fig.add_axes([0.03 + k * 0.155, 0.27, 0.145, 0.16]); field_img(a2, P[i, k], MK[k], vmin, vmax)
                if k == 0:
                    a1.set_ylabel("CFD", color=FG, fontsize=16); a2.set_ylabel("network", color=ACC, fontsize=16)
            sl = slice(0, i + 1)
            x = np.maximum(ep[sl], 0) + 1
            ax = fig.add_axes([0.68, 0.70, 0.29, 0.16]); ax.set_facecolor(BG)
            ax.semilogy(x, np.maximum(loss[sl], 1e-6), color=ACC, lw=2.5); ax.set_xlim(1, n_epochs)
            ax.set_title(f"training loss: {loss[i]:.3g}", color=FG, fontsize=15, loc="left")
            ax2 = fig.add_axes([0.68, 0.42, 0.29, 0.16]); ax2.set_facecolor(BG)
            ax2.semilogy(x, np.maximum(val_ld[sl] * 100, 1e-3), color=ACC2, lw=2.5); ax2.set_xlim(1, n_epochs)
            ax2.set_title(f"L/D error on unseen airfoils: {val_ld[i] * 100:.2f} %", color=FG, fontsize=15, loc="left")
            ax3 = fig.add_axes([0.68, 0.14, 0.29, 0.16]); ax3.set_facecolor(BG)
            mc = np.median(cos[sl], axis=1)
            ax3.plot(x, mc, color="#6fd08c", lw=2.5); ax3.set_xlim(1, n_epochs); ax3.set_ylim(-1, 1.02)
            ax3.axhline(0.98, color=MUTED, lw=1, ls="--")
            ax3.set_title(f"gradient direction vs adjoint CFD (cosine): {mc[-1]:.3f}", color=FG, fontsize=15, loc="left")
            for a in (ax, ax2, ax3):
                a.tick_params(which="both", colors=MUTED, labelsize=11)
                a.set_xlabel("epoch", color=MUTED, fontsize=11)
                for s in a.spines.values():
                    s.set_color("#2a2f3a")
            for _ in range(args.hold):
                writer.grab_frame()
        for _ in range(int(2 * fps)):
            writer.grab_frame()
        # 4. accuracy is not enough
        fig.clf(); fig.patch.set_facecolor(BG)
        fig.text(0.5, 0.92, "Do the network's design directions match adjoint CFD?", ha="center", color=FG, fontsize=30, fontweight="bold")
        fig.text(0.5, 0.86, "per-airfoil cosine between the network's L/D gradient and the adjoint-CFD gradient (1 = same direction)",
                 ha="center", color=MUTED, fontsize=17)
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.62]); ax.set_facecolor(BG)
        c = cos[-1]
        ax.bar(np.arange(len(c)), c, color=[("#6fd08c" if v >= 0.98 else ACC2) for v in c])
        ax.axhline(0.98, color=MUTED, ls="--", lw=1); ax.set_ylim(min(-0.2, c.min() - 0.05), 1.05)
        ax.set_xticks(np.arange(len(c))); ax.set_xticklabels([s.replace("D1_", "") for s in meta["grad_cases"]], rotation=60, color=MUTED, fontsize=10)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_color("#2a2f3a")
        for _ in range(int(6 * fps)):
            writer.grab_frame()
        # 5. end card (test-set numbers from the evaluation report, if available)
        if met:
            e = met["ensemble"]
            lines = ["Final results on 128 never-seen test airfoils (5-model ensemble)",
                     f"lift-to-drag error (median): {e['median_rel_err_LD'] * 100:.2f} %",
                     f"design-gradient error (median, relative): {e['median_rel_grad_err_LD'] * 100:.1f} %",
                     f"gradient direction cosine (median): {e['median_cos']:.3f}",
                     "next: certify when a predicted gradient can be trusted (CDGC)"]
            text_card(fig, lines, [30, 24, 24, 24, 20], [FG, ACC2, ACC2, "#6fd08c", MUTED])
            for _ in range(int(6 * fps)):
                writer.grab_frame()
    plt.close(fig)
    print("VIDEO", out, os.path.getsize(out), "bytes", flush=True)


if __name__ == "__main__":
    main()
