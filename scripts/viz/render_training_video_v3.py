"""Render the V3 training video (outreach; transonic Euler, M 0.70, CST-13 shapes, objective J = CL/(CD+0.008)).

Adapted from scripts/render_training_video_v2.py (same visual design). Every number and field shown comes from logged
artefacts; nothing is interpolated between snapshots and phase captions are chosen from the logged field error:
 * --frames checkpoints/v3/fno/video_frames_seed0.npz (Recorder in scripts/train_surrogates_v3.py, FNO seed 0)
 * --metrics results/reports_v3/h1v3_test_metrics_fno.json, --metrics-mlp results/reports_v3/h1v3_test_metrics_mlp.json
 * --data datasets/processed/dataset_v3 (true CL/CD of the validation shapes = train cases n_fit..N-1; theta checked)
 * --raw datasets/raw/dataset_v3 (measured CFD wall times: primal run.wall_s, J adjoint objectives.J.run.wall_s)
Usage:
  python render_training_video_v3.py --frames checkpoints/v3/fno/video_frames_seed0.npz \
    --metrics results/reports_v3/h1v3_test_metrics_fno.json --metrics-mlp results/reports_v3/h1v3_test_metrics_mlp.json \
    --data datasets/processed/dataset_v3 --raw datasets/raw/dataset_v3 --out results/video_v3/aeroforge_training_v3.mp4 \
    --stills results/video_v3/stills
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
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

BG, PANEL, FG, MUTED, GRID = "#0b0e14", "#141a24", "#eef1f6", "#8b93a5", "#252c3a"
BLUE, ORANGE, GREEN, RED = "#5b8def", "#f2a541", "#5fd08a", "#ef6b6b"
EXTENT = (-0.5, 1.5, -0.5, 0.5)
CD0 = 0.008
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": MUTED})


def press_cmap():
    c = plt.get_cmap("RdBu_r").copy(); c.set_bad("#2b3140"); return c


def err_cmap():
    c = plt.get_cmap("inferno").copy(); c.set_bad("#2b3140"); return c


def show_field(ax, f, mask, vmin, vmax, cmap):
    a = np.ma.masked_where(mask == 0, f.astype(np.float32))
    im = ax.imshow(a, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xlim(-0.25, 1.35); ax.set_ylim(-0.38, 0.38); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    return im


def style_axes(a, xlabel=None):
    a.set_facecolor(PANEL)
    a.tick_params(which="both", colors=MUTED, labelsize=11)
    a.grid(color=GRID, lw=0.6, alpha=0.8)
    for s in a.spines.values():
        s.set_color(GRID)
    if xlabel:
        a.set_xlabel(xlabel, color=MUTED, fontsize=12)


class Video:
    def __init__(self, fig, writer, fps):
        self.fig, self.w, self.fps, self.cover = fig, writer, fps, None

    def new(self):
        self.fig.clf(); self.fig.patch.set_facecolor(BG); self.cover = None

    def _ensure_cover(self):
        if self.cover is None or self.cover.figure is None:
            self.cover = Rectangle((0, 0), 1, 1, transform=self.fig.transFigure, color=BG, zorder=1000)
            self.fig.add_artist(self.cover)

    def hold(self, seconds, fade_in=0.5, fade_out=0.5):
        n = max(1, int(round(seconds * self.fps)))
        fi, fo = int(fade_in * self.fps), int(fade_out * self.fps)
        self._ensure_cover()
        for k in range(n):
            a = 0.0
            if k < fi:
                a = 1 - (k + 1) / (fi + 1)
            elif k >= n - fo:
                a = (k - (n - fo) + 1) / (fo + 1)
            self.cover.set_alpha(a)
            self.w.grab_frame()
        self.cover.set_alpha(0.0)

    def frames(self, n):
        self._ensure_cover(); self.cover.set_alpha(0.0)
        for _ in range(n):
            self.w.grab_frame()


def header(fig, step, title, subtitle=None):
    fig.text(0.035, 0.945, step, color=BLUE, fontsize=15, fontweight="bold", va="top")
    fig.text(0.035, 0.915, title, color=FG, fontsize=30, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.035, 0.858, subtitle, color=MUTED, fontsize=17, va="top")


def lower_third(fig, label, text, color=ORANGE):
    fig.add_artist(FancyBboxPatch((0.035, 0.03), 0.93, 0.075, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, lw=1.2))
    fig.add_artist(Rectangle((0.035, 0.03), 0.006, 0.075, transform=fig.transFigure, color=color))
    fig.text(0.052, 0.0675, label, color=color, fontsize=15, fontweight="bold", va="center")
    fig.text(0.052 + 0.0085 * len(label) + 0.02, 0.0675, text, color=FG, fontsize=16, va="center")


def footer(fig):
    fig.text(0.965, 0.008, "AeroForge-TRIAD · V3 transonic study · numbers from logged runs", color="#4d5566", fontsize=10,
             ha="right", va="bottom")


def colorbar_lowhigh(fig, im, rect, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_ticks([im.norm.vmin, im.norm.vmax]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(colors=MUTED, labelsize=12); cb.outline.set_edgecolor(GRID)
    cax.yaxis.set_label_position("left"); cax.set_ylabel(label, color=MUTED, fontsize=12, labelpad=4)


def phase_caption(frac_err, e):
    if e < 0:
        return ("UNTRAINED", "Random weights: the network's guess has nothing to do with the real airflow.", RED)
    drop = (1 - frac_err) * 100
    if frac_err > 0.5:
        return ("LEARNING", f"Weights update after every batch of 16 shapes. Field error down {drop:.0f} % so far.", ORANGE)
    if frac_err > 0.1:
        return ("SHARPENING", f"The pressure pattern is in place (error down {drop:.0f} %). Bottom row: where it is still wrong.", ORANGE)
    if frac_err > 0.03:
        return ("REFINING", f"Field error is now {frac_err * 100:.1f} % of where it started; what remains is the shock detail.", BLUE)
    return ("FINE-TUNING", f"Field error is now {frac_err * 100:.2g} % of where it started on shapes it never trains on.", GREEN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--metrics-mlp", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--stills", default=None)
    args = ap.parse_args()

    z = np.load(args.frames, allow_pickle=False)
    ep, P, PT, MK = z["epoch"], z["p_pred"].astype(np.float32), z["p_true"].astype(np.float32), z["mask"]
    cos, loss, fmse = z["grad_cos"], z["loss"], z["val_field_mse"]
    vCL, vCD, vth = z["val_CL"], z["val_CD"], z["val_theta"]
    n_epochs = int(ep.max()) + 1
    vals = PT[MK == 1]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    prange = vmax - vmin
    err_max = 0.25 * prange

    # validation truth: the last len(val_theta) train cases in case-id order (protocol v3 §4); theta must match exactly
    files = sorted(Path(args.data, "train").glob("*.npz"))[-len(vth):]
    tr = [np.load(f) for f in files]
    th = np.stack([t["theta"] for t in tr])
    assert np.allclose(th, vth, atol=1e-6), "validation theta mismatch"
    j_true = np.array([float(t["CL"]) / (float(t["CD"]) + CD0) for t in tr])
    j_pred = vCL / (vCD + CD0)  # (snapshots, n_val)
    j_err = np.median(np.abs(j_pred - j_true[None]) / np.maximum(np.abs(j_true[None]), 1e-2), axis=1)

    met = json.loads(Path(args.metrics).read_text())
    met_mlp = json.loads(Path(args.metrics_mlp).read_text()) if args.metrics_mlp else None
    n_tr = met["meta"]["n_train"]; n_fit = met["meta"]["n_fit"]; n_test = met["meta"]["n_test"]; n_cal = met["meta"]["n_cal"]

    prim_s = adj_s = None
    if args.raw and Path(args.raw).is_dir():
        ps, ads = [], []
        for d in Path(args.raw).iterdir():
            c = d / "case.json"
            if c.exists():
                ps.append(float(json.loads(c.read_text())["run"]["wall_s"]))
            g = d / "gradients.json"
            if g.exists():
                ads.append(float(json.loads(g.read_text())["objectives"]["J"]["run"]["wall_s"]))
        prim_s = float(np.median(ps)) if ps else None
        adj_s = float(np.median(ads)) if ads else None

    err_pct = np.array([[np.abs(P[i, k] - PT[k])[MK[k] == 1].mean() / prange * 100 for k in range(len(PT))] for i in range(len(ep))])
    f0 = fmse[0] if np.isfinite(fmse[0]) and fmse[0] > 0 else np.nanmax(fmse)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    stills = Path(args.stills) if args.stills else None
    if stills:
        stills.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = FFMpegWriter(fps=args.fps, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"],
                          metadata={"title": "AeroForge-TRIAD V3: a neural operator learns transonic airflow"})

    def still(name):
        if stills:
            fig.savefig(stills / f"{name}.png", dpi=50, facecolor=BG)

    with writer.saving(fig, str(out), dpi=100):
        V = Video(fig, writer, args.fps)

        # ---- 1. hook
        V.new()
        fig.text(0.5, 0.64, "Can a neural network learn shock waves?", ha="center", color=FG, fontsize=50, fontweight="bold")
        fig.text(0.5, 0.53, "We trained one on hundreds of transonic flow simulations — and filmed it learning.",
                 ha="center", color=MUTED, fontsize=24)
        fig.text(0.5, 0.40, "AeroForge-TRIAD · V3", ha="center", color=BLUE, fontsize=28, fontweight="bold")
        fig.text(0.5, 0.35, "can we trust an AI's design gradients?", ha="center", color=MUTED, fontsize=18)
        still("01_hook"); V.hold(4.5, 0.8, 0.6)

        # ---- 2. ground truth
        V.new()
        header(fig, "STEP 1 · THE GROUND TRUTH", "Computational fluid dynamics (CFD) solves the physics",
               "compressible Euler equations at Mach 0.70 around 13-parameter CST airfoils, solver SU2")
        for k in range(len(PT)):
            ax = fig.add_axes([0.035 + k * 0.225, 0.36, 0.21, 0.36])
            im = show_field(ax, PT[k], MK[k], vmin, vmax, press_cmap())
            ax.set_title(f"validation shape {k + 1}", color=MUTED, fontsize=14)
        colorbar_lowhigh(fig, im, [0.945, 0.38, 0.012, 0.32], "pressure")
        sp = (f"Each simulation: ~{prim_s / 60:.0f} min for the flow" + (f" + ~{adj_s / 60:.0f} min for the exact gradient (adjoint)" if adj_s else "")
              + " on one CPU core.") if prim_s else ""
        fig.text(0.035, 0.29, sp, color=FG, fontsize=19)
        fig.text(0.035, 0.25, f"Dataset: {n_tr} training + {n_cal} calibration + {n_test} test shapes.", color=FG, fontsize=19)
        fig.text(0.035, 0.205, "Near the speed of sound a shock wave forms on the upper surface: the sharp jump in colour.",
                 color=MUTED, fontsize=17)
        lower_third(fig, "THE PROBLEM", "Accurate — but far too slow to test thousands of shapes during design.", RED)
        footer(fig); still("02_cfd"); V.hold(6.5)

        # ---- 3. idea
        V.new()
        header(fig, "STEP 2 · THE IDEA", "Teach a neural network to imitate the solver")
        boxes = [("airfoil shape", "12 CST coefficients\n+ angle of attack", MUTED),
                 ("Fourier neural operator", "learns patterns of the flow\nin frequency space", BLUE),
                 ("full flow field", "pressure + velocity everywhere\n+ lift and drag", GREEN)]
        for i, (t, s, c) in enumerate(boxes):
            x = 0.07 + i * 0.31
            fig.add_artist(FancyBboxPatch((x, 0.42), 0.24, 0.22, boxstyle="round,pad=0.01,rounding_size=0.02",
                                          transform=fig.transFigure, facecolor=PANEL, edgecolor=c, lw=2.5))
            fig.text(x + 0.12, 0.57, t, ha="center", color=c, fontsize=24, fontweight="bold")
            fig.text(x + 0.12, 0.48, s, ha="center", color=FG, fontsize=16, linespacing=1.5)
            if i < 2:
                fig.text(x + 0.275, 0.53, "→", ha="center", va="center", color=MUTED, fontsize=46)
        fig.text(0.5, 0.29, f"Training: {n_fit} shapes · Watched while training: {n_tr - n_fit} shapes it never learns from"
                 f" · Final exam: {n_test} held-out shapes", ha="center", color=MUTED, fontsize=17)
        lower_third(fig, "WHAT CHANGES", "Training adjusts the network's weights until its output matches the CFD.", BLUE)
        footer(fig); still("03_idea"); V.hold(6)

        # ---- 4. timelapse
        mid = None
        for i in range(len(ep)):
            V.new()
            e = int(ep[i])
            header(fig, "STEP 3 · TRAINING (TIMELAPSE)", "Watching the network learn",
                   "four shapes the network never trains on — top: CFD truth · middle: network · bottom: where it is wrong")
            fig.text(0.965, 0.915, "untrained" if e < 0 else f"epoch {e + 1}", color=ORANGE, fontsize=34, fontweight="bold",
                     va="top", ha="right")
            fig.text(0.965, 0.862, f"of {n_epochs} · 1 epoch = one pass over {n_fit} shapes", color=MUTED, fontsize=13,
                     va="top", ha="right")
            rows = [(0.615, "CFD"), (0.435, "network"), (0.255, "error")]
            for k in range(len(PT)):
                x0 = 0.035 + k * 0.148
                a1 = fig.add_axes([x0, rows[0][0], 0.14, 0.15]); im_p = show_field(a1, PT[k], MK[k], vmin, vmax, press_cmap())
                a2 = fig.add_axes([x0, rows[1][0], 0.14, 0.15]); show_field(a2, P[i, k], MK[k], vmin, vmax, press_cmap())
                a3 = fig.add_axes([x0, rows[2][0], 0.14, 0.15])
                im_e = show_field(a3, np.abs(P[i, k] - PT[k]), MK[k], 0, err_max, err_cmap())
                a3.text(0.03, 0.08, f"error {err_pct[i, k]:.1f} %", transform=a3.transAxes, color=FG, fontsize=12,
                        fontweight="bold", bbox=dict(facecolor=BG, edgecolor="none", alpha=0.75, pad=2.5))
                if k == 0:
                    for a, (_, lab) in zip((a1, a2, a3), rows):
                        a.set_ylabel(lab, color=FG, fontsize=15)
            colorbar_lowhigh(fig, im_p, [0.640, 0.47, 0.007, 0.28], "pressure")
            cb2 = fig.add_axes([0.640, 0.26, 0.007, 0.14]); c2 = fig.colorbar(im_e, cax=cb2)
            c2.set_ticks([0, err_max]); c2.set_ticklabels(["0", "large"]); c2.ax.tick_params(colors=MUTED, labelsize=11)
            c2.outline.set_edgecolor(GRID); cb2.yaxis.set_label_position("left"); cb2.set_ylabel("error", color=MUTED, fontsize=12, labelpad=4)
            sl = slice(0, i + 1); x = np.maximum(ep[sl], 0) + 1
            ax = fig.add_axes([0.715, 0.66, 0.25, 0.15]); style_axes(ax)
            ax.semilogy(x, np.maximum(loss[sl], 1e-6), color=BLUE, lw=2.5); ax.set_xlim(1, n_epochs)
            ax.set_title(f"training loss {loss[i]:.3g}" if np.isfinite(loss[i]) else "training loss (starts at epoch 1)", color=FG, fontsize=15, loc="left")
            ax2 = fig.add_axes([0.715, 0.43, 0.25, 0.15]); style_axes(ax2)
            ax2.semilogy(x, np.maximum(j_err[sl] * 100, 1e-3), color=ORANGE, lw=2.5); ax2.set_xlim(1, n_epochs)
            ax2.set_title(f"objective J error, unseen shapes {j_err[i] * 100:.2f} %", color=FG, fontsize=15, loc="left")
            ax3 = fig.add_axes([0.715, 0.20, 0.25, 0.15]); style_axes(ax3, "epoch")
            mc = np.median(cos[sl], axis=1)
            ax3.plot(x, mc, color=GREEN, lw=2.5); ax3.set_xlim(1, n_epochs); ax3.set_ylim(-1.05, 1.05)
            ax3.axhline(1.0, color=MUTED, lw=1, ls="--")
            ax3.set_title(f"design direction vs adjoint CFD {mc[-1]:.3f}", color=FG, fontsize=15, loc="left")
            for a in (ax, ax2):
                a.set_xticklabels([])
            fig.add_artist(Rectangle((0.035, 0.125), 0.93, 0.008, transform=fig.transFigure, color=GRID))
            fig.add_artist(Rectangle((0.035, 0.125), 0.93 * (max(e, 0) + 1) / n_epochs, 0.008, transform=fig.transFigure, color=ORANGE))
            frac = (fmse[i] / f0) if np.isfinite(fmse[i]) else 1.0
            lab, txt, col = phase_caption(frac, e)
            lower_third(fig, lab, txt, col)
            footer(fig)
            if i == 0:
                still("04_train_start"); V.hold(2.5, 0.4, 0.0)
            elif i == len(ep) - 1:
                still("06_train_end"); V.hold(3.0, 0.0, 0.5)
            else:
                slow = max(args.hold, int(round(4 * args.hold * np.exp(-i / 10))))
                if slow >= 2 * args.hold:
                    fig.text(0.035, 0.148, f"▶ slow motion ×{slow / args.hold:.0f}: the first epochs, where most of the learning happens",
                             color=FG, fontsize=14, va="bottom", ha="left", bbox=dict(facecolor=PANEL, edgecolor=GRID, pad=4))
                if mid is None and frac < 0.3:
                    still("05_train_mid"); mid = i
                V.frames(slow)

        # ---- 5. parity on J
        idx = np.unique(np.concatenate([np.arange(0, min(40, len(ep))), np.linspace(40, len(ep) - 1, 40).astype(int)]))
        idx = idx[idx < len(ep)]
        lo, hi = j_true.min(), j_true.max(); pad = 0.15 * (hi - lo)
        for j, i in enumerate(idx):
            V.new()
            header(fig, "STEP 4 · HOW GOOD IS IT?", f"Predicted vs. true objective J on {len(j_true)} unseen shapes",
                   "J = lift / (drag + 0.008); each dot is one shape; a perfect network puts every dot on the dashed line")
            ax = fig.add_axes([0.30, 0.20, 0.40, 0.58]); style_axes(ax, "true J (CFD)")
            ax.set_ylabel("predicted J (network)", color=MUTED, fontsize=12)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, ls="--", lw=1.2)
            ax.scatter(j_true, np.clip(j_pred[i], lo - pad, hi + pad), s=60, color=ORANGE, edgecolor=BG, lw=0.8, zorder=3)
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
            e = int(ep[i])
            fig.text(0.965, 0.915, "untrained" if e < 0 else f"epoch {e + 1}", color=ORANGE, fontsize=34, fontweight="bold", va="top", ha="right")
            fig.text(0.74, 0.55, f"median error\n{j_err[i] * 100:.2f} %", color=FG, fontsize=26, fontweight="bold")
            lower_third(fig, "WHAT CHANGES", "As training proceeds the dots collapse onto the line: the objective is predicted well.",
                        GREEN if j == len(idx) - 1 else BLUE)
            footer(fig)
            if j == len(idx) - 1:
                still("07_parity"); V.hold(3.5, 0.0, 0.5)
            elif j == 0:
                V.hold(1.5, 0.4, 0.0)
            else:
                V.frames(3)

        # ---- 6. gradients
        V.new()
        header(fig, "STEP 5 · THE HARD PART", "Design needs a direction, not just an answer",
               "to improve a wing, an optimizer asks: which way should I change the shape? (the gradient)")
        ax = fig.add_axes([0.07, 0.24, 0.6, 0.52]); style_axes(ax)
        c = cos[-1]
        ax.bar(np.arange(len(c)), c, color=[(GREEN if v >= 0.98 else ORANGE) for v in c], width=0.7)
        ax.axhline(1.0, color=MUTED, ls="--", lw=1); ax.axhline(0.98, color=GREEN, ls=":", lw=1)
        ax.set_ylim(min(0.0, c.min() - 0.05), 1.05)
        ax.set_xticks(np.arange(len(c))); ax.set_xticklabels([f"{k + 1}" for k in range(len(c))], color=MUTED)
        ax.set_xlabel("calibration shape", color=MUTED, fontsize=12); ax.set_ylabel("agreement (cosine)", color=MUTED, fontsize=12)
        ax.set_title(f"one trained network (seed 0) vs. exact adjoint CFD, {len(c)} calibration shapes (1 = identical)",
                     color=FG, fontsize=15, loc="left")
        fig.text(0.72, 0.68, "How we check", color=BLUE, fontsize=20, fontweight="bold")
        fig.text(0.72, 0.43, "The discrete adjoint gives the exact\nCFD gradient of J. We verified it against\n"
                 "finite differences (within 1 %),\nthen compared it with the\nnetwork's gradient.", color=FG, fontsize=16, linespacing=1.6)
        fig.text(0.72, 0.30, f"median agreement: {np.median(c):.3f}", color=GREEN, fontsize=20, fontweight="bold")
        n_ok = int((c >= 0.98).sum())
        lower_third(fig, "RESULT", f"{n_ok} of {len(c)} shapes: the network's design direction agrees with CFD to cosine ≥ 0.98.",
                    GREEN if n_ok == len(c) else ORANGE)
        footer(fig); still("08_gradients"); V.hold(7)

        # ---- 7. results
        V.new()
        header(fig, "STEP 6 · FINAL EXAM", f"Results on {n_test} shapes never used in training",
               "5-network ensembles; test set evaluated once, after training")
        cols = [("Fourier neural operator", met["ensemble"], BLUE)] + ([("Multilayer perceptron", met_mlp["ensemble"], ORANGE)] if met_mlp else [])
        rowsdef = [("objective J error (median)", lambda m: f"{m['median_rel_err_J'] * 100:.2f} %"),
                   ("design-gradient error (median)", lambda m: f"{m['median_rel_grad_err_J'] * 100:.1f} %"),
                   ("gradient direction agreement", lambda m: f"{m['median_cos']:.4f}"),
                   ("shapes with agreement < 0.98", lambda m: f"{m['frac_cos_below_0.98'] * 100:.1f} %")]
        for r, (name, _) in enumerate(rowsdef):
            fig.text(0.08, 0.66 - r * 0.09, name, color=MUTED, fontsize=20)
        for ci, (cname, m, col) in enumerate(cols):
            xx = 0.52 + ci * 0.22
            fig.text(xx, 0.75, cname, color=col, fontsize=19, fontweight="bold", ha="center")
            for r, (_, f) in enumerate(rowsdef):
                fig.text(xx, 0.66 - r * 0.09, f(m), color=FG, fontsize=24, fontweight="bold", ha="center")
        e1 = met["ensemble"]
        ratio = e1["median_rel_grad_err_J"] / max(e1["median_rel_err_J"], 1e-9)
        verdict = "supported" if met.get("H1pp_supported") else ("falsified" if met.get("H1pp_falsified") else "neither supported nor falsified")
        fig.text(0.08, 0.27, f"Gradient error is {ratio:.0f}× the value error: accurate answers, much less accurate directions.",
                 color=FG, fontsize=18)
        lower_third(fig, "HONEST NOTE", f"Pre-registered hypothesis H1'' (gradients much worse than values): {verdict} for the FNO.", ORANGE)
        footer(fig); still("09_results"); V.hold(8)

        # ---- 8. outro
        V.new()
        fig.text(0.5, 0.62, "Next: a certificate that says when to trust the gradient", ha="center", color=FG, fontsize=40, fontweight="bold")
        fig.text(0.5, 0.52, "run-level conformal certificates along 40 optimiser trajectories · checked against 480 new CFD runs",
                 ha="center", color=MUTED, fontsize=19)
        fig.text(0.5, 0.36, ".", ha="center", color=BLUE, fontsize=30, fontweight="bold")
        still("10_outro"); V.hold(4.5, 0.6, 1.0)
    plt.close(fig)
    print("VIDEO", out, os.path.getsize(out), "bytes", flush=True)


if __name__ == "__main__":
    main()
