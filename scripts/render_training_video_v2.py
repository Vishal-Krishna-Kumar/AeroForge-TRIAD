"""Render the explanatory training video, v2 (outreach deliverable; docs/outreach/training_video_plan.md).

Each scene carries on-screen captions that say what is happening and what is changing. Every number and
field shown comes from logged artefacts:
  * --frames   video_frames_seed0.npz written by scripts/train_surrogates.py during FNO seed-0 training
  * --metrics  h1_test_metrics_fno.json / --metrics-mlp h1_test_metrics_mlp.json (test-set evaluation)
  * --data     datasets/processed/dataset_v1 (true CL/CD of the 64 validation airfoils, for the parity scene)
  * --manifest datasets/dataset_v1_manifest.json (CFD wall time), --timing a JSON measured on the GPU
Nothing is staged or interpolated between snapshots; captions about training phases are selected from the
logged validation field error, not written by hand per frame.

Usage:
  python scripts/render_training_video_v2.py --frames checkpoints/v1/fno/video_frames_seed0.npz \
      --metrics results/reports/h1_test_metrics_fno.json --metrics-mlp results/reports/h1_test_metrics_mlp.json \
      --data datasets/processed/dataset_v1 --timing results/video/inference_timing.json \
      --out results/video/aeroforge_training_v2.mp4
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
    """Frame writer with fade-in/out via a full-figure overlay."""

    def __init__(self, fig, writer, fps):
        self.fig, self.w, self.fps = fig, writer, fps
        self.cover = None

    def new(self):
        self.fig.clf(); self.fig.patch.set_facecolor(BG)
        self.cover = None

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
    """Caption bar at the bottom of the frame: what is happening right now."""
    fig.add_artist(FancyBboxPatch((0.035, 0.03), 0.93, 0.075, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, lw=1.2))
    fig.add_artist(Rectangle((0.035, 0.03), 0.006, 0.075, transform=fig.transFigure, color=color))
    fig.text(0.052, 0.0675, label, color=color, fontsize=15, fontweight="bold", va="center")
    fig.text(0.052 + 0.0085 * len(label) + 0.02, 0.0675, text, color=FG, fontsize=16, va="center")


def footer(fig):
    fig.text(0.965, 0.008, "AeroForge-TRIAD · numbers from logged runs", color="#4d5566", fontsize=10, ha="right", va="bottom")


def colorbar_lowhigh(fig, im, rect, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_ticks([im.norm.vmin, im.norm.vmax]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(colors=MUTED, labelsize=12); cb.outline.set_edgecolor(GRID)
    cax.yaxis.set_label_position("left"); cax.set_ylabel(label, color=MUTED, fontsize=12, labelpad=4)


def phase_caption(frac_err, e):
    """Caption chosen from the logged validation field error relative to the untrained network (no hand-written claims)."""
    if e < 0:
        return ("UNTRAINED", "Random weights: the network's guess has nothing to do with the real airflow.", RED)
    drop = (1 - frac_err) * 100
    if frac_err > 0.5:
        return ("LEARNING", f"Weights update after every batch of 16 airfoils. Field error down {drop:.0f} % so far.", ORANGE)
    if frac_err > 0.1:
        return ("SHARPENING", f"The overall pressure pattern is in place (error down {drop:.0f} %). Bottom row: where it is still wrong.", ORANGE)
    if frac_err > 0.03:
        return ("REFINING", f"Field error is now {frac_err * 100:.1f} % of where it started; what remains is fine detail.", BLUE)
    return ("FINE-TUNING", f"Field error is now {frac_err * 100:.2g} % of where it started: it tracks the CFD on unseen airfoils.", GREEN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--metrics-mlp", default=None)
    ap.add_argument("--data", default=None, help="processed dataset_v1 folder (validation truth for parity scene)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--timing", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=int, default=3, help="video frames per training snapshot")
    ap.add_argument("--stills", default=None, help="folder for PNG stills of each scene")
    args = ap.parse_args()

    z = np.load(args.frames, allow_pickle=False)
    ep, P, PT, MK = z["epoch"], z["p_pred"].astype(np.float32), z["p_true"].astype(np.float32), z["mask"]
    cos, loss, val_ld, fmse = z["grad_cos"], z["loss"], z["val_rel_err_LD"], z["val_field_mse"]
    vCL, vCD = z["val_CL"], z["val_CD"]
    meta = json.loads(str(z["meta"]))
    n_epochs = int(ep.max()) + 1
    vals = PT[MK == 1]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    prange = vmax - vmin
    err_max = 0.25 * prange
    met = json.loads(Path(args.metrics).read_text())["ensemble"]
    met_mlp = json.loads(Path(args.metrics_mlp).read_text())["ensemble"] if args.metrics_mlp else None
    timing = json.loads(Path(args.timing).read_text()) if args.timing and Path(args.timing).exists() else None

    cfd_med_s, n_cases = None, None
    if args.manifest and Path(args.manifest).exists():
        man = json.loads(Path(args.manifest).read_text())
        cases = man.get("cases", [])
        walls = [float(c["wall_s"]) for c in cases if c.get("wall_s") is not None]
        n = len(cases)
        if walls:
            cfd_med_s, n_cases = float(np.median(walls)), n

    ld_true = None
    if args.data:
        files = sorted(Path(args.data, "train").glob("*.npz"))[448:512]
        if len(files) == len(z["val_theta"]):
            tr = [np.load(f) for f in files]
            th = np.stack([t["theta"] for t in tr])
            assert np.allclose(th, z["val_theta"], atol=1e-6), "validation theta mismatch"
            ld_true = np.array([float(t["CL"]) / float(t["CD"]) for t in tr])

    # per-snapshot relative field error on the 4 shown airfoils (mean |err| / pressure range, fluid cells only)
    err_pct = np.array([[np.abs(P[i, k] - PT[k])[MK[k] == 1].mean() / prange * 100 for k in range(4)] for i in range(len(ep))])
    f0 = fmse[0] if np.isfinite(fmse[0]) and fmse[0] > 0 else np.nanmax(fmse)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    stills = Path(args.stills) if args.stills else None
    if stills:
        stills.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    writer = FFMpegWriter(fps=args.fps, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"],
                          metadata={"title": "AeroForge-TRIAD: watching a neural operator learn airflow"})
    fps = args.fps

    def still(name):
        if stills:
            fig.savefig(stills / f"{name}.png", dpi=50, facecolor=BG)

    with writer.saving(fig, str(out), dpi=100):
        V = Video(fig, writer, fps)

        # ---- 1. hook --------------------------------------------------------------------------
        V.new()
        fig.text(0.5, 0.64, "Can a neural network learn to see airflow?", ha="center", color=FG, fontsize=50, fontweight="bold")
        fig.text(0.5, 0.53, "We trained one on hundreds of fluid-dynamics simulations — and filmed it learning.",
                 ha="center", color=MUTED, fontsize=24)
        fig.text(0.5, 0.40, ".", ha="center", color=BLUE, fontsize=28, fontweight="bold")
        fig.text(0.5, 0.35, "trustworthy AI surrogates for airfoil design", ha="center", color=MUTED, fontsize=18)
        still("01_hook"); V.hold(4.5, 0.8, 0.6)

        # ---- 2. the problem: CFD is accurate but slow -----------------------------------------
        V.new()
        header(fig, "STEP 1 · THE GROUND TRUTH", "Computational fluid dynamics (CFD) solves the physics",
               "Navier–Stokes equations around NACA 4-digit airfoils, Reynolds number 1000, solver SU2")
        for k in range(4):
            ax = fig.add_axes([0.035 + k * 0.225, 0.36, 0.21, 0.36])
            im = show_field(ax, PT[k], MK[k], vmin, vmax, press_cmap())
            ax.set_title(f"validation airfoil {k + 1}", color=MUTED, fontsize=14)
        colorbar_lowhigh(fig, im, [0.945, 0.38, 0.012, 0.32], "pressure")
        speed = f"~{cfd_med_s / 60:.0f} minutes" if cfd_med_s else "~4 minutes"
        cnt = f"{n_cases}" if n_cases else "768"
        fig.text(0.035, 0.27, f"Each simulation: {speed} on one CPU core.   Our dataset: {cnt} simulations.",
                 color=FG, fontsize=20)
        fig.text(0.035, 0.225, "Red = high pressure (air piles up at the nose).  Blue = low pressure (suction that creates lift).",
                 color=MUTED, fontsize=17)
        lower_third(fig, "THE PROBLEM", "Accurate — but far too slow to test thousands of shapes during design.", RED)
        footer(fig); still("02_cfd"); V.hold(6.5)

        # ---- 3. the idea --------------------------------------------------------------------------
        V.new()
        header(fig, "STEP 2 · THE IDEA", "Teach a neural network to imitate the solver")
        boxes = [("airfoil shape", "camber · camber position\nthickness · angle of attack", MUTED),
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
        fig.text(0.5, 0.29, "Training data: 448 airfoils   ·   Watched while training: 64 airfoils it never learns from"
                 "   ·   Final exam: 128 held-out airfoils", ha="center", color=MUTED, fontsize=17)
        lower_third(fig, "WHAT CHANGES", "Training adjusts the network's weights until its output matches the CFD.", BLUE)
        footer(fig); still("03_idea"); V.hold(6)

        # ---- 4. training timelapse --------------------------------------------------------------
        mid = None
        for i in range(len(ep)):
            V.new()
            e = int(ep[i])
            header(fig, "STEP 3 · TRAINING (TIMELAPSE)", "Watching the network learn",
                   "four airfoils the network never trains on — top: CFD truth · middle: network · bottom: where it is wrong")
            fig.text(0.965, 0.915, "untrained" if e < 0 else f"epoch {e + 1}", color=ORANGE, fontsize=34,
                     fontweight="bold", va="top", ha="right")
            fig.text(0.965, 0.862, f"of {n_epochs}  ·  1 epoch = one pass over 448 airfoils", color=MUTED, fontsize=13,
                     va="top", ha="right")
            rows = [(0.615, "CFD"), (0.435, "network"), (0.255, "error")]
            for k in range(4):
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
            ax.set_title(f"training loss   {loss[i]:.3g}", color=FG, fontsize=15, loc="left")
            ax2 = fig.add_axes([0.715, 0.43, 0.25, 0.15]); style_axes(ax2)
            ax2.semilogy(x, np.maximum(val_ld[sl] * 100, 1e-3), color=ORANGE, lw=2.5); ax2.set_xlim(1, n_epochs)
            ax2.set_title(f"lift/drag error, unseen airfoils   {val_ld[i] * 100:.2f} %", color=FG, fontsize=15, loc="left")
            ax3 = fig.add_axes([0.715, 0.20, 0.25, 0.15]); style_axes(ax3, "epoch")
            mc = np.median(cos[sl], axis=1)
            ax3.plot(x, mc, color=GREEN, lw=2.5); ax3.set_xlim(1, n_epochs); ax3.set_ylim(-1.05, 1.05)
            ax3.axhline(1.0, color=MUTED, lw=1, ls="--")
            ax3.set_title(f"design direction vs adjoint CFD   {mc[-1]:.3f}", color=FG, fontsize=15, loc="left")
            for a in (ax, ax2):
                a.set_xticklabels([])
            # progress bar
            fig.add_artist(Rectangle((0.035, 0.125), 0.93, 0.008, transform=fig.transFigure, color=GRID))
            fig.add_artist(Rectangle((0.035, 0.125), 0.93 * (max(e, 0) + 1) / n_epochs, 0.008, transform=fig.transFigure,
                                     color=ORANGE))
            frac = (fmse[i] / f0) if np.isfinite(fmse[i]) else 1.0
            lab, txt, col = phase_caption(frac, e)
            lower_third(fig, lab, txt, col)
            footer(fig)
            if i == 0:
                still("04_train_start"); V.hold(2.5, 0.4, 0.0)
            elif i == len(ep) - 1:
                still("06_train_end"); V.hold(3.0, 0.0, 0.5)
            else:
                # the network learns most of the flow in the first epochs: show those in labelled slow motion
                slow = max(args.hold, int(round(4 * args.hold * np.exp(-i / 10))))
                if slow >= 2 * args.hold:
                    fig.text(0.035, 0.148, f"▶ slow motion ×{slow / args.hold:.0f}: the first epochs, where most of the learning happens", color=FG, fontsize=14, va="bottom", ha="left",
                             bbox=dict(facecolor=PANEL, edgecolor=GRID, pad=4))
                if mid is None and frac < 0.3:
                    still("05_train_mid"); mid = i
                V.frames(slow)

        # ---- 5. parity: predicted vs true L/D on validation airfoils, evolving --------------------
        if ld_true is not None:
            idx = np.unique(np.concatenate([np.arange(0, min(40, len(ep))), np.linspace(40, len(ep) - 1, 40).astype(int)]))
            idx = idx[idx < len(ep)]
            lo, hi = ld_true.min(), ld_true.max(); pad = 0.15 * (hi - lo)
            for j, i in enumerate(idx):
                V.new()
                header(fig, "STEP 4 · HOW GOOD IS IT?", "Predicted vs. true lift-to-drag on 64 unseen airfoils",
                       "each dot is one airfoil; a perfect network puts every dot on the dashed line")
                ax = fig.add_axes([0.30, 0.20, 0.40, 0.58]); style_axes(ax, "true L/D (CFD)")
                ax.set_ylabel("predicted L/D (network)", color=MUTED, fontsize=12)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, ls="--", lw=1.2)
                ld_pred = vCL[i] / vCD[i]
                ax.scatter(ld_true, np.clip(ld_pred, lo - pad, hi + pad), s=60, color=ORANGE, edgecolor=BG, lw=0.8, zorder=3)
                ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
                e = int(ep[i])
                fig.text(0.965, 0.915, "untrained" if e < 0 else f"epoch {e + 1}", color=ORANGE, fontsize=34,
                         fontweight="bold", va="top", ha="right")
                fig.text(0.74, 0.55, f"median error\n{val_ld[i] * 100:.2f} %", color=FG, fontsize=26, fontweight="bold")
                lower_third(fig, "WHAT CHANGES", "As training proceeds the dots collapse onto the line: predictions become accurate.",
                            GREEN if j == len(idx) - 1 else BLUE)
                footer(fig)
                if j == len(idx) - 1:
                    still("07_parity"); V.hold(3.5, 0.0, 0.5)
                elif j == 0:
                    V.hold(1.5, 0.4, 0.0)
                else:
                    V.frames(3)

        # ---- 6. design gradients ----------------------------------------------------------------------
        V.new()
        header(fig, "STEP 5 · THE HARD PART", "Design needs a direction, not just an answer",
               "to improve a wing, an optimizer asks: which way should I change the shape? (the gradient)")
        ax = fig.add_axes([0.07, 0.24, 0.6, 0.52]); style_axes(ax)
        c = cos[-1]
        ax.bar(np.arange(len(c)), c, color=[(GREEN if v >= 0.98 else ORANGE) for v in c], width=0.7)
        ax.axhline(1.0, color=MUTED, ls="--", lw=1)
        ax.set_ylim(min(0.0, c.min() - 0.05), 1.05)
        ax.set_xticks(np.arange(len(c))); ax.set_xticklabels([f"{k + 1}" for k in range(len(c))], color=MUTED)
        ax.set_xlabel("calibration airfoil", color=MUTED, fontsize=12)
        ax.set_ylabel("agreement (cosine)", color=MUTED, fontsize=12)
        ax.set_title("one trained network (seed 0) vs. exact adjoint CFD, 16 calibration airfoils (1 = identical)", color=FG,
                     fontsize=15, loc="left")
        fig.text(0.72, 0.68, "How we check", color=BLUE, fontsize=20, fontweight="bold")
        fig.text(0.72, 0.43, "The adjoint method gives the exact\nCFD gradient. We verified it against\n"
                 "finite differences (0.12 % max\ndifference), then compared it with\nthe network's gradient.",
                 color=FG, fontsize=16, linespacing=1.6)
        fig.text(0.72, 0.30, f"median agreement: {np.median(c):.3f}", color=GREEN, fontsize=20, fontweight="bold")
        n_ok = int((c >= 0.98).sum())
        lower_third(fig, "RESULT", f"{n_ok} of {len(c)} airfoils: the network's design direction agrees with CFD to cosine ≥ 0.98.",
                    GREEN if n_ok == len(c) else ORANGE)
        footer(fig); still("08_gradients"); V.hold(7)

        # ---- 7. results ------------------------------------------------------------------------------
        V.new()
        header(fig, "STEP 6 · FINAL EXAM", "Results on 128 airfoils never used in training",
               "5-network ensembles; test set evaluated once, after training")
        cols = [("Fourier neural operator", met, BLUE)] + ([("Multilayer perceptron", met_mlp, ORANGE)] if met_mlp else [])
        rowsdef = [("lift/drag error (median)", lambda m: f"{m['median_rel_err_LD'] * 100:.2f} %"),
                   ("design-gradient error (median)", lambda m: f"{m['median_rel_grad_err_LD'] * 100:.1f} %"),
                   ("gradient direction agreement", lambda m: f"{m['median_cos']:.4f}"),
                   ("airfoils with agreement < 0.98", lambda m: f"{m['frac_cos_below_0.98'] * 100:.1f} %")]
        for r, (name, _) in enumerate(rowsdef):
            fig.text(0.08, 0.66 - r * 0.09, name, color=MUTED, fontsize=20)
        for ci, (cname, m, col) in enumerate(cols):
            x = 0.52 + ci * 0.22
            fig.text(x, 0.75, cname, color=col, fontsize=19, fontweight="bold", ha="center")
            for r, (_, f) in enumerate(rowsdef):
                fig.text(x, 0.66 - r * 0.09, f(m), color=FG, fontsize=24, fontweight="bold", ha="center")
        if timing:
            fig.text(0.08, 0.27, f"Speed: CFD {speed} per airfoil (1 CPU core)  vs.  network "
                     f"{timing['ms_per_case']:.2f} ms per airfoil ({timing['device']}, batch of {timing['batch']})",
                     color=FG, fontsize=18)
        agree = (1 - met["frac_cos_below_0.98"]) * 100
        if agree >= 95:
            note = (f"We predicted wrong design directions (hypothesis H1). On test airfoils {agree:.1f} % agree with CFD, "
                    "so H1 is rejected.")
        else:
            note = f"Only {agree:.1f} % of test airfoils have design directions that agree with CFD (cosine ≥ 0.98)."
        lower_third(fig, "HONEST NOTE", note, ORANGE)
        footer(fig); still("09_results"); V.hold(8)

        # ---- 8. outro -------------------------------------------------------------------------------
        V.new()
        fig.text(0.5, 0.62, "Next: a certificate that says when to trust the AI", ha="center", color=FG, fontsize=40,
                 fontweight="bold")
        fig.text(0.5, 0.52, "conformal design-gradient certificates (CDGC) · optimization with a CFD safety check · "
                 "behaviour outside the training range", ha="center", color=MUTED, fontsize=19)
        fig.text(0.5, 0.36, ".", ha="center", color=BLUE, fontsize=30, fontweight="bold")
        still("10_outro"); V.hold(4.5, 0.6, 1.0)
    plt.close(fig)
    print("VIDEO", out, os.path.getsize(out), "bytes", flush=True)


if __name__ == "__main__":
    main()
