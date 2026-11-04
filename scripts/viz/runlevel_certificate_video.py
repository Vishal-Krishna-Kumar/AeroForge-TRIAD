"""Animated 2-D toy illustration of run-level conformal certificates for surrogate gradients (AeroForge-TRIAD).

NOT CFD and not a result of the paper: a toy built to show the mechanism. Smooth objective J with three bumps; the
surrogate gradient has a smooth error field whose size grows near the optima (where optimisers go and where training
data are thin), and an uncertainty scale sigma that captures that growth only partly. An optimiser follows the
surrogate gradient (normalised projected ascent, h = 0.05, 15 steps). Two certificates (alpha = 0.1):
  i.i.d.     q from 600 uniform design points             -> covers single points, not whole runs
  run-level  q from the MAX score over each of 40 runs    -> Theorem 1: P(all iterates of a new run covered) >= 0.9
At each iterate the disk of radius q*sigma around the surrogate gradient is drawn; the step is covered if the true
gradient lies inside it. Coverage is the mean over 200 independent calibration/test draws (200 test runs each), the
quantity Theorem 1 bounds; the demo run comes from the first draw.
Usage: python runlevel_certificate_video.py OUT.mp4 [--seed 3]
"""
from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

BLUE, ORANGE, GRAY, RED, GREEN = "#2a78d6", "#eb6834", "#a3a29d", "#e34948", "#1baf7a"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BUMPS = np.array([[0.30, 0.70, 1.0, 0.18], [0.72, 0.62, 0.8, 0.14], [0.55, 0.25, 0.6, 0.20]])


def J(u):
    u = np.atleast_2d(u)
    return sum(a * np.exp(-((u[:, 0] - x) ** 2 + (u[:, 1] - y) ** 2) / (2 * s * s)) for x, y, a, s in BUMPS)


def grad(u):
    g = np.zeros(2)
    for x, y, a, s in BUMPS:
        e = a * math.exp(-((u[0] - x) ** 2 + (u[1] - y) ** 2) / (2 * s * s))
        g += e * np.array([-(u[0] - x), -(u[1] - y)]) / (s * s)
    return g


def field(u):
    return 0.9 * np.array([math.sin(7 * u[0] + 1) * math.cos(5 * u[1]), math.cos(6 * u[0]) * math.sin(8 * u[1] + 2)])


def err(u):
    return (0.3 + 1.2 * float(J(u)[0])) * field(u)


def sigma(u):
    return 0.35 * np.linalg.norm(field(u)) * (0.3 + 0.4 * float(J(u)[0])) + 0.12


def surrogate(u):
    return grad(u) + err(u)


def run(u0, T=15, h=0.05):
    us = [np.array(u0, float)]
    for _ in range(T - 1):
        gh = surrogate(us[-1])
        us.append(np.clip(us[-1] + h * gh / max(np.linalg.norm(gh), 1e-12), 0, 1))
    return np.array(us)


def score(u):
    return np.linalg.norm(grad(u) - surrogate(u)) / sigma(u)


def quantile(s, alpha=0.1):
    s = np.sort(s)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return s[k - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    s_pt = np.array([score(u) for u in rng.uniform(0, 1, (20000, 2))])  # pool of i.i.d. design points
    runs = [run(u0) for u0 in rng.uniform(0, 1, (4000, 2))]  # pool of optimiser runs
    S = np.array([[score(u) for u in r] for r in runs])
    cov = {"iid": [], "run": [], "pt_iid": [], "q_iid": [], "q_run": []}
    for rep in range(200):
        q_i = quantile(s_pt[rng.choice(len(s_pt), 600, replace=False)])
        perm = rng.permutation(len(runs))
        cal, te = perm[:40], perm[40:240]
        q_r = quantile(S[cal].max(1))
        for k, v in (("iid", np.mean(S[te].max(1) <= q_i)), ("run", np.mean(S[te].max(1) <= q_r)),
                     ("pt_iid", np.mean(S[te] <= q_i)), ("q_iid", q_i), ("q_run", q_r)):
            cov[k].append(v)
        if rep == 0:
            q_iid, q_run, tests = q_i, q_r, [runs[j] for j in te]
    cov_iid, cov_run, pt_iid = (float(np.mean(cov[k])) for k in ("iid", "run", "pt_iid"))
    # demo run: covered by the run-level certificate and with the most i.i.d. misses (chosen by rule, shown as such)
    demo = max(tests, key=lambda r: (all(score(u) <= q_run for u in r), sum(score(u) > q_iid for u in r)))
    print(f"draw0 q_iid {q_iid:.3f} q_run {q_run:.3f}; mean over 200 draws: whole-run iid {cov_iid:.3f} "
          f"(sd {np.std(cov['iid']):.3f}) run {cov_run:.3f} (sd {np.std(cov['run']):.3f}) pointwise iid {pt_iid:.3f}; "
          f"mean q_iid {np.mean(cov['q_iid']):.3f} q_run {np.mean(cov['q_run']):.3f}")

    xs = np.linspace(0, 1, 160)
    X, Y = np.meshgrid(xs, xs)
    Z = J(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.patch.set_facecolor(SURF)
    w = FFMpegWriter(fps=12, bitrate=6000)

    def header(sub):
        fig.text(0.5, 0.965, "Can an optimiser trust its surrogate's gradient for a WHOLE run?", ha="center",
                 fontsize=19, color=INK, weight="bold")
        fig.text(0.5, 0.935, sub, ha="center", fontsize=12.5, color=INK2)
        fig.text(0.5, 0.012, "2-D toy illustration (not CFD) · AeroForge-TRIAD · run-level conformal certificates",
                 ha="center", fontsize=10, color=INK2)

    with w.saving(fig, a.out, dpi=100):
        for t_total in range(15 * 8 + 1):
            t = min(t_total // 8, 14)
            fig.clf()
            header("Surrogate gradient ĝ (orange) vs true gradient g (blue) at each optimiser step, α = 0.1")
            ax = fig.add_axes([0.05, 0.30, 0.46, 0.58])
            bx = fig.add_axes([0.56, 0.30, 0.40, 0.58])
            for z in (ax, bx):
                z.set_facecolor(SURF)
            ax.contourf(X, Y, Z, levels=18, cmap="Blues", alpha=0.55)
            ax.plot(demo[: t + 1, 0], demo[: t + 1, 1], "-", color=INK2, lw=1.4)
            for i in range(t + 1):
                bad = score(demo[i]) > q_iid
                ax.plot(*demo[i], "o", ms=7, color=RED if bad else GREEN, mec=SURF, mew=1.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("design space: objective J and the optimiser's path\n(red = step NOT covered by the i.i.d. certificate)",
                         fontsize=11, color=INK)
            u = demo[t]
            g, gh, s = grad(u), surrogate(u), sigma(u)
            R = max(q_run * s, np.linalg.norm(gh), np.linalg.norm(g)) * 1.25
            bx.add_patch(Circle(gh, q_run * s, fill=False, ec=BLUE, lw=2.2, label=f"run-level disk (q = {q_run:.2f})"))
            bx.add_patch(Circle(gh, q_iid * s, fill=False, ec=GRAY, lw=2, ls="--", label=f"i.i.d. disk (q = {q_iid:.2f})"))
            bx.annotate("", gh, (0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
            bx.annotate("", g, (0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5))
            bx.text(*gh, "  ĝ", color=ORANGE, fontsize=13, weight="bold")
            bx.text(*g, "  g", color=BLUE, fontsize=13, weight="bold")
            bx.set_xlim(-R, R); bx.set_ylim(-R, R); bx.set_aspect("equal"); bx.axhline(0, color="#e6e5e1"); bx.axvline(0, color="#e6e5e1")
            bx.set_xticks([]); bx.set_yticks([])
            bx.legend(loc="lower left", fontsize=10, frameon=False)
            bx.set_title(f"gradient space at step {t + 1}/15", fontsize=11, color=INK)
            n_iid = sum(score(v) <= q_iid for v in demo[: t + 1])
            fig.text(0.07, 0.22, f"i.i.d. certificate: {n_iid}/{t + 1} steps covered", fontsize=15, color=RED if n_iid < t + 1 else INK)
            fig.text(0.07, 0.18, f"run-level certificate: {t + 1}/{t + 1} steps covered", fontsize=15, color=BLUE)
            fig.text(0.07, 0.11, "Run-level calibration uses the WORST step of each calibration run,\nso its guarantee covers every step of a new run (Theorem 1).",
                     fontsize=12, color=INK2)
            w.grab_frame()
        for f in range(48):
            fig.clf()
            header("Whole-run coverage of new runs, mean over 200 calibration/test draws (target ≥ 0.90)")
            ax = fig.add_axes([0.12, 0.25, 0.80, 0.55])
            ax.set_facecolor(SURF)
            grow = min(1.0, (f + 1) / 24)
            vals = [cov_iid * grow, cov_run * grow]
            ax.bar([0, 1], vals, color=[GRAY, BLUE], width=0.55)
            ax.axhline(0.9, color=INK, ls="--", lw=1.2)
            ax.text(0.5, 0.915, "target 0.90", ha="center", fontsize=12, color=INK)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=18, color=INK, weight="bold")
            ax.set_xticks([0, 1], ["i.i.d. calibration", "run-level (run-max)"], fontsize=13)
            ax.set_ylim(0, 1.08); ax.set_ylabel("fraction of runs with EVERY step covered", fontsize=12, color=INK2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.text(0.12, 0.12, f"i.i.d. calibration covers {pt_iid:.0%} of single steps but only {cov_iid:.0%} of whole runs:\n"
                                 "runs end near optima, where this surrogate is least accurate, and get 15 chances to miss.",
                     fontsize=12.5, color=INK2)
            w.grab_frame()
    plt.close(fig)
    print("VIDEO", a.out)


if __name__ == "__main__":
    main()
