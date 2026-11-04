"""EXPLORATORY (post hoc, not pre-registered): run-level calibration variants on the H4 trajectory points.

Written 2026-10-13 AFTER the H2-trajectory/H4 results were known. Nothing here changes any pre-registered verdict;
it only checks the theory draft (docs/theory/run_level_certificates.md) on existing data and sizes the next study.

Unit = start index (all gates from one start share their first point; 10 starts). Variants, each calibrated on
the other starts and tested on a held-out start:
  pooled      : every point of the calibration starts pooled (what results/h4 did; no finite-sample guarantee
                for a new run because points within a run are dependent and runs have different lengths)
  subsample   : one uniformly drawn point per calibration start, split conformal on those R scores
                (exact 1-alpha for a uniformly drawn point of a new run); averaged over draws
  runmax      : per-start max score, split conformal (exact 1-alpha that ALL points of a new run are covered)
Coverage of a held-out start is reported two ways: pointwise (fraction of its points covered, then averaged over
starts so each start has weight 1) and simultaneous (1 if all its points are covered).

Usage: python scripts/explore_runlevel_calibration.py <h4_trajectory_points.npz> [out.json]
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

SPAN = np.array([0.04, 0.4, 0.06, 6.0])  # = models.training.THETA_SPAN (Dataset V1)
ALPHAS = (0.1, 0.2)


def cq(scores, alpha):
    s = np.sort(np.asarray(scores, float)); k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else float(s[k - 1])


def scores(z):
    d = (z["g_true"] - z["g_hat"].astype(float)) * SPAN      # M = diag(span^2) -> ||v||_M = ||v * span||
    return np.linalg.norm(d, axis=1) / z["sigma"]


def evaluate(s, starts, cal, test, alpha, rng, draws=2000):
    by = {k: s[starts == k] for k in np.unique(starts)}
    out = {}
    q = cq(np.concatenate([by[k] for k in cal]), alpha)
    out["pooled"] = q
    q = cq([by[k].max() for k in cal], alpha)
    out["runmax"] = q
    qs = [cq([rng.choice(by[k]) for k in cal], alpha) for _ in range(draws)]
    out["subsample"] = qs
    res = {}
    for name, q in out.items():
        qv = np.atleast_1d(q)
        point = np.mean([np.mean([(by[k] <= qq).mean() for qq in qv]) for k in test])
        simul = np.mean([np.mean([float((by[k] <= qq).all()) for qq in qv]) for k in test])
        fin = qv[np.isfinite(qv)]
        res[name] = {"pointwise": float(point), "simultaneous": float(simul),
                     "qhat_median": float(np.median(fin)) if len(fin) else math.inf,
                     "frac_infinite_qhat": float(np.mean(~np.isfinite(qv)))}
    return res


def main():
    z = np.load(sys.argv[1]); s = scores(z); starts = z["starts"]
    rng = np.random.default_rng(20261013)
    u = sorted(np.unique(starts).tolist())
    rep = {"n_points": int(len(s)), "points_per_start": {int(k): int((starts == k).sum()) for k in u},
           "median_score": float(np.median(s)), "label": "EXPLORATORY post hoc analysis; not pre-registered"}
    for a in ALPHAS:
        halves = {f"{A[0]}-{A[-1]}->{B[0]}-{B[-1]}": evaluate(s, starts, A, B, a, rng)
                  for A, B in ((u[:5], u[5:]), (u[5:], u[:5]))}
        loso = [evaluate(s, starts, [k for k in u if k != t], [t], a, rng, draws=500) for t in u]
        agg = {m: {f: float(np.mean([r[m][f] for r in loso])) for f in ("pointwise", "simultaneous", "frac_infinite_qhat")}
               for m in loso[0]}
        rep[f"alpha={a}"] = {"halves": halves, "leave_one_start_out_mean": agg}
    txt = json.dumps(rep, indent=1, default=float)
    print(txt)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(txt)


if __name__ == "__main__":
    main()
