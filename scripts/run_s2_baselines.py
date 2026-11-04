"""DESCRIPTIVE baselines for the S2 trajectory study (NOT pre-registered; written 2026-10-14 before any trajectory
data exist). Uses exactly the same data selection, ensemble statistics and scores as scripts/run_s2_analysis.py
(imported, not copied), then reports on the test runs, for α ∈ {0.1, 0.2}:
  - gaussian (uncalibrated ensemble UQ), bonferroni (i.i.d. split conformal at α / run length),
  - weighted (Tibshirani et al. 2019, density ratio p_traj/p_iid from logistic regression fitted on the
    i.i.d. calibration designs vs the calibration-run iterates),
each with whole-run and per-iterate coverage and the median threshold (width cost).
Output: results/v2_traj/s2_baselines_descriptive.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import run_s2_analysis as A  # noqa: E402  (sets sys.path, ROOT, SPAN, design rules)
from gradient_certification.baselines import (aci_run_thresholds, bonferroni_runs, fit_density_ratio, gaussian_threshold,  # noqa: E402
                                               weighted_conformal_threshold)
from models.training import load_split  # noqa: E402


ACI_GAMMAS = (0.01, 0.05)   # fixed 2026-10-14 before any trajectory result was seen


def main():
    cal = load_split(A.PROC / "dataset_v2" / "cal", with_fields=False)
    tr = load_split(A.PROC / "dataset_v2_traj" / "traj", with_fields=False)
    tr, _ = A.add_raw_gradient_points(tr)   # same data selection as run_s2_analysis.py (§9 amendment 2026-10-14)
    okc = ~np.isnan(cal["g_ld"]).any(1); okt = ~np.isnan(tr["g_ld"]).any(1)
    run = np.array([int(c[4:6]) for c in tr["case_id"]]); step = np.array([int(c[8:10]) for c in tr["case_id"]])
    raw = A.ROOT / "datasets" / "raw" / "dataset_v2_traj"
    full = all((raw / f"T2_s{r:02d}_t{j:02d}" / "case.json").exists() for r in range(40) for j in range(A.N_IT))
    cal_runs, test_runs, n_it = (A.CAL_RUNS, A.TEST_RUNS, A.N_IT) if full else (A.RED_CAL, A.RED_TEST, A.RED_IT)
    okt = okt & np.isin(run, cal_runs + test_runs) & (step < n_it)
    G = A.member_grads({"cal": cal["theta"][okc], "traj": tr["theta"][okt]})
    ghc, sgc = A.stats(G["cal"]); ght, sgt = A.stats(G["traj"])
    floor = 1e-6 * float(np.median(sgc))
    s_cal = A.score(cal["g_ld"][okc], ghc, sgc, floor)
    s_tr = A.score(tr["g_ld"][okt], ght, sgt, floor)
    th_tr, run, step_t = tr["theta"][okt], run[okt], step[okt]
    in_cal = np.isin(run, cal_runs)
    wfun = fit_density_ratio(cal["theta"][okc], th_tr[in_cal], C=1.0, span=A.SPAN)
    w_cal = wfun(cal["theta"][okc])
    d = cal["g_ld"].shape[1]
    rep = {"design": "full 40x15" if full else "reduced 30x10", "note": "descriptive, not pre-registered", "alphas": {}}
    for a in A.ALPHAS:
        out = {}
        qg, qb = gaussian_threshold(d, a), bonferroni_runs(s_cal, a, n_it)
        for name, qfun in (("gaussian", lambda r, i: qg), ("bonferroni", lambda r, i: qb),
                           ("weighted", lambda r, i: weighted_conformal_threshold(s_cal, w_cal, float(wfun(th_tr[i:i + 1])[0]), a))):
            whole, point, qs = [], [], []
            for r in test_runs:
                idx = np.flatnonzero(run == r)
                if not len(idx):
                    continue
                q = np.array([qfun(r, i) for i in idx]); qs += list(q)
                cov = s_tr[idx] <= q
                whole.append(float(cov.all())); point.append(float(cov.mean()))
            out[name] = {"whole_run": float(np.mean(whole)), "whole_run_CI95": A.clopper_pearson(int(np.sum(whole)), len(whole)),
                         "per_iterate": float(np.mean(point)), "median_threshold": float(np.median(qs)), "n_test_runs": len(whole)}
        for gam in ACI_GAMMAS:   # adaptive conformal inference, sequential within each test run (descriptive)
            whole, point, qs = [], [], []
            for r in test_runs:
                idx = np.flatnonzero(run == r)
                if not len(idx):
                    continue
                idx = idx[np.argsort(step_t[idx])]
                q = aci_run_thresholds(s_cal, s_tr[idx], a, gam); qs += list(q)
                cov = s_tr[idx] <= q
                whole.append(float(cov.all())); point.append(float(cov.mean()))
            out[f"aci_gamma{gam}"] = {"whole_run": float(np.mean(whole)), "whole_run_CI95": A.clopper_pearson(int(np.sum(whole)), len(whole)),
                                      "per_iterate": float(np.mean(point)), "median_threshold": float(np.median(qs)), "n_test_runs": len(whole),
                                      "note": "needs a CFD gradient after every step"}
        rep["alphas"][str(a)] = out
    out = A.ROOT / "results" / "v2_traj" / "s2_baselines_descriptive.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["alphas"], default=float))


if __name__ == "__main__":
    sys.exit(main())
