"""DESCRIPTIVE (not pre-registered; written 2026-10-14 after H5-H7 were computed, question raised by the width cost):
how informative are the trajectory certificates? For the S2 test runs (20-39, FNO ensemble, same data selection,
ensemble statistics and scores as scripts/run_s2_analysis.py — imported, not copied) and each threshold q̂ already
stored in results/v2_traj/s2_h5_h6_h7.json (i.i.d. / pooled / run-max / random-iterate median), report:
  - frac_steps_certified: steepest-ascent step d = M ĝ is certified iff ||ĝ||_M > q̂ σ̂ (same rule as H2′),
  - frac_certified_steps_true_ascent: among certified steps, CFD gradient g has gᵀ M ĝ > 0,
  - frac_certified_steps_covered: among certified steps, the score is inside q̂ (certificate actually valid),
  - frac_runs_all_steps_certified, median steps certified per run, median radius / ||ĝ||_M,
  - frac_all_steps_true_ascent (independent of q̂).
Output: results/v2_traj/s2_certified_fraction_descriptive.json (refuses to overwrite an existing file).
"""
from __future__ import annotations

import json
import sys

import numpy as np

import run_s2_analysis as A  # noqa: E402  (sets sys.path, ROOT, SPAN)
from models.training import load_split  # noqa: E402

OUT = A.ROOT / "results" / "v2_traj" / "s2_certified_fraction_descriptive.json"


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; not overwritten")
    rep5 = json.loads((A.ROOT / "results" / "v2_traj" / "s2_h5_h6_h7.json").read_text())
    assert rep5["design"].startswith("full"), rep5["design"]
    cal = load_split(A.PROC / "dataset_v2" / "cal", with_fields=False)
    tr = load_split(A.PROC / "dataset_v2_traj" / "traj", with_fields=False)
    tr, fallback = A.add_raw_gradient_points(tr)
    okc = ~np.isnan(cal["g_ld"]).any(1); okt = ~np.isnan(tr["g_ld"]).any(1)
    run = np.array([int(c[4:6]) for c in tr["case_id"]]); step = np.array([int(c[8:10]) for c in tr["case_id"]])
    okt = okt & np.isin(run, A.CAL_RUNS + A.TEST_RUNS) & (step < A.N_IT)
    G = A.member_grads({"cal": cal["theta"][okc], "traj": tr["theta"][okt]})
    ghc, sgc = A.stats(G["cal"]); ght, sgt = A.stats(G["traj"])
    floor = 1e-6 * float(np.median(sgc))
    gt = tr["g_ld"][okt]
    s_tr = A.score(gt, ght, sgt, floor)
    assert abs(float(np.median(s_tr)) - rep5["median_score_traj"]) < 1e-6, "scores differ from run_s2_analysis"
    run = run[okt]
    test = np.isin(run, A.TEST_RUNS)
    gnorm = np.linalg.norm(ght * A.SPAN, axis=1)
    sig = np.maximum(sgt, floor)
    ascent = np.einsum("bi,bi->b", gt * A.SPAN, ght * A.SPAN) > 0   # gᵀ M ĝ with M = diag(span²)
    rep = {"note": "descriptive, not pre-registered", "n_test_points": int(test.sum()), "n_test_runs": len(set(run[test])),
           "raw_gradient_fallback": fallback, "frac_all_steps_true_ascent": float(ascent[test].mean()), "alphas": {}}
    for a, r in rep5["alphas"].items():
        qs = {"iid": r["H7_iid"]["qhat"], "pooled": r["pooled_baseline"]["qhat"], "runmax": r["H5_runmax"]["qhat"],
              "random_iterate_median": r["H6_subsample"]["qhat_median"]}
        out = {}
        for name, q in qs.items():
            cert = gnorm > q * sig
            ct, cov = cert & test, s_tr <= q
            per_run = [int(cert[(run == k)].sum()) for k in A.TEST_RUNS]
            out[name] = {"qhat": q, "frac_steps_certified": float(cert[test].mean()),
                         "frac_certified_steps_true_ascent": float(ascent[ct].mean()) if ct.any() else None,
                         "frac_certified_steps_covered": float(cov[ct].mean()) if ct.any() else None,
                         "frac_runs_all_steps_certified": float(np.mean([c == A.N_IT for c in per_run])),
                         "median_steps_certified_per_run": float(np.median(per_run)),
                         "median_radius_over_grad_norm": float(np.median(q * sig[test] / gnorm[test]))}
        rep["alphas"][a] = out
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    sys.exit(main())
