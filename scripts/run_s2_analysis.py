"""Protocol v2 §4: H5 (whole-run certificate), H6 (random-iterate certificate), H7 (i.i.d. shift replication).

Written 2026-10-13 BEFORE any V2 trajectory CFD exists (pod session 1 running, trajectories not yet generated).
Inputs: V2 cal split (i.i.d., datasets/processed/dataset_v2/cal, dLD_dtheta), trajectory iterates with ground truth
(datasets/processed/dataset_v2_traj/traj/T2_sKK_tJJ.npz), FNO members checkpoints/v2/fno/seed{0..4}.pt.
Score s = ||g - ĝ||_M / max(σ̂, 1e-6 × median σ̂_cal), M = diag(SPAN²), ĝ/σ̂ = ensemble mean / spread (ddof S-1).
Unit = run (start). Calibration runs 0–19, test runs 20–39 (pre-specified); if the 40 x 15 design is incomplete,
the reduced design of protocol v2 §9 (cal 0–14, test 20–34, iterates 0–9) is used. α ∈ {0.1, 0.2}. Decision: point
estimate ≥ 1 − α − 0.05 (H5/H6 supported); H7 replicated if whole-run OR per-iterate i.i.d.-calibrated coverage on
the test runs < 1 − α − 0.05 at either α. CIs: Clopper–Pearson (whole-run fraction), run bootstrap (seed 20261024,
2000 resamples) for per-iterate coverage. Iterates without converged ground truth are excluded and counted; a run
missing iterates is evaluated on the iterates it has and flagged. Pooled-point calibration is reported as a
baseline without guarantee; leave-one-run-out results are descriptive.
Usage: python scripts/run_s2_analysis.py   -> results/v2_traj/s2_h5_h6_h7.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

ALPHAS = (0.1, 0.2)
CAL_RUNS, TEST_RUNS, N_IT = list(range(20)), list(range(20, 40)), 15
# protocol v2 §9 (budget amendment, fixed before any trajectory exists): if the full 40 x 15 design is not complete,
# the analysis uses the REDUCED design — calibration runs 0-14, test runs 20-34, iterates 0-9 of each run
RED_CAL, RED_TEST, RED_IT = list(range(15)), list(range(20, 35)), 10
PROC = ROOT / "datasets" / "processed"
_D = json.loads((ROOT / "datasets" / "designs" / "dataset_v2_design.json").read_text())
SPAN = np.asarray(_D["box_hi"]) - np.asarray(_D["box_lo"])


def cq(scores, alpha):
    s = np.sort(np.asarray(scores, float)); k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else float(s[k - 1])


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def member_grads(thetas: dict) -> dict:
    import torch
    from data.postprocess import GridSpec
    from models.training import TrainConfig, build_model, load_split, predict_gradients
    gx, gy = GridSpec().coords()
    fit = load_split(PROC / "dataset_v2" / "train", with_fields=False)
    n_fit = int(json.loads((ROOT / "checkpoints" / "v2" / "fno" / "seed0.json").read_text())["n_fit"])
    fit = {k: (v[:n_fit] if isinstance(v, np.ndarray) and len(v) == len(fit["theta"]) else v) for k, v in fit.items()}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {k: [] for k in thetas}
    for s in range(5):
        cfg = TrainConfig(model="fno", seed=s, device=dev, batch=16, lr=1e-3, width=32, modes=(16, 24), layers=4)
        m = build_model(cfg, gx, gy, fit).to(dev)
        m.load_state_dict(torch.load(ROOT / "checkpoints" / "v2" / "fno" / f"seed{s}.pt", map_location=dev), strict=True)
        m.eval()
        for k, th in thetas.items():
            out[k].append(predict_gradients(m, th.astype(np.float32), device=dev, batch=32)[1])
    return {k: np.stack(v) for k, v in out.items()}


def stats(G):
    gh = G.mean(0)
    sq = np.stack([np.linalg.norm((G[k] - gh) * SPAN, axis=1) ** 2 for k in range(G.shape[0])])
    return gh, np.sqrt(sq.sum(0) / (G.shape[0] - 1))


def score(gt, gh, sg, floor):
    return np.linalg.norm((gt - gh) * SPAN, axis=1) / np.maximum(sg, floor)


def add_raw_gradient_points(tr: dict) -> tuple[dict, list]:
    """Protocol v2 §9 amendment (2026-10-14 ~18:50Z, BEFORE any H5-H7 number was computed): a trajectory point whose
    CFD converged and whose adjoint L/D gradient exists, but whose field post-processing failed (no npz; e.g.
    T2_s37_t04 'fluid grid point not covered by the mesh'), still has ground truth for the gradient score. Such points
    are taken from the raw records (theta from the frozen trajectory_points.json, gradient from gradients.json) instead
    of being dropped. Every point added this way is listed in the report."""
    pts = json.loads((ROOT / "results" / "v2_traj" / "trajectory_points.json").read_text())["points"]
    have, add = set(tr["case_id"].tolist()), []
    for p in pts:
        d = ROOT / "datasets" / "raw" / "dataset_v2_traj" / p["case_id"]
        if p["case_id"] in have or not (d / "gradients.json").exists() or not (d / "case.json").exists():
            continue
        if json.loads((d / "case.json").read_text()).get("status") != "converged":
            continue
        g = json.loads((d / "gradients.json").read_text())
        gl = g.get("objectives", {}).get("LD", {}).get("grad")
        if gl:
            add.append((p["case_id"], np.asarray(p["theta"], np.float32), np.array([gl[k] for k in g["dv"]], np.float64)))
    if add:
        tr = dict(tr)
        tr["case_id"] = np.concatenate([tr["case_id"], [a[0] for a in add]])
        tr["theta"] = np.concatenate([tr["theta"], np.stack([a[1] for a in add])])
        tr["g_ld"] = np.concatenate([tr["g_ld"], np.stack([a[2] for a in add])])
    return tr, [a[0] for a in add]


def main():
    from models.training import load_split
    cal = load_split(PROC / "dataset_v2" / "cal", with_fields=False)
    tr = load_split(PROC / "dataset_v2_traj" / "traj", with_fields=False)
    tr, fallback = add_raw_gradient_points(tr)
    okc = ~np.isnan(cal["g_ld"]).any(1); okt = ~np.isnan(tr["g_ld"]).any(1)
    run = np.array([int(c[4:6]) for c in tr["case_id"]]); step = np.array([int(c[8:10]) for c in tr["case_id"]])
    raw = ROOT / "datasets" / "raw" / "dataset_v2_traj"   # "complete" = every point SIMULATED (any status)
    full = all((raw / f"T2_s{r:02d}_t{j:02d}" / "case.json").exists() for r in range(40) for j in range(N_IT))
    cal_runs, test_runs, n_it = (CAL_RUNS, TEST_RUNS, N_IT) if full else (RED_CAL, RED_TEST, RED_IT)
    okt = okt & np.isin(run, cal_runs + test_runs) & (step < n_it)
    G = member_grads({"cal": cal["theta"][okc], "traj": tr["theta"][okt]})
    ghc, sgc = stats(G["cal"]); ght, sgt = stats(G["traj"])
    floor = 1e-6 * float(np.median(sgc))
    s_cal = score(cal["g_ld"][okc], ghc, sgc, floor)
    s_tr = score(tr["g_ld"][okt], ght, sgt, floor)
    run = run[okt]
    by = {r: s_tr[run == r] for r in range(40) if (run == r).any()}
    rep = {"n_cal_iid": int(okc.sum()), "n_traj_points": int(okt.sum()), "n_traj_missing": len(cal_runs + test_runs) * n_it - int(okt.sum()),
           "raw_gradient_fallback": fallback, "runs_present": sorted(by), "design": "full 40x15" if full else "reduced 30x10 (protocol v2 §9)",
           "runs_incomplete": [r for r in cal_runs + test_runs if len(by.get(r, [])) < n_it],
           "median_score_iid": float(np.median(s_cal)), "median_score_traj": float(np.median(s_tr)), "alphas": {}}
    rng = np.random.default_rng(20261024)
    test = [r for r in test_runs if r in by]; calr = [r for r in cal_runs if r in by]

    def cov_runs(q):
        whole = np.array([float((by[r] <= q).all()) for r in test]); point = np.array([(by[r] <= q).mean() for r in test])
        boot = [point[rng.integers(0, len(point), len(point))].mean() for _ in range(2000)]
        return {"qhat": q, "whole_run": float(whole.mean()), "whole_run_CI95": clopper_pearson(int(whole.sum()), len(whole)),
                "per_iterate": float(point.mean()), "per_iterate_CI95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    for a in ALPHAS:
        r = {"target": 1 - a}
        r["H5_runmax"] = cov_runs(cq([by[k].max() for k in calr], a))
        subs = [cq([rng.choice(by[k]) for k in calr], a) for _ in range(1000)]
        h6 = [cov_runs(q)["per_iterate"] for q in subs[:200]]
        r["H6_subsample"] = {"qhat_median": float(np.median(subs)), "per_iterate_mean_over_draws": float(np.mean(h6))}
        r["H7_iid"] = cov_runs(cq(s_cal, a))
        r["pooled_baseline"] = cov_runs(cq(np.concatenate([by[k] for k in calr]), a))
        thr = 1 - a - 0.05
        r["H5_supported"] = r["H5_runmax"]["whole_run"] >= thr
        r["H6_supported"] = r["H6_subsample"]["per_iterate_mean_over_draws"] >= thr
        r["H7_replicated_at_alpha"] = (r["H7_iid"]["whole_run"] < thr) or (r["H7_iid"]["per_iterate"] < thr)
        loo = [(by[t] <= cq([by[k].max() for k in by if k != t], a)).all() for t in by]
        r["descriptive_LOO_runmax_whole_run"] = float(np.mean(loo))
        rep["alphas"][str(a)] = r
    rep["H7_replicated"] = any(rep["alphas"][str(a)]["H7_replicated_at_alpha"] for a in ALPHAS)
    out = ROOT / "results" / "v2_traj" / "s2_h5_h6_h7.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({a: {k: v for k, v in rep["alphas"][a].items() if k.endswith(("supported", "at_alpha"))} for a in rep["alphas"]}))


if __name__ == "__main__":
    main()
