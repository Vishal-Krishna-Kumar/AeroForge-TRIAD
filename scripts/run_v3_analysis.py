"""Protocol v3 §6 analyses: H2'' (i.i.d.), H5'' (run-max), H6'' (random iterate), H7'' (i.i.d. shift), H9 (decision value).

Written 2026-10-19 BEFORE any V3 trajectory exists (protocol v3 §9: analysis code smoke-tested on fake data first).
Same statistics as V2 (scripts/run_s2_analysis.py, scripts/run_h2v2_calibration.py, left unchanged) with V3 constants:
objective J = CL/(CD+0.008), gradient key dJ_dtheta (adjoint), surrogate dJ/dtheta from each member's (CL, CD) Jacobian
(scripts/train_surrogates_v3.j_and_grad), M = diag(SPAN^2) of the V3 box, N_IT = 12 iterates per run.
Score s = ||g - ĝ||_M / max(σ̂, 1e-6 x median σ̂_cal), ĝ/σ̂ = ensemble mean / spread (ddof S-1) over the 5 members.
* H2'': split conformal, V3 cal -> V3 test, FNO and MLP, α ∈ {0.1, 0.2}; coverage with Clopper-Pearson CI; reported with the
  V2 rule (coverage >= 1 - α - 0.05).
* Trajectories (FNO): unit = run; calibration runs 0-19, test runs 20-39. Design = the largest COMPLETE design of
  [40 runs x 12 iterates, 40 x 10, 30 x 10 (cal 0-14, test 20-34)] (protocol v3 §8 reductions in order; "complete" =
  every point of the design was simulated, any status). Iterates without ground truth are excluded and counted.
  Points that converged with an adjoint but failed post-processing enter from the raw records (protocol v3 §5).
* H5'': run-max q̂; supported at α if whole-run coverage of the test runs >= 1 - α - 0.05.
* H6'': one random iterate per calibration run, 1000 draws (seed 20261034); supported if the mean per-run covered fraction
  >= 1 - α - 0.05 at BOTH α.
* H7'': i.i.d.-calibrated certificate (V3 cal) on the test runs; replicated if whole-run OR per-iterate coverage
  < 1 - α - 0.05 at either α.
* H9 (α = 0.1, run-max q̂): a test-run step is non-ascent if gᵀ M ĝ <= 0 and certified if ||ĝ||_M > q̂ σ̂. (a) non-ascent
  fraction < 5 % -> "not testable in this regime"; (b) otherwise supported iff non-ascent fraction among certified steps
  <= 1/3 of that among uncertified steps (Fisher exact p reported; no uncertified step -> not testable; no certified step -> not testable, since
  the certified fraction is then undefined — r2 fix after the fake-data smoke test reported 'supported' with 0 certified).
* Descriptive (not pre-registered): pooled-point baseline, leave-one-run-out run-max, certified fractions.
CIs: Clopper-Pearson for whole-run fractions; run bootstrap (seed 20261035, 2000 resamples) for per-iterate coverage.
Usage: python scripts/run_v3_analysis.py  -> results/v3_analysis/v3_analysis.json (+ member-gradient npz)
Environment overrides (smoke tests only): AF_DATA (dataset_v3), AF_TRAJ_DATA (dataset_v3_traj/traj), AF_RAW_TRAJ,
AF_TRAJ_POINTS, AF_CKPT_V3 (checkpoints/v3), AF_OUT (results/v3_analysis).
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
from scipy.stats import beta as beta_dist, fisher_exact  # noqa: E402

ALPHAS = (0.1, 0.2)
N_IT, CD0 = 12, 0.008
DESIGNS = [("40x12", list(range(20)), list(range(20, 40)), 12), ("40x10", list(range(20)), list(range(20, 40)), 10),
           ("30x10", list(range(15)), list(range(20, 35)), 10)]
DATA = Path(os.environ.get("AF_DATA", ROOT / "datasets" / "processed" / "dataset_v3"))
TRAJ = Path(os.environ.get("AF_TRAJ_DATA", ROOT / "datasets" / "processed" / "dataset_v3_traj" / "traj"))
RAW = Path(os.environ.get("AF_RAW_TRAJ", ROOT / "datasets" / "raw" / "dataset_v3_traj"))
POINTS = Path(os.environ.get("AF_TRAJ_POINTS", ROOT / "results" / "v3_traj" / "trajectory_points.json"))
CKPT = Path(os.environ.get("AF_CKPT_V3", ROOT / "checkpoints" / "v3"))
OUT = Path(os.environ.get("AF_OUT", ROOT / "results" / "v3_analysis"))
_D = json.loads((ROOT / "datasets" / "designs" / "dataset_v3_design.json").read_text())
SPAN = np.asarray(_D["box_hi"]) - np.asarray(_D["box_lo"])
M = np.diag(SPAN ** 2)
HP = {"fno": dict(batch=16, lr=1e-3, width=32, modes=(16, 24), layers=4), "mlp": dict(batch=64, lr=3e-3)}


def cq(scores, alpha):
    s = np.sort(np.asarray(scores, float)); k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else float(s[k - 1])


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def load(folder: Path) -> dict:
    from train_surrogates_v3 import load_v3
    return load_v3(folder, with_fields=False)


def member_grads(kind: str, thetas: dict) -> dict:
    import torch
    from data.postprocess import GridSpec
    from models.training import TrainConfig, build_model, load_split
    from train_surrogates_v3 import j_and_grad
    gx, gy = GridSpec().coords()
    fit = load_split(DATA / "train", with_fields=False)
    n_fit = int(json.loads((CKPT / kind / "seed0.json").read_text())["n_fit"])
    fit = {k: (v[:n_fit] if isinstance(v, np.ndarray) and len(v) == len(fit["theta"]) else v) for k, v in fit.items()}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {k: [] for k in thetas}
    for s in range(5):
        cfg = TrainConfig(model=kind, seed=s, device=dev, **HP[kind])
        m = build_model(cfg, gx, gy, fit).to(dev)
        m.load_state_dict(torch.load(CKPT / kind / f"seed{s}.pt", map_location=dev), strict=True)
        m.eval()
        for k, th in thetas.items():
            out[k].append(j_and_grad(m, th.astype(np.float32), dev)[1] if len(th) else np.zeros((0, len(SPAN))))
    return {k: np.stack(v) for k, v in out.items()}


def stats(G):
    gh = G.mean(0)
    sq = np.stack([np.linalg.norm((G[k] - gh) * SPAN, axis=1) ** 2 for k in range(G.shape[0])])
    return gh, np.sqrt(sq.sum(0) / (G.shape[0] - 1))


def score(gt, gh, sg, floor):
    return np.linalg.norm((gt - gh) * SPAN, axis=1) / np.maximum(sg, floor)


def h9_verdict(na: np.ndarray, certified: np.ndarray) -> dict:
    """H9 decision rule (protocol v3 §6) from boolean arrays over the test-run steps: na = non-ascent, certified."""
    h9 = {"n_steps": int(len(na)), "non_ascent_fraction": float(na.mean()), "n_certified": int(certified.sum()),
          "n_uncertified": int((~certified).sum())}
    if na.mean() < 0.05:
        h9["verdict"] = "not testable in this regime (non-ascent fraction < 5 %)"
    elif not (~certified).any():
        h9["verdict"] = "not testable (no uncertified step)"
    elif not certified.any():  # the certified non-ascent fraction is 0/0: undefined, so H9 cannot be supported
        h9.update({"non_ascent_among_uncertified": float(na.mean()), "verdict": "not testable (no certified step)"})
    else:
        f_c = float(na[certified].mean())
        f_u = float(na[~certified].mean())
        table = [[int((na & certified).sum()), int((~na & certified).sum())], [int((na & ~certified).sum()), int((~na & ~certified).sum())]]
        h9.update({"non_ascent_among_certified": f_c, "non_ascent_among_uncertified": f_u, "table": table,
                   "fisher_p": float(fisher_exact(table)[1]), "verdict": "supported" if f_c <= f_u / 3 else "not supported"})
    return h9


def add_raw_gradient_points(tr: dict) -> tuple[dict, list]:
    """Protocol v3 §5 (as the V2 §9 amendment): converged points with an adjoint J gradient but no npz enter from raw."""
    pts = json.loads(POINTS.read_text())["points"]
    have, add = set(tr["case_id"].tolist()), []
    for p in pts:
        d = RAW / p["case_id"]
        if p["case_id"] in have or not (d / "gradients.json").exists() or not (d / "case.json").exists():
            continue
        if json.loads((d / "case.json").read_text()).get("status") != "converged":
            continue
        g = json.loads((d / "gradients.json").read_text())
        gj = g.get("objectives", {}).get("J", {}).get("grad")
        if gj:
            add.append((p["case_id"], np.asarray(p["theta"], np.float32), np.array([gj[k] for k in g["dv"]], np.float64)))
    if add:
        tr = dict(tr)
        tr["case_id"] = np.concatenate([tr["case_id"], [a[0] for a in add]])
        tr["theta"] = np.concatenate([tr["theta"], np.stack([a[1] for a in add])])
        tr["g_J"] = np.concatenate([tr["g_J"], np.stack([a[2] for a in add])])
    return tr, [a[0] for a in add]


def h2(kind: str, cal: dict, te: dict) -> dict:
    okc, okt = ~np.isnan(cal["g_J"]).any(1), ~np.isnan(te["g_J"]).any(1)
    G = member_grads(kind, {"cal": cal["theta"][okc], "test": te["theta"][okt]})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"member_gradients_iid_{kind}.npz", G_cal=G["cal"], G_test=G["test"],
             cal_case_id=cal["case_id"][okc], test_case_id=te["case_id"][okt])
    ghc, sgc = stats(G["cal"]); ght, sgt = stats(G["test"])
    floor = 1e-6 * float(np.median(sgc))
    sc, st = score(cal["g_J"][okc], ghc, sgc, floor), score(te["g_J"][okt], ght, sgt, floor)
    res = {"n_cal": int(okc.sum()), "n_test": int(okt.sum()), "alphas": {}}
    for a in ALPHAS:
        q = cq(sc, a); k = int((st <= q).sum())
        res["alphas"][str(a)] = {"qhat": q, "coverage": k / len(st), "coverage_CI95": clopper_pearson(k, len(st)),
                                 "meets_v2_rule": bool(k / len(st) >= 1 - a - 0.05)}
    return res, (sc, floor, ghc, sgc)


def main():
    cal, te = load(DATA / "cal"), load(DATA / "test")
    rep = {"objective": "J = CL/(CD+0.008)", "metric": "M = diag(span^2), V3 box", "H2pp": {}}
    for kind in ("fno", "mlp"):
        rep["H2pp"][kind], keep = h2(kind, cal, te)
        if kind == "fno":
            s_cal, floor = keep[0], keep[1]
    tr = load(TRAJ)
    tr, fallback = add_raw_gradient_points(tr)
    okt = ~np.isnan(tr["g_J"]).any(1)
    run = np.array([int(c[4:6]) for c in tr["case_id"]]); step = np.array([int(c[8:10]) for c in tr["case_id"]])
    simulated = lambda r, j: (RAW / f"T3_s{r:02d}_t{j:02d}" / "case.json").exists()
    name, cal_runs, test_runs, n_it = next((d for d in DESIGNS if all(simulated(r, j) for r in d[1] + d[2] for j in range(d[3]))),
                                           DESIGNS[-1])
    complete = all(simulated(r, j) for r in cal_runs + test_runs for j in range(n_it))
    okt = okt & np.isin(run, cal_runs + test_runs) & (step < n_it)
    G = member_grads("fno", {"traj": tr["theta"][okt]})
    np.savez(OUT / "member_gradients_traj_fno.npz", G=G["traj"], case_id=tr["case_id"][okt])
    gh, sg = stats(G["traj"])
    gt = tr["g_J"][okt]
    s_tr = score(gt, gh, sg, floor)
    run, step = run[okt], step[okt]
    by = {r: s_tr[run == r] for r in sorted(set(run.tolist()))}
    rep.update({"design": name, "design_complete": complete, "n_traj_points": int(okt.sum()),
                "n_traj_missing": len(cal_runs + test_runs) * n_it - int(okt.sum()), "raw_gradient_fallback": fallback,
                "runs_incomplete": [r for r in cal_runs + test_runs if len(by.get(r, [])) < n_it],
                "median_score_iid_cal": float(np.median(s_cal)), "median_score_traj": float(np.median(s_tr)), "alphas": {}})
    test = [r for r in test_runs if r in by]; calr = [r for r in cal_runs if r in by]
    rng_boot, rng_h6 = np.random.default_rng(20261035), np.random.default_rng(20261034)

    def cov_runs(q, boot=True):
        whole = np.array([float((by[r] <= q).all()) for r in test]); point = np.array([(by[r] <= q).mean() for r in test])
        out = {"qhat": q, "whole_run": float(whole.mean()), "whole_run_CI95": clopper_pearson(int(whole.sum()), len(whole)),
               "per_iterate": float(point.mean())}
        if boot:
            b = [point[rng_boot.integers(0, len(point), len(point))].mean() for _ in range(2000)]
            out["per_iterate_CI95"] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
        return out

    for a in ALPHAS:
        thr = 1 - a - 0.05
        r = {"target": 1 - a, "threshold": thr}
        r["H5_runmax"] = cov_runs(cq([by[k].max() for k in calr], a))
        subs = [cq([rng_h6.choice(by[k]) for k in calr], a) for _ in range(1000)]
        h6 = [cov_runs(q, boot=False)["per_iterate"] for q in subs]
        r["H6_random_iterate"] = {"qhat_median": float(np.median(subs)), "per_iterate_mean_over_draws": float(np.mean(h6))}
        r["H7_iid"] = cov_runs(cq(s_cal, a))
        r["pooled_baseline_descriptive"] = cov_runs(cq(np.concatenate([by[k] for k in calr]), a))
        r["H5_supported"] = r["H5_runmax"]["whole_run"] >= thr
        r["H6_ok_at_alpha"] = r["H6_random_iterate"]["per_iterate_mean_over_draws"] >= thr
        r["H7_replicated_at_alpha"] = (r["H7_iid"]["whole_run"] < thr) or (r["H7_iid"]["per_iterate"] < thr)
        r["LOO_runmax_whole_run_descriptive"] = float(np.mean([(by[t] <= cq([by[k].max() for k in by if k != t], a)).all() for t in by]))
        rep["alphas"][str(a)] = r
    rep["H6_supported"] = all(rep["alphas"][str(a)]["H6_ok_at_alpha"] for a in ALPHAS)
    rep["H7_replicated"] = any(rep["alphas"][str(a)]["H7_replicated_at_alpha"] for a in ALPHAS)
    # H9 on the test-run steps, alpha 0.1, run-max qhat
    q = rep["alphas"]["0.1"]["H5_runmax"]["qhat"]
    sel = np.isin(run, test)
    ascent = np.einsum("bi,ij,bj->b", gt[sel], M, gh[sel]) > 0
    certified = np.linalg.norm(gh[sel] * SPAN, axis=1) > q * np.maximum(sg[sel], floor)
    h9 = h9_verdict(~ascent, certified)
    rep["H9"] = h9
    rep["certified_fraction_descriptive"] = {k: float((np.linalg.norm(gh[sel] * SPAN, axis=1) > rep["alphas"]["0.1"][v]["qhat"] * np.maximum(sg[sel], floor)).mean())
                                            for k, v in (("runmax", "H5_runmax"), ("iid", "H7_iid"), ("pooled", "pooled_baseline_descriptive"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v3_analysis.json").write_text(json.dumps(rep, indent=1, default=float))
    print("V3_ANALYSIS", json.dumps({"design": name, "H2pp": {k: {a: v["coverage"] for a, v in rep["H2pp"][k]["alphas"].items()} for k in rep["H2pp"]},
                                     "H5": {a: rep["alphas"][a]["H5_supported"] for a in rep["alphas"]}, "H6": rep["H6_supported"],
                                     "H7": rep["H7_replicated"], "H9": h9["verdict"]}), flush=True)


if __name__ == "__main__":
    main()
