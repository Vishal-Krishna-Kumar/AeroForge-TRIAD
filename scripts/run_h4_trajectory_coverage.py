"""H2 (trajectory part) and H4: coverage of conformal design-gradient certificates on optimiser trajectories.

Written and fixed on 2026-10-13 before any trajectory gradient existed (experiment log, same date).

Data: every CFD-verified iterate of the H3 runs (results/h3/<gate>_B<budget>_start<k>.json) that has a
converged primal and adjoint gradients of CL and CD (datasets/raw/optimizer_calls/<case_id>/). A point is
counted once (de-duplicated by case id) and belongs to the start index k of the runs that visited it (all runs
from one start share their first point, so the start is the exchangeable unit: "split by run", protocol §1).

Certificate (as in scripts/run_h2_calibration.py): ĝ = ensemble-mean L/D gradient of the 5 FNO members,
σ̂ = sqrt(Σ_k ||g_k − ĝ||²_M / (S−1)), M = diag(span²), σ̂ floored at 1e-6 × the calibration median, score
s = ||g − ĝ||_M / σ̂, split conformal quantile q̂ at α ∈ {0.1, 0.2}.

H2 trajectory test (protocol §4). q̂ is calibrated on i.i.d. data: (a) the pre-registered cal split and
(b) the same 5 random 128-point re-splits of cal ∪ test (seeds 0-4) as the i.i.d. analysis. The decision
statistic is the trajectory coverage averaged over (b); (a) is reported alongside. H2 is FALSIFIED if, at either
α, that average is below 1 − α − 0.05. Uncertainty: start-level bootstrap (2000 resamples of the 10 starts).

H4 (protocol §6). Trajectory-calibrated certificate: calibrate on the points of starts A, test on starts B, for
the two halves A = {0..4}, B = {5..9} and the reverse; reported next to the i.i.d.-calibrated certificate on the
same test points, with start-level bootstrap CIs. Descriptive breakdowns by gate are reported, not tested.

Secondary (descriptive, not implemented in this version): the same analysis for the penalised H3 objective J.

Usage: python scripts/run_h4_trajectory_coverage.py      (writes results/h4/h4_trajectory_coverage.json)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gradient_certification.cdgc import conformal_quantile, ld_gradient, m_norm  # noqa: E402
from models.training import THETA_SPAN, load_split  # noqa: E402
from optimization.cfd_oracle import CFDOracle  # noqa: E402
from run_h2_calibration import ensemble_stats, member_gradients  # noqa: E402

PROC = Path(os.environ.get("AF_DATA", ROOT / "datasets" / "processed" / "dataset_v1"))
H3 = ROOT / "results" / "h3"
CALLS = ROOT / "datasets" / "raw" / "optimizer_calls"
OUT = ROOT / "results" / "h4"
ALPHAS = (0.1, 0.2)
M = np.diag(THETA_SPAN ** 2)
DV = ("m", "p", "t", "aoa_deg")


def trajectory_points() -> tuple[list[dict], dict]:
    pts: dict[str, dict] = {}
    info = {"runs": 0, "verified_points_total": 0, "missing_gradient": [], "not_converged": []}
    for f in sorted(H3.glob("*_B*_start*.json")):
        if f.name.endswith(".failed.json"):
            continue
        rec = json.loads(f.read_text()); info["runs"] += 1
        k = int(f.stem.rsplit("start", 1)[1])
        for i in rec["verified_steps"]:
            th = np.asarray(rec["thetas"][i], dtype=float); info["verified_points_total"] += 1
            cid = CFDOracle.case_id(th)
            p = pts.setdefault(cid, {"case_id": cid, "theta": th, "start": k, "visits": []})
            p["visits"].append({"gate": rec["gate"], "budget": rec["budget"], "step": i})
    usable = []
    for cid, p in pts.items():
        cj, gj = CALLS / cid / "case.json", CALLS / cid / "gradients.json"
        if not cj.exists() or json.loads(cj.read_text()).get("status") != "converged":
            info["not_converged"].append(cid); continue
        if not gj.exists():
            info["missing_gradient"].append(cid); continue
        a = json.loads(cj.read_text())["assessment"]; g = json.loads(gj.read_text())["objectives"]
        try:
            p["dCL"] = np.array([g["CL"]["grad"][d] for d in DV]); p["dCD"] = np.array([g["CD"]["grad"][d] for d in DV])
        except KeyError:
            info["missing_gradient"].append(cid); continue
        p["CL"], p["CD"] = float(a["CL"]), float(a["CD"])
        usable.append(p)
    info["unique_points"] = len(pts); info["usable_points"] = len(usable)
    return usable, info


def scores(gt, gh, sg, floor):
    return m_norm(gt - gh, M) / np.maximum(sg, floor)


def boot_cov(covered: np.ndarray, starts: np.ndarray, rng, B: int = 2000) -> list[float]:
    u = np.unique(starts)
    if len(u) == 0:
        return [float("nan"), float("nan")]
    idx = {s: np.flatnonzero(starts == s) for s in u}
    vals = []
    for _ in range(B):
        pick = rng.choice(u, size=len(u), replace=True)
        sel = np.concatenate([idx[s] for s in pick])
        vals.append(covered[sel].mean())
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def analyse(name, gt_c, gh_c, sg_c, gt_p, gh_p, sg_p, starts, gates, rng, n_cal: int) -> dict:
    floor = 1e-6 * float(np.median(sg_c))
    s_c, s_p = scores(gt_c, gh_c, sg_c, floor), scores(gt_p, gh_p, sg_p, floor)
    res = {"quantity": name, "n_points": int(len(s_p)), "alphas": {}}
    # s_c is the pool cal ∪ test in that order; its first n_cal entries are the pre-registered calibration split,
    # and the re-splits use the same seeds and pool order as scripts/run_h2_calibration.py
    for a in ALPHAS:
        r = {}
        q_pre = conformal_quantile(s_c[:n_cal], a)            # first half of the pool = the pre-registered cal split
        cov = s_p <= q_pre
        r["iid_preregistered"] = {"qhat": q_pre, "coverage": float(cov.mean()), "CI95_start_bootstrap": boot_cov(cov, starts, rng)}
        covs = []
        for seed in range(5):
            perm = np.random.default_rng(seed).permutation(len(s_c))
            covs.append(float((s_p <= conformal_quantile(s_c[perm[:n_cal]], a)).mean()))
        r["iid_resampled_mean_coverage"] = float(np.mean(covs)); r["iid_resampled_each"] = covs
        r["target"] = 1 - a
        r["H2_falsified_at_this_alpha"] = bool(np.mean(covs) < 1 - a - 0.05)
        halves = []
        for A, Bh in (((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), ((5, 6, 7, 8, 9), (0, 1, 2, 3, 4))):
            ca, tb = np.isin(starts, A), np.isin(starts, Bh)
            if ca.sum() == 0 or tb.sum() == 0:
                halves.append({"cal_starts": list(A), "test_starts": list(Bh), "n_cal": int(ca.sum()), "n_test": int(tb.sum()),
                               "note": "no points on one side; not evaluable"})
                continue
            q_tr = conformal_quantile(s_p[ca], a)
            cov_tr, cov_iid = s_p[tb] <= q_tr, s_p[tb] <= q_pre
            halves.append({"cal_starts": list(A), "test_starts": list(Bh), "n_cal": int(ca.sum()), "n_test": int(tb.sum()),
                           "qhat_trajectory": q_tr, "coverage_trajectory_calibrated": float(cov_tr.mean()),
                           "CI95_traj": boot_cov(cov_tr, starts[tb], rng),
                           "coverage_iid_calibrated": float(cov_iid.mean()), "CI95_iid": boot_cov(cov_iid, starts[tb], rng)})
        r["H4_run_level_split"] = halves
        r["by_gate_iid_preregistered_coverage"] = {g: float(cov[[g in gs for gs in gates]].mean())
                                                   for g in sorted({x for gs in gates for x in gs})
                                                   if any(g in gs for gs in gates)}
        res["alphas"][str(a)] = r
    res["median_score_cal"], res["median_score_traj"] = float(np.median(s_c)), float(np.median(s_p))
    return res


def main() -> None:
    torch.set_num_threads(int(os.environ.get("AF_TORCH_THREADS", "8")))
    pts, info = trajectory_points()
    print("H4_POINTS", json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in info.items()}), flush=True)
    fit = load_split(PROC / "train", with_fields=False)
    fit = {k: (v[:448] if isinstance(v, np.ndarray) and len(v) == len(fit["theta"]) else v) for k, v in fit.items()}
    cal, te = load_split(PROC / "cal", with_fields=False), load_split(PROC / "test", with_fields=False)
    th_p = np.stack([p["theta"] for p in pts])
    G = member_gradients("fno", fit, {"cal": cal["theta"], "test": te["theta"], "traj": th_p})
    # i.i.d. pool in the fixed order cal then test (first half = pre-registered calibration split)
    gt_parts, gh_parts, sg_parts = [], [], []
    for name, d in (("cal", cal), ("test", te)):
        ok = ~np.isnan(d["jac"]).any(axis=(1, 2))
        gt = ld_gradient(d["CL"], d["CD"], d["jac"][:, 0], d["jac"][:, 1])
        gh, sg = ensemble_stats(G[name])
        gt_parts.append(gt[ok]); gh_parts.append(gh[ok]); sg_parts.append(sg[ok])
    gt_c, gh_c, sg_c = (np.concatenate(x) for x in (gt_parts, gh_parts, sg_parts))
    gt_p = ld_gradient(np.array([p["CL"] for p in pts]), np.array([p["CD"] for p in pts]),
                       np.stack([p["dCL"] for p in pts]), np.stack([p["dCD"] for p in pts]))
    gh_p, sg_p = ensemble_stats(G["traj"])
    starts = np.array([p["start"] for p in pts]); gates = [{v["gate"] for v in p["visits"]} for p in pts]
    rng = np.random.default_rng(20261013)
    out = {"info": info, "primary_LD": analyse("L/D gradient", gt_c, gh_c, sg_c, gt_p, gh_p, sg_p, starts, gates, rng,
                                               n_cal=len(gt_parts[0]))}
    out["note_secondary"] = "the penalised-objective (J) variant named in the docstring is not computed by this version"
    a1 = out["primary_LD"]["alphas"]
    out["H2_trajectory_falsified"] = bool(any(a1[str(a)]["H2_falsified_at_this_alpha"] for a in ALPHAS))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "h4_trajectory_points.npz", theta=th_p, g_true=gt_p, g_hat=gh_p, sigma=sg_p, starts=starts,
             G_members=G["traj"], case_id=np.array([p["case_id"] for p in pts]))
    (OUT / "h4_trajectory_coverage.json").write_text(json.dumps(out, indent=1, default=float))
    for a in ALPHAS:
        r = a1[str(a)]
        print("H2T", f"alpha={a}", "iid_pre", round(r["iid_preregistered"]["coverage"], 3), r["iid_preregistered"]["CI95_start_bootstrap"],
              "iid_resampled_mean", round(r["iid_resampled_mean_coverage"], 3), "falsified" if r["H2_falsified_at_this_alpha"] else "ok", flush=True)
        for h in r["H4_run_level_split"]:
            if "note" in h:
                print("H4", f"alpha={a}", h["note"], flush=True); continue
            print("H4", f"alpha={a}", "cal", h["cal_starts"], "traj", round(h["coverage_trajectory_calibrated"], 3),
                  "iid", round(h["coverage_iid_calibrated"], 3), flush=True)
    print("AF_H4_DONE H2_trajectory_falsified=", out["H2_trajectory_falsified"], flush=True)


if __name__ == "__main__":
    main()
