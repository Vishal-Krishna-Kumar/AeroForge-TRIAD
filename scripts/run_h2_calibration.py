"""H2 — CDGC calibration on i.i.d. data (protocol §4), for the trained surrogate ensembles.

Pre-registered (research/protocol/experiment_protocol_v1.md §4):
  score s = ||g - ĝ||_M / σ̂, ĝ = ensemble mean L/D gradient, σ̂ = ensemble std of ĝ in the M-norm
  (floored at 1e-6 of the calibration median), M = diag(span²); split conformal at α ∈ {0.1, 0.2};
  coverage on test (i.i.d.) and on held-out optimiser trajectories; repeated over cal resamplings.
  Falsified if trajectory coverage < 1 - α - 0.05 (averaged over seeds).

This script covers the i.i.d. part only. Trajectory coverage needs CFD-verified gradients along optimiser
runs (H3 / run_h3_optimization.py) and is reported separately; H2 cannot be decided from this script alone.

Interpretation of "repeated over 5 ensemble seeds/cal resamplings": one 5-member ensemble exists per model
(seeds 0-4), so the repetition is over cal resamplings: (a) the pre-registered cal→test split, and
(b) 5 random re-splits (seeds 0-4) of the pooled cal ∪ test points into 128/128, which keeps exchangeability.

Analyses marked "additional" below were not pre-registered and are reported as descriptive only.

Usage: python scripts/run_h2_calibration.py --models fno mlp
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import beta  # noqa: E402

from data.postprocess import GridSpec  # noqa: E402
from gradient_certification.cdgc import GradientCertificate, ld_gradient, m_norm  # noqa: E402
from models.training import THETA_SPAN, TrainConfig, build_model, load_split, predict_gradients  # noqa: E402

PROC = Path(os.environ.get("AF_DATA", ROOT / "datasets" / "processed" / "dataset_v1"))
HP = {"fno": dict(batch=16, lr=1e-3, width=32, modes=(16, 24), layers=4), "mlp": dict(batch=64, lr=3e-3)}
EPOCHS = {"fno": 500, "mlp": 3000}
ALPHAS = (0.1, 0.2)
M = np.diag(THETA_SPAN ** 2)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def member_gradients(kind: str, fit: dict, thetas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-member L/D gradients, {split: (S, N, 4)}."""
    gx, gy = GridSpec().coords()
    out = {k: [] for k in thetas}
    folder = ROOT / "checkpoints" / "v1" / kind
    seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.pt"))
    assert seeds == [0, 1, 2, 3, 4], seeds
    for s in seeds:
        cfg = TrainConfig(model=kind, seed=s, epochs=EPOCHS[kind], **HP[kind])
        model = build_model(cfg, gx, gy, fit).to(cfg.device)
        model.load_state_dict(torch.load(folder / f"seed{s}.pt", map_location=cfg.device), strict=True)
        model.eval()
        for k, th in thetas.items():
            _, g, _ = predict_gradients(model, th, device=cfg.device, batch=32)
            out[k].append(g)
    return {k: np.stack(v) for k, v in out.items()}


def ensemble_stats(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ĝ = member mean; σ̂ = sqrt(Σ_k ||g_k - ĝ||²_M / (S-1))."""
    gh = G.mean(0)
    S = G.shape[0]
    sq = np.stack([m_norm(G[k] - gh, M) ** 2 for k in range(S)])
    return gh, np.sqrt(sq.sum(0) / (S - 1))


def evaluate(gt_c, gh_c, sg_c, gt_t, gh_t, sg_t, alpha) -> dict:
    floor = 1e-6 * float(np.median(sg_c))
    sg_c, sg_t = np.maximum(sg_c, floor), np.maximum(sg_t, floor)
    cert = GradientCertificate(M=M, alpha=alpha).calibrate(gt_c, gh_c, sg_c)
    cov = cert.covered(gt_t, gh_t, sg_t)
    k, n = int(cov.sum()), len(cov)
    rad = cert.radius(sg_t)
    gnorm = m_norm(gh_t, M)
    certified = gnorm > rad  # steepest-ascent step d = M ĝ is certified iff ||ĝ||_M > q̂ σ̂  (additional)
    true_ascent = np.einsum("bi,ij,bj->b", gt_t, M, gh_t) > 0
    return {"alpha": alpha, "n_cal": cert.n_cal, "n_test": n, "qhat": cert.qhat, "coverage": k / n,
            "coverage_CI95": clopper_pearson(k, n), "target": 1 - alpha,
            "additional": {"median_radius_over_grad_norm": float(np.median(rad / gnorm)),
                           "frac_steps_certified": float(certified.mean()),
                           "frac_certified_steps_true_ascent": float(true_ascent[certified].mean()) if certified.any() else None,
                           "frac_all_steps_true_ascent": float(true_ascent.mean())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["fno", "mlp"])
    ap.add_argument("--resamples", type=int, default=5)
    args = ap.parse_args()
    fit = load_split(PROC / "train", with_fields=False)
    fit = {k: (v[:448] if isinstance(v, np.ndarray) and len(v) == len(fit["theta"]) else v) for k, v in fit.items()}
    cal = load_split(PROC / "cal", with_fields=False)
    te = load_split(PROC / "test", with_fields=False)
    out_dir = ROOT / "results" / "h2"; out_dir.mkdir(parents=True, exist_ok=True)
    for kind in args.models:
        G = member_gradients(kind, fit, {"cal": cal["theta"], "test": te["theta"]})
        res = {"model": kind, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "metric": "M = diag(span^2)",
               "objective": "L/D", "protocol": "experiment_protocol_v1.md §4 (i.i.d. part)", "splits": {}}
        parts = {}
        for name, d in (("cal", cal), ("test", te)):
            ok = ~np.isnan(d["jac"]).any(axis=(1, 2))
            gt = ld_gradient(d["CL"], d["CD"], d["jac"][:, 0], d["jac"][:, 1])
            gh, sg = ensemble_stats(G[name])
            parts[name] = (gt[ok], gh[ok], sg[ok])
            res["splits"][name] = {"n": int(len(ok)), "n_with_adjoint": int(ok.sum())}
        res["preregistered_split"] = [evaluate(*parts["cal"], *parts["test"], a) for a in ALPHAS]
        pool = [np.concatenate([parts["cal"][i], parts["test"][i]]) for i in range(3)]
        n_c = len(parts["cal"][0])
        res["resamples"] = []
        for r in range(args.resamples):
            perm = np.random.default_rng(r).permutation(len(pool[0]))
            c, t = perm[:n_c], perm[n_c:]
            res["resamples"].append({"seed": r, "results": [evaluate(*(p[c] for p in pool), *(p[t] for p in pool), a) for a in ALPHAS]})
        res["resample_mean_coverage"] = {str(a): float(np.mean([rr["results"][i]["coverage"] for rr in res["resamples"]]))
                                         for i, a in enumerate(ALPHAS)}
        res["trajectory_coverage"] = "pending: needs CFD-verified optimiser trajectories (H3 runs); H2 falsification test not yet evaluable"
        np.savez(out_dir / f"h2_member_gradients_{kind}.npz", G_cal=G["cal"], G_test=G["test"],
                 cal_case_id=cal["case_id"], test_case_id=te["case_id"])
        (out_dir / f"h2_iid_{kind}.json").write_text(json.dumps(res, indent=1, default=float))
        for e in res["preregistered_split"]:
            print("H2", kind, f"alpha={e['alpha']}", f"coverage={e['coverage']:.3f}", "CI95", [round(x, 3) for x in e["coverage_CI95"]],
                  f"qhat={e['qhat']:.3g}", json.dumps(e["additional"]), flush=True)
        print("H2", kind, "resample mean coverage", res["resample_mean_coverage"], flush=True)
    print("AF_H2_DONE", flush=True)


if __name__ == "__main__":
    main()
