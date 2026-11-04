"""H3/H4 study: gated surrogate optimisation with CFD fallback (protocol §5–6).

For each gate in {never, always, cdgc, output_uq, distance}, each in-loop CFD budget B in {5, 10} and each of
10 Sobol starting points (seed 7), run the optimiser (optimization.gated_optimizer.optimise) with the trained
surrogate ensemble, then CFD-verify every iterate (primal only; cached) and the final design (primal +
adjoint). Results: results/h3/<gate>_B<budget>_start<k>.json and a summary with the paired Wilcoxon tests.

The CDGC certificate is calibrated on the cal split (i.i.d.) for the main runs; the H4 analysis recomputes
coverage on the CFD-verified trajectory points of runs whose starts are disjoint from the test runs.

Usage: python scripts/run_h3_optimization.py --models checkpoints/v1/fno --gates never cdgc output_uq distance always
Parallel use (one process per start; CFD cases are locked per case id, so shared points are computed once):
  python scripts/run_h3_optimization.py --models checkpoints/v1/fno --start-index 3 --stage run
  python scripts/run_h3_optimization.py --models checkpoints/v1/fno --stage summary      # after all runs
A run whose CFD fails is written as <name>.failed.json (kept and reported, never dropped silently).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import qmc, wilcoxon  # noqa: E402

from data.postprocess import GridSpec  # noqa: E402
from gradient_certification.cdgc import GradientCertificate  # noqa: E402
from models.surrogates import FNOSurrogate, MLPSurrogate  # noqa: E402
from models.training import load_split  # noqa: E402
from optimization.cfd_oracle import CFDOracle  # noqa: E402
from optimization.gated_optimizer import Box, Gate, optimise  # noqa: E402
from optimization.objective import EnsembleSurrogate, objective  # noqa: E402

PROC = ROOT / "datasets" / "processed" / "dataset_v1"
OUT = ROOT / "results" / "h3"
VERIFY_EVERY = 5   # cost control: 7 verified points per 30-step run (protocol v1 change log)


def load_ensemble(folder: Path, kind: str) -> list:
    gx, gy = GridSpec().coords()
    models = []
    for f in sorted(folder.glob("seed*.pt")):
        m = FNOSurrogate(torch.tensor(gx), torch.tensor(gy)) if kind == "fno" else MLPSurrogate()
        m.load_state_dict(torch.load(f, map_location="cpu"), strict=True)  # strict: a silent shape/key mismatch would load a wrong model
        models.append(m.eval())
    return models


def calibrate(ens: EnsembleSurrogate, cal: dict, M: np.ndarray, alpha: float) -> tuple[GradientCertificate, dict]:
    ok = ~np.isnan(cal["jac"]).any(axis=(1, 2))
    g_true, g_hat, sig, Js = [], [], [], []
    for th, cl, cd, J in zip(cal["theta"][ok], cal["CL"][ok], cal["CD"][ok], cal["jac"][ok]):
        _, g_t = objective(float(cl), float(cd), J[0], J[1], float(th[2]))
        Jh, gh, sg, sJ = ens.predict(th)
        g_true.append(g_t); g_hat.append(gh); sig.append(sg); Js.append(sJ)
    cert = GradientCertificate(M=M, alpha=alpha).calibrate(np.array(g_true), np.array(g_hat), np.array(sig))
    return cert, {"sigma_J_q90": float(np.quantile(Js, 0.9)), "n_cal": int(ok.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--kind", default="fno", choices=["fno", "mlp"])
    ap.add_argument("--gates", nargs="+", default=["never", "cdgc", "output_uq", "distance", "always"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--start-index", type=int, nargs="*", default=None, help="subset of start indices (default all)")
    ap.add_argument("--stage", choices=["run", "summary"], default="run")
    args = ap.parse_args()
    torch.set_num_threads(1)
    design = json.loads((ROOT / "datasets" / "designs" / "dataset_v1_design.json").read_text())
    lo = np.array([design["box"][k][0] for k in design["variables"]]); hi = np.array([design["box"][k][1] for k in design["variables"]])
    box = Box(lo, hi)
    M = np.diag(box.span ** 2)
    ens = EnsembleSurrogate(load_ensemble(ROOT / args.models, args.kind), M)
    cal = load_split(PROC / "cal", with_fields=False)
    train = load_split(PROC / "train", with_fields=False)
    cert, cal_info = calibrate(ens, cal, M, args.alpha)
    nn = [np.min(np.linalg.norm((train["theta"] - th) / box.span, axis=1)) for th in cal["theta"]]
    dist_thr = float(np.quantile(nn, 0.9))
    oracle = CFDOracle(ROOT, Path(os.environ.get("AF_WORK", "/content/runs")) / "h3", os.environ.get("AF_SU2_AD", "/content/mamba/su2ad/bin/SU2_CFD_AD"))
    starts = lo + qmc.Sobol(d=4, scramble=True, seed=7).random(args.starts) * box.span
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"cert_qhat": cert.qhat, "cal": cal_info, "dist_threshold": dist_thr, "runs": {}}
    for B in args.budgets:
        for gname in args.gates:
            gate = Gate(kind=gname, cert=cert, uq_threshold=cal_info["sigma_J_q90"], dist_threshold=dist_thr)
            finals, missing = [], []
            for k, th0 in enumerate(starts):
                if args.start_index is not None and k not in args.start_index:
                    continue
                f = OUT / f"{gname}_B{B}_start{k}.json"
                if f.exists():
                    finals.append(json.loads(f.read_text())["final_J_cfd"]); continue
                if args.stage == "summary":
                    missing.append(k); continue
                try:
                    log = optimise(th0, ens, gate, oracle, box, train["theta"], step0=0.05, n_steps=30, cfd_budget=B)
                    # post-hoc CFD verification (primal only, cached): every VERIFY_EVERY-th iterate and the final one
                    idx = sorted(set(range(0, len(log.thetas), VERIFY_EVERY)) | {len(log.thetas) - 1})
                    verified = [oracle.value(log.thetas[i]) for i in idx]
                except Exception as exc:  # recorded, reported in the summary, never dropped silently
                    (OUT / f"{gname}_B{B}_start{k}.failed.json").write_text(json.dumps(
                        {"gate": gname, "budget": B, "start": th0.tolist(), "error": repr(exc)}))
                    print("H3_FAILED", gname, B, k, repr(exc), flush=True)
                    missing.append(k); continue
                failed = int(np.sum(np.diff(verified) < 0))
                rec = {"gate": gname, "budget": B, "start": th0.tolist(), "thetas": [t.tolist() for t in log.thetas],
                       "J_hat": log.J_hat, "verified_steps": idx, "J_cfd": verified, "cfd_calls_in_loop": len(log.cfd_calls),
                       "failed_steps": failed, "final_J_cfd": verified[-1]}
                f.write_text(json.dumps(rec))
                finals.append(verified[-1])
                print("H3", gname, B, k, json.dumps({kk: rec[kk] for kk in ("final_J_cfd", "failed_steps", "cfd_calls_in_loop")}), flush=True)
            summary["runs"][f"{gname}_B{B}"] = finals
            if missing:
                summary.setdefault("missing_or_failed", {})[f"{gname}_B{B}"] = missing
        if args.stage == "summary" and "cdgc" in args.gates and "output_uq" in args.gates and \
                len(summary["runs"][f"cdgc_B{B}"]) == len(summary["runs"][f"output_uq_B{B}"]) == args.starts:
            a, b = summary["runs"][f"cdgc_B{B}"], summary["runs"][f"output_uq_B{B}"]
            summary[f"wilcoxon_cdgc_vs_output_uq_B{B}"] = float(wilcoxon(a, b).pvalue) if np.any(np.array(a) != np.array(b)) else 1.0
    if args.stage == "summary":
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("AF_H3_DONE", flush=True)


if __name__ == "__main__":
    main()
