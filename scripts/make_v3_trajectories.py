"""Protocol v3 §5: the 40 surrogate-only optimiser runs whose iterates get CFD primal + J adjoint in phase 3.

Starts: first 40 accepted points of a scrambled Sobol stream (seed 20261033) in the V3 box (same acceptance as the
dataset, scripts/run_dataset_v3.sample). Optimiser as V2 (src/optimization/gated_optimizer.optimise, gate "never"):
maximise J = CL/(CD+CD0) with the FNO-ensemble-mean gradient (5 members, checkpoints/v3/fno), normalised step
h = 0.05 in box-scaled coordinates, projection onto the box, 11 steps -> 12 iterates per run (start included).
Iterates leaving the acceptance set are kept and flagged. Output (frozen; sha256 printed and logged BEFORE any of these
points is simulated; existing files are never overwritten):
  results/v3_traj/trajectories.json      {meta, runs: [{start, iterates, accepted, J_hat}]}
  results/v3_traj/trajectory_points.json {meta, points: [{case_id, theta, start, step}]}  (input of run_dataset_v3.py traj)
Usage (Colab, GPU optional): python scripts/make_v3_trajectories.py   [AF_CKPT to override the checkpoint folder in smoke tests]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_dataset_v3 as d3  # noqa: E402
from data.postprocess import GridSpec  # noqa: E402
from models.training import TrainConfig, build_model, load_split, predict_coefficients, predict_gradients  # noqa: E402
from optimization.gated_optimizer import Box, Gate, optimise  # noqa: E402

N_RUNS, SEED, STEP, N_STEPS = 40, 20261033, 0.05, 11
CK = Path(os.environ.get("AF_CKPT", ROOT / "checkpoints" / "v3" / "fno"))
OUT = Path(os.environ.get("AF_TRAJ_OUT", ROOT / "results" / "v3_traj"))
HP = dict(batch=16, lr=1e-3, width=32, modes=(16, 24), layers=4)  # = scripts/train_surrogates_v3.HP["fno"]


def load_members():
    gx, gy = GridSpec().coords()
    fit = load_split(Path(os.environ.get("AF_DATA", ROOT / "datasets" / "processed" / "dataset_v3")) / "train", with_fields=False)
    n_fit = int(json.loads((CK / "seed0.json").read_text())["n_fit"])  # = training fit set (normalisation statistics)
    fit = {k: (v[:n_fit] if isinstance(v, np.ndarray) and len(v) == len(fit["theta"]) else v) for k, v in fit.items()}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members = []
    for s in range(5):
        cfg = TrainConfig(model="fno", seed=s, device=dev, **HP)
        m = build_model(cfg, gx, gy, fit).to(dev)
        m.load_state_dict(torch.load(CK / f"seed{s}.pt", map_location=dev), strict=True)
        members.append(m.eval())
    return members, dev


class Ensemble:
    def __init__(self, members, dev):
        self.members, self.dev = members, dev

    def predict(self, theta):
        th = theta[None, :].astype(np.float32)
        Js, gs = [], []
        for m in self.members:
            c = predict_coefficients(m, th, device=self.dev, batch=1)[0]
            _, _, jac = predict_gradients(m, th, device=self.dev, batch=1)
            den = c[1] + d3.CD0
            Js.append(c[0] / den)
            gs.append(jac[0, 0] / den - c[0] / den ** 2 * jac[0, 1])
        Js, gs = np.asarray(Js), np.stack(gs)
        return float(Js.mean()), gs.mean(0).astype(float), 0.0, float(Js.std(ddof=1))


def main():
    if (OUT / "trajectory_points.json").exists():  # frozen files are never overwritten
        raise SystemExit(f"{OUT / 'trajectory_points.json'} already exists; refusing to overwrite frozen trajectories")
    lo, hi = d3.box()
    starts, _ = d3.sample(N_RUNS, SEED)
    ens = Ensemble(*load_members())
    runs, pts = [], []
    for k, th0 in enumerate(starts):
        log = optimise(th0, ens, Gate("never"), cfd=None, box=Box(lo, hi), data_points=np.zeros((0, len(lo))),
                       step0=STEP, n_steps=N_STEPS, cfd_budget=0)
        it = [np.asarray(t, dtype=float) for t in log.thetas]
        assert len(it) == N_STEPS + 1
        acc = [bool(d3.accepted(t)[0]) for t in it]
        runs.append({"start": k, "iterates": [t.round(10).tolist() for t in it], "accepted": acc,
                     "J_hat": [float(v) for v in log.J_hat]})
        pts += [{"case_id": f"T3_s{k:02d}_t{j:02d}", "theta": t.round(10).tolist(), "start": k, "step": j}
                for j, t in enumerate(it)]
        print("TRAJ", k, "accepted", sum(acc), "/", len(acc), "J_hat", round(log.J_hat[0], 3), "->", round(log.J_hat[-1], 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"n_runs": N_RUNS, "seed": SEED, "step": STEP, "n_steps": N_STEPS, "objective": "J = CL/(CD+0.008) (FNO ensemble mean)",
            "checkpoints": {f"seed{s}": hashlib.sha256((CK / f"seed{s}.pt").read_bytes()).hexdigest() for s in range(5)}}
    for name, obj in (("trajectories.json", {"meta": meta, "runs": runs}), ("trajectory_points.json", {"meta": meta, "points": pts})):
        (OUT / name).write_text(json.dumps(obj, indent=1))
        print("FROZEN", name, hashlib.sha256((OUT / name).read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
