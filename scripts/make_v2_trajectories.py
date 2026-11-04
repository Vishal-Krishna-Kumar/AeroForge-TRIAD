"""Protocol v2 §4: the 40 surrogate-only optimizer runs whose iterates get CFD ground truth in pod session 2.

Starts: first 40 accepted points of a scrambled Sobol stream (seed 20261023) in the V2 box (same acceptance as the
dataset). Optimizer: maximise L/D with the FNO-ensemble-mean gradient ĝ (5 members, checkpoints/v2/fno), normalized
step h = 0.05 in box-scaled coordinates (u = theta / span), projection onto the box, 14 steps -> 15 iterates per run
(src/optimization/gated_optimizer.optimise with gate "never"). Iterates leaving the acceptance set are kept and
flagged. Output (frozen, sha256 printed and logged BEFORE any of these points is simulated):
  results/v2_traj/trajectories.json  {runs: [{start, iterates: [theta...], accepted: [bool...]}]}
  results/v2_traj/trajectory_points.json  {points: [{case_id, theta, start, step}]}  (input of pod session 2)
Usage (Colab, GPU optional): python scripts/make_v2_trajectories.py
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

import run_dataset_v2 as d2  # noqa: E402
from data.postprocess import GridSpec  # noqa: E402
from models.training import TrainConfig, build_model, load_split, predict_gradients  # noqa: E402
from optimization.gated_optimizer import Box, Gate, optimise  # noqa: E402

N_RUNS, SEED, STEP, N_STEPS = 40, 20261023, 0.05, 14
CK = ROOT / "checkpoints" / "v2" / "fno"
OUT = ROOT / "results" / "v2_traj"
HP = dict(batch=16, lr=1e-3, width=32, modes=(16, 24), layers=4)   # = scripts/train_surrogates_v2.HP["fno"]


def load_members():
    gx, gy = GridSpec().coords()
    fit = load_split(ROOT / "datasets" / "processed" / "dataset_v2" / "train", with_fields=False)
    n_fit = int(json.loads((CK / "seed0.json").read_text())["n_fit"])   # = training fit set (protocol v2 §9 budget rule)
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
        out = [predict_gradients(m, theta[None, :].astype(np.float32), device=self.dev, batch=1) for m in self.members]
        ld = np.array([o[0][0] for o in out]); g = np.stack([o[1][0] for o in out])
        return float(ld.mean()), g.mean(0).astype(float), 0.0, float(ld.std(ddof=1))


def main():
    if (OUT / "trajectory_points.json").exists():   # frozen files are never overwritten
        raise SystemExit(f"{OUT / 'trajectory_points.json'} already exists; refusing to overwrite frozen trajectories")
    lo, hi = d2.box()
    starts, _ = d2.sample(N_RUNS, SEED)
    ens = Ensemble(*load_members())
    runs, pts = [], []
    for k, th0 in enumerate(starts):
        log = optimise(th0, ens, Gate("never"), cfd=None, box=Box(lo, hi), data_points=np.zeros((0, len(lo))),
                       step0=STEP, n_steps=N_STEPS, cfd_budget=0)
        it = [np.asarray(t, dtype=float) for t in log.thetas]
        assert len(it) == N_STEPS + 1
        acc = [bool(d2.accepted(t)[0]) for t in it]
        runs.append({"start": k, "iterates": [t.round(10).tolist() for t in it], "accepted": acc,
                     "J_hat": [float(v) for v in log.J_hat]})
        pts += [{"case_id": f"T2_s{k:02d}_t{j:02d}", "theta": t.round(10).tolist(), "start": k, "step": j}
                for j, t in enumerate(it)]
        print("TRAJ", k, "accepted", sum(acc), "/", len(acc), "J_hat", round(log.J_hat[0], 3), "->", round(log.J_hat[-1], 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"n_runs": N_RUNS, "seed": SEED, "step": STEP, "n_steps": N_STEPS, "objective": "L/D (FNO ensemble mean)",
            "checkpoints": {f"seed{s}": hashlib.sha256((CK / f"seed{s}.pt").read_bytes()).hexdigest() for s in range(5)}}
    for name, obj in (("trajectories.json", {"meta": meta, "runs": runs}), ("trajectory_points.json", {"meta": meta, "points": pts})):
        (OUT / name).write_text(json.dumps(obj, indent=1))
        print("FROZEN", name, hashlib.sha256((OUT / name).read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
