"""Protocol v3 amendment 4 (i): the 40 extension runs (runs 40-79) for phase 3b.

Identical to scripts/make_v3_trajectories.py (imported, not copied: same FNO ensemble, optimiser, step 0.05, 11 steps)
except the starts: accepted Sobol points 41-80 of the same stream (seed 20261033; d3.sample(80) is prefix-stable, so its first
40 points are the existing runs 0-39, which this script checks against results/v3_traj/trajectories.json before going on).
Output (frozen, never overwritten): results/v3_traj_ext/trajectories.json, results/v3_traj_ext/trajectory_points.json
(case ids T3_s40_t00 ... T3_s79_t11, start = run index 40-79).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("AF_TRAJ_OUT", str(ROOT / "results" / "v3_traj_ext"))
import numpy as np  # noqa: E402

import make_v3_trajectories as T  # noqa: E402
from optimization.gated_optimizer import Box, Gate, optimise  # noqa: E402

FIRST, N_EXT = 40, 40


def main():
    out = T.OUT
    if (out / "trajectory_points.json").exists():
        raise SystemExit(f"{out / 'trajectory_points.json'} already exists; refusing to overwrite frozen trajectories")
    lo, hi = T.d3.box()
    allstarts, _ = T.d3.sample(FIRST + N_EXT, T.SEED)
    old = json.loads((ROOT / "results" / "v3_traj" / "trajectories.json").read_text())["runs"]
    assert len(old) == FIRST and all(np.allclose(np.asarray(r["iterates"][0]), allstarts[k], atol=1e-8) for k, r in enumerate(old)), \
        "Sobol prefix differs from the frozen runs 0-39"
    ens = T.Ensemble(*T.load_members())
    runs, pts = [], []
    for k in range(FIRST, FIRST + N_EXT):
        log = optimise(allstarts[k], ens, Gate("never"), cfd=None, box=Box(lo, hi), data_points=np.zeros((0, len(lo))),
                       step0=T.STEP, n_steps=T.N_STEPS, cfd_budget=0)
        it = [np.asarray(t, dtype=float) for t in log.thetas]
        assert len(it) == T.N_STEPS + 1
        acc = [bool(T.d3.accepted(t)[0]) for t in it]
        runs.append({"start": k, "iterates": [t.round(10).tolist() for t in it], "accepted": acc, "J_hat": [float(v) for v in log.J_hat]})
        pts += [{"case_id": f"T3_s{k:02d}_t{j:02d}", "theta": t.round(10).tolist(), "start": k, "step": j} for j, t in enumerate(it)]
        print("TRAJ", k, "accepted", sum(acc), "/", len(acc), "J_hat", round(log.J_hat[0], 3), "->", round(log.J_hat[-1], 3), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"amendment": 4, "runs": [FIRST, FIRST + N_EXT - 1], "seed": T.SEED, "step": T.STEP, "n_steps": T.N_STEPS,
            "objective": "J = CL/(CD+0.008) (FNO ensemble mean)",
            "checkpoints": {f"seed{s}": hashlib.sha256((T.CK / f"seed{s}.pt").read_bytes()).hexdigest() for s in range(5)}}
    for name, obj in (("trajectories.json", {"meta": meta, "runs": runs}), ("trajectory_points.json", {"meta": meta, "points": pts})):
        (out / name).write_text(json.dumps(obj, indent=1))
        print("FROZEN", name, hashlib.sha256((out / name).read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
