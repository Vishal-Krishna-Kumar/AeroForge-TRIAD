"""Adjoint gradients at the CFD-verified points of the H3 optimiser trajectories (data for H2 trajectory
coverage and H4; protocol §1, §4, §6).

Collects every verified iterate (every 5th step and the final one) of every finished H3 run
(results/h3/<gate>_B<budget>_start<k>.json), de-duplicates by CFD case id, and computes the primal (cached
from the verification) plus the discrete adjoints of CL and CD with the same oracle as the in-loop calls.
Runs in a process pool; the oracle's per-case locks make concurrent use safe. Failures are written to
results/h3/trajectory_gradients_failed.json, never dropped.

Order and cost cap (fixed before any trajectory gradient exists, so the cap cannot select by result):
points are processed gate by gate in the order never, cdgc, output_uq, distance, always (budget 5 before 10,
start 0..9, step order); with --deadline-utc no new point is started after that time, and the summary lists
which points were not computed.

Usage: python scripts/run_trajectory_gradients.py --workers 14 [--deadline-utc 2026-10-13T20:00:00]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from optimization.cfd_oracle import CFDOracle  # noqa: E402

OUT = ROOT / "results" / "h3"
_oracle = None


def _init():
    global _oracle
    _oracle = CFDOracle(ROOT, Path(os.environ.get("AF_WORK", "/content/runs")) / "h3",
                        os.environ.get("AF_SU2_AD", "/content/mamba/su2ad/bin/SU2_CFD_AD"))


def _grad(theta: list[float]) -> tuple[str, str | None]:
    th = np.asarray(theta, dtype=float)
    try:
        _oracle(th)
        return CFDOracle.case_id(th), None
    except Exception as exc:
        return CFDOracle.case_id(th), repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deadline-utc", default=None, help="ISO time (UTC) after which no new point is started")
    args = ap.parse_args()
    deadline = datetime.datetime.fromisoformat(args.deadline_utc).replace(tzinfo=datetime.timezone.utc) if args.deadline_utc else None
    order = {g: i for i, g in enumerate(["never", "cdgc", "output_uq", "distance", "always"])}
    runs = []
    for f in OUT.glob("*_B*_start*.json"):
        if f.name.endswith(".failed.json"):
            continue
        rec = json.loads(f.read_text())
        runs.append((order.get(rec["gate"], 99), rec["budget"], int(f.stem.rsplit("start", 1)[1]), rec))
    pts: dict[str, list[float]] = {}
    for _, _, _, rec in sorted(runs, key=lambda r: r[:3]):
        for i in rec["verified_steps"]:
            th = rec["thetas"][i]
            pts.setdefault(CFDOracle.case_id(np.asarray(th)), th)
    print("TRAJ_POINTS", len(pts), flush=True)
    failed, not_started, done = {}, [], 0
    todo = list(pts.items())
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        running = set()
        while todo or running:
            while todo and len(running) < args.workers:
                if deadline and datetime.datetime.now(datetime.timezone.utc) > deadline:
                    not_started = [cid for cid, _ in todo]; todo = []
                    print("TRAJ_DEADLINE reached; not started:", len(not_started), flush=True)
                    break
                cid, th = todo.pop(0)
                running.add(ex.submit(_grad, th))
            if not running:
                break
            fu = next(as_completed(running))
            running.discard(fu)
            cid, err = fu.result(); done += 1
            if err:
                failed[cid] = err
            print("TRAJ_GRAD", done, len(pts), cid, err or "ok", flush=True)
    (OUT / "trajectory_gradients_failed.json").write_text(json.dumps({"failed": failed, "not_started_deadline": not_started}, indent=1))
    print("AF_TRAJ_DONE", len(pts), "failed", len(failed), "not_started", len(not_started), flush=True)


if __name__ == "__main__":
    main()
