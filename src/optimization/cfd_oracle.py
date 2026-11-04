"""CFD oracle for optimiser runs: primal (P-coarse, settled) + discrete-adjoint gradients at any theta.

Same settings as Dataset V1 (ADR-003), so oracle values are directly comparable with the training data.
Every call is a recorded case under datasets/raw/optimizer_calls/<case_id> (never overwritten; results
are cached by a hash of theta, so a repeated query costs nothing and returns the identical record).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from cfd.gradients import adjoint_gradients
from cfd.pipeline import ensure_su2, existing, run_case
from cfd.su2_case import save_json
from optimization.objective import objective
from meshing.gmsh_airfoil import MeshSpec

P_SPEC = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015,
              h_wake=0.015, wake_length=4.0, h_wake_near=0.008, near_wake_length=1.0,
              growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
RE = 1000.0


class CFDOracle:
    def __init__(self, root: Path, work: Path, su2_ad: str, tag: str = "opt"):
        self.root = Path(root)
        self.out = self.root / "datasets" / "raw" / "optimizer_calls"
        self.work = Path(work)
        self.su2_zip = self.root / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
        self.su2 = ensure_su2(self.su2_zip, Path(os.environ.get("AF_SU2_DIR", "/content/su2")))
        self.su2_ad = su2_ad
        self.env = dict(os.environ, LD_LIBRARY_PATH=str(Path(su2_ad).resolve().parents[1] / "lib"))
        self.spec = MeshSpec(**P_SPEC).scaled(1.5, "P_coarse")
        self.tag = tag
        self.n_calls = 0
        self.locks = self.out / ".locks"
        self.locks.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _lock(self, cid: str):
        """Exclusive per-case lock, so parallel optimiser processes never run the same case twice:
        a second process waits, then finds the finished record and reuses it."""
        with open(self.locks / f"{cid}.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    @staticmethod
    def case_id(theta: np.ndarray) -> str:
        key = ",".join(f"{v:.8f}" for v in np.asarray(theta, dtype=float))
        return "OC_" + hashlib.sha256(key.encode()).hexdigest()[:16]

    def primal(self, theta: np.ndarray) -> dict:
        th = np.asarray(theta, dtype=float)
        cid = self.case_id(th)
        with self._lock(cid):
            rec = existing(self.out / cid)
            if rec is None:
                rec = run_case(cid, {"m": th[0], "p": th[1], "t": th[2]}, RE, th[3], self.spec, self.out, self.work,
                               self.su2, self.su2_zip, f"G5_optimizer_{self.tag}", conv="cauchy")
                self.n_calls += 1
        if rec.get("status") != "converged":
            raise RuntimeError(f"{cid}: CFD did not converge ({rec.get('status')}); recorded, not used")
        return rec

    def value(self, theta: np.ndarray) -> float:
        """Objective J (protocol §5) from a primal solve only (used for post-hoc verification)."""
        a = self.primal(theta)["assessment"]
        return objective(a["CL"], a["CD"], np.zeros(4), np.zeros(4), float(theta[2]))[0]

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective J and its gradient (primal + discrete adjoints of CL and CD)."""
        th = np.asarray(theta, dtype=float)
        rec = self.primal(th)
        cid = rec["case_id"]
        gfile = self.out / cid / "gradients.json"
        with self._lock(cid + "_adj"):
            if gfile.exists():
                g = json.loads(gfile.read_text())
            else:
                g = adjoint_gradients(self.out / cid, self.work / f"{cid}_adj", th, self.su2_ad, env=self.env)
                save_json(gfile, g)
        a = rec["assessment"]
        dcl = np.array([g["objectives"]["CL"]["grad"][k] for k in ("m", "p", "t", "aoa_deg")])
        dcd = np.array([g["objectives"]["CD"]["grad"][k] for k in ("m", "p", "t", "aoa_deg")])
        return objective(a["CL"], a["CD"], dcl, dcd, float(th[2]))
