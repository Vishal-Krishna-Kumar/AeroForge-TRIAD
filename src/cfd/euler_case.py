"""V3 case runner: CST geometry -> Gmsh mesh -> SU2 Euler -> assessment -> persisted record (protocol v3).

Same persistence/provenance conventions as cfd.pipeline (case.json with versions, checksums, timings; resumable;
failed attempts archived, never overwritten). Convergence rule (protocol v3 §2): rms[rho] <= 1e-10 within the
iteration cap AND the last 100 iterations settled (CL relative spread < 1e-4, CD absolute spread < 1e-6 —
absolute for CD because inviscid CD can be ~0). Anything else is kept and flagged 'not_converged'.
"""
from __future__ import annotations

import datetime
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from cfd.pipeline import _archive_failed, _persist, existing, sha256
from cfd.su2_case import read_history, run_su2
from cfd.su2_euler import RES_KEY, write_euler_config
from meshing.gmsh_airfoil import MeshSpec, build_mesh_isolated

KEEP_EULER = ("mesh.su2", "config.cfg", "history.csv", "solver.log", "surface_flow.csv", "surface_flow.vtk",
              "restart_flow.dat")
SU2_THREADS = int(os.environ.get("AF_SU2_THREADS", "2"))


def assess_euler(case_dir: Path, res_min: float = -10.0, window: int = 100) -> dict:
    h = read_history(case_dir)
    r, cl, cd = h[RES_KEY], h["CL"], h["CD"]
    w = min(window, len(r))
    return {"iterations": int(len(r)), "final_rms_rho_log10": float(r[-1]), "residual_drop_orders": float(r[0] - r[-1]),
            "reached_res_min": bool(r[-1] <= res_min), "CL": float(cl[-1]), "CD": float(cd[-1]),
            "CL_tail_rel_spread": float(np.ptp(cl[-w:]) / max(abs(cl[-w:].mean()), 1e-12)),
            "CD_tail_abs_spread": float(np.ptp(cd[-w:])), "finite": bool(np.isfinite(cl[-1]) and np.isfinite(cd[-1])),
            "tail_window": int(w)}


def run_case_euler(case_id: str, au, al, aoa: float, mach: float, spec: MeshSpec, out_root: Path, work_root: Path,
                   su2: Path, su2_zip: Path, stage: str, iters: int = 3000, res_min: float = -10.0,
                   force: bool = False) -> dict:
    from geometry.cst import cst_surfaces, shape_metrics
    dest = out_root / case_id
    if not force:
        prev = existing(dest)
        if prev is not None:
            prev["resumed"] = True
            return prev
    _archive_failed(dest, out_root)
    wd = work_root / case_id
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    rec = {"case_id": case_id, "stage": stage, "geometry": {"family": "CST", "au": list(map(float, au)), "al": list(map(float, al))},
           "conditions": {"Mach": mach, "aoa_deg": aoa, "model": "2D compressible Euler, steady (JST, MG3)"},
           "mesh_spec": spec.__dict__ if hasattr(spec, "__dict__") else str(spec),
           "su2_zip_sha256": sha256(su2_zip) if Path(su2_zip).exists() else None, "python": platform.python_version(),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    up, lo = cst_surfaces(np.asarray(au, float), np.asarray(al, float))
    rec["geometry"]["checks"] = shape_metrics(np.asarray(au, float), np.asarray(al, float))
    try:
        t0 = time.time()
        rec["mesh"] = build_mesh_isolated(up, lo, str(wd / "mesh.su2"), spec, aoa_deg=aoa)
        rec["mesh_wall_s"] = round(time.time() - t0, 1)
    except Exception as exc:  # recorded, never hidden
        rec["status"], rec["error"] = "mesh_failed", repr(exc)
        return _persist(wd, dest, rec, KEEP_EULER)
    write_euler_config(wd, mach, iters=iters, res_min=res_min)
    rec["run"] = run_su2(wd, str(su2), threads=SU2_THREADS, timeout_s=3 * 3600)
    rec["run"]["threads"] = SU2_THREADS
    if rec["run"]["returncode"] != 0 or not (wd / "history.csv").exists():
        rec["status"] = "solver_failed"
        return _persist(wd, dest, rec, KEEP_EULER)
    a = assess_euler(wd, res_min)
    rec["assessment"] = a
    settled = a["CL_tail_rel_spread"] < 1e-4 and a["CD_tail_abs_spread"] < 1e-6
    rec["status"] = "converged" if (a["finite"] and a["reached_res_min"] and settled) else "not_converged"
    return _persist(wd, dest, rec, KEEP_EULER)
