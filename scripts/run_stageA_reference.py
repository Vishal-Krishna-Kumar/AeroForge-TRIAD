"""Stage A, Step 8: reference SU2 simulations (pipeline verification, NOT dataset ground truth).

Cases: NACA 0012, Re = 1000, alpha = 0 and 4 deg, medium mesh, far field 50c.
Outputs: raw solver files -> datasets/raw/stageA_pipeline/<case_id>/, summary -> results/reports/.
Usage (Colab): python scripts/run_stageA_reference.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

from geometry.naca import naca4, geometry_checks  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec, build_mesh  # noqa: E402
from cfd.su2_case import write_config, run_su2, assess, integrate_surface, save_json  # noqa: E402

SU2_ZIP = ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
WORK = Path(os.environ.get("AF_WORK", "/content/runs"))
OUT = ROOT / "datasets" / "raw" / "stageA_pipeline"
RES_MIN = -10.0


def ensure_su2() -> Path:
    exe = SU2_DIR / "bin" / "SU2_CFD"
    if not exe.exists():
        SU2_DIR.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(SU2_ZIP).extractall(SU2_DIR)
        for inner in list(SU2_DIR.rglob("*.zip")):
            zipfile.ZipFile(inner).extractall(inner.parent)
            inner.unlink()
        for p in SU2_DIR.rglob("SU2_*"):
            if p.is_file():
                p.chmod(0o755)
    return exe


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case_id: str, m: float, p: float, t: float, re: float, aoa: float, spec: MeshSpec,
             su2: Path) -> dict:
    wd = WORK / case_id
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    up, lo = naca4(m, p, t)
    gchk = geometry_checks(up, lo, t)
    t0 = time.time()
    mstat = build_mesh(up, lo, str(wd / "mesh.su2"), spec, aoa_deg=aoa)
    mesh_s = round(time.time() - t0, 1)
    write_config(wd, re, iters=4000, cfl=50.0, res_min=RES_MIN)
    run = run_su2(wd, str(su2), threads=2, timeout_s=3600)
    rec = {"case_id": case_id, "stage": "A_pipeline_verification",
           "geometry": {"family": "NACA4", "m": m, "p": p, "t": t, "checks": gchk},
           "conditions": {"Re": re, "aoa_deg": aoa, "model": "steady 2D incompressible laminar NS"},
           "mesh": mstat, "mesh_wall_s": mesh_s, "run": run, "su2_zip_sha256": sha256(SU2_ZIP),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "python": platform.python_version()}
    if run["returncode"] == 0 and (wd / "history.csv").exists():
        a = assess(wd, RES_MIN)
        rec["assessment"] = a
        try:
            rec["independent_integration"] = integrate_surface(wd)
            ii = rec["independent_integration"]
            rec["consistency"] = {
                "CL_rel_diff": abs(ii["CL_total"] - a["CL"]) / max(abs(a["CL"]), 1e-12),
                "CD_rel_diff": abs(ii["CD_total"] - a["CD"]) / max(abs(a["CD"]), 1e-12)}
        except Exception as exc:  # recorded, not hidden
            rec["independent_integration_error"] = repr(exc)
        rec["status"] = "converged" if a["reached_res_min"] and a["finite"] else "not_converged"
    else:
        rec["status"] = "solver_failed"
    dest = OUT / case_id
    if dest.exists():
        shutil.rmtree(dest)  # only our own Stage A pipeline outputs, regenerated deterministically
    dest.mkdir(parents=True)
    for name in ("mesh.su2", "config.cfg", "history.csv", "solver.log", "surface_flow.csv",
                 "surface_flow.vtk", "flow.vtu", "restart_flow.dat"):
        if (wd / name).exists():
            shutil.copy2(wd / name, dest / name)
    rec["files_sha256"] = {f.name: sha256(f) for f in sorted(dest.iterdir()) if f.is_file()}
    save_json(dest / "case.json", rec)
    return rec


def main() -> None:
    su2 = ensure_su2()
    spec = MeshSpec()  # medium, far field 50c
    cases = [("A001_naca0012_Re1000_a0", 0.0, 0.0, 0.12, 1000.0, 0.0),
             ("A002_naca0012_Re1000_a4", 0.0, 0.0, 0.12, 1000.0, 4.0)]
    summary = []
    for cid, m, p, t, re, aoa in cases:
        r = run_case(cid, m, p, t, re, aoa, spec, su2)
        a = r.get("assessment", {})
        line = {"case_id": cid, "status": r["status"], "wall_s": r["run"]["wall_s"],
                "iters": a.get("iterations"), "rms_p": a.get("final_rms_pressure_log10"),
                "CL": a.get("CL"), "CD": a.get("CD"),
                "CL_indep": r.get("independent_integration", {}).get("CL_total"),
                "CD_indep": r.get("independent_integration", {}).get("CD_total"),
                "cells": r["mesh"]["n_cells_2d"], "minSICN": r["mesh"]["min_SICN"]}
        summary.append(line)
        print("CASE", json.dumps(line), flush=True)
    rep = ROOT / "results" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    save_json(rep / "stageA_step8_reference_summary.json", {"cases": summary, "note": (
        "Stage A pipeline verification only. Not publication ground truth. Validation reference: "
        "Kurtulus 2015 (IJMAV 7(3):301-326) reports Cd ~ 0.12 at alpha=0 and lift slope ~ pi per rad "
        "for NACA0012 at Re=1000 (Fluent, far field 15c).")})
    print("AF_STEP8_DONE", flush=True)


if __name__ == "__main__":
    main()
