"""Stage A, Step 9e: production-mesh verification at the Dataset V1 operating point, Re = 1000.

Why: the steady solver does not settle for NACA 4412 at Re 2000, alpha 4 deg on the P-fine mesh
(CL oscillating 0.205-0.227 after 4,000 iterations on Colab; same behaviour locally at CFL 50 and 10),
which suggests the flow is unsteady there (time-accurate confirmation: Step 9d, case R1). A suspect
case cannot serve as a steady verification case, so Dataset V1 is set at Re = 1000 (ADR-003, provisional
until the regime map) and Step 9b's far-field phase on that case was stopped and is replaced here.
(An earlier local time-accurate run was invalid: it was interrupted by SIGTERM at t = 1.0 and SU2 then
continued with zero inner iterations; it is recorded as a failed experiment and not used.)

Phase A  GCI for the P family (coarse, medium, fine; ratio 1.5) on NACA 4412, Re 1000, alpha 4 deg,
         Cauchy stop + settle-by-continuation (100-iteration spread < 1e-4).
Phase B  Far-field independence on the same case, P-medium inner mesh, R = 100, 200, 400 chords.

Usage: python scripts/run_stageA_re1000_verification.py [A] [B]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cfd.pipeline import ensure_su2, run_case  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402
from run_stageA_verification import gci  # noqa: E402

SU2_ZIP = ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
WORK = Path(os.environ.get("AF_WORK", "/content/runs"))
OUT = ROOT / "datasets" / "raw" / "stageA_verification"
REPORT = ROOT / "results" / "reports" / "stageA_step9e_re1000_verification.json"
P_SPEC = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015,
              h_wake=0.015, wake_length=4.0, h_wake_near=0.008, near_wake_length=1.0,
              growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
R = 1.5
LEVELS = {"coarse": R, "medium": 1.0, "fine": 1.0 / R}
GEOM, RE, AOA, TAG = {"m": 0.04, "p": 0.4, "t": 0.12}, 1000.0, 4.0, "naca4412_Re1000_a4"


def coeffs(rec: dict) -> dict:
    a = rec.get("assessment", {})
    cl, cd = a.get("CL"), a.get("CD")
    return {"CL": cl, "CD": cd, "LD": (cl / cd) if (cl is not None and cd) else None, "status": rec.get("status"),
            "cells": rec.get("mesh", {}).get("n_cells_2d"), "wall_s": rec.get("run", {}).get("wall_s"),
            "iters": a.get("iterations"), "rms_p": a.get("final_rms_pressure_log10"),
            "CL_spread100": a.get("CL_tail_rel_spread"), "CD_spread100": a.get("CD_tail_rel_spread"),
            "continuation_segments": len(rec.get("continuation", {}).get("segments", []))}


if __name__ == "__main__":
    phases = sys.argv[1:] or ["A", "B"]
    su2 = ensure_su2(SU2_ZIP, SU2_DIR)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    common = dict(out_root=OUT, work_root=WORK, su2=su2, su2_zip=SU2_ZIP, stage="A_verification_P_Re1000", conv="cauchy")
    if "A" in phases:
        res = {}
        for lvl, fac in LEVELS.items():
            res[lvl] = coeffs(run_case(f"V6_P_{TAG}_{lvl}", GEOM, RE, AOA, MeshSpec(**P_SPEC).scaled(fac, f"P_{lvl}"), **common))
            print("P9E_A", lvl, json.dumps(res[lvl]), flush=True)
        ok = all(res[k]["status"] == "converged" for k in LEVELS)
        report["gci_P"] = {"levels": res, "gci": {q: gci(res["fine"][q], res["medium"][q], res["coarse"][q], R)
                                                  for q in ("CL", "CD", "LD")} if ok else "incomplete"}
        print("P9E_GCI", json.dumps(report["gci_P"]["gci"]), flush=True)
        save_json(REPORT, report)
    if "B" in phases:
        far = {}
        for radius in (100.0, 200.0, 400.0):
            spec = MeshSpec(**{**P_SPEC, "farfield_radius": radius, "label": f"P_R{int(radius)}"})
            far[f"R{int(radius)}"] = coeffs(run_case(f"V7_P_{TAG}_R{int(radius)}", GEOM, RE, AOA, spec, **common))
            print("P9E_B", radius, json.dumps(far[f"R{int(radius)}"]), flush=True)
        ref = far.get("R400", {})
        for v in far.values():
            if ref.get("CL") and v.get("CL"):
                v["CL_rel_to_R400"] = (v["CL"] - ref["CL"]) / ref["CL"]
                v["CD_rel_to_R400"] = (v["CD"] - ref["CD"]) / ref["CD"]
        report["farfield_P"] = far
        save_json(REPORT, report)
    print("AF_STEP9E_DONE", flush=True)
