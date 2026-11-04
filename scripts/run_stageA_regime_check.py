"""Stage A, Step 9d: time-accurate regime check on the production mesh family (ADR-001 item 4).

Why: on the P-fine mesh the steady solver does not settle for NACA 4412, Re 2000, alpha 4 deg
(CL oscillates 0.205-0.227 after 3,000+ iterations on Colab; a local run on a different mesh
realisation shows the same growing oscillation, also at CFL 10). Implicit pseudo-time stepping
with large local steps can converge to an unstable steady state on coarser meshes, so the steady
results there do not prove the flow is steady. A time-accurate run from an impulsive start decides.

Cases (dual time stepping, dt = 0.05 c/U, 30 convective times, 8 inner iterations):
  R1  NACA 4412, Re 2000, a 4  on P-medium  (the suspect case)
  R2  NACA 4412, Re 2000, a 4  on P-coarse  (does the cheaper mesh capture the same regime?)
  R3  NACA 0012, Re 1000, a 4  on P-coarse  (control; expected steady)

Usage: python scripts/run_stageA_regime_check.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

from cfd.pipeline import ensure_su2, run_case  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402

SU2_ZIP = ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
WORK = Path(os.environ.get("AF_WORK", "/content/runs"))
OUT = ROOT / "datasets" / "raw" / "stageA_verification"
REPORT = ROOT / "results" / "reports" / "stageA_step9d_regime_check.json"
P_SPEC = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015,
              h_wake=0.015, wake_length=4.0, h_wake_near=0.008, near_wake_length=1.0,
              growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
UNSTEADY = dict(dt=0.05, t_end=30.0, inner=8)
CASES = (("R1_naca4412_Re2000_a4_Pmedium", {"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, 1.0, "P_medium"),
         ("R2_naca4412_Re2000_a4_Pcoarse", {"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, 1.5, "P_coarse"),
         ("R3_naca0012_Re1000_a4_Pcoarse", {"m": 0.0, "p": 0.0, "t": 0.12}, 1000.0, 4.0, 1.5, "P_coarse"))

if __name__ == "__main__":
    su2 = ensure_su2(SU2_ZIP, SU2_DIR)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {"settings": UNSTEADY, "cases": {}}
    for cid, geom, re, aoa, fac, label in CASES:
        spec = MeshSpec(**P_SPEC).scaled(fac, label)
        rec = run_case(cid, geom, re, aoa, spec, OUT, WORK, su2, SU2_ZIP, "A_regime_check_P", unsteady=UNSTEADY)
        out = {"status": rec.get("status"), "cells": rec.get("mesh", {}).get("n_cells_2d"),
               "wall_s": rec.get("run", {}).get("wall_s"), "regime": rec.get("regime")}
        report["cases"][cid] = out
        save_json(REPORT, report)
        print("RC", cid, json.dumps(out), flush=True)
    print("AF_STEP9D_DONE", flush=True)
