"""Stage A, Step 9: numerical verification of the CFD set-up (ADR-001 obligations).

Phase 1  Mesh independence: coarse / medium / fine (uniform spacing factor 1.5 between levels) for
         (a) NACA 0012, Re 1000, alpha 4 deg and (b) NACA 4412, Re 2000, alpha 4 deg.
         Grid Convergence Index (Roache; Celik et al. 2008 procedure) for CL, CD and L/D.
Phase 2  Far-field independence: radius 25c / 50c / 100c (medium mesh), NACA 0012, Re 1000, alpha 4.
Phase 3  Regime pilot (time-accurate, impulsive start, coarse mesh): envelope corners at Re 2000 and a
         low-Re control. Classifies each case as steady or unsteady (vortex shedding).

All cases are resumable and persisted to datasets/raw/stageA_verification/<case_id>/.
Usage: python scripts/run_stageA_verification.py [phase ...]   (default: 1 2 3)
"""
from __future__ import annotations

import json
import math
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
REPORT = ROOT / "results" / "reports" / "stageA_step9_verification.json"

R_REFINE = 1.5
LEVELS = {"coarse": R_REFINE, "medium": 1.0, "fine": 1.0 / R_REFINE}
NACA0012 = {"m": 0.0, "p": 0.0, "t": 0.12}
NACA4412 = {"m": 0.04, "p": 0.4, "t": 0.12}


def gci(f1: float, f2: float, f3: float, r: float = R_REFINE) -> dict:
    """Celik et al. (2008) GCI with constant refinement ratio r. f1 = fine, f3 = coarse."""
    e21, e32 = f2 - f1, f3 - f2
    out = {"fine": f1, "medium": f2, "coarse": f3}
    if e21 == 0 or e32 == 0 or e32 / e21 <= 0:
        out["note"] = "oscillatory or exact convergence; observed order undefined"
        out["gci_fine_pct"] = 100.0 * 1.25 * abs(e21) / max(abs(f1), 1e-12) if e21 else 0.0
        return out
    p = abs(math.log(abs(e32 / e21))) / math.log(r)
    f_ext = (r**p * f1 - f2) / (r**p - 1)
    out.update({"observed_order_p": p, "richardson_extrapolated": f_ext,
                "gci_fine_pct": 100.0 * 1.25 * abs(e21 / f1) / (r**p - 1),
                "gci_medium_pct": 100.0 * 1.25 * abs(e32 / f2) / (r**p - 1),
                "asymptotic_ratio": (1.25 * abs(e32 / f2) / (r**p - 1)) / (r**p * 1.25 * abs(e21 / f1) / (r**p - 1))})
    return out


def coeffs(rec: dict) -> dict:
    a = rec.get("assessment", {})
    cl, cd = a.get("CL"), a.get("CD")
    return {"CL": cl, "CD": cd, "LD": (cl / cd) if (cl is not None and cd) else None,
            "status": rec.get("status"), "cells": rec.get("mesh", {}).get("n_cells_2d"),
            "wall_s": rec.get("run", {}).get("wall_s"), "iters": a.get("iterations")}


def main(phases: list[str]) -> None:
    su2 = ensure_su2(SU2_ZIP, SU2_DIR)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    common = dict(out_root=OUT, work_root=WORK, su2=su2, su2_zip=SU2_ZIP, stage="A_verification")

    if "1" in phases:
        studies = {}
        for tag, geom, re, aoa in (("naca0012_Re1000_a4", NACA0012, 1000.0, 4.0),
                                   ("naca4412_Re2000_a4", NACA4412, 2000.0, 4.0)):
            res = {}
            for lvl, fac in LEVELS.items():
                spec = MeshSpec().scaled(fac, lvl)
                rec = run_case(f"V1_{tag}_{lvl}", geom, re, aoa, spec, **common)
                res[lvl] = coeffs(rec)
                print("P1", tag, lvl, json.dumps(res[lvl]), flush=True)
            ok = all(res[k]["status"] == "converged" for k in LEVELS)
            studies[tag] = {"levels": res, "gci": {q: gci(res["fine"][q], res["medium"][q], res["coarse"][q])
                                                   for q in ("CL", "CD", "LD")} if ok else "incomplete"}
            print("P1_GCI", tag, json.dumps(studies[tag]["gci"]), flush=True)
        report["mesh_independence"] = studies
        save_json(REPORT, report)

    if "2" in phases:
        far = {}
        for radius in (25.0, 50.0, 100.0):
            spec = MeshSpec(farfield_radius=radius, label=f"medium_R{int(radius)}")
            rec = run_case(f"V2_naca0012_Re1000_a4_R{int(radius)}", NACA0012, 1000.0, 4.0, spec, **common)
            far[f"R{int(radius)}"] = coeffs(rec)
            print("P2", radius, json.dumps(far[f"R{int(radius)}"]), flush=True)
        ref = far.get("R100", {})
        for k, v in far.items():
            if ref.get("CD") and v.get("CD"):
                v["CD_rel_to_R100"] = (v["CD"] - ref["CD"]) / ref["CD"]
                v["CL_rel_to_R100"] = (v["CL"] - ref["CL"]) / ref["CL"]
        report["farfield_independence"] = far
        save_json(REPORT, report)

    if "3" in phases:
        regime = {}
        uns = {"dt": 0.05, "t_end": 30.0, "inner": 10}
        coarse = MeshSpec().scaled(R_REFINE, "coarse")
        cases = [("V3_naca0012_Re2000_a6", NACA0012, 2000.0, 6.0),
                 ("V3_naca6412_Re2000_a6", {"m": 0.06, "p": 0.4, "t": 0.12}, 2000.0, 6.0),
                 ("V3_naca6412_Re2000_am4", {"m": 0.06, "p": 0.4, "t": 0.12}, 2000.0, -4.0),
                 ("V3_naca0008_Re2000_a6", {"m": 0.0, "p": 0.0, "t": 0.08}, 2000.0, 6.0),
                 ("V3_naca0018_Re2000_a6", {"m": 0.0, "p": 0.0, "t": 0.18}, 2000.0, 6.0),
                 ("V3_naca6412_Re500_a6", {"m": 0.06, "p": 0.4, "t": 0.12}, 500.0, 6.0)]
        for cid, geom, re, aoa in cases:
            rec = run_case(cid, geom, re, aoa, coarse, unsteady=uns, **common)
            regime[cid] = {"status": rec.get("status"), **rec.get("regime", {}),
                           "wall_s": rec.get("run", {}).get("wall_s")}
            print("P3", cid, json.dumps(regime[cid]), flush=True)
            report["regime_pilot"] = regime
            save_json(REPORT, report)
    print("AF_STEP9_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["1", "2", "3"])
