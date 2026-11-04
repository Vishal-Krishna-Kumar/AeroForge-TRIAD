"""Stage A, Step 9b: verification of the candidate PRODUCTION mesh family "P" (ADR-002 draft).

Why: Step 9 phase 1 showed the v1 medium mesh carries ~12-14 % error in CL and L/D at Re 2000
(NACA 4412), driven by wake, boundary-layer and outer-field resolution (local dev experiments).

Phase A  GCI for the P family (coarse = P x1.5, P, fine = P / 1.5) on the same two test cases as Step 9
         (NACA 0012 Re 1000 a4; NACA 4412 Re 2000 a4), coefficient (Cauchy) convergence.
Phase B  Far-field independence with the v2 graded mesh: R = 100, 200, 400 (inner mesh unchanged).
Phase C  Re-classify the Step 9 phase-3 histories with classifier v2 (no new solves).

Usage: python scripts/run_stageA_production_verification.py [A] [B] [C]   (default: A B C)
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
from cfd.su2_unsteady import classify  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402
from run_stageA_verification import gci  # noqa: E402

SU2_ZIP = ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
WORK = Path(os.environ.get("AF_WORK", "/content/runs"))
OUT = ROOT / "datasets" / "raw" / "stageA_verification"
REPORT = ROOT / "results" / "reports" / "stageA_step9b_production_mesh.json"

P_SPEC = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015,
              h_wake=0.015, wake_length=4.0, h_wake_near=0.008, near_wake_length=1.0,
              growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
R = 1.5
LEVELS = {"coarse": R, "medium": 1.0, "fine": 1.0 / R}
CASES = (("naca0012_Re1000_a4", {"m": 0.0, "p": 0.0, "t": 0.12}, 1000.0, 4.0),
         ("naca4412_Re2000_a4", {"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0))


def coeffs(rec: dict) -> dict:
    a = rec.get("assessment", {})
    cl, cd = a.get("CL"), a.get("CD")
    return {"CL": cl, "CD": cd, "LD": (cl / cd) if (cl is not None and cd) else None,
            "status": rec.get("status"), "cells": rec.get("mesh", {}).get("n_cells_2d"),
            "wall_s": rec.get("run", {}).get("wall_s"), "iters": a.get("iterations"),
            "rms_p": a.get("final_rms_pressure_log10")}


def main(phases: list[str]) -> None:
    su2 = ensure_su2(SU2_ZIP, SU2_DIR)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    common = dict(out_root=OUT, work_root=WORK, su2=su2, su2_zip=SU2_ZIP, stage="A_verification_P",
                  conv="cauchy")
    if "A" in phases:
        studies = report.get("gci_P", {})
        for tag, geom, re, aoa in CASES:
            res = {}
            for lvl, fac in LEVELS.items():
                spec = MeshSpec(**P_SPEC).scaled(fac, f"P_{lvl}")
                res[lvl] = coeffs(run_case(f"V4_P_{tag}_{lvl}", geom, re, aoa, spec, **common))
                print("PA", tag, lvl, json.dumps(res[lvl]), flush=True)
            ok = all(res[k]["status"] == "converged" for k in LEVELS)
            studies[tag] = {"levels": res, "gci": {q: gci(res["fine"][q], res["medium"][q], res["coarse"][q], R)
                                                   for q in ("CL", "CD", "LD")} if ok else "incomplete"}
            print("PA_GCI", tag, json.dumps(studies[tag]["gci"]), flush=True)
            report["gci_P"] = studies
            save_json(REPORT, report)
    if "B" in phases:
        far = {}
        for radius in (100.0, 200.0, 400.0):
            spec = MeshSpec(**{**P_SPEC, "farfield_radius": radius, "label": f"P_R{int(radius)}"})
            far[f"R{int(radius)}"] = coeffs(run_case(f"V5_P_naca4412_Re2000_a4_R{int(radius)}",
                                                      {"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, spec, **common))
            print("PB", radius, json.dumps(far[f"R{int(radius)}"]), flush=True)
        ref = far.get("R400", {})
        for v in far.values():
            if ref.get("CL") and v.get("CL"):
                v["CL_rel_to_R400"] = (v["CL"] - ref["CL"]) / ref["CL"]
                v["CD_rel_to_R400"] = (v["CD"] - ref["CD"]) / ref["CD"]
        report["farfield_P"] = far
        save_json(REPORT, report)
    if "C" in phases:
        recl = {}
        for d in sorted(OUT.glob("V3_*")):
            if (d / "history.csv").exists():
                try:
                    recl[d.name] = classify(d)
                except Exception as exc:
                    recl[d.name] = {"error": repr(exc)}
                print("PC", d.name, json.dumps(recl[d.name]), flush=True)
        report["regime_pilot_reclassified_v2"] = recl
        report["regime_pilot_note"] = ("Step 9 phase 3 ran on the v1 coarse mesh, which phase 1 showed to be inadequate "
                                       "at Re 2000 (CL error ~30 %); its regime labels are provisional and are superseded "
                                       "by a regime map on the production mesh.")
        save_json(REPORT, report)
    print("AF_STEP9B_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
