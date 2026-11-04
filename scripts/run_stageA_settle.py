"""Stage A, Step 9c: settle the Step 9b production-mesh cases and compute the GCI / far-field results.

Why: Step 9b stopped SU2 on its built-in Cauchy test (eps 1e-5). That test averages per-iteration
relative changes, which carry ~1e-6 jitter on the P mesh, so it stops before CL has settled to 1e-4
over 100 iterations (NACA 4412 P-coarse: CL spread 4.3e-4 at 622 iterations -> "not_converged").
Each Step 9b case (V4_*, V5_*) is therefore continued from its own persisted mesh and restart file
(cfd.pipeline.continue_case) until the 100-iteration relative spread of CL and CD is < 1e-4, and
recorded as a NEW case "<id>_S". The Step 9b records are kept unchanged as evidence.
Locally, a continued run reproduces an uninterrupted run bit-for-bit (CL 0.2092920717 both ways).

Usage: python scripts/run_stageA_settle.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cfd.pipeline import continue_case, ensure_su2  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402
from run_stageA_verification import gci  # noqa: E402

SU2_ZIP = ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
WORK = Path(os.environ.get("AF_WORK", "/content/runs"))
OUT = ROOT / "datasets" / "raw" / "stageA_verification"
REPORT = ROOT / "results" / "reports" / "stageA_step9b_production_mesh.json"
R = 1.5
TAGS = ("naca0012_Re1000_a4", "naca4412_Re2000_a4")
LEVELS = ("coarse", "medium", "fine")
FAR = ("R100", "R200", "R400")


def summary(rec: dict) -> dict:
    a = rec.get("assessment", {})
    cl, cd = a.get("CL"), a.get("CD")
    return {"CL": cl, "CD": cd, "LD": (cl / cd) if (cl is not None and cd) else None, "status": rec.get("status"),
            "cells": rec.get("mesh", {}).get("n_cells_2d"), "iters": a.get("iterations"),
            "rms_p": a.get("final_rms_pressure_log10"), "CL_spread100": a.get("CL_tail_rel_spread"),
            "CD_spread100": a.get("CD_tail_rel_spread"), "continuation_segments": len(rec.get("continuation", {}).get("segments", [])),
            "wall_s_total": rec.get("run", {}).get("wall_s_including_source_run"),
            "source_status": rec.get("continued_from", {}).get("status")}


def settle_one(src_id: str) -> dict | None:
    src = OUT / src_id
    if not (src / "case.json").exists():
        print("S9C_MISSING", src_id, flush=True)
        return None
    rec = continue_case(src, f"{src_id}_S", OUT, WORK, su2, "A_verification_P_settled")
    s = summary(rec)
    print("S9C", src_id, json.dumps(s), flush=True)
    return s


if __name__ == "__main__":
    su2 = ensure_su2(SU2_ZIP, SU2_DIR)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    gci_s = {}
    for tag in TAGS:
        res = {lvl: settle_one(f"V4_P_{tag}_{lvl}") for lvl in LEVELS}
        ok = all(res[l] and res[l]["status"] == "converged" for l in LEVELS)
        gci_s[tag] = {"levels": res, "gci": {q: gci(res["fine"][q], res["medium"][q], res["coarse"][q], R)
                                            for q in ("CL", "CD", "LD")} if ok else "incomplete"}
        print("S9C_GCI", tag, json.dumps(gci_s[tag]["gci"]), flush=True)
        report["gci_P_settled"] = gci_s
        save_json(REPORT, report)
    far = {}
    for key in FAR:
        s = settle_one(f"V5_P_naca4412_Re2000_a4_{key}")
        if s:
            far[key] = s
    ref = far.get("R400", {})
    for v in far.values():
        if ref.get("CL") and v.get("CL"):
            v["CL_rel_to_R400"] = (v["CL"] - ref["CL"]) / ref["CL"]
            v["CD_rel_to_R400"] = (v["CD"] - ref["CD"]) / ref["CD"]
    report["farfield_P_settled"] = far
    report["settle_note"] = ("Step 9c: Step 9b cases continued from their own mesh and restart files until the 100-iteration "
                             "relative spread of CL and CD is < 1e-4 (cfd.pipeline.continue_case); records '<id>_S'. "
                             "gci_P_settled and farfield_P_settled supersede gci_P and farfield_P.")
    save_json(REPORT, report)
    print("AF_STEP9C_DONE", flush=True)
