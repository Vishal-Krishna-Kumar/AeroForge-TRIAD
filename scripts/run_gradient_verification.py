"""Gradient verification (G2/G3 prerequisite for CDGC): discrete-adjoint design gradients vs central finite
differences on deformed meshes, both through the same theta -> mesh map (cfd.shape_sensitivity).

Base case: NACA 4412, Re 1000, alpha 4 deg, P-coarse (Step 9e record V6_P_naca4412_Re1000_a4_coarse).
Design variables theta = (m, p, t, alpha_deg); objectives CL and CD.

Acceptance (to be recorded, not tuned afterwards): relative difference |g_adj - g_fd| / |g_fd| <= 2 % for
every derivative whose |g_fd| is not negligible (> 1e-3 of the largest derivative of that objective),
with FD solves settled to an absolute 100-iteration spread well below the FD increment of the objective.

Usage: python scripts/run_gradient_verification.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from cfd.gradients import adjoint_gradients, fd_gradients  # noqa: E402
from cfd.pipeline import ensure_su2  # noqa: E402
from cfd.shape_sensitivity import DV_NAMES, DesignMap, chain_rule, read_su2_mesh  # noqa: E402
from cfd.gradients import read_volume_sensitivity, sensitivity_csv  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402

SU2_ZIP = ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
SU2_AD = os.environ.get("AF_SU2_AD", "/content/mamba/su2ad/bin/SU2_CFD_AD")
AD_ENV = dict(os.environ, LD_LIBRARY_PATH=str(Path(SU2_AD).resolve().parents[1] / "lib"))
CASE = ROOT / "datasets" / "raw" / "stageA_verification" / "V6_P_naca4412_Re1000_a4_coarse"
THETA0 = np.array([0.04, 0.4, 0.12, 4.0])
WORK = Path(os.environ.get("AF_WORK", "/content/runs")) / "gradcheck_naca4412_Re1000_a4_coarse"
REPORT = ROOT / "results" / "reports" / "gradient_verification_naca4412_Re1000_a4_coarse.json"
STEPS = (1e-3, 1e-2, 1e-3, 0.1)

if __name__ == "__main__":
    su2 = str(ensure_su2(SU2_ZIP, SU2_DIR))
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report["case"] = CASE.name
    report["theta0"] = THETA0.tolist()
    if "adjoint" not in report or not all("grad" in v for v in report["adjoint"]["objectives"].values()):
        if "adjoint" in report:  # earlier attempt without gradients (2026-10-12: CSV name mismatch) - kept on record
            report.setdefault("failed_attempts", []).append(report.pop("adjoint"))
        report["adjoint"] = adjoint_gradients(CASE, WORK / "adj", THETA0, SU2_AD, threads=2, radius=0.5, env=AD_ENV)
        save_json(REPORT, report)
    print("GV_ADJ", json.dumps({k: v.get("grad") for k, v in report["adjoint"]["objectives"].items()}), flush=True)
    # RBF-radius sensitivity of the chain rule (no new solves)
    mesh = read_su2_mesh(WORK / "adj" / "mesh.su2")
    radii = {}
    for r in (0.25, 1.0):
        dX = DesignMap(mesh, THETA0, radius=r).dX_dtheta()
        radii[str(r)] = {q: dict(zip(DV_NAMES, map(float, chain_rule(
            read_volume_sensitivity(sensitivity_csv(WORK / "adj" / f"adj_{q}"), len(mesh.points)), dX))))
            for q in ("CL", "CD")}
    report["adjoint_radius_check"] = radii
    save_json(REPORT, report)
    print("GV_RADII", json.dumps(radii), flush=True)
    if "fd" not in report:
        report["fd"] = fd_gradients(CASE, WORK / "fd", THETA0, su2, steps=STEPS, radius=0.5, iters=1200, threads=2)
        save_json(REPORT, report)
    print("GV_FD", json.dumps(report["fd"]["grad"]), flush=True)
    cmp = {}
    for q in ("CL", "CD"):
        g_adj, g_fd = report["adjoint"]["objectives"][q]["grad"], report["fd"]["grad"][q]
        scale = max(abs(v) for v in g_fd.values())
        cmp[q] = {k: {"adj": g_adj[k], "fd": g_fd[k],
                      "rel_diff": abs(g_adj[k] - g_fd[k]) / abs(g_fd[k]) if abs(g_fd[k]) > 1e-3 * scale else None}
                  for k in g_fd}
    report["comparison"] = cmp
    report["passes_2pct"] = all(v["rel_diff"] is None or v["rel_diff"] <= 0.02 for q in cmp.values() for v in q.values())
    save_json(REPORT, report)
    print("GV_CMP", json.dumps(cmp), "PASS" if report["passes_2pct"] else "FAIL", flush=True)
    print("AF_GRADCHECK_DONE", flush=True)
