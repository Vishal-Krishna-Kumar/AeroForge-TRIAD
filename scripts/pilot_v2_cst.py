"""Dataset V2 pilot (cost and feasibility only, not data): CST-13 sections at Re 1000 on the V1 P-coarse mesh.

Box (DRAFT, not yet fixed): baseline = 6+6 CST fit of NACA 2411 (centre of the V1 box), +-0.06 per coefficient,
alpha in [-2, 4] deg; accepted if t_max in [0.08, 0.14], camber in [-0.01, 0.045], no self-intersection.
Pilot cases (chosen by rule before running): P0 baseline at alpha 4; P1 the accepted Sobol sample with the
largest t_max * camber_max at alpha 4 (worst regime corner); P2 the sample with the most curvature sign changes
on the upper surface (waviest) at alpha 4.
Usage: python scripts/pilot_v2_cst.py CASE_INDEX   (one case per process; run several in parallel)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from cfd.pipeline import run_case  # noqa: E402
from geometry.cst import fit_cst, shape_metrics, surface_y  # noqa: E402
from geometry.naca import naca4  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402

P_SPEC = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015,
              h_wake=0.015, wake_length=4.0, h_wake_near=0.008, near_wake_length=1.0,
              growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")   # = scripts/run_dataset_v1.P_SPEC
DELTA, NC = 0.06, 6


def accepted(th):
    m = shape_metrics(th[:NC], th[NC:], n=801)
    ok = m["self_intersection_free"] and 0.08 <= m["t_max"] <= 0.14 and -0.01 <= m["camber_min"] and m["camber_max"] <= 0.045
    return ok, m


def waviness(th):
    x = 0.5 * (1 - np.cos(np.linspace(0.05, np.pi - 0.05, 400)))
    y = surface_y(x, th[:NC]); d2 = np.gradient(np.gradient(y, x), x)
    return int(np.sum(np.diff(np.sign(d2)) != 0))


def pilot_cases():
    up, lo = naca4(0.02, 0.4, 0.11, n=401)
    au0, al0, _ = fit_cst(up, lo, NC)
    base = np.r_[au0, al0]
    X = base + DELTA * (qmc.Sobol(2 * NC, seed=1).random(1024) * 2 - 1)
    acc = [(th, m) for th in X for ok, m in [accepted(th)] if ok]
    p1 = max(acc, key=lambda a: a[1]["t_max"] * a[1]["camber_max"])[0]
    p2 = max(acc, key=lambda a: waviness(a[0]))[0]
    return [("P0_base", base), ("P1_thick_cambered", p1), ("P2_wavy", p2)]


def main():
    k = int(sys.argv[1])
    name, th = pilot_cases()[k]
    out, work = ROOT / "results" / "v2_pilot" / "raw", ROOT / "runs" / "v2_pilot"
    out.mkdir(parents=True, exist_ok=True)
    su2 = Path(os.environ["AF_SU2_EXE"]); su2zip = Path(os.environ["AF_SU2_ZIP"])
    spec = MeshSpec(**P_SPEC).scaled(1.5, "P_coarse")
    geom = {"family": "CST", "au": [round(float(v), 6) for v in th[:NC]], "al": [round(float(v), 6) for v in th[NC:]]}
    rec = run_case(f"V2pilot_{name}_a4", geom, 1000.0, 4.0, spec, out, work, su2, su2zip, stage="v2_pilot", conv="cauchy")
    keep = {k2: rec.get(k2) for k2 in ("case_id", "status", "geometry", "mesh_wall_s", "run", "assessment")}
    print("PILOT_DONE", json.dumps(keep, default=str)[:2000], flush=True)


if __name__ == "__main__":
    main()
