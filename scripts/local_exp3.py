import sys, json, time; sys.path.insert(0, "src")
from pathlib import Path
from cfd.pipeline import run_case, ensure_su2
from meshing.gmsh_airfoil import MeshSpec
z = Path("su2dl/su2omp.zip"); su2 = ensure_su2(z, Path("localsu2"))
out, work = Path("localexp_out"), Path("localexp_work")
P = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015, h_wake=0.015, wake_length=4.0,
         h_wake_near=0.008, near_wake_length=1.0, growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="Pa")
cases = {"Pa_4412": ({"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, MeshSpec(**P))}
if len(sys.argv) > 1 and sys.argv[1] == "fine":
    cases = {"Pa_fine_4412": ({"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, MeshSpec(**P).scaled(1/1.5, "Pa_fine"))}
for name, (g, re, a, spec) in cases.items():
    t = time.time(); r = run_case("E_" + name, g, re, a, spec, out, work, su2, z, "dev_experiment"); s = r.get("assessment", {})
    print(name, json.dumps({"CL": s.get("CL"), "CD": s.get("CD"), "cells": r.get("mesh", {}).get("n_cells_2d"), "minSICN": r.get("mesh", {}).get("min_SICN"), "iters": s.get("iterations"), "status": r.get("status"), "wall": round(time.time() - t)}), flush=True)
