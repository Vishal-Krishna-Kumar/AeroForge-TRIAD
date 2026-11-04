import sys, json, time; sys.path.insert(0, "src")
from pathlib import Path
from cfd.pipeline import run_case, ensure_su2
from meshing.gmsh_airfoil import MeshSpec
z = Path("su2dl/su2omp.zip"); su2 = ensure_su2(z, Path("localsu2"))
P = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015, h_wake=0.015, wake_length=4.0,
         h_wake_near=0.008, near_wake_length=1.0, growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
t = time.time()
r = run_case("E_P_medium_cauchy_4412", {"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, MeshSpec(**P), Path("localexp_out"), Path("localexp_work"), su2, z, "dev_experiment", conv="cauchy")
a = r.get("assessment", {})
print(json.dumps({"status": r["status"], "CL": a.get("CL"), "CD": a.get("CD"), "iters": a.get("iterations"), "rms": a.get("final_rms_pressure_log10"), "cells": r["mesh"]["n_cells_2d"], "wall_s": r["run"]["wall_s"], "total_s": round(time.time() - t)}), flush=True)
