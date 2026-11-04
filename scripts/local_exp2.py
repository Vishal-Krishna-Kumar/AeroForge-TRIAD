import sys, json, time; sys.path.insert(0, "src")
from pathlib import Path
from cfd.pipeline import run_case, ensure_su2
from meshing.gmsh_airfoil import MeshSpec
z = Path("su2dl/su2omp.zip"); su2 = ensure_su2(z, Path("localsu2"))
out, work = Path("localexp_out"), Path("localexp_work")
geom = {"m": 0.04, "p": 0.4, "t": 0.12}
V = {"wake_half": MeshSpec(h_wake=0.01, wake_length=5.0),
     "bl_fine": MeshSpec(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06),
     "nearfield": MeshSpec(h_far=1.5)}
for name in sys.argv[1:]:
    t = time.time(); r = run_case("E_" + name, geom, 2000.0, 4.0, V[name], out, work, su2, z, "dev_experiment"); a = r.get("assessment", {})
    print(name, json.dumps({"CL": a.get("CL"), "CD": a.get("CD"), "cells": r.get("mesh", {}).get("n_cells_2d"), "iters": a.get("iterations"), "status": r.get("status"), "wall": round(time.time() - t)}), flush=True)
