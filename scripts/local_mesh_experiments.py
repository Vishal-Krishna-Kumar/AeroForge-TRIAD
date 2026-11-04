"""Local (dev-only) mesh-sensitivity experiments for the worst verification case:
NACA 4412, Re 2000, alpha 4. Not provenance-grade; the chosen production spec is re-verified in Colab."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from cfd.pipeline import ensure_su2, run_case  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402

z = Path("su2dl/su2omp.zip")
su2 = ensure_su2(z, Path("localsu2"))
out, work = Path("localexp_out"), Path("localexp_work")
geom = {"m": 0.04, "p": 0.4, "t": 0.12}
variants = {
    "base_medium": MeshSpec(),
    "surf_half": MeshSpec(h_surface=0.003, h_edges=0.001),
    "wake_half": MeshSpec(h_wake=0.01, wake_length=5.0),
    "bl_fine": MeshSpec(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06),
    "nearfield": MeshSpec(h_far=1.5),
    "R100": MeshSpec(farfield_radius=100.0),
}
for name, spec in variants.items():
    t = time.time()
    r = run_case("E_" + name, geom, 2000.0, 4.0, spec, out, work, su2, z, "dev_experiment")
    a = r.get("assessment", {})
    print(name, json.dumps({"CL": a.get("CL"), "CD": a.get("CD"), "cells": r.get("mesh", {}).get("n_cells_2d"),
                            "iters": a.get("iterations"), "status": r.get("status"),
                            "wall": round(time.time() - t)}), flush=True)
