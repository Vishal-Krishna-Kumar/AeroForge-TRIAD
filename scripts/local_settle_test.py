import sys, json, time; sys.path.insert(0, "src")
from pathlib import Path
from cfd.pipeline import run_case, ensure_su2
from meshing.gmsh_airfoil import MeshSpec
z = Path("su2dl/su2omp.zip"); su2 = ensure_su2(z, Path("localsu2"))
P = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015, h_wake=0.015, wake_length=4.0,
         h_wake_near=0.008, near_wake_length=1.0, growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
t = time.time()
r = run_case("T_P_coarse_4412_settle", {"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, MeshSpec(**P).scaled(1.5, "P_coarse"),
             Path("settle_test/out"), Path("settle_test/work"), su2, z, "dev_test", conv="cauchy", force=True)
a = r["assessment"]
print(json.dumps({"status": r["status"], "CL": a["CL"], "CD": a["CD"], "iters": a["iterations"], "rms": a["final_rms_pressure_log10"],
                  "spreads": [a["CL_tail_rel_spread"], a["CD_tail_rel_spread"]], "cont": r["continuation"], "consistency": r.get("consistency"),
                  "total_s": round(time.time() - t)}, indent=1), flush=True)
