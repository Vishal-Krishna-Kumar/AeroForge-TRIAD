import sys, json, time; sys.path.insert(0, "src")
from pathlib import Path
import cfd.pipeline as pl
from meshing.gmsh_airfoil import MeshSpec
z = Path("su2dl/su2omp.zip"); su2 = pl.ensure_su2(z, Path("localsu2"))
P = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015, h_wake=0.015, wake_length=4.0,
         h_wake_near=0.008, near_wake_length=1.0, growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
real = pl.settle
pl.settle = lambda *a, **k: {"segments": [], "note": "disabled for test (emulates a V4 record)"}
src = pl.run_case("T_src", {"m": 0.04, "p": 0.4, "t": 0.12}, 2000.0, 4.0, MeshSpec(**P).scaled(1.5, "P_coarse"),
                  Path("settle_test/out"), Path("settle_test/work"), su2, z, "dev_test", conv="cauchy", force=True)
print("SRC", src["status"], src["assessment"]["iterations"], src["assessment"]["CL"], src["assessment"]["CL_tail_rel_spread"])
pl.settle = real
t = time.time()
r = pl.continue_case(Path("settle_test/out/T_src"), "T_src_S", Path("settle_test/out"), Path("settle_test/work"), su2, "dev_test", force=True)
a = r["assessment"]
print("CONT", r["status"], a["iterations"], a["CL"], a["CD"], a["CL_tail_rel_spread"], a["CD_tail_rel_spread"], a["final_rms_pressure_log10"],
      r["run"], len(r["continuation"]["segments"]), r["continued_from"]["case_id"], r.get("consistency"), round(time.time() - t))
r2 = pl.continue_case(Path("settle_test/out/T_src_S"), "T_src_SS", Path("settle_test/out"), Path("settle_test/work"), su2, "dev_test", force=True)
print("NOOP", r2["status"], r2["assessment"]["iterations"], r2["assessment"]["CL"], len(r2["continuation"]["segments"]))
print(sorted(r["files_sha256"]))
