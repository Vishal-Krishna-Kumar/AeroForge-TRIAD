"""Local dev: run a P-family case in resumable chunks (each call <= ~9 min), using the same
continuation mechanism as cfd.pipeline.settle. Usage: local_chunked.py <tag> <level> <chunk_iters>"""
import json, shutil, sys, time
sys.path.insert(0, "src")
from pathlib import Path
from cfd.pipeline import ensure_su2, _settled
from cfd.su2_case import write_config, run_su2, assess, continuation_config, append_history, integrate_surface
from geometry.naca import naca4
from meshing.gmsh_airfoil import MeshSpec, build_mesh
P = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015, h_wake=0.015, wake_length=4.0,
         h_wake_near=0.008, near_wake_length=1.0, growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
CASES = {"naca0012_Re1000_a4": ((0.0, 0.0, 0.12), 1000.0, 4.0), "naca4412_Re2000_a4": ((0.04, 0.4, 0.12), 2000.0, 4.0)}
FAC = {"coarse": 1.5, "medium": 1.0, "fine": 1 / 1.5}
tag, lvl, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
su2 = ensure_su2(Path("su2dl/su2omp.zip"), Path("localsu2"))
(m, p, t), re, aoa = CASES[tag]
wd = Path("localP") / f"{tag}_{lvl}"; t0 = time.time()
if not (wd / "history.csv").exists():
    wd.mkdir(parents=True, exist_ok=True)
    up, lo = naca4(m, p, t)
    st = build_mesh(up, lo, str(wd / "mesh.su2"), MeshSpec(**P).scaled(FAC[lvl], f"P_{lvl}"), aoa_deg=aoa)
    json.dump(st, open(wd / "mesh_stats.json", "w"))
    write_config(wd, re, iters=chunk, cfl=50.0, res_min=-10.0, conv="cauchy")
    (wd / "config_base.cfg").write_text((wd / "config.cfg").read_text())
    r = run_su2(wd, str(su2), threads=2, timeout_s=3600)
else:
    base = (wd / "config_base.cfg").read_text().replace(f"ITER= {chunk}", "ITER= 4000")
    shutil.copy2(wd / "restart_flow.dat", wd / "solution_flow.dat")
    shutil.move(str(wd / "history.csv"), str(wd / "history_main.csv"))
    (wd / "config.cfg").write_text(continuation_config(base, chunk))
    r = run_su2(wd, str(su2), threads=2, timeout_s=3600)
    seg = wd / f"history_seg{len(list(wd.glob('history_seg*.csv'))) + 1}.csv"
    shutil.move(str(wd / "history.csv"), str(seg)); shutil.move(str(wd / "history_main.csv"), str(wd / "history.csv"))
    append_history(wd, seg)
a = assess(wd, -10.0, window=100); a.pop("history_columns")
a["settled"] = _settled(a); a["rc"] = r["returncode"]; a["chunk_s"] = round(time.time() - t0)
a["cells"] = json.load(open(wd / "mesh_stats.json"))["n_cells_2d"]
if a["settled"]:
    ii = integrate_surface(wd); a["CL_indep"] = ii["CL_total"]; a["CD_indep"] = ii["CD_total"]
print("LOCAL", tag, lvl, json.dumps(a), flush=True)
