"""Dev-only solver-acceleration test on the Pa_coarse mesh (NACA 4412, Re 2000, a4):
baseline (no multigrid, residual -10) vs multigrid (MGLEVEL 2/3) vs Cauchy stopping on CL/CD."""
import sys, json, re, shutil
sys.path.insert(0, "src")
from pathlib import Path
from geometry.naca import naca4
from meshing.gmsh_airfoil import MeshSpec, build_mesh
from cfd.su2_case import write_config, run_su2, assess
from cfd.pipeline import ensure_su2
su2 = ensure_su2(Path("su2dl/su2omp.zip"), Path("localsu2"))
P = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015, h_wake=0.015, wake_length=4.0,
         h_wake_near=0.008, near_wake_length=1.0, growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="Pa")
spec = MeshSpec(**P).scaled(1.5, "Pa_coarse")
u, l = naca4(0.04, 0.4, 0.12)
base = Path("localexp4"); base.mkdir(exist_ok=True)
mesh = base / "mesh.su2"
if not mesh.exists():
    build_mesh(u, l, str(mesh), spec, aoa_deg=4.0)
variants = {
  "mg0_res10": {},
  "mg2_res10": {"MGLEVEL": "2", "MGCYCLE": "W_CYCLE"},
  "mg0_cauchy": {"CONV_FIELD": "(LIFT, DRAG)", "CONV_CAUCHY_ELEMS": "100", "CONV_CAUCHY_EPS": "1E-6"},
  "mg2_cauchy": {"MGLEVEL": "2", "MGCYCLE": "W_CYCLE", "CONV_FIELD": "(LIFT, DRAG)", "CONV_CAUCHY_ELEMS": "100", "CONV_CAUCHY_EPS": "1E-6"},
}
for name in sys.argv[1:]:
    wd = base / name
    if wd.exists(): shutil.rmtree(wd)
    wd.mkdir(); shutil.copy(mesh, wd / "mesh.su2")
    cfg = write_config(wd, 2000.0, iters=4000, cfl=50.0, res_min=-10.0)
    txt = cfg.read_text()
    for k, v in variants[name].items():
        if re.search(rf"(?m)^{k}=", txt): txt = re.sub(rf"(?m)^{k}=.*$", f"{k}= {v}", txt)
        else: txt += f"{k}= {v}\n"
    cfg.write_text(txt)
    r = run_su2(wd, str(su2), threads=2)
    a = assess(wd, -10.0); a.pop("history_columns", None)
    print(name, json.dumps({"rc": r["returncode"], "wall_s": r["wall_s"], "iters": a["iterations"], "rms_p": round(a["final_rms_pressure_log10"], 2), "CL": a["CL"], "CD": a["CD"]}), flush=True)
