"""Dataset V3 (protocol v3): steady transonic Euler, M 0.70, CST-13 design box with alpha in [-1, 3] deg.

Objective J = CL / (CD + CD0), CD0 = 0.008 (fixed profile-drag allowance, protocol v3 §3); its design gradient comes
from ONE discrete adjoint of wL*CL + wD*CD with wL = 1/(CD+CD0), wD = -CL/(CD+CD0)^2 (cal, test and trajectory
points; train is primal-only as in V2). Cases are never dropped: any status other than 'converged' is kept and flagged.

Usage: python scripts/run_dataset_v3.py SPLIT [--start i] [--stop j] [--no-adjoint] | --design-only
       SPLIT in train | cal | test | traj  (traj = frozen results/v3_traj/trajectory_points.json)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "/workspace/AeroForge-TRIAD"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from cfd.euler_case import run_case_euler  # noqa: E402
from cfd.pipeline import ensure_su2, existing  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402

SU2_ZIP = Path(os.environ.get("AF_SU2_ZIP", ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"))
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/workspace/su2"))
SU2_AD = os.environ.get("AF_SU2_AD", "/workspace/mamba/su2ad/bin/SU2_CFD_AD")
AD_ENV = dict(os.environ, LD_LIBRARY_PATH=str(Path(SU2_AD).resolve().parents[1] / "lib"))
WORK = Path(os.environ.get("AF_WORK", "/workspace/runs"))
RAW = ROOT / "datasets" / "raw" / "dataset_v3"
PROC = ROOT / "datasets" / "processed" / "dataset_v3"
DESIGN = ROOT / "datasets" / "designs" / "dataset_v3_design.json"
MACH, CD0, NC, DELTA, AOA = 0.70, 0.008, 6, 0.06, (-1.0, 3.0)
ADJ_ITERS, ADJ_EXTRA = 8000, {"MGLEVEL": "0"}  # protocol v3 amendment 3: single-grid adjoint, 8000-iteration cap
SPLITS = {"train": (384, 20261030), "cal": (128, 20261031), "test": (64, 20261032)}
E_SPEC = dict(h_wall=0.003, bl_ratio=1.15, bl_thickness=0.01, h_surface=0.006, h_edges=0.002, h_wake=0.04,
              wake_length=1.0, h_wake_near=None, growth_rate=0.08, h_far=4.0, farfield_radius=100.0, label="E100")
KEEP = ("mesh.su2", "config.cfg", "history.csv", "solver.log", "surface_flow.csv", "surface_flow.vtk", "restart_flow.dat")


def baseline() -> np.ndarray:
    from geometry.cst import fit_cst
    from geometry.naca import naca4
    up, lo = naca4(0.02, 0.4, 0.11, n=401)
    au, al, _ = fit_cst(up, lo, NC)
    return np.r_[au, al]


def box() -> tuple[np.ndarray, np.ndarray]:
    b = baseline()
    return np.r_[b - DELTA, AOA[0]], np.r_[b + DELTA, AOA[1]]


def accepted(th: np.ndarray) -> tuple[bool, dict]:
    """Same acceptance rules as V2 (t_max in [0.08, 0.14], camber in [-0.01, 0.045], no self-intersection)."""
    from geometry.cst import shape_metrics
    m = shape_metrics(th[:NC], th[NC:2 * NC], n=801)
    ok = (m["self_intersection_free"] and 0.08 <= m["t_max"] <= 0.14
          and m["camber_min"] >= -0.01 and m["camber_max"] <= 0.045)
    return ok, m


def sample(n: int, seed: int) -> tuple[list[np.ndarray], int]:
    from scipy.stats import qmc
    lo, hi = box()
    eng = qmc.Sobol(d=2 * NC + 1, scramble=True, seed=seed)
    out, rej = [], 0
    while len(out) < n:
        for u in eng.random(256):
            th = lo + u * (hi - lo)
            if accepted(th)[0]:
                out.append(th)
                if len(out) == n:
                    break
            else:
                rej += 1
    return out, rej


def make_design() -> dict:
    lo, hi = box()
    d = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "Mach": MACH, "CD0": CD0,
         "family": "CST", "n_coef_per_side": NC, "box_lo": lo.round(8).tolist(), "box_hi": hi.round(8).tolist(),
         "acceptance": "t_max in [0.08,0.14], camber in [-0.01,0.045], no self-intersection", "mesh": E_SPEC,
         "splits": {}, "rejected": {}}
    for name, (n, seed) in SPLITS.items():
        pts, rej = sample(n, seed)
        d["rejected"][name] = rej
        d["splits"][name] = [{"case_id": f"D3_{name}_{i:04d}", "theta": [round(float(v), 8) for v in th]}
                             for i, th in enumerate(pts)]
    return d


def load_design() -> dict:
    if DESIGN.exists():
        d = json.loads(DESIGN.read_text())
        lo, hi = box()
        if not (np.allclose(d["box_lo"], lo, atol=1e-7) and np.allclose(d["box_hi"], hi, atol=1e-7)):
            raise SystemExit("design box on disk differs from this script's box; refusing to mix designs")
        return d
    DESIGN.parent.mkdir(parents=True, exist_ok=True)
    d = make_design()
    save_json(DESIGN, d)
    return d


def j_weights(cl: float, cd: float) -> tuple[float, float]:
    den = cd + CD0
    return 1.0 / den, -cl / den ** 2


def postprocess(case_dir: Path, th: np.ndarray, out: Path, grads: dict | None) -> dict:
    from data.postprocess import GridSpec
    from data.postprocess_euler import fields_on_grid_euler, surface_on_beta_euler
    a = json.loads((case_dir / "case.json").read_text())["assessment"]
    surf = surface_on_beta_euler(case_dir, th)
    fld = fields_on_grid_euler(case_dir, th, MACH, GridSpec())
    arrays = {"theta": th.astype(np.float32), "CL": np.float64(a["CL"]), "CD": np.float64(a["CD"]), "CD0": np.float64(CD0),
              **{k: v for k, v in surf.items() if isinstance(v, np.ndarray)}, **fld}
    g = (grads or {}).get("objectives", {}).get("J", {}).get("grad")
    if g:
        arrays["dJ_dtheta"] = np.array([g[k] for k in grads["dv"]], dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    return {"npz": out.name, "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "surface_gap": surf["surface_gap"]}


def j_gradient(case_dir: Path, wd: Path, th: np.ndarray, rec: dict) -> dict:
    from cfd.gradients import adjoint_gradients
    from cfd.su2_euler import ADJ_CONV_FIELD
    a = rec["assessment"]
    w = j_weights(float(a["CL"]), float(a["CD"]))
    g = adjoint_gradients(case_dir, wd, th, SU2_AD, threads=int(os.environ.get("AF_SU2_THREADS", "1")), radius=0.5,
                          objectives=(), combined={"J": w}, env=AD_ENV, adj_conv_field=ADJ_CONV_FIELD, adj_res_min=-10.0,
                          adj_iters=ADJ_ITERS, adj_extra=ADJ_EXTRA)
    g["j_weights"] = list(w)
    g["CD0"] = CD0
    return g


def mesh_spec() -> MeshSpec:
    return MeshSpec(**E_SPEC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", nargs="?", choices=list(SPLITS) + ["traj"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--no-adjoint", action="store_true")
    ap.add_argument("--design-only", action="store_true")
    args = ap.parse_args()
    design = load_design()
    if args.design_only or args.split is None:
        print("AF_DESIGN_V3", DESIGN, {k: len(v) for k, v in design["splits"].items()}, design["rejected"], flush=True)
        return
    su2 = ensure_su2(SU2_ZIP, SU2_DIR) if not os.environ.get("AF_SU2_EXE") else Path(os.environ["AF_SU2_EXE"])
    spec = mesh_spec()
    want_adj = args.split in ("cal", "test", "traj") and not args.no_adjoint
    raw, proc = RAW, PROC
    if args.split == "traj":
        raw, proc = ROOT / "datasets" / "raw" / "dataset_v3_traj", ROOT / "datasets" / "processed" / "dataset_v3_traj"
        cases = json.loads((ROOT / "results" / "v3_traj" / "trajectory_points.json").read_text())["points"]
    else:
        cases = design["splits"][args.split]
    for c in cases[args.start: args.stop]:
        cid, th = c["case_id"], np.asarray(c["theta"], dtype=float)
        t0 = time.time()
        rec = existing(raw / cid)
        if rec is None:
            rec = run_case_euler(cid, th[:NC], th[NC:2 * NC], float(th[-1]), MACH, spec, raw, WORK, su2, SU2_ZIP,
                                 f"V3_{args.split}")
        a = rec.get("assessment", {})
        info = {"status": rec.get("status"), "CL": a.get("CL"), "CD": a.get("CD"), "iters": a.get("iterations"),
                "solve_s": rec.get("run", {}).get("wall_s")}
        grads, gfile = None, raw / cid / "gradients.json"
        if rec.get("status") == "converged":
            if want_adj:
                t1 = time.time()
                grads = json.loads(gfile.read_text()) if gfile.exists() else j_gradient(raw / cid, WORK / f"{cid}_adj", th, rec)
                if not gfile.exists():
                    save_json(gfile, grads)
                info["grad_ok"] = "grad" in grads["objectives"].get("J", {})
                info["adj_s"] = round(time.time() - t1, 1)
            npz = proc / args.split / f"{cid}.npz"
            if not npz.exists():
                try:
                    info["post"] = postprocess(raw / cid, th, npz, grads)
                except Exception as exc:  # recorded, never hidden
                    info["post_error"] = repr(exc)
        info["wall_s"] = round(time.time() - t0, 1)
        print("D3", cid, json.dumps(info), flush=True)
    print("AF_DATASET_V3_BATCH_DONE", args.split, args.start, args.stop, flush=True)


if __name__ == "__main__":
    main()
