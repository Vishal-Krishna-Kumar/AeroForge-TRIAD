"""Dataset V2 generation (protocol v2 §1): steady 2D laminar CST-13 sections at Re 1000, P-coarse mesh.

theta = (A_u[0..5], A_l[0..5], alpha_deg). Box: CST fit of NACA 2411 +- 0.06 per coefficient, alpha in [-2, 4] deg.
Acceptance: t_max in [0.08, 0.14], camber in [-0.01, 0.045], no self-intersection. Splits are independent scrambled
Sobol streams (seeds 20261020/21/22); rejected points are skipped and COUNTED in the design file, never hidden;
each split takes the first n accepted points in sequence order.
cal and test cases get the L/D design gradient from ONE adjoint of w_L*CL + w_D*CD with w_L = 1/CD, w_D = -CL/CD^2
(validated against two adjoints by scripts/v2_checks.py before any dataset case; see protocol v2 §1).
Cases are never dropped: any status other than "converged" is kept and flagged.

Usage: python scripts/run_dataset_v2.py SPLIT [--start i] [--stop j] [--no-adjoint] | --design-only
       python scripts/run_dataset_v2.py traj [--start i] [--stop j]   (pod session 2: the frozen trajectory
       iterates of results/v2_traj/trajectory_points.json, primal + L/D adjoint, protocol v2 §4)
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

ROOT = Path(os.environ.get("AF_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from cfd.pipeline import ensure_su2, existing, run_case  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402

SU2_ZIP = Path(os.environ.get("AF_SU2_ZIP", ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"))
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
SU2_AD = os.environ.get("AF_SU2_AD", "/content/mamba/su2ad/bin/SU2_CFD_AD")
AD_ENV = dict(os.environ, LD_LIBRARY_PATH=str(Path(SU2_AD).resolve().parents[1] / "lib"))
WORK = Path(os.environ.get("AF_WORK", "/content/runs"))
RAW = ROOT / "datasets" / "raw" / "dataset_v2"
PROC = ROOT / "datasets" / "processed" / "dataset_v2"
DESIGN = ROOT / "datasets" / "designs" / "dataset_v2_design.json"
RE, NC, DELTA, AOA = 1000.0, 6, 0.06, (-2.0, 4.0)
SPLITS = {"train": (1024, 20261020), "cal": (256, 20261021), "test": (128, 20261022)}
P_SPEC = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015,
              h_wake=0.015, wake_length=4.0, h_wake_near=0.008, near_wake_length=1.0,
              growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")   # identical to V1
KEEP = ("mesh.su2", "config.cfg", "history.csv", "solver.log", "surface_flow.csv", "surface_flow.vtk",
        "restart_flow.dat")


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
    from geometry.cst import shape_metrics
    m = shape_metrics(th[:NC], th[NC:2 * NC], n=801)
    ok = (m["self_intersection_free"] and 0.08 <= m["t_max"] <= 0.14
          and m["camber_min"] >= -0.01 and m["camber_max"] <= 0.045)
    return ok, m


def sample(n: int, seed: int) -> tuple[list[np.ndarray], int]:
    """First n accepted points of a scrambled Sobol stream; returns (points, number rejected)."""
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
    d = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "Re": RE, "family": "CST",
         "n_coef_per_side": NC, "box_lo": lo.round(8).tolist(), "box_hi": hi.round(8).tolist(),
         "acceptance": "t_max in [0.08,0.14], camber in [-0.01,0.045], no self-intersection",
         "mesh": "P_coarse (P x 1.5)", "splits": {}, "rejected": {}}
    for name, (n, seed) in SPLITS.items():
        pts, rej = sample(n, seed)
        d["rejected"][name] = rej
        d["splits"][name] = [{"case_id": f"D2_{name}_{i:04d}", "theta": [round(float(v), 8) for v in th]}
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


def ld_weights(cl: float, cd: float) -> tuple[float, float]:
    return 1.0 / cd, -cl / cd ** 2


def postprocess(case_dir: Path, th: np.ndarray, out: Path, grads: dict | None) -> dict:
    from data.postprocess import GridSpec, fields_on_grid, surface_on_beta
    a = json.loads((case_dir / "case.json").read_text())["assessment"]
    surf = surface_on_beta(case_dir, th)
    fld = fields_on_grid(case_dir, th, GridSpec())
    arrays = {"theta": th.astype(np.float32), "CL": np.float64(a["CL"]), "CD": np.float64(a["CD"]),
              **{k: v for k, v in surf.items() if isinstance(v, np.ndarray)}, **fld}
    g = (grads or {}).get("objectives", {}).get("LD", {}).get("grad")
    if g:
        arrays["dLD_dtheta"] = np.array([g[k] for k in grads["dv"]], dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    return {"npz": out.name, "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "surface_gap": surf["surface_gap"]}


def ld_gradient(case_dir: Path, wd: Path, th: np.ndarray, rec: dict) -> dict:
    from cfd.gradients import adjoint_gradients
    a = rec["assessment"]
    w = ld_weights(float(a["CL"]), float(a["CD"]))
    g = adjoint_gradients(case_dir, wd, th, SU2_AD, threads=int(os.environ.get("AF_SU2_THREADS", "1")), radius=0.5,
                          objectives=(), combined={"LD": w}, env=AD_ENV)
    g["ld_weights"] = list(w)
    return g


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
        print("AF_DESIGN_V2", DESIGN, {k: len(v) for k, v in design["splits"].items()}, design["rejected"], flush=True)
        return
    su2 = ensure_su2(SU2_ZIP, SU2_DIR) if not os.environ.get("AF_SU2_EXE") else Path(os.environ["AF_SU2_EXE"])
    spec = MeshSpec(**P_SPEC).scaled(1.5, "P_coarse")
    want_adj = args.split in ("cal", "test", "traj") and not args.no_adjoint
    if args.split == "traj":
        global RAW, PROC
        RAW, PROC = ROOT / "datasets" / "raw" / "dataset_v2_traj", ROOT / "datasets" / "processed" / "dataset_v2_traj"
        cases = json.loads((ROOT / "results" / "v2_traj" / "trajectory_points.json").read_text())["points"]
    else:
        cases = design["splits"][args.split]
    for c in cases[args.start: args.stop]:
        cid, th = c["case_id"], np.asarray(c["theta"], dtype=float)
        t0 = time.time()
        rec = existing(RAW / cid)
        if rec is None:
            geom = {"family": "CST", "au": th[:NC].tolist(), "al": th[NC:2 * NC].tolist()}
            rec = run_case(cid, geom, RE, float(th[-1]), spec, RAW, WORK, su2, SU2_ZIP,
                           f"S2_dataset_v2_{args.split}", conv="cauchy", keep=KEEP)
        a = rec.get("assessment", {})
        info = {"status": rec.get("status"), "CL": a.get("CL"), "CD": a.get("CD"), "iters": a.get("iterations")}
        grads, gfile = None, RAW / cid / "gradients.json"
        if rec.get("status") == "converged":
            if want_adj:
                grads = json.loads(gfile.read_text()) if gfile.exists() else ld_gradient(RAW / cid, WORK / f"{cid}_adj", th, rec)
                if not gfile.exists():
                    save_json(gfile, grads)
                info["grad_ok"] = "grad" in grads["objectives"].get("LD", {})
            npz = PROC / args.split / f"{cid}.npz"
            if not npz.exists():
                try:
                    info["post"] = postprocess(RAW / cid, th, npz, grads)
                except Exception as exc:  # recorded, never hidden
                    info["post_error"] = repr(exc)
        info["wall_s"] = round(time.time() - t0, 1)
        print("D2", cid, json.dumps(info), flush=True)
    print("AF_DATASET_V2_BATCH_DONE", args.split, args.start, args.stop, flush=True)


if __name__ == "__main__":
    main()
