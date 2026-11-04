"""Dataset V1 generation (G3): steady 2D laminar NACA 4-digit cases at Re = 1000 on the P-coarse mesh.

Design (fixed before any case is run, written once to datasets/designs/dataset_v1_design.json):
  theta = (m, p, t, alpha_deg), box m in [0, 0.04], p in [0.2, 0.6], t in [0.08, 0.14], alpha in [-2, 4] deg
  (steady envelope from the regime map; see ADR-003).
  Splits are independent scrambled Sobol sequences (different seeds), processed in sequence order so
  any prefix is space-filling:  train (512), cal (128), test (128).
  cal and test cases also get discrete-adjoint gradients of CL and CD w.r.t. theta.

Per case: cfd.pipeline.run_case (cold start from uniform flow, Cauchy stop + settle-by-continuation;
a warm start from the nearest converged case saved only ~6 % of iterations in a local test, 651 vs 694,
so the simpler, order-independent cold start is used), then
post-processing to fixed-format arrays (surface Cp/Cf on a beta grid, volume p/u/v/SDF on a 192x96 grid)
saved as datasets/processed/dataset_v1/<case>.npz. Cases are never dropped: any status other than
"converged" is kept and flagged in the manifest.

Usage: python scripts/run_dataset_v1.py SPLIT [--start i] [--stop j] [--no-adjoint]
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

SU2_ZIP = ROOT / "env" / "cache" / "SU2-v8.5.0-linux64-omp.zip"
SU2_DIR = Path(os.environ.get("AF_SU2_DIR", "/content/su2"))
SU2_AD = os.environ.get("AF_SU2_AD", "/content/mamba/su2ad/bin/SU2_CFD_AD")
AD_ENV = dict(os.environ, LD_LIBRARY_PATH=str(Path(SU2_AD).resolve().parents[1] / "lib"))
WORK = Path(os.environ.get("AF_WORK", "/content/runs"))
RAW = ROOT / "datasets" / "raw" / "dataset_v1"
PROC = ROOT / "datasets" / "processed" / "dataset_v1"
DESIGN = ROOT / "datasets" / "designs" / "dataset_v1_design.json"
RE = 1000.0
# Final box (ADR-003, fixed 2026-10-12 before any dataset case was run): the steady envelope from the Re 1000
# regime map. Round 1-2 found self-excited unsteadiness for thick + cambered sections at alpha 4 (m 0.04, p 0.2,
# t 0.18) and at m 0.06 alpha 6; m <= 0.04, t <= 0.14, alpha in [-2, 4] is the conservative envelope whose corners
# are checked by regime-map round 3.
BOX = {"m": (0.0, 0.04), "p": (0.2, 0.6), "t": (0.08, 0.14), "alpha_deg": (-2.0, 4.0)}
SPLITS = {"train": (512, 20261012), "cal": (128, 20261013), "test": (128, 20261014)}
P_SPEC = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015,
              h_wake=0.015, wake_length=4.0, h_wake_near=0.008, near_wake_length=1.0,
              growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P")
KEEP = ("mesh.su2", "config.cfg", "history.csv", "solver.log", "surface_flow.csv", "surface_flow.vtk",
        "restart_flow.dat")  # flow.vtu dropped: the volume solution is in restart_flow.dat


def make_design() -> dict:
    from scipy.stats import qmc

    lo = np.array([v[0] for v in BOX.values()]); hi = np.array([v[1] for v in BOX.values()])
    design = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "Re": RE,
              "variables": list(BOX), "box": BOX, "mesh": "P_coarse (P x 1.5)", "splits": {}}
    for name, (n, seed) in SPLITS.items():
        u = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
        x = lo + u * (hi - lo)
        design["splits"][name] = [{"case_id": f"D1_{name}_{i:04d}", "m": round(float(r[0]), 6), "p": round(float(r[1]), 6),
                                   "t": round(float(r[2]), 6), "alpha_deg": round(float(r[3]), 4)} for i, r in enumerate(x)]
    return design


def load_design() -> dict:
    if DESIGN.exists():
        d = json.loads(DESIGN.read_text())
        if d["box"] != {k: list(v) for k, v in BOX.items()}:
            raise SystemExit(f"design box on disk {d['box']} differs from this script's {BOX}; refusing to mix designs")
        return d
    DESIGN.parent.mkdir(parents=True, exist_ok=True)
    d = make_design()
    save_json(DESIGN, d)
    return d


def theta_of(c: dict) -> np.ndarray:
    return np.array([c["m"], c["p"], c["t"], c["alpha_deg"]], dtype=float)


def nearest_converged(c: dict, done: list[dict]) -> Path | None:
    if not done:
        return None
    span = np.array([v[1] - v[0] for v in BOX.values()])
    th = theta_of(c) / span
    d = [np.linalg.norm(theta_of(o) / span - th) for o in done]
    return RAW / done[int(np.argmin(d))]["case_id"]


def postprocess(case_dir: Path, c: dict, out: Path, grads: dict | None) -> dict:
    from data.postprocess import GridSpec, fields_on_grid, surface_on_beta

    th = theta_of(c)
    rec = json.loads((case_dir / "case.json").read_text())
    a = rec["assessment"]
    surf = surface_on_beta(case_dir, th)
    fld = fields_on_grid(case_dir, th, GridSpec())
    arrays = {"theta": th.astype(np.float32), "CL": np.float64(a["CL"]), "CD": np.float64(a["CD"]),
              **{k: v for k, v in surf.items() if isinstance(v, np.ndarray)}, **fld}
    if grads:
        for q in ("CL", "CD"):
            g = grads["objectives"].get(q, {}).get("grad")
            if g:
                arrays[f"d{q}_dtheta"] = np.array([g[k] for k in ("m", "p", "t", "aoa_deg")], dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    return {"npz": out.name, "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "surface_gap": surf["surface_gap"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=list(SPLITS))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--no-adjoint", action="store_true")
    ap.add_argument("--design-only", action="store_true", help="write the design file and exit (run once before parallel workers)")
    args = ap.parse_args()
    design = load_design()
    if args.design_only:
        print("AF_DESIGN", DESIGN, {k: len(v) for k, v in design["splits"].items()}, flush=True)
        return
    cases = design["splits"][args.split][args.start: args.stop]
    su2 = ensure_su2(SU2_ZIP, SU2_DIR)
    spec = MeshSpec(**P_SPEC).scaled(1.5, "P_coarse")
    want_adj = args.split in ("cal", "test") and not args.no_adjoint
    done = [c for c in design["splits"][args.split] if (existing(RAW / c["case_id"]) or {}).get("status") == "converged"]
    for c in cases:
        cid = c["case_id"]
        t0 = time.time()
        prev = existing(RAW / cid)
        if prev is None:
            rec = run_case(cid, {"m": c["m"], "p": c["p"], "t": c["t"]}, RE, c["alpha_deg"], spec, RAW, WORK, su2, SU2_ZIP,
                           f"G3_dataset_v1_{args.split}", conv="cauchy", keep=KEEP)
        else:
            rec = prev
        info = {"status": rec.get("status"), "CL": rec.get("assessment", {}).get("CL"),
                "CD": rec.get("assessment", {}).get("CD"), "iters": rec.get("assessment", {}).get("iterations")}
        grads = None
        gfile = RAW / cid / "gradients.json"
        if rec.get("status") == "converged":
            done.append(c)
            if want_adj:
                if gfile.exists():
                    grads = json.loads(gfile.read_text())
                else:
                    from cfd.gradients import adjoint_gradients
                    grads = adjoint_gradients(RAW / cid, WORK / f"{cid}_adj", theta_of(c), SU2_AD, threads=int(os.environ.get("AF_SU2_THREADS", "2")),
                                              radius=0.5, env=AD_ENV)
                    save_json(gfile, grads)
                info["grad_ok"] = all("grad" in v for v in grads["objectives"].values())
            npz = PROC / args.split / f"{cid}.npz"
            if not npz.exists():
                try:
                    info["post"] = postprocess(RAW / cid, c, npz, grads)
                except Exception as exc:  # recorded, never hidden
                    info["post_error"] = repr(exc)
        info["wall_s"] = round(time.time() - t0, 1)
        print("D1", cid, json.dumps(info), flush=True)
    print("AF_DATASET_V1_BATCH_DONE", args.split, args.start, args.stop, flush=True)


if __name__ == "__main__":
    main()
