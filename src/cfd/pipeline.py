"""Reusable case pipeline: geometry -> mesh -> SU2 -> assessment -> persisted record (AeroForge-TRIAD).

Every case directory in Drive contains the inputs, solver log, outputs and a case.json with
provenance (versions, checksums, timings). Runs are resumable: a case whose case.json already
records a terminal status is skipped unless force=True.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import shutil
import time
import zipfile
from pathlib import Path

from cfd.su2_case import (append_history, assess, continuation_config, integrate_surface, run_su2,
                          save_json, write_config)
from cfd.su2_unsteady import classify, write_unsteady_config
from geometry.naca import geometry_checks, naca4
from meshing.gmsh_airfoil import MeshSpec, build_mesh_isolated

# Outcomes that are kept on resume. Failures (solver_failed, mesh_failed) are retried, and the
# failed attempt is archived first so it stays on record (never silently overwritten).
TERMINAL = {"converged", "not_converged", "steady", "unsteady", "not_settled"}
FAILED = {"solver_failed", "mesh_failed"}
KEEP_FILES = ("mesh.su2", "config.cfg", "history.csv", "solver.log", "surface_flow.csv",
              "surface_flow.vtk", "flow.vtu", "restart_flow.dat")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_su2(su2_zip: Path, su2_dir: Path) -> Path:
    exe = su2_dir / "bin" / "SU2_CFD"
    if not exe.exists():
        su2_dir.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(su2_zip).extractall(su2_dir)
        for inner in list(su2_dir.rglob("*.zip")):
            zipfile.ZipFile(inner).extractall(inner.parent)
            inner.unlink()
        for p in su2_dir.rglob("SU2_*"):
            if p.is_file():
                p.chmod(0o755)
    return exe


def _persist(wd: Path, dest: Path, rec: dict, keep: tuple[str, ...] = KEEP_FILES) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in keep:
        if (wd / name).exists():
            shutil.copy2(wd / name, dest / name)
    # unsteady runs write time-stamped final files; keep the last of each kind
    for pattern in ("surface_flow_*.vtk", "flow_*.vtu"):
        found = sorted(wd.glob(pattern))
        if found:
            shutil.copy2(found[-1], dest / found[-1].name)
    rec["files_sha256"] = {f.name: sha256(f) for f in sorted(dest.iterdir()) if f.is_file()}
    save_json(dest / "case.json", rec)
    return rec


def existing(dest: Path) -> dict | None:
    f = dest / "case.json"
    if f.exists():
        rec = json.loads(f.read_text())
        if rec.get("status") in TERMINAL:
            return rec
    return None


def _archive_failed(dest: Path, out_root: Path) -> None:
    """Move a previous failed attempt to <out_root>/_failed_attempts/<case>__<created_utc>/."""
    f = dest / "case.json"
    if not f.exists():
        return
    rec = json.loads(f.read_text())
    if rec.get("status") not in FAILED:
        return
    stamp = str(rec.get("created_utc", "unknown")).replace(":", "").replace("+", "")
    arch = out_root / "_failed_attempts" / f"{dest.name}__{stamp}"
    arch.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dest), str(arch))


SU2_THREADS = int(os.environ.get("AF_SU2_THREADS", "2"))  # OpenMP threads per SU2 run
SETTLE_TOL = 1e-4      # relative spread of CL and CD over the last 100 iterations
SETTLE_WINDOW = 100
SETTLE_CHUNK = 200     # continuation chunk (iterations)
ITER_CAP = 4000        # total iterations per case, initial run plus continuations


def _settled(a: dict) -> bool:
    return a["CL_tail_rel_spread"] < SETTLE_TOL and a["CD_tail_rel_spread"] < SETTLE_TOL


def settle(wd: Path, su2: Path, res_min: float, chunk: int = SETTLE_CHUNK, cap: int = ITER_CAP) -> dict:
    """Continue a finished steady run from its restart file, `chunk` iterations at a time, until
    CL and CD have settled (100-iteration relative spread < SETTLE_TOL) or the total reaches `cap`.
    Returns a provenance record of the continuation segments."""
    segs = []
    a = assess(wd, res_min, window=SETTLE_WINDOW)
    base_cfg = (wd / "config.cfg").read_text()
    while not _settled(a) and a["iterations"] < cap:
        n = min(chunk, cap - a["iterations"])
        shutil.copy2(wd / "restart_flow.dat", wd / "solution_flow.dat")
        shutil.move(str(wd / "history.csv"), str(wd / "history_main.csv"))
        (wd / "config.cfg").write_text(continuation_config(base_cfg, n))
        prev_log = (wd / "solver.log").read_text() if (wd / "solver.log").exists() else ""
        run = run_su2(wd, str(su2), threads=SU2_THREADS, timeout_s=6 * 3600)
        (wd / "solver.log").write_text(prev_log + f"\n##### AeroForge continuation segment {len(segs) + 1}"
                                       f" ({n} iterations from restart) #####\n" + (wd / "solver.log").read_text())
        seg_hist = wd / f"history_seg{len(segs) + 1}.csv"
        if (wd / "history.csv").exists():
            shutil.move(str(wd / "history.csv"), str(seg_hist))
        shutil.move(str(wd / "history_main.csv"), str(wd / "history.csv"))
        segs.append({"iterations_requested": n, **run})
        if run["returncode"] != 0 or not seg_hist.exists():
            segs[-1]["failed"] = True
            break
        append_history(wd, seg_hist)
        a = assess(wd, res_min, window=SETTLE_WINDOW)
        segs[-1].update({"total_iterations": a["iterations"], "CL_spread": a["CL_tail_rel_spread"],
                         "CD_spread": a["CD_tail_rel_spread"], "rms_p": a["final_rms_pressure_log10"]})
    (wd / "config.cfg").write_text(base_cfg)  # the persisted config is the case's own (initial) config
    return {"segments": segs, "chunk": chunk, "cap": cap, "tol": SETTLE_TOL, "window": SETTLE_WINDOW}


def _finish_steady(wd: Path, rec: dict, res_min: float, conv: str, cauchy_res_floor: float) -> None:
    a = assess(wd, res_min, window=SETTLE_WINDOW if conv == "cauchy" else None)
    a.pop("history_columns", None)
    rec["assessment"] = a
    try:
        ii = integrate_surface(wd)
        rec["independent_integration"] = ii
        rec["consistency"] = {
            "CL_abs_diff": abs(ii["CL_total"] - a["CL"]),
            "CD_rel_diff": abs(ii["CD_total"] - a["CD"]) / max(abs(a["CD"]), 1e-12)}
    except Exception as exc:
        rec["independent_integration_error"] = repr(exc)
    if conv == "cauchy":
        # converged = CL and CD settled (100-iteration spread) AND residual below a floor
        ok = a["finite"] and _settled(a) and a["final_rms_pressure_log10"] <= cauchy_res_floor
        rec["hit_iteration_cap"] = a["iterations"] >= ITER_CAP
        rec["status"] = "converged" if ok else "not_converged"
    else:
        rec["status"] = "converged" if a["reached_res_min"] and a["finite"] else "not_converged"


def run_case(case_id: str, geom: dict, re: float, aoa: float, spec: MeshSpec, out_root: Path,
             work_root: Path, su2: Path, su2_zip: Path, stage: str, unsteady: dict | None = None,
             res_min: float = -10.0, force: bool = False, conv: str = "residual",
             cauchy_res_floor: float = -8.0, init_from: Path | None = None,
             keep: tuple[str, ...] = KEEP_FILES) -> dict:
    """geom = {"m":..., "p":..., "t":...}; unsteady = None (steady) or kwargs for
    write_unsteady_config (dt, t_end, inner). init_from: a converged steady case directory whose
    solution, interpolated onto this mesh, is the initial condition (warm start; steady runs only).
    The converged steady state does not depend on the initial condition when the flow is steady,
    but a warm start must not be used for time-accurate runs, where it could mask an instability."""
    dest = out_root / case_id
    if not force:
        prev = existing(dest)
        if prev is not None:
            prev["resumed"] = True
            return prev
    _archive_failed(dest, out_root)
    wd = work_root / case_id
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    family = geom.get("family", "NACA4")   # V1 records are unchanged: NACA4 is the default
    rec = {"case_id": case_id, "stage": stage, "geometry": {"family": family, **{k: v for k, v in geom.items() if k != "family"}},
           "conditions": {"Re": re, "aoa_deg": aoa,
                          "model": "2D incompressible laminar NS, " + ("time-accurate" if unsteady else "steady")},
           "su2_zip_sha256": sha256(su2_zip), "python": platform.python_version(),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if family == "NACA4":
        up, lo = naca4(geom["m"], geom["p"], geom["t"])
        rec["geometry"]["checks"] = geometry_checks(up, lo, geom["t"])
    elif family == "CST":
        from geometry.cst import cst_surfaces, shape_metrics
        up, lo = cst_surfaces(geom["au"], geom["al"])
        rec["geometry"]["checks"] = shape_metrics(geom["au"], geom["al"])
    else:
        raise ValueError(f"unknown geometry family {family}")
    try:
        t0 = time.time()
        rec["mesh"] = build_mesh_isolated(up, lo, str(wd / "mesh.su2"), spec, aoa_deg=aoa)
        rec["mesh_wall_s"] = round(time.time() - t0, 1)
    except Exception as exc:  # recorded, never hidden
        rec["status"], rec["error"] = "mesh_failed", repr(exc)
        return _persist(wd, dest, rec, keep)
    if unsteady:
        write_unsteady_config(wd, re, **unsteady)
        rec["unsteady_settings"] = unsteady
    else:
        write_config(wd, re, iters=4000, cfl=50.0, res_min=res_min, conv=conv)
        rec["convergence_mode"] = conv
        if init_from is not None:
            from data.postprocess import warm_start_restart
            rec["warm_start"] = warm_start_restart(init_from, wd / "mesh.su2", wd / "solution_flow.dat")
            cfg = (wd / "config.cfg").read_text()
            cfg = cfg.replace("RESTART_SOL= NO", "RESTART_SOL= YES")
            if "SOLUTION_FILENAME=" not in cfg:
                cfg += "SOLUTION_FILENAME= solution_flow.dat\n"
            (wd / "config.cfg").write_text(cfg)
    rec["run"] = run_su2(wd, str(su2), threads=SU2_THREADS, timeout_s=6 * 3600)
    rec["run"]["threads"] = SU2_THREADS
    if rec["run"]["returncode"] != 0 or not (wd / "history.csv").exists():
        rec["status"] = "solver_failed"
        return _persist(wd, dest, rec, keep)
    if unsteady:
        rec["regime"] = classify(wd)
        rec["status"] = rec["regime"]["regime"]
    else:
        if conv == "cauchy":
            rec["continuation"] = settle(wd, su2, res_min)
        _finish_steady(wd, rec, res_min, conv, cauchy_res_floor)
    return _persist(wd, dest, rec, keep)


def continue_case(src_dir: Path, case_id: str, out_root: Path, work_root: Path, su2: Path, stage: str,
                  res_min: float = -10.0, cauchy_res_floor: float = -8.0, force: bool = False) -> dict:
    """Continue an existing steady (Cauchy-mode) case from its persisted mesh and restart file until
    settled, and record it as a NEW case (`case_id`); the source record is never modified.
    The mesh is reused byte-for-byte, so the continued solution belongs to the same discretisation."""
    dest = out_root / case_id
    if not force:
        prev = existing(dest)
        if prev is not None:
            prev["resumed"] = True
            return prev
    _archive_failed(dest, out_root)
    src = json.loads((src_dir / "case.json").read_text())
    for name in ("mesh.su2", "config.cfg", "restart_flow.dat", "history.csv"):
        if sha256(src_dir / name) != src["files_sha256"][name]:
            raise RuntimeError(f"{src_dir.name}/{name} does not match its recorded checksum")
    wd = work_root / case_id
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    for name in KEEP_FILES:  # mesh, config, restart, history, log and the (pre-continuation) outputs
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, wd / name)
    rec = {k: src[k] for k in ("geometry", "conditions", "mesh", "mesh_wall_s", "su2_zip_sha256", "python",
                               "convergence_mode") if k in src}
    rec.update({"case_id": case_id, "stage": stage,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "continued_from": {"case_id": src["case_id"], "status": src.get("status"),
                                   "assessment": src.get("assessment"), "run": src.get("run"),
                                   "files_sha256": {k: src["files_sha256"][k] for k in
                                                    ("mesh.su2", "config.cfg", "restart_flow.dat", "history.csv")}}})
    t0 = time.time()
    rec["continuation"] = settle(wd, su2, res_min)
    rec["run"] = {"returncode": 0 if not any(g.get("failed") for g in rec["continuation"]["segments"]) else 1,
                  "wall_s": round(time.time() - t0, 1),
                  "wall_s_including_source_run": round(time.time() - t0 + src.get("run", {}).get("wall_s", 0.0), 1)}
    if rec["run"]["returncode"] != 0:
        rec["status"] = "solver_failed"
        return _persist(wd, dest, rec)
    _finish_steady(wd, rec, res_min, "cauchy", cauchy_res_floor)
    return _persist(wd, dest, rec)
