"""Reference design gradients for NACA 4-digit cases: SU2 discrete adjoint + mesh-deformation chain rule,
verified against central finite differences on deformed meshes (AeroForge-TRIAD).

Both routes use the same theta -> mesh map (cfd.shape_sensitivity.DesignMap), so they must agree up to
iterative and truncation error. The adjoint route gives all four derivatives (m, p, t, aoa) of one
objective from one adjoint solve; the FD route costs two primal solves per derivative and is used to
verify the adjoint, not to produce the dataset gradients.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from cfd.shape_sensitivity import DV_NAMES, DesignMap, dv_names, chain_rule, min_cell_jacobian_ratio, read_su2_mesh, write_deformed_mesh
from cfd.su2_case import read_history

OBJECTIVES = {"CL": "LIFT", "CD": "DRAG"}


def set_key(text: str, key: str, value: str) -> str:
    if re.search(r"(?m)^" + re.escape(key) + r"\s*=", text):
        return re.sub(r"(?m)^" + re.escape(key) + r"\s*=.*$", f"{key}= {value}", text)
    return text.rstrip("\n") + f"\n{key}= {value}\n"


def drop_keys(text: str, prefixes: tuple[str, ...]) -> str:
    return "".join(l for l in text.splitlines(keepends=True) if not l.strip().startswith(prefixes))


def adjoint_config(base_cfg: str, objective: str, iters: int = 3000, res_min: float = -11.0,
                   weights: str | None = None, conv_field: str = "RMS_ADJ_PRESSURE", extra: dict | None = None) -> str:
    """Discrete-adjoint config derived from the case's own primal config (same physics/numerics).
    weights: SU2 OBJECTIVE_WEIGHT for a combined objective, e.g. objective "LIFT, DRAG" with weights "a, b"
    gives the gradient of a*CL + b*CD from ONE adjoint solve (V2 cost saving; validated against two solves).
    conv_field: adjoint convergence field; the default is the V1/V2 (incompressible) one, V3 Euler passes
    RMS_ADJ_DENSITY (2026-10-14; defaults unchanged so V1/V2 configs are reproduced exactly)."""
    t = drop_keys(base_cfg, ("CONV_CAUCHY_",))
    for k, v in (("MATH_PROBLEM", "DISCRETE_ADJOINT"), ("RESTART_SOL", "NO"),
                 ("SOLUTION_FILENAME", "solution_flow.dat"), ("OBJECTIVE_FUNCTION", objective),
                 ("ITER", str(iters)), ("CONV_FIELD", conv_field), ("CONV_RESIDUAL_MINVAL", str(res_min)),
                 ("CONV_STARTITER", "10"), ("OUTPUT_FILES", "(RESTART, CSV, SURFACE_CSV)"),
                 ("VOLUME_OUTPUT", "(COORDINATES, SOLUTION, SENSITIVITY)"),
                 ("HISTORY_OUTPUT", "(ITER, RMS_RES, SENSITIVITY)"),
                 ("SCREEN_OUTPUT", "(INNER_ITER, RMS_ADJ_PRESSURE, SENS_AOA)"), ("OUTPUT_WRT_FREQ", str(iters)),
                 ("RESTART_ADJ_FILENAME", "restart_adj.dat"), ("SOLUTION_ADJ_FILENAME", "solution_adj.dat"),
                 ("VOLUME_ADJ_FILENAME", "adjoint"), ("SURFACE_ADJ_FILENAME", "surface_adjoint")):
        t = set_key(t, k, v)
    if weights is not None:
        t = set_key(t, "OBJECTIVE_WEIGHT", weights)
    for k, v in (extra or {}).items():  # protocol v3 amendment 3: e.g. MGLEVEL 0 for the V3 adjoint
        t = set_key(t, k, v)
    return t


def read_volume_sensitivity(csv_path: Path, n_points: int) -> np.ndarray:
    """(N, 2) dJ/dX from SU2's volume CSV (columns Sensitivity_x/_y, rows in point order or with PointID)."""
    with open(csv_path) as fh:
        rows = list(csv.reader(fh))
    head = [h.strip().strip('"') for h in rows[0]]
    data = np.array([[float(v) for v in r] for r in rows[1:] if r])
    sx = next(i for i, h in enumerate(head) if h.lower() in ("sensitivity_x", "sens_x"))
    sy = next(i for i, h in enumerate(head) if h.lower() in ("sensitivity_y", "sens_y"))
    out = np.zeros((n_points, 2))
    if "PointID" in head:
        pid = data[:, head.index("PointID")].astype(int)
        out[pid, 0], out[pid, 1] = data[:, sx], data[:, sy]
    else:
        if len(data) != n_points:
            raise ValueError(f"{csv_path}: {len(data)} rows for {n_points} points and no PointID column")
        out[:, 0], out[:, 1] = data[:, sx], data[:, sy]
    return out


def sensitivity_csv(sub: Path) -> Path | None:
    """Volume sensitivity file of an adjoint run. SU2 8.5 writes the volume CSV as the ASCII adjoint restart
    with the objective suffix (restart_adj_cl.csv / restart_adj_cd.csv; columns PointID, x, y, Adjoint_*,
    Sensitivity_x, Sensitivity_y, Surface_Sensitivity), not as <VOLUME_ADJ_FILENAME>.csv."""
    hits = sorted(sub.glob("restart_adj*.csv")) or sorted(sub.glob("adjoint*.csv"))
    return hits[0] if hits else None


def adjoint_finished(sub: Path) -> bool:
    log = sub / "adjoint.log"
    return log.exists() and "Exit Success" in log.read_text(errors="replace") and sensitivity_csv(sub) is not None


def run(cmd: list[str], wd: Path, log: str, env: dict | None = None, timeout: int = 6 * 3600) -> dict:
    t0 = time.time()
    with open(wd / log, "w") as fh:
        p = subprocess.run(cmd, cwd=wd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return {"returncode": p.returncode, "wall_s": round(time.time() - t0, 1)}


def adjoint_gradients(case_dir: Path, wd: Path, theta0, su2_ad: str, threads: int = 2, radius: float = 0.5,
                      objectives=("CL", "CD"), env: dict | None = None, reuse: bool = True,
                      combined: dict | None = None, adj_conv_field: str = "RMS_ADJ_PRESSURE",
                      adj_res_min: float = -11.0, adj_iters: int = 3000, adj_extra: dict | None = None) -> dict:
    """Adjoint design gradients of CL and CD for a converged steady case (mesh.su2, config.cfg,
    restart_flow.dat in case_dir). combined = {name: (wL, wD)} adds one adjoint per entry for the
    objective wL*CL + wD*CD (e.g. the L/D gradient with wL = 1/CD, wD = -CL/CD^2)."""
    names = dv_names(theta0)
    wd.mkdir(parents=True, exist_ok=True)
    for f in ("mesh.su2", "config.cfg", "restart_flow.dat"):
        shutil.copy2(case_dir / f, wd / f)
    shutil.copy2(wd / "restart_flow.dat", wd / "solution_flow.dat")
    mesh = read_su2_mesh(wd / "mesh.su2")
    dmap = DesignMap(mesh, theta0, radius=radius)
    dX = dmap.dX_dtheta()
    out = {"theta0": list(map(float, theta0)), "dv": list(names), "surface_gap": dmap.surface_gap,
           "rbf_radius": radius, "objectives": {}}
    base = (wd / "config.cfg").read_text()
    jobs = [(n, OBJECTIVES[n], None) for n in objectives]
    jobs += [(n, "LIFT, DRAG", f"{w[0]!r}, {w[1]!r}") for n, w in (combined or {}).items()]
    for name, su2_obj, weights in jobs:
        sub = wd / f"adj_{name}"
        sub.mkdir(exist_ok=True)
        cfg = adjoint_config(base, su2_obj, iters=adj_iters, weights=weights, conv_field=adj_conv_field,
                             res_min=adj_res_min, extra=adj_extra)
        if (reuse and adjoint_finished(sub) and (sub / "config.cfg").read_text() == cfg
                and (sub / "solution_flow.dat").read_bytes() == (wd / "solution_flow.dat").read_bytes()):
            r = {"returncode": 0, "reused": True}  # same config and primal solution: identical inputs, not re-solved
        else:
            for f in ("mesh.su2", "solution_flow.dat"):
                shutil.copy2(wd / f, sub / f)
            (sub / "config.cfg").write_text(cfg)
            r = run([su2_ad, "-t", str(threads), "config.cfg"], sub, "adjoint.log", env=env)
        rec = {"run": r}
        csv_path = sensitivity_csv(sub)
        if r["returncode"] == 0 and csv_path is None:
            rec["error"] = "adjoint exited 0 but no sensitivity CSV was found"
        if r["returncode"] == 0 and csv_path is not None:
            rec["sensitivity_file"] = csv_path.name
            sens = read_volume_sensitivity(csv_path, len(mesh.points))
            g = chain_rule(sens, dX)
            h = read_history(sub)
            rk = next(k for k in h if "rms[A_P]" in k or "ADJ_PRESSURE" in k.upper() or "rms[A_Rho]" in k)
            rec.update({"grad": dict(zip(names, map(float, g))), "iterations": int(len(h[rk])),
                        "final_rms_adj_p": float(h[rk][-1]), "adj_residual_key": rk,
                        "surface_share": float(np.abs(chain_rule(sens[dmap.surf_nodes], dX[dmap.surf_nodes])).sum()
                                               / max(np.abs(g).sum(), 1e-30))})
        out["objectives"][name] = rec
    return out


def primal_on_points(case_dir: Path, wd: Path, points: np.ndarray, su2: str, iters: int = 1500,
                     threads: int = 2, env: dict | None = None, conv_field: str = "RMS_PRESSURE",
                     res_key: str = "rms[P]") -> dict:
    """Primal solve on the case mesh with moved nodes, restarted from the base solution, fixed iteration
    count (unreachable residual target). Returns final CL, CD and their last-100-iteration spreads.
    conv_field/res_key: V1/V2 defaults; V3 Euler passes RMS_DENSITY / rms[Rho]."""
    wd.mkdir(parents=True, exist_ok=True)
    mesh = read_su2_mesh(case_dir / "mesh.su2")
    write_deformed_mesh(mesh, points, wd / "mesh.su2")
    shutil.copy2(case_dir / "restart_flow.dat", wd / "solution_flow.dat")
    t = drop_keys((case_dir / "config.cfg").read_text(), ("CONV_CAUCHY_",))
    for k, v in (("RESTART_SOL", "YES"), ("SOLUTION_FILENAME", "solution_flow.dat"), ("ITER", str(iters)),
                 ("CONV_FIELD", conv_field), ("CONV_RESIDUAL_MINVAL", "-16"), ("OUTPUT_WRT_FREQ", str(iters)),
                 ("OUTPUT_FILES", "(RESTART)")):
        t = set_key(t, k, v)
    (wd / "config.cfg").write_text(t)
    r = run([su2, "-t", str(threads), "config.cfg"], wd, "solver.log", env=env)
    h = read_history(wd)
    cl, cd = h["CL"], h["CD"]
    rp = next(k for k in h if res_key in k)
    return {"run": r, "CL": float(cl[-1]), "CD": float(cd[-1]), "rms_p": float(h[rp][-1]),
            "CL_abs_spread100": float(np.ptp(cl[-100:])), "CD_abs_spread100": float(np.ptp(cd[-100:])),
            "min_area_ratio": min_cell_jacobian_ratio(mesh, points)}


def fd_gradients(case_dir: Path, wd: Path, theta0, su2: str, steps=None, radius: float = 0.5,
                 iters: int = 1500, threads: int = 2, env: dict | None = None, which=None,
                 conv_field: str = "RMS_PRESSURE", res_key: str = "rms[P]") -> dict:
    """steps default: V1 NACA (1e-3, 1e-2, 1e-3, 0.1); CST 1e-3 per coefficient and 0.1 deg for the angle."""
    names = dv_names(theta0)
    if steps is None:
        steps = (1e-3, 1e-2, 1e-3, 0.1) if len(names) == 4 else (1e-3,) * (len(names) - 1) + (0.1,)
    which = names if which is None else which
    mesh = read_su2_mesh(case_dir / "mesh.su2")
    dmap = DesignMap(mesh, theta0, radius=radius)
    out = {"steps": dict(zip(names, steps)), "solves": {}, "grad": {"CL": {}, "CD": {}}}
    for k, name in enumerate(names):
        if name not in which:
            continue
        res = {}
        for sgn in (+1, -1):
            e = np.zeros(len(names)); e[k] = sgn * steps[k]
            res[sgn] = primal_on_points(case_dir, wd / f"fd_{name}_{'p' if sgn > 0 else 'm'}",
                                        dmap.points(np.asarray(theta0) + e), su2, iters, threads, env,
                                        conv_field=conv_field, res_key=res_key)
        out["solves"][name] = {"plus": res[1], "minus": res[-1]}
        for q in ("CL", "CD"):
            out["grad"][q][name] = (res[1][q] - res[-1][q]) / (2 * steps[k])
    return out
