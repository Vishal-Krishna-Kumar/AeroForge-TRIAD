"""Learning targets for V3 (steady compressible Euler) cases, in the same array layout as data.postprocess (V1/V2),
so the V2 surrogate code can be reused unchanged:
* volume fields on the same fixed grid: p := pressure coefficient Cp, u := u/U_inf, v := v/U_inf
  (non-dimensional, computed from SU2's Pressure_Coefficient and conservative variables in restart_flow.dat);
* surface: Cp on the same beta grid per side; the skin-friction arrays cfx/cfy are identically zero (inviscid).
Nothing is smoothed or invented: values are SU2's own outputs, interpolated linearly on the mesh cells.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from cfd.shape_sensitivity import locate_surface_nodes, sides_from_loop
from data.postprocess import GridSpec, mesh_triangles, read_restart, signed_distance

GAMMA, R_GAS, T_INF = 1.4, 287.058, 288.15


def u_inf(mach: float) -> float:
    return mach * math.sqrt(GAMMA * R_GAS * T_INF)


P_INF = 101325.0


def primitive_nondim(rs: dict, mach: float) -> dict[str, np.ndarray]:
    """Cp, u/U_inf, v/U_inf from the conservative variables of the restart (the restart holds only
    Density, Momentum_x/_y, Energy). Free stream as in su2_euler: p 101325 Pa, T 288.15 K, R 287.058, gamma 1.4."""
    rho, mx, my, en = rs["Density"], rs["Momentum_x"], rs["Momentum_y"], rs["Energy"]
    p = (GAMMA - 1.0) * (en - 0.5 * (mx * mx + my * my) / rho)
    ui = u_inf(mach)
    q_inf = 0.5 * (P_INF / (R_GAS * T_INF)) * ui * ui
    return {"p": (p - P_INF) / q_inf, "u": mx / rho / ui, "v": my / rho / ui}


def fields_on_grid_euler(case_dir: str | Path, theta, mach: float, grid: GridSpec = GridSpec()) -> dict[str, np.ndarray]:
    import matplotlib.tri as mtri

    case_dir = Path(case_dir)
    rs = read_restart(case_dir / "restart_flow.dat")
    vals = primitive_nondim(rs, mach)
    tri = mtri.Triangulation(rs["x"], rs["y"], mesh_triangles(case_dir / "mesh.su2"))
    gx, gy = grid.coords()
    sdf = signed_distance(gx, gy, theta)
    out = {"sdf": sdf.astype(np.float32), "mask_fluid": (sdf > 0).astype(np.uint8)}
    for name, arr_pts in vals.items():
        f = mtri.LinearTriInterpolator(tri, arr_pts)(gx, gy)
        arr = np.asarray(f.filled(np.nan), dtype=np.float64)
        arr[sdf <= 0] = 0.0  # inside the body: masked
        out[name] = arr.astype(np.float32)
    n_nan = int(np.isnan(out["p"]).sum())
    if n_nan:
        raise ValueError(f"{case_dir.name}: {n_nan} fluid grid points not covered by the mesh")
    return out


def surface_on_beta_euler(case_dir: str | Path, theta, n_beta: int = 129) -> dict[str, np.ndarray]:
    import meshio

    m = meshio.read(Path(case_dir) / "surface_flow.vtk")
    xy = m.points[:, :2]
    cp = np.asarray(m.point_data["Pressure_Coefficient"]).reshape(-1)
    lines = np.vstack([c.data for c in m.cells if c.type == "line"])
    used = np.unique(lines.ravel())
    loc = np.searchsorted(used, lines)
    side = sides_from_loop(xy[used], loc, theta)
    side, beta, gap = locate_surface_nodes(xy[used], np.asarray(theta, dtype=float), side=side)
    cp = cp[used]
    bgrid = np.linspace(0.0, np.pi, n_beta)
    out = {"beta": bgrid, "surface_gap": gap}
    for s, tag in ((1.0, "upper"), (-1.0, "lower")):
        sel = (side == s) | (beta < 1e-3) | (beta > np.pi - 1e-3)
        order = np.argsort(beta[sel])
        b, uniq = np.unique(beta[sel][order], return_index=True)
        out[f"cp_{tag}"] = np.interp(bgrid, b, cp[sel][order][uniq]).astype(np.float32)
        out[f"cfx_{tag}"] = np.zeros(n_beta, np.float32)
        out[f"cfy_{tag}"] = np.zeros(n_beta, np.float32)
    return out
