"""Turn a finished steady case into fixed-format learning targets (AeroForge-TRIAD Dataset V1).

Per case:
* coefficients: CL, CD (from SU2 history; independently re-integrated values are kept in case.json);
* surface distributions: Cp and the skin-friction vector on a fixed parametric grid beta in [0, pi]
  per side (upper / lower), beta = arccos(1 - 2 x) on the *unrotated* airfoil;
* volume fields: pressure and velocity on a fixed Cartesian grid in the wind frame, with a mask for
  grid points inside the body and the signed distance to the airfoil surface.

Everything here is read from SU2's own outputs (restart_flow.dat, surface_flow.vtk, mesh.su2); no value
is invented or smoothed. Interpolation is linear on the actual mesh cells.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfd.shape_sensitivity import locate_surface_nodes, read_su2_mesh, sides_from_loop, surface_point

RESTART_MAGIC = 535532


def read_restart(path: str | Path) -> dict[str, np.ndarray]:
    """SU2 binary restart (v7+ format): header of 5 int32, field names (33 bytes each), float64 data."""
    raw = Path(path).read_bytes()
    magic, nf, npnt = struct.unpack("3i", raw[:12])
    if magic != RESTART_MAGIC:
        raise ValueError(f"{path}: not an SU2 binary restart (magic {magic})")
    off = 20
    names = [raw[off + 33 * i: off + 33 * (i + 1)].split(b"\x00")[0].decode() for i in range(nf)]
    off += 33 * nf
    data = np.frombuffer(raw[off: off + 8 * nf * npnt], dtype="<f8").reshape(npnt, nf)
    return {n: data[:, i].copy() for i, n in enumerate(names)}


@dataclass(frozen=True)
class GridSpec:
    """Fixed Cartesian grid in the wind frame (free stream along +x), chord 1, LE of the unrotated
    airfoil at the origin, rotation about (0.25, 0)."""
    x0: float = -0.5
    x1: float = 1.5
    y0: float = -0.5
    y1: float = 0.5
    nx: int = 192
    ny: int = 96

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(np.linspace(self.x0, self.x1, self.nx), np.linspace(self.y0, self.y1, self.ny))


def mesh_triangles(mesh_path: str | Path) -> np.ndarray:
    """All 2D cells as triangles (quads split along a diagonal)."""
    lines = Path(mesh_path).read_text().splitlines()
    for i, l in enumerate(lines):
        if l.strip().startswith("NELEM="):
            nel = int(l.split("=")[1])
            tris = []
            for k in range(nel):
                v = [int(x) for x in lines[i + 1 + k].split()]
                if v[0] == 5:
                    tris.append(v[1:4])
                elif v[0] == 9:
                    tris.append([v[1], v[2], v[3]])
                    tris.append([v[1], v[3], v[4]])
            return np.asarray(tris, dtype=int)
    raise ValueError(f"{mesh_path}: no NELEM block")


def signed_distance(gx: np.ndarray, gy: np.ndarray, theta, n_surf: int = 2001) -> np.ndarray:
    """Signed distance to the (rotated) airfoil surface: negative inside the body."""
    from matplotlib.path import Path as MPath
    from scipy.spatial import cKDTree

    b = np.linspace(0.0, np.pi, n_surf)
    up = surface_point(np.ones_like(b), b, theta)
    lo = surface_point(-np.ones_like(b), b, theta)
    loop = np.vstack([up, lo[::-1][1:]])
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    d, _ = cKDTree(loop).query(pts)
    inside = MPath(loop).contains_points(pts)
    return np.where(inside, -d, d).reshape(gx.shape)


def fields_on_grid(case_dir: str | Path, theta, grid: GridSpec = GridSpec()) -> dict[str, np.ndarray]:
    import matplotlib.tri as mtri

    case_dir = Path(case_dir)
    rs = read_restart(case_dir / "restart_flow.dat")
    tri = mtri.Triangulation(rs["x"], rs["y"], mesh_triangles(case_dir / "mesh.su2"))
    gx, gy = grid.coords()
    sdf = signed_distance(gx, gy, theta)
    out = {"sdf": sdf.astype(np.float32), "mask_fluid": (sdf > 0).astype(np.uint8)}
    for name, key in (("p", "Pressure"), ("u", "Velocity_x"), ("v", "Velocity_y")):
        f = mtri.LinearTriInterpolator(tri, rs[key])(gx, gy)
        arr = np.asarray(f.filled(np.nan), dtype=np.float64)
        arr[sdf <= 0] = 0.0  # inside the body: no flow (no-slip wall; pressure undefined, set to 0 and masked)
        out[name] = arr.astype(np.float32)
    n_nan = int(np.isnan(out["p"]).sum())
    if n_nan:
        raise ValueError(f"{case_dir.name}: {n_nan} fluid grid points not covered by the mesh")
    return out


def surface_on_beta(case_dir: str | Path, theta, n_beta: int = 129) -> dict[str, np.ndarray]:
    """Cp and skin-friction (x, y) on a fixed beta grid per side, from surface_flow.vtk."""
    import meshio

    m = meshio.read(Path(case_dir) / "surface_flow.vtk")
    xy = m.points[:, :2]
    cp = np.asarray(m.point_data["Pressure_Coefficient"]).reshape(-1)
    cf = np.asarray(m.point_data["Skin_Friction_Coefficient"])[:, :2]
    lines = np.vstack([c.data for c in m.cells if c.type == "line"])
    used = np.unique(lines.ravel())
    loc = np.searchsorted(used, lines)
    side_u = sides_from_loop(xy[used], loc, theta)
    side_u, beta_u, gap = locate_surface_nodes(xy[used], np.asarray(theta, dtype=float), side=side_u)
    xy, cp, cf = xy[used], cp[used], cf[used]
    side, beta = side_u, beta_u
    bgrid = np.linspace(0.0, np.pi, n_beta)
    out = {"beta": bgrid, "surface_gap": gap}
    for s, tag in ((1.0, "upper"), (-1.0, "lower")):
        sel = side == s
        # the LE and TE nodes are shared; include them on both sides
        sel |= (beta < 1e-3) | (beta > np.pi - 1e-3)  # LE / TE nodes (node spacing there is >> 1e-3 in beta)
        order = np.argsort(beta[sel])
        b = beta[sel][order]
        b, uniq = np.unique(b, return_index=True)
        for name, arr in (("cp", cp), ("cfx", cf[:, 0]), ("cfy", cf[:, 1])):
            vals = arr[sel][order][uniq]
            out[f"{name}_{tag}"] = np.interp(bgrid, b, vals).astype(np.float32)
    return out


def write_restart(path: str | Path, fields: dict[str, np.ndarray]) -> None:
    """Write an SU2 binary restart (same layout as read_restart; the two trailing header ints are 0,
    as in the files SU2 v8.5.0 writes for steady runs)."""
    names = list(fields)
    n = len(next(iter(fields.values())))
    hdr = struct.pack("5i", RESTART_MAGIC, len(names), n, 0, 0)
    nm = b"".join(name.encode().ljust(33, b"\x00")[:33] for name in names)
    data = np.column_stack([np.asarray(fields[k], dtype="<f8") for k in names]).astype("<f8").tobytes()
    Path(path).write_bytes(hdr + nm + data)


def warm_start_restart(donor_case: str | Path, target_mesh: str | Path, out_path: str | Path) -> dict:
    """Initial solution for a new mesh by linear interpolation of a converged donor solution on the
    donor's own cells (nearest donor node where the target node is outside the donor's fluid, e.g.
    inside the donor airfoil). Wall nodes get zero velocity (no-slip)."""
    import matplotlib.tri as mtri
    from scipy.spatial import cKDTree

    donor_case = Path(donor_case)
    rs = read_restart(donor_case / "restart_flow.dat")
    tri = mtri.Triangulation(rs["x"], rs["y"], mesh_triangles(donor_case / "mesh.su2"))
    tgt = read_su2_mesh(target_mesh)
    x, y = tgt.points[:, 0], tgt.points[:, 1]
    out = {"x": x, "y": y}
    _, nn = cKDTree(np.column_stack([rs["x"], rs["y"]])).query(tgt.points)
    n_fill = 0
    for key in ("Pressure", "Velocity_x", "Velocity_y"):
        v = mtri.LinearTriInterpolator(tri, rs[key])(x, y)
        arr = np.asarray(v.filled(np.nan), dtype=float)
        miss = np.isnan(arr)
        arr[miss] = rs[key][nn[miss]]
        n_fill = int(miss.sum())
        out[key] = arr
    wall = np.unique(tgt.markers["airfoil"].ravel())
    out["Velocity_x"][wall] = 0.0
    out["Velocity_y"][wall] = 0.0
    write_restart(out_path, out)
    return {"donor": donor_case.name, "n_points": int(len(x)), "n_nearest_fill": n_fill}
