"""Design-variable sensitivities for NACA 4-digit airfoils on a fixed mesh topology (AeroForge-TRIAD).

Design variables theta = (m, p, t, aoa_deg). The mesh of the base design is deformed (never re-meshed)
to represent a perturbed design:

1. Every airfoil surface node is assigned its parametric coordinate (side, beta) on the analytic
   NACA surface (x = (1 - cos beta) / 2, rotated by aoa about the pivot).
2. A perturbation of theta moves each surface node to the same (side, beta) on the perturbed surface.
3. The surface displacement is blended into the volume with a compactly supported Wendland C2 radial
   basis function interpolation (support radius R), so near-wall cells move almost rigidly and the
   far field does not move.

The map theta -> node coordinates X(theta) is therefore smooth and fully defined by this module. Two
consumers use the same map, which makes them consistent by construction:

* adjoint chain rule: dJ/dtheta = sum_nodes dJ/dX . dX/dtheta, with dJ/dX the SU2 discrete-adjoint
  volume sensitivity;
* finite-difference verification: J(X(theta + h e_k)) - J(X(theta - h e_k)) over 2h, solved on the
  deformed meshes written by `write_deformed_mesh`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from geometry.naca import A4_CLOSED_TE

DV_NAMES = ("m", "p", "t", "aoa_deg")


def dv_names(theta) -> tuple[str, ...]:
    """Design-variable names: NACA (len 4) -> DV_NAMES; CST (len 2n+1) -> au0.., al0.., aoa_deg."""
    k = len(theta)
    if k == 4:
        return DV_NAMES
    n = (k - 1) // 2
    return tuple(f"au{i}" for i in range(n)) + tuple(f"al{i}" for i in range(n)) + ("aoa_deg",)


# --------------------------------------------------------------------------------------------- mesh I/O
@dataclass
class SU2Mesh:
    lines: list[str]            # the file, line by line (with line endings)
    point_line0: int            # index of the first point line
    points: np.ndarray          # (N, 2) coordinates, in file order (SU2 point index = row)
    markers: dict[str, np.ndarray]  # marker -> (E, 2) edge node indices


def read_su2_mesh(path: str | Path) -> SU2Mesh:
    lines = Path(path).read_text().splitlines(keepends=True)
    i, n = 0, len(lines)
    points, point_line0, markers = None, -1, {}
    while i < n:
        s = lines[i].strip()
        if s.startswith("NPOIN="):
            npoin = int(s.split("=")[1].split()[0])
            point_line0 = i + 1
            pts = np.array([[float(v) for v in lines[point_line0 + k].split()[:2]] for k in range(npoin)])
            points = pts
            i = point_line0 + npoin
            continue
        if s.startswith("MARKER_TAG="):
            name = s.split("=")[1].strip()
            nel = int(lines[i + 1].split("=")[1])
            edges = np.array([[int(v) for v in lines[i + 2 + k].split()[1:3]] for k in range(nel)], dtype=int)
            markers[name] = edges
            i += 2 + nel
            continue
        i += 1
    if points is None:
        raise ValueError(f"{path}: no NPOIN block")
    return SU2Mesh(lines, point_line0, points, markers)


def write_deformed_mesh(mesh: SU2Mesh, new_points: np.ndarray, out_path: str | Path) -> None:
    """Rewrite only the coordinates; every other line is kept byte-identical."""
    if new_points.shape != mesh.points.shape:
        raise ValueError("point array shape mismatch")
    out = list(mesh.lines)
    for k, (x, y) in enumerate(new_points):
        old = out[mesh.point_line0 + k].split()
        tail = " ".join(old[2:])
        out[mesh.point_line0 + k] = f"{x:.17g}\t{y:.17g}\t{tail}\n"
    Path(out_path).write_text("".join(out))


# ------------------------------------------------------------------------------------- analytic surface
def surface_point(side: np.ndarray, beta: np.ndarray, theta: np.ndarray, pivot=(0.25, 0.0)) -> np.ndarray:
    """Point on the rotated airfoil surface for parametric coordinates (side=+1 upper, -1 lower; beta).
    The geometry family follows from len(theta): 4 -> NACA 4-digit (m, p, t, aoa_deg) (V1, unchanged);
    2n+1 -> CST (A_u[0..n-1], A_l[0..n-1], aoa_deg) (V2, geometry.cst).
    NACA: the camber formula is used for every m (m = 0 gives zero camber),
    so derivatives with respect to m exist at m = 0 for any p in (0, 1)."""
    if len(theta) != 4:
        return _surface_point_cst(side, beta, theta, pivot)
    m, p, t, aoa = (float(v) for v in theta)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + A4_CLOSED_TE * x**4)
    front = x < p
    yc = np.where(front, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(front, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xs = x - side * yt * np.sin(th)
    ys = yc + side * yt * np.cos(th)
    a = np.deg2rad(aoa)
    c, s = np.cos(a), np.sin(a)
    px, py = pivot
    xr = (xs - px) * c + (ys - py) * s + px
    yr = -(xs - px) * s + (ys - py) * c + py
    return np.column_stack([xr, yr])


def _rotate_xy(xs, ys, aoa, pivot):
    a = np.deg2rad(aoa)
    c, s = np.cos(a), np.sin(a)
    px, py = pivot
    return np.column_stack([(xs - px) * c + (ys - py) * s + px, -(xs - px) * s + (ys - py) * c + py])


def _surface_point_cst(side, beta, theta, pivot=(0.25, 0.0)) -> np.ndarray:
    """CST surface point: x = (1 - cos beta)/2 fixed, y from the upper (side +1) or lower (side -1) CST polynomial."""
    from geometry.cst import split_theta, surface_y
    theta = np.asarray(theta, dtype=float)
    au, al, aoa = split_theta(theta, (len(theta) - 1) // 2)
    side = np.broadcast_to(np.asarray(side, dtype=float), np.shape(beta))
    x = 0.5 * (1.0 - np.cos(beta))
    ys = np.where(side > 0, surface_y(x, au), surface_y(x, al))
    return _rotate_xy(x, ys, aoa, pivot)


def sides_from_loop(xy: np.ndarray, edges_local: np.ndarray, theta) -> np.ndarray:
    """Side (+1 upper, -1 lower) of every surface node from the closed wall loop topology: the loop is
    split at the leading-edge node (min x) and trailing-edge node (max x) of the unrotated airfoil, and
    the chain with the larger mean y is the upper side. Distance-based side assignment is ambiguous at
    the thin trailing edge, so topology is used instead. edges_local index rows of xy."""
    aoa = np.deg2rad(float(theta[3]))
    c, s = np.cos(aoa), np.sin(aoa)
    px, py = 0.25, 0.0
    xu = (xy[:, 0] - px) * c - (xy[:, 1] - py) * s + px  # inverse of the nose-up rotation
    yu = (xy[:, 0] - px) * s + (xy[:, 1] - py) * c + py
    nbr = {i: [] for i in range(len(xy))}
    for a, b in edges_local:
        nbr[int(a)].append(int(b)); nbr[int(b)].append(int(a))
    if any(len(v) != 2 for v in nbr.values()):
        raise ValueError("wall marker is not a single closed loop")
    le, te = int(np.argmin(xu)), int(np.argmax(xu))
    chains = []
    for start in nbr[le]:
        chain, prev, cur = [le], le, start
        while cur != te:
            chain.append(cur)
            nxt = nbr[cur][0] if nbr[cur][0] != prev else nbr[cur][1]
            prev, cur = cur, nxt
        chains.append(chain[1:])
    side = np.zeros(len(xy))
    up_first = np.mean(yu[chains[0]]) > np.mean(yu[chains[1]])
    side[chains[0]] = 1.0 if up_first else -1.0
    side[chains[1]] = -1.0 if up_first else 1.0
    side[le] = side[te] = 1.0  # shared points: beta = 0 (LE) or pi (TE) gives the same point on both sides
    return side


def locate_surface_nodes(xy: np.ndarray, theta: np.ndarray, n_dense: int = 40001,
                         side: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Parametric coordinates (side, beta) of mesh surface nodes on the analytic surface of design theta.
    If `side` is given (recommended: sides_from_loop), beta is searched on that side only.
    Returns (side, beta, max_distance): max_distance is the largest node-to-surface gap (should be
    ~1e-6 or less; the Gmsh spline through 201 points differs slightly from the analytic curve)."""
    b = np.linspace(0.0, np.pi, n_dense)
    if side is None:
        cand_side = np.concatenate([np.ones(n_dense), -np.ones(n_dense)])
        cand_beta = np.concatenate([b, b])
        _, idx = cKDTree(surface_point(cand_side, cand_beta, theta)).query(xy)
        side, beta = cand_side[idx].copy(), cand_beta[idx].copy()
    else:
        side = np.asarray(side, dtype=float).copy()
        beta = np.zeros(len(xy))
        for sv in (1.0, -1.0):
            sel = side == sv
            if sel.any():
                _, idx = cKDTree(surface_point(np.full(n_dense, sv), b, theta)).query(xy[sel])
                beta[sel] = b[idx]
    # refine beta locally by golden-section on each node's own side (distance is unimodal locally)
    h = np.pi / (n_dense - 1)
    lo, hi = np.clip(beta - h, 0.0, np.pi), np.clip(beta + h, 0.0, np.pi)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):
        b1 = hi - gr * (hi - lo)
        b2 = lo + gr * (hi - lo)
        d1 = np.linalg.norm(surface_point(side, b1, theta) - xy, axis=1)
        d2 = np.linalg.norm(surface_point(side, b2, theta) - xy, axis=1)
        left = d1 < d2
        hi = np.where(left, b2, hi)
        lo = np.where(left, lo, b1)
    beta = 0.5 * (lo + hi)
    gap = float(np.max(np.linalg.norm(surface_point(side, beta, theta) - xy, axis=1)))
    return side, beta, gap


# --------------------------------------------------------------------------------- volume deformation
def wendland_c2(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, 0.0, 1.0)
    return (1.0 - r) ** 4 * (4.0 * r + 1.0)


class RBFDeformer:
    """Linear map from surface-node displacements to volume-node displacements (Wendland C2, support R)."""

    def __init__(self, points: np.ndarray, surf_nodes: np.ndarray, radius: float = 1.0):
        self.points = points
        self.surf_nodes = np.asarray(surf_nodes, dtype=int)
        self.radius = float(radius)
        xs = points[self.surf_nodes]
        tree_s = cKDTree(xs)
        pairs = tree_s.query_pairs(self.radius, output_type="ndarray")
        n = len(xs)
        r = np.linalg.norm(xs[pairs[:, 0]] - xs[pairs[:, 1]], axis=1) / self.radius
        v = wendland_c2(r)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        vals = np.concatenate([v, v, np.ones(n)])
        self._lu = splu(csr_matrix((vals, (rows, cols)), shape=(n, n)).tocsc())
        # evaluation matrix volume <- surface centres (only nodes within the support)
        tree_v = cKDTree(points)
        sm = tree_s.sparse_distance_matrix(tree_v, self.radius, output_type="coo_matrix")
        phi = wendland_c2(sm.data / self.radius)
        self._E = csr_matrix((phi, (sm.col, sm.row)), shape=(len(points), n))

    def volume_displacement(self, surf_disp: np.ndarray) -> np.ndarray:
        w = self._lu.solve(np.asarray(surf_disp, dtype=float))
        d = self._E @ w
        d[self.surf_nodes] = surf_disp  # exact on the surface (the interpolant already is, up to round-off)
        return d


# ------------------------------------------------------------------------------------- design mapping
class DesignMap:
    """theta -> deformed node coordinates, built around the base design theta0 and its mesh."""

    def __init__(self, mesh: SU2Mesh, theta0, wall_marker: str = "airfoil", radius: float = 1.0):
        self.mesh = mesh
        self.theta0 = np.asarray(theta0, dtype=float)
        edges = mesh.markers[wall_marker]
        self.surf_nodes = np.unique(edges.ravel())
        xy = mesh.points[self.surf_nodes]
        local = np.searchsorted(self.surf_nodes, edges)
        side = sides_from_loop(xy, local, self.theta0)
        self.side, self.beta, self.surface_gap = locate_surface_nodes(xy, self.theta0, side=side)
        # offset between the mesh node and the analytic point is carried along unchanged
        self._offset = xy - surface_point(self.side, self.beta, self.theta0)
        self.deformer = RBFDeformer(mesh.points, self.surf_nodes, radius)

    def surface_xy(self, theta) -> np.ndarray:
        return surface_point(self.side, self.beta, np.asarray(theta, dtype=float)) + self._offset

    def points(self, theta) -> np.ndarray:
        disp = self.surface_xy(theta) - self.mesh.points[self.surf_nodes]
        return self.mesh.points + self.deformer.volume_displacement(disp)

    def dX_dtheta(self, h=None) -> np.ndarray:
        """(N, 2, k) derivative of all node coordinates w.r.t. theta at theta0 (central differences on
        the analytic map, which is linear in the surface displacement, so h only needs to be small).
        Default h: 1e-6 for shape variables, 1e-5 for the angle of attack (last entry); for NACA this is
        exactly the V1 default (1e-6, 1e-6, 1e-6, 1e-5)."""
        k_dv = len(self.theta0)
        if h is None:
            h = (1e-6,) * (k_dv - 1) + (1e-5,)
        out = np.zeros((len(self.mesh.points), 2, k_dv))
        for k, hk in enumerate(h):
            e = np.zeros(k_dv); e[k] = hk
            ds = (self.surface_xy(self.theta0 + e) - self.surface_xy(self.theta0 - e)) / (2 * hk)
            out[:, :, k] = self.deformer.volume_displacement(ds)
        return out


def chain_rule(sens_xy: np.ndarray, dXdtheta: np.ndarray) -> np.ndarray:
    """dJ/dtheta from volume sensitivities dJ/dX (N, 2) and dX/dtheta (N, 2, k)."""
    return np.einsum("ni,nik->k", sens_xy, dXdtheta)


def min_cell_jacobian_ratio(mesh: SU2Mesh, new_points: np.ndarray) -> float:
    """Smallest ratio (deformed / original) of signed triangle areas over all 2D cells; > 0 means no
    cell inverted. Quads are checked as their two triangles."""
    ratios = []
    i = 0
    lines = mesh.lines
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("NELEM="):
            nel = int(s.split("=")[1])
            for k in range(nel):
                v = [int(x) for x in lines[i + 1 + k].split()]
                typ, nodes = v[0], v[1:]
                tris = [nodes[:3]] if typ == 5 else ([[nodes[0], nodes[1], nodes[2]], [nodes[0], nodes[2], nodes[3]]] if typ == 9 else [])
                for a, b, c in tris:
                    def area(P):
                        return 0.5 * ((P[b, 0] - P[a, 0]) * (P[c, 1] - P[a, 1]) - (P[c, 0] - P[a, 0]) * (P[b, 1] - P[a, 1]))
                    a0 = area(mesh.points)
                    ratios.append(area(new_points) / a0 if a0 != 0 else 1.0)
            break
        i += 1
    return float(min(ratios)) if ratios else float("nan")
