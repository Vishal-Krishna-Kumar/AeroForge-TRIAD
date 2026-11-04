"""2D airfoil meshing with Gmsh (AeroForge-TRIAD).

Hybrid mesh: structured quad boundary layer on the airfoil (Gmsh BoundaryLayer field with a
fan at the sharp trailing edge), unstructured triangles elsewhere, circular far field.
Writes SU2 native format with markers named by Gmsh physical groups: 'airfoil', 'farfield'.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from geometry.naca import rotate


@dataclass
class MeshSpec:
    farfield_radius: float = 50.0      # chords, centred on the pivot
    h_wall: float = 1.0e-3             # first boundary-layer cell height (chords)
    bl_ratio: float = 1.12             # boundary-layer growth ratio
    bl_thickness: float = 0.04         # total quad-layer thickness (chords)
    h_surface: float = 0.006           # spacing along the airfoil surface (chords)
    h_edges: float = 0.002             # spacing at the leading and trailing edge
    h_wake: float = 0.02               # spacing in the near wake box
    wake_length: float = 3.0           # chords downstream of the TE that are refined
    h_far: float = 2.5                 # spacing at the far field
    label: str = "medium"
    # --- optional (v2) controls; the defaults reproduce the v1 meshes exactly
    growth_rate: float | None = None   # linear size growth per chord of wall distance (replaces the
                                       # R-dependent Threshold, so domain size no longer changes inner cells)
    h_wake_near: float | None = None   # finer spacing in a near-wake box just behind the TE
    near_wake_length: float = 1.0      # chords downstream covered by the near-wake box

    def scaled(self, factor: float, label: str) -> "MeshSpec":
        """Uniform refinement: all spacings multiplied by `factor` (factor < 1 -> finer)."""
        d = asdict(self)
        for k in ("h_wall", "h_surface", "h_edges", "h_wake", "h_far", "h_wake_near"):
            if d[k] is not None:
                d[k] *= factor
        if d["growth_rate"] is not None:
            d["growth_rate"] *= factor
        d["label"] = label
        return MeshSpec(**d)


def build_mesh(upper: np.ndarray, lower: np.ndarray, out_path: str, spec: MeshSpec,
               aoa_deg: float = 0.0, pivot=(0.25, 0.0), verbose: bool = False) -> dict:
    """Mesh the airfoil from unrotated upper/lower surfaces ordered LE -> TE.

    The geometry is rotated nose-up by aoa_deg about `pivot`; the free stream stays along +x.
    Returns a dict of mesh statistics (counts, quality, marker names found in the file)."""
    import gmsh

    up = rotate(upper, aoa_deg, pivot)
    lo = rotate(lower, aoa_deg, pivot)
    cx, cy = pivot
    R = spec.farfield_radius

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        # Reproducibility: single-threaded meshing gives run-to-run identical meshes
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.model.add("airfoil")
        geo = gmsh.model.geo

        # --- airfoil: shared LE (index 0) and TE (index -1) points, two splines LE -> TE
        p_up = [geo.addPoint(x, y, 0.0) for x, y in up]
        p_lo = [p_up[0]] + [geo.addPoint(x, y, 0.0) for x, y in lo[1:-1]] + [p_up[-1]]
        te = p_up[-1]
        c_up = geo.addSpline(p_up)
        c_lo = geo.addSpline(p_lo)

        # --- circular far field (4 arcs)
        pc = geo.addPoint(cx, cy, 0.0)
        pf = [geo.addPoint(cx + R * np.cos(a), cy + R * np.sin(a), 0.0)
              for a in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)]
        arcs = [geo.addCircleArc(pf[i], pc, pf[(i + 1) % 4]) for i in range(4)]

        loop_far = geo.addCurveLoop(arcs)
        loop_af = geo.addCurveLoop([c_up, -c_lo])
        surf = geo.addPlaneSurface([loop_far, loop_af])
        geo.synchronize()

        # --- size fields
        fld = gmsh.model.mesh.field
        f_dist = fld.add("Distance")
        fld.setNumbers(f_dist, "CurvesList", [c_up, c_lo])
        fld.setNumber(f_dist, "Sampling", 400)
        if spec.growth_rate is None:  # v1: linear ramp to h_far at 0.8 R (depends on R)
            f_thr = fld.add("Threshold")
            fld.setNumber(f_thr, "InField", f_dist)
            fld.setNumber(f_thr, "SizeMin", spec.h_surface)
            fld.setNumber(f_thr, "SizeMax", spec.h_far)
            fld.setNumber(f_thr, "DistMin", 0.0)
            fld.setNumber(f_thr, "DistMax", 0.8 * R)
        else:  # v2: size = h_surface + growth_rate * d, capped at h_far (independent of R)
            f_thr = fld.add("MathEval")
            fld.setString(f_thr, "F", f"min({spec.h_surface} + {spec.growth_rate}*F{f_dist}, {spec.h_far})")

        # LE/TE refinement with Ball fields (a point-based Distance field stalls in Gmsh 4.15.2)
        f_balls = []
        for xy in (up[0], up[-1]):
            fb = fld.add("Ball")
            fld.setNumber(fb, "XCenter", float(xy[0]))
            fld.setNumber(fb, "YCenter", float(xy[1]))
            fld.setNumber(fb, "ZCenter", 0.0)
            fld.setNumber(fb, "Radius", 0.02)
            fld.setNumber(fb, "Thickness", 0.06)
            fld.setNumber(fb, "VIn", spec.h_edges)
            fld.setNumber(fb, "VOut", spec.h_far)
            f_balls.append(fb)

        te_xy = up[-1]
        f_box = fld.add("Box")
        fld.setNumber(f_box, "VIn", spec.h_wake)
        fld.setNumber(f_box, "VOut", spec.h_far)
        fld.setNumber(f_box, "XMin", te_xy[0] - 0.05)
        fld.setNumber(f_box, "XMax", te_xy[0] + spec.wake_length)
        fld.setNumber(f_box, "YMin", te_xy[1] - 0.35)
        fld.setNumber(f_box, "YMax", te_xy[1] + 0.35)
        fld.setNumber(f_box, "Thickness", 1.0)

        extra = []
        if spec.h_wake_near is not None:
            f_nw = fld.add("Box")
            fld.setNumber(f_nw, "VIn", spec.h_wake_near)
            fld.setNumber(f_nw, "VOut", spec.h_far)
            fld.setNumber(f_nw, "XMin", te_xy[0] - 0.02)
            fld.setNumber(f_nw, "XMax", te_xy[0] + spec.near_wake_length)
            fld.setNumber(f_nw, "YMin", te_xy[1] - 0.2)
            fld.setNumber(f_nw, "YMax", te_xy[1] + 0.2)
            fld.setNumber(f_nw, "Thickness", 0.3)
            extra.append(f_nw)

        f_min = fld.add("Min")
        fld.setNumbers(f_min, "FieldsList", [f_thr, f_box] + f_balls + extra)
        fld.setAsBackgroundMesh(f_min)

        f_bl = fld.add("BoundaryLayer")
        fld.setNumbers(f_bl, "CurvesList", [c_up, c_lo])
        fld.setNumbers(f_bl, "FanPointsList", [te])
        fld.setNumber(f_bl, "Size", spec.h_wall)
        fld.setNumber(f_bl, "Ratio", spec.bl_ratio)
        fld.setNumber(f_bl, "Thickness", spec.bl_thickness)
        fld.setNumber(f_bl, "Quads", 1)
        fld.setAsBoundaryLayer(f_bl)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        # --- physical groups -> SU2 marker names
        gmsh.model.addPhysicalGroup(1, [c_up, c_lo], name="airfoil")
        gmsh.model.addPhysicalGroup(1, arcs, name="farfield")
        gmsh.model.addPhysicalGroup(2, [surf], name="fluid")

        gmsh.model.mesh.generate(2)

        # --- statistics and quality
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        etypes, etags, _ = gmsh.model.mesh.getElements(2)
        counts, qmins = {}, []
        for et, tags in zip(etypes, etags):
            name = gmsh.model.mesh.getElementProperties(et)[0]
            counts[name] = len(tags)
            q = gmsh.model.mesh.getElementQualities(tags, "minSICN")
            qmins.append(float(np.min(q)))
        gmsh.write(out_path)
    finally:
        gmsh.finalize()

    markers = []
    with open(out_path) as fh:
        for line in fh:
            if line.startswith("MARKER_TAG"):
                markers.append(line.split("=")[1].strip())
    return {
        "spec": asdict(spec),
        "aoa_deg": aoa_deg,
        "n_nodes": int(len(node_tags)),
        "n_cells_2d": int(sum(counts.values())),
        "cell_types": counts,
        "min_SICN": float(min(qmins)),
        "markers_in_file": markers,
    }


def build_mesh_isolated(upper: np.ndarray, lower: np.ndarray, out_path: str, spec: MeshSpec,
                        aoa_deg: float = 0.0, pivot=(0.25, 0.0), timeout_s: float = 1800.0) -> dict:
    """build_mesh in a fresh Python interpreter (subprocess running this module as a script).

    Gmsh keeps hidden state across initialize/finalize in one process: a second mesh of the same
    geometry in the same process came out with 49,190 instead of 49,288 cells (2026-10-12), which
    changed CL by 5e-4. A fresh interpreter gives the same mesh every time on a given platform.
    (A multiprocessing 'spawn' child was tried first; it re-imports the caller's __main__ and fails
    for scripts without a main guard, so a plain subprocess is used.)"""
    import json
    import pickle
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / "in.pkl", Path(td) / "out.json"
        inp.write_bytes(pickle.dumps((np.asarray(upper), np.asarray(lower), str(out_path), asdict(spec), float(aoa_deg), tuple(pivot))))
        src = str(Path(__file__).resolve().parents[1])
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                            "from meshing.gmsh_airfoil import _worker_main; _worker_main(sys.argv[2], sys.argv[3])",
                            src, str(inp), str(outp)], capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0 or not outp.exists():
            raise RuntimeError(f"meshing subprocess failed (rc {r.returncode}): {r.stderr[-2000:]}")
        return json.loads(outp.read_text())


def _worker_main(inp: str, outp: str) -> None:
    import json
    import pickle
    from pathlib import Path

    upper, lower, out_path, spec_d, aoa, pivot = pickle.loads(Path(inp).read_bytes())
    stats = build_mesh(upper, lower, out_path, MeshSpec(**spec_d), aoa_deg=aoa, pivot=pivot)
    Path(outp).write_text(json.dumps(stats))
