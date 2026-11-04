"""Protocol v3 amendment 4 (iii): descriptive mesh-refinement check of the CFD truth.

The first 10 test designs (D3_test_0000-0009) at two finer meshes: every E100 cell size (h_wall, h_surface, h_edges, h_wake)
x 1/sqrt(2) ("E141") and x 1/2 ("E200") via MeshSpec.scaled (also h_far and growth rate; amendment 4a).
Same primal (scripts/run_dataset_v3.run_case_euler path) and J adjoint (run_dataset_v3.j_gradient, amendment-3 settings).
Raw records: datasets/raw/v3_checks/meshref/<case_id>_<label>/ (case.json, gradients.json, history.csv, config.cfg ...).
Never used for calibration or testing. Jobs are numbered 0-19: 0-9 = E200 (longest first), 10-19 = E141.
Usage (pod): python scripts/run_v3_meshref.py --start i --stop j
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "/workspace/AeroForge-TRIAD"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

import run_dataset_v3 as d3  # noqa: E402

RAWM = ROOT / "datasets" / "raw" / "v3_checks" / "meshref"
LEVELS = [("E200", 0.5), ("E141", 2 ** -0.5)]
SCALED = ("h_wall", "h_surface", "h_edges", "h_wake", "h_far", "growth_rate")


def jobs():
    test = d3.load_design()["splits"]["test"][:10]
    return [(c["case_id"], np.asarray(c["theta"], float), lab, f) for lab, f in LEVELS for c in test]


def spec_for(label, f):
    return d3.mesh_spec().scaled(f, label)  # amendment 4a: uniform refinement incl. h_far and growth rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()
    su2 = d3.ensure_su2(d3.SU2_ZIP, d3.SU2_DIR) if not os.environ.get("AF_SU2_EXE") else Path(os.environ["AF_SU2_EXE"])
    RAWM.mkdir(parents=True, exist_ok=True)
    for cid0, th, lab, f in jobs()[args.start:args.stop]:
        cid, t0 = f"{cid0}_{lab}", time.time()
        spec = spec_for(lab, f)
        rec = d3.existing(RAWM / cid)
        if rec is None:
            rec = d3.run_case_euler(cid, th[:d3.NC], th[d3.NC:2 * d3.NC], float(th[-1]), d3.MACH, spec, RAWM, d3.WORK, su2,
                                    d3.SU2_ZIP, "V3_meshref")
        info = {"status": rec.get("status"), "CL": rec.get("assessment", {}).get("CL"), "CD": rec.get("assessment", {}).get("CD"),
                "spec": {k: getattr(spec, k) for k in SCALED}}
        gfile = RAWM / cid / "gradients.json"
        if rec.get("status") == "converged" and not gfile.exists():
            g = d3.j_gradient(RAWM / cid, d3.WORK / f"{cid}_adj", th, rec)
            d3.save_json(gfile, g)
            info["adj_iterations"] = g["objectives"]["J"].get("iterations")
        info["wall_s"] = round(time.time() - t0, 1)
        print("MESHREF", cid, json.dumps(info), flush=True)
    print("AF_MESHREF_BATCH_DONE", args.start, args.stop, flush=True)


if __name__ == "__main__":
    main()
