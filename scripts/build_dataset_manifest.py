"""Build the Dataset V1 manifest and summary statistics from the stored records (no solver runs).

Reads datasets/designs/dataset_v1_design.json, datasets/raw/dataset_v1/<case>/case.json (+ gradients.json) and
datasets/processed/dataset_v1/<split>/<case>.npz. Writes datasets/dataset_v1_manifest.json (one entry per
designed case, including cases that are missing or did not converge) and prints a markdown summary.

Usage: python scripts/build_dataset_manifest.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AF_ROOT", "/workspace/AeroForge-TRIAD"))
RAW = ROOT / "datasets" / "raw" / "dataset_v1"
PROC = ROOT / "datasets" / "processed" / "dataset_v1"
DESIGN = ROOT / "datasets" / "designs" / "dataset_v1_design.json"
OUT = ROOT / "datasets" / "dataset_v1_manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    design = json.loads(DESIGN.read_text())
    entries, summary = [], {}
    for split, cases in design["splits"].items():
        st = {"designed": len(cases), "status": {}, "npz": 0, "grad_ok": 0, "iters": [], "CL": [], "CD": [], "wall_s": []}
        for c in cases:
            cid = c["case_id"]
            e = {"case_id": cid, "split": split, "theta": [c["m"], c["p"], c["t"], c["alpha_deg"]]}
            cj = RAW / cid / "case.json"
            if not cj.exists():
                e["status"] = "missing"
            else:
                rec = json.loads(cj.read_text())
                a = rec.get("assessment", {})
                e.update({"status": rec.get("status"), "CL": a.get("CL"), "CD": a.get("CD"), "iterations": a.get("iterations"),
                          "wall_s": rec.get("run", {}).get("wall_s"), "mesh_cells": rec.get("mesh", {}).get("n_cells")})
                if e["status"] == "converged":
                    st["iters"].append(a.get("iterations")); st["CL"].append(a.get("CL")); st["CD"].append(a.get("CD"))
                    st["wall_s"].append(rec.get("run", {}).get("wall_s"))
            g = RAW / cid / "gradients.json"
            if g.exists():
                gr = json.loads(g.read_text())
                e["grad_ok"] = all("grad" in v for v in gr["objectives"].values())
                e["adjoint_iterations"] = {k: v.get("iterations") for k, v in gr["objectives"].items()}
                st["grad_ok"] += int(e["grad_ok"])
            npz = PROC / split / f"{cid}.npz"
            if npz.exists():
                e["npz"] = str(npz.relative_to(ROOT)); e["npz_sha256"] = sha256(npz); st["npz"] += 1
            st["status"][e["status"]] = st["status"].get(e["status"], 0) + 1
            entries.append(e)
        summary[split] = st
    OUT.write_text(json.dumps({"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "design_sha256": sha256(DESIGN), "box": design["box"], "Re": design["Re"],
                               "cases": entries}, indent=1))
    print(f"manifest: {OUT} ({len(entries)} cases)")
    print("| split | designed | converged | other status | npz | adjoint gradients | iterations (min/median/max) | CL range | CD range | wall s (median) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for split, st in summary.items():
        conv = st["status"].get("converged", 0)
        other = ", ".join(f"{k} {v}" for k, v in st["status"].items() if k != "converged") or "none"
        it = np.array([x for x in st["iters"] if x is not None]); cl = np.array(st["CL"], float); cd = np.array(st["CD"], float)
        ws = np.array([x for x in st["wall_s"] if x is not None], float)
        fmt = lambda arr, f: f"{arr.min():{f}} – {arr.max():{f}}" if arr.size else "–"
        print(f"| {split} | {st['designed']} | {conv} | {other} | {st['npz']} | {st['grad_ok'] if split != 'train' else '–'} | "
              f"{(f'{it.min()}/{int(np.median(it))}/{it.max()}' if it.size else '–')} | {fmt(cl, '.4f')} | {fmt(cd, '.4f')} | "
              f"{(f'{np.median(ws):.0f}' if ws.size else '–')} |")


if __name__ == "__main__":
    sys.exit(main())
