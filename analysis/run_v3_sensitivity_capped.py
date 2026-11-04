"""DESCRIPTIVE sensitivity analysis (not pre-registered): scripts/run_v3_analysis.py re-run with every J adjoint that hit the
amendment-3 iteration cap (8000) removed. Protocol v3 §2 keeps such cases in all pre-registered analyses ("kept and flagged,
never dropped"); this run only shows whether the conclusions depend on them.
Capped = gradients.json objectives.J.iterations >= 8000, read from datasets/raw/dataset_v3 (cal/test; 17 cases on
2026-10-19) and datasets/raw/dataset_v3_traj (trajectory points). Capped trajectory points are treated like points without
ground truth (excluded and counted), so the run-level design check reports any run they make incomplete.
Output: results/v3_analysis_sensitivity_capped/v3_analysis.json + capped_cases.json (refuses to overwrite).
Usage (Colab, after phase 3 is installed): python scripts/run_v3_sensitivity_capped.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "."))
OUT = ROOT / "results" / "v3_analysis_sensitivity_capped"
if (OUT / "v3_analysis.json").exists():
    raise SystemExit(f"{OUT} exists; refusing to overwrite")
os.environ["AF_OUT"] = str(OUT)
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import run_v3_analysis as A  # noqa: E402


def capped(raw: Path) -> set:
    s = set()
    if raw.is_dir():
        for d in raw.iterdir():
            g = d / "gradients.json"
            if g.exists() and json.loads(g.read_text())["objectives"]["J"].get("iterations", 0) >= 8000:
                s.add(d.name)
    return s


CAP = capped(ROOT / "datasets" / "raw" / "dataset_v3") | capped(A.RAW)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "capped_cases.json").write_text(json.dumps(sorted(CAP), indent=1))
print("CAPPED", len(CAP), flush=True)
_load = A.load


def load_without_capped(folder):
    d = _load(folder)
    hit = np.isin(d["case_id"], sorted(CAP))
    d = dict(d); d["g_J"] = d["g_J"].copy(); d["g_J"][hit] = np.nan  # no ground truth -> excluded and counted, like a missing adjoint
    print("EXCLUDED", folder.name, int(hit.sum()), flush=True)
    return d


A.load = load_without_capped
_raw = A.add_raw_gradient_points


def raw_without_capped(tr):
    tr2, fb = _raw(tr)
    hit = np.isin(tr2["case_id"], sorted(CAP))
    tr2 = dict(tr2); tr2["g_J"] = tr2["g_J"].copy(); tr2["g_J"][hit] = np.nan
    return tr2, fb


A.add_raw_gradient_points = raw_without_capped
A.main()
