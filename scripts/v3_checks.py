"""Protocol v3 phase-0 gate checks (run on ONE pod before any further V3 compute; stop rules in protocol v3 §6).

Check designs (fixed by rule, not by result): from a 4096-point Sobol pool of the V3 box (seed 7), K=0 the accepted
point closest to the box centre with alpha set to 2.5 deg (moderate shock), K=1 the accepted point closest to the
box centre with alpha set to 0.5 deg (weak shock).
  primal K        steady Euler primal of check design K (cfd.euler_case.run_case_euler)
  adjoint K       discrete adjoints of CL, CD and of the weighted objective J = CL/(CD+CD0) on design K
  fd K I          central FD (two deformed-mesh primals, restarted from the base solution) for design variable I
  report          pass/fail:
                  (a) weighted J gradient vs (1/(CD+CD0)) dCL - CL/(CD+CD0)^2 dCD from the two single adjoints,
                      relative norm difference <= 1e-5;
                  (b) adjoint vs FD of dJ/dtheta for the FD variables, relative difference <= 1 % for every
                      component larger than 1 % of the largest |component| (same rule as V2); the FD reference is
                      the Richardson value (4 D(h/2) - D(h)) / 3 of central differences at h and h/2 (shocks make
                      the forces visibly nonlinear in theta; both single-step values are reported);
                  plus timing of every solve (for the cost projection, protocol v3 §7).
FD variables (fixed): au0, au3, al2, aoa_deg -> indices 0, 3, 8, 12; steps 1e-3 (CST) and 0.025 deg (alpha; protocol v3 amendment 3, was 0.1 deg).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", "/workspace/AeroForge-TRIAD"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import run_dataset_v3 as d3  # noqa: E402
from cfd.euler_case import run_case_euler  # noqa: E402
from cfd.pipeline import ensure_su2  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402

OUT = ROOT / "datasets" / "raw" / "v3_checks"
REP = ROOT / "results" / "v3_checks"
WORK = Path(os.environ.get("AF_WORK", "/workspace/runs")) / "v3_checks"
FD_IDX = (0, 3, 8, 12)
ALPHAS = (2.5, 0.5)


def su2() -> Path:
    if os.environ.get("AF_SU2_EXE"):
        return Path(os.environ["AF_SU2_EXE"])
    return ensure_su2(d3.SU2_ZIP, d3.SU2_DIR)


def check_design(k: int) -> np.ndarray:
    from scipy.stats import qmc
    lo, hi = d3.box()
    pts = lo + qmc.Sobol(2 * d3.NC + 1, scramble=True, seed=7).random(4096) * (hi - lo)
    ok = [p for p in pts if d3.accepted(p)[0]]
    c = 0.5 * (lo + hi)
    best = min(ok, key=lambda p: float(np.sum(((p - c) / (hi - lo))[:-1] ** 2)))
    th = best.copy()
    th[-1] = ALPHAS[k]
    return th


def cid(k: int) -> str:
    return f"V3CHK_{k}"


def main() -> None:
    from cfd.gradients import adjoint_gradients, fd_gradients
    from cfd.su2_euler import ADJ_CONV_FIELD, RES_KEY
    a = sys.argv[1:]
    for d in (OUT, REP, WORK):
        d.mkdir(parents=True, exist_ok=True)
    thr = int(os.environ.get("AF_SU2_THREADS", "1"))
    if a[0] == "primal":
        k = int(a[1]); th = check_design(k)
        rec = run_case_euler(cid(k), th[:d3.NC], th[d3.NC:2 * d3.NC], float(th[-1]), d3.MACH, d3.mesh_spec(), OUT, WORK,
                             su2(), d3.SU2_ZIP, "V3_checks")
        print("V3CHK_PRIMAL", k, rec.get("status"), json.dumps(rec.get("assessment")), rec.get("run"), flush=True)
    elif a[0] == "adjoint":
        k = int(a[1]); th = check_design(k)
        rec = json.loads((OUT / cid(k) / "case.json").read_text())
        w = d3.j_weights(rec["assessment"]["CL"], rec["assessment"]["CD"])
        g = adjoint_gradients(OUT / cid(k), WORK / f"{cid(k)}_adj", th, d3.SU2_AD, threads=thr, radius=0.5,
                              objectives=("CL", "CD"), combined={"J": w}, env=d3.AD_ENV, adj_conv_field=ADJ_CONV_FIELD,
                              adj_res_min=-10.0, adj_iters=d3.ADJ_ITERS, adj_extra=d3.ADJ_EXTRA)
        g["j_weights"] = list(w)
        save_json(REP / f"adjoint_{k}.json", g)
        print("V3CHK_ADJOINT", k, {n: (o.get("run"), o.get("iterations"), o.get("final_rms_adj_p")) for n, o in g["objectives"].items()}, flush=True)
    elif a[0] == "fd":
        k, i = int(a[1]), int(a[2]); th = check_design(k)
        names = None
        from cfd.shape_sensitivity import dv_names
        names = dv_names(th)
        base_steps = (1e-3,) * (len(names) - 1) + (0.025,)  # amendment 3: alpha 0.025/0.0125 deg (was 0.1/0.05)
        runs = {}
        for tag, fac in (("h", 1.0), ("h2", 0.5)):   # central FD at h and h/2; Richardson (4 D(h/2) - D(h)) / 3
            runs[tag] = fd_gradients(OUT / cid(k), WORK / f"{cid(k)}_fd{i}_{tag}", th, str(su2()),
                                     steps=tuple(s * fac for s in base_steps), radius=0.5, iters=1500, threads=thr,
                                     which=[names[i]], conv_field="RMS_DENSITY", res_key=RES_KEY)
        n = names[i]
        g = {"h": runs["h"], "h2": runs["h2"],
             "grad": {q: {n: (4 * runs["h2"]["grad"][q][n] - runs["h"]["grad"][q][n]) / 3} for q in ("CL", "CD")}}
        save_json(REP / f"fd_{k}_{i}.json", g)
        print("V3CHK_FD", k, i, json.dumps(g["grad"]), json.dumps({t: r["grad"] for t, r in runs.items()}), flush=True)
    elif a[0] == "report":
        rep = {"designs": {}, "pass": True}
        for k in (0, 1):
            th = check_design(k)
            rec = json.loads((OUT / cid(k) / "case.json").read_text()) if (OUT / cid(k) / "case.json").exists() else {}
            if rec.get("status") != "converged" or not (REP / f"adjoint_{k}.json").exists():
                rep["designs"][k] = {"status": rec.get("status"), "adjoint_file": (REP / f"adjoint_{k}.json").exists(), "pass": False}
                rep["pass"] = False
                continue
            cl, cd = rec["assessment"]["CL"], rec["assessment"]["CD"]
            den = cd + d3.CD0
            adj = json.loads((REP / f"adjoint_{k}.json").read_text())
            if not all("grad" in adj["objectives"].get(n, {}) for n in ("CL", "CD", "J")):
                rep["designs"][k] = {"status": rec["status"], "adjoint_objectives": {n: o.get("run") for n, o in adj["objectives"].items()},
                                     "pass": False}
                rep["pass"] = False
                continue
            names = adj["dv"]
            gcl = np.array([adj["objectives"]["CL"]["grad"][n] for n in names])
            gcd = np.array([adj["objectives"]["CD"]["grad"][n] for n in names])
            gj = np.array([adj["objectives"]["J"]["grad"][n] for n in names])
            comb = gcl / den - cl / den ** 2 * gcd
            a_ok = float(np.linalg.norm(gj - comb) / np.linalg.norm(comb))
            big = np.abs(gj).max()
            fd_rows = {}
            for i in FD_IDX:
                f = REP / f"fd_{k}_{i}.json"
                if not f.exists():
                    fd_rows[names[i]] = {"missing": True}
                    continue
                fd = json.loads(f.read_text())
                dcl, dcd = fd["grad"]["CL"][names[i]], fd["grad"]["CD"][names[i]]
                dj = dcl / den - cl / den ** 2 * dcd
                rel = abs(gj[i] - dj) / max(abs(dj), 1e-30)
                checked = abs(gj[i]) > 0.01 * big
                single = {}
                for t in ("h", "h2"):
                    sc, sd = fd[t]["grad"]["CL"][names[i]], fd[t]["grad"]["CD"][names[i]]
                    single[t] = float(sc / den - cl / den ** 2 * sd)
                fd_rows[names[i]] = {"adjoint": float(gj[i]), "fd_richardson": float(dj), "fd_h": single["h"],
                                     "fd_h2": single["h2"], "rel_diff": float(rel), "checked": bool(checked),
                                     "ok": bool((not checked) or rel <= 0.01)}
            ok = a_ok <= 1e-5 and all(r.get("ok", False) for r in fd_rows.values())
            rep["designs"][k] = {"theta": th.tolist(), "CL": cl, "CD": cd, "status": rec["status"],
                                 "primal_wall_s": rec["run"]["wall_s"], "iterations": rec["assessment"]["iterations"],
                                 "adjoint_runs": {n: o.get("run") for n, o in adj["objectives"].items()},
                                 "adjoint_iterations": {n: o.get("iterations") for n, o in adj["objectives"].items()},
                                 "weighted_vs_combined_rel": a_ok, "fd": fd_rows, "pass": bool(ok)}
            rep["pass"] = rep["pass"] and ok
        save_json(REP / "v3_checks_report.json", rep)
        print("V3CHK_REPORT", "PASS" if rep["pass"] else "FAIL", json.dumps(rep)[:3000], flush=True)
    else:
        raise SystemExit(f"unknown command {a}")


if __name__ == "__main__":
    main()
