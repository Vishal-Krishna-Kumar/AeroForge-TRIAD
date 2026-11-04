"""Protocol v2 gate checks, run on the pod BEFORE any Dataset V2 case (stop rules in protocol v2 §6).

  regime I          I = 0..3: time-accurate run (V1 regime-map settings: dt 0.05, t_end 30, 8 inner) of the
                    most extreme accepted shapes at alpha 4 deg, chosen by rule from a 4096-point Sobol pool
                    (seed 7): 0 max t*camber, 1 max t, 2 max camber, 3 max waviness (upper-surface curvature
                    sign changes). Pass: status 'steady' for all four.
  grad-primal       steady primal of the check case (pool point closest to the box centre, alpha 4 deg).
  grad-adjoint      adjoints of CL, CD and the weighted L/D objective on that case.
  grad-fd K         central FD (two deformed-mesh primals) for design variable index K.
  grad-report       pass/fail: (a) weighted L/D gradient vs (1/CD) dCL - (CL/CD^2) dCD from the two
                    single adjoints, relative norm difference <= 1e-5; (b) adjoint vs FD for the variables run,
                    relative difference <= 1 % for every component larger than 1 % of the largest one.
FD variables (fixed): au0, au3, al2, aoa_deg -> indices 0, 3, 8, 12.
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

import run_dataset_v2 as d2  # noqa: E402
from cfd.pipeline import ensure_su2, existing, run_case  # noqa: E402
from cfd.su2_case import save_json  # noqa: E402
from meshing.gmsh_airfoil import MeshSpec  # noqa: E402

OUT = ROOT / "datasets" / "raw" / "v2_checks"
REP = ROOT / "results" / "v2_checks"
WORK = Path(os.environ.get("AF_WORK", "/workspace/runs")) / "v2_checks"
UNS = dict(dt=0.05, t_end=30.0, inner=8)
FD_IDX = (0, 3, 8, 12)


def su2():
    if os.environ.get("AF_SU2_EXE"):
        return Path(os.environ["AF_SU2_EXE"])
    return ensure_su2(d2.SU2_ZIP, d2.SU2_DIR)


def pool() -> list[tuple[np.ndarray, dict]]:
    from scipy.stats import qmc
    lo, hi = d2.box()
    u = qmc.Sobol(2 * d2.NC + 1, scramble=True, seed=7).random(4096)
    out = []
    for th in lo + u * (hi - lo):
        ok, m = d2.accepted(th)
        if ok:
            out.append((th, m))
    return out


def waviness(th) -> int:
    from geometry.cst import surface_y
    x = 0.5 * (1 - np.cos(np.linspace(0.05, np.pi - 0.05, 400)))
    y = surface_y(x, th[:d2.NC])
    return int(np.sum(np.diff(np.sign(np.gradient(np.gradient(y, x), x))) != 0))


def extremes() -> list[tuple[str, np.ndarray]]:
    p = pool()
    keys = [("max_t_camber", lambda a: a[1]["t_max"] * a[1]["camber_max"]), ("max_t", lambda a: a[1]["t_max"]),
            ("max_camber", lambda a: a[1]["camber_max"]), ("max_waviness", lambda a: waviness(a[0]))]
    out = []
    for name, f in keys:
        th = max(p, key=f)[0].copy()
        th[-1] = 4.0
        out.append((name, th))
    return out


def check_case() -> np.ndarray:
    lo, hi = d2.box()
    c = 0.5 * (lo + hi)
    p = pool()
    th = min(p, key=lambda a: np.linalg.norm((a[0] - c) / (hi - lo)))[0].copy()
    th[-1] = 4.0
    return th


def geom(th):
    return {"family": "CST", "au": th[:d2.NC].tolist(), "al": th[d2.NC:2 * d2.NC].tolist()}


def spec():
    return MeshSpec(**d2.P_SPEC).scaled(1.5, "P_coarse")


def main():
    cmd = sys.argv[1]
    REP.mkdir(parents=True, exist_ok=True)   # launch 1 failed: save_json needs the directory to exist
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "regime":
        i = int(sys.argv[2])
        name, th = extremes()[i]
        rec = run_case(f"V2gate_regime_{i}_{name}", geom(th), d2.RE, 4.0, spec(), OUT, WORK / f"rg{i}", su2(), d2.SU2_ZIP,
                       "S2_regime_gate", unsteady=UNS)
        rg = rec.get("regime", {})
        print("GATE_REGIME", i, name, rec.get("status"), json.dumps({k: rg.get(k) for k in (
            "CL_mean", "CL_osc_abs_p2p", "rms_ratio_second_over_first_half", "dominant_freq_chord")}), flush=True)
    elif cmd == "grad-primal":
        th = check_case()
        rec = run_case("V2gate_grad_base", geom(th), d2.RE, 4.0, spec(), OUT, WORK / "gbase", su2(), d2.SU2_ZIP,
                       "S2_grad_gate", conv="cauchy", keep=d2.KEEP)
        save_json(REP / "grad_check_theta.json", {"theta": th.tolist()})
        print("GATE_GRAD_PRIMAL", rec.get("status"), rec.get("assessment", {}).get("iterations"), flush=True)
    elif cmd == "grad-adjoint":
        from cfd.gradients import adjoint_gradients
        th = np.array(json.loads((REP / "grad_check_theta.json").read_text())["theta"])
        rec = existing(OUT / "V2gate_grad_base")
        w = d2.ld_weights(rec["assessment"]["CL"], rec["assessment"]["CD"])
        g = adjoint_gradients(OUT / "V2gate_grad_base", WORK / "gadj", th, d2.SU2_AD, threads=1, radius=0.5,
                              objectives=("CL", "CD"), combined={"LD": w}, env=d2.AD_ENV)
        g["ld_weights"] = list(w)
        save_json(REP / "grad_check_adjoint.json", g)
        print("GATE_GRAD_ADJOINT", {k: "grad" in v for k, v in g["objectives"].items()}, flush=True)
    elif cmd == "grad-fd":
        from cfd.gradients import fd_gradients
        from cfd.shape_sensitivity import dv_names
        k = int(sys.argv[2])
        th = np.array(json.loads((REP / "grad_check_theta.json").read_text())["theta"])
        name = dv_names(th)[k]
        r = fd_gradients(OUT / "V2gate_grad_base", WORK / f"gfd_{name}", th, str(su2()), radius=0.5, iters=1200,
                         threads=1, which=(name,))
        save_json(REP / f"grad_check_fd_{name}.json", r)
        print("GATE_GRAD_FD", name, r["grad"], flush=True)
    elif cmd == "grad-report":
        from cfd.shape_sensitivity import dv_names
        g = json.loads((REP / "grad_check_adjoint.json").read_text())
        names = g["dv"]
        G = {q: np.array([g["objectives"][q]["grad"][n] for n in names]) for q in ("CL", "CD", "LD")}
        rec = existing(OUT / "V2gate_grad_base")["assessment"]
        cl, cd = rec["CL"], rec["CD"]
        combo = G["CL"] / cd - cl / cd ** 2 * G["CD"]
        rel_w = float(np.linalg.norm(G["LD"] - combo) / np.linalg.norm(combo))
        fd_rows, ok_fd = [], True
        for k in FD_IDX:
            n = dv_names(np.zeros(len(names)))[k]
            f = REP / f"grad_check_fd_{n}.json"
            if not f.exists():
                ok_fd = False
                fd_rows.append({"dv": n, "missing": True})
                continue
            fd = json.loads(f.read_text())["grad"]
            for q in ("CL", "CD"):
                big = np.abs(G[q]).max()
                a, b = G[q][k], fd[q][n]
                rel = abs(a - b) / max(abs(b), 1e-30)
                need = abs(a) > 0.01 * big
                ok_fd &= (rel <= 0.01) or not need
                fd_rows.append({"dv": n, "q": q, "adjoint": a, "fd": b, "rel_diff": rel, "tested": bool(need)})
        rep = {"weighted_vs_two_adjoints_rel": rel_w, "weighted_pass": rel_w <= 1e-5, "fd": fd_rows,
               "fd_pass": bool(ok_fd)}
        rep["PASS"] = rep["weighted_pass"] and rep["fd_pass"]
        save_json(REP / "grad_check_report.json", rep)
        print("GATE_GRAD_REPORT", json.dumps(rep), flush=True)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
