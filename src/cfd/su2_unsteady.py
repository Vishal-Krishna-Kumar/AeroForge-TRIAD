"""Time-accurate SU2 runs for the steady/unsteady regime map (AeroForge-TRIAD, ADR-001 item 4).

Impulsive start from uniform flow (no steady restart, which could mask an instability), dual time
stepping (2nd order), convective time unit T = c / U_inf = 1. The regime is classified from the
lift-coefficient history over the final window.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from cfd.su2_case import CONFIG_TEMPLATE, read_history

UNSTEADY_BLOCK = """\
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= {dt}
MAX_TIME= {t_end}
TIME_ITER= {n_steps}
INNER_ITER= {inner}
HISTORY_WRT_FREQ_TIME= 1
"""


def write_unsteady_config(case_dir: Path, reynolds: float, dt: float = 0.02, t_end: float = 40.0,
                          inner: int = 15, cfl: float = 50.0) -> Path:
    n_steps = int(round(t_end / dt))
    base = CONFIG_TEMPLATE.format(mu=1.0 / reynolds, iters=inner, cfl=cfl, res_min=-6.0)
    # Line-anchored edits only: a plain substring replace of "ITER= 10" once corrupted
    # "CONV_STARTITER= 10" (bug found in the Step 9 regime pilot, 2026-10-12).
    base = re.sub(r"(?m)^ITER= .*\n", "", base)  # ITER is replaced by INNER_ITER below
    base = re.sub(r"(?m)^OUTPUT_WRT_FREQ= .*$", "OUTPUT_WRT_FREQ= {}".format(n_steps), base)
    base = base.replace("HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)",
                        "HISTORY_OUTPUT= (TIME_ITER, CUR_TIME, RMS_RES, AERO_COEFF)")
    text = base + UNSTEADY_BLOCK.format(dt=dt, t_end=t_end, n_steps=n_steps, inner=inner)
    validate_config_text(text)
    cfg = case_dir / "config.cfg"
    cfg.write_text(text)
    return cfg


def validate_config_text(text: str) -> None:
    """Fail fast on malformed SU2 config text: every non-comment, non-blank line must be
    'KEY= value', and no key may appear twice."""
    seen = set()
    for ln, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if "=" not in s:
            raise ValueError(f"config line {ln} has no '=': {s!r}")
        key = s.split("=", 1)[0].strip()
        if key in seen:
            raise ValueError(f"duplicate config key {key!r} at line {ln}")
        seen.add(key)


def classify(case_dir: Path, window: float = 15.0, rel_tol: float = 1e-3, abs_tol: float = 1e-4,
             min_cycles: float = 3.0, sustain_ratio: float = 0.7) -> dict:
    """Classify the final window of CL(t) (v2, after the v1 false positive of 2026-10-12).

    Steps: remove a quadratic trend (start-up transients are smooth and slow), then look for an
    oscillation that is (a) large enough (rel_tol of |mean CL| and abs_tol absolute), (b) periodic
    with at least `min_cycles` cycles inside the window (dominant frequency >= min_cycles / window),
    and (c) sustained: the RMS of the second half of the window is >= sustain_ratio x the first half.
      * 'unsteady'     (a) and (b) and (c): self-sustained shedding; Strouhal (chord-based) reported
      * 'not_settled'  anything else that still varies by more than rel_tol: decaying or slow transient.
                       This is NOT counted as steady; such cases need a longer run.
      * 'steady'       variation below rel_tol (or below abs_tol in absolute CL)."""
    h = read_history(case_dir)
    keys = list(h.keys())
    tkey = next(k for k in keys if k in ("Cur_Time", "CUR_TIME", "Time"))
    t, cl = h[tkey], h[next(k for k in keys if k in ("CL", "LIFT"))]
    cd = h[next(k for k in keys if k in ("CD", "DRAG"))]
    mask = t >= (t[-1] - window)
    tw, clw, cdw = t[mask], cl[mask], cd[mask]
    scale = max(abs(float(np.mean(clw))), 1e-2)
    total_rel = float(np.ptp(clw)) / scale
    coef = np.polyfit(tw - tw[0], clw, 2) if len(tw) > 3 else np.array([0.0, 0.0, float(np.mean(clw))])
    resid = clw - np.polyval(coef, tw - tw[0])
    osc_abs = float(np.ptp(resid))
    osc_rel = osc_abs / scale
    out = {"t_end": float(t[-1]), "window": window, "CL_mean": float(np.mean(clw)),
           "CD_mean": float(np.mean(cdw)), "CL_total_rel_p2p": total_rel, "CL_osc_rel_p2p": osc_rel,
           "CL_osc_abs_p2p": osc_abs, "classifier": "v2"}
    periodic = sustained = False
    if len(tw) > 16:
        x = resid * np.hanning(len(resid))
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=float(np.mean(np.diff(tw))))
        f_dom = float(freqs[1:][np.argmax(spec[1:])])
        out["dominant_freq_chord"] = f_dom
        periodic = f_dom >= min_cycles / window
        half = len(resid) // 2
        r1, r2 = float(np.std(resid[:half])), float(np.std(resid[half:]))
        out["rms_ratio_second_over_first_half"] = r2 / max(r1, 1e-15)
        sustained = r2 >= sustain_ratio * r1
    if osc_rel >= rel_tol and osc_abs >= abs_tol and periodic and sustained:
        out["regime"] = "unsteady"
        out["strouhal_chord"] = out["dominant_freq_chord"]
    elif total_rel >= rel_tol and float(np.ptp(clw)) >= abs_tol:
        out["regime"] = "not_settled"
    else:
        out["regime"] = "steady"
    return out
