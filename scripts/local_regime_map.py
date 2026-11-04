"""Local (Claude workspace container) regime-map runs at Re 1000 on P-medium, time-accurate.
Records go to localRM/out/<case>/case.json (same pipeline as Colab); summary lines 'RM ...' to stdout."""
import json, platform, sys, time
sys.path.insert(0, "src")
from pathlib import Path
from cfd.pipeline import ensure_su2, run_case
from meshing.gmsh_airfoil import MeshSpec
P = dict(h_wall=5e-4, bl_ratio=1.08, bl_thickness=0.06, h_surface=0.004, h_edges=0.0015, h_wake=0.015, wake_length=4.0,
         h_wake_near=0.008, near_wake_length=1.0, growth_rate=0.05, h_far=3.0, farfield_radius=100.0, label="P_medium")
UNS = dict(dt=0.05, t_end=30.0, inner=8)
z = Path("su2dl/su2omp.zip"); su2 = ensure_su2(z, Path("localsu2"))
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "medium"   # "coarse" = P x 1.5 (dataset level), "medium" = P
FAC = {"coarse": 1.5, "medium": 1.0}[LEVEL]
# Round 1 (2026-10-12 16:58 UTC): m=0.06, p=0.4, t=0.12, a=6 is UNSTEADY (self-excited CL oscillation,
# f*c/U ~ 0.93, amplitude growing from t~10 and saturating). The a=8, m=0.06 corners were therefore
# dropped as uninformative and the map re-centred on a reduced box: m <= 0.04, a <= 4.
CASES = [(0.04, 0.6, 0.18, 4.0), (0.04, 0.2, 0.18, 4.0), (0.04, 0.6, 0.08, 4.0), (0.04, 0.2, 0.08, 4.0),
         (0.0, 0.4, 0.18, 4.0), (0.04, 0.6, 0.18, -2.0), (0.06, 0.6, 0.18, 2.0), (0.0, 0.4, 0.08, 6.0)]
if LEVEL == "coarse":
    # first: the known-unsteady medium-mesh case, to check that P-coarse reproduces the regime
    CASES = [(0.06, 0.4, 0.12, 6.0)] + CASES
    # noise-floor control (added ~18:55 UTC after m0.04 p0.6 t0.18 a4 showed a tiny late-time CL oscillation,
    # p2p ~2e-4 = 0.25 % of CL): a case expected to be steady, to measure the residual oscillation that the
    # time-accurate method itself produces with 8 inner iterations
    CASES = CASES[:2] + [(0.0, 0.4, 0.12, 2.0)] + CASES[2:]
for m, p, t, a in CASES:
    cid = f"RM_Re1000_m{m:.2f}_p{p:.1f}_t{t:.2f}_a{a:+.0f}" + ("_Pcoarse" if LEVEL == "coarse" else "")
    t0 = time.time()
    rec = run_case(cid, {"m": m, "p": p, "t": t}, 1000.0, a, MeshSpec(**P).scaled(FAC, f"P_{LEVEL}"), Path("localRM/out"), Path("localRM/work"),
                   su2, z, f"A_regime_map_local[{platform.node()}]", unsteady=UNS)
    rg = rec.get("regime", {})
    print("RM", cid, rec.get("status"), json.dumps({k: rg.get(k) for k in ("CL_mean", "CD_mean", "CL_osc_abs_p2p", "CL_total_rel_p2p", "dominant_freq_chord", "rms_ratio_second_over_first_half")}), round(time.time() - t0), flush=True)
print("RM_DONE", flush=True)
