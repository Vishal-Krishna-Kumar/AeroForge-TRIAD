# Experiment protocol v3 — second flow regime: steady transonic inviscid flow (AeroForge-TRIAD)

Status: FROZEN 2026-10-15 (UTC), before any V3 cloud compute and before any V3 surrogate, trajectory or gradient
exists. Written by Claude for Mohammad's review; Mohammad approved the budget (hard cap $12.00) on 2026-10-15
01:12Z. Changes after freezing are allowed only as dated amendments in §10, each written before the data it affects.
Local pilot (exploratory, container CPU, no cloud): docs/plan_v3_transonic_DRAFT.md and the experiment log.

## 1. Purpose
V1/V2 (laminar, Re 1000) showed that run-level calibration keeps whole-run coverage along optimiser trajectories
(H5 supported) where i.i.d. calibration fails (H7 replicated), but surrogate gradient directions were almost always
right (299/300 V2 test-run steps were true ascent), so no certificate could show decision value. V3 repeats the
run-level study in a regime with shock waves, where gradient quality is expected — not assumed — to be worse.

## 2. Physics, numerics, convergence (code: src/cfd/su2_euler.py, src/cfd/euler_case.py)
SU2 8.5.0, steady 2-D compressible Euler; JST (sensor 0.5, 0.02), implicit Euler, 3-level W-cycle multigrid, CFL 10;
free stream M = 0.70, p 101325 Pa, T 288.15 K; angle of attack applied by rotating the geometry. Mesh "E100" (Gmsh
pipeline: first cell 0.003 c, ratio 1.15, quad layer 0.01 c, surface 0.006 c, edges 0.002 c, wake box 0.04 c over 1 c,
growth 0.08, far field 4 c spacing at radius 100 c). Converged = finite, rms[ρ] ≤ 1e-10 within 3000 iterations,
CL relative spread < 1e-4 and CD absolute spread < 1e-6 over the last 100 iterations; otherwise 'not_converged'
(kept and flagged, never dropped). Gradients: SU2_CFD_AD discrete adjoint of the weighted objective (§3), residual
rms[A_ρ] ≤ 1e-10 within 3000 iterations, chain rule through the same RBF mesh-deformation map as V2 (radius 0.5).
The CFD "truth" is this fixed discrete model; its discretization error (pilot: CD ≈ ±0.0015 on weak-shock designs,
CL insensitive to refinement) is a stated limitation.

## 3. Design space, objective, data (code: scripts/run_dataset_v3.py; design file sha256 below)
θ = 12 CST coefficients (6 per side) of the NACA 2411 fit ± 0.06, plus α ∈ [−1°, 3°]; acceptance as V2
(t_max ∈ [0.08, 0.14], camber ∈ [−0.01, 0.045], no self-intersection).
Objective J = CL / (CD + CD0), CD0 = 0.008 (fixed profile-drag allowance; inviscid CD can be ≈ 0). dJ/dθ from ONE
adjoint of wL·CL + wD·CD, wL = 1/(CD+CD0), wD = −CL/(CD+CD0)², weights from the converged primal.
Splits (scrambled Sobol, first accepted points): train 384 (seed 20261030; primal only), cal 128 (20261031),
test 64 (20261032) — cal/test with adjoint gradients. Frozen design file datasets/designs/dataset_v3_design.json,
sha256 dac6e75382c82080d7c2a6412b73d22bcfc28e555e215bf0147e362ba4f94d50 (generated 2026-10-15 before any V3 run).
Learning targets: npz in the V2 layout (fields p := Cp, u/U∞, v/U∞ on the V2 grid; surface Cp; cf ≡ 0; dJ_dtheta).

## 4. Surrogates (phase 2)
FNO and MLP ensembles of 5 seeds with exactly the V2 architectures and training settings (scripts/train_surrogates_v2.py
hyper-parameters; FNO 500 epochs, MLP 3000 epochs); fit set = first ⌊0.875 N_train⌋ train cases in case-id order,
validation = the rest. Value prediction targets CL and CD; J is computed from them.

## 5. Trajectories (phase 3)
40 runs: starts = first 40 accepted points of a scrambled Sobol stream (seed 20261033) in the V3 box. Optimiser as
V2 (src/optimization/gated_optimizer.optimise, gate "never"): maximise J with the FNO-ensemble-mean gradient, normalized
step h = 0.05 in box-scaled coordinates, projection onto the box, 11 steps → 12 iterates per run (start included).
Frozen trajectory files (sha256 logged) before any trajectory point is simulated. Every iterate gets primal + J adjoint.
Points whose CFD converged and whose adjoint exists but whose post-processing failed enter from the raw records
(as the V2 §9 amendment).

## 6. Hypotheses and decision rules (α ∈ {0.1, 0.2}; threshold 1 − α − 0.05 as v2)
Score s = ‖g − ĝ‖_M / σ̂ with M = diag(span²) of the V3 box, ĝ/σ̂ = FNO-ensemble mean / spread (ddof S−1).
Unit = run. Calibration runs 0–19, test runs 20–39.
- H1″: as H1′ (value vs gradient accuracy, i.i.d. test, n = 64) — reported with the V2 rule.
- H2″: i.i.d. split-conformal coverage cal → test, both models — reported with the V2 rule.
- H5″ (Theorem 1): run-max calibration; supported at α if the fraction of test runs with ALL iterates covered ≥ threshold.
- H6″ (Theorem 2): one random iterate per calibration run (seeded 20261034, 1000 draws); supported if mean per-run
  covered fraction ≥ threshold at both α.
- H7″: i.i.d.-calibrated certificate (V3 cal split) on test runs; replicated if whole-run OR per-iterate coverage
  < threshold at either α.
- H9 (decision value, new): on the 20 test runs, a step is "non-ascent" if gᵀ M ĝ ≤ 0 (CFD gradient g, surrogate
  steepest-ascent direction M ĝ); it is "certified" at α = 0.1 if ‖ĝ‖_M > q̂ σ̂ with the run-max q̂ (H5″).
  (a) report the non-ascent fraction; if it is < 5 %, H9 is "not testable in this regime" (reported as such).
  (b) otherwise H9 is supported if the non-ascent fraction among certified steps is ≤ 1/3 of that among
  uncertified steps (Fisher exact test p reported alongside; if no step is uncertified, H9 is not testable).
Descriptive (not pre-registered): pooled-point calibration, Gaussian, Bonferroni, weighted conformal, ACI (γ 0.01,
0.05), certified fractions for every threshold, leave-one-run-out — same code as S2 (scripts/run_s2_*), V3 paths.

## 7. Stop rules, budget and data safety
Hard cap $12.00 of RunPod spend for all of V3 (approved). Phase 0 (one 16-vCPU CPU pod, ≤ 1 h, ≤ $0.60):
scripts/v3_checks.py — (a) weighted-adjoint J gradient vs the combination of separate CL and CD adjoints, relative
norm difference ≤ 1e-5; (b) adjoint vs FD of dJ/dθ for θ indices 0, 3, 8, 12 on two fixed check designs (scripts/v3_checks.py: the
accepted Sobol-pool point closest to the box centre at α 2.5° and at α 0.5°), FD reference = Richardson
extrapolation (4 D(h/2) − D(h))/3 of central differences with h = 1e-3 (CST) and 0.1° (α) — forces are visibly
nonlinear in θ near shocks (local pilot) — relative difference ≤ 1 % for components > 1 % of the largest. Phase 0 also runs the first 6 cal cases (kept as data;
production setting, 1 thread each, under full load) to measure cost. The go/no-go rule is implemented in
scripts/runpod/v3_projection.py (projected total = spend so far + phase 0 + phase 1 + phase 3 at the measured rate,
packing efficiency 0.85, 0.5 pod-hour overhead per phase). Phase 1 starts automatically ONLY if (a) and (b) pass,
≥ 5 of the 6 phase-0 cal cases converged with a J gradient, and the projected total ≤ $12.00. Otherwise STOP and
report to Mohammad; if only the cost rule fails, the reductions of §8 are proposed and Mohammad asked again.
Pods: no new case after a per-phase deadline computed from the measured rate and the remaining budget; every pod
stops itself only after the Colab receiver confirms a sha256-verified install on Drive (ACK); exports via
token-protected HTTP on the pod's proxy port (no fixed transfer codes); balance checked before each phase.

## 8. Pre-specified reductions (applied in this order only if §7 requires, decided before any V3 data)
(1) trajectory iterates 12 → 10 per run; (2) i.i.d. sizes train 384 → 320, cal 128 → 96, test 64 → 48;
(3) trajectory runs 40 → 30 (cal 0–14, test 20–34). A design cut by a deadline is analysed with the largest of
these designs that is complete, as in v2 §9, and reported as underpowered, never re-defined.

## 9. Analysis code
Written and smoke-tested on fake data BEFORE any V3 trajectory exists; V3 analyses import the S2 code paths with V3
constants (design, N_IT = 12, J gradient key dJ_dtheta).

## 10. Amendments
Amendment 1 — 2026-10-15 05:55Z, written before any V3 data exists (no V3 pod had been created), requested by Mohammad.
(i) Phase-1 go/no-go (§7): in addition to the existing conditions, PROCEED requires the phase-1 deadline
(now + 1.5 × projected phase-1 time) to fall at least 0.5 h before the finisher's hard deadline AF_HARD_DEADLINE_UTC;
a missing or unparseable hard deadline means STOP. Otherwise STOP and ask Mohammad. Reason: the hard deadline (deploy
+ 7 h ≈ $4 at $0.57/h) could otherwise cut a phase 1 the cost rule had approved, producing an unplanned partial design.
(ii) Phase-0 time limit: if P0_DECISION has not appeared 1.5 h after P0_START, the finisher stops all workers and
exports what exists (§7 expects phase 0 ≤ 1 h / ≤ $0.60). Implemented in scripts/runpod/v3_projection.py and
scripts/runpod/finish_v3.sh; the stop reasons are written to results/v3_checks/projection.json.

Amendment 2 — 2026-10-19 05:25Z, written before any V3 data exists (no V3 pod has been created; RunPod console
2026-10-19 05:17–05:20Z: all 11 project pods stopped, balance $19.04), requested by Mohammad. Transfer, data-safety
and wording only: no physics, design, split, surrogate, trajectory, hypothesis, threshold or decision-rule change.
(i) Colab receiver scripts/colab_v2_http_watcher.py: a Drive code file (scripts/, src/, research/) that is replaced
by a differing pod version is backed up as env/code_backups/<relative path with '/' -> '__'>.<UTC stamp> instead of
<file name>.<stamp>, so two changed files with the same name in different folders cannot overwrite each other's
backup within one install (same naming as the amendment-1 installer).
(ii) runpodctl on Colab: notebook 17 cell 4 (wget of a fixed asset URL) failed. Deployment uses notebook 18, whose
cells 6–7 take the v1.14.15 release (the pods' version) from the GitHub API, pick the linux amd64 binary and check it
against the release checksum file when one is published (otherwise the binary's sha256 is printed and logged).
(iii) Wording of amendment 1 (i), matching the code installed with it on 2026-10-15 06:08Z
(scripts/runpod/v3_projection.py): "projected phase-1 time" is the projected phase-1 compute time without the fixed
0.5 h per-phase overhead, i.e. phase-1 deadline = decision time + 1.5 × (h1 − 0.5 h). The cost rule is unchanged
(it includes the overhead).

Amendment 3 — 2026-10-19 07:50Z, approved by Mohammad in chat (2026-10-19, after reading the diagnosis), written after phase 0
(gate FAILED → STOP; V3 spend $0.18) and after an exploratory Colab diagnosis (results/v3_diag; experiment log
2026-10-19 06:27–07:33Z), before any further V3 data. Adjoint numerics, the gate's α reference and the rerun's pod limits
only: no change to the primal physics/numerics, design, splits, surrogates, trajectories, hypotheses, thresholds or
decision rules.
(i) Discrete adjoint (every V3 adjoint: check designs and cal/test/trajectory points): MGLEVEL 0 (no multigrid) in the
adjoint configuration; the primal keeps the 3-level W-cycle. Convergence target unchanged (rms[A_ρ] ≤ 1e-10); iteration cap
8000 instead of 3000 (§2). Reason: on check design 0 the multigrid adjoint converges (rms −10, and −14 gives the same
gradient) to shape sensitivities 8–46 % away from the FD derivative of the same discrete primal, while the single-grid
adjoint agrees to ≤ 1.1 % (results/v3_diag/diag_report.json); converged single-grid CL/CD adjoints needed 3554–3975
iterations (results/v3_diag/adj_mg0L_*).
(ii) Gate check (b): the α FD reference is the Richardson value of central differences at h = 0.025° and h/2 = 0.0125°
(previously 0.1°/0.05°); CST steps unchanged (1e-3, 5e-4). Reason: near the shock the forces are not smooth enough in α for
h = 0.1°; the central-difference sequence 0.1°/0.05°/0.025°/0.0125° of dJ/dα still changes by more than 1 % per halving on
design 0 (2.6 %, 2.6 %, 1.8 %; results/v3_diag/fd_aoa_*), so the old reference cannot support a 1 % test. With (i) and (ii)
the exploratory check passes (all 8 compared components within 1 %, max 0.92 %; results/v3_diag/gate_b_mg0_smallalpha.json);
the rerun must pass on its own.
(iii) Phase 0 is re-run in full on a fresh pod (TAG v3p0b: check designs 0/1 primal, adjoints, FD; the 6 phase-0 cal cases
with the amended adjoint). Before the rerun, the first phase-0 outputs that the rerun re-creates (results/v3_checks,
datasets/raw/v3_checks, datasets/raw/dataset_v3/D3_cal_0000–0005, datasets/processed/dataset_v3/cal/D3_cal_0000–0005.npz)
are moved — not deleted — to archive/v3p0_mg3/ (same relative paths, sha256 manifest) so the receiver installs the rerun
at the canonical paths; logs/v3/v3p0 and the raw archive datasets/archives/af_v3_v3p0_raw.tar.gz stay unchanged. The
go/no-go rule (§7, amendments 1–2) applies unchanged to the rerun, with spend so far = $0.18.
(iv) Code: src/cfd/gradients.py adjoint_config/adjoint_gradients gain optional extra-config-key and iteration-cap
arguments whose defaults reproduce the V1/V2 configurations exactly (checked: identical config text); scripts/run_dataset_v3.py
and scripts/v3_checks.py pass MGLEVEL 0 and 8000; scripts/v3_checks.py uses α steps 0.025°/0.0125°. Pod bundle
v3_bundle_a3.tgz = bundle a1 with these three files and this protocol replaced (manifest updated).
(v) Rerun pod: one 16-vCPU CPU pod as in §7 (live price checked before renting). The finisher's hard deadline
AF_HARD_DEADLINE_UTC is deploy + 10 h instead of deploy + 7 h (amendment 1), because single-grid adjoints roughly double
the adjoint cost; at ≤ $0.57/h this bounds the rerun pod at ≤ $5.70, keeping V3 below the $12 cap. Amendment 1 (i)
(phase-1 deadline ≥ 0.5 h before the hard deadline) applies unchanged.

Amendment 4 — 2026-10-19 19:37Z, approved by Mohammad in chat (2026-10-19, extra budget for a stronger study), written while phase 3 is
running and BEFORE any phase-3 result is analysed: only case counts and solver return codes of phase 3 had been viewed (84/480
cases done, all rc 0); no phase-3 gradient, J or score had been compared with any surrogate prediction. Adds data and one
descriptive check; changes nothing in §§2–6 for the existing 40-run design.
(i) Trajectory extension (phase 3b). 40 further runs, starts = accepted points 41–80 of the same scrambled Sobol stream
(seed 20261033; points 1–40 are the existing runs), same frozen FNO checkpoints, optimiser, step, 11 steps (12 iterates),
case ids T3_s40_t00 … T3_s79_t11, every iterate primal + J adjoint with the amendment-3 adjoint settings. Runs 40–59 are
calibration runs, 60–79 test runs. Frozen file results/v3_traj_ext/trajectory_points.json (sha256 logged before any of its
points is simulated; existing files never overwritten); the pod bundle carries it at the path the phase-3 driver reads.
(ii) Analyses. PRIMARY = the pre-registered §6 analysis of the original 40-run design (runs 0–39), unchanged. SECONDARY
(pre-registered here, same rules and thresholds as §6: H5'' run-max, H6'' random iterate with the same seed and draws, H7'',
H9) on the combined 80-run design: calibration runs 0–19 and 40–59, test runs 20–39 and 60–79. If the extension pod is cut by
its deadline, the secondary analysis uses the original 40 runs plus every extension run with all 12 iterates simulated,
reported with its actual size; it is never re-defined after results are seen. Capped adjoints (iteration cap reached) are
kept and flagged as in §2 / experiment log 2026-10-19; a run without them is descriptive only.
(iii) Mesh-refinement check (descriptive, no decision rule). The first 10 test designs (D3_test_0000–0009) are re-run at two
finer meshes: all E100 cell sizes (first cell, surface, edges, wake) × 1/√2 ("E141") and × 1/2 ("E200"); growth rate,
boundary-layer ratio/thickness and far field unchanged. Primal + J adjoint with the same settings. Reported: relative change of
J and of dJ/dθ (M-norm, cosine) versus the E100 values, as an estimate of the CFD-truth uncertainty to set beside the surrogate
errors. These cases are never used for calibration or testing.
(iv) Budget and pod. V3 hard cap raised from $12.00 to $27.00 (Mohammad approved up to $15 more); one 5 GHz Compute-Optimized
pod (16 vCPU, or 8 vCPU if 16 is unavailable), live price checked, deployed only after phase 3 is installed; finisher hard
deadline deploy + 24 h; per-phase deadline rule of §7 with AF_SPENT_USD = actual V3 spend.

Amendment 4a — 2026-10-19 19:42Z, correction of amendment 4 (iii) written before any mesh-refinement case exists. The refinement uses the
project's uniform-refinement method src/meshing/gmsh_airfoil.MeshSpec.scaled (as in the V1/V2 mesh studies): h_wall, h_surface,
h_edges, h_wake, h_far and the size growth rate are all multiplied by the factor (1/√2 for "E141", 1/2 for "E200");
boundary-layer ratio/thickness, wake length and far-field radius unchanged. Reason: with the E100 settings the cell size away
from the wall is h_surface + 0.08 × wall distance, so keeping the growth rate fixed (as 4 (iii) said) would leave the shock
region almost unrefined. Everything else in amendment 4 is unchanged.
