# Experiment protocol v2 — FROZEN 2026-10-13 before any Dataset V2 case — AeroForge-TRIAD, study S2 (CST-13)

Status: FROZEN 2026-10-13 (sha256 of this file recorded in the experiment log) before any Dataset V2 case was run.
Mohammad approved the compute budget on 2026-10-13 (balance $24.13, no further top-up). Changes after freezing go
to §9 with date and reason. (A local pilot of 3 shapes, cost/feasibility only, preceded this; it is not V2 data.)
Context: protocol v1 results — H2 (i.i.d.) supported at α 0.1; H2-trajectory falsified; H3 falsified; H4
supported (descriptive, 10 starts). Theory draft: docs/theory/run_level_certificates.md.

## 1. Data
- Geometry: CST, 6 upper + 6 lower Bernstein coefficients + α (13 variables), src/geometry/cst.py. Box: CST fit of
  NACA 2411 ± 0.06 per coefficient, α ∈ [−2°, 4°]; shape accepted iff t_max ∈ [0.08, 0.14], camber ∈ [−0.01, 0.045],
  no self-intersection (rejected Sobol points are skipped and counted, never silently).
- Flow: as V1 (2D incompressible laminar, Re 1000, P-coarse mesh, Cauchy stop + settle). Regime gate: the 4 most
  extreme accepted shapes (max t·camber, max t, max camber, max waviness) at α 4°, time-accurate as V1 round 3;
  any unsteady → box shrunk and re-checked before data.
- Splits (independent scrambled Sobol, fixed seeds 20261020/21/22; first n accepted points; rejections counted):
  train 1024 (primal), cal 256 and test 128 (primal + ∇(L/D) by one weighted adjoint, w_L = 1/CD, w_D = −CL/CD²).
  The design file datasets/designs/dataset_v2_design.json is generated once in the Claude workspace (scipy 1.18.1)
  and shipped unchanged (sha256 eae20311…f6b3704).
- Gate checks (scripts/v2_checks.py), all before any V2 case is used: (a) regime gate as above; (b) weighted-adjoint
  validation: relative norm difference to (1/CD)∇CL − (CL/CD²)∇CD from two single adjoints ≤ 1e-5; (c) adjoint vs
  central FD on deformed meshes for au0, au3, al2, α on the pool shape nearest the box centre at α 4°: ≤ 1 % for every
  component larger than 1 % of the largest. To use idle cores, train cases may run while the gates run; if any gate
  fails they are moved to datasets/raw/_pregate_discarded and never used.

## 2. Surrogates
FNO ensemble (5 seeds) and MLP ensemble as V1, retrained on V2 train only; hyper-parameters fixed as V1 except
input dimension. No tuning on cal/test.

## 3. H1′ / H2′ — i.i.d. replication in 13-D (cal → test)
Same statistics and decision rules as protocol v1 §3–4, on V2 cal/test.

## 4. S2 trajectory study (primary)
- Starts: 40 accepted Sobol points (seed 20261023). Optimizer: surrogate-only ("never" gate), FNO ensemble mean
  ĝ, normalized step h = 0.05 in box-scaled coordinates, projection onto the coefficient box, 14 steps →
  15 iterates per run including the start. Shapes leaving the acceptance set are still evaluated and flagged.
- Ground truth at EVERY iterate: primal + ∇(L/D). Unit = run (start).
- Score s = ‖g − ĝ‖_M / σ̂ with M = diag(span²) (span = 0.12 per coefficient, 6° for α), σ̂ as v1.
- Pre-specified split: calibration runs = starts 0–19, test runs = starts 20–39.

H5 (whole-run certificate, Theorem 1). Run-max calibration on runs 0–19; on runs 20–39 the fraction of runs
with ALL iterates covered. Supported at α if that fraction ≥ 1 − α − 0.05 for α ∈ {0.1, 0.2}.
H6 (random-iterate certificate, Theorem 2). Calibration by one random iterate per run (seeded, 1000 draws,
mean coverage reported); test = per-run fraction of covered iterates averaged over runs 20–39.
Supported if ≥ 1 − α − 0.05 at both α.
H7 (shift replication). The i.i.d.-calibrated certificate (V2 cal split) on runs 20–39: H2-trajectory is
replicated if whole-run OR per-iterate coverage < 1 − α − 0.05 at either α.
Reported for all: Clopper–Pearson / run-bootstrap CIs (seed 20261024), q̂ values (width cost of simultaneity),
leave-one-run-out descriptive results, pooled-point calibration as a baseline without guarantee.
Note on power: 20 test runs give a CI half-width ≈ ±0.13 at 0.9; decisions use the point estimate as in v1 and
the CIs are reported next to them.

## 5. Gating (H3 in 13-D) — NOT in this compute phase
Deferred to a later, separately approved phase; stated here so it cannot be added post hoc to this phase.

## 6. Stop rules
Regime gate fails; gradient check fails; weighted-adjoint validation fails (→ two adjoints, re-estimated cost
reported to Mohammad before continuing); pod balance deadline reached (blocks run in order regime → checks →
train → cal/test → trajectories; a cut loses trajectory runs from the end of the test range first — the cut is
recorded and H5–H7 are then reported as underpowered, not re-defined).

## 7. Compute
RunPod CPU pod (32 vCPU), estimate in docs/dataset_v2_plan_draft.md (approved). Session 1: no new chunk after
start + 8.5 h, workers stopped at start + 9.5 h (≤ ≈ $13.2 incl. export); session 2 gets the remaining balance.
The pod self-stops after the data leave it; total spend never exceeds the $24.13 balance.

## 9. Change log
- 2026-10-13: draft created; gate tolerances, design sha and session deadlines added; frozen.
- 2026-10-13 (after freezing, before any trajectory exists; session 1 running): clarification of §4 — the
  surrogate-only optimizer maximises L/D (the quantity whose gradient the certificate covers), with ĝ the FNO
  ensemble-mean L/D gradient; starts are the first 40 accepted points of the Sobol stream seed 20261023 (same
  acceptance as §1). Implemented in scripts/make_v2_trajectories.py; surrogates by scripts/train_surrogates_v2.py
  (V1 procedure and hyper-parameters, fit on train 0–895, validation 896–1023, FNO 500 epochs, MLP 3000 epochs).
- 2026-10-13 19:25Z (after the gate results, before any V2 dataset case is used): launch 2 ran the gates (18:37–19:00Z).
  Gradient gate PASS (weighted vs two adjoints 4.4e-7; adjoint vs FD ≤ 5.9e-5). Regime gate labels: max_t steady;
  max_t·camber, max_camber, max_waviness `not_settled` (CL oscillation p2p 4.9e-5–8.8e-5, second/first-half RMS ratio
  0.66–0.73, i.e. still decaying at t = 30; none `unsteady`). §1 says the box is changed only if a corner is
  *unsteady*; scripts/runpod/run_v2_session1.sh (launch 2) instead required the label `steady` for all four — an
  implementation stricter than the protocol — and stopped the pod. V1's accepted box corners showed the same decaying
  behaviour (p2p ≈ 1.2e-4, ratio 0.68–0.75, also labelled `not_settled`). Decision: the protocol text governs; the
  regime gate is recorded as PASS under §1, the script is corrected to fail only on `unsteady`/missing, and the data
  run is relaunched (launch 3) without repeating the gates. This is stated here and will be stated in the paper.
- 2026-10-13 21:30Z (budget amendment; no V2 result has been examined — only case counts and wall times were read;
  no trajectory exists). Measured cost on the session-1 pod is ≈ 3–4× the plan (cal cases 2000–3300 s instead of
  ≈ 720 s; host load ≈ 190 on a shared host, 2 threads per SU2 process; our own process count is not the cause).
  A 5 GHz 8-vCPU pod measured ≈ 1.65× more single-thread work per dollar. The full plan no longer fits the fixed
  $24.13 balance, so, decided now and applied whatever the results:
  (a) Session 1 runs a case-level queue in this priority order: train 0–511 → test 0–63 → cal 0–127 →
      test 64–127 → cal 128–255 → train 512–1023 (scripts/runpod/run_v2_session1b.sh). Each split is a prefix of
      its Sobol stream, so whatever completes is still a space-filling design. Minimum targets: train 512,
      test 64, cal 128. Cases completed by launch 3 before the switch are kept (same code, same design file).
  (b) Surrogates are fitted on the first ⌊0.875 N⌋ completed train cases in Sobol order and validated on the
      rest (N = completed train cases; 896/128 if all 1024 complete). Hyper-parameters unchanged. The fit size
      is read from the checkpoint metadata by every later script.
  (c) H1′/H2′ use the completed cal/test prefixes; the sizes are reported.
  (d) Trajectories are generated exactly as in §4 (40 × 15). Session 2 simulates the REDUCED design first —
      calibration runs 0–14 and test runs 20–34, iterates 0–9 — then the remaining points
      (scripts/runpod/run_v2_session2.sh). If every one of the 600 points is simulated, H5–H7 use the full
      design; otherwise they use the reduced design (15 cal runs, 15 test runs, 10 iterates each). Nothing else
      in §4 changes; the lower power (15 test runs: CI half-width ≈ ±0.15 at 0.9) is reported.
- 2026-10-13 22:05Z (same day, still before any V2 result is examined): Mohammad approved additional funds, so the
  study goes back to the FULL design (all 1024/256/128 cases; 40 × 15 trajectory points). The work moved to three
  5 GHz 16-vCPU pods (scripts/runpod/run_v2_worker.sh; queue scripts/runpod/v2_move_queue.py, same priority order
  as (a), item i → pod i mod 3). Same code, design file and solver settings; only the host CPU differs (AMD EPYC
  4564P vs 9655), which is recorded per case in the pod logs. Rules (a)–(d) above stay in force only as the
  fallback if a deadline or the budget still cuts the run; the analysis rule in (d) already selects the full
  design whenever all 600 points are simulated.

**§9 amendment, 2026-10-14 ~18:50Z (before any H5–H7 number was computed; Claude, logged for Mohammad's review).**
Ground truth for the trajectory score is the converged primal + adjoint L/D gradient. A trajectory point whose CFD
converged and whose adjoint gradient exists but whose *field* post-processing failed (no processed npz; first case:
T2_s37_t04, "1 fluid grid point not covered by the mesh") is therefore included, with θ from the frozen
`trajectory_points.json` and g from `gradients.json`, instead of being dropped as "missing". Points added this way are
listed in the report (`raw_gradient_fallback`). Implemented in `scripts/run_s2_analysis.py` (sha a27bac1e…) and used
identically by `scripts/run_s2_baselines.py` (sha 9b396191…). Smoke-tested on fake data only.
