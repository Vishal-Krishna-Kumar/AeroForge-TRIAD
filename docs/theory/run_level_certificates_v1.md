# Run-level conformal design-gradient certificates — theory draft v1 (2026-10-14)

Supersedes v0 (`run_level_certificates.md`, kept unchanged for the record). Changes: full proofs of the quantile
lemma and Theorems 1–2 (incl. the upper bound and ties), a calibration-conditional remark, Proposition 3 with the
M-inner-product algebra written out, Corollary 4 (two-stage gating, which removes the circularity flagged in v0),
Proposition 5 (stationarity) with constants, a list of what is and is not new, and a Monte Carlo sanity check.
Status: proofs written out and self-checked once; they must still be checked by a second reader before submission.
Items marked [CHECK] are claims about other papers that have not yet been verified against the full text.

## 1. Setting and notation
- Design variables θ ∈ Θ = [lo, hi] ⊂ R^d (a box). True objective J: Θ → R, gradient g(θ) = ∇J(θ), available only
  through a deterministic oracle (here CFD primal + discrete adjoint).
- Surrogate ensemble: point estimate ĝ(θ) and scale σ̂(θ) > 0, both fixed functions once the training data D₀
  are fixed. M ≻ 0 symmetric; ⟨a, b⟩_M = aᵀMb, ‖a‖_M = (aᵀMa)^½. Nonconformity score s(θ) = ‖g(θ) − ĝ(θ)‖_M / σ̂(θ).
- Optimizer 𝒜: maps a start θ₀ (and, optionally, independent internal randomness ξ) to a finite set of iterates
  V(θ₀, ξ) ⊂ Θ — the points where a certificate is required (every point where a surrogate step is taken).
- Starts θ₀^(1), …, θ₀^(R+1) i.i.d. from π, independent of D₀; ξ^(r) i.i.d., independent of everything else.
- Run r produces W_r = {(θ, s(θ)) : θ ∈ V_r}. Everything below is conditional on D₀.

**Assumption A (run-level i.i.d.).** Given D₀, W_1, …, W_{R+1} are i.i.d.
Sufficient condition: 𝒜, ĝ, σ̂, the oracle and any threshold used inside 𝒜 are fixed given D₀ (or given data
independent of the R+1 runs), and runs differ only through (θ₀^(r), ξ^(r)). Nothing is assumed about the
dependence between iterates of the same run, about run lengths |V_r| (they may differ), or about the existence of a
density of the iterates.

## 2. The quantile lemma
For scalars S_1, …, S_R and α ∈ (0,1), let k = ⌈(R+1)(1−α)⌉ and q̂ = S_(k) (k-th smallest) if k ≤ R, q̂ = +∞ otherwise.

**Lemma 0.** If S_1, …, S_{R+1} are exchangeable, then P(S_{R+1} ≤ q̂) ≥ 1 − α. If moreover ties occur with
probability 0, then P(S_{R+1} ≤ q̂) ≤ 1 − α + 1/(R+1).
Proof. If k > R the claim is trivial (q̂ = ∞). Otherwise, S_{R+1} ≤ S_(k) (order statistic of the first R) holds iff
S_{R+1} is among the k smallest of all R+1 values, where ties are broken in favour of S_{R+1}
(i.e. rank(S_{R+1}) ≤ k with rank counted as 1 + #{j ≤ R : S_j < S_{R+1}}). By exchangeability the rank of S_{R+1}
among the R+1 values is stochastically no larger than uniform on {1, …, R+1} under this tie-breaking, so
P(rank ≤ k) ≥ k/(R+1) ≥ 1 − α. Without ties the rank is exactly uniform, so P = k/(R+1) < (R+1)(1−α)/(R+1) + 1/(R+1). ∎
(Standard: Vovk, Gammerman & Shafer 2005; Lei et al. 2018, Thm 1; Angelopoulos & Bates 2023, Thm D.1. [CHECK exact
theorem numbers before citing.])

**Remark (calibration-conditional validity).** For fixed calibration runs, the conditional coverage
C = P(S_{R+1} ≤ q̂ | S_1..S_R) of a continuous score is Beta(k, R+1−k)-distributed (Vovk 2012 [CHECK]). Computed
(scipy 2026-10-14): R = 40, α = 0.1 (k = 37): E C = 0.902, P(C ≥ 0.80) = 0.972; R = 20 (full V2 design): α = 0.1
(k = 19) P(C ≥ 0.80) = 0.931, α = 0.2 (k = 17) P(C ≥ 0.70) = 0.893; R = 15 (reduced design): α = 0.1 (k = 15,
E C = 0.938) P(C ≥ 0.80) = 0.965, α = 0.2 (k = 13) P(C ≥ 0.70) = 0.873. With R = 15–20 a single calibration batch
can therefore give a realised coverage several points away from nominal; the H5 rule's "− 0.05" slack and the
Clopper–Pearson intervals on 15–20 test runs must be read with that in mind.

## 3. Run-level certificates

**Theorem 1 (simultaneous whole-run certificate).** Let S_r = max_{θ ∈ V_r} s(θ) and q̂ as in §2. Under A,
  P( ‖g(θ) − ĝ(θ)‖_M ≤ q̂ σ̂(θ) for all θ ∈ V_{R+1} ) ≥ 1 − α,
and if the S_r are a.s. distinct the probability is ≤ 1 − α + 1/(R+1). The probability is over the R calibration
runs and the new run (conditional on D₀).
Proof. S_r is a measurable function of W_r, so by A the S_r are i.i.d., hence exchangeable. Because σ̂ > 0,
{‖g − ĝ‖_M ≤ q̂σ̂ ∀θ ∈ V_{R+1}} = {max_θ s(θ) ≤ q̂} = {S_{R+1} ≤ q̂}. Apply Lemma 0. ∎

**Theorem 2 (random-iterate certificate).** For each run draw τ_r uniformly from V_r (independently, given W_r) and
let S_r = s(θ_{τ_r}). Under A, P( s(θ_{τ_{R+1}}^{(R+1)}) ≤ q̂ ) ≥ 1 − α. Equivalently, E[ (1/|V_{R+1}|) Σ_{θ∈V_{R+1}}
1{s(θ) ≤ q̂} ] ≥ 1 − α: the expected fraction of covered iterates of a new run, each run weighted equally.
Proof. (W_r, τ_r) are i.i.d., so the S_r are i.i.d.; Lemma 0 gives the first statement. For the second, condition on
q̂ and W_{R+1}: P(s(θ_τ) ≤ q̂ | q̂, W_{R+1}) = (1/|V|)Σ 1{s(θ) ≤ q̂}; take expectations. ∎
Note (checked against the arXiv abstracts 2026-10-14; full texts still to be read). Dunn, Wasserman & Ramdas,
"Distribution-free prediction sets for two-layer hierarchical models" (arXiv 1809.07441; JASA 2023 [CHECK venue/year])
propose CDF pooling (asymptotic coverage), single subsampling and repeated subsampling (recommended when a guarantee
is required). Theorem 2 is, in substance, their single-subsampling construction applied to runs as groups — it is
NOT claimed as new; the paper should cite it as such and consider their repeated-subsampling variant instead of one
draw. Lee, Barber & Willett, "Distribution-free inference with hierarchical data" (arXiv 2306.06342, 2023) derive a
hierarchical exchangeability and a second-moment coverage guarantee for repeated measurements.
Dunn et al. full text (arXiv v4, 24 Feb 2022; read via WebFetch summary 2026-10-14 — re-read the PDF directly before
citing): the coverage target is ONE new observation from a new group; Method 2 (single subsampling) has finite-sample
coverage ≥ 1 − α (their Thm 5, via Thm 1), Method 3 (repeated subsampling, averaged p-values) has a formal 1 − 2α
guarantee (their Thm 6) and empirically ≈ 1 − α; Method 0 (double conformal) summarises groups by interval endpoints;
no group-maximum construction for simultaneous coverage of all observations in a new group. Lee–Barber–Willett full
text (arXiv v4, read via WebFetch summary 2026-10-14; re-read before citing): the formal target is one response of a
new group; "second-moment coverage" is E[α_D(X_test)²] ≤ α²; their HCP² uses within-group pairwise minima of scores
(a second-moment device), not maxima; simultaneous coverage of all measurements is mentioned as a motivation (ensemble
forecasts) but not constructed. So neither paper states Theorem 1's group-maximum construction. That construction is
nonetheless elementary (it is split conformal applied to a group summary), so the paper should present it as a
simple, explicitly-acknowledged adaptation and rest the novelty claim on items 1–4 of §5, after a further search for
"simultaneous" / "family-wise" grouped conformal (e.g. multi-output / conformal for trajectories / time series).
**Found (2026-10-14): Cleaveland, Lee, Pappas & Lindemann, "Conformal prediction regions for time series using linear
complementarity programming" (arXiv 2304.01075; AAAI 2024 [CHECK venue]) use exactly this mechanism for trajectories:
nonconformity R = max_t α_t ‖Y_t − Ŷ_t‖ (their eq. 3) with joint coverage of all time steps (their Thm 2 /
Corollary 1), contrasted with per-step regions + union bound. Their weights α_t are fixed per time index; ours are
state-dependent (1/σ̂(θ)) and the "trajectory" is produced by the optimiser following the surrogate's own gradient.
Consequence: Theorem 1 is NOT a new mechanism and must be presented as an application of the max-over-trajectory
score (cite Cleaveland et al. and the conformal-planning line of Lindemann et al. [CHECK which paper first]).
The novelty claim is restricted to items 1–4 of §5.**

**What pooling does not give.** The pooled quantile of all calibration iterates has no finite-sample guarantee under A:
iterates of a run are dependent and runs have different lengths, so the pooled scores are not exchangeable with the
scores of a new run. (Simulation in §6: pointwise ≈ nominal, whole-run far below nominal.)

## 4. From certificates to optimisation guarantees

**Proposition 3 (certified ascent direction).** Fix c ∈ (0, 1). At a point θ with ‖g − ĝ‖_M ≤ c‖ĝ‖_M:
 (i) ⟨g, ĝ⟩_M ≥ (1 − c)‖ĝ‖_M² > 0, so the step direction d = Mĝ is an ascent direction of J
     (directional derivative gᵀd = ⟨g, ĝ⟩_M);
 (ii) (1 − c)‖ĝ‖_M ≤ ‖g‖_M ≤ (1 + c)‖ĝ‖_M;
 (iii) ⟨g, ĝ⟩_M / (‖g‖_M ‖ĝ‖_M) ≥ (1 − c)/(1 + c)  (cosine in the M-geometry).
Proof. (i) ⟨g, ĝ⟩_M = ‖ĝ‖² + ⟨g − ĝ, ĝ⟩ ≥ ‖ĝ‖² − ‖g − ĝ‖‖ĝ‖ ≥ (1 − c)‖ĝ‖² (Cauchy–Schwarz in ⟨·,·⟩_M). (ii) Triangle
inequality. (iii) Divide (i) by ‖g‖‖ĝ‖ and use the upper bound in (ii). ∎
On the event of Theorem 1, the hypothesis holds at every θ ∈ V_{R+1} with q̂σ̂(θ) ≤ c‖ĝ(θ)‖_M ("gate open").

**Corollary 4 (two-stage gating, removes the circularity).** If the gate inside 𝒜 uses a threshold q̂₀ computed from
data independent of the R+1 runs (a pilot batch or D₀), Assumption A holds. Let q̂ be the Theorem-1 threshold from the
calibration runs. On the Theorem-1 event, every gated-open point satisfies ‖g − ĝ‖_M ≤ q̂σ̂ = (q̂/q̂₀)·q̂₀σ̂ ≤
c' ‖ĝ‖_M with c' = c · max(1, q̂/q̂₀). Hence Proposition 3 holds with c' whenever c' < 1, with probability ≥ 1 − α.
Proof. Immediate from Theorem 1 and Proposition 3. ∎
Practical reading: choose c so that c · q̂/q̂₀ < 1 with margin; q̂/q̂₀ is observable after calibration. (In V1/H3 the gate
used the i.i.d. q̂ from D₀, so A held for the H3 runs.)

**Proposition 5 (stationarity of a certified run).** Work in scaled coordinates u = S⁻¹θ with S = diag(hi − lo), and take
M = S² (as in our experiments). Then ∇_u J = S g, ‖∇_u J‖₂ = ‖g‖_M, and the optimiser step u_{t+1} = u_t + h d_t with
d_t = Sĝ / ‖Sĝ‖₂ is exactly normalised ascent in u. Assume (a) J is L-smooth in u on the box; (b) no projection is
active along the run (iterates stay in the interior); (c) every step uses either the oracle gradient or a surrogate
gradient satisfying the hypothesis of Proposition 3 with constant c. Then for every t,
  J(u_{t+1}) ≥ J(u_t) + h κ ‖∇_u J(u_t)‖₂ − L h²/2,   κ = (1 − c)/(1 + c),
and with J* = sup_Θ J,
  min_{t<T} ‖∇_u J(u_t)‖₂ ≤ (1/κ) [ (J* − J(u_0)) / (hT) + L h / 2 ].
Proof. ⟨∇_u J, d_t⟩ = ⟨g, ĝ⟩_M / ‖ĝ‖_M ≥ (1 − c)‖ĝ‖_M ≥ κ ‖g‖_M = κ‖∇_u J‖ by Prop. 3 (i)–(ii) (oracle steps: c = 0).
The ascent lemma for L-smooth J with ‖d_t‖ = 1 gives the one-step bound. Sum over t < T, use J(u_T) ≤ J*, bound the sum
of gradient norms below by T·min. ∎
On the Theorem-1 event (probability ≥ 1 − α, or Corollary 4 with c'), hypothesis (c) holds for the whole run.
Limitations: (b) fails when box constraints become active — a projected-gradient-mapping version is needed (open);
L is unknown, so the bound is qualitative; the bound concerns the true J only at the iterates.

## 5. What is new, and what is not (for the paper's claims)
Not new: Lemma 0, Theorem 2 (single subsampling of grouped data, Dunn et al.), Theorem 1's max-over-trajectory
score (Cleaveland et al. 2023/24 for time series), the use of a group-level statistic for grouped data, Proposition 3 (i)–(ii) as algebra, the ascent lemma. Claimed as new, pending the full-text check of Dunn et al. 2023, Lee/Barber/Willett 2023, Fannjiang et al. 2022,
Prinster et al. 2024 and Choi 2026 [CHECK]:
1. Calibrating *gradient* (design-sensitivity) errors of a surrogate rather than values.
2. Run-level calibration for *deterministic* optimisers: no density or likelihood ratio of the iterates is needed
   (contrast: weighted / feedback-covariate-shift conformal requires one).
3. The chain Theorem 1 → Corollary 4 → Proposition 5: one probability statement for a whole design run that implies an
   ascent-direction and stationarity guarantee for the *true* objective, with the gating circularity resolved.
4. Empirical evidence on a real CFD + adjoint pipeline (V1, V2) that i.i.d. calibration and pooled calibration do not
   deliver run-level validity while run-max does (V1 exploratory: §6 of v0; V2: pre-registered H5–H7, not yet run).
5. (added 2026-10-14) Feedback-free deployment: online/adaptive conformal methods (adaptive conformal inference,
   Gibbs & Candès, NeurIPS 2021; conformal PID control, Angelopoulos, Candès & Tibshirani 2023 [CHECK]) adapt their
   threshold from the observed error at every step, i.e. they need the TRUE gradient (one CFD + adjoint solve) after
   each optimiser step — which removes the reason for using a surrogate — and they bound long-run average
   miscoverage, not the probability that a whole finite run is covered. The run-level certificate needs no oracle
   call while the optimiser runs; its cost is paid once, in the calibration runs. ACI is included as a descriptive
   baseline (scripts/run_s2_baselines.py, γ ∈ {0.01, 0.05}, fixed before any trajectory result) so the comparison is
   empirical, not only argued.

## 6. Monte Carlo sanity check (toy problem, not CFD)
Script `scripts/theory/sim_runlevel.py` (d = 4, smooth J, fixed smooth surrogate error, partly informative σ̂,
deterministic normalised ascent with box projection, T = 15, R = 40 calibration runs, α = 0.1, 2000 repetitions;
run 2026-10-14 in Claude's workspace, ≈ 70 s):
- Theorem 1 (run-max): P(all test-run iterates covered) = 0.909 (theory: ≥ 0.900, ≤ 0.924).
- Theorem 2 (one random iterate): 0.906 (theory ≥ 0.900).
- Pooled baseline: pointwise 0.899 but whole-run only 0.741.
- Proposition 3 (i) on the Theorem-1 event with c = 0.5: held at all 23,137 gated-open test iterates.
Monte Carlo s.e. ≈ 0.0065. This checks the algebra and the exchangeability argument; it says nothing about airfoils.

## 7. Open items before submission
1. Second-reader check of every proof; fix [CHECK] citations against full texts.
2. Projected (box-constrained) version of Proposition 5.
3. (done 2026-10-14, §2 remark) Beta-distribution figures for R = 15 / 20.
4. Decide whether Theorem 2's one-draw scheme is reported or replaced by the Dunn et al. grouped estimator.
