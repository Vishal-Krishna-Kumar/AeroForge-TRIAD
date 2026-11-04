"""Baseline gradient certificates for comparison with run-level calibration (AeroForge-TRIAD).

Written 2026-10-14, before any V2 trajectory data exist. These are DESCRIPTIVE baselines (not pre-registered in
protocol v2; results from them must be labelled as such). All return a threshold q such that the certificate at a
point is ||g - ghat||_M <= q * sigma, i.e. score <= q, with score = ||g - ghat||_M / sigma.

1. gaussian_threshold      : uncalibrated ensemble UQ. Treat g - ghat ~ N(0, (sigma^2/d) I) in the M-geometry, so
                             score^2 ~ chi2_d / d; q = sqrt(chi2_{d,1-a} / d). No calibration data used.
2. bonferroni_runs         : i.i.d. split-conformal quantile at level a / n_steps, applied to every step of a run
                             (union bound for a whole run if the i.i.d. guarantee held at each step; it does not
                             under optimiser-induced shift, which is the point of the comparison).
3. weighted_conformal      : Tibshirani et al. (NeurIPS 2019) covariate-shift conformal with likelihood ratio
                             w(x) = p_traj(x) / p_iid(x) estimated by a probabilistic classifier (logistic
                             regression on design variables) trained to separate i.i.d.-calibration points from
                             calibration-run trajectory points. Point-wise (marginal) guarantee only if w is exact
                             and the test point is drawn from p_traj independently, neither of which holds exactly
                             here (iterates are dependent); reported as the standard density-ratio baseline.
4. aci_run_thresholds      : adaptive conformal inference (Gibbs & Candes, NeurIPS 2021) along each run:
                             alpha_{t+1} = alpha_t + gamma (alpha - err_t), q_t = i.i.d. split-conformal quantile at
                             level alpha_t. NEEDS the true gradient (one oracle/CFD call) after every step to update,
                             i.e. more information than the run-level certificate; its guarantee is a long-run average
                             miscoverage bound, not a finite-sample whole-run statement. Added 2026-10-14 (before any
                             trajectory result was seen) as the standard online/adaptive baseline.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2

from gradient_certification.cdgc import conformal_quantile


def gaussian_threshold(d: int, alpha: float) -> float:
    return float(math.sqrt(chi2.ppf(1.0 - alpha, d) / d))


def bonferroni_runs(iid_scores: np.ndarray, alpha: float, n_steps: int) -> float:
    return conformal_quantile(np.asarray(iid_scores, float), alpha / max(int(n_steps), 1))


def fit_density_ratio(x_iid: np.ndarray, x_traj: np.ndarray, C: float = 1.0, span=None):
    """Return w(x) ≈ p_traj(x)/p_iid(x) from a logistic classifier (class 1 = trajectory), corrected for class sizes.
    Inputs are scaled by `span` (design-box span) if given. Pure NumPy (Newton / IRLS), no sklearn dependency."""
    X = np.vstack([x_iid, x_traj]).astype(float)
    if span is not None:
        X = X / np.asarray(span, float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    y = np.r_[np.zeros(len(x_iid)), np.ones(len(x_traj))]
    beta = np.zeros(Z.shape[1])
    lam = 1.0 / C
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ beta))
        Wd = p * (1 - p) + 1e-9
        R = np.diag(np.r_[0.0, np.full(Z.shape[1] - 1, lam)])
        Hm = Z.T @ (Z * Wd[:, None]) + R
        gr = Z.T @ (y - p) - R @ beta
        step = np.linalg.solve(Hm, gr)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    prior = len(x_iid) / len(x_traj)

    def w(x):
        x = np.atleast_2d(np.asarray(x, float))
        if span is not None:
            x = x / np.asarray(span, float)
        z = np.hstack([np.ones((len(x), 1)), (x - mu) / sd])
        p = 1 / (1 + np.exp(-z @ beta))
        return prior * p / np.maximum(1 - p, 1e-12)
    return w


def weighted_conformal_threshold(cal_scores: np.ndarray, cal_w: np.ndarray, test_w: float, alpha: float) -> float:
    """Tibshirani et al. 2019: smallest q with sum_i p_i 1{s_i <= q} >= 1 - alpha, where p_i = w_i / (sum w + w_test)
    and the test point's mass w_test sits at +inf."""
    s = np.asarray(cal_scores, float); w = np.asarray(cal_w, float)
    order = np.argsort(s); s, w = s[order], w[order]
    tot = w.sum() + float(test_w)
    cdf = np.cumsum(w) / tot
    idx = np.searchsorted(cdf, 1.0 - alpha - 1e-12)
    return float("inf") if idx >= len(s) else float(s[idx])


def aci_run_thresholds(iid_scores: np.ndarray, run_scores: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Thresholds q_1..q_T used at the T steps of one run (q_t depends only on errors at steps < t)."""
    cal = np.asarray(iid_scores, float)
    a_t, qs = float(alpha), []
    for s in np.asarray(run_scores, float):
        q = float("inf") if a_t <= 0 else (float(cal.min()) if a_t >= 1 else conformal_quantile(cal, a_t))
        qs.append(q)
        a_t += gamma * (alpha - float(s > q))
    return np.asarray(qs)
