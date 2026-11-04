"""CDGC: conformal design-gradient certificates (AeroForge-TRIAD).

Setting. A surrogate gives a design gradient ghat(theta) of the objective J (here L/D) and a scale
sigma(theta) > 0 (e.g. ensemble spread of the gradient). CFD gives the reference gradient g(theta)
(discrete adjoint). The normalised nonconformity score is

    s(theta) = || g(theta) - ghat(theta) ||_M / sigma(theta),

with ||v||_M = sqrt(v^T M v) for a fixed SPD metric M (M = diag(span^2) makes the design variables
commensurate: with u = theta/span, grad_u J = span * g, so ||g||_M = ||grad_u J||_2; docstring corrected
2026-10-14 — it previously said diag(1/span^2), while all scripts use diag(span^2)). Split conformal prediction (Vovk et al. 2005; Lei et al. 2018) on n calibration points
gives qhat = the ceil((n+1)(1-a))-th smallest score, and for an exchangeable test point

    P( || g - ghat ||_M <= qhat * sigma ) >= 1 - a.

Certified step. For a step d, the true directional derivative satisfies, on that event,

    g.d >= ghat.d - qhat * sigma * || d ||_{M^-1}       (Cauchy-Schwarz in the M geometry),

so a step is *certified* as an ascent direction when the right-hand side is > 0; a certified
first-order model decrease bound is available for trust-region acceptance tests. When the certificate
fails, the optimiser falls back to CFD (adjoint) at that point.

Exchangeability caveat (the reason for H4). Points visited by an optimiser are not exchangeable with
i.i.d. design samples. `TrajectoryCalibrator` therefore calibrates on scores collected along complete
optimiser runs that are independent of the test run (run-level exchangeability), using the per-run
maximum score (a conservative "whole-trajectory" certificate) or per-step scores with a run-level
split (the per-step guarantee then holds marginally over steps of a new run).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small for level alpha."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def m_norm(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(v)
    return np.sqrt(np.einsum("bi,ij,bj->b", v, M, v))


@dataclass
class GradientCertificate:
    M: np.ndarray                      # SPD metric on gradient space
    alpha: float = 0.1
    qhat: float = float("nan")
    n_cal: int = 0
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def score(self, g_true: np.ndarray, g_hat: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return m_norm(np.asarray(g_true) - np.asarray(g_hat), self.M) / np.maximum(np.asarray(sigma, dtype=float), 1e-12)

    def calibrate(self, g_true: np.ndarray, g_hat: np.ndarray, sigma: np.ndarray) -> "GradientCertificate":
        self.scores = self.score(g_true, g_hat, sigma)
        self.n_cal = len(self.scores)
        self.qhat = conformal_quantile(self.scores, self.alpha)
        return self

    def radius(self, sigma: np.ndarray) -> np.ndarray:
        return self.qhat * np.asarray(sigma, dtype=float)

    def covered(self, g_true: np.ndarray, g_hat: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return self.score(g_true, g_hat, sigma) <= self.qhat

    def worst_case_directional(self, g_hat: np.ndarray, sigma: float, d: np.ndarray) -> float:
        """Lower bound on g.d over the certified ball (ascent if > 0)."""
        Minv = np.linalg.inv(self.M)
        return float(np.dot(g_hat, d) - self.qhat * sigma * math.sqrt(float(d @ Minv @ d)))

    def certifies_ascent(self, g_hat: np.ndarray, sigma: float, d: np.ndarray, margin: float = 0.0) -> bool:
        return self.worst_case_directional(g_hat, sigma, d) > margin


@dataclass
class TrajectoryCalibrator:
    """Run-level calibration on optimiser trajectories (H4).

    runs: list of arrays of per-step scores, one array per independent optimiser run.
    mode 'max'  -> certificate for every step of a new run simultaneously (conservative);
    mode 'step' -> pooled per-step scores, quantile at level computed over runs (marginal per step)."""
    alpha: float = 0.1
    mode: str = "max"
    qhat: float = float("nan")

    def calibrate(self, runs: list[np.ndarray]) -> "TrajectoryCalibrator":
        if self.mode == "max":
            self.qhat = conformal_quantile(np.array([np.max(r) for r in runs]), self.alpha)
        elif self.mode == "step":
            self.qhat = conformal_quantile(np.concatenate(runs), self.alpha)
        else:
            raise ValueError(self.mode)
        return self


def ld_gradient(cl: np.ndarray, cd: np.ndarray, dcl: np.ndarray, dcd: np.ndarray) -> np.ndarray:
    """Gradient of L/D from coefficient values and gradients (quotient rule), batched."""
    cl, cd = np.asarray(cl)[..., None], np.asarray(cd)[..., None]
    return (np.asarray(dcl) * cd - cl * np.asarray(dcd)) / cd**2


def cosine(a: np.ndarray, b: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if M is None:
        M = np.eye(a.shape[1])
    return np.einsum("bi,ij,bj->b", a, M, b) / (m_norm(a, M) * m_norm(b, M) + 1e-300)
