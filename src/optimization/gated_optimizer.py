"""Surrogate-driven design optimisation with CFD fallback gates (AeroForge-TRIAD, H3).

Problem: maximise J(theta) = L/D at Re = 1000 over the Dataset V1 box (projected onto the box).
Each iteration proposes a projected-gradient step from the surrogate gradient. A *gate* decides
whether the step can be taken on the surrogate alone or whether CFD (primal + adjoint) must be called
at the current point first. A CFD call returns the true J and gradient; the step is then taken with
the CFD gradient (model-management fallback) and the point is logged.

Gates (all compared at equal CFD budget):
  never      : never call CFD (ungated surrogate optimisation)
  always     : call CFD every step (gradient ascent on CFD; the budget-limited reference)
  cdgc       : call CFD unless the conformal gradient certificate certifies ascent for the step
  output_uq  : call CFD if the ensemble std of predicted J exceeds a threshold
  distance   : call CFD if the distance to the nearest training/CFD point exceeds a threshold
                (Cheng et al. 2025-style trigger)

Every final design is verified by CFD (not counted against the in-loop budget, reported separately).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from gradient_certification.cdgc import GradientCertificate


class Surrogate(Protocol):
    def predict(self, theta: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        """-> (J_hat, grad_hat, sigma_grad, sigma_J) at one theta."""


CFDOracle = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self.lo), self.hi)

    @property
    def span(self) -> np.ndarray:
        return self.hi - self.lo


@dataclass
class Gate:
    kind: str = "never"
    cert: GradientCertificate | None = None
    uq_threshold: float = float("inf")
    dist_threshold: float = float("inf")
    margin: float = 0.0

    def needs_cfd(self, theta, J_hat, g_hat, sig_g, sig_J, d, data_points: np.ndarray, span: np.ndarray) -> bool:
        if self.kind == "never":
            return False
        if self.kind == "always":
            return True
        if self.kind == "cdgc":
            return not self.cert.certifies_ascent(g_hat, sig_g, d, self.margin)
        if self.kind == "output_uq":
            return sig_J > self.uq_threshold
        if self.kind == "distance":
            if len(data_points) == 0:
                return True
            dist = np.min(np.linalg.norm((data_points - theta) / span, axis=1))
            return dist > self.dist_threshold
        raise ValueError(self.kind)


@dataclass
class RunLog:
    thetas: list = field(default_factory=list)
    J_hat: list = field(default_factory=list)
    cfd_calls: list = field(default_factory=list)      # (step, theta, J, grad)
    gated: list = field(default_factory=list)


def optimise(theta0: np.ndarray, surrogate: Surrogate, gate: Gate, cfd: CFDOracle, box: Box,
             data_points: np.ndarray, step0: float = 0.1, n_steps: int = 30, cfd_budget: int = 10) -> RunLog:
    """Projected gradient ascent in box-normalised coordinates with a fixed normalised step length
    (step0 x box span), CFD fallback per the gate, at most `cfd_budget` in-loop CFD calls."""
    log = RunLog()
    theta = box.project(np.asarray(theta0, dtype=float))
    pts = np.array(data_points, dtype=float).reshape(-1, len(theta))
    for k in range(n_steps):
        J_hat, g_hat, sig_g, sig_J = surrogate.predict(theta)
        gn = g_hat * box.span                      # gradient in normalised coordinates
        d = gn / (np.linalg.norm(gn) + 1e-300) * step0 * box.span   # step in physical coordinates
        call = gate.needs_cfd(theta, J_hat, g_hat, sig_g, sig_J, d, pts, box.span) and len(log.cfd_calls) < cfd_budget
        log.thetas.append(theta.copy()); log.J_hat.append(J_hat); log.gated.append(bool(call))
        if call:
            J, g = cfd(theta)
            log.cfd_calls.append((k, theta.copy(), float(J), np.asarray(g, dtype=float)))
            pts = np.vstack([pts, theta])
            gn = np.asarray(g) * box.span
            d = gn / (np.linalg.norm(gn) + 1e-300) * step0 * box.span
        theta = box.project(theta + d)
    log.thetas.append(theta.copy())
    return log
