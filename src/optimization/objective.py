"""Design objective of the H3 study (protocol §5) and the ensemble-surrogate adapter used by the optimiser.

J(theta) = L/D - W * ( max(0, CL_MIN - CL)^2 + max(0, T_MIN - t)^2 ),   W = 100, CL_MIN = 0.25, T_MIN = 0.10
(fixed before any data was generated). Its gradient follows from CL, CD and their gradients.
"""
from __future__ import annotations

import numpy as np
import torch

from models.surrogates import lift_to_drag_and_grad

W, CL_MIN, T_MIN = 100.0, 0.25, 0.10


def objective(cl: float, cd: float, dcl: np.ndarray, dcd: np.ndarray, t: float) -> tuple[float, np.ndarray]:
    ld = cl / cd
    g = (dcl * cd - cl * dcd) / cd**2
    vcl, vt = max(0.0, CL_MIN - cl), max(0.0, T_MIN - t)
    J = ld - W * (vcl**2 + vt**2)
    g = g + 2 * W * vcl * dcl
    g = g.copy(); g[2] += 2 * W * vt
    return float(J), g


class EnsembleSurrogate:
    """Adapter: list of trained models -> (J_hat, grad_hat, sigma_grad, sigma_J) at one theta, where the
    hats are ensemble means and the sigmas ensemble standard deviations (sigma_grad in the M-norm)."""

    def __init__(self, models: list, M: np.ndarray, device: str = "cpu"):
        self.models, self.M, self.device = models, M, device

    def predict(self, theta: np.ndarray):
        th = torch.tensor(np.asarray(theta, dtype=np.float32)[None], device=self.device)
        Js, gs = [], []
        for m in self.models:
            _, _, jac = lift_to_drag_and_grad(m, th)
            c = m.coefficients(th).detach().cpu().numpy()[0]
            J, g = objective(float(c[0]), float(c[1]), jac[0, 0].cpu().numpy(), jac[0, 1].cpu().numpy(), float(theta[2]))
            Js.append(J); gs.append(g)
        Js, gs = np.array(Js), np.array(gs)
        gm = gs.mean(0)
        dev = gs - gm
        sig_g = float(np.sqrt(np.mean(np.einsum("ki,ij,kj->k", dev, self.M, dev))))
        return float(Js.mean()), gm, sig_g, float(Js.std())
