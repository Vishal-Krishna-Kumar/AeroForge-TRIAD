"""Surrogate models for Dataset V1 (AeroForge-TRIAD).

* FNOSurrogate: 2D Fourier neural operator (Li et al., ICLR 2021) on the fixed wind-frame grid. Input
  channels: signed distance, fluid mask, x, y. Outputs: pressure and velocity fields (masked to the
  fluid) plus CL and CD from a pooled head. Design gradients come from autograd through the
  differentiable SDF (models.geometry_torch), i.e. d(output)/d(theta) for theta = (m, p, t, alpha_deg).
* MLPSurrogate: theta -> (CL, CD), the parametric baseline.
Both expose `coefficients(theta)` returning (B, 2) [CL, CD] so they are interchangeable in the
optimiser and in the gradient-certification code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry_torch import sdf_grid


class SpectralConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, m1: int, m2: int):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, : self.m1, : self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, : self.m1, : self.m2], self.w1)
        out[:, :, -self.m1:, : self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, : self.m2], self.w2)
        return torch.fft.irfft2(out, s=(h, w))


class FNO2d(nn.Module):
    def __init__(self, cin: int, cout: int, width: int = 32, modes: tuple[int, int] = (16, 24), layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(cin, width, 1)
        self.spec = nn.ModuleList(SpectralConv2d(width, width, *modes) for _ in range(layers))
        self.skip = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, cout, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.lift(x)
        for k, (s, w) in enumerate(zip(self.spec, self.skip)):
            h = s(h) + w(h)
            if k < len(self.spec) - 1:
                h = F.gelu(h)
        return self.proj(h), h


class FNOSurrogate(nn.Module):
    """Inputs are built from theta inside forward(), so autograd gives d(outputs)/d(theta)."""

    def __init__(self, gx: torch.Tensor, gy: torch.Tensor, width: int = 32, modes=(16, 24), layers: int = 4,
                 sdf_scale: float = 0.1, n_theta: int = 4):
        super().__init__()
        self.register_buffer("gx", gx.float())
        self.register_buffer("gy", gy.float())
        self.sdf_scale = sdf_scale
        self.net = FNO2d(4, 3, width, modes, layers)
        self.head = nn.Sequential(nn.Linear(2 * width, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))
        self.register_buffer("theta_mu", torch.zeros(n_theta))   # n_theta = 4 (V1 NACA) or 13 (V2 CST)
        self.register_buffer("theta_sd", torch.ones(n_theta))
        self.register_buffer("y_mu", torch.zeros(2))
        self.register_buffer("y_sd", torch.ones(2))
        self.register_buffer("field_mu", torch.zeros(3))  # (p, u, v) standardisation, set by training
        self.register_buffer("field_sd", torch.ones(3))

    def physical_fields(self, theta: torch.Tensor, sdf: torch.Tensor | None = None) -> torch.Tensor:
        """(p, u, v) in solver units on the grid, zero inside the body."""
        f, _ = self.forward(theta, sdf)
        fluid = (self.encode(theta, sdf)[:, 1:2] > 0.5).float()
        return (f * self.field_sd[None, :, None, None] + self.field_mu[None, :, None, None]) * fluid

    def encode(self, theta: torch.Tensor, sdf: torch.Tensor | None = None) -> torch.Tensor:
        if sdf is None:
            sdf = sdf_grid(theta, self.gx, self.gy)
        b = theta.shape[0]
        fluid = torch.sigmoid(sdf / 2e-3)  # smooth mask so the input stays differentiable in theta
        xy = torch.stack([self.gx, self.gy])[None].expand(b, -1, -1, -1)
        return torch.cat([torch.tanh(sdf / self.sdf_scale)[:, None], fluid[:, None], xy], dim=1)

    def forward(self, theta: torch.Tensor, sdf: torch.Tensor | None = None):
        x = self.encode(theta, sdf)
        fields, h = self.net(x)
        fluid = x[:, 1:2]
        # pure operator model: the coefficient head sees only pooled features of the geometry-driven
        # field representation (no direct theta input), so its design gradients flow through the SDF
        pooled = torch.cat([(h * fluid).mean((2, 3)), (h * fluid).amax((2, 3))], dim=1)
        coeff = self.head(pooled) * self.y_sd + self.y_mu
        return fields * fluid, coeff

    def coefficients(self, theta: torch.Tensor) -> torch.Tensor:
        return self.forward(theta)[1]


class MLPSurrogate(nn.Module):
    def __init__(self, width: int = 128, depth: int = 4, n_theta: int = 4):
        super().__init__()
        layers, d = [], n_theta
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
        self.register_buffer("theta_mu", torch.zeros(n_theta))
        self.register_buffer("theta_sd", torch.ones(n_theta))
        self.register_buffer("y_mu", torch.zeros(2))
        self.register_buffer("y_sd", torch.ones(2))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net((theta - self.theta_mu) / self.theta_sd) * self.y_sd + self.y_mu

    def coefficients(self, theta: torch.Tensor) -> torch.Tensor:
        return self.forward(theta)


def lift_to_drag_and_grad(model: nn.Module, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L/D, its gradient w.r.t. theta, and the (B, 2, 4) Jacobian of (CL, CD), all by autograd."""
    th = theta.detach().clone().requires_grad_(True)
    c = model.coefficients(th)
    jac = []
    for k in range(2):
        g, = torch.autograd.grad(c[:, k].sum(), th, retain_graph=True)
        jac.append(g)
    jac = torch.stack(jac, dim=1)
    cl, cd = c[:, 0], c[:, 1]
    ld = cl / cd
    gld = (jac[:, 0] * cd[:, None] - cl[:, None] * jac[:, 1]) / cd[:, None] ** 2
    return ld.detach(), gld.detach(), jac.detach()
