"""Differentiable NACA 4-digit geometry and grid encodings in PyTorch (AeroForge-TRIAD).

Mirrors cfd.shape_sensitivity.surface_point exactly (same closed-TE thickness polynomial, same camber
formula for every m, same nose-up rotation about (0.25, 0)), so that surrogate design gradients are taken
with respect to the same design variables theta = (m, p, t, alpha_deg) as the CFD adjoint gradients.
"""
from __future__ import annotations

import math

import torch

A4_CLOSED_TE = -0.1036


def surface_points(theta: torch.Tensor, n_beta: int = 257) -> torch.Tensor:
    """theta (B, 4) NACA or (B, 2n+1) CST -> closed surface polyline (B, 2*n_beta - 1, 2):
    upper LE->TE then lower TE->LE."""
    if theta.shape[1] != 4:
        return _surface_points_cst(theta, n_beta)
    m, p, t, aoa = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4]
    beta = torch.linspace(0.0, math.pi, n_beta, dtype=theta.dtype, device=theta.device)[None, :]
    x = 0.5 * (1.0 - torch.cos(beta))
    yt = 5.0 * t * (0.2969 * torch.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + A4_CLOSED_TE * x**4)
    front = x < p
    yc = torch.where(front, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = torch.where(front, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = torch.atan(dyc)
    xu, yu = x - yt * torch.sin(th), yc + yt * torch.cos(th)
    xl, yl = x + yt * torch.sin(th), yc - yt * torch.cos(th)
    xs = torch.cat([xu, xl.flip(-1)[:, 1:]], dim=1)
    ys = torch.cat([yu, yl.flip(-1)[:, 1:]], dim=1)
    a = torch.deg2rad(aoa)
    c, s = torch.cos(a), torch.sin(a)
    xr = (xs - 0.25) * c + ys * s + 0.25
    yr = -(xs - 0.25) * s + ys * c
    return torch.stack([xr, yr], dim=-1)


def _surface_points_cst(theta: torch.Tensor, n_beta: int) -> torch.Tensor:
    """CST (geometry.cst): y = sqrt(x)(1-x) * sum_i A_i B_i^n(x), x fixed by beta; mirrors
    cfd.shape_sensitivity._surface_point_cst."""
    k = (theta.shape[1] - 1) // 2
    au, al, aoa = theta[:, :k], theta[:, k:2 * k], theta[:, 2 * k:2 * k + 1]
    beta = torch.linspace(0.0, math.pi, n_beta, dtype=theta.dtype, device=theta.device)
    x = 0.5 * (1.0 - torch.cos(beta))
    n = k - 1
    i = torch.arange(k, dtype=theta.dtype, device=theta.device)
    binom = torch.tensor([math.comb(n, j) for j in range(k)], dtype=theta.dtype, device=theta.device)
    B = binom[None, :] * x[:, None] ** i[None, :] * (1.0 - x[:, None]) ** (n - i)[None, :]   # (n_beta, k)
    C = torch.sqrt(x) * (1.0 - x)
    yu, yl = C[None, :] * (au @ B.T), C[None, :] * (al @ B.T)
    xb = x[None, :].expand(theta.shape[0], -1)
    xs = torch.cat([xb, xb.flip(-1)[:, 1:]], dim=1)
    ys = torch.cat([yu, yl.flip(-1)[:, 1:]], dim=1)
    a = torch.deg2rad(aoa)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([(xs - 0.25) * c + ys * s + 0.25, -(xs - 0.25) * s + ys * c], dim=-1)


def segment_distance(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Unsigned distance from grid points (G, 2) to each polyline (B, P, 2) -> (B, G).
    Exact point-to-segment distance; differentiable w.r.t. the polyline almost everywhere."""
    a = poly[:, :-1, :]                     # (B, S, 2)
    b = poly[:, 1:, :]
    ab = b - a
    ap = pts[None, :, None, :] - a[:, None, :, :]      # (B, G, S, 2)
    denom = (ab * ab).sum(-1).clamp_min(1e-14)[:, None, :]
    tpar = ((ap * ab[:, None, :, :]).sum(-1) / denom).clamp(0.0, 1.0)
    proj = a[:, None, :, :] + tpar[..., None] * ab[:, None, :, :]
    d2 = ((pts[None, :, None, :] - proj) ** 2).sum(-1)  # (B, G, S)
    return torch.sqrt(d2.min(dim=-1).values + 1e-16)


def inside_polygon(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Even-odd rule, no gradient. pts (G, 2), poly (B, P, 2) closed (first point == last point not required)."""
    with torch.no_grad():
        x, y = pts[:, 0][None, :, None], pts[:, 1][None, :, None]
        x1, y1 = poly[:, :-1, 0][:, None, :], poly[:, :-1, 1][:, None, :]
        x2, y2 = poly[:, 1:, 0][:, None, :], poly[:, 1:, 1][:, None, :]
        cond = (y1 > y) != (y2 > y)
        xint = x1 + (y - y1) * (x2 - x1) / torch.where((y2 - y1).abs() < 1e-14, torch.full_like(y2, 1e-14), y2 - y1)
        cross = cond & (x < xint)
        return (cross.sum(-1) % 2) == 1


def sdf_grid(theta: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor, n_beta: int = 257, chunk: int = 4096) -> torch.Tensor:
    """Signed distance (negative inside) on a grid, (B, ny, nx), differentiable w.r.t. theta."""
    poly = surface_points(theta, n_beta)
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(theta)
    out = []
    for i in range(0, pts.shape[0], chunk):
        p = pts[i: i + chunk]
        d = segment_distance(p, poly)
        ins = inside_polygon(p, poly)
        out.append(torch.where(ins, -d, d))
    return torch.cat(out, dim=1).reshape(theta.shape[0], *gx.shape)
