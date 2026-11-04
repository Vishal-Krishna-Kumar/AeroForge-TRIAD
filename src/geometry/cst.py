"""CST (class-shape transformation, Kulfan 2008) airfoil geometry for Dataset V2 (AeroForge-TRIAD).

Conventions match geometry.naca: chord 1, LE (0, 0), TE (1, 0), closed (sharp) trailing edge, cosine spacing,
surfaces ordered LE -> TE, angle of attack applied later by geometry.naca.rotate.

y_s(x) = C(x) * sum_i A_s,i * B_i^n(x),  C(x) = sqrt(x) * (1 - x),  B_i^n Bernstein polynomials of order n.
Design vector for V2: theta = (A_u[0..n], A_l[0..n], aoa_deg). Lower coefficients are usually negative.
Unlike NACA 4-digit, thickness is added vertically, so a surface node keeps its x under a design change:
the parametric coordinate beta (x = (1 - cos beta) / 2) moves only in y.
"""
from __future__ import annotations

from math import comb

import numpy as np


def bernstein(x: np.ndarray, n: int) -> np.ndarray:
    """(len(x), n+1) matrix of Bernstein basis polynomials B_i^n(x)."""
    x = np.asarray(x, dtype=float)[:, None]
    i = np.arange(n + 1)[None, :]
    return np.array([comb(n, k) for k in range(n + 1)])[None, :] * x**i * (1.0 - x) ** (n - i)


def class_fn(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(x, 0.0, None)) * (1.0 - x)


def surface_y(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    coef = np.asarray(coef, dtype=float)
    return class_fn(x) * (bernstein(x, len(coef) - 1) @ coef)


def cst_surfaces(au, al, n: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """(upper, lower), each (n, 2), LE -> TE, cosine spacing; shared LE and TE points."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    return np.column_stack([x, surface_y(x, au)]), np.column_stack([x, surface_y(x, al)])


def split_theta(theta, n_coef: int) -> tuple[np.ndarray, np.ndarray, float]:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2 * n_coef + 1,):
        raise ValueError(f"theta must have {2 * n_coef + 1} entries, got {theta.shape}")
    return theta[:n_coef], theta[n_coef:2 * n_coef], float(theta[-1])


def fit_cst(upper: np.ndarray, lower: np.ndarray, n_coef: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares CST coefficients of given surfaces (x must be in [0, 1]); returns (au, al, max_abs_error).
    End points are excluded because C(x) vanishes there."""
    out, err = [], 0.0
    for s in (upper, lower):
        x, y = s[1:-1, 0], s[1:-1, 1]
        A = class_fn(x)[:, None] * bernstein(x, n_coef - 1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append(c)
        err = max(err, float(np.max(np.abs(A @ c - y))))
    return out[0], out[1], err


def shape_metrics(au, al, n: int = 2001) -> dict:
    """Thickness and camber of the unrotated section (vertical measures, which is what CST controls)."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    yu, yl = surface_y(x, au), surface_y(x, al)
    th, cb = yu - yl, 0.5 * (yu + yl)
    inner = slice(1, -1)
    return {"t_max": float(th.max()), "x_t_max": float(x[np.argmax(th)]),
            "camber_max": float(cb.max()), "camber_min": float(cb.min()), "x_camber_max": float(x[np.argmax(cb)]),
            "min_interior_thickness": float(th[inner].min()),
            "self_intersection_free": bool((th[inner] > 0).all())}
