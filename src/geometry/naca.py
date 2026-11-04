"""NACA 4-digit airfoil geometry (AeroForge-TRIAD).

Conventions
-----------
* Chord c = 1, leading edge at (0, 0), trailing edge at (1, 0) before rotation.
* m: maximum camber (fraction of chord), p: position of max camber (fraction of chord),
  t: maximum thickness (fraction of chord). NACA 2412 -> m=0.02, p=0.4, t=0.12.
* Closed (sharp) trailing edge: a4 = -0.1036 instead of the original -0.1015, so the
  thickness is exactly zero at x = 1 (0.2969 - 0.1260 - 0.3516 + 0.2843 - 0.1036 = 0).
* Cosine spacing clusters points at the leading and trailing edges.
* Angle of attack is applied by rotating the geometry nose-up about the quarter chord,
  with the free stream fixed along +x. Lift is then +y force and drag is +x force.
"""
from __future__ import annotations

import numpy as np

A4_CLOSED_TE = -0.1036
A4_OPEN_TE = -0.1015


def thickness(x: np.ndarray, t: float, closed_te: bool = True) -> np.ndarray:
    a4 = A4_CLOSED_TE if closed_te else A4_OPEN_TE
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)


def camber(x: np.ndarray, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean camber line y_c(x) and its slope dy_c/dx."""
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    front = x < p
    yc = np.where(front, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(front, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, dyc


def naca4(m: float, p: float, t: float, n: int = 201, closed_te: bool = True):
    """Return (upper, lower) surfaces, each ordered from the leading edge (index 0) to the
    trailing edge (index -1). Both share the same LE and TE points when closed_te=True."""
    if not (0.0 <= m < 0.1 and 0.0 <= p < 1.0 and 0.0 < t < 0.4):
        raise ValueError(f"parameters out of supported range: m={m}, p={p}, t={t}")
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = thickness(x, t, closed_te)
    yc, dyc = camber(x, m, p)
    th = np.arctan(dyc)
    upper = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    lower = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])
    return upper, lower


def rotate(points: np.ndarray, aoa_deg: float, pivot=(0.25, 0.0)) -> np.ndarray:
    """Rotate nose-up by aoa_deg about the pivot (free stream along +x)."""
    a = np.deg2rad(aoa_deg)
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, s], [-s, c]])
    piv = np.asarray(pivot, dtype=float)
    return (points - piv) @ rot.T + piv


def geometry_checks(upper: np.ndarray, lower: np.ndarray, t: float) -> dict:
    """Basic validity checks used before meshing."""
    thick = upper[:, 1] - lower[:, 1]
    return {
        "le_match": bool(np.allclose(upper[0], lower[0])),
        "te_match": bool(np.allclose(upper[-1], lower[-1], atol=1e-12)),
        "min_thickness_interior": float(thick[1:-1].min()),
        "max_thickness": float(thick.max()),
        "max_thickness_rel_error": float(abs(thick.max() - t) / t),
        "self_intersection_free": bool((thick[1:-1] > 0).all()),
    }
