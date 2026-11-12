"""Valley-resolved band-edge search for diamond-structure crystals.

Valleys are labelled in the unstrained cubic frame (units of 2*pi/a):
  G  = (0,0,0)
  L1..L4 = (-1,1,1)/2, (1,-1,1)/2, (1,1,-1)/2, (1,1,1)/2  (TRIMs; reduced coords (.5,0,0) ...)
  Dx, Dy, Dz = conduction-band minima on the Gamma-X lines (inversion pairs merged)
Reduced coordinates of k are invariant under homogeneous strain, so basins are defined
with the unstrained reciprocal lattice and the k-points are then used in the strained cell.
"""
from __future__ import annotations

import itertools

import numpy as np

from .materials import FCC_PRIM

# Unstrained reciprocal basis (rows) in units of 2*pi/a for the FCC_PRIM cell.
B0 = np.linalg.inv(FCC_PRIM).T

VALLEY_CENTERS = {
    "G": [(0, 0, 0)],
    "L1": [(-0.5, 0.5, 0.5), (0.5, -0.5, -0.5)],
    "L2": [(0.5, -0.5, 0.5), (-0.5, 0.5, -0.5)],
    "L3": [(0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)],
    "L4": [(0.5, 0.5, 0.5), (-0.5, -0.5, -0.5)],
    "Dx": [(0.85, 0, 0), (-0.85, 0, 0)],
    "Dy": [(0, 0.85, 0), (0, -0.85, 0)],
    "Dz": [(0, 0, 0.85), (0, 0, -0.85)],
}
VALLEYS = list(VALLEY_CENTERS)
_G_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)


def frac_to_cart0(f: np.ndarray) -> np.ndarray:
    return np.asarray(f) @ B0


def cart0_to_frac(c: np.ndarray) -> np.ndarray:
    return np.asarray(c) @ np.linalg.inv(B0)


def fold_to_bz(f: np.ndarray) -> np.ndarray:
    """Map reduced k-points to their first-Brillouin-zone image (unstrained metric)."""
    f = np.atleast_2d(f)
    cands = f[:, None, :] + _G_SHIFTS[None, :, :]
    d = np.linalg.norm(cands @ B0, axis=2)
    return cands[np.arange(len(f)), d.argmin(axis=1)]


def uniform_grid(n: int) -> np.ndarray:
    """Gamma-centred n^3 grid reduced by time reversal (k ~ -k)."""
    pts, seen = [], set()
    for i, j, k in itertools.product(range(n), repeat=3):
        key = (i, j, k)
        mkey = ((-i) % n, (-j) % n, (-k) % n)
        if mkey in seen:
            continue
        seen.add(key)
        pts.append((i / n, j / n, k / n))
    return fold_to_bz(np.array(pts))


TRIM_VALLEYS = ["G", "L1", "L2", "L3", "L4"]
DELTA_VALLEYS = ["Dx", "Dy", "Dz"]
_AXES = {"Dx": 0, "Dy": 1, "Dz": 2}


def trim_frac(v: str) -> np.ndarray:
    """Reduced coordinates of a TRIM valley (Gamma or an L point)."""
    return cart0_to_frac(np.array(VALLEY_CENTERS[v][0], dtype=float))


def delta_line(v: str, s=np.linspace(0.6, 1.0, 11)) -> np.ndarray:
    """Points on the Gamma-X line of valley v at fractions s of |Gamma X| (cubic frame)."""
    c = np.zeros((len(s), 3))
    c[:, _AXES[v]] = s
    return cart0_to_frac(c)


def delta_region(kfrac: np.ndarray, v: str, max_angle_deg: float = 30.0, kmin: float = 0.3):
    """Mask of k-points inside the cone around the +/- axis of a Delta valley."""
    c = frac_to_cart0(fold_to_bz(kfrac))
    r = np.linalg.norm(c, axis=1)
    cosang = np.abs(c[:, _AXES[v]]) / np.maximum(r, 1e-12)
    return (r > kmin) & (cosang > np.cos(np.radians(max_angle_deg)))


def stencil(center_frac: np.ndarray, step: float, m: int = 5) -> np.ndarray:
    o = (np.arange(m) - (m - 1) / 2) * step
    off = np.array(list(itertools.product(o, o, o)))
    return center_frac[None, :] + off


def quad_fit_min(kfrac: np.ndarray, e: np.ndarray, bstrained: np.ndarray):
    """Fit E = c + g.x + x.H.x/2 in Cartesian k (1/Angstrom); return min energy, location, H."""
    x = kfrac @ bstrained
    x0 = x[np.argmin(e)]
    dx = x - x0
    cols = [np.ones(len(dx))] + [dx[:, i] for i in range(3)]
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    cols += [dx[:, i] * dx[:, j] * (0.5 if i == j else 1.0) for i, j in pairs]
    a = np.array(cols).T
    coef, *_ = np.linalg.lstsq(a, e, rcond=None)
    g = coef[1:4]
    h = np.zeros((3, 3))
    for (i, j), c in zip(pairs, coef[4:]):
        h[i, j] = h[j, i] = c
    try:
        step = -np.linalg.solve(h, g)
    except np.linalg.LinAlgError:
        step = np.zeros(3)
    if np.linalg.norm(step) > np.max(np.linalg.norm(dx, axis=1)):
        return float(e.min()), x0, h, False   # minimum outside stencil: keep grid value
    emin = coef[0] + g @ step + 0.5 * step @ h @ step
    return float(min(emin, e.min())), x0 + step, h, True


HBAR2_OVER_ME = 7.619964  # eV * Angstrom^2  (hbar^2 / m_e)


def mass_eigs(h_ev_a2: np.ndarray) -> list[float]:
    """Effective-mass eigenvalues (m_e) from the band Hessian d2E/dk2 (eV A^2)."""
    w = np.linalg.eigvalsh(h_ev_a2)
    return [float(HBAR2_OVER_ME / x) if abs(x) > 1e-8 else float("inf") for x in w]
