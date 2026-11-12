"""Strain conventions, crystal symmetry and elastic energy density.

Conventions (fixed for the whole project):
- Strain is given in Voigt order (e11, e22, e33, e23, e13, e12) with TENSOR shear
  components (not engineering shear 2*e_ij).
- The default measure is Green-Lagrange (GL), matching Shi et al. PNAS 2024.
  The rotation-free deformation gradient is F = sqrtm(I + 2E).
  measure="linear" uses F = I + eps instead (small-strain / Biot-like).
- Lattice vectors are rows of A; a deformed crystal has rows A' = A F^T (= A F for symmetric F).
- h = (E(eps) - E0) / V0 in meV/A^3.
"""
from __future__ import annotations

import itertools
import numpy as np

RY_TO_EV = 13.605693122994
EV_PER_A3_TO_MEV_PER_A3 = 1000.0


def voigt_to_tensor(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.shape != (6,):
        raise ValueError("strain must have 6 Voigt components (e11,e22,e33,e23,e13,e12)")
    e11, e22, e33, e23, e13, e12 = v
    return np.array([[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]])


def tensor_to_voigt(t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[0, 2], t[0, 1]])


def _sym_sqrtm(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    if np.any(w <= 0):
        raise ValueError("I + 2E is not positive definite; strain is unphysical")
    return (v * np.sqrt(w)) @ v.T


def deformation_gradient(voigt, measure: str = "gl") -> np.ndarray:
    e = voigt_to_tensor(voigt)
    if measure == "gl":
        return _sym_sqrtm(np.eye(3) + 2.0 * e)
    if measure == "linear":
        return np.eye(3) + e
    raise ValueError(f"unknown strain measure {measure!r}")


def convert(voigt, src: str, dst: str) -> np.ndarray:
    """Convert a strain between 'gl' and 'linear' (rotation-free) measures."""
    if src == dst:
        return np.asarray(voigt, dtype=float)
    f = deformation_gradient(voigt, src)
    if dst == "gl":
        return tensor_to_voigt(0.5 * (f.T @ f - np.eye(3)))
    if dst == "linear":
        return tensor_to_voigt(f - np.eye(3))
    raise ValueError(dst)


def strained_cell(cell0: np.ndarray, voigt, measure: str = "gl") -> np.ndarray:
    f = deformation_gradient(voigt, measure)
    return np.asarray(cell0) @ f.T


def oh_operations() -> list[np.ndarray]:
    """The 48 proper and improper rotations of the cubic point group O_h (Cartesian)."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                r[i, j] = s
            ops.append(r)
    return ops


def rotate_strain(voigt, r: np.ndarray) -> np.ndarray:
    return tensor_to_voigt(r @ voigt_to_tensor(voigt) @ r.T)


def h_from_energies(e_ry: float, e0_ry: float, v0_a3: float) -> float:
    """Elastic strain energy density in meV/A^3 from total energies in Ry (per cell)."""
    return (e_ry - e0_ry) * RY_TO_EV / v0_a3 * EV_PER_A3_TO_MEV_PER_A3


def strain_norm(voigt) -> float:
    """Euclidean norm with tensor shear counted twice (PNAS 2024, Eq. 6)."""
    v = np.asarray(voigt, dtype=float)
    return float(np.sqrt(np.sum(v[:3] ** 2) + 2.0 * np.sum(v[3:] ** 2)))
