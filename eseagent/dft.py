"""Per-strain first-principles evaluation with a content-addressed cache.

compute_point(material, strain) runs (1) a fixed-cell relaxation of the internal coordinates,
(2) an NSCF on a time-reversal-reduced uniform grid, (3) an NSCF on local stencils around each
valley's coarse minimum, and returns valley-resolved band edges, the gap, h and the stress.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import bands, qe
from .materials import DIAMOND_POS, MATERIALS, Material
from .strain import convert, h_from_energies, strained_cell

CODE_VERSION = "pbe-v3"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "cache"
SCRATCH = ROOT / "runs" / "scratch"
NGRID = 6


def _key(mat: Material, strain, level: str) -> str:
    payload = dict(mat=mat.name, a0=mat.a0, ecut=mat.ecutwfc, k=mat.kmesh, pp=mat.pseudo,
                   level=level, ngrid=NGRID, code=CODE_VERSION,
                   strain=[round(float(x), 7) for x in strain])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def cached(mat: Material, strain, level: str = "pbe"):
    p = CACHE / f"{_key(mat, strain, level)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def reference_energy(mat: Material) -> float:
    rec = compute_point(mat, np.zeros(6), need_e0=False)
    return rec["energy_ry"]


def compute_point(mat: Material | str, strain, level: str = "pbe", need_e0: bool = True,
                  keep_raw: bool = True) -> dict:
    if isinstance(mat, str):
        mat = MATERIALS[mat]
    strain = np.asarray(strain, dtype=float)
    key = _key(mat, strain, level)
    out = CACHE / f"{key}.json"
    if out.exists():
        return json.loads(out.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    work = SCRATCH / key
    if work.exists():
        shutil.rmtree(work)
    t0 = time.time()
    cell = strained_cell(mat.cell0(), strain, "gl")
    common = dict(prefix="x", outdir="./", species=mat.species(), symbols=mat.symbols(),
                  ecutwfc=mat.ecutwfc, nbnd=8)
    relax, retry = None, 0
    for conv, smear in ((1e-8, 0.005), (1e-7, 0.02)):     # retry looser if SCF stalls
        try:
            relax = qe.run_pw(qe.write_input(calculation="relax", cell_a=cell, positions_frac=DIAMOND_POS,
                                             kmesh=(mat.kmesh,) * 3, forc_conv_thr=1e-4, conv_thr=conv,
                                             smearing=smear, **common), work, "x", keep=True, timeout=900)
            break
        except (RuntimeError, __import__("subprocess").TimeoutExpired):
            retry += 1
            shutil.rmtree(work, ignore_errors=True)
    if relax is None:
        raise RuntimeError("SCF did not converge after retries")
    pos = relax.positions_frac
    bstr = 2 * np.pi * np.linalg.inv(relax.cell_a).T          # strained reciprocal rows (1/A)
    nscf_kw = dict(common, nbnd=6, conv_thr=1e-9)
    # Stage 1: coarse uniform grid + the three Gamma-X lines
    lines = np.vstack([bands.delta_line(v) for v in bands.DELTA_VALLEYS])
    grid = np.vstack([bands.uniform_grid(NGRID), lines])
    nscf1 = qe.run_pw(qe.write_input(calculation="nscf", cell_a=cell, positions_frac=pos,
                                     kpts_frac=grid, **nscf_kw), work, "x", keep=True)
    e1 = nscf1.eigs_ev
    nocc = int(round(nscf1.nelec / 2))
    vb, cb = e1[:, nocc - 1], e1[:, nocc]
    # Stage 2: stencils. TRIM valleys are evaluated AT the TRIM (a critical point by symmetry);
    # Delta valleys are refined around their coarse minimum inside a 30-degree cone.
    blocks = {v: bands.stencil(bands.trim_frac(v), 0.01, m=1) for v in bands.TRIM_VALLEYS}
    for v in bands.DELTA_VALLEYS:
        mask = bands.delta_region(grid, v)
        k0 = grid[mask][np.argmin(cb[mask])]
        blocks[v] = bands.stencil(k0, 0.02, m=5)
    names = list(blocks)
    kp2 = np.vstack([blocks[v] for v in names])
    nscf2 = qe.run_pw(qe.write_input(calculation="nscf", cell_a=cell, positions_frac=pos,
                                     kpts_frac=kp2, **nscf_kw), work, "x", keep=False)
    e2 = nscf2.eigs_ev
    valleys, warnings, start = {}, [], 0
    for v in names:
        n = len(blocks[v])
        sl = slice(start, start + n)
        start += n
        if v in bands.TRIM_VALLEYS:
            hess = None
            emin, kf, ok = float(e2[sl, nocc][0]), kp2[sl][0], True
        else:
            emin, xmin, hess, ok = bands.quad_fit_min(kp2[sl], e2[sl, nocc], bstr)
            kf = xmin @ np.linalg.inv(bstr)
            if not ok:
                warnings.append(f"{v}: minimum outside stencil")
        valleys[v] = dict(e_abs=emin, k_frac=np.asarray(kf).tolist(),
                          masses=bands.mass_eigs(hess) if hess is not None else None, fit_ok=bool(ok))
    g = slice(0, len(blocks["G"]))
    vbm = float(e2[g, nocc - 1].max())
    vbm_k = kp2[g][np.argmax(e2[g, nocc - 1])]
    if vb.max() > vbm + 1e-3:
        vbm = float(vb.max())
        vbm_k = grid[np.argmax(vb)]
        warnings.append(f"VBM off Gamma at {np.round(vbm_k, 3).tolist()}")
    for v in valleys:
        valleys[v]["e"] = valleys[v]["e_abs"] - vbm
    cbm_valley = min(valleys, key=lambda v: valleys[v]["e"])
    grid_cbm = float(cb.min())
    if grid_cbm < valleys[cbm_valley]["e_abs"] - 1e-3:
        warnings.append("grid CBM below all refined valley minima")
    eg = min(valleys[cbm_valley]["e"], grid_cbm - vbm)
    rec = dict(material=mat.name, level=level, key=key, strain_gl=strain.tolist(),
               strain_linear=convert(strain, "gl", "linear").tolist(), cell_a=cell.tolist(),
               positions_frac=pos.tolist(), energy_ry=relax.energy_ry,
               stress_kbar=relax.stress_kbar.tolist(), volume_a3=float(abs(np.linalg.det(cell))),
               vbm_ev=vbm, vbm_k=vbm_k.tolist(), valleys=valleys, eg_ev=float(eg),
               cbm_valley=cbm_valley, direct=bool(cbm_valley == "G"), warnings=warnings,
               converged=bool(relax.converged), scf_retry=retry, wall_s=time.time() - t0)
    if need_e0:
        e0 = reference_energy(mat)
        v0 = abs(np.linalg.det(mat.cell0()))
        rec["h_mev_a3"] = h_from_energies(rec["energy_ry"], e0, v0)
    else:
        rec["h_mev_a3"] = 0.0
    if keep_raw:
        np.savez_compressed(CACHE / f"{key}_raw.npz", grid=grid, e_grid=e1, k_stencil=kp2,
                            e_stencil=e2)
    out.write_text(json.dumps(rec))
    shutil.rmtree(work, ignore_errors=True)
    return rec
