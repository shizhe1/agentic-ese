"""Per-valley HSE06 corrections.

For a strain eps, HSE06 (nq=4) and PBE are run on the same Gamma-centred 8^3 grid without
symmetry reduction (time reversal kept). The 8^3 grid contains Gamma, the L points, X and the
0.75X points on each Gamma-X line, so a correction is defined per valley family:
    D_v(eps) = [E_HSE(k_v) - VBM_HSE] - [E_PBE(k_v) - VBM_PBE],
with k_G = Gamma, k_Li = L_i and k_Dx/y/z = the 0.75X point of that axis (closest grid point
to the Delta minimum). Corrected valley energies are PBE valley minima + D_v.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import bands, qe
from .dft import CACHE, SCRATCH
from .materials import MATERIALS, Material
from .strain import strained_cell

HSE_VERSION = "hse-v1"
NQ, NK = 4, 8
VALLEY_K = {"G": (0, 0, 0)}
VALLEY_K.update({v: tuple(bands.trim_frac(v)) for v in ("L1", "L2", "L3", "L4")})
VALLEY_K.update({v: tuple(bands.delta_line(v, np.array([0.75]))[0]) for v in bands.DELTA_VALLEYS})
STRICT = dict(nk=8, nq=4, fock=2.0, conv=1e-9)          # settings used for all agent verify_hse calls


def _valley_k(nk):
    k = dict(VALLEY_K)
    s = 0.75 if nk == 8 else 4.0 / nk                    # Delta-line grid point nearest the minimum
    k.update({v: tuple(bands.delta_line(v, np.array([s]))[0]) for v in bands.DELTA_VALLEYS})
    return k


def _key(mat: Material, strain, settings=None) -> str:
    payload = dict(mat=mat.name, a0=mat.a0, ecut=mat.ecutwfc, v=HSE_VERSION,
                   strain=[round(float(x), 7) for x in strain])
    if settings and settings != STRICT:
        payload["settings"] = settings
    return "hse_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _valley_energies(res, nocc, vk=None):
    kf = res.kpts_frac
    vbm = float(res.eigs_ev[:, nocc - 1].max())
    out = {}
    for v, k in (vk or VALLEY_K).items():
        k = np.asarray(k)
        d = np.min([np.linalg.norm((kf - s * k) - np.round(kf - s * k), axis=1) for s in (1, -1)], axis=0)
        i = int(np.argmin(d))
        if d[i] > 1e-4:
            raise RuntimeError(f"k-point for {v} not on grid")
        out[v] = float(res.eigs_ev[i, nocc]) - vbm
    return out, vbm


def hse_correction(mat: Material | str, strain, positions_frac=None, settings: dict | None = None) -> dict:
    if isinstance(mat, str):
        mat = MATERIALS[mat]
    strain = np.asarray(strain, dtype=float)
    st = dict(STRICT, **(settings or {}))
    key = _key(mat, strain, st)
    out = CACHE / f"{key}.json"
    if out.exists():
        return json.loads(out.read_text())
    from .dft import compute_point
    pbe = compute_point(mat, strain)            # supplies relaxed positions and PBE valleys
    pos = np.asarray(positions_frac if positions_frac is not None else pbe["positions_frac"])
    cell = strained_cell(mat.cell0(), strain, "gl")
    work = SCRATCH / key
    shutil.rmtree(work, ignore_errors=True)
    common = dict(prefix="x", outdir="./", cell_a=cell, species=mat.species(), symbols=mat.symbols(),
                  positions_frac=pos, ecutwfc=mat.ecutwfc, kmesh=(st["nk"],) * 3, nbnd=8, nosym=True,
                  tstress=False, smearing=0.002, conv_thr=st["conv"])
    rp = qe.run_pw(qe.write_input(calculation="scf", **common), work / "pbe", "x")
    rh = qe.run_pw(qe.write_input(calculation="scf", hybrid=dict(nq=st["nq"], ecutfock=st["fock"] * mat.ecutwfc),
                                  **common), work / "hse", "x")
    nocc = int(round(rp.nelec / 2))
    vk = _valley_k(st["nk"])
    ep, _ = _valley_energies(rp, nocc, vk)
    eh, _ = _valley_energies(rh, nocc, vk)
    corr = {v: eh[v] - ep[v] for v in ep}
    valleys = {v: pbe["valleys"][v]["e"] + corr[v] for v in corr}
    cbm = min(valleys, key=valleys.get)
    rec = dict(material=mat.name, key=key, strain_gl=strain.tolist(), corrections=corr,
               valleys_hse=valleys, eg_hse=float(valleys[cbm]), cbm_hse=cbm,
               h_mev_a3=pbe["h_mev_a3"], pbe_key=pbe["key"], settings=st, wall_s=rp.wall_s + rh.wall_s)
    out.write_text(json.dumps(rec))
    shutil.rmtree(work, ignore_errors=True)
    return rec
