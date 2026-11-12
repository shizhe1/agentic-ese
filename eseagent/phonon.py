"""Phonon (DFPT) and elastic (Born) stability checks for strained 2-atom cells."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import qe
from .materials import MATERIALS, Material
from .strain import strained_cell

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "runs" / "phonon"


def _run(exe: str, inp: str, work: Path, timeout: float = 4 * 3600) -> None:
    (work / f"{exe}.in").write_text(inp)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR="/tmp")
    with open(work / f"{exe}.out", "w") as fo:
        p = subprocess.run([str(qe.ENV_BIN / f"{exe}.x"), "-in", f"{exe}.in"], cwd=work,
                           stdout=fo, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{exe}.x failed in {work}: " + (work / f"{exe}.out").read_text()[-1500:])


def phonon_check(mat: Material | str, strain, positions_frac=None, nq: int = 4,
                 dense: int = 16, tag: str | None = None, workers: int = 10) -> dict:
    """DFPT on an nq^3 grid, Fourier-interpolated to a dense mesh; returns the lowest frequency.

    Stable if the lowest interpolated frequency (excluding the three Gamma acoustic modes)
    exceeds -5 cm^-1 (numerical noise of the acoustic sum rule)."""
    if isinstance(mat, str):
        mat = MATERIALS[mat]
    strain = np.asarray(strain, dtype=float)
    work = SCRATCH / (tag or f"{mat.name}_{abs(hash(tuple(np.round(strain, 7)))) % 10**10}_q{nq}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    t0 = time.time()
    cell = strained_cell(mat.cell0(), strain, "gl")
    pos = positions_frac if positions_frac is not None else np.array([[0, 0, 0], [.25, .25, .25]])
    qe.run_pw(qe.write_input(calculation="scf", prefix="x", outdir="./", cell_a=cell,
                             species=mat.species(), symbols=mat.symbols(), positions_frac=pos,
                             ecutwfc=mat.ecutwfc, kmesh=(mat.kmesh,) * 3, nbnd=8, smearing=None,
                             conv_thr=1e-12), work, "x", keep=True)
    # q-points split across processes: each worker copies the ground state and runs start_q..last_q
    _run("ph", f"ph\n&inputph\n prefix='x', outdir='./', fildyn='dyn', tr2_ph=1.0d-15,\n"
               f" ldisp=.true., nq1={nq}, nq2={nq}, nq3={nq}, start_q=1, last_q=1\n/\n", work)
    nirr_q = int(re.search(r"\(\s*(\d+)\s*q-points\)", (work / "ph.out").read_text()).group(1))
    from concurrent.futures import ThreadPoolExecutor

    def one(iq):
        w = work / f"q{iq}"
        w.mkdir()
        shutil.copytree(work / "x.save", w / "x.save")
        shutil.copy(work / "x.xml", w / "x.xml") if (work / "x.xml").exists() else None
        _run("ph", f"ph\n&inputph\n prefix='x', outdir='./', fildyn='dyn', tr2_ph=1.0d-15,\n"
                   f" ldisp=.true., nq1={nq}, nq2={nq}, nq3={nq}, start_q={iq}, last_q={iq}\n/\n", w)
        shutil.copy(w / f"dyn{iq}", work / f"dyn{iq}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(2, nirr_q + 1)))
    _run("q2r", "&input\n fildyn='dyn', zasr='crystal', flfrc='fc'\n/\n", work)
    _run("matdyn", f"&input\n asr='crystal', flfrc='fc', flfrq='freq', dos=.true.,\n"
                   f" nk1={dense}, nk2={dense}, nk3={dense}, fldos='dos'\n/\n", work)
    data = np.loadtxt(str(work / "freq") + ".gp")
    freqs, gam = data[:, 1:], data[:, 0] == 0         # Gamma rows: zero path distance
    fg = freqs[gam].ravel()
    rest = np.concatenate([fg[np.argsort(np.abs(fg))[3:]], freqs[~gam].ravel()])   # drop only the 3 near-zero translations at Gamma
    omin = float(rest.min())
    return dict(material=mat.name, strain_gl=strain.tolist(), nq=nq, dense=dense,
                omega_min_cm1=omin, omega_min_thz=omin / 33.35641, stable=bool(omin > -5.0),
                wall_s=time.time() - t0, workdir=str(work))


def _read_freq(path: Path) -> np.ndarray:
    """Frequencies (cm^-1) from matdyn's gnuplot file: k-distance followed by nbnd columns."""
    data = np.loadtxt(str(path) + ".gp")
    return data[:, 1:]


def born_stable(c: np.ndarray) -> dict:
    """Elastic (Born) stability of a 6x6 Voigt stiffness matrix at the strained state."""
    w = np.linalg.eigvalsh(0.5 * (c + c.T))
    return dict(min_eig_gpa=float(w.min()), stable=bool(w.min() > 0))
