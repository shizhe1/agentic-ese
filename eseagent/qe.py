"""Minimal Quantum ESPRESSO (pw.x) driver: input writer, runner and XML parser."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HA_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903
# Directory holding pw.x / ph.x: $ESE_QE_BIN, else wherever pw.x is on PATH.
ENV_BIN = Path(os.environ.get("ESE_QE_BIN") or os.path.dirname(shutil.which("pw.x") or "") or ".")
PSEUDO_DIR = Path(__file__).resolve().parent.parent / "pseudo"


@dataclass
class PWResult:
    energy_ry: float
    cell_a: np.ndarray            # rows, Angstrom
    positions_frac: np.ndarray
    forces_ry_bohr: np.ndarray | None
    stress_kbar: np.ndarray | None
    kpts_frac: np.ndarray         # (nk, 3) reduced coordinates of the reciprocal basis
    eigs_ev: np.ndarray           # (nk, nbnd)
    nelec: float
    wall_s: float
    converged: bool
    extra: dict = field(default_factory=dict)


def _fmt_block(rows) -> str:
    return "\n".join("  " + " ".join(f"{x:.12f}" for x in r) for r in rows)


def write_input(*, calculation: str, prefix: str, outdir: str, cell_a: np.ndarray,
                species: list[tuple[str, float, str]], symbols: list[str],
                positions_frac: np.ndarray, ecutwfc: float, kmesh=None, kpts_frac=None,
                nbnd: int | None = None, smearing: float | None = 0.005,
                conv_thr: float = 1e-10, forc_conv_thr: float = 1e-5, nosym: bool = False,
                tstress: bool = True, hybrid: dict | None = None, extra_system: dict | None = None,
                pseudo_dir: Path = PSEUDO_DIR) -> str:
    system = {"ibrav": 0, "nat": len(symbols), "ntyp": len(species), "ecutwfc": ecutwfc}
    if nbnd:
        system["nbnd"] = nbnd
    if smearing is not None:
        system.update(occupations="'smearing'", smearing="'mv'", degauss=smearing)
    else:
        system["occupations"] = "'fixed'"
    if nosym:
        system["nosym"] = ".true."
    if hybrid:
        system.update(input_dft="'hse'", nqx1=hybrid["nq"], nqx2=hybrid["nq"], nqx3=hybrid["nq"],
                      exxdiv_treatment="'gygi-baldereschi'", x_gamma_extrapolation=".true.")
        if "ecutfock" in hybrid:
            system["ecutfock"] = hybrid["ecutfock"]
    if extra_system:
        system.update(extra_system)
    control = {"calculation": f"'{calculation}'", "prefix": f"'{prefix}'", "outdir": f"'{outdir}'",
               "pseudo_dir": f"'{pseudo_dir}'", "tprnfor": ".true.",
               "tstress": ".true." if tstress else ".false.", "forc_conv_thr": forc_conv_thr,
               "disk_io": "'low'" if calculation != "nscf" else "'low'"}
    electrons = {"conv_thr": conv_thr, "mixing_beta": 0.3, "mixing_mode": "'local-TF'",
                 "electron_maxstep": 300}
    if calculation == "nscf":
        electrons = {"conv_thr": conv_thr, "diago_full_acc": ".true."}
    lines = ["&CONTROL"] + [f"  {k} = {v}" for k, v in control.items()] + ["/", "&SYSTEM"]
    lines += [f"  {k} = {v}" for k, v in system.items()] + ["/", "&ELECTRONS"]
    lines += [f"  {k} = {v}" for k, v in electrons.items()] + ["/"]
    if calculation in ("relax",):
        lines += ["&IONS", "/"]
    lines += ["ATOMIC_SPECIES"] + [f"  {s} {m} {pp}" for s, m, pp in species]
    lines += ["CELL_PARAMETERS angstrom", _fmt_block(cell_a)]
    lines += ["ATOMIC_POSITIONS crystal"]
    lines += [f"  {s} " + " ".join(f"{x:.12f}" for x in p) for s, p in zip(symbols, positions_frac)]
    if kpts_frac is not None:
        kp = np.asarray(kpts_frac)
        lines += [f"K_POINTS crystal", f"  {len(kp)}"]
        lines += [f"  {k[0]:.12f} {k[1]:.12f} {k[2]:.12f} 1.0" for k in kp]
    else:
        n = kmesh
        lines += ["K_POINTS automatic", f"  {n[0]} {n[1]} {n[2]} 0 0 0"]
    return "\n".join(lines) + "\n"


def parse_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    out = root.find("output")
    alat = float(out.find("atomic_structure").get("alat"))
    cell_b = np.array([[float(x) for x in out.find(f"atomic_structure/cell/a{i}").text.split()]
                       for i in (1, 2, 3)])
    pos_b = np.array([[float(x) for x in a.text.split()]
                      for a in out.findall("atomic_structure/atomic_positions/atom")])
    etot = float(out.find("total_energy/etot").text) * 2.0  # Ha -> Ry
    bs = out.find("band_structure")
    nelec = float(bs.find("nelec").text)
    kxml, eig = [], []
    for ks in bs.findall("ks_energies"):
        kxml.append([float(x) for x in ks.find("k_point").text.split()])
        eig.append([float(x) * HA_TO_EV for x in ks.find("eigenvalues").text.split()])
    kxml = np.array(kxml)
    kfrac = kxml @ cell_b.T / alat          # f_i = k . a_i / alat (k in 2pi/alat)
    forces = out.find("forces")
    forces = (np.array([float(x) for x in forces.text.split()]).reshape(-1, 3) * 2.0
              if forces is not None else None)  # Ha/bohr -> Ry/bohr
    stress = out.find("stress")
    stress = (np.array([float(x) for x in stress.text.split()]).reshape(3, 3)
              * 294210.2648 if stress is not None else None)  # Ha/bohr^3 -> kbar
    conv = root.find("output/convergence_info/scf_conv/convergence_achieved")
    return dict(energy_ry=etot, cell_a=cell_b * BOHR_TO_A, pos_bohr=pos_b, kfrac=kfrac,
                eigs_ev=np.array(eig), nelec=nelec, forces=forces, stress=stress,
                converged=(conv is None or conv.text.strip() == "true"))


def run_pw(inp: str, workdir: Path, prefix: str, nthreads: int = 1, timeout: float = 7200,
           keep: bool = False) -> PWResult:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "pw.in").write_text(inp)
    env = dict(os.environ, OMP_NUM_THREADS=str(nthreads), OPENBLAS_NUM_THREADS=str(nthreads),
               TMPDIR="/tmp")
    t0 = time.time()
    with open(workdir / "pw.out", "w") as fo:
        p = subprocess.run([str(ENV_BIN / "pw.x"), "-in", "pw.in"], cwd=workdir, stdout=fo,
                           stderr=subprocess.STDOUT, env=env, timeout=timeout)
    wall = time.time() - t0
    xml = workdir / f"{prefix}.xml"
    if p.returncode != 0 or not xml.exists():
        tail = (workdir / "pw.out").read_text()[-2000:]
        raise RuntimeError(f"pw.x failed (rc={p.returncode}) in {workdir}:\n{tail}")
    d = parse_xml(xml)
    cell = d["cell_a"]
    pos_frac = d["pos_bohr"] * BOHR_TO_A @ np.linalg.inv(cell)
    res = PWResult(energy_ry=d["energy_ry"], cell_a=cell, positions_frac=pos_frac,
                   forces_ry_bohr=d["forces"], stress_kbar=d["stress"], kpts_frac=d["kfrac"],
                   eigs_ev=d["eigs_ev"], nelec=d["nelec"], wall_s=wall, converged=d["converged"])
    if not keep:
        for f in workdir.glob(f"{prefix}.save"):
            shutil.rmtree(f, ignore_errors=True)
    return res
