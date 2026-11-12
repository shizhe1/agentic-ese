"""Material definitions (diamond-structure group-IV crystals)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FCC_PRIM = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
DIAMOND_POS = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])


@dataclass(frozen=True)
class Material:
    name: str
    symbol: str
    mass: float
    pseudo: str
    a0: float          # PBE equilibrium lattice constant (A), set after Gate-1 validation
    ecutwfc: float     # Ry
    kmesh: int         # SCF Monkhorst-Pack n x n x n, fixed in reciprocal coordinates
    exp_gap_ev: float  # experimental (low T) indirect gap, for reporting only
    exp_a0: float

    def cell0(self) -> np.ndarray:
        return self.a0 * FCC_PRIM

    def species(self):
        return [(self.symbol, self.mass, self.pseudo)]

    def symbols(self):
        return [self.symbol, self.symbol]


# a0: PBE Birch-Murnaghan fits (runs/gate1_eos/eos.json); ecut converged in runs/gate1.
SI = Material("Si", "Si", 28.0855, "Si.upf", a0=5.4689, ecutwfc=50.0, kmesh=8,
              exp_gap_ev=1.17, exp_a0=5.431)
C = Material("C", "C", 12.011, "C.upf", a0=3.5724, ecutwfc=100.0, kmesh=8,
             exp_gap_ev=5.48, exp_a0=3.567)
MATERIALS = {"Si": SI, "C": C}
