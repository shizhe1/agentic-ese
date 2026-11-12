"""Valley-resolved, symmetry-equivariant surrogates and baselines.

Each valley family (Delta, L) has ONE network evaluated in that valley's frame:
    y_v(eps) = g_family(R_v eps R_v^T),
where R_v in O_h maps valley v onto the family's reference valley (Dz, L4). Invariance under
the reference valley's stabilizer (D4h for Dz, D3d for L4) is exact by group averaging.
Gamma and h are O_h-invariant (averaged over all 48 operations). Inputs include the
Bir-Pikus invariants sqrt(J_b), sqrt(J_d), which carry the conical splitting of the
degenerate Gamma valence/conduction states. Targets are valley energies relative to the VBM;
the gap is the minimum over valleys and its uncertainty the spread of that minimum across
an ensemble.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .strain import oh_operations, rotate_strain

OPS = oh_operations()
DELTA_AXIS = {"Dx": np.array([1.0, 0, 0]), "Dy": np.array([0, 1.0, 0]), "Dz": np.array([0, 0, 1.0])}
L_AXIS = {"L1": np.array([-1.0, 1, 1]), "L2": np.array([1.0, -1, 1]), "L3": np.array([1.0, 1, -1]),
          "L4": np.array([1.0, 1, 1])}
FAMILY = {**{v: "D" for v in DELTA_AXIS}, **{v: "L" for v in L_AXIS}, "G": "G"}
REF_AXIS = {"D": DELTA_AXIS["Dz"], "L": L_AXIS["L4"]}


def _maps(axis_from, axis_to):
    """An O_h operation taking +/-axis_from to +/-axis_to (valleys are inversion pairs)."""
    for r in OPS:
        v = r @ axis_from
        if np.allclose(v, axis_to) or np.allclose(v, -axis_to):
            return r
    raise ValueError("no O_h map")


def _stabilizer(axis):
    return [r for r in OPS if np.allclose(r @ axis, axis) or np.allclose(r @ axis, -axis)]


FRAME = {v: _maps(a, REF_AXIS["D"]) for v, a in DELTA_AXIS.items()}
FRAME.update({v: _maps(a, REF_AXIS["L"]) for v, a in L_AXIS.items()})
STAB = {"D": _stabilizer(REF_AXIS["D"]), "L": _stabilizer(REF_AXIS["L"]), "G": OPS, "h": OPS}
VALLEYS = ["G", "L1", "L2", "L3", "L4", "Dx", "Dy", "Dz"]


def features(e6: np.ndarray) -> np.ndarray:
    """(n,6) Voigt strains -> (n,8) inputs incl. Bir-Pikus invariants."""
    e = np.atleast_2d(e6)
    jb = (e[:, 0] - e[:, 1]) ** 2 + (e[:, 1] - e[:, 2]) ** 2 + (e[:, 2] - e[:, 0]) ** 2
    jd = e[:, 3] ** 2 + e[:, 4] ** 2 + e[:, 5] ** 2
    return np.hstack([e, np.sqrt(jb + 1e-12)[:, None], np.sqrt(jd + 1e-12)[:, None]])


def _orbit(e6: np.ndarray, ops) -> np.ndarray:
    """(n,6) -> (n, |ops|, 8) features of the strain under each operation."""
    out = np.stack([np.array([rotate_strain(x, r) for x in e6]) for r in ops], axis=1)
    n, m, _ = out.shape
    return features(out.reshape(-1, 6)).reshape(n, m, 8)


class MLP(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers, d = [], 8
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (n, m, 8) orbit -> group average
        return self.net(x).mean(dim=1).squeeze(-1)


def _fit_one(x, y, seed, epochs=600, lr=5e-3, wd=1e-6):
    torch.manual_seed(seed)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(m(xt), yt)
        loss.backward()
        opt.step()
        sched.step()
    return m


class _Head:
    """Ensemble for one target family with standardisation."""

    def __init__(self, ops, n_models=5, scale=20.0):
        self.ops, self.n, self.scale = ops, n_models, scale   # inputs scaled: strain*20 ~ O(1)

    def fit(self, e6, y, seed=0):
        x = _orbit(e6 * self.scale, self.ops)
        self.mu, self.sd = float(y.mean()), float(y.std() + 1e-9)
        self.models = [_fit_one(x, (y - self.mu) / self.sd, seed + i) for i in range(self.n)]
        return self

    def predict(self, e6):  # -> (n_models, n)
        x = torch.tensor(_orbit(np.atleast_2d(e6) * self.scale, self.ops), dtype=torch.float32)
        with torch.no_grad():
            return np.stack([m(x).numpy() * self.sd + self.mu for m in self.models])


class ValleySurrogate:
    """Valley-resolved, O_h-covariant ensemble surrogate for valley energies, gap and h."""

    def __init__(self, n_models=5):
        self.n_models = n_models

    def fit(self, records: list[dict]):
        e = np.array([r["strain_gl"] for r in records])
        y = {v: np.array([r["valleys"][v]["e"] for r in records]) for v in VALLEYS}
        rd = np.vstack([np.array([rotate_strain(x, FRAME[v]) for x in e]) for v in DELTA_AXIS])
        yd = np.concatenate([y[v] for v in DELTA_AXIS])
        rl = np.vstack([np.array([rotate_strain(x, FRAME[v]) for x in e]) for v in L_AXIS])
        yl = np.concatenate([y[v] for v in L_AXIS])
        jobs = {"D": (STAB["D"], rd, yd, 0), "L": (STAB["L"], rl, yl, 10),
                "G": (STAB["G"], e, y["G"], 20),
                "h": (STAB["h"], e, np.array([r["h_mev_a3"] for r in records]), 30)}
        from concurrent.futures import ThreadPoolExecutor
        torch.set_num_threads(1)
        with ThreadPoolExecutor(4) as ex:
            futs = {k: ex.submit(lambda a: _Head(a[0], self.n_models).fit(a[1], a[2], seed=a[3]), j)
                    for k, j in jobs.items()}
            self.heads = {k: f.result() for k, f in futs.items()}
        return self

    def predict(self, e6) -> dict:
        e = np.atleast_2d(np.asarray(e6, dtype=float))
        vals = {}
        for v in VALLEYS:
            fam = FAMILY[v]
            x = e if fam == "G" else np.array([rotate_strain(s, FRAME[v]) for s in e])
            vals[v] = self.heads[fam].predict(x)            # (n_models, n)
        stack = np.stack([vals[v] for v in VALLEYS])         # (8, n_models, n)
        eg_members = stack.min(axis=0)
        h = self.heads["h"].predict(e)
        cbm = np.array(VALLEYS)[stack.mean(axis=1).argmin(axis=0)]
        return dict(eg=eg_members.mean(0), eg_std=eg_members.std(0), h=h.mean(0), h_std=h.std(0),
                    valleys={v: vals[v].mean(0) for v in VALLEYS}, cbm=cbm)


class DirectGapSurrogate:
    """Baseline: ensemble regressing the gap directly on the 6 strain components (2019-style)."""

    def __init__(self, n_models=5):
        self.head = _Head([np.eye(3)], n_models)

    def fit(self, records):
        e = np.array([r["strain_gl"] for r in records])
        self.head.fit(e, np.array([r["eg_ev"] for r in records]))
        return self

    def predict(self, e6):
        p = self.head.predict(np.atleast_2d(e6))
        return dict(eg=p.mean(0), eg_std=p.std(0))


class BandGridSurrogate:
    """Baseline (simplified band-on-grid MLP; borrows the band-on-grid idea of Tsymbalov et al. 2021 but not their CNN,
    multi-band output or k-interpolation):
    an ensemble predicts the lowest conduction band on the fixed k-grid of the stage-1 NSCF
    (plus the Gamma-X line points) and the valence-band maximum; Eg = min_k CB - VBM."""

    def __init__(self, n_models=5, epochs=600):
        self.n, self.epochs = n_models, epochs

    @staticmethod
    def targets(rec):
        raw = np.load(str(Path(__file__).resolve().parents[1] / "runs" / "cache" / f"{rec['key']}_raw.npz"))
        e, nocc = raw["e_grid"], 4
        return np.concatenate([e[:, nocc], [e[:, nocc - 1].max()]])

    def fit(self, records):
        x = np.array([r["strain_gl"] for r in records]) * 20.0
        y = np.array([self.targets(r) for r in records])
        self.mu, self.sd = y.mean(0), y.std(0) + 1e-6
        xt = torch.tensor(x, dtype=torch.float32); yt = torch.tensor((y - self.mu) / self.sd, dtype=torch.float32)
        self.models = []
        for s in range(self.n):
            torch.manual_seed(100 + s)
            m = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, y.shape[1]))
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for _ in range(self.epochs):
                opt.zero_grad(); l = nn.functional.mse_loss(m(xt), yt); l.backward(); opt.step()
            self.models.append(m)
        return self

    def predict(self, e6):
        x = torch.tensor(np.atleast_2d(e6) * 20.0, dtype=torch.float32)
        with torch.no_grad():
            p = np.stack([m(x).numpy() * self.sd + self.mu for m in self.models])
        eg = p[:, :, :-1].min(axis=2) - p[:, :, -1]
        return dict(eg=eg.mean(0), eg_std=eg.std(0))
