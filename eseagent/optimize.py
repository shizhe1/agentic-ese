"""Deterministic constrained inverse design on a surrogate: minimise h subject to band-edge
constraints, with multi-start SLSQP. The LLM never does this search; it only formalises the
problem (objective, constraints, subspace, bounds, margins) that this function solves."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

CONSTRAINT_KINDS = {
    "gap_max": "Eg <= value (eV)",
    "gap_min": "Eg >= value (eV)",
    "direct": "Gamma valley is the CBM by at least `value` eV",
    "indirect": "some non-Gamma valley is the CBM by at least `value` eV",
}


def _constraint_fns(pred, cons, kappa):
    # pred(x)['eg'] and ['valleys'] already include any level-of-theory shift
    fns = []
    for c in cons:
        kind, val = c["kind"], float(c.get("value", 0.0))
        if kind == "gap_max":
            fns.append(lambda x, v=val: v - (pred(x)["eg"] + kappa * pred(x)["eg_std"]))
        elif kind == "gap_min":
            fns.append(lambda x, v=val: (pred(x)["eg"] - kappa * pred(x)["eg_std"]) - v)
        elif kind == "direct":
            def f(x, v=val):
                p = pred(x)["valleys"]
                return min(p[k] for k in p if k != "G") - p["G"] - v
            fns.append(f)
        elif kind == "indirect":
            def f(x, v=val):
                p = pred(x)["valleys"]
                return p["G"] - min(p[k] for k in p if k != "G") - v
            fns.append(f)
        else:
            raise ValueError(f"unknown constraint kind {kind!r}; allowed: {list(CONSTRAINT_KINDS)}")
    return fns


def optimize_strain(model, constraints: list[dict], free: list[int] | None = None,
                    bounds: tuple[float, float] = (-0.10, 0.10), fixed: dict | None = None,
                    n_starts: int = 32, kappa: float = 0.0, seed: int = 0,
                    shift: dict | None = None) -> list[dict]:
    """Minimise predicted h over the free Voigt components (0..5); others fixed (default 0).

    kappa > 0 makes constraints robust: they must hold at mean +/- kappa * ensemble std."""
    free = list(range(6)) if free is None else list(free)
    base = np.zeros(6)
    for i, v in (fixed or {}).items():
        base[int(i)] = float(v)

    def full(x):
        e = base.copy()
        e[free] = x
        return e

    cache = {}

    def pred(x):
        k = tuple(np.round(x, 10))
        if k not in cache:
            p = model.predict(full(x)[None, :])
            vals = {v: float(a[0]) + (shift or {}).get(v, 0.0) for v, a in p["valleys"].items()}
            eg = min(vals.values()) if shift else float(p["eg"][0])
            cache[k] = dict(eg=eg, eg_std=float(p["eg_std"][0]), h=float(p["h"][0]), valleys=vals)
        return cache[k]

    cons = [dict(type="ineq", fun=f) for f in _constraint_fns(pred, constraints, kappa)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_starts):
        x0 = rng.uniform(bounds[0], bounds[1], len(free)) * rng.uniform(0.2, 1.0)
        r = minimize(lambda x: pred(x)["h"], x0, method="SLSQP", constraints=cons,
                     bounds=[bounds] * len(free), options=dict(maxiter=200, ftol=1e-9))
        ok = r.success and all(c["fun"](r.x) >= -1e-6 for c in cons)
        if ok:
            p = pred(r.x)
            out.append(dict(strain_gl=full(r.x).tolist(), h_pred=p["h"], eg_pred=p["eg"],
                            eg_std=p["eg_std"], cbm_pred=min(p["valleys"], key=p["valleys"].get)))
    out.sort(key=lambda d: d["h_pred"])
    # drop near-duplicates (symmetry-equivalent states are kept: they differ in strain)
    uniq = []
    for d in out:
        if all(np.linalg.norm(np.subtract(d["strain_gl"], u["strain_gl"])) > 1e-3 for u in uniq):
            uniq.append(d)
    return uniq[:8]
