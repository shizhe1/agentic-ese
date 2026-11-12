"""Non-LLM baseline: constrained Bayesian optimisation directly on DFT (BoTorch, frozen settings).

Minimises h over a strain subspace subject to inequality constraints c_j(eps) >= 0 computed
from DFT valley energies (same constraint kinds as optimize.py). One GP per objective/constraint
(SingleTaskGP, Matern-5/2, ARD); acquisition = constrained expected improvement (feasibility-
weighted EI) optimised by random multi-start. Settings are frozen and identical across tasks.

Naming: internal file names, keys and logs say "scbo"/"SCBO" for historical reasons. This is NOT
trust-region SCBO (Eriksson & Poloczek 2021); the paper calls this arm cBO (Gardner et al. 2014).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm, qmc

from .dft import compute_point

N_INIT, BATCH, N_CAND = 20, 10, 4000


def constraint_values(rec: dict, constraints: list[dict], valleys_override: dict | None = None) -> list[float]:
    v = valleys_override or {k: rec["valleys"][k]["e"] for k in rec["valleys"]}
    eg = min(v.values())
    out = []
    for c in constraints:
        k, val = c["kind"], float(c.get("value", 0.0))
        if k == "gap_max":
            out.append(val - eg)
        elif k == "gap_min":
            out.append(eg - val)
        elif k == "direct":
            out.append(min(x for n, x in v.items() if n != "G") - v["G"] - val)
        elif k == "indirect":
            out.append(v["G"] - min(x for n, x in v.items() if n != "G") - val)
    return out


def _gp(x, y):
    m = SingleTaskGP(torch.tensor(x), torch.tensor(y)[:, None])
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    return m


def run_scbo(material: str, constraints: list[dict], budget: int, free=None, fixed=None,
             bound: float = 0.10, seed: int = 0, correction=None, log_path: Path | None = None) -> dict:
    """correction: optional callable(rec)->valleys dict (e.g. HSE-corrected) for constraints."""
    free = list(range(6)) if free is None else list(free)
    base = np.zeros(6)
    for i, v in (fixed or {}).items():
        base[int(i)] = float(v)
    rng = np.random.default_rng(seed)

    def full(u):  # u in [0,1]^d -> strain
        e = base.copy()
        e[free] = (2 * u - 1) * bound
        return e

    X, H, Cs, hist = [], [], [], []

    def evaluate(us):
        from concurrent.futures import ThreadPoolExecutor
        def safe(u):
            try:
                return compute_point(material, full(u))
            except RuntimeError:
                return None
        with ThreadPoolExecutor(10) as ex:
            recs = list(ex.map(safe, us))
        for u, rec in zip(us, recs):
            if rec is None:
                continue
            vo = correction(rec) if correction else None
            c = constraint_values(rec, constraints, vo)
            X.append(u), H.append(rec["h_mev_a3"]), Cs.append(c)
            feas = [h for h, cc in zip(H, Cs) if min(cc) >= 0]
            hist.append(dict(n=len(H), best_h=min(feas) if feas else None))

    evaluate(qmc.LatinHypercube(d=len(free), seed=seed).random(min(N_INIT, budget)))
    while len(H) < budget:
        x = np.array(X, dtype=np.float64)
        gh = _gp(x, (np.array(H) - np.mean(H)) / (np.std(H) + 1e-9))
        gcs = [_gp(x, np.array(Cs)[:, j] / (np.std(np.array(Cs)[:, j]) + 1e-9))
               for j in range(len(constraints))]
        cand = rng.uniform(0, 1, (N_CAND, len(free)))
        ct = torch.tensor(cand)
        with torch.no_grad():
            ph = gh.posterior(ct)
            mu, sd = ph.mean.squeeze(-1).numpy(), ph.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
            pf = np.ones(N_CAND)
            for g in gcs:
                p = g.posterior(ct)
                pf *= norm.cdf(p.mean.squeeze(-1).numpy() / p.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy())
        feas = [(h - np.mean(H)) / (np.std(H) + 1e-9) for h, cc in zip(H, Cs) if min(cc) >= 0]
        if feas:
            best = min(feas)
            z = (best - mu) / sd
            acq = ((best - mu) * norm.cdf(z) + sd * norm.pdf(z)) * pf
        else:
            acq = pf                                          # find feasibility first
        pick = []
        for i in np.argsort(-acq):
            if all(np.linalg.norm(cand[i] - cand[j]) > 0.02 for j in pick):
                pick.append(i)
            if len(pick) >= min(BATCH, budget - len(H)):
                break
        evaluate(cand[pick])
    feas = [(h, X[i]) for i, (h, cc) in enumerate(zip(H, Cs)) if min(cc) >= 0]
    best = min(feas, key=lambda t: t[0]) if feas else None
    res = dict(material=material, constraints=constraints, budget=budget, seed=seed,
               best_h=best[0] if best else None,
               best_strain=full(np.asarray(best[1])).tolist() if best else None, history=hist,
               points=[dict(strain=full(np.asarray(x)).tolist(), h=h, feasible=bool(min(c) >= 0)) for x, h, c in zip(X, H, Cs)])
    if log_path:
        Path(log_path).write_text(json.dumps(res))
    return res
