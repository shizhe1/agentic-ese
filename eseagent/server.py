"""MCP tool server ("ese") exposing deep-ESE tools to the agent.

Per-run isolation: everything a run computes or fits lives in $ESE_RUN_DIR (its own data store,
budget ledger and notebook). The shared DFT cache only avoids recomputation; a cache hit is
charged exactly like a miss and returns identical output. Strains are Green-Lagrange, Voigt
order (e11,e22,e33,e23,e13,e12), TENSOR shear, as fractions (0.01 = 1%).
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from mcp.server.mcpserver import Context, MCPServer
from scipy.stats import qmc

from . import dft
from .materials import MATERIALS
from .optimize import CONSTRAINT_KINDS, optimize_strain as _opt
from .hse import hse_correction
from .surrogate import ValleySurrogate

RUN = Path(os.environ.get("ESE_RUN_DIR", "/tmp/ese_run"))
RUN.mkdir(parents=True, exist_ok=True)
BUDGET = float(os.environ.get("ESE_BUDGET", "400"))
COST = {"pbe": 1.0, "hse_verify": 15.0, "elastic_stability": 12.0}
MAX_BATCH = 60
POOL = ThreadPoolExecutor(int(os.environ.get("ESE_WORKERS", "10")))
LEDGER = RUN / "ledger.jsonl"
NOTEBOOK = RUN / "notebook.jsonl"

server = MCPServer("ese", instructions=(
    "Tools for deep elastic strain engineering of diamond-structure crystals (materials: Si, C). "
    "Strain: Green-Lagrange, Voigt order (e11,e22,e33,e23,e13,e12), tensor shear, fractions. "
    "h = elastic strain energy density (meV/A^3). Gaps from PBE DFT. Every DFT call is charged "
    "against a fixed budget; check describe_material for costs and the remaining budget."))


def _log(path: Path, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["cost"] for l in LEDGER.read_text().splitlines() if l.strip())


def _records(material: str) -> list[dict]:
    p = RUN / f"data_{material}.jsonl"
    if not p.exists():
        return []
    keys = [json.loads(l)["key"] for l in p.read_text().splitlines() if l.strip()]
    return [json.loads((dft.CACHE / f"{k}.json").read_text()) for k in dict.fromkeys(keys)]


def _check_material(material: str):
    if material not in MATERIALS:
        raise ValueError(f"material must be one of {list(MATERIALS)}")


def _check_strains(strains) -> np.ndarray:
    a = np.asarray(strains, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6:
        raise ValueError("strains must be a list of 6-component Voigt vectors")
    if np.abs(a).max() > 0.25:
        raise ValueError("strain components must be fractions with |e| <= 0.25 (0.01 = 1%)")
    return a


def scissor(material: str) -> dict:
    """Per-valley HSE06 corrections at zero strain (cached; computed once per material)."""
    return hse_correction(material, np.zeros(6))["corrections"]


def _brief(r: dict) -> dict:
    v = {k: round(r["valleys"][k]["e"], 3) for k in ("G", "L1", "L2", "L3", "L4", "Dx", "Dy", "Dz")}
    sc = scissor(r["material"])
    vc = {k: v[k] + sc[k] for k in v}
    cc = min(vc, key=vc.get)
    return dict(strain=[round(x, 5) for x in r["strain_gl"]], h=round(r["h_mev_a3"], 3),
                eg_pbe=round(r["eg_ev"], 3), cbm_pbe=r["cbm_valley"], valleys_pbe=v,
                eg_hse_scissor=round(vc[cc], 3), cbm_hse_scissor=cc,
                **({"warn": r["warnings"]} if r["warnings"] else {}))


@server.tool()
def describe_material(material: str) -> dict:
    """Reference data, conventions, costs and remaining budget for a material."""
    _check_material(material)
    m = MATERIALS[material]
    z = dft.compute_point(m, np.zeros(6), need_e0=False)
    sc = scissor(material)
    return dict(material=material, levels="PBE valley energies; 'hse_scissor' adds per-valley HSE06 "
                "corrections fitted at zero strain (cheap). verify_hse runs full HSE06 at one strain.",
                hse_scissor_eV={k: round(v, 3) for k, v in sc.items()},
                zero_strain_eg_hse_eV=round(hse_correction(material, np.zeros(6))["eg_hse"], 3), a0_A=m.a0,
                zero_strain=dict(eg_eV=round(z["eg_ev"], 3), cbm=z["cbm_valley"],
                                 valleys_eV={k: round(z["valleys"][k]["e"], 3) for k in z["valleys"]}),
                experiment=dict(gap_eV=m.exp_gap_ev, a0_A=m.exp_a0),
                conventions="GL strain, Voigt (e11,e22,e33,e23,e13,e12), tensor shear, fractions; "
                            "valley energies are relative to the valence-band maximum",
                costs=COST, budget_total=BUDGET, budget_left=round(BUDGET - _spent(), 1),
                constraint_kinds=CONSTRAINT_KINDS)


@server.tool()
def sample_strains(material: str, n: int, normal_max: float = 0.05, shear_max: float = 0.03,
                   fixed: dict | None = None, seed: int = 0) -> dict:
    """Latin-hypercube strains (no DFT). `fixed` maps Voigt index (0-5) to a fixed value."""
    _check_material(material)
    n = int(min(max(n, 1), 200))
    x = qmc.LatinHypercube(d=6, seed=seed).random(n)
    lo = np.array([-normal_max] * 3 + [-shear_max] * 3)
    e = qmc.scale(x, lo, -lo)
    for i, v in (fixed or {}).items():
        e[:, int(i)] = float(v)
    return dict(strains=np.round(e, 5).tolist())


@server.tool()
async def run_dft(material: str, strains: list[list[float]], ctx: Context) -> dict:
    """PBE DFT at each strain (internal relaxation, valley-resolved band edges, h). Cost 1 unit each."""
    _check_material(material)
    e = _check_strains(strains)
    if len(e) > MAX_BATCH:
        raise ValueError(f"at most {MAX_BATCH} strains per call")
    cost = COST["pbe"] * len(e)
    if _spent() + cost > BUDGET + 1e-9:
        raise ValueError(f"budget exceeded: {round(BUDGET - _spent(), 1)} units left, {cost} requested")
    _log(LEDGER, dict(t=time.time(), tool="run_dft", n=len(e), cost=cost))
    loop = asyncio.get_running_loop()
    def safe(x):
        try:
            return dft.compute_point(material, x)
        except RuntimeError:
            return dict(failed=True, strain_gl=list(x))
    futs = [loop.run_in_executor(POOL, safe, x) for x in e]
    done, out = 0, []
    for f in asyncio.as_completed(futs):
        out.append(await f)
        done += 1
        await ctx.report_progress(done, len(e), f"{done}/{len(e)} DFT points")
    order = {tuple(np.round(r["strain_gl"], 7)): r for r in out}
    res = [order[tuple(np.round(x, 7))] for x in e]
    for r in res:
        if not r.get("failed"):
            _log(RUN / f"data_{material}.jsonl", dict(key=r["key"]))
    return dict(results=[dict(strain=[round(x, 5) for x in r["strain_gl"]], error="SCF did not converge")
                         if r.get("failed") else _brief(r) for r in res],
                budget_left=round(BUDGET - _spent(), 1))


@server.tool()
def fit_surrogate(material: str) -> dict:
    """Fit the valley-resolved surrogate on THIS run's DFT points; reports held-out error."""
    _check_material(material)
    recs = _records(material)
    if len(recs) < 20:
        raise ValueError(f"need >= 20 DFT points, have {len(recs)}")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(recs))
    nte = max(4, len(recs) // 5)
    te, tr = [recs[i] for i in idx[:nte]], [recs[i] for i in idx[nte:]]
    m = ValleySurrogate(n_models=3).fit(tr)
    p = m.predict(np.array([r["strain_gl"] for r in te]))
    mae = float(np.mean(np.abs(p["eg"] - np.array([r["eg_ev"] for r in te]))))
    hmae = float(np.mean(np.abs(p["h"] - np.array([r["h_mev_a3"] for r in te]))))
    final = ValleySurrogate(n_models=3).fit(recs)
    (RUN / f"surrogate_{material}.pkl").write_bytes(pickle.dumps(final))
    return dict(n_points=len(recs), holdout_gap_mae_eV=round(mae, 4), holdout_h_mae=round(hmae, 4))


def _model(material):
    p = RUN / f"surrogate_{material}.pkl"
    if not p.exists():
        raise ValueError("no surrogate yet: call fit_surrogate first")
    return pickle.loads(p.read_bytes())


@server.tool()
def query_surrogate(material: str, strains: list[list[float]]) -> dict:
    """Surrogate predictions (gap, its ensemble std, h, CBM valley). No DFT, no cost."""
    e = _check_strains(strains)[:100]
    p = _model(material).predict(e)
    return dict(predictions=[dict(eg=round(float(p["eg"][i]), 3), eg_std=round(float(p["eg_std"][i]), 3),
                                  h=round(float(p["h"][i]), 3), cbm=str(p["cbm"][i]))
                             for i in range(len(e))])


@server.tool()
def optimize_strain(material: str, constraints: list[dict], free: list[int] | None = None,
                    fixed: dict | None = None, bound: float = 0.10, n_starts: int = 32,
                    kappa: float = 0.0, level: str = "pbe") -> dict:
    """Minimise surrogate h subject to constraints [{kind, value}] over free Voigt indices.
    kinds: gap_max, gap_min, direct, indirect. kappa>0 enforces constraints at mean +/- kappa*std.
    level: 'pbe' or 'hse_scissor' (constraints evaluated on HSE-corrected valley energies)."""
    if level not in ("pbe", "hse_scissor"):
        raise ValueError("level must be 'pbe' or 'hse_scissor'")
    cands = _opt(_model(material), constraints, free=free, fixed=fixed, bounds=(-bound, bound),
                 n_starts=n_starts, kappa=kappa, shift=scissor(material) if level == "hse_scissor" else None)
    return dict(candidates=[{k: (np.round(v, 5).tolist() if k == "strain_gl" else round(v, 4)
                                  if isinstance(v, float) else v) for k, v in c.items()}
                            for c in cands[:5]])


@server.tool()
def propose_active_learning(material: str, n: int, constraints: list[dict] | None = None,
                            fixed: dict | None = None, bound: float = 0.10, seed: int = 0) -> dict:
    """Pick n new strains where the surrogate is most uncertain, favouring low h and states near
    the constraint boundary (if constraints are given). No DFT, no cost."""
    m = _model(material)
    rng = np.random.default_rng(seed)
    e = rng.uniform(-bound, bound, (4000, 6)) * rng.uniform(0.1, 1.0, (4000, 1))
    for i, v in (fixed or {}).items():
        e[:, int(i)] = float(v)
    p = m.predict(e)
    score = p["eg_std"] / (1.0 + np.maximum(p["h"], 0) / 5.0)
    for c in constraints or []:
        if c["kind"] in ("gap_max", "gap_min"):
            score *= np.exp(-np.abs(p["eg"] - float(c.get("value", 0))) / 0.15)
    pick = []
    for i in np.argsort(-score):
        if all(np.linalg.norm(e[i] - e[j]) > 0.01 for j in pick):
            pick.append(i)
        if len(pick) >= int(n):
            break
    return dict(strains=np.round(e[pick], 5).tolist())


@server.tool()
def map_envelope(material: str, h_max: float, n_bins: int = 12, bound: float = 0.10) -> dict:
    """Surrogate-sampled lower/upper envelope of the gap vs h (density-of-bandgap style)."""
    m = _model(material)
    rng = np.random.default_rng(1)
    e = rng.uniform(-bound, bound, (20000, 6)) * rng.uniform(0.05, 1.0, (20000, 1))
    p = m.predict(e)
    edges = np.linspace(0, h_max, n_bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (p["h"] >= a) & (p["h"] < b)
        if s.sum() >= 5:
            rows.append(dict(h=round((a + b) / 2, 3), eg_min=round(float(p["eg"][s].min()), 3),
                             eg_max=round(float(p["eg"][s].max()), 3), n=int(s.sum())))
    return dict(envelope=rows)


@server.tool()
async def verify_hse(material: str, strain: list[float]) -> dict:
    """Full HSE06 band edges at ONE strain (slow, ~30 min; cost 15 units). For final answers."""
    _check_material(material)
    e = _check_strains([strain])[0]
    if _spent() + COST["hse_verify"] > BUDGET + 1e-9:
        raise ValueError("budget exceeded")
    _log(LEDGER, dict(t=time.time(), tool="verify_hse", n=1, cost=COST["hse_verify"]))
    r = await asyncio.get_running_loop().run_in_executor(POOL, hse_correction, material, e)
    return dict(strain=[round(x, 5) for x in e], eg_hse=round(r["eg_hse"], 3), cbm_hse=r["cbm_hse"],
                valleys_hse={k: round(v, 3) for k, v in r["valleys_hse"].items()},
                h=round(r["h_mev_a3"], 3), budget_left=round(BUDGET - _spent(), 1))


@server.tool()
async def elastic_stability(material: str, strain: list[float], delta: float = 0.005) -> dict:
    """Born (elastic) stability at a strained state: stiffness from DFT stresses under +/-delta
    on each Voigt component (12 PBE points, cost 12 units). Stable if all eigenvalues > 0."""
    _check_material(material)
    e0 = _check_strains([strain])[0]
    if _spent() + COST["elastic_stability"] > BUDGET + 1e-9:
        raise ValueError("budget exceeded")
    _log(LEDGER, dict(t=time.time(), tool="elastic_stability", n=12, cost=COST["elastic_stability"]))
    pts = []
    for j in range(6):
        for sgn in (1, -1):
            e = e0.copy(); e[j] += sgn * delta; pts.append(e)
    loop = asyncio.get_running_loop()
    recs = await asyncio.gather(*[loop.run_in_executor(POOL, dft.compute_point, material, x) for x in pts])
    def voigt_stress(r):
        t = np.array(r["stress_kbar"]) / 10.0          # GPa, QE sign: positive = tensile reaction
        return -np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[0, 2], t[0, 1]])
    c = np.zeros((6, 6))
    for j in range(6):
        ds = voigt_stress(recs[2 * j]) - voigt_stress(recs[2 * j + 1])
        c[:, j] = ds / (2 * delta) / (2.0 if j >= 3 else 1.0)   # tensor-shear -> engineering
    w = np.linalg.eigvalsh(0.5 * (c + c.T))
    return dict(strain=[round(x, 5) for x in e0], min_stiffness_eig_GPa=round(float(w.min()), 1),
                stable=bool(w.min() > 0), budget_left=round(BUDGET - _spent(), 1))


@server.tool()
def log_finding(text: str) -> dict:
    """Append a finding to this run's lab notebook."""
    _log(NOTEBOOK, dict(t=time.time(), text=text[:2000]))
    return dict(ok=True)


if __name__ == "__main__":
    server.run("stdio")
