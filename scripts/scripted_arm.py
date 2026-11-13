"""Scripted non-LLM arm (post-review amendment): the agent's own MCP tool functions, cost ledger and
150-unit budget, driven by a fixed schedule with the expert formalisation. Isolates what the LLM adds.

usage: scripted_arm.py TASK SEED [FORMAL]   (TASK in H1, H2, H3; FORMAL = expert (default) | naive)

FORMAL=naive (2026-10-28, author's request; after the data freeze): the same tools, budget, schedule and verifier,
but the pre-registered naive formalization (rule d): raw-PBE thresholds, no subspace, no stability in the search.
Isolates the formalization with search and verification held fixed. Success is still the pre-registered task
criterion (full HSE06, elastic stability, and e33 = 0 on H3).
"""
import asyncio, json, os, sys, time
from pathlib import Path

TASK, SEED = sys.argv[1], int(sys.argv[2])
FORMAL = sys.argv[3] if len(sys.argv) > 3 else "expert"
R = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("ESE_RUNS_DIR") or R / "runs") / "scripted" / (f"{TASK}_s{SEED}" if FORMAL == "expert" else f"{TASK}_s{SEED}_{FORMAL}")
RUN.mkdir(parents=True, exist_ok=True)
os.environ.update(ESE_RUN_DIR=str(RUN), ESE_BUDGET="150", ESE_WORKERS=os.environ.get("ESE_WORKERS", "4"))
sys.path.insert(0, str(R))
from eseagent import server  # noqa: E402  (reads ESE_* at import)

# expert formalisation (formalizations/expert.json), at the HSE-scissor level used during search
SPEC = {"H1": dict(mat="C", cons=[{"kind": "gap_min", "value": 1.97}, {"kind": "gap_max", "value": 2.03}],
                   free=None, fixed=None, ok=lambda g: abs(g - 2.0) <= 0.03),
        "H2": dict(mat="Si", cons=[{"kind": "gap_max", "value": 0.80}], free=None, fixed=None, ok=lambda g: g <= 0.80),
        "H3": dict(mat="Si", cons=[{"kind": "gap_max", "value": 0.0}], free=[0, 1, 3, 4, 5], fixed={2: 0.0},
                   ok=lambda g: g <= 0.0)}[TASK]
TASK_OK = SPEC["ok"]                                       # pre-registered success criterion (gap part)
SUBSPACE = (lambda e: abs(e[2]) < 1e-6) if TASK == "H3" else (lambda e: True)
LEVEL, GAP = "hse_scissor", "eg_hse_scissor"
if FORMAL == "naive":                                     # rule (d): raw PBE, full 6D, no stability in the search
    SPEC = dict(SPEC, free=None, fixed=None)
    LEVEL, GAP = "pbe", "eg_pbe"
MAT = SPEC["mat"]


class _Ctx:
    async def report_progress(self, *a, **k):
        pass


def dft(strains):
    return asyncio.run(server.run_dft(MAT, strains, _Ctx()))["results"]


def feasible(r, cons):
    if "error" in r:
        return False
    g = r[GAP]
    return all((g <= c["value"]) if c["kind"] == "gap_max" else (g >= c["value"]) for c in cons)


def one_round(cons, data):
    server.fit_surrogate(MAT)
    cands = server.optimize_strain(MAT, cons, free=SPEC["free"], fixed=SPEC["fixed"], bound=0.10,
                                   level=LEVEL)["candidates"][:3]
    data += dft([c["strain_gl"] for c in cands])


def select(cons, data, exclude=()):
    ok = [r for r in data if feasible(r, cons) and tuple(r["strain"]) not in exclude]
    return min(ok, key=lambda r: r["h"]) if ok else None


def verify(state):
    v = asyncio.run(server.verify_hse(MAT, state["strain"]))
    b = asyncio.run(server.elastic_stability(MAT, state["strain"]))
    return dict(strain=state["strain"], h=state["h"], eg_scissor=state[GAP], eg_hse=v["eg_hse"],
                cbm_hse=v["cbm_hse"], born_min_eig_GPa=b["min_stiffness_eig_GPa"], stable=b["stable"],
                subspace_ok=bool(SUBSPACE(state["strain"])),
                passed=bool(TASK_OK(v["eg_hse"]) and b["stable"] and SUBSPACE(state["strain"])))


t0 = time.time(); log = []
data = dft(server.sample_strains(MAT, 20, fixed=SPEC["fixed"], seed=SEED)["strains"])
cons = SPEC["cons"]
for _ in range(3):
    one_round(cons, data)
state = select(cons, data)
result = verify(state) if state else None
log.append(dict(stage="first", constraints=cons, result=result))
if result and not result["passed"]:                       # one scripted re-target (mirrors the verifier round)
    exclude = {tuple(state["strain"])}
    if TASK_OK(result["eg_hse"]):                          # gap fine, Born (or subspace) failed: next-best state
        cons2 = cons
    else:                                                  # shift window by the observed full-HSE - scissor offset
        off = result["eg_hse"] - result["eg_scissor"]
        cons2 = [dict(c, value=round(c["value"] - off, 4)) for c in cons]
        one_round(cons2, data)
    state = select(cons2, data, exclude)
    result = verify(state) if state else None
    log.append(dict(stage="retarget", constraints=cons2, result=result))
spent = sum(json.loads(l)["cost"] for l in open(RUN / "ledger.jsonl"))
final = dict(task=TASK, seed=SEED, formalization=FORMAL, level=LEVEL, verdict="VERIFIED" if result and result["passed"] else "FAILED",
             final=result, rounds=log, spent=spent, n_dft=len(data), minutes=round((time.time() - t0) / 60, 1))
(RUN / "final.json").write_text(json.dumps(final, indent=1))
print(json.dumps({k: final[k] for k in ("task", "seed", "verdict", "spent", "minutes")}),
      (result or {}).get("h"), (result or {}).get("eg_hse"), flush=True)
