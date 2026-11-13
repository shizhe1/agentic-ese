"""Launch one isolated headless agent campaign.

usage: run_agent.py TASK_ID RUN_ID [--model sonnet|opus] [--budget 150] [--architecture multi|single]

Isolation: fresh empty cwd outside the project; only the `ese` MCP server; built-in tools limited
to `Agent` (sub-agents), so no file, shell or web access anywhere; per-run data store
($ESE_RUN_DIR). Full stream-json transcript (incl. sub-agent text) is written to the run dir.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PY = Path(sys.executable)
CLAUDE = Path(os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude")

TASKS = {  # verbatim from preregistration.md
    "D1": "Find the lowest-elastic-energy state of uniaxial stress along a cube axis that closes the band gap of silicon.",
    "T1": "Find the strain state that turns silicon into a metal (zero band gap) at the lowest possible elastic strain energy.",
    "T2": "Find the strain state that makes silicon a direct-band-gap semiconductor at the lowest possible elastic strain energy, and assess whether that state is stable.",
    "T3": "Find the strain state that makes diamond a direct-band-gap semiconductor at the lowest possible elastic strain energy.",
    "H1": "Find the strain state that lowers the band gap of diamond to 2.0 eV at the lowest possible elastic strain energy, and compare it with simple uniaxial loading along <100>, <110> and <111>.",
    "H2": "Make silicon absorb light at 1550 nm at the lowest possible elastic strain energy, while keeping the crystal stable.",
    "H3": "A silicon film is constrained so that its normal strain perpendicular to the (001) surface is zero. Find the lowest-elastic-energy strain state that closes its band gap.",
}

SYSTEM = """You are the orchestrator of a computational materials-science team doing deep elastic strain \
engineering with first-principles tools. You have no file, shell or web access: you act only through \
the `ese` tools and two sub-agents.
- Delegate problem formalisation and the inverse-design campaign to the `designer` sub-agent.
- Delegate independent checking of any candidate answer to the `verifier` sub-agent.
- Respect the DFT budget reported by the tools. Do not report a result that has not been verified by DFT.
Finish with the final answer in the requested JSON format: the formalised problem you solved, the \
best verified strain (Green-Lagrange, Voigt, tensor shear, fractions), its h, its gap and CBM \
valley, whether the constraints are satisfied at DFT level, and a short physical explanation."""

DESIGNER = """You are the inverse-design specialist. First write down the optimisation problem precisely: \
objective, every constraint with its numerical threshold and the level of theory at which it must be \
evaluated, the strain subspace and bounds, and any stability requirement. Then run an efficient \
campaign with the ese tools (sample -> DFT -> fit surrogate -> active learning -> optimise -> DFT check), \
staying within budget. The numerical search is done by optimize_strain; you choose its inputs. \
Report candidates with their DFT values and your reasoning in terms of conduction-band valleys."""

VERIFIER = """You are an independent verifier. You did not design the candidate. Re-evaluate it with DFT, \
check symmetry-equivalent strains, units and conventions (fractions, tensor shear, Green-Lagrange), \
whether every constraint really holds at the right level of theory, and stability where requested. \
Reject any claim that lacks DFT evidence. Report clearly: VERIFIED or REJECTED, with numbers."""

DESIGN_TOOLS = ["describe_material", "sample_strains", "run_dft", "fit_surrogate", "query_surrogate",
                "optimize_strain", "propose_active_learning", "map_envelope", "log_finding"]
VERIFY_TOOLS = ["describe_material", "run_dft", "verify_hse", "elastic_stability", "log_finding"]

SCHEMA = {"type": "object", "properties": {
    "formalization": {"type": "object", "properties": {
        "objective": {"type": "string"}, "constraints": {"type": "array", "items": {"type": "string"}},
        "level_of_theory": {"type": "string"}, "subspace": {"type": "string"},
        "stability": {"type": "string"}}},
    "strain_gl": {"type": "array", "items": {"type": "number"}},
    "h_mev_a3": {"type": "number"}, "eg_ev": {"type": "number"}, "cbm_valley": {"type": "string"},
    "constraints_satisfied": {"type": "boolean"}, "explanation": {"type": "string"}},
    "required": ["formalization", "strain_gl", "h_mev_a3", "eg_ev", "constraints_satisfied"]}


VERIFY_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["VERIFIED", "REJECTED"]},
    "strain_gl": {"type": "array", "items": {"type": "number"}}, "h_mev_a3": {"type": "number"},
    "eg_ev": {"type": "number"}, "eg_level": {"type": "string"}, "stable": {"type": "string"},
    "issues": {"type": "array", "items": {"type": "string"}}},
    "required": ["verdict", "issues"]}


def session(prompt, system, tools, schema, model, run_dir, tag, env, cwd):
    cmd = [str(CLAUDE), "-p", prompt, "--model", model, "--output-format", "stream-json", "--verbose",
           "--strict-mcp-config", "--mcp-config", str(run_dir / "mcp.json"), "--system-prompt", system,
           "--json-schema", json.dumps(schema), "--permission-mode", "dontAsk", "--no-session-persistence",
           "--setting-sources", "", "--tools", "", "--allowedTools", *[f"mcp__ese__{t}" for t in tools]]
    with open(run_dir / f"transcript_{tag}.jsonl", "w") as fo, open(run_dir / f"stderr_{tag}.log", "w") as fe:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL)
    out = None
    for line in open(run_dir / f"transcript_{tag}.jsonl"):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("type") == "result":
            out = m.get("structured_output")
    return out


def pipeline(a, run_dir, env, cwd):
    task = TASKS[a.task]
    feedback, final = "", None
    for rnd in (1, 2):
        d = session(task + feedback, DESIGNER + "\nFinish with the requested JSON: your formalisation and your single best DFT-checked candidate.",
                    DESIGN_TOOLS, SCHEMA, a.model, run_dir, f"designer{rnd}", env, cwd)
        if not d or not d.get("strain_gl"):
            final = dict(stage=f"designer{rnd}", error="no candidate", designer=d)
            break
        vprompt = (f"Task given to the design team: {task}\nTheir formalisation: {json.dumps(d.get('formalization'))}\n"
                   f"Their candidate strain (Green-Lagrange, Voigt, tensor shear, fractions): {d['strain_gl']}\n"
                   f"Their claimed h={d.get('h_mev_a3')} meV/A^3, gap={d.get('eg_ev')} eV. Verify independently.")
        v = session(vprompt, VERIFIER, VERIFY_TOOLS, VERIFY_SCHEMA, a.model, run_dir, f"verifier{rnd}", env, cwd)
        final = dict(round=rnd, designer=d, verifier=v)
        if v and v.get("verdict") == "VERIFIED":
            break
        feedback = ("\n\nAn independent verifier REJECTED your previous candidate " + json.dumps(d.get("strain_gl"))
                    + " for these reasons: " + json.dumps((v or {}).get("issues")) + ". Revise and try again.")
    (run_dir / "final.json").write_text(json.dumps(final, indent=1))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("run_id")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--architecture", default="pipeline", choices=["pipeline", "multi", "single"])
    a = ap.parse_args()
    run_dir = Path(os.environ.get("ESE_RUNS_DIR") or ROOT / "runs") / "agent" / f"{a.task}_{a.run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cwd = Path(tempfile.mkdtemp(prefix="ese_agent_"))          # empty, outside the project
    env = dict(os.environ, ESE_RUN_DIR=str(run_dir), ESE_BUDGET=str(a.budget),
               CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT="0", MCP_TOOL_TIMEOUT="10800000",
               PYTHONPATH=str(ROOT))
    mcp = {"mcpServers": {"ese": {"command": str(PY), "args": ["-m", "eseagent.server"],
                                  "env": {"ESE_RUN_DIR": str(run_dir), "ESE_BUDGET": str(a.budget),
                                          "PYTHONPATH": str(ROOT),
                                          **{k: os.environ[k] for k in ("ESE_RUNS_DIR", "ESE_QE_BIN", "ESE_WORKERS")
                                             if k in os.environ}}}}}
    (run_dir / "mcp.json").write_text(json.dumps(mcp))
    allowed = [f"mcp__ese__{t}" for t in sorted(set(DESIGN_TOOLS) | set(VERIFY_TOOLS))]
    cmd = [str(CLAUDE), "-p", TASKS[a.task], "--model", a.model, "--output-format", "stream-json",
           "--verbose", "--strict-mcp-config", "--mcp-config", str(run_dir / "mcp.json"),
           "--system-prompt", SYSTEM if a.architecture == "multi" else SYSTEM.split("- Delegate")[0]
           + DESIGNER, "--json-schema", json.dumps(SCHEMA), "--permission-mode", "dontAsk",
           "--no-session-persistence", "--setting-sources", ""]
    if a.architecture == "multi":
        agents = {"designer": {"description": "Formalises the goal and runs the inverse-design campaign.",
                               "prompt": DESIGNER, "tools": [f"mcp__ese__{t}" for t in DESIGN_TOOLS]},
                  "verifier": {"description": "Independently re-checks candidate answers with DFT.",
                               "prompt": VERIFIER, "tools": [f"mcp__ese__{t}" for t in VERIFY_TOOLS]}}
        cmd += ["--tools", "Agent", "--agents", json.dumps(agents), "--forward-subagent-text",
                "--allowedTools", *allowed, "Agent"]
    else:
        cmd += ["--tools", "", "--allowedTools", *allowed]
    if a.architecture == "pipeline":
        meta = dict(task=a.task, prompt=TASKS[a.task], model=a.model, budget=a.budget,
                    architecture="pipeline", start=time.time())
        final = pipeline(a, run_dir, env, cwd)
        meta.update(end=time.time(), verdict=((final or {}).get("verifier") or {}).get("verdict"))
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(dict(run_dir=str(run_dir), verdict=meta["verdict"], minutes=round((meta["end"] - meta["start"]) / 60, 1))))
        return 0
    meta = dict(task=a.task, prompt=TASKS[a.task], model=a.model, budget=a.budget,
                architecture=a.architecture, start=time.time(), cmd=cmd[:1] + ["..."])
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    with open(run_dir / "transcript.jsonl", "w") as fo, open(run_dir / "stderr.log", "w") as fe:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL)
    meta.update(end=time.time(), returncode=p.returncode)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(dict(run_dir=str(run_dir), rc=p.returncode, minutes=round((meta["end"] - meta["start"]) / 60, 1))))


if __name__ == "__main__":
    sys.exit(main())
