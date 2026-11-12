# agentic-ese

Companion code for the paper

> Zhe Shi, **Agentic Deep Elastic Strain Engineering of Semiconductor Electronic Properties**
> (submitted to *Computational Materials Science*, 2026).

A language-model "designer" agent turns a one-sentence goal (for example *"make silicon a
direct-gap semiconductor at the lowest elastic strain energy"*) into a constrained inverse problem
and solves it with first-principles active learning on a valley-resolved, symmetry-equivariant
surrogate. An independent "verifier" agent re-checks each candidate with DFT, HSE06 band edges and
elastic stability. This repository holds the toolkit, the tool server the agents call, the agent
harness, and a small demo that runs in about a minute without any DFT code.

It is a lightweight companion, not a full reproduction package: the raw Quantum ESPRESSO outputs
and the DFT cache (about 31 GB) are not included.

## Contents

| Path | What it is |
| --- | --- |
| `eseagent/` | The toolkit: strain conventions, Quantum ESPRESSO driver, per-strain DFT evaluation with valley-resolved band edges, HSE06 correction, phonon and Born stability checks, the surrogate, the constrained optimizer, the Bayesian-optimization baseline, and the MCP tool server (`server.py`). |
| `scripts/run_agent.py` | Harness that launches one isolated designer → verifier agent campaign. |
| `demo/` | `run_demo.py`, its input data (319 PBE points for strained Si) and the expected output. |
| `results/results.json` | Summary numbers behind the paper's tables and figures. |
| `preregistration.md` | The pre-registered tasks, pass rules and amendments. |
| `pseudo/` | Norm-conserving ONCVPSP pseudopotentials for Si and C (PseudoDojo). |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The demo needs only `numpy`, `scipy` and `torch`. Tested with Python 3.12.

## Demo (no DFT code needed)

```bash
python demo/run_demo.py
```

This fits the surrogate on 255 stored PBE points for strained Si, tests it on 64 held-out points,
and then finds the lowest-elastic-energy strain that makes Si direct-gap. It takes about one minute
on a laptop (`--quick` is faster) and should print numbers close to `demo/expected_output.json`:

```
held-out test: {"n_train": 255, "n_test": 64, "gap_mae_mev": 42.1, "h_mae_mev_a3": 0.063, "cbm_valley_accuracy": 0.984}
lowest-h direct-gap Si (surrogate, not DFT-verified): {"strain_gl": [0.0435, 0.0367, 0.0809, 0.0385, 0.0374, 0.0535], "h_pred": 9.35, ...}
```

The optimum is a surrogate prediction only. In the paper each candidate is recomputed with DFT
and checked by the verifier before it is reported.

Conventions used everywhere: Green-Lagrange strain in Voigt order (e11, e22, e33, e23, e13, e12),
tensor shear, as fractions (0.01 = 1%); `h` is the elastic strain energy density in meV/Å³.

## Running new DFT points and the agents (optional)

- **DFT.** Install Quantum ESPRESSO 7.x (for example `conda install -c conda-forge qe`) so that
  `pw.x` and `ph.x` are on your `PATH`, or set `ESE_QE_BIN` to their folder. Then
  `python -c "from eseagent.dft import compute_point; print(compute_point('Si', [0.01,0,0,0,0,0])['eg_ev'])"`
  computes one point. Results are cached under `runs/cache/`.
- **Agents.** `scripts/run_agent.py` uses the [Claude Code](https://claude.com/claude-code) CLI
  (`claude` on your `PATH`, or set `CLAUDE_BIN`) and the `mcp` Python package:
  `python scripts/run_agent.py T2 demo1 --model sonnet --budget 150`.
  Each run writes its transcripts and final answer to `runs/agent/<task>_<run>/`.
  A campaign runs many DFT calculations and can take hours.

## Citation

See `CITATION.cff`. The citation will be updated when the paper is published.

## License

Code: MIT (see `LICENSE`). The pseudopotentials in `pseudo/` come from
[PseudoDojo](http://www.pseudo-dojo.org) and keep their original license.
