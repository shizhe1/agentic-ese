# agentic-ese

Companion code for the paper

> Zhe Shi, **Agentic Deep Elastic Strain Engineering of Semiconductor Electronic Properties**
> (submitted, 2026).

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
| `scripts/scripted_arm.py` | The scripted, LLM-free arm: the same tools, budget and verification on a fixed schedule. |
| `scripts/figures/` | Scripts that redraw Figs. 1–6 of the paper from the data in this repository. |
| `demo/` | `run_demo.py` and an example of its output. |
| `data/refsets/` | PBE reference sets: Si (319 points), Si small-strain (307) and diamond (316), with valley-resolved band edges and `h`. |
| `data/agent_runs/`, `data/scripted/` | Final answers of the 15 agent runs and 12 scripted runs, plus the full transcripts of the run shown in Fig. 3 (`H2_s3`). |
| `data/learning_curves/`, `data/uniaxial_C.json`, `data/fig5_reference_states.json` | Remaining source data for the figures. |
| `results/results.json` | Summary numbers behind the paper's tables and figures. |
| `preregistration.md` | The pre-registered tasks, pass rules and amendments. |
| `pseudo/` | Norm-conserving ONCVPSP pseudopotentials for Si and C (PseudoDojo). |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The demo needs only `numpy`, `scipy` and `torch`; the figure scripts also need `matplotlib`. Tested with Python 3.12.

## Demo (no DFT code needed)

```bash
python demo/run_demo.py
```

This fits the surrogate on 255 stored PBE points for strained Si, tests it on 64 held-out points,
and then finds the lowest-elastic-energy strain that makes Si direct-gap. It takes about one minute
on a laptop (`--quick` is faster). Example output (also in `demo/expected_output.json`):

```
held-out test: {"n_train": 255, "n_test": 64, "gap_mae_mev": 41.8, "h_mae_mev_a3": 0.064, "cbm_valley_accuracy": 0.984}
lowest-h direct-gap Si (surrogate, not DFT-verified): {"strain_gl": [0.0352, 0.0532, 0.0548, 0.042, -0.0509, -0.0551], "h_pred": 9.21, ...}
```

The ensemble members are trained in parallel threads, so numbers change a little from run to run:
expect a gap error of 41–43 meV and a predicted optimum `h` of roughly 8.7–9.4 meV/Å³ (the
DFT-verified agent states in the paper are at 9.5). The optimum is a surrogate prediction only. In
the paper each candidate is recomputed with DFT and checked by the verifier before it is reported.

## Figures

```bash
cd scripts/figures && for i in 1 2 3 4 5 6; do python fig$i.py; done
```

This writes `figures/fig1.png` … `fig6.png` (and PDFs) from `results/results.json` and `data/`.
No DFT code is needed. The figures use Arial.

Conventions used everywhere: Green-Lagrange strain in Voigt order (e11, e22, e33, e23, e13, e12),
tensor shear, as fractions (0.01 = 1%); `h` is the elastic strain energy density in meV/Å³.

## Running new DFT points and the agents (optional)

- **DFT.** Install Quantum ESPRESSO 7.x (for example `conda install -c conda-forge qe`) so that
  `pw.x` and `ph.x` are on your `PATH`, or set `ESE_QE_BIN` to their folder. Then
  `python -c "from eseagent.dft import compute_point; print(compute_point('Si', [0.01,0,0,0,0,0])['eg_ev'])"`
  computes one point. Results are cached under `runs/cache/`.
- **Scripted arm (no LLM).** `python scripts/scripted_arm.py H2 0` runs one full campaign
  (task `H1`, `H2` or `H3`; then a seed): sampling, DFT, surrogate fitting, optimization, HSE06
  verification and the elastic-stability check. It takes roughly an hour on a laptop and writes
  `runs/scripted/H2_s0/final.json`.
- **Agents.** `scripts/run_agent.py` uses the [Claude Code](https://claude.com/claude-code) CLI
  (`claude` on your `PATH`, or set `CLAUDE_BIN`) and the `mcp` Python package:
  `python scripts/run_agent.py T2 demo1 --model sonnet --budget 150`.
  Each run writes its transcripts and final answer to `runs/agent/<task>_<run>/`.
  A campaign runs many DFT calculations and can take from 20 minutes to a few hours.

Everything computed is written under `runs/` (set `ESE_RUNS_DIR` to put it elsewhere).
`ESE_WORKERS` sets how many DFT calculations run in parallel.

## Citation

See `CITATION.cff`. The citation will be updated when the paper is published.

## License

Code: MIT (see `LICENSE`). The pseudopotentials in `pseudo/` come from
[PseudoDojo](http://www.pseudo-dojo.org) and keep their original license.
