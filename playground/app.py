"""ESE playground: a small local web app on top of the valley-resolved surrogate.

    pip install -r playground/requirements.txt
    python playground/app.py            # then open http://127.0.0.1:7860

Two tabs:
  * Predict:        set a strain with sliders and see the predicted band edges, gap and h.
  * Inverse design: pick a target (gap window, direct gap, ...) and get the lowest-h strains.

Everything here is a SURROGATE prediction. The surrogates ship pre-trained (playground/models/, trained by
playground/surrogates.py on about 2,000 Si and 7,000 diamond PBE points from the paper), so the app starts
in seconds.
No DFT and no language model run in the playground; in the paper each candidate is re-checked with
DFT and HSE06 by an independent verifier before it is reported.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eseagent.optimize import optimize_strain  # noqa: E402
from eseagent.surrogate import VALLEYS, ValleySurrogate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surrogates  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

MATERIALS = {   # training range (GL strain, |normal| and |tensor shear|)
    "Si": dict(label="Silicon", normal=0.10, shear=0.06),
    "C": dict(label="Diamond", normal=0.10, shear=0.08),
}
STATS = {m: json.load(open(ROOT / "playground" / "models" / f"{m}.json")) for m in MATERIALS}
_res = json.load(open(ROOT / "results" / "results.json"))
SCISSOR = {m: _res["validation"][m]["hse_scissor_eV"] for m in MATERIALS}   # per-valley HSE06 shift (eV)
COMP = ["e11", "e22", "e33", "e23", "e13", "e12"]
VCOLOR = {"G": "#D55E00", **{f"L{i}": "#0072B2" for i in range(1, 5)}, **{v: "#009E73" for v in ("Dx", "Dy", "Dz")}}

_models: dict[str, ValleySurrogate] = {}
_lock = threading.Lock()


def model(mat: str) -> ValleySurrogate:
    with _lock:
        if mat not in _models:
            _models[mat] = surrogates.load(mat)
        return _models[mat]


def _strain(vals) -> np.ndarray:
    return np.array([float(v) / 100.0 for v in vals])          # sliders are in percent


def _valley_plot(valleys: dict, title: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=120)
    names = VALLEYS
    e = [valleys[v] for v in names]
    cbm = min(names, key=lambda v: valleys[v])
    bars = ax.bar(names, e, color=[VCOLOR[v] for v in names], alpha=0.85)
    bars[names.index(cbm)].set_edgecolor("black")
    bars[names.index(cbm)].set_linewidth(2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Valley energy above VBM (eV)")
    ax.set_title(title, fontsize=10)
    ax.text(0.99, 0.97, f"CBM: {cbm}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _in_range(mat, e) -> str:
    lim = MATERIALS[mat]
    if np.abs(e[:3]).max() > lim["normal"] + 1e-9 or np.abs(e[3:]).max() > lim["shear"] + 1e-9:
        return " ⚠️ Outside the training range: treat with extra caution."
    return ""


def predict(mat, level, *vals):
    e = _strain(vals)
    p = model(mat).predict(e[None, :])
    shift = SCISSOR[mat] if level == "HSE06 (scissor)" else {v: 0.0 for v in VALLEYS}
    valleys = {v: float(p["valleys"][v][0]) + shift[v] for v in VALLEYS}
    cbm = min(valleys, key=valleys.get)
    eg = valleys[cbm]
    kind = "metal (gap closed)" if eg <= 0 else ("direct gap at Γ" if cbm == "G" else f"indirect gap, CBM at {cbm}")
    md = (f"### {MATERIALS[mat]['label']}: E<sub>g</sub> = {eg:.3f} eV ({level})\n"
          f"- **Character:** {kind}\n"
          f"- **Elastic strain energy density h:** {float(p['h'][0]):.3f} ± {float(p['h_std'][0]):.3f} meV/Å³\n"
          f"- **Ensemble spread of the PBE gap:** ± {float(p['eg_std'][0]) * 1000:.0f} meV"
          + _in_range(mat, e))
    return md, _valley_plot(valleys, f"{MATERIALS[mat]['label']}, strain (%) = {np.round(e * 100, 2).tolist()}")


GOALS = {
    "Close the gap (metal)": lambda x: [dict(kind="gap_max", value=0.0)],
    "Gap at most x eV": lambda x: [dict(kind="gap_max", value=x)],
    "Gap at least x eV": lambda x: [dict(kind="gap_min", value=x)],
    "Gap equal to x eV (± 0.03)": lambda x: [dict(kind="gap_min", value=x - 0.03), dict(kind="gap_max", value=x + 0.03)],
    "Direct gap at Γ (margin x eV)": lambda x: [dict(kind="direct", value=max(x, 0.0))],
}
SUBSPACES = {"Full 6D": (None, None), "Normal strains only": ([0, 1, 2], None),
             "Thin film, e33 = 0": ([0, 1, 3, 4, 5], {2: 0.0})}


def design(mat, goal, x, level, subspace, bound_pct, n_starts):
    cons = GOALS[goal](float(x))
    free, fixed = SUBSPACES[subspace]
    shift = SCISSOR[mat] if level == "HSE06 (scissor)" else None
    sols = optimize_strain(model(mat), cons, free=free, fixed=fixed, n_starts=int(n_starts),
                           bounds=(-bound_pct / 100, bound_pct / 100), shift=shift, seed=0)
    if not sols:
        return ("No feasible strain found within these bounds. Try a larger bound, another subspace or a "
                "looser target.", None, None)
    rows = [[i + 1, round(s["h_pred"], 3), round(s["eg_pred"], 3), s["cbm_pred"],
             *np.round(np.array(s["strain_gl"]) * 100, 2).tolist()] for i, s in enumerate(sols)]
    best = sols[0]
    e = np.array(best["strain_gl"])
    p = model(mat).predict(e[None, :])
    sh = shift or {v: 0.0 for v in VALLEYS}
    valleys = {v: float(p["valleys"][v][0]) + sh[v] for v in VALLEYS}
    md = (f"### Lowest-h strain: h ≈ {best['h_pred']:.3f} meV/Å³, E<sub>g</sub> ≈ {best['eg_pred']:.3f} eV "
          f"({level}), CBM at {best['cbm_pred']}\n"
          f"Strain (Green-Lagrange, %, Voigt e11 e22 e33 e23 e13 e12, tensor shear): "
          f"`{np.round(e * 100, 2).tolist()}`" + _in_range(mat, e) + "\n\n"
          "This is a surrogate optimum, not a DFT-verified result. Re-check it with DFT "
          "(`eseagent.dft.compute_point`) before relying on it.")
    return md, rows, _valley_plot(valleys, "Best candidate: predicted valley energies")


def build() -> gr.Blocks:
    with gr.Blocks(title="ESE playground") as demo:
        gr.Markdown(
            "# Deep elastic strain engineering playground\n"
            "Explore how a six-component strain changes the band gap of **silicon** and **diamond**, "
            "using the valley-resolved surrogate from the paper. "
            "Strain is Green-Lagrange in percent, Voigt order, tensor shear. *h* is the elastic strain "
            "energy density. Predictions only: nothing here runs DFT. Held-out PBE gap error: "
            f"Si {STATS['Si']['test_gap_mae_mev']:.0f} meV, diamond {STATS['C']['test_gap_mae_mev']:.0f} meV.")
        with gr.Tab("Predict"):
            with gr.Row():
                mat = gr.Radio(list(MATERIALS), value="Si", label="Material")
                level = gr.Radio(["PBE", "HSE06 (scissor)"], value="HSE06 (scissor)", label="Level of theory")
            with gr.Row():
                sl = [gr.Slider(-10, 10, value=0, step=0.1, label=f"{c} (%)") for c in COMP[:3]]
            with gr.Row():
                sl += [gr.Slider(-8, 8, value=0, step=0.1, label=f"{c} (%)") for c in COMP[3:]]
            btn = gr.Button("Predict", variant="primary")
            out_md = gr.Markdown()
            out_plot = gr.Plot()
            btn.click(predict, [mat, level, *sl], [out_md, out_plot])
            gr.Examples([["Si", "HSE06 (scissor)", 0, 0, 0, 0, 0, 0],
                         ["Si", "HSE06 (scissor)", -8.5, 2.65, 2.65, 0, 0, 0],
                         ["C", "HSE06 (scissor)", 6, 6, 6, 3, 3, 3]],
                        [mat, level, *sl], label="Examples: unstrained Si, Si under [001] compression, diamond under triaxial tension with shear")
        with gr.Tab("Inverse design"):
            with gr.Row():
                mat2 = gr.Radio(list(MATERIALS), value="Si", label="Material")
                level2 = gr.Radio(["PBE", "HSE06 (scissor)"], value="HSE06 (scissor)", label="Level of theory")
            with gr.Row():
                goal = gr.Dropdown(list(GOALS), value="Direct gap at Γ (margin x eV)", label="Target")
                x = gr.Number(value=0.02, label="x (eV)")
                sub = gr.Dropdown(list(SUBSPACES), value="Full 6D", label="Strain subspace")
            with gr.Row():
                bound = gr.Slider(1, 10, value=8, step=0.5, label="Bound on each component (%)")
                starts = gr.Slider(4, 48, value=16, step=4, label="Optimizer restarts")
            btn2 = gr.Button("Find the lowest-energy strain", variant="primary")
            d_md = gr.Markdown()
            d_tab = gr.Dataframe(headers=["#", "h (meV/Å³)", "Eg (eV)", "CBM", *[f"{c} %" for c in COMP]],
                                 label="Candidates, lowest h first")
            d_plot = gr.Plot()
            btn2.click(design, [mat2, goal, x, level2, sub, bound, starts], [d_md, d_tab, d_plot])
        gr.Markdown("Code and paper: see the repository README. Surrogate predictions are not DFT results.")
    return demo


if __name__ == "__main__":
    for m in MATERIALS:
        model(m)
    build().launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)))
