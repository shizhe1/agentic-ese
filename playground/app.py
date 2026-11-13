"""ESE playground: a small local web app on top of the valley-resolved surrogate.

    pip install -r playground/requirements.txt
    python playground/app.py            # then open http://127.0.0.1:7860

Two tabs:
  * Forward model:  set a strain tensor and see the predicted band edges, gap and h (updates live).
  * Inverse design: pick a target (gap window, direct gap, ...) and get the lowest-h strains.

Everything here is a SURROGATE prediction. The surrogates ship pre-trained (playground/models/, trained by
playground/surrogates.py on about 2,000 Si and 7,400 diamond PBE points from the paper), so the app starts
in seconds. No DFT and no language model run in the playground; in the paper each candidate is re-checked
with DFT and HSE06 by an independent verifier before it is reported.
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
from matplotlib.gridspec import GridSpec  # noqa: E402

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
LEVELS = ["HSE06 (scissor)", "PBE"]

# ---- palette (one meaning per colour) -------------------------------------------------------------
INK, MUTED, GRID = "#0f172a", "#64748b", "#e2e8f0"
FAM = {"G": "#c2410c", "L": "#1d4ed8", "D": "#0f766e"}
FAM_NAME = {"G": "Γ", "L": "L", "D": "Δ"}
VFAM = {v: ("G" if v == "G" else v[0]) for v in VALLEYS}
VLABEL = {"G": "Γ", **{f"L{i}": f"L{i}" for i in range(1, 5)}, "Dx": "Δx", "Dy": "Δy", "Dz": "Δz"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": "#94a3b8", "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 10, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "savefig.facecolor": "white", "figure.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})

_models: dict[str, ValleySurrogate] = {}
_lock = threading.Lock()


def model(mat: str) -> ValleySurrogate:
    with _lock:
        if mat not in _models:
            _models[mat] = surrogates.load(mat)
        return _models[mat]


# ---- physics helpers ------------------------------------------------------------------------------
def evaluate(mat: str, e: np.ndarray, level: str) -> dict:
    p = model(mat).predict(e[None, :])
    shift = SCISSOR[mat] if level.startswith("HSE") else {v: 0.0 for v in VALLEYS}
    valleys = {v: float(p["valleys"][v][0]) + shift[v] for v in VALLEYS}
    cbm = min(valleys, key=valleys.get)
    return dict(valleys=valleys, cbm=cbm, eg=valleys[cbm], h=float(p["h"][0]), h_std=float(p["h_std"][0]),
                eg_std=float(p["eg_std"][0]))


def tensor(e: np.ndarray) -> np.ndarray:
    return np.array([[e[0], e[5], e[4]], [e[5], e[1], e[3]], [e[4], e[3], e[2]]])


def character(r: dict) -> str:
    if r["eg"] <= 0:
        return "Metallic (gap closed)"
    return "Direct gap at Γ" if r["cbm"] == "G" else "Indirect gap"


def in_range(mat: str, e: np.ndarray) -> bool:
    lim = MATERIALS[mat]
    return np.abs(e[:3]).max() <= lim["normal"] + 1e-9 and np.abs(e[3:]).max() <= lim["shear"] + 1e-9


# ---- figures --------------------------------------------------------------------------------------
def figure(mat: str, e: np.ndarray, r: dict, level: str):
    fig = plt.figure(figsize=(10.0, 4.7), dpi=160)
    gs = GridSpec(1, 2, width_ratios=[2.25, 1], wspace=0.22, left=0.075, right=0.965, top=0.9, bottom=0.2)
    ax, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) band-edge level diagram
    xpos = {"G": 0.0, "L1": 1.3, "L2": 1.9, "L3": 2.5, "L4": 3.1, "Dx": 4.4, "Dy": 5.0, "Dz": 5.6}
    vals = r["valleys"]
    lo = min(-0.35, min(vals.values()) - 0.25)
    hi = max(vals.values()) + 0.45
    ax.axhspan(lo, 0, color="#f1f5f9", zorder=0)
    ax.axhline(0, color=INK, lw=1.2, zorder=2)
    ax.text(6.85, 0.012 * (hi - lo), "VBM at Γ", va="bottom", ha="right", fontsize=8, color=INK)
    ax.text(6.85, lo / 2, "valence band", va="center", ha="right", fontsize=8, color=MUTED, style="italic")
    for v, x in xpos.items():
        c = FAM[VFAM[v]]
        is_cbm = v == r["cbm"]
        ax.hlines(vals[v], x - 0.24, x + 0.24, color=c, lw=4.2 if is_cbm else 2.6, zorder=4,
                  capstyle="round")
        ax.text(x, vals[v] + 0.05 * (hi - lo), f"{vals[v]:.2f}", ha="center", va="bottom", fontsize=7.5,
                color=c, family="monospace")
    # gap arrow to the CBM
    xc, ec = xpos[r["cbm"]], r["eg"]
    col = "#b91c1c" if ec <= 0 else INK
    ax.annotate("", xy=(xc + 0.36, ec), xytext=(xc + 0.36, 0),
                arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.1, shrinkA=0, shrinkB=0, mutation_scale=9))
    ax.text(xc + 0.44, ec / 2, f"$E_g$ = {ec:.3f} eV", va="center", ha="left", fontsize=9, color=col,
            fontweight="semibold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xticks(list(xpos.values()), [VLABEL[v] for v in xpos])
    for lab, v in zip(ax.get_xticklabels(), xpos):
        lab.set_color(FAM[VFAM[v]])
    for fam, (x0, x1) in {"G": (-0.4, 0.4), "L": (1.0, 3.4), "D": (4.1, 5.9)}.items():
        ax.annotate("", xy=(x0, -0.1), xytext=(x1, -0.1), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=FAM[fam], lw=1.0))
        ax.text((x0 + x1) / 2, -0.125, {"G": "Γ valley", "L": "L valleys (×4)", "D": "Δ valleys (×3)"}[fam],
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=FAM[fam])
    ax.set_xlim(-0.6, 6.9)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Energy relative to VBM (eV)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_title(f"a   Conduction-band valleys · {MATERIALS[mat]['label']} · {level}", loc="left")

    # (b) strain tensor map
    t = tensor(e) * 100
    vmax = max(2.0, float(np.abs(t).max()))
    im = axt.imshow(t, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(3):
        for j in range(3):
            axt.text(j, i, f"{t[i, j]:+.2f}", ha="center", va="center", fontsize=9, family="monospace",
                     color="white" if abs(t[i, j]) > 0.6 * vmax else INK)
    axt.set_xticks(range(3), ["x", "y", "z"])
    axt.set_yticks(range(3), ["x", "y", "z"])
    axt.tick_params(length=0)
    for s in axt.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axt, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.set_label("strain (%)", fontsize=8, color=MUTED)
    axt.set_title("b   Strain tensor ε (Green-Lagrange)", loc="left")
    return fig


# ---- HTML pieces ----------------------------------------------------------------------------------
def card(label, value, sub="", tone="", text=False):
    return (f'<div class="ese-card {tone}"><div class="ese-card-label">{label}</div>'
            f'<div class="ese-card-value{" txt" if text else ""}">{value}</div><div class="ese-card-sub">{sub}</div></div>')


def cards(mat, e, r, level):
    tone = "bad" if r["eg"] <= 0 else ("good" if r["cbm"] == "G" else "")
    warn = "" if in_range(mat, e) else ('<div class="ese-warn">Outside the training range of the surrogate. '
                                        'Treat this prediction with extra caution.</div>')
    return ('<div class="ese-cards">'
            + card("Band gap E<sub>g</sub>", f"{r['eg']:.3f} <span>eV</span>",
                   f"{level} · ensemble ± {r['eg_std'] * 1000:.0f} meV", tone)
            + card("Character", character(r), f"CBM valley: {VLABEL[r['cbm']]}", tone, text=True)
            + card("Strain energy density h", f"{r['h']:.3f} <span>meV Å<sup>−3</sup></span>",
                   f"ensemble ± {r['h_std']:.3f}")
            + card("Strain magnitude ‖ε‖", f"{np.linalg.norm(tensor(e)) * 100:.2f} <span>%</span>",
                   f"max |ε<sub>ij</sub>| = {np.abs(e).max() * 100:.2f} %")
            + "</div>" + warn)


# ---- callbacks ------------------------------------------------------------------------------------
def predict(mat, level, *vals):
    e = np.array([float(v) / 100.0 for v in vals])
    r = evaluate(mat, e, level)
    fig = figure(mat, e, r, level)
    return cards(mat, e, r, level), fig


PRESETS = {
    "Unstrained crystal": [0, 0, 0, 0, 0, 0],
    "[001] uniaxial compression (Si gap closes)": [-8.5, 2.65, 2.65, 0, 0, 0],
    "(001) biaxial tension": [3, 3, -2.3, 0, 0, 0],
    "Triaxial tension with shear (direct gap)": [5.9, 6.5, 5.8, 3.6, -4.3, -4.0],
    "[111] uniaxial tension": [2.4, 2.4, 2.4, 3.3, 3.3, 3.3],
}


def apply_preset(name):
    return [gr.update(value=v) for v in PRESETS[name]]


GOALS = {
    "Close the gap (metal)": lambda x: [dict(kind="gap_max", value=0.0)],
    "Gap ≤ target": lambda x: [dict(kind="gap_max", value=x)],
    "Gap ≥ target": lambda x: [dict(kind="gap_min", value=x)],
    "Gap = target ± 0.03 eV": lambda x: [dict(kind="gap_min", value=x - 0.03), dict(kind="gap_max", value=x + 0.03)],
    "Direct gap at Γ (margin = target)": lambda x: [dict(kind="direct", value=max(x, 0.0))],
}
SUBSPACES = {"Full 6D strain": (None, None), "Normal strains only": ([0, 1, 2], None),
             "Thin film (e33 = 0)": ([0, 1, 3, 4, 5], {2: 0.0})}


def design(mat, goal, x, level, subspace, bound_pct, n_starts):
    cons = GOALS[goal](float(x))
    free, fixed = SUBSPACES[subspace]
    shift = SCISSOR[mat] if level.startswith("HSE") else None
    sols = optimize_strain(model(mat), cons, free=free, fixed=fixed, n_starts=int(n_starts),
                           bounds=(-bound_pct / 100, bound_pct / 100), shift=shift, seed=0)
    if not sols:
        return ('<div class="ese-warn">No feasible strain found within these bounds. Try a larger bound, '
                'another subspace or a looser target.</div>', None, None)
    rows = [[i + 1, round(s["h_pred"], 3), round(s["eg_pred"], 3), VLABEL[s["cbm_pred"]],
             *np.round(np.array(s["strain_gl"]) * 100, 2).tolist()] for i, s in enumerate(sols)]
    e = np.array(sols[0]["strain_gl"])
    r = evaluate(mat, e, level)
    head = ('<div class="ese-note">Lowest-energy solution of the constrained problem on the surrogate. '
            'It is a prediction, not a DFT-verified result: re-check it with '
            '<code>eseagent.dft.compute_point</code> before relying on it.</div>')
    return head + cards(mat, e, r, level), figure(mat, e, r, level), rows


# ---- layout ---------------------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
#ese-hero {background: radial-gradient(1200px 300px at 0% 0%, #1e3a8a 0%, #0f172a 55%, #020617 100%);
  border-radius: 14px; padding: 26px 30px 22px; color: #e2e8f0; position: relative; overflow: hidden;}
#ese-hero:after {content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .18;
  background-image: radial-gradient(#93c5fd 1px, transparent 1px); background-size: 18px 18px;
  mask-image: linear-gradient(90deg, transparent 35%, black 100%);}
#ese-hero h1 {font-size: 28px; font-weight: 650; letter-spacing: -0.02em; margin: 0 0 6px; color: #f8fafc;}
#ese-hero p {margin: 0; max-width: 860px; color: #cbd5e1; font-size: 14.5px; line-height: 1.55;}
.ese-eyebrow {font-family: var(--font-mono); font-size: 11.5px; letter-spacing: .14em; text-transform: uppercase;
  color: #93c5fd; margin-bottom: 8px;}
.ese-pills {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.ese-pill {font-family: var(--font-mono); font-size: 11.5px; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(148,163,184,.35); background: rgba(15,23,42,.45); color: #e2e8f0;}
.ese-pill b {color: #93c5fd; font-weight: 600;}
.ese-panel-title {font-size: 12px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: #475569; margin: 2px 0 4px;}
.ese-sub {font-size: 12.5px; font-weight: 600; color: #64748b; margin: 0 0 -4px 2px;}
.ese-cards {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px;}
@media (max-width: 900px) {.ese-cards {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.ese-card {border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; background: #fff;
  border-top: 3px solid #334155;}
.ese-card.good {border-top-color: #0f766e;} .ese-card.bad {border-top-color: #b91c1c;}
.ese-card-label {font-size: 12.5px; font-weight: 600; color: #475569;}
.ese-card-value {font-family: var(--font-mono); font-size: 21px; font-weight: 600; color: #0f172a;
  margin-top: 4px; line-height: 1.25;}
.ese-card-value.txt {font-family: var(--font); font-size: 17px; line-height: 1.3;}
.ese-card-value span {font-size: 12.5px; color: #64748b; font-weight: 500;}
.ese-card-sub {font-size: 12px; color: #64748b; margin-top: 4px;}
.ese-warn {margin-top: 10px; padding: 9px 12px; border-radius: 8px; background: #fef2f2; color: #991b1b;
  border: 1px solid #fecaca; font-size: 13px;}
.ese-note {margin-bottom: 10px; padding: 9px 12px; border-radius: 8px; background: #f8fafc; color: #334155;
  border: 1px solid #e2e8f0; font-size: 13px;}
#ese-foot {color: #64748b; font-size: 12.5px; border-top: 1px solid #e2e8f0; padding-top: 10px;}
footer {display: none !important;}
"""

THEME = gr.themes.Base(primary_hue=gr.themes.colors.indigo, secondary_hue=gr.themes.colors.slate,
                       neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_sm,
                       spacing_size=gr.themes.sizes.spacing_md, text_size=gr.themes.sizes.text_md).set(
    body_background_fill="#f8fafc", block_background_fill="#ffffff", block_border_width="1px",
    block_border_color="#e2e8f0", block_shadow="0 1px 2px rgba(15,23,42,.04)", block_label_text_weight="600",
    block_title_text_weight="600", button_primary_background_fill="#1e3a8a",
    button_primary_background_fill_hover="#1e40af", button_primary_text_color="#ffffff",
    slider_color="#1e3a8a", input_background_fill="#ffffff")

FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"


def hero() -> str:
    s, c = STATS["Si"], STATS["C"]
    return f"""
<div id="ese-hero">
  <div class="ese-eyebrow">Agentic deep elastic strain engineering · companion playground</div>
  <h1>Strain → band structure, in real time</h1>
  <p>Set a six-component strain tensor and see how the conduction-band valleys of silicon and diamond move,
  or ask for the lowest-energy strain that reaches a target band gap. Powered by the valley-resolved,
  Oₕ-equivariant surrogate from the paper; no DFT runs here.</p>
  <div class="ese-pills">
    <span class="ese-pill"><b>Si</b> {s['n_train']:,} PBE points · held-out gap MAE {s['test_gap_mae_mev']:.0f} meV</span>
    <span class="ese-pill"><b>C</b> {c['n_train']:,} PBE points · held-out gap MAE {c['test_gap_mae_mev']:.0f} meV</span>
    <span class="ese-pill"><b>ε</b> Green-Lagrange · Voigt · tensor shear</span>
    <span class="ese-pill"><b>HSE06</b> per-valley scissor</span>
  </div>
</div>"""


def build() -> gr.Blocks:
    with gr.Blocks(title="ESE playground") as demo:
        gr.HTML(hero())
        with gr.Tabs():
            with gr.Tab("Forward model"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=340):
                        gr.HTML('<div class="ese-panel-title">Crystal and level of theory</div>')
                        with gr.Row():
                            mat = gr.Radio(list(MATERIALS), value="Si", label="Material", scale=1)
                            level = gr.Radio(LEVELS, value=LEVELS[0], label="Band edges", scale=2)
                        gr.HTML('<div class="ese-panel-title">Strain tensor (%)</div>')
                        preset = gr.Dropdown(list(PRESETS), value="Unstrained crystal", label="Preset")
                        with gr.Row():
                            with gr.Column(min_width=150):
                                gr.HTML('<div class="ese-sub">Normal</div>')
                                sl = [gr.Slider(-10, 10, value=0, step=0.05, label=f"ε{c[1:]}") for c in COMP[:3]]
                            with gr.Column(min_width=150):
                                gr.HTML('<div class="ese-sub">Shear (tensor)</div>')
                                sl += [gr.Slider(-8, 8, value=0, step=0.05, label=f"ε{c[1:]}") for c in COMP[3:]]
                    with gr.Column(scale=7):
                        out_html = gr.HTML()
                        out_plot = gr.Plot(show_label=False)
                inputs = [mat, level, *sl]
                preset.change(apply_preset, preset, sl).then(predict, inputs, [out_html, out_plot])
                for comp in sl:
                    comp.release(predict, inputs, [out_html, out_plot])
                for comp in (mat, level):
                    comp.change(predict, inputs, [out_html, out_plot])
                demo.load(predict, inputs, [out_html, out_plot])
            with gr.Tab("Inverse design"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=340):
                        gr.HTML('<div class="ese-panel-title">Design problem</div>')
                        with gr.Row():
                            mat2 = gr.Radio(list(MATERIALS), value="Si", label="Material", scale=1)
                            level2 = gr.Radio(LEVELS, value=LEVELS[0], label="Band edges", scale=2)
                        goal = gr.Dropdown(list(GOALS), value="Direct gap at Γ (margin = target)", label="Constraint")
                        x = gr.Number(value=0.02, label="Target (eV)")
                        sub = gr.Dropdown(list(SUBSPACES), value="Full 6D strain", label="Strain subspace")
                        gr.HTML('<div class="ese-panel-title">Search</div>')
                        bound = gr.Slider(1, 10, value=8, step=0.5, label="Bound on each component (%)")
                        starts = gr.Slider(4, 48, value=16, step=4, label="Multi-start SLSQP restarts")
                        btn = gr.Button("Minimise elastic energy  →", variant="primary", size="lg")
                    with gr.Column(scale=7):
                        d_html = gr.HTML('<div class="ese-note">Objective: minimise the elastic strain energy '
                                         'density h subject to the band-edge constraint. Choose a problem on '
                                         'the left and run the search (about 15 s).</div>')
                        d_plot = gr.Plot(show_label=False)
                        d_tab = gr.Dataframe(headers=["#", "h (meV Å⁻³)", "Eg (eV)", "CBM",
                                                      *[f"ε{c[1:]} %" for c in COMP]],
                                             label="Candidates, lowest h first", interactive=False)
                btn.click(design, [mat2, goal, x, level2, sub, bound, starts], [d_html, d_plot, d_tab])
        gr.HTML('<div id="ese-foot">Surrogate predictions only (PBE valley energies; HSE06 via a per-valley '
                'scissor fitted at zero strain). In the paper every reported state is recomputed with DFT, '
                'checked at full HSE06 and for elastic stability by an independent verifier agent. '
                'Code and data: see the repository README.</div>')
    return demo


if __name__ == "__main__":
    for m in MATERIALS:
        model(m)
    build().launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)),
                   theme=THEME, css=CSS, js=FORCE_LIGHT)
