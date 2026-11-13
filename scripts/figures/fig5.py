"""Fig. 5: HSE-scissor gap vs elastic energy density h from DFT reference points (Si, diamond),
direct-gap states highlighted; verified agent optima (full HSE06) and references added from results.json."""
import json
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import style as S

R = S.ROOT; DATA = R / "data"
res = json.load(open(S.ROOT / "results/results.json"))
NBINS = 15                                            # equal-count bins in h for the lower envelope


def load(tags, mat):
    sc = res["validation"][mat]["hse_scissor_eV"]; pts = []
    for t in tags:
        for c in json.load(open(DATA / "refsets" / f"{t}.json")):
            v = {k: c["valleys"][k]["e"] + sc[k] for k in c["valleys"]}
            cbm = min(v, key=v.get); pts.append((c["h_mev_a3"], v[cbm], cbm == "G"))
    return np.array(pts)


TMAT = {"T1": "Si", "T2": "Si", "H2": "Si", "H3": "Si", "H1": "C"}


def best_states(mat):
    out = {}
    for t, m in TMAT.items():
        if m != mat: continue
        ag = [v for k, v in res["agent_runs"].items() if k.startswith(t) and v["verdict"] == "VERIFIED"]
        out[t] = min(ag, key=lambda v: v["h"])
    return out


def scissor_gap(mat, strain):
    """HSE-scissor gap of a stored PBE state (no new DFT: computed in its campaign)."""
    c = next(r for r in json.load(open(DATA / "fig5_reference_states.json"))
             if r["material"] == mat and np.allclose(r["strain_gl"], strain, atol=1e-6))
    sc = res["validation"][mat]["hse_scissor_eV"]
    return min(c["valleys"][k]["e"] + sc[k] for k in c["valleys"])


def pair(ax, h, eg_sc, eg_full, ref=False):
    """Star at the scissor level of the map (y-axis), bar at the same state's full-HSE06 gap (2026-10-28 fix)."""
    ax.plot([h, h], [eg_sc, eg_full], color=S.INK, lw=0.6, zorder=5)
    ax.scatter([h], [eg_full], marker="_", s=110, color=S.INK, lw=1.5, zorder=8)
    ax.scatter([h], [eg_sc], **(S.REFSTAR if ref else S.STAR))


def draw(ax, mat, p, inset=False):
    h, eg, d = p[:, 0], p[:, 1], p[:, 2].astype(bool)
    ax.scatter(h[~d], eg[~d], s=4, color=S.BG_PT, alpha=0.35, lw=0, rasterized=True)
    if d.any():
        ax.scatter(h[d], eg[d], s=S.S_SCATTER * 0.6, color=S.DIRECT, lw=0, rasterized=True)
    bins = np.quantile(h, np.linspace(0, 1, NBINS + 1))
    lo = [(0.5 * (a + b), eg[(h >= a) & (h < b)].min()) for a, b in zip(bins[:-1], bins[1:]) if ((h >= a) & (h < b)).any()]
    ax.plot(*zip(*lo), color=S.INK, lw=S.LW_DATA, drawstyle="steps-mid", zorder=3)   # lo holds bin centres
    ax.axhline(0, **S.REFLINE)
    for t, b in best_states(mat).items():
        pair(ax, b["h"], b["eg_claimed"], b["eg_verifier"])          # eg_claimed = designer's scissor-level gap
        rf = res["reference"].get(t)
        if rf and rf["kind"] == "scbo":
            sc = res["scbo"][rf["source"]]
            pair(ax, sc["h"], scissor_gap(mat, sc["strain_gl"]), sc["eg_hse"], ref=True)


fig = S.figure("fig5")
axs = fig.subplots(1, 2)
for ax, (mat, tags, lab, title) in zip(axs, [("Si", ["Si", "Si_small"], "a", "Si"), ("C", ["C"], "b", "Diamond")]):
    p = load(tags, mat)
    draw(ax, mat, p)
    ax.set_xscale("log")
    S.logfmt(ax, "x", [0.1, 1, 10] if mat == "Si" else [10, 20, 30, 100, 300])
    ax.set_xlabel("$\\mathit{h}$ (meV Å$^{-3}$)"); ax.set_ylabel("$\\mathit{E}_\\mathrm{g}$, HSE scissor (eV)"); ax.set_title(title, fontweight="bold")
    S.panel_label(ax, lab); S.grid(ax)
    bs = best_states(mat)
    hl = [Line2D([], [], ls="", marker="o", ms=3, mfc=S.BG_PT, mec="none"),
          Line2D([], [], color=S.INK, lw=S.LW_DATA, drawstyle="steps-mid"),
          Line2D([], [], ls="", marker="*", ms=9, mfc=S.GOLD, mec=S.GOLD_EDGE, mew=0.6)]
    ll = ["CBM not at Γ (DFT)", f"Lower envelope ({NBINS} equal-count bins)", "Verified agent state (scissor level)"]
    hl.append(Line2D([], [], ls="", marker="_", ms=8, mec=S.INK, mew=1.5)); ll.append("Same state at full HSE06")
    if p[:, 2].astype(bool).any():
        hl.insert(1, Line2D([], [], ls="", marker="o", ms=3.5, mfc=S.DIRECT, mec="none")); ll.insert(1, "Γ CBM (DFT)")
    if mat == "Si":
        for t, b in bs.items():
            if t in ("T1", "H3"): continue
            ax.annotate(f"{t}, agent", (b["h"], b["eg_claimed"]), xytext=(9, 4), textcoords="offset points",
                        fontsize=S.FS_ANNOT, fontweight="bold")
        hl.append(Line2D([], [], ls="", marker="*", ms=10, mfc="none", mec=S.INK, mew=0.8)); ll.append("H2 reference (cBO-naive)")
        # T1 and H3 nearly coincide: one label with a leader line
        t1, h3 = bs["T1"], bs["H3"]
        ax.annotate("T1 and H3, agent\n(overlapping)", (np.sqrt(t1["h"] * h3["h"]), 0.5 * (t1["eg_claimed"] + h3["eg_claimed"])),
                    xytext=(0.9, -0.85), textcoords="data", ha="right", va="center", fontsize=S.FS_ANNOT, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.5, shrinkB=4))
        LEG = dict(zip(ll, hl))
    else:
        b = bs["H1"]
        ax.axhspan(1.97, 2.03, color=S.BG_PT, alpha=0.35, lw=0, zorder=1)
        for yv in (1.97, 2.03):
            ax.axhline(yv, color=S.MUTED, lw=0.5, ls="--", zorder=1)
        ax.annotate("H1, agent", (b["h"], b["eg_claimed"]), xytext=(-8, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=S.FS_ANNOT, fontweight="bold")
        hl += [Patch(fc=S.BG_PT, alpha=0.35, ec=S.MUTED, ls="--", lw=0.5)]
        ll += ["Target band, 2.0 ± 0.03 eV"]
        ax.text(0.99, 2.03, "target 2.0 ± 0.03 eV", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=S.FS_ANNOT, color=S.MUTED)
        for k, v in zip(ll, hl):
            LEG.setdefault(k, v)
order = ["CBM not at Γ (DFT)", "Γ CBM (DFT)", f"Lower envelope ({NBINS} equal-count bins)",
         "Verified agent state (scissor level)", "Same state at full HSE06", "H2 reference (cBO-naive)"]
fig.legend([LEG[k] for k in order], order, loc="outside upper center", ncol=3, columnspacing=2.0)
S.save(fig, "fig5", legacy="fig5_envelopes")
