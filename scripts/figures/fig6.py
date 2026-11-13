"""Fig. 6: physics of the verified states.
(a) T2 path (zero strain -> agent's direct-gap state): Gamma minus the next-lowest conduction valley (HSE scissor);
(b) Born stability along the same path, with the DFPT endpoint;
(c) diamond: the agent's H1 state vs uniaxial tensile stress along <100>/<110>/<111> (HSE-scissor gaps)."""
import json
from matplotlib.lines import Line2D
import style as S

R = S.ROOT
res = json.load(open(S.ROOT / "results/results.json"))

v0 = res["validation"]["Si"]; e0 = {k: v0["valleys_pbe_eV"][k] + v0["hse_scissor_eV"][k] for k in v0["valleys_pbe_eV"]}
path = [dict(frac=0.0, G_minus_next=e0["G"] - min(v for k, v in e0.items() if k != "G"),
             born_min_eig_GPa=res["born_ref"]["zero_Si"]["elastic_min_eig_pk2_GPa"])] + \
    [dict(p, born_min_eig_GPa=p["elastic_min_eig_pk2_GPa"]) for p in res["T2_path"]]   # finite-strain (2nd PK) criterion
f = [p["frac"] for p in path]

fig = S.figure("fig6")
gs = fig.add_gridspec(2, 2, width_ratios=[0.42, 0.58])
axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[1, 0], sharex=axa); axc = fig.add_subplot(gs[:, 1])

ax = axa
ax.axhspan(-0.5, 0, color=S.DIRECT, alpha=0.10, lw=0)
ax.plot(f, [p["G_minus_next"] for p in path], color=S.BLUE, marker="o", lw=S.LW_DATA)
ax.axhline(0, **S.REFLINE); ax.text(0.02, -0.1, "Γ is the CBM", fontsize=S.FS_ANNOT, color=S.DIRECT, va="center")
ax.set_ylabel("Γ-valley offset,\n$\\mathit{E}_\\Gamma - \\mathit{E}_\\mathrm{next}$ (eV)")
ax.tick_params(labelbottom=False)
S.grid(ax); S.panel_label(ax, "a")

ax = axb
ax.plot(f, [p["born_min_eig_GPa"] for p in path], color=S.OK_C, marker="o", lw=S.LW_DATA)
ax.axhline(0, **S.REFLINE)
w = res["T2_dfpt"]["omega_min_cm1"]
ax.annotate(f"DFPT: $\\omega_\\mathrm{{min}}$ = +{w:.1f} cm$^{{-1}}$\n(stable, 4³→16³ q)", (1.0, path[-1]["born_min_eig_GPa"]),
            xytext=(0.03, 6), textcoords="data", fontsize=S.FS_LABEL, color=S.OK_C, va="bottom",
            arrowprops=dict(arrowstyle="-", color=S.OK_C, lw=0.5))
ax.set_ylim(-5, 85); ax.set_xlabel("Fraction of path to T2 state")
ax.set_ylabel("Minimum elastic stiffness\neigenvalue (GPa)")
S.grid(ax); S.panel_label(ax, "b")

ax = axc
U = json.load(open(R / "data/uniaxial_C.json"))
hl, ll = [], []
LABOFF = {"100": (5, 0), "110": (5, 0), "111": (5, -2)}
for d, st in S.DIRS.items():
    rows = sorted([u for u in U if u["dir"] == d], key=lambda u: u["axial"])
    rows = [u for u in rows if u["sigma_axial_GPa"] > 0 and u["max_nonaxial_GPa"] < 2.0]  # drop the 18% <111> relaxation that left the uniaxial branch (SI S7)
    imax = max(range(len(rows)), key=lambda i: rows[i]["sigma_axial_GPa"])
    ok, bad = rows[:imax + 1], rows[imax + 1:]                     # beyond the axial-stress maximum: unstable
    ax.plot([u["h"] for u in ok], [u["eg_scissor"] for u in ok], color=st["c"], marker=st["m"], ms=3.5, ls=st["ls"],
            lw=S.LW_DATA)
    ax.annotate(f"$\\langle {d}\\rangle$", (ok[-1]["h"], ok[-1]["eg_scissor"]), xytext=LABOFF[d], textcoords="offset points",
                fontsize=S.FS_ANNOT, fontweight="bold", color=st["c"], va="center",
                bbox=dict(fc="white", ec="none", pad=0.5))
    if bad:
        ax.scatter([u["h"] for u in bad], [u["eg_scissor"] for u in bad], marker="x", s=40, color=st["c"], lw=1.2, zorder=5)
h1 = res["agent_runs"]["H1_s1"]
ax.scatter([h1["h"]], [h1["eg_claimed"]], **S.STAR)
ax.annotate("H1, agent", (h1["h"], h1["eg_claimed"]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
            fontsize=S.FS_ANNOT, fontweight="bold")
ax.axhline(2.0, **S.REFLINE); ax.set_xlim(0, 78); ax.set_ylim(1.7, 5.4)
ax.set_xlabel("$\\mathit{h}$ (meV Å$^{-3}$)"); ax.set_ylabel("$\\mathit{E}_\\mathrm{g}$, HSE scissor (eV)")
ax.set_title("Diamond, uniaxial tensile stress", fontweight="bold")
hl += [Line2D([], [], ls="", marker="x", ms=5, mew=1.2, color="#A3A29C"),
       Line2D([], [], ls="", marker="*", ms=9, mfc=S.GOLD, mec=S.GOLD_EDGE, mew=0.6)]
ll += ["Beyond axial-stress maximum", "H1, agent (scissor level)"]
ax.legend(hl, ll, loc="upper right")
S.grid(ax); S.panel_label(ax, "c")
S.save(fig, "fig6", legacy="fig6_physics")
