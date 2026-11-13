"""Fig. 2: held-out gap error vs training size for three surrogates on three reference sets (5 random splits)."""
import json
import os
import numpy as np
import style as S

R = S.ROOT / "data/learning_curves"
d = json.load(open(R / "learning_curves_splits.json"))   # 5 random splits
# Panel (c) extended with the post-campaign diamond states (learning_curves_ext.py, 2026-10-28), shown up to
# MAXN = 1,000 training points (author's choice); the full curve to 6,456 points is SI Table S4.
EXT = os.environ.get("FIG2_EXT", "1") == "1"
MAXN = int(os.environ.get("FIG2_MAXN", 1000))
if EXT:
    for sp in (7, 11, 13, 17, 19):
        d += [r for r in json.load(open(R / f"learning_curves_ext_{sp}.json")) if r["n_train"] <= MAXN]
# Si sets extended the same way (si_extend.py + learning_curves_ext.py, 2026-10-28), also to MAXN (author's choice,
# 2026-10-30); FIG2_SI="" drops them (benchmark-size Si panels)
SI_EXT = [t for t in os.environ.get("FIG2_SI", "Si_small,Si").split(",") if t] if EXT else []
for tag in SI_EXT:
    for sp in (7, 11, 13, 17, 19):
        d += [r for r in json.load(open(R / f"learning_curves_ext_{tag}_{sp}.json")) if r["n_train"] <= MAXN]
LAB = {"valley": "Valley-resolved, equivariant (this work)", "direct": "Direct gap regression",
       "bandgrid": "Band on fixed k-grid (simplified)"}
OFF = {"valley": 1.0, "direct": 0.97, "bandgrid": 1.03}            # +-3 % offset in log-x for the baselines
titles = {"Si_small": "Si, small strain (|ε| ≤ 3%)", "Si": "Si, wide range (|ε| ≤ 10%)", "C": "Diamond (|ε| ≤ 10%)"}
YT = {"Si_small": [10, 20, 50], "Si": [50, 100, 200], "C": [100, 200, 500]}

fig = S.figure("fig2")
axs = fig.subplots(1, 3)
for ax, s, lab in zip(axs, ("Si_small", "Si", "C"), "abc"):
    for m, st in S.SURR.items():
        rows = [r for r in d if r["set"] == s and r["model"] == m]
        ns = sorted({r["n_train"] for r in rows})
        mu = np.array([np.mean([r["mae_meV"] for r in rows if r["n_train"] == n]) for n in ns])
        sd = np.array([np.std([r["mae_meV"] for r in rows if r["n_train"] == n]) for n in ns])
        x = np.array(ns) * OFF[m]
        if m == "valley":     # filled +-1 s.d. band for this work only
            ax.fill_between(ns, mu - sd, mu + sd, color=st["c"], alpha=0.15, lw=0)
            ax.plot(ns, mu, color=st["c"], marker=st["m"], lw=st["lw"], label=LAB[m], zorder=4)
        else:                 # baselines as error bars
            ax.errorbar(x, mu, yerr=sd, color=st["c"], marker=st["m"], lw=st["lw"], elinewidth=0.6, capsize=1.5,
                        mfc=st.get("mfc", st["c"]), mec=st["c"], mew=0.8, ms=4, label=LAB[m], zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(titles[s], fontweight="bold")
    ax.set_xlabel("DFT training points")
    S.panel_label(ax, lab); S.grid(ax)
    S.logfmt(ax, "x", [t for t in (50, 250, 1000, 5000) if t <= MAXN] if ((EXT and s == "C") or s in SI_EXT) else [50, 100, 250])
    S.logfmt(ax, "y", YT[s])
axs[0].set_ylabel("Held-out gap MAE (meV)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, columnspacing=2.5, handlelength=2.2, labelcolor="linecolor")
if EXT and MAXN == 1000 and sorted(SI_EXT) == ["Si", "Si_small"]:
    S.save(fig, "fig2", legacy="fig2_surrogate")
else:
    S.save(fig, f"fig2_variant_{MAXN if EXT else 'noext'}" + ("_si" + "_".join(SI_EXT) if EXT else ""))
