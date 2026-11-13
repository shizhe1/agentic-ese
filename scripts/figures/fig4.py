"""Fig. 4: (a) arm outcomes on the held-out tasks (h relative to the reference; filled = meets the
full-HSE constraint and verified, open = fails); (b) verifier fault-injection outcomes; (c) formalization
rubric scores from two automated judges."""
import json
import os
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import style as S

res = json.load(open(S.ROOT / "results/results.json"))
ARMS = ["agent", "expert", "agentform", "naive", "scripted"]
TICK = {"agent": "Agent", "expert": "cBO-\nexpert", "agentform": "cBO-\nagent", "naive": "cBO-\nnaive",
        "scripted": "Scripted"}
TASKS = ["H1", "H2", "H3"]
# y-range variants (2026-10-28; the author chose A): C = shared 0.38-2.3 (high runs off scale, top edge);
# A = shared 0.7-5.5 (H3 cBO-naive off scale, bottom edge); B = a range per panel, nothing off scale
VAR = os.environ.get("FIG4_VARIANT", "A")
LIM = {"C": dict.fromkeys(TASKS, (0.38, 2.3)), "A": dict.fromkeys(TASKS, (0.7, 5.5)),
       "B": {"H1": (0.7, 5.5), "H2": (0.8, 2.3), "H3": (0.38, 1.5)}}[VAR]

fig = S.figure("fig4")
top, bot = fig.subfigures(2, 1, height_ratios=[1.3, 1], hspace=0.02)
axs = list(top.subplots(1, 3, sharey=VAR != "B"))

for i, (ax, t) in enumerate(zip(axs, TASKS)):
    ref = res["reference"][t]["h"]
    YMIN, YMAX = LIM[t]
    for j, arm in enumerate(ARMS):
        st = S.ARM[arm]
        if arm == "scripted":
            pts = [(v["h"], v["verdict"] == "VERIFIED") for v in res.get("scripted", {}).values() if v["task"] == t and v["h"]]
        elif arm == "agent":
            runs = {k: v for k, v in res["agent_runs"].items() if k.startswith(t)}
            pts = [(v["h"], v["verdict"] == "VERIFIED") for v in runs.values()]   # no star here (2026-10-28)
        elif t == "H1" and arm in ("expert", "naive"):     # corrected two-sided window rerun
            pts = [(v["h_checked"][0], v["passes"]) for k, v in res["scbo_h1_window"].items() if f"_{arm}_" in k]
        else:
            pts = [(v["h"], bool(v["passes"])) for v in res["scbo"].values() if v["task"] == t and v["arm"] == arm]
        if not pts:
            ax.text(j, 0.75, "not run", ha="center", va="center", fontsize=S.FS_ANNOT, color=S.MUTED, style="italic")
        xs = j + np.linspace(-0.15, 0.15, len(pts)) if len(pts) > 1 else [j] * len(pts)
        n_off = 0
        for x, p in zip(xs, pts):
            h, ok = p
            r = h / ref
            if r > YMAX:                                   # off scale: the arm's marker on the top edge, arrow above, value beside
                left = n_off == 0; n_off += 1              # first one left of the column center, the next one right
                x = j + (-0.14 if left else 0.14)
                ax.scatter([x], [YMAX], s=S.S_SCATTER * 1.6, marker=st["m"], zorder=5, lw=0.9, clip_on=False,
                           facecolor=st["c"] if ok else "white", edgecolor=st["c"])
                ax.annotate("", xy=(x, YMAX * 1.16), xytext=(x, YMAX * 1.04), annotation_clip=False, zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=st["c"], lw=1.0, mutation_scale=7,
                                            shrinkA=0, shrinkB=0))
                ax.annotate(f"{r:.1f}×", (x, YMAX), xytext=(-6 if left else 6, 0), textcoords="offset points",
                            ha="right" if left else "left", va="center", fontsize=S.FS_ANNOT, color=st["c"],
                            annotation_clip=False)
            elif r < YMIN:                                 # below scale: marker just inside the bottom edge, arrow down to it
                ym = YMIN * 1.14
                ax.scatter([x], [ym], s=S.S_SCATTER * 1.6, marker=st["m"], zorder=5, lw=0.9,
                           facecolor=st["c"] if ok else "white", edgecolor=st["c"])
                ax.annotate("", xy=(x, YMIN), xytext=(x, YMIN * 1.075), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=st["c"], lw=1.0, mutation_scale=7,
                                            shrinkA=0, shrinkB=0))
                ax.annotate(f"{r:.2f}×", (x, ym), xytext=(7, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=S.FS_ANNOT, color=st["c"], annotation_clip=False)
            else:
                ax.scatter([x], [r], s=S.S_SCATTER * 1.6, marker=st["m"], zorder=4, lw=0.9,
                           facecolor=st["c"] if ok else "white", edgecolor=st["c"])
    ax.axhline(1, **S.REFLINE)
    if t == "H3":                                     # label the reference line where the panel is empty
        ax.text(3, 1.03, "reference", ha="center", va="bottom", fontsize=S.FS_ANNOT, color=S.MUTED)
    ax.set_yscale("log"); ax.set_ylim(YMIN, YMAX)
    S.logfmt(ax, "y", [v for v in (0.5, 0.8, 1, 1.5, 2, 3, 5) if YMIN <= v <= YMAX])
    ax.set_xlim(-0.6, len(ARMS) - 0.4); ax.set_xticks(range(len(ARMS)))
    ax.set_title(t, fontweight="bold")
    S.grid(ax, "y")
    ax.set_xticklabels([TICK[a] for a in ARMS], fontsize=S.FS_TICK)
    ax.tick_params(axis="x", length=0)
    if i == 0:
        ax.set_ylabel("$\\mathit{h}/\\mathit{h}_\\mathrm{ref}$ (lower is better)")
        S.panel_label(ax, "a")
    elif VAR != "B":
        ax.tick_params(labelleft=False)

# figure-level key (arms are named on the x axes): fill = pass/fail, off-scale arrow
kh = [Line2D([], [], ls="", marker="o", mfc=S.INK, mec=S.INK), Line2D([], [], ls="", marker="o", mfc="white", mec=S.INK),
       Line2D([], [], ls="", marker="$\\uparrow$" if VAR == "C" else "$\\downarrow$", ms=9, mfc=S.MUTED, mec=S.MUTED)]
kl = ["meets target", "misses target", "off scale (> 2.3; value given)" if VAR == "C" else "off scale (< 0.7; value given)"]
kh, kl = kh[:2], kl[:2]                                # off-scale entry dropped (author, 2026-10-28): arrow + value suffice
kh[0] = Line2D([], [], ls="", marker="o", mfc=S.MUTED, mec=S.MUTED)
kh[1] = Line2D([], [], ls="", marker="o", mfc="white", mec=S.MUTED)
top.legend(kh, kl, loc="outside upper center", ncol=len(kh), columnspacing=1.8, handletextpad=0.3)

# (b) fault injection: dot matrix, 8 rows x 3 transcripts
axb, axc = bot.subplots(1, 2, width_ratios=[1, 1.1])
pf = res["fault_injection"]["per_fault"]
NAMES = {"pbe_as_hse": "PBE gap reported as HSE", "gap_violates": "Gap violates target", "h_mismatch": "$\\mathit{h}$ misreported",
         "unconverged": "Unconverged SCF", "unstable": "Elastically unstable", "strain_mismatch": "Strain misreported",
         "percent_units": "Strain in percent units", "clean": "Clean control (no fault)"}
order = [k for k in NAMES if k in pf]
y = np.arange(len(order))[::-1]
for yy, k in zip(y, order):
    n, rej = pf[k]["n"], pf[k]["rejected"]
    good = (n - rej) if k == "clean" else rej
    for c in range(n):
        if c < good:
            axb.scatter([c + 1], [yy], s=S.S_SCATTER * 1.1, color=S.OK_C, zorder=3)
        else:
            axb.scatter([c + 1], [yy], s=24, marker="x", color=S.MISS_C, lw=1.3, zorder=3)
    lab = f"{good}/{n} " + ("verified" if k == "clean" else "caught") if good else f"{good}/{n} caught"
    axb.text(n + 0.6, yy, lab, va="center", fontsize=S.FS_ANNOT, color=S.OK_C if good else S.MISS_C)
axb.set_yticks(y); axb.set_yticklabels([NAMES[k] for k in order], fontsize=S.FS_TICK)
axb.set_xlim(0.4, 5.2); axb.set_xticks([1, 2, 3]); axb.set_ylim(-0.7, len(order) - 0.3)
axb.spines["left"].set_visible(False); axb.tick_params(axis="y", length=0)
axb.spines["bottom"].set_bounds(1, 3)
axb.set_xlabel("Replayed transcripts")
S.panel_label(axb, "b")

# (c) formalization rubric
fr = res["formalization_rating"]; crit = ["R1", "R2", "R3", "R4", "R5"]
CL = {"R1": "Minimize $\\mathit{h}$", "R2": "Threshold", "R3": "Level of theory", "R4": "Stability (n = 10)", "R5": "Subspace"}
jst = {fr["judges"][0]: S.JUDGE["light"], fr["judges"][1]: S.JUDGE["dark"]}
for k, j in enumerate(fr["judges"]):
    pc = fr["per_criterion"][j]; yy = np.arange(len(crit))[::-1] + (0.19 if k == 0 else -0.19)
    st = jst[j]
    axc.barh(yy, [pc[c]["frac"] for c in crit], height=0.34, color=st["c"], edgecolor="none",
             lw=0.0, label=f"Claude {j.capitalize()} (judge)")
    for yv, c in zip(yy, crit):
        axc.text(pc[c]["frac"] + 0.02, yv, f"{pc[c]['passed']}/{pc[c]['n']}", va="center", fontsize=S.FS_ANNOT)
axc.set_yticks(np.arange(len(crit))[::-1]); axc.set_yticklabels([CL[c] for c in crit])
axc.set_ylim(-0.6, len(crit) + 0.5)                    # headroom for the legend inside, top-left
axc.set_xlim(0, 1.18); axc.set_xticks([0, 0.5, 1]); axc.set_xlabel("Fraction of answers scored 1")
axc.spines["bottom"].set_bounds(0, 1)
axc.legend(loc="upper left", ncol=2, borderaxespad=0.1, handlelength=1.6, columnspacing=1.0)
S.panel_label(axc, "c")
if VAR == "A":
    S.save(fig, "fig4", legacy="fig4_evaluation")
else:
    S.save(fig, f"fig4_variant{VAR}")
