"""Fig. 3: (a) verbatim excerpt of the H2 run selected by the pre-registered rule (H2_s3);
(b) anytime best h vs DFT budget for the cBO arms on H2, with the agent runs' verified results."""
import json, re
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.lines import Line2D
import style as S
import icons as I
from matplotlib.patches import Circle, Polygon

R = S.ROOT
res = json.load(open(S.ROOT / "results/results.json"))
RUN = R / "data/agent_runs/H2_s3"

# ---- verbatim sources
src = RUN.joinpath("transcript_designer1.jsonl").read_text() + RUN.joinpath("transcript_verifier1.jsonl").read_text()
task = re.search(r'"H2": "([^"]+)"', (R / "scripts/run_agent.py").read_text()).group(1)
fin = json.load(open(RUN / "final.json")); de, ve = fin["designer"], fin["verifier"]
expl = de["explanation"]


def q(s, text=expl):
    assert s in text, f"not verbatim: {s!r}"
    return s


for s in ("70.9", f"{ve['eg_ev']}", "0.0288"):
    assert s in src, s

SANS, MONO = "sans-serif", "monospace"
# (header, [(body text, family)], tint)
boxes = [
    ("User", [(f"“{task}”", SANS)], "user"),
    ("Designer · tools and formalization",
     [("describe_material → sample_strains → run_dft ×20 → fit_surrogate (hold-out gap MAE 28.8 meV) → "
       "optimize_strain(gap_max 0.78 eV, level hse_scissor) → run_dft ×2", MONO),
      ("“" + q(de["formalization"]["constraints"][0], de["formalization"]["constraints"][0]) + "”", SANS)], "designer"),
    ("Designer · reasoning",
     [("“" + q("Compressing along z lowers Dz relative to Dx and Dy, so the gap closes at small strain.")
       + " … " + q("The gap is indirect, so absorption near 0.75 eV is phonon-assisted and weak.") + " … "
       + q("The tool permissions for both were denied, so the 0.753 eV is a scissor estimate and stability is unconfirmed.")
       + " " + q("For that reason I set constraints_satisfied to false.") + "”", SANS)], "designer"),
    ("Harness", [("verify_hse, elastic_stability: denied to the designer (verifier-only tools)", MONO)], "harness"),
    ("Verifier · tools",
     [("elastic_stability → min stiffness eigenvalue 70.9 GPa (stable)", MONO),
      (f"verify_hse → Eg(HSE06) = {ve['eg_ev']:.3f} eV, CBM Δz", MONO),
      (f"verdict: {ve['verdict']}", MONO)], "verifier"),
]

FS_BODY, FS_HEAD, LS = 7.0, 7.5, 1.22


def width_pt(s, family, size, weight="normal"):
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=family, weight=weight)).get_extents().width


def wrap(s, family, size, maxw):
    """Greedy wrap on measured glyph widths (breaks at spaces)."""
    out, cur = [], ""
    for w in s.split(" "):
        t = (cur + " " + w) if cur else w
        if width_pt(t, family, size) <= maxw or not cur:
            cur = t
        else:
            out.append(cur); cur = w
    return out + [cur]


fig = S.figure("fig3", layout=None)
Wpt, Hpt = fig.get_size_inches() * 72
FA = 0.50                                            # width ratio of panel a
axa = fig.add_axes([0, 0, FA, 1]); axa.axis("off"); axa.set_xlim(0, FA * Wpt); axa.set_ylim(0, Hpt)
# hidden anchor axes for the panel letter (D0): same -12/+6 pt offset from the box column's top-left
axa.text(3, Hpt - 3, "a", fontsize=S.FS_LETTER, fontweight="bold", ha="left", va="top")
PAD, X0, X1, GAP = 4.0, 34.0, FA * Wpt - 6, 5.0      # pt; PAD = 1.5 mm; avatars in x < X0
AV = 26 / 2 + 6                                      # avatar centre x
TOOLS = set(re.findall(r"[a-z]+_[a-z_]+", boxes[1][1][0][0] + boxes[3][1][0][0] + boxes[4][1][0][0]))
CW = width_pt("x" * 40, MONO, FS_BODY) / 40          # monospace advance


def tok_color(tk):
    if tk in TOOLS:
        return I.CODE_FN
    if tk in ("VERIFIED",):
        return I.ROLE["verifier"]["mid"]
    if re.fullmatch(r"×?\d+(\.\d+)?", tk):
        return I.CODE_NUM
    if re.fullmatch(r"[a-z]+_[a-z]+|level|verdict|min|stiffness|eigenvalue", tk):
        return I.CODE_ARG
    if tk in ("→", "(", ")", ",", ":", "=") :
        return I.CODE_CMT
    return I.CODE_FG


def code_line(ax, x, y, line):
    col = 0
    for tk in re.findall(r"×?\d+(?:\.\d+)?|\w+|\s+|.", line):
        if not tk.isspace():
            ax.text(x + col * CW, y, tk, fontsize=FS_BODY, family=MONO, va="top", color=tok_color(tk), zorder=5)
        col += len(tk)


def avatar(ax, role, x, y):
    if role == "user":
        I.human(ax, x, y, 20, I.ROLE["user"]["c"])
    elif role == "harness":
        ax.add_patch(Circle((x, y), 9.5, fc=I.ROLE["harness"]["c"], ec="none", zorder=4))
        I.shield(ax, x, y, 11, "white", z=5)
    else:
        I.robot(ax, x, y, 15, I.ROLE[role]["c"])


y = Hpt - 18
for head, parts, role in boxes:
    rc = I.ROLE[role]
    inner = X1 - X0 - 2 * PAD
    blocks = []                                      # (family, [lines])
    for txt, fam in parts:                           # consecutive code parts share one block
        ls = wrap(txt, fam, FS_BODY, inner - (8 if fam == MONO else 0))
        if blocks and fam == MONO and blocks[-1][0] == MONO:
            blocks[-1][1].extend(ls)
        else:
            blocks.append((fam, ls))
    lh = FS_BODY * LS
    hgt = PAD + 15 + sum(lh * len(ls) + (6 if fam == MONO else 0) + 2 for fam, ls in blocks) + PAD - 2
    axa.add_patch(FancyBboxPatch((X0, y - hgt), X1 - X0, hgt, boxstyle="round,pad=0,rounding_size=5",
                                 fc=rc["bg"], ec=rc["mid"], lw=0.8, zorder=1))
    # speech-bubble tail toward the avatar
    axa.add_patch(Polygon([(X0 + 0.4, y - 9), (X0 - 6, y - 13), (X0 + 0.4, y - 17)], closed=True, fc=rc["bg"],
                          ec="none", zorder=2))
    axa.plot([X0, X0 - 6, X0], [y - 9, y - 13, y - 17], color=rc["mid"], lw=0.8, zorder=2, solid_joinstyle="miter")
    avatar(axa, role, AV, y - 13)
    hd = head.split(" · ")
    pw = I.pill(axa, X0 + PAD, y - PAD - 5.5, hd[0], rc["c"], size=7, h=11, pad=3.5)
    if len(hd) > 1:
        axa.text(X0 + PAD + pw + 4, y - PAD - 1.5, hd[1], fontsize=FS_BODY, color=rc["c"], fontweight="bold", va="top")
    yy = y - PAD - 15
    for fam, ls in blocks:
        if fam == MONO:
            bh = lh * len(ls) + 6
            axa.add_patch(FancyBboxPatch((X0 + PAD, yy - bh), inner, bh, boxstyle="round,pad=0,rounding_size=3",
                                         fc=I.CODE_BG, ec="none", zorder=3))
            yy -= 3
            for l in ls:
                code_line(axa, X0 + PAD + 4, yy, l)
                if l.startswith("verdict: VERIFIED"):
                    I.badge(axa, X0 + PAD + 4 + (len(l) + 1.8) * CW, yy - FS_BODY * 0.55, 3.8, ok=True, z=6)
                yy -= lh
            yy -= 3 + 2
        else:
            for l in ls:
                axa.text(X0 + PAD, yy, l, fontsize=FS_BODY, family=fam, va="top", color=S.INK,
                         style="italic" if l.startswith("“") or role != "user" else "normal", zorder=5)
                yy -= lh
            yy -= 2
    y -= hgt + GAP
print("panel a slack pt", round(y, 1)); assert y > 0, "panel a overflows"

# ---- panel b
axb = fig.add_axes([FA + 0.085, 0.10, 1 - FA - 0.10, 0.86])
fig.text(FA + 0.012, 1 - 3 / Hpt, "b", fontsize=S.FS_LETTER, fontweight="bold", ha="left", va="top")
sc = {k: v for k, v in res["scbo"].items() if v["task"] == "H2"}
ref = res["reference"]["H2"]["h"]


def draw_b(ax, inset=False):
    exp = []
    for k, v in sc.items():
        a = v["anytime"]; x = range(1, len(a) + 1); st = S.ARM[v["arm"]]
        if v["arm"] == "expert":
            exp.append(a)
            ax.plot(x, a, color=st["c"], lw=S.LW_SEED, alpha=0.6, drawstyle="steps-post")
        else:
            ax.plot(x, a, color=st["c"], lw=S.LW_DATA, ls=st["ls"], drawstyle="steps-post", zorder=3)
            if not inset:
                ax.plot([len(a)], [a[-1]], marker=st["m"], color=st["c"], ls="", ms=4, zorder=4)
    med = np.median(np.array(exp), axis=0)
    ax.plot(range(1, len(med) + 1), med, color=S.ARM["expert"]["c"], lw=S.LW_DATA, drawstyle="steps-post", zorder=3)
    if not inset:
        ax.plot([len(med)], [med[-1]], marker=S.ARM["expert"]["m"], color=S.ARM["expert"]["c"], ls="", ms=4, zorder=4)
    ag = {k: v for k, v in res["agent_runs"].items() if k.startswith("H2") and v["verdict"] == "VERIFIED"}
    best = min(ag, key=lambda k: ag[k]["h"])
    oth = [v for k, v in ag.items() if k != best]
    ax.scatter([v["spent"] for v in oth], [v["h"] for v in oth], s=S.S_SCATTER, marker="o", color=S.ARM["agent"]["c"],
               edgecolors="white", linewidths=0.5, zorder=6)
    ax.scatter([ag[best]["spent"]], [ag[best]["h"]], **S.STAR)
    scr = [v for v in res.get("scripted", {}).values() if v["task"] == "H2" and v["verdict"] == "VERIFIED"]
    ax.scatter([v["spent"] for v in scr], [v["h"] for v in scr], s=S.S_SCATTER, marker="D", color=S.ARM["scripted"]["c"],
               edgecolors="white", linewidths=0.5, zorder=6)
    ax.axhline(ref, **S.REFLINE)
    return ag, best


ag, best = draw_b(axb)
allmax = max(max(v["anytime"]) for v in sc.values())
axb.axvline(100, color=S.GRID, lw=0.6, zorder=0)
axb.set_yscale("log"); axb.set_xlim(0, 118); axb.set_ylim(0.22, allmax * 1.5)
S.logfmt(axb, "y", [0.3, 1, 3, 10])
axb.set_xlabel("DFT budget units"); axb.set_ylabel("Best $\\mathit{h}$ found (meV Å$^{-3}$)")
S.grid(axb)

axb.annotate("agent cap 150 →", xy=(1.0, 0.0), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
             ha="right", va="bottom", fontsize=S.FS_ANNOT, color=S.MUTED)
axb.text(99, allmax * 1.4, "cBO budget", fontsize=S.FS_ANNOT, color=S.MUTED, ha="right", va="top")

axb.annotate("H2_s3 (panel a)", (ag["H2_s3"]["spent"], ag["H2_s3"]["h"]), xytext=(-38, -2), textcoords="offset points",
             ha="right", va="center", fontsize=S.FS_ANNOT, color=S.MUTED, arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.5, shrinkB=3))
axb.text(2, ref * 0.97, "reference", fontsize=S.FS_ANNOT, color=S.MUTED, ha="left", va="top")
axb.annotate("H2, agent", (ag[best]["spent"], ag[best]["h"]), xytext=(0, 9), textcoords="offset points",
             fontsize=S.FS_ANNOT, fontweight="bold", ha="center", va="bottom")

A = S.ARM
hl = [Line2D([], [], color=A["expert"]["c"], lw=S.LW_SEED, alpha=0.6),
      Line2D([], [], color=A["expert"]["c"], lw=S.LW_DATA, marker=A["expert"]["m"]),
      Line2D([], [], color=A["agentform"]["c"], lw=S.LW_DATA, ls="--", marker=A["agentform"]["m"]),
      Line2D([], [], color=A["naive"]["c"], lw=S.LW_DATA, marker=A["naive"]["m"]),
      Line2D([], [], color=A["agent"]["c"], ls="", marker="o", mec="white", mew=0.5),
      Line2D([], [], color=S.GOLD, ls="", marker="*", ms=9, mec=S.GOLD_EDGE, mew=0.6),
      Line2D([], [], color=A["scripted"]["c"], ls="", marker="D", mec="white", mew=0.5)]
ll = ["cBO-expert, 5 seeds", "cBO-expert, median", "cBO-agent formalization", "cBO-naive formalization",
      "Agent, verified (Sonnet, 3 runs)", "Agent, verified (Opus; lowest $\\mathit{h}$)", "Scripted (no LLM), 3 seeds"]
axb.legend(hl, ll, loc="upper right", bbox_to_anchor=(1.0, 0.93), ncol=1, handlelength=1.8, borderaxespad=0.3,
           facecolor="white", framealpha=1.0, frameon=True, edgecolor="none", labelcolor=[h.get_color() if i != 5 else S.INK for i, h in enumerate(hl)])
S.save(fig, "fig3", legacy="fig3_campaign")
