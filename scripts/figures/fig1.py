"""Fig. 1: framework architecture (vector artwork drawn from the harness definition in run_agent.py).
Harness -> designer agent (formalise, active-learning loop, candidate) -> verifier agent (accept/reject, one
feedback round); MCP tool layer below with the designer and verifier tool sets. Icon style: icons.py."""
import re
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import style as S
import icons as I

# tool sets read from the harness, not retyped
src = (S.ROOT / "scripts/run_agent.py").read_text()
DES = re.findall(r'"(\w+)"', re.search(r"DESIGN_TOOLS = \[(.*?)\]", src, re.S).group(1))
VER = re.findall(r'"(\w+)"', re.search(r"VERIFY_TOOLS = \[(.*?)\]", src, re.S).group(1))
VONLY = [t for t in VER if t not in DES]
CAT = {  # tool -> category (glyph colour)
    "describe_material": "dft", "run_dft": "dft", "verify_hse": "check", "elastic_stability": "check",
    "sample_strains": "surr", "fit_surrogate": "surr", "query_surrogate": "surr", "propose_active_learning": "surr",
    "optimize_strain": "opt", "map_envelope": "opt", "log_finding": "log",
}
CATC = {"dft": "#4D4D4A", "surr": "#7B5CC4", "opt": "#E07B39", "check": "#009E73", "log": "#A3A29C"}
CATN = {"dft": "DFT", "surr": "Surrogate", "opt": "Optimizer", "check": "Verification", "log": "Log"}

fig = S.figure("fig1", layout=None)
W, H = fig.get_size_inches() * 72
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")
R = I.ROLE


def tw(s, size=7, family="sans-serif", weight="normal"):
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=family, weight=weight)).get_extents().width


def box(x, y, w, h, fc, ec="none", lw=0.6, r=5, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))


def arrow(p, q, style="-|>", rad=0.0, color=S.INK, lw=1.0, ls="-", z=5, ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def t(x, y, s, size=7, **kw):
    kw.setdefault("va", "top"); kw.setdefault("color", S.INK)
    return ax.text(x, y, s, fontsize=size, zorder=6, **kw)


def chip(x, y, name, owner, tag=None):
    """Tool chip: category dot + mono name (+ tag). Returns width."""
    w = tw(name, family="monospace") + 16
    if tag:
        w += tw(tag) + 6
    box(x, y - 11, w, 11, "white", ec=R[owner]["mid"], lw=0.7, r=5.5, z=3)
    ax.add_patch(Circle((x + 6, y - 5.5), 2.2, fc=CATC[CAT.get(name, "log")], ec="none", zorder=4))
    t(x + 11, y - 2.0, name, family="monospace")
    if tag:
        t(x + w - 4, y - 2.0, tag, ha="right", color=R["verifier"]["c"], style="italic", fontweight="bold")
    return w


# ---------------- geometry (pt)
TOP0, TOP1 = 112, H - 6               # agent row
BOT0, BOT1 = 6, 100                   # tool layer
XU0, XU1 = 6, 80                      # user goal / output columns
XH0, XH1 = 88, W - 76                 # harness frame
XD0, XD1 = XH0 + 8, XH0 + 212         # designer box
XV0, XV1 = XD1 + 50, XH1 - 8          # verifier box
XO0, XO1 = XH1 + 12, W - 6            # output

# harness frame
box(XH0, TOP0, XH1 - XH0, TOP1 - TOP0, R["harness"]["bg"], ec=R["harness"]["mid"], lw=0.8)
pw = I.pill(ax, XH0 + 8, TOP1 - 10, "Harness", R["harness"]["c"])
t(XH0 + 8 + pw + 6, TOP1 - 6.2, "budget ledger · tool permissions · structured hand-off", color="#8A6512")

# user
yb0, yb1 = TOP0 + 18, TOP1 - 22
box(XU0, yb0, XU1 - XU0, yb1 - yb0, R["user"]["bg"], ec=R["user"]["mid"], lw=0.8)
I.human(ax, XU0 + 17, yb1 - 18, 22, R["user"]["c"])
t(XU0 + 32, yb1 - 12, "User", 8, fontweight="bold", color=R["user"]["c"])
box(XU0 + 4, yb0 + 5, XU1 - XU0 - 8, yb1 - yb0 - 42, "white", ec="none", r=5, z=2)
t(XU0 + 8, yb1 - 40, "One-sentence\ngoal, e.g.\n“make Si absorb\nat 1550 nm at the\nlowest strain\nenergy, stable”",
  linespacing=1.25, style="italic")

# designer
DY0, DY1 = TOP0 + 10, TOP1 - 22
box(XD0, DY0, XD1 - XD0, DY1 - DY0, R["designer"]["bg"], ec=R["designer"]["mid"], lw=0.8)
I.robot(ax, XD0 + 14, DY1 - 13, 15, R["designer"]["c"])
I.pill(ax, XD0 + 27, DY1 - 12, "Designer agent", R["designer"]["c"])
sx, sw = XD0 + 8, XD1 - XD0 - 16
stages = [("Formalize", "objective, constraints with thresholds,\nlevel of theory, strain subspace, stability"),
          ("Active-learning loop", "sample → DFT → fit surrogate → optimize\n→ DFT check, within the budget"),
          ("Candidate", "strain, $\\mathit{h}$, gap and CBM valley with DFT evidence")]
sy = DY1 - 26
SH = [30, 30, 20]
for k, ((head, body), hh) in enumerate(zip(stages, SH)):
    box(sx, sy - hh, sw, hh, "white", ec="none", r=4, z=2)
    ax.add_patch(Circle((sx + 9, sy - 8), 5, fc=R["designer"]["c"], ec="none", zorder=4))
    t(sx + 9, sy - 8, str(k + 1), 7, ha="center", va="center", color="white", fontweight="bold")
    t(sx + 18, sy - 3.5, head, 7, fontweight="bold", color=R["designer"]["c"])
    t(sx + 18, sy - 12.5, body, linespacing=1.2)
    sy -= hh + 5
    if k < 2:
        arrow((sx + sw / 2, sy + 5), (sx + sw / 2, sy), lw=0.9, color=R["designer"]["c"], ms=6)
ly = DY1 - 26 - SH[0] - 5
I.loop_arrow(ax, sx + sw - 14, ly - SH[1] / 2, 16, R["designer"]["c"])

# verifier
box(XV0, DY0, XV1 - XV0, DY1 - DY0, R["verifier"]["bg"], ec=R["verifier"]["mid"], lw=0.8)
I.robot(ax, XV0 + 14, DY1 - 13, 15, R["verifier"]["c"])
I.pill(ax, XV0 + 27, DY1 - 12, "Verifier agent", R["verifier"]["c"])
t(XV0 + 8, DY1 - 23, "independent; did not design\nthe candidate", color="#00704F", style="italic", size=7, linespacing=1.15)
vx, vw = XV0 + 8, XV1 - XV0 - 16
vy = DY1 - 44
box(vx, vy - 30, vw, 30, "white", ec="none", r=4, z=2)
I.shield(ax, vx + 9, vy - 9, 12, R["verifier"]["c"])
t(vx + 18, vy - 3.5, "Re-check", 7, fontweight="bold", color=R["verifier"]["c"])
t(vx + 18, vy - 12.5, "full HSE06 band edges;\nelastic stability; units", linespacing=1.2)
vy -= 35
arrow((vx + vw / 2, vy + 5), (vx + vw / 2, vy), lw=0.9, color=R["verifier"]["c"], ms=6)
box(vx, vy - 34, vw, 34, "white", ec="none", r=4, z=2)
t(vx + 6, vy - 3.5, "Verdict", 7, fontweight="bold", color=R["verifier"]["c"])
I.badge(ax, vx + 10, vy - 17, 4.2, ok=True)
t(vx + 17, vy - 13.5, "VERIFIED", 7, fontweight="bold", color=R["verifier"]["c"])
I.badge(ax, vx + 10, vy - 28, 4.2, ok=False)
t(vx + 17, vy - 24.5, "REJECTED", 7, fontweight="bold", color=R["reject"]["c"])

# designer -> verifier (candidate) and one feedback round
YC = DY1 - 60
arrow((XD1, YC), (XV0, YC), lw=1.3, color=R["designer"]["c"])
t((XD1 + XV0) / 2, YC + 8, "candidate", ha="center", va="bottom", size=7, color=R["designer"]["c"], fontweight="bold")
YF = DY0 + 52
arrow((XV0, YF), (XD1, YF), lw=1.1, ls="--", color=R["reject"]["c"])
t((XD1 + XV0) / 2, YF - 3, "reject\n+ reason", ha="center", va="top", color=R["reject"]["c"], linespacing=1.1)
t((XD1 + XV0) / 2, YF - 22, "one\nfeedback\nround", ha="center", va="top", color=S.MUTED, style="italic",
  linespacing=1.1)

# user -> designer ; verifier -> output
ymid = (yb0 + yb1) / 2
arrow((XU1, ymid), (XD0, ymid), lw=1.3, color=R["user"]["c"])
arrow((XV1, ymid), (XO0, ymid), lw=1.3, color=R["verifier"]["c"])
box(XO0, yb0, XO1 - XO0, yb1 - yb0, R["verifier"]["bg"], ec=R["verifier"]["mid"], lw=0.8)
t(XO0 + 6, yb1 - 6, "Output", 8, fontweight="bold", color=R["verifier"]["c"])
ax.scatter([XO0 + 14], [yb1 - 28], marker="*", s=150, color=S.GOLD, edgecolors=S.GOLD_EDGE, linewidths=0.6, zorder=7)
I.badge(ax, XO0 + 24, yb1 - 34, 4.5, ok=True)
t(XO0 + 6, yb1 - 50, "verified\nstrain state\nat full\nHSE06, with\nevidence", linespacing=1.25)

# ---------------- MCP tool layer
box(XU0, BOT0, XO1 - XU0, BOT1 - BOT0, R["tools"]["bg"], ec=R["tools"]["mid"], lw=0.8)
I.pill(ax, XU0 + 6, BOT1 - 11, "MCP tool layer", R["tools"]["c"])
# category key with glyphs
gy = BOT1 - 32
for k, (glyph, cat) in enumerate(((I.crystal, "dft"), (I.network, "surr"), (I.target, "opt"), (I.shield, "check"))):
    yy = gy - k * 16
    glyph(ax, XU0 + 13, yy, 12, CATC[cat])
    t(XU0 + 23, yy + 3.5, CATN[cat], 7, color=CATC[cat], fontweight="bold")


def tool_group(x0, x1, title, tools, ncol, owner):
    t(x0, BOT1 - 6, title, 7.5, fontweight="bold", color=R[owner]["c"])
    colw = (x1 - x0) / ncol
    for i, name in enumerate(tools):
        c, r = i % ncol, i // ncol
        chip(x0 + c * colw, BOT1 - 18 - r * 14, name, owner, "verifier only" if name in VONLY else None)


tool_group(XD0, XD1 + 34, f"Designer tools ({len(DES)})", DES, 2, "designer")
tool_group(XV0, XO1 - 4, f"Verifier tools ({len(VER)})", VER, 1, "verifier")
# agents <-> tools
for x, own in (((XD0 + XD1) / 2, "designer"), ((XV0 + XV1) / 2, "verifier")):
    arrow((x, DY0 - 10), (x, BOT1 + 1), style="<|-|>", lw=1.0, color=R[own]["c"], ms=6)
S.save(fig, "fig1")
