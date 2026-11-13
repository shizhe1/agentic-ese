"""Shared figure style (revision plan item D0). Import in every figure script:

    import style as S
    fig = S.figure("fig2")          # 190 mm wide, height from S.HEIGHT_MM
    ...
    S.save(fig, "fig2", legacy="fig2_surrogate")

Fonts: Arial (fallback Liberation Sans), mono DejaVu Sans Mono; 7 pt ticks/legends/annotations,
8 pt axis labels and panel titles, 9 pt bold panel letters; TrueType (Type 42) embedding.
Colours/markers: one fixed entity palette (Okabe-Ito based). Arm colours are used only for arms.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib._mathtext as _mt
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[2]          # repository root
OUT = ROOT / "figures"
QA = ROOT / "figures/qa"                     # text-size records read by check_figs.py

MM = 1 / 25.4
WIDTH_MM = 190.0
HEIGHT_MM = {"fig1": 95, "fig2": 75, "fig3": 125, "fig4": 130, "fig5": 78, "fig6": 95}

# ---- sizes (pt)
FS_TICK = FS_LEGEND = FS_ANNOT = 7
FS_LABEL = FS_TITLE = 8
FS_LETTER = 9
# Sub/superscripts must stay >= 6 pt: mathtext scripts are drawn at SHRINK x the base size, so any text
# that carries a script is set at >= 8 pt (8 x 0.75 = 6.0 pt). check_figs.py enforces this.
SHRINK = 0.75
_mt.SHRINK_FACTOR = SHRINK
MIN_PT, MIN_SCRIPT_PT = 7.0, 6.0

# ---- colours: ONE meaning per colour across all figures
#   blue  BLUE   = the agent / designer / this work      green GREEN = verifier, verified, stable
#   red   RED    = rejected, missed                        purple PURPLE = LLM tools and judges
#   gold star    = verified agent state                    vermilion DIRECT = direct-gap (Gamma CBM) states
#   baselines and comparison loads use orange/pink/brown/greys, never blue, green or purple
BLUE, GREEN, RED, PURPLE = "#0072B2", "#009E73", "#D64545", "#7B5CC4"
GOLD, GOLD_EDGE = "#F2B84B", "#8A6512"
INK, MUTED, GRID = "#1F1E1D", "#6B6A66", "#ECEBE6"
BG_PT = "#BDBDB8"                 # background DFT points (alpha 0.35)
DIRECT = "#D55E00"                # direct-gap (Gamma CBM) points
ARM = {  # colour, marker, linestyle, label
    "agent":     dict(c="#0072B2", m="o", ls="-",  label="Agent"),
    "scripted":  dict(c="#A6611A", m="D", ls="-",  label="Scripted (no LLM)"),
    "expert":    dict(c="#7A7A75", m="s", ls="-",  label="cBO-expert"),
    "agentform": dict(c="#CC79A7", m="^", ls="--", label="cBO-agent"),
    "naive":     dict(c="#E69F00", m="v", ls="-",  label="cBO-naive"),
}
SURR = {  # Fig. 2 surrogates
    "valley":   dict(c=BLUE, m="o", lw=1.6),                  # this work
    "direct":   dict(c="#E07B39", m="s", lw=0.9),
    "bandgrid": dict(c="#7A7A75", m="^", lw=0.9, mfc="white"),
}
DIRS = {  # uniaxial directions (Fig. 6c)
    "100": dict(c="#4D4D4A", m="o", ls="-"),
    "110": dict(c="#E07B39", m="s", ls="--"),
    "111": dict(c="#A3A29C", m="^", ls=":"),
}
JUDGE = {"dark": dict(c=PURPLE, hatch=None), "light": dict(c="#C9B9EA", hatch=None)}
OK_C, MISS_C = GREEN, RED
TINT = {"designer": "#E5F0F8", "verifier": "#EDEDEA", "harness": "#F6EFE0", "user": "#FFFFFF"}

# ---- lines / markers
LW_DATA, LW_SEED, LW_REF = 1.2, 0.6, 0.6
S_SCATTER, S_STAR, S_REFSTAR = 20, 80, 112       # reference star 1.4x the verified star
STAR = dict(marker="*", s=S_STAR * 1.3, color=GOLD, edgecolors=GOLD_EDGE, linewidths=0.6, zorder=7)
REFSTAR = dict(marker="*", s=S_REFSTAR, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=7)
REFLINE = dict(color=INK, lw=LW_REF, ls=":")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.monospace": ["DejaVu Sans Mono"],
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.titleweight": "normal",
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.fallback": "stixsans",
    "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.4,
    "lines.linewidth": LW_DATA, "lines.markersize": 4, "scatter.edgecolors": "face",
    "errorbar.capsize": 2,
    "figure.dpi": 150, "savefig.dpi": 600, "figure.constrained_layout.use": False,
    "axes.unicode_minus": True,
})


def figure(name, height_mm=None, **kw):
    """New figure at the D0 size (190 mm wide) with constrained layout unless layout=None is passed."""
    h = height_mm or HEIGHT_MM[name]
    kw.setdefault("layout", "constrained")
    return plt.figure(figsize=(WIDTH_MM * MM, h * MM), **kw)


def grid(ax, axis="both"):
    ax.grid(True, which="major", axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def panel_label(ax, s, fig=None):
    """Bold panel letter at a fixed offset (-12 pt, +6 pt) from the axes' top-left corner."""
    fig = fig or ax.figure
    return ax.text(0, 1, s, transform=ax.transAxes + ScaledTranslation(-12 / 72, 6 / 72, fig.dpi_scale_trans),
                   fontsize=FS_LETTER, fontweight="bold", va="bottom", ha="right")


def logfmt(ax, axis="x", ticks=None):
    """Plain-number log tick labels (D3) with minor labels suppressed."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def _has_script(s):
    return "$" in s and ("^" in s or "_" in s)


def text_records(fig):
    """All visible, non-empty text in the figure with its size (pt), for check_figs.py."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    out = []
    hidden = set()                    # tick labels of ticks outside the view limits are never drawn
    for ax in fig.findobj(matplotlib.axes.Axes):
        for axis in (ax.xaxis, ax.yaxis):
            drawn = {id(tk) for tk in axis._update_ticks()}
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if id(tk) not in drawn or not ax.axison or not axis.get_visible():
                    hidden.update({id(tk.label1), id(tk.label2)})
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if id(t) in hidden or not t.get_visible() or not s.strip():
            continue
        # skip text clipped away (e.g. tick labels outside the view)
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width == 0:
            continue
        inside = bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= W + 1 and bb.y1 <= H + 1
        out.append(dict(text=s, size=float(t.get_fontsize()), script=_has_script(s),
                        family=t.get_fontfamily(), font=t.get_fontname(), inside=bool(inside)))
    return out


def save(fig, name, legacy=None):
    """Save <name>.png (600 dpi, review) and <name>.pdf (vector, submission); record text sizes for QA."""
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    rec = dict(name=name, size_in=list(fig.get_size_inches()), fonttype=plt.rcParams["pdf.fonttype"],
               shrink=SHRINK, texts=text_records(fig))
    (QA / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    names = [name] + ([legacy] if legacy else [])
    for n in names:
        fig.savefig(OUT / f"{n}.png", dpi=600)
        fig.savefig(OUT / f"{n}.pdf")
    print("saved", ", ".join(names))
