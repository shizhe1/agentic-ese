"""Flat vector icons and colour roles for the agentic schematics (Fig. 1, Fig. 3a, graphical abstract).
All icons are drawn with matplotlib patches in data units (pt) of an axes whose x and y scales are equal."""
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle, Wedge, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

# colour roles: (strong, light panel, mid)
ROLE = {
    "user":     dict(c="#E07B39", bg="#FDF0E6", mid="#F3C29D"),
    "designer": dict(c="#0072B2", bg="#E6F1F8", mid="#9CC6E3"),
    "verifier": dict(c="#009E73", bg="#E4F5EF", mid="#97D8C2"),
    "harness":  dict(c="#D9961A", bg="#FDF6E6", mid="#F2D49A"),
    "tools":    dict(c="#7B5CC4", bg="#F3EFFB", mid="#C9B9EA"),
    "reject":   dict(c="#D64545", bg="#FBEAEA", mid="#EFB0B0"),
}
CODE_BG, CODE_FG = "#25272E", "#E8E8E3"
CODE_FN, CODE_ARG, CODE_NUM, CODE_CMT = "#F2B84B", "#8FD18F", "#7FC4F2", "#9A9AA0"
SKIN, HAIR = "#F2C7A0", "#3A2E28"


def robot(ax, x, y, s, color, z=10):
    """Robot head centred at (x, y), height ~ s."""
    w, h = 1.05 * s, 0.82 * s
    ax.add_line(Line2D([x, x], [y + h / 2, y + h / 2 + 0.16 * s], color="#4A4A4A", lw=0.8, zorder=z))
    ax.add_patch(Circle((x, y + h / 2 + 0.2 * s), 0.07 * s, fc=color, ec="none", zorder=z + 1))
    for sx in (-1, 1):
        ax.add_patch(FancyBboxPatch((x + sx * w / 2 - (0.1 * s if sx > 0 else 0) + (0 if sx > 0 else -0.0), y - 0.14 * s),
                                    0.1 * s, 0.28 * s, boxstyle=f"round,pad=0,rounding_size={0.03 * s}",
                                    fc="#5A5A5A", ec="none", zorder=z - 1, transform=ax.transData))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={0.2 * s}",
                                fc=color, ec="none", zorder=z))
    fw, fh = 0.78 * s, 0.48 * s
    ax.add_patch(FancyBboxPatch((x - fw / 2, y - fh / 2 + 0.02 * s), fw, fh, boxstyle=f"round,pad=0,rounding_size={0.12 * s}",
                                fc="#20232A", ec="none", zorder=z + 1))
    for sx in (-1, 1):
        ax.add_patch(Ellipse((x + sx * 0.17 * s, y + 0.05 * s), 0.13 * s, 0.17 * s, fc="white", ec="none", zorder=z + 2))
    ax.add_patch(Wedge((x, y - 0.06 * s), 0.1 * s, 200, 340, width=0.035 * s, fc="white", ec="none", zorder=z + 2))


def human(ax, x, y, s, shirt, z=10):
    """Bust centred at (x, y), height ~ s."""
    ax.add_patch(Wedge((x, y - 0.5 * s), 0.46 * s, 0, 180, fc=shirt, ec="none", zorder=z))
    ax.add_patch(Circle((x, y + 0.12 * s), 0.25 * s, fc=SKIN, ec="none", zorder=z + 1))
    ax.add_patch(Wedge((x, y + 0.15 * s), 0.27 * s, 15, 165, fc=HAIR, ec="none", zorder=z + 2))
    for sx in (-1, 1):
        ax.add_patch(Circle((x + sx * 0.09 * s, y + 0.08 * s), 0.03 * s, fc=HAIR, ec="none", zorder=z + 3))


def badge(ax, x, y, r, ok=True, z=12):
    """Round check (ok) or cross badge."""
    c = ROLE["verifier"]["c"] if ok else ROLE["reject"]["c"]
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=0.8, zorder=z))
    if ok:
        ax.add_line(Line2D([x - 0.45 * r, x - 0.1 * r, x + 0.5 * r], [y, y - 0.38 * r, y + 0.4 * r], color="white",
                           lw=1.2, solid_capstyle="round", zorder=z + 1))
    else:
        for d in (1, -1):
            ax.add_line(Line2D([x - 0.4 * r, x + 0.4 * r], [y - d * 0.4 * r, y + d * 0.4 * r], color="white", lw=1.2,
                               solid_capstyle="round", zorder=z + 1))


def pill(ax, x, y, text, color, size=7.5, pad=4, h=12, z=12, ha="left"):
    """Filled rounded label; (x, y) = left (or centre) of its baseline box. Returns width."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    w = TextPath((0, 0), text, size=size, prop=FontProperties(family="sans-serif", weight="bold")).get_extents().width + 2 * pad
    x0 = x - w / 2 if ha == "center" else x
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}", fc=color,
                                ec="none", zorder=z))
    ax.text(x0 + w / 2, y, text, fontsize=size, fontweight="bold", color="white", ha="center", va="center_baseline",
            zorder=z + 1)
    return w


# ---- tool-category glyphs (s = size)
def crystal(ax, x, y, s, color, z=12):
    pts = [(-0.35, -0.3), (0.35, -0.3), (0.35, 0.3), (-0.35, 0.3), (0, 0)]
    for a, b in ((0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (1, 2), (2, 3), (3, 0)):
        ax.add_line(Line2D([x + pts[a][0] * s, x + pts[b][0] * s], [y + pts[a][1] * s, y + pts[b][1] * s], color=color,
                           lw=0.7, zorder=z))
    for i, (px, py) in enumerate(pts):
        ax.add_patch(Circle((x + px * s, y + py * s), (0.13 if i == 4 else 0.1) * s, fc=color if i == 4 else "white",
                            ec=color, lw=0.8, zorder=z + 1))


def network(ax, x, y, s, color, z=12):
    L = [[-0.25, 0.25], [-0.35, 0, 0.35], [0]]
    xs = [-0.4, 0, 0.4]
    for i in range(2):
        for a in L[i]:
            for b in L[i + 1]:
                ax.add_line(Line2D([x + xs[i] * s, x + xs[i + 1] * s], [y + a * s, y + b * s], color=color, lw=0.5,
                                   alpha=0.7, zorder=z))
    for i, col in enumerate(L):
        for a in col:
            ax.add_patch(Circle((x + xs[i] * s, y + a * s), 0.09 * s, fc=color, ec="white", lw=0.5, zorder=z + 1))


def target(ax, x, y, s, color, z=12):
    for r, fc in ((0.42, "white"), (0.28, color), (0.14, "white")):
        ax.add_patch(Circle((x, y), r * s, fc=fc, ec=color, lw=0.8, zorder=z))
    ax.add_patch(Circle((x, y), 0.05 * s, fc=color, ec="none", zorder=z + 1))


def shield(ax, x, y, s, color, z=12):
    p = np.array([(0, 0.45), (0.38, 0.32), (0.32, -0.12), (0, -0.45), (-0.32, -0.12), (-0.38, 0.32)]) * s
    ax.add_patch(Polygon(p + [x, y], closed=True, fc=color, ec="none", zorder=z))
    ax.add_line(Line2D([x - 0.15 * s, x - 0.03 * s, x + 0.18 * s], [y, y - 0.13 * s, y + 0.14 * s], color="white", lw=1.1,
                       solid_capstyle="round", zorder=z + 1))


def loop_arrow(ax, x, y, s, color, z=12):
    """Circular 'repeat' arrow of diameter ~ s centred at (x, y)."""
    from matplotlib.patches import Arc
    r = 0.4 * s
    ax.add_patch(Arc((x, y), 2 * r, 2 * r, theta1=40, theta2=330, color=color, lw=1.1, zorder=z))
    a = np.deg2rad(40)
    tip = np.array([x + r * np.cos(a), y + r * np.sin(a)])
    d = np.array([-np.sin(a), np.cos(a)])          # counter-clockwise tangent
    p = [tip + 0.0 * d, tip - 0.35 * s * d + 0.18 * s * np.array([np.cos(a), np.sin(a)]),
         tip - 0.35 * s * d - 0.18 * s * np.array([np.cos(a), np.sin(a)])]
    ax.add_patch(Polygon(p, closed=True, fc=color, ec="none", zorder=z + 1))
