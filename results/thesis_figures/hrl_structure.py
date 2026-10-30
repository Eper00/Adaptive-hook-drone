"""Block diagram of the hierarchical agent, with the low-pass filter between
the two levels (same style as safety_filter_structure.py).

Usage: python hrl_structure.py hrl_structure.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "cm"})
FS = 21
LW = 2.5

fig, ax = plt.subplots(figsize=(13.2, 9.6))
ax.set_xlim(0, 13.2)
ax.set_ylim(0.0, 9.9)
ax.axis("off")


def box(x0, y0, x1, y1, color, title, lines, step=0.52):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=color, ec="black", lw=LW))
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    top = yc + step * len(lines) / 2
    ax.text(xc, top, title, ha="center", va="center", fontsize=FS + 1, weight="bold")
    for i, line in enumerate(lines):
        ax.text(xc, top - step * (i + 1), line, ha="center", va="center", fontsize=FS)


def arrow(points, label=None, lpos=None, ha="left", va="center"):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color="black", lw=LW, solid_capstyle="butt")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.4", color="black",
                                lw=LW, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, ha=ha, va=va, fontsize=FS)


XC = 6.6                                        # centre column
HL = (4.1, 7.85, 9.1, 9.2)
LP = (4.1, 5.55, 9.1, 6.75)
LL = (4.1, 2.95, 9.1, 4.25)
ENV = (4.1, 0.35, 9.1, 1.65)

box(*HL, "#dce8f5", "High-level policy", [r"$\pi^{\mathrm{H}}_{\zeta}$"])
box(*LP, "#f3f3f3", "Low-pass filter", [r"$\alpha = 0.1$"])
box(*LL, "#e3efdf", "Low-level policy", [r"$\pi^{\mathrm{L}}_{\nu}$ (frozen)"])
box(*ENV, "#ececec", "Environment", ["MuJoCo simulation"])

# action of the high-level policy
ax.text(XC, HL[3] + 0.45, r"$A_t^{\mathrm{H}} = [v_{\mathrm{ref},t}^T,\ \psi_{\mathrm{ref},t},\ a_{5,t},\ a_{6,t}]^T$",
        ha="center", va="center", fontsize=FS)
# main path (top to bottom)
arrow([(XC, HL[1]), (XC, LP[3])], r"$v_{\mathrm{ref},t}$", (XC + 0.2, (HL[1] + LP[3]) / 2))
arrow([(XC, LP[1]), (XC, LL[3])], r"$\bar{v}_{\mathrm{ref},t}$", (XC + 0.2, (LP[1] + LL[3]) / 2))
arrow([(XC, LL[1]), (XC, ENV[3])], r"$A^{\mathrm{L}}_t = [a_{1,t}, \ldots, a_{4,t}]^T$", (XC + 0.2, (LL[1] + ENV[3]) / 2))
# the yaw reference bypasses the filter
arrow([(HL[0] + 0.6, HL[1]), (HL[0] + 0.6, 7.2), (3.5, 7.2), (3.5, 3.9), (LL[0], 3.9)])
ax.text(3.35, 5.3, r"$\psi_{\mathrm{ref},t}$", ha="right", va="center", fontsize=FS)
# low-level observation
arrow([(ENV[2], 1.2), (10.45, 1.2), (10.45, 3.6), (LL[2], 3.6)],
      r"$\eta_t,\ v_t,\ \omega_t,$" "\n" r"$l_{1,t},\ l_{2,t}$", (10.6, 2.4))
# observation and reward of the high-level policy
arrow([(ENV[0], 0.75), (1.6, 0.75), (1.6, 8.5), (HL[0], 8.5)],
      r"$O_t^{\mathrm{H}} = O_t,$" "\n" r"$R_{t+1}$", (1.45, 6.3), ha="right")
# tendon commands
arrow([(HL[2], 8.5), (12.3, 8.5), (12.3, 0.75), (ENV[2], 0.75)])
ax.text(11.0, 6.1, "tendon\ncommands\n" r"$a_{5,t},\ a_{6,t}$", ha="center", va="center", fontsize=FS)

fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight", pad_inches=0.1)
