"""Block diagram of the hierarchical agent with the predictive safety filter
(same style and orientation as figures/hrl_structure.png).

Usage: python safety_filter_structure.py safety_filter_structure.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "cm"})
FS = 21
LW = 2.5

fig, ax = plt.subplots(figsize=(13.2, 12.2))
ax.set_xlim(0, 13.2)
ax.set_ylim(-0.7, 11.5)
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
HL = (4.1, 9.75, 9.1, 11.1)
SF = (3.6, 6.25, 9.6, 8.85)
LP = (4.6, 4.5, 8.6, 5.35)
LL = (4.1, 2.55, 9.1, 3.85)
ENV = (4.1, 0.35, 9.1, 1.65)

# closed loop identified by the ResNet
ax.add_patch(Rectangle((2.95, -0.05), 9.35, 5.8, fill=False, ec="0.35", lw=2, ls=(0, (6, 4))))
ax.text(7.62, -0.4, r"identified closed loop $F_\kappa$", ha="center", va="center",
        fontsize=FS - 2, style="italic", color="0.25")

box(*HL, "#dce8f5", "High-level policy", [r"$\pi^{\mathrm{H}}_{\zeta}$"])
box(*SF, "#fbe5cf", "Predictive safety filter",
    [r"$u_k = \pi^{\mathrm{S}}(x_k,\, u_{\mathrm{RL},k})$",
     r"prediction with $F_\kappa$, $K$ steps",
     r"$u_{\mathrm{RL},k} \in \mathcal{U}_{\mathrm{safe}}(x_k)$: $u_k = u_{\mathrm{RL},k}$",
     r"otherwise: $u_k = u^{\star}_{0|k}$"])
box(*LP, "#f3f3f3", "Low-pass filter", [])
box(*LL, "#e3efdf", "Low-level policy", [r"$\pi^{\mathrm{L}}_{\nu}$ (frozen)"])
box(*ENV, "#ececec", "Environment", ["MuJoCo simulation"])

# main path (top to bottom)
arrow([(XC, HL[1]), (XC, SF[3])], r"$u_{\mathrm{RL},k}$", (XC + 0.2, (HL[1] + SF[3]) / 2))
arrow([(XC, SF[1]), (XC, LP[3])], r"$u_k$", (XC + 0.2, 6.0))
arrow([(XC, LP[1]), (XC, LL[3])], r"$\bar{v}_{\mathrm{ref},k}$", (XC + 0.2, (LP[1] + LL[3]) / 2))
arrow([(XC, LL[1]), (XC, ENV[3])], r"$A^{\mathrm{L}}_t = [a_{1,t}, \ldots, a_{4,t}]^T$", (XC + 0.2, (LL[1] + ENV[3]) / 2))
# low-level observation (inside the closed loop)
arrow([(ENV[2], 1.2), (10.4, 1.2), (10.4, 3.2), (LL[2], 3.2)],
      r"$\eta_t,\ v_t,\ \omega_t,$" "\n" r"$l_{1,t},\ l_{2,t}$", (10.55, 2.2))
# feedback: measured state to the filter, observation and reward to the high-level policy
arrow([(ENV[0], 1.3), (2.45, 1.3), (2.45, 7.55), (SF[0], 7.55)], r"$x_k$", (2.3, 4.4), ha="right")
arrow([(ENV[0], 0.75), (1.3, 0.75), (1.3, 10.4), (HL[0], 10.4)],
      r"$O_t,$" "\n" r"$R_{t+1}$", (1.15, 8.0), ha="right")
# tendon commands bypass the filter
arrow([(HL[2], 10.4), (12.5, 10.4), (12.5, 0.75), (ENV[2], 0.75)])
ax.text(11.05, 8.2, "tendon\ncommands\n" r"$a_{5,t},\ a_{6,t}$", ha="center", va="center", fontsize=FS)

fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight", pad_inches=0.1)
