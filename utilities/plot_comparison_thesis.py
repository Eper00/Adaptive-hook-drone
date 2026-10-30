"""Figure 6.1 of the thesis: the overview of compare_methods, sized for the thesis.

The overview of compare_methods is made for the screen (17 in wide, 8-9 pt
text, stacked 0-100 % bars that hide the few failures). This script draws
the thesis version from the cached episodes: failures counted per type,
success rate per payload mass, and box plots of the time to goal, max tilt,
filtered command roughness and computation time, with larger fonts and short
method names.

Usage:
    python -m utilities.plot_comparison_thesis              # -> results/thesis_figures
    python -m utilities.plot_comparison_thesis --out ../documentation/thesis/figures
"""

import argparse
import os
import pickle

from utilities import compare_methods as cm

SHORT = {"director": "RL", "director_mpc": "MPC", "director_safety": "RL+SF"}


def plot_overview_thesis(results, out):
    """Overview for the thesis: failures counted per type (the stacked 0-100 %
    bars of compare_methods hide the few failures), success rate per payload
    mass, and box plots of the time to goal, max tilt, command roughness and
    computation time."""
    import matplotlib.pyplot as plt
    import numpy as np
    from utilities.analyse import OUTCOME_COLORS

    kinds = list(results)
    n = len(results[kinds[0]])
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "legend.fontsize": 13})
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.6))

    ax = axs[0, 0]                                         # failures by type
    fails = ["stability", "payload", "unfinished"]
    w = 0.26
    for i, o in enumerate(fails):
        cnt = [sum(e["outcome"] == o for e in results[k]) for k in kinds]
        x = np.arange(len(kinds)) + (i - 1) * w
        ax.bar(x, cnt, w, color=OUTCOME_COLORS[o], label=o)
        for xi, c in zip(x, cnt):
            ax.text(xi, c + 0.4, str(c), ha="center", va="bottom", fontsize=13)
    succ = [sum(e["outcome"] == "success" for e in results[k]) for k in kinds]
    ax.set_xticks(range(len(kinds)), [f"{SHORT[k]}\n{100 * s_ / n:.1f} %" for k, s_ in zip(kinds, succ)])
    ax.set_ylabel(f"failed episodes (of {n})")
    ax.set_title("Failures (below: success rate)")
    ax.set_ylim(0, 1.25 * max(1, max(sum(e["outcome"] == o for e in results[k]) for k in kinds for o in fails)))
    ax.legend(loc="upper right")

    ax = axs[0, 1]                                         # success vs payload mass
    masses = np.array([e["mass"] for e in results[kinds[0]]])
    edges = np.quantile(masses, np.linspace(0, 1, 6))
    edges[-1] += 1e-9
    centers = 0.5 * (edges[:-1] + edges[1:])
    for k in kinds:
        rate = [100 * np.mean([e["outcome"] == "success" for e in results[k] if lo <= e["mass"] < hi])
                for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centers, rate, "o-", color=cm.COLORS[k], label=SHORT[k], lw=2.5, ms=8)
    ax.set_xlabel("payload mass [kg]")
    ax.set_ylabel("success [%]")
    ax.set_title("Success vs payload mass")
    ax.legend()

    def box(ax, data, title, ylabel, hline=None, log=False):
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], patch_artist=True, widths=0.55)
        for patch, k in zip(bp["boxes"], kinds):
            patch.set_facecolor(cm.COLORS[k])
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_xticks(range(1, len(kinds) + 1), [SHORT[k] for k in kinds])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, color="#d1453b", ls=":", lw=1.8)
        if log:
            from matplotlib.ticker import ScalarFormatter
            ax.set_yscale("log")
            ax.set_yticks([1, 2, 5, 10, 20, 50])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())

    box(axs[0, 2], [[e["event_time"] for e in results[k] if e["outcome"] == "success"] for k in kinds],
        "Time to goal (successes)", "[s]")
    box(axs[1, 0], [[cm.tilt_deg(e).max() for e in results[k]] for k in kinds], "Max tilt per episode",
        "[deg]", 15.0)
    # roughness of the low-pass filtered command (3.20), which every method passes
    box(axs[1, 1], [[np.mean(np.linalg.norm(np.diff(e["cmd_filtered"], axis=0), axis=1)) * 48
                     for e in results[k]] for k in kinds], "Filtered command roughness", "[m/s$^2$]")
    box(axs[1, 2], [[1e3 * np.median(e["step_time"]) for e in results[k]] for k in kinds],
        "Computation per step", "[ms]", 1e3 / 48, log=True)
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    f = os.path.join(out, "comparison_overview.png")
    fig.savefig(f, dpi=200, bbox_inches="tight")
    fig.savefig(f[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return f



def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", default="results/analysis/methods/episodes.pkl")
    p.add_argument("--out", default="results/thesis_figures")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    with open(a.episodes, "rb") as f:
        results = pickle.load(f)
    cm.reclassify(results)          # current success rule (goal + hold)
    cm._style()
    os.makedirs(a.out, exist_ok=True)
    print("saved:", plot_overview_thesis(results, a.out))


if __name__ == "__main__":
    main()
