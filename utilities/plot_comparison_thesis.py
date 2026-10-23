"""overview.png and rescue_matrix.png of compare_methods, sized for the thesis.

The plots of compare_methods are made for the screen (17 in wide, 8-9 pt
text), which become unreadable at text width. This script runs the same
plot functions on the cached episodes with a smaller figure, larger fonts,
short method names and no suptitle (the caption says it), by patching a
copy of their source. If compare_methods changes so that a patch no longer
applies, the script stops with the patch that failed.

Usage:
    python -m utilities.plot_comparison_thesis
    python -m utilities.plot_comparison_thesis --out ../documentation/thesis/figures   # default: results/thesis_figures
"""

import argparse
import inspect
import os
import pickle
import types

from utilities import compare_methods

PATCHES = [
    ('"font.size": 9', '"font.size": 12'),
    ('"axes.titlesize": 10', '"axes.titlesize": 12'),
    ("figsize=(17, 9.5)", "figsize=(12, 7.8)"),
    ("fontsize=8", "fontsize=11"),
    ("fig.suptitle(", "(lambda *a, **k: None)("),
    ('names = [NAMES[k].replace(" ", "\\n", 1) for k in kinds]',
     'names = [{"director": "RL", "director_mpc": "MPC", "director_safety": "RL+SF"}[k]'
     ' for k in kinds]'),
    ('ax.set_title(f"Outcomes ({len(results[kinds[0]])} scenarios; success: goal + "\n'
     '                 f"{GOAL_HOLD_TIME:g} s hold; bar: 95% CI)")', 'ax.set_title("Outcomes")'),
    ('ax.legend(ncol=4, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.13))',
     'ax.legend(ncol=2, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.1))'),
    ('"payload mass [kg] (quintile bins)"', '"payload mass [kg]"'),
    ('"Time to the goal with the payload (successes)"', '"Time to goal (successes)"'),
    ('"Max tilt per episode (dotted: filter limit)"', '"Max tilt per episode"'),
    ('"Command roughness  mean |d cmd/dt|"', '"Command roughness"'),
    ('"Computation per control step (episode median)"', '"Computation per step"'),
    ("figsize=(5.2 * len(others), 4.6)", "figsize=(4.9 * len(others), 4.3)"),
    ('fontsize=12,\n                            fontweight="bold"',
     'fontsize=15,\n                            fontweight="bold"'),
    ("ax.set_xticks(range(len(OUTCOMES)), OUTCOMES)",
     'ax.set_xticks(range(len(OUTCOMES)), OUTCOMES, rotation=25, ha="right")'),
    ("fig.savefig(f, dpi=140)", 'fig.savefig(f, dpi=200, bbox_inches="tight")'),
]


def thesis_module():
    src = inspect.getsource(compare_methods)
    for old, new in PATCHES:
        if old not in src:
            raise RuntimeError(f"patch does not apply to compare_methods:\n{old}")
        src = src.replace(old, new)
    mod = types.ModuleType("compare_methods_thesis")
    exec(compile(src, compare_methods.__file__, "exec"), mod.__dict__)
    return mod

SHORT = {"director": "RL", "director_mpc": "MPC", "director_safety": "RL+SF"}


def plot_overview_thesis(cm, results, out):
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
    box(axs[1, 1], [[np.mean(np.linalg.norm(np.diff(e["cmd"], axis=0), axis=1)) * 48 for e in results[k]]
                    for k in kinds], "Command roughness", "[m/s$^2$]")
    box(axs[1, 2], [[1e3 * np.median(e["step_time"]) for e in results[k]] for k in kinds],
        "Computation per step", "[ms]", 1e3 / 48, log=True)
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    f = os.path.join(out, "comparison_overview.png")
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f



def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", default="results/analysis/methods/episodes.pkl")
    p.add_argument("--out", default="results/thesis_figures")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    cm = thesis_module()
    with open(a.episodes, "rb") as f:
        results = pickle.load(f)
    cm.reclassify(results)          # current success rule (goal + hold)
    cm._style()
    os.makedirs(a.out, exist_ok=True)
    print("saved:", plot_overview_thesis(cm, results, a.out))
    for f in cm.plot_rescue_matrix(results, a.out):
        dst = os.path.join(a.out, "comparison_" + os.path.basename(f))   # name in the thesis
        os.replace(f, dst)
        print("saved:", dst)


if __name__ == "__main__":
    main()
