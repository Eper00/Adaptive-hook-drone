"""Training of two curriculum runs side by side, as a figure for the thesis.

Two panels over the environment steps (cut at --max_steps): episode return
and episode length, training (rollout/*, mean of the last 100 episodes; solid)
and evaluation (deterministic policy, smoothed mean; dashed) of every run. The
curriculum level changes of each run are drawn as vertical lines in the run's
colour, labelled at the top. The event file of each run is chosen as in
plot_curriculum_training (the one matching the run's evaluations.npz).

Default: director vs transport curriculum (the director returns are the
transport reward minus a small action penalty, both have 480-step episodes).
Saves <name>.png (300 dpi) and <name>.pdf to --out.

Usage
-----
    python -m utilities.plot_curriculum_compare
    python -m utilities.plot_curriculum_compare --max_steps 10e6
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utilities.plot_curriculum_training import DEFAULT_OUT, ROOT, pick_event, scalars, smooth

RUNS = [
    ("director (velocity commands)", os.path.join(ROOT, "results", "final", "rl_adaptive_director_curriculum"),
     "tab:blue"),
    ("transport (motor commands)", os.path.join(ROOT, "results", "final", "rl_adaptive_transport_curriculum"),
     "tab:green"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max_steps", type=float, default=5e6)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--name", default="director_vs_transport_curriculum")
    p.add_argument("--smooth", type=float, default=0.6, help="EMA weight of the evaluation mean")
    p.add_argument("--font", type=float, default=32)
    args = p.parse_args()
    f = args.font
    plt.rcParams.update({
        "font.size": f, "axes.titlesize": f + 2, "axes.labelsize": f,
        "xtick.labelsize": f - 2, "ytick.labelsize": f - 2,
        "legend.fontsize": f - 4, "lines.linewidth": 2.5,
    })
    scale, end = 1e6, args.max_steps

    fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(16, 15), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 2]))
    handles = []
    for k, (label, run, color) in enumerate(RUNS):
        ea, evals = pick_event(run)
        for ax, tag, key in ((ax_r, "rollout/ep_rew_mean", "results"),
                             (ax_l, "rollout/ep_len_mean", "ep_lengths")):
            s, v = scalars(ea, tag)
            keep = s <= end
            line, = ax.plot(s[keep] / scale, v[keep], color=color, label=f"{label}, training")
            if evals is not None:
                t = evals["timesteps"]
                keep = t <= end
                ev, = ax.plot(t[keep] / scale, smooth(evals[key].mean(axis=1), args.smooth)[keep],
                              color=color, ls="--", lw=2, alpha=0.8, label=f"{label}, evaluation")
            if ax is ax_r:
                handles += [line] + ([ev] if evals is not None else [])
        # level changes of this run
        s_lv, lv = scalars(ea, "curriculum/level")
        changes = [(s_lv[i], int(lv[i])) for i in range(1, len(lv)) if lv[i] != lv[i - 1] and s_lv[i] <= end]
        for ax in (ax_r, ax_l):
            for s, _ in changes:
                ax.axvline(s / scale, color=color, alpha=0.45, lw=2.5, ls=":")
        # labels at the top, the two runs on two rows
        for i, (s, l) in enumerate(changes):
            close = i > 0 and (s - changes[i - 1][0]) < 0.05 * end
            ax_r.text(s / scale, 1.01 + 0.07 * k, f" L{l}" if close else f"L{l} ",
                      transform=ax_r.get_xaxis_transform(), ha="left" if close else "right",
                      va="bottom", color=color, fontsize=f - 6)
        print(label, "level changes:", [(int(s), l) for s, l in changes])

    ax_r.set_ylabel("Episode return")
    ax_l.set_ylabel("Episode length\n[steps]")
    ax_l.set_xlabel("Environment steps [×10⁶]")
    for ax in (ax_r, ax_l):
        ax.grid(alpha=0.3)
        ax.set_xlim(0, end / scale)
    ax_r.set_title("Curriculum training: director vs transport", pad=3.2 * f)
    fig.align_ylabels((ax_r, ax_l))
    fig.tight_layout()
    level_handle = plt.Line2D([], [], color="gray", ls=":", lw=2.5, label="level change (run colour)")
    fig.legend(handles=handles + [level_handle], loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=1, frameon=False)

    os.makedirs(args.out, exist_ok=True)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        path = os.path.join(args.out, f"{args.name}.{ext}")
        fig.savefig(path, bbox_inches="tight", **kw)
        print("saved", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
