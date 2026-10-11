"""Training progress of one curriculum run, as a figure for the thesis.

Three panels over the environment steps:
  * episode return: training (rollout/ep_rew_mean, mean of the last 100
    episodes of the 8 envs) and evaluation (deterministic policy, mean and
    min-max of the evaluation episodes from evaluations.npz), with the
    curriculum advance threshold,
  * episode length: training and evaluation,
  * curriculum level (highest level of the 8 training envs).
The level changes are drawn as faint red vertical lines on every panel.

The event file whose eval/mean_reward matches the run's evaluations.npz is
used (the run of the saved models); --event selects one explicitly.
The advance threshold and the level descriptions are taken from the preset of
the environment in the run name (velocity / transport / director, see
CurriculumConfig and adjust_difficulty in utilities/learn.py); --threshold and
--levels override them. Saves <name>.png (300 dpi) and <name>.pdf to --out.

Usage
-----
    python -m utilities.plot_curriculum_training
    python -m utilities.plot_curriculum_training --run results/final/rl_adaptive_director_curriculum
    python -m utilities.plot_curriculum_training --run results/final/rl_adaptive_transport_curriculum
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUN = os.path.join(ROOT, "results", "final", "rl_adaptive_velocity_curriculum")
DEFAULT_OUT = os.path.join(ROOT, "results", "analysis", "curriculum")

# per environment: threshold_advance of its CurriculumConfig preset and the
# levels of adjust_difficulty (utilities/learn.py)
TRANSPORT_LEVELS = ["L0: waypoints only", "L1: light payload", "L2: full payload range",
                    "L3: final task"]
PRESETS = {
    "velocity": dict(threshold=110, levels=["L0: no payload", "L1: light payload, random yaw",
                                            "L2: full payload range"]),
    "transport": dict(threshold=1000, levels=TRANSPORT_LEVELS),
    # results/final/rl_adaptive_director_curriculum was trained (2026-09-28) while
    # the velocity preset was active in CurriculumConfig: threshold 110,
    # 20-episode window, 3 levels, so it never reached L3
    "director": dict(threshold=110, levels=TRANSPORT_LEVELS[:3]),
}


def preset(run):
    name = os.path.basename(os.path.normpath(run))
    for key, value in PRESETS.items():
        if key in name:
            return value
    raise ValueError(f"no curriculum preset for {name}: give --threshold and --levels")


def scalars(ea, tag):
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def pick_event(run, event=None):
    """The event file of the saved models (matching evaluations.npz), else the longest."""
    if event:
        files = [event]
    else:
        files = sorted(glob.glob(os.path.join(run, "tb", "*", "events.out.tfevents.*")))
    evals = os.path.join(run, "evaluations.npz")
    ref = np.load(evals)["results"].mean(axis=1) if os.path.exists(evals) else None
    best, best_score = None, None
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if "rollout/ep_rew_mean" not in ea.Tags()["scalars"]:
            continue
        # (matches evaluations.npz, last step)
        match = False
        if ref is not None and "eval/mean_reward" in ea.Tags()["scalars"]:
            v = scalars(ea, "eval/mean_reward")[1]
            match = len(v) == len(ref) and np.allclose(v, ref, atol=1e-3)
        score = (match, scalars(ea, "rollout/ep_rew_mean")[0][-1])
        if best_score is None or score > best_score:
            best, best_score = (f, ea), score
    print(f"using {best[0]}")
    return best[1], (np.load(evals) if ref is not None else None)


def smooth(y, w):
    if w <= 0:
        return y
    out, last = np.empty(len(y)), y[0]
    for i, v in enumerate(y):
        last = w * last + (1 - w) * v
        out[i] = last
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=DEFAULT_RUN, help="training output directory (with tb/ and evaluations.npz)")
    p.add_argument("--event", default=None, help="event file to use instead of the automatic choice")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--name", default=None, help="output file name (default: <run dir name>_training)")
    p.add_argument("--threshold", type=float, default=None,
                   help="curriculum advance threshold to draw (default: preset of the run's env)")
    p.add_argument("--levels", nargs="+", default=None,
                   help="level descriptions (default: preset of the run's env)")
    p.add_argument("--smooth", type=float, default=0.6, help="EMA weight of the evaluation mean")
    p.add_argument("--font", type=float, default=32)
    args = p.parse_args()
    if args.threshold is None or args.levels is None:
        pre = preset(args.run)
        args.threshold = pre["threshold"] if args.threshold is None else args.threshold
        args.levels = args.levels or pre["levels"]
    levels_txt = args.levels

    ea, evals = pick_event(args.run, args.event)
    s_rew, rew = scalars(ea, "rollout/ep_rew_mean")
    s_len, ep_len = scalars(ea, "rollout/ep_len_mean")
    s_lv, lv = scalars(ea, "curriculum/level")
    changes = [(s_lv[i], int(lv[i])) for i in range(1, len(lv)) if lv[i] != lv[i - 1]]
    end = max(s_rew[-1], s_lv[-1])
    for s, l in changes:
        print(f"level {l} from step {s}")

    plt.rcParams.update({
        "font.size": args.font, "axes.titlesize": args.font + 2, "axes.labelsize": args.font,
        "xtick.labelsize": args.font - 2, "ytick.labelsize": args.font - 2,
        "legend.fontsize": args.font - 4, "lines.linewidth": 2.5,
    })
    # the level panel grows with the number of levels (room for the descriptions)
    n_levels = len(levels_txt)
    fig, axes = plt.subplots(3, 1, figsize=(16, 15 + 0.9 * n_levels), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 2, 0.55 * n_levels]))
    ax_r, ax_l, ax_c = axes
    scale = 1e6

    # episode return
    ax_r.plot(s_rew / scale, rew, color="tab:blue", label="training (mean of last 100 episodes)")
    if evals is not None:
        t, r = evals["timesteps"] / scale, evals["results"]
        ax_r.fill_between(t, r.min(1), r.max(1), color="tab:orange", alpha=0.15, lw=0, label="evaluation min–max")
        ax_r.plot(t, smooth(r.mean(1), args.smooth), color="tab:orange", label="evaluation (deterministic, mean)")
    ax_r.axhline(args.threshold, color="gray", ls="--", lw=1.8, label=f"advance threshold ({args.threshold:g})")
    ax_r.set_ylabel("Episode return")

    # episode length
    ax_l.plot(s_len / scale, ep_len, color="tab:blue", label="training (mean of last 100 episodes)")
    if evals is not None:
        ax_l.plot(evals["timesteps"] / scale, smooth(evals["ep_lengths"].mean(1), args.smooth),
                  color="tab:orange", label="evaluation (deterministic, mean)")
    ax_l.set_ylabel("Episode length\n[steps]")

    # curriculum level
    ax_c.step(s_lv / scale, lv, where="post", color="tab:red")
    n_lv = len(levels_txt)
    ax_c.set_yticks(range(n_lv))
    ax_c.set_yticklabels([f"L{i}" for i in range(n_lv)])
    ax_c.set_ylim(-0.4, n_lv - 0.2)
    ax_c.set_ylabel("Curriculum\nlevel")
    ax_c.set_xlabel("Environment steps [×10⁶]")

    # level boundaries and descriptions
    bounds = [0] + [s / scale for s, _ in changes] + [end / scale]
    for ax in axes:
        for i, (s, _) in enumerate(changes):
            ax.axvline(s / scale, color="red", alpha=0.3, lw=2,
                       label="level change" if (ax is ax_r and i == 0) else None)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, end / scale)
    # description of each level just after its start, above its line
    span = end / scale
    for i in range(min(n_lv, len(bounds) - 1)):
        ax_c.text(bounds[i] + 0.01 * span, i + 0.12, levels_txt[i], ha="left", va="bottom",
                  fontsize=args.font - 5, color="tab:red",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    # level names above the top panel; close changes are pushed apart
    xs = [s / scale for s, _ in changes]
    for i, (x, (_, l)) in enumerate(zip(xs, changes)):
        near_next = i + 1 < len(xs) and xs[i + 1] - x < 0.05 * span
        near_prev = i > 0 and x - xs[i - 1] < 0.05 * span
        ha = "right" if near_next else "left" if near_prev else "center"
        ax_r.text(x, 1.01, f" L{l} " if ha != "center" else f"L{l}", transform=ax_r.get_xaxis_transform(),
                  ha=ha, va="bottom", color="red", fontsize=args.font - 3)

    ax_r.set_title("Training progress with curriculum", pad=1.8 * args.font)
    fig.align_ylabels(axes)
    fig.tight_layout()
    # one legend for the whole figure, below it, so that it covers no data
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"{os.path.basename(os.path.normpath(args.run))}_training"
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        path = os.path.join(args.out, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", **kw)
        print("saved", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
