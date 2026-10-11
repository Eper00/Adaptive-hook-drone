"""Training with vs. without curriculum, from the TensorBoard logs.

Reads the scalars of the two PPO runs and plots them over the environment
steps. The curriculum level changes (``curriculum/level``) are drawn as faint
red vertical lines, and the part after the final level is reached (from there
both runs train on the same task) is shaded.

  ep_rew_mean.png   rollout/ep_rew_mean of both runs
  metrics.png       ep_rew_mean, eval/mean_reward, ep_len_mean, train/std,
                    train/explained_variance, train/value_loss
  summary.csv       steps to reach the reward thresholds (on the final task)

Note: before the final level the curriculum run is rewarded on an easier task,
so its reward is not directly comparable with the baseline there. The fair
comparison is the part after the last red line (and the steps needed to reach
a given reward on the final task, summary.csv).

If a log directory has several event files (several runs written into the same
PPO_1), the one reaching the most steps is used (the last one on a tie);
--baseline / --curriculum also accept an event file.

Usage
-----
    python -m utilities.compare_curriculum
    python -m utilities.compare_curriculum --smooth 0.9 --thresholds 500 1000 1500
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DEFAULT_BASELINE = os.path.join(ROOT, "final", "rl_adaptive_transport", "tb", "PPO_1")
DEFAULT_CURRICULUM = os.path.join(ROOT, "final", "rl_adaptive_transport_curriculum", "tb", "PPO_1")
DEFAULT_OUT = os.path.join(ROOT, "analysis", "curriculum")

METRICS = [
    ("rollout/ep_rew_mean", "training episode reward (mean)"),
    ("eval/mean_reward", "evaluation reward (mean)"),
    ("rollout/ep_len_mean", "training episode length (mean)"),
    ("train/std", "policy std"),
    ("train/explained_variance", "explained variance"),
    ("train/value_loss", "value loss"),
]
STYLE = {"baseline": dict(color="tab:blue", label="without curriculum"),
         "curriculum": dict(color="tab:green", label="with curriculum")}


def load(path):
    """Scalars {tag: (steps, values)} of the event file (or of the longest run in a directory)."""
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"no event file in {path}")
    best, best_step = None, -1
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if "rollout/ep_rew_mean" not in tags:
            continue
        last = ea.Scalars("rollout/ep_rew_mean")[-1].step
        if last >= best_step:
            best, best_step = (f, ea), last
    f, ea = best
    print(f"using {f} ({best_step} steps)")
    return {t: (np.array([e.step for e in ea.Scalars(t)]), np.array([e.value for e in ea.Scalars(t)]))
            for t in ea.Tags()["scalars"]}


def smooth(y, w):
    """TensorBoard-style exponential moving average (w = 0: none)."""
    if w <= 0 or len(y) == 0:
        return y
    out, last = np.empty_like(y, dtype=float), y[0]
    for i, v in enumerate(y):
        last = w * last + (1 - w) * v
        out[i] = last
    return out


def level_changes(data):
    """[(step, new level)] from curriculum/level."""
    if "curriculum/level" not in data:
        return []
    steps, lv = data["curriculum/level"]
    return [(int(steps[i]), int(lv[i])) for i in range(1, len(lv)) if lv[i] != lv[i - 1]]


def mark_levels(ax, changes, final_step, end_step, label=True):
    for i, (s, lv) in enumerate(changes):
        ax.axvline(s, color="red", alpha=0.3, lw=1.2, ls="-",
                   label="curriculum level change" if (label and i == 0) else None)
        if label:
            ax.text(s, 1.0, f" L{lv}", color="red", alpha=0.7, fontsize=8,
                    transform=ax.get_xaxis_transform(), va="top")
    if final_step is not None:
        ax.axvspan(final_step, end_step, color="red", alpha=0.04)
    ax.set_xlim(0, end_step)


def first_reach(steps, values, threshold, start=0):
    """First step >= start where values >= threshold (None if never)."""
    idx = np.where((steps >= start) & (values >= threshold))[0]
    return int(steps[idx[0]]) if len(idx) else None


def plot_curve(ax, runs, tag, w):
    for name, data in runs.items():
        if tag not in data:
            continue
        s, v = data[tag]
        ax.plot(s, v, color=STYLE[name]["color"], alpha=0.2, lw=0.8)
        ax.plot(s, smooth(v, w), lw=1.8, **STYLE[name])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.add_argument("--curriculum", default=DEFAULT_CURRICULUM)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--smooth", type=float, default=0.8, help="EMA weight as in TensorBoard (0 = none)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[500, 1000, 1500],
                   help="reward thresholds for the steps-to-reach table")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {"baseline": load(args.baseline), "curriculum": load(args.curriculum)}
    changes = level_changes(runs["curriculum"])
    final_step = changes[-1][0] if changes else None
    end_step = max(d["rollout/ep_rew_mean"][0][-1] for d in runs.values())
    for s, lv in changes:
        print(f"level {lv} from step {s}")

    # 1) rollout/ep_rew_mean
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, runs, "rollout/ep_rew_mean", args.smooth)
    mark_levels(ax, changes, final_step, end_step)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("rollout/ep_rew_mean")
    ax.set_title("Training reward with and without curriculum\n"
                 "(shaded: curriculum on the final task, same task as the baseline)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(args.out, "ep_rew_mean.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)

    # 2) metric panel
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for ax, (tag, title) in zip(axes.flat, METRICS):
        plot_curve(ax, runs, tag, args.smooth)
        mark_levels(ax, changes, final_step, end_step, label=(tag == METRICS[0][0]))
        ax.set_title(f"{title}  ({tag})", fontsize=10)
        ax.grid(alpha=0.3)
        if tag == "train/value_loss":
            ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("environment steps")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("PPO training with vs. without curriculum (red lines: curriculum level changes)")
    fig.tight_layout()
    path = os.path.join(args.out, "metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)

    # 3) steps to reach the thresholds on the final task (smoothed curves)
    rows = []
    for tag in ("rollout/ep_rew_mean", "eval/mean_reward"):
        b_s, b_v = runs["baseline"][tag]
        c_s, c_v = runs["curriculum"][tag]
        b_v, c_v = smooth(b_v, args.smooth), smooth(c_v, args.smooth)
        start = final_step or 0
        for th in args.thresholds:
            b = first_reach(b_s, b_v, th)
            c = first_reach(c_s, c_v, th, start=start)
            rows.append(dict(metric=tag, threshold=th, baseline_steps=b, curriculum_total_steps=c,
                             curriculum_steps_on_final_task=None if c is None else c - start))
    path = os.path.join(args.out, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)
    print(f"\nsteps to reach the reward on the final task (smoothing {args.smooth}):")
    print(f"{'metric':22s} {'thr':>6s} {'baseline':>10s} {'curr. total':>12s} {'curr. on final':>15s}")
    fmt = lambda v: "-" if v is None else f"{v / 1e6:.2f}M"
    for r in rows:
        print(f"{r['metric']:22s} {r['threshold']:6.0f} {fmt(r['baseline_steps']):>10s} "
              f"{fmt(r['curriculum_total_steps']):>12s} {fmt(r['curriculum_steps_on_final_task']):>15s}")
    for name, data in runs.items():
        s, v = data["rollout/ep_rew_mean"]
        n = max(1, len(v) // 10)
        print(f"{STYLE[name]['label']}: final ep_rew_mean (mean of the last 10 %) = {v[-n:].mean():.0f}")


if __name__ == "__main__":
    main()
