"""Training curves of several seeds per method, as figures for the thesis.

Every group is a set of training directories (one per seed, written by
utilities/learn.py, see utilities/train_seeds.sh). The training curves
(rollout/ep_rew_mean and rollout/ep_len_mean, mean of the last 100 episodes
of the 8 envs) of the seeds are interpolated to a common step grid and drawn
as the mean over the seeds (line) with the min-max range of the seeds
(shaded). For the curriculum groups the bottom panel shows the curriculum
level of every seed, and the level changes are marked by thin vertical lines
in the group's colour on the other panels.

Presets (--preset):
  cl        flat agent without vs with curriculum      (thesis Fig. 3.1, with --overlay)
  director  director vs flat agent, both with curriculum (Section 3.3.8)

Besides <name>.png (300 dpi) and <name>.pdf, <name>_summary.csv lists per
run: the step of every level change, the step from which the run trains on
the final task (the first step on level --final_level; 0 without
curriculum; empty if the run never got there), the steps at which the
return on the final task first reaches each --thresholds value, and the
mean training return and episode length over the last 10 % of the run.

The steps to a threshold are counted only on the final task, in two ways:
  steps_to_<th>_train  training return (rollout/ep_rew_mean). It is the mean
                       of the last 100 episodes, which right after the level
                       change still contains episodes of the easier level, so
                       the count starts 100 episodes (100 x episode length
                       steps) after the change. The 8 envs change level
                       independently (the log shows the most advanced one);
                       they change within a few episodes of each other, as
                       their 100-episode windows fill at the same rate.
  steps_to_<th>_eval   evaluation return (deterministic policy, mean of the
                       evaluation episodes in evaluations.npz). The
                       evaluation env is switched together with the logged
                       level, so every evaluation after the change is on the
                       final task.

Usage
-----
    python -m utilities.plot_seeds --preset cl
    python -m utilities.plot_seeds --preset director
    python -m utilities.plot_seeds --preset cl --overlay --out results/thesis_figures   # thesis Fig. 3.1
    python -m utilities.plot_seeds --group "flat=results/seeds/transport_s*_curriculum" \\
        --group "director=results/seeds/director_s*_curriculum" --name my_plot
"""

import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from utilities.plot_curriculum_training import (ROOT, TRANSPORT_LEVELS, pick_event, scalars,
                                                smooth)

SEEDS = os.path.join(ROOT, "results", "seeds")
DEFAULT_OUT = os.path.join(ROOT, "results", "analysis", "seeds")
COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]


def presets(root):
    """Run groups of the presets, for the runs of utilities/train_seeds.sh in ``root``."""
    return {
        "cl": dict(name="transport_cl_vs_no_cl_seeds",
                   title="Flat agent: training with and without curriculum",
                   groups=[("without CL", os.path.join(root, "transport_s[0-9]")),
                           ("with CL", os.path.join(root, "transport_s[0-9]_curriculum"))]),
        "director": dict(name="director_vs_transport_seeds",
                         title="Curriculum training: hierarchical vs flat agent",
                         groups=[("flat agent", os.path.join(root, "transport_s[0-9]_curriculum")),
                                 ("hierarchical agent",
                                  os.path.join(root, "director_s[0-9]_curriculum"))]),
    }


def load_run(run):
    ea, evals = pick_event(run)
    tags = ea.Tags()["scalars"]
    data = {t: scalars(ea, t) for t in ("rollout/ep_rew_mean", "rollout/ep_len_mean") if t in tags}
    data["level"] = scalars(ea, "curriculum/level") if "curriculum/level" in tags else None
    data["eval"] = None if evals is None else (evals["timesteps"], evals["results"].mean(axis=1))
    return data


def final_task_start(level, final_level):
    """First step on ``final_level`` (0 without curriculum, None if never reached)."""
    if level is None:
        return 0
    s, lv = level
    idx = np.where(lv >= final_level)[0]
    return int(s[idx[0]]) if len(idx) else None


def level_changes(level):
    if level is None:
        return []
    s, lv = level
    return [(int(s[i]), int(lv[i])) for i in range(1, len(lv)) if lv[i] != lv[i - 1]]


def on_grid(runs, tag, grid):
    """(n_runs, len(grid)) array of the runs' curves, NaN outside each run."""
    out = np.full((len(runs), len(grid)), np.nan)
    for i, d in enumerate(runs):
        s, v = d[tag]
        keep = (grid >= s[0]) & (grid <= s[-1])
        out[i, keep] = np.interp(grid[keep], s, v)
    return out


def first_reach(s, v, threshold, start):
    """First step >= start where v >= threshold (None if never, or if start is None)."""
    if start is None:
        return None
    idx = np.where((s >= start) & (v >= threshold))[0]
    return int(s[idx[0]]) if len(idx) else None


def summary_rows(loaded, args):
    """One row per run (see the module docstring)."""
    rows = []
    for label, dirs, runs in loaded:
        for d, run in zip(runs, dirs):
            changes = level_changes(d["level"])
            s, v = d["rollout/ep_rew_mean"]
            s_len, v_len = d["rollout/ep_len_mean"]
            start = final_task_start(d["level"], args.final_level)
            # training return: from 100 episodes after the change (see module docstring)
            train_start = None if start is None else (
                start + (100 * float(np.interp(start, s_len, v_len)) if start > 0 else 0))
            final_level = int(d["level"][1][-1]) if d["level"] is not None else None
            n = max(1, len(v) // 10)
            row = dict(group=label, run=os.path.relpath(run, ROOT), steps=int(s[-1]), final_level=final_level,
                       level_changes=" ".join(f"L{l}@{st}" for st, l in changes),
                       final_task_from=start,
                       final_return_last10pct=round(float(v[-n:].mean()), 1),
                       final_ep_len_last10pct=round(float(v_len[-max(1, len(v_len) // 10):].mean()), 1))
            for th in args.thresholds:
                row[f"steps_to_{th:g}_train"] = first_reach(s, v, th, train_start)
                row[f"steps_to_{th:g}_eval"] = (None if d["eval"] is None
                                                else first_reach(*d["eval"], th, start))
            rows.append(row)
    return rows


def draw_group(label, dirs, runs, args, grid, end, title, name):
    """One group of seeds in the style and colours of plot_curriculum_training:
    training blue, evaluation orange (mean of the seeds, min-max shaded),
    curriculum level and level changes red."""
    f, scale = args.font, 1e6
    plt.rcParams.update({
        "font.size": f, "axes.titlesize": f + 2, "axes.labelsize": f,
        "xtick.labelsize": f - 2, "ytick.labelsize": f - 2,
        "legend.fontsize": f - 4, "lines.linewidth": 2.5,
    })
    has_cl = any(d["level"] is not None for d in runs)
    levels_txt = TRANSPORT_LEVELS
    n_lv = len(levels_txt)
    n_panels = 3 if has_cl else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 15 + (0.9 * n_lv if has_cl else 0)),
                             sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 0.55 * n_lv][:n_panels]))
    ax_r, ax_l = axes[0], axes[1]
    ns = len(runs)
    for ax, tag, key in ((ax_r, "rollout/ep_rew_mean", "results"), (ax_l, "rollout/ep_len_mean", "ep_lengths")):
        y = on_grid(runs, tag, grid)
        ok = ~np.all(np.isnan(y), axis=0)
        ax.fill_between(grid[ok] / scale, np.nanmin(y[:, ok], 0), np.nanmax(y[:, ok], 0),
                        color="tab:blue", alpha=0.15, lw=0, label=f"training min–max of the {ns} seeds")
        ax.plot(grid[ok] / scale, np.nanmean(y[:, ok], 0), color="tab:blue",
                label=f"training (mean of last 100 episodes, mean of {ns} seeds)")
        Y = []
        for run in dirs:
            path = os.path.join(run, "evaluations.npz")
            if not os.path.exists(path):
                continue
            e = np.load(path)
            t = e["timesteps"]
            keep = (grid >= t[0]) & (grid <= t[-1])
            yy = np.full(len(grid), np.nan)
            yy[keep] = np.interp(grid[keep], t, smooth(e[key].mean(1), args.smooth))
            Y.append(yy)
        if Y:
            Y = np.asarray(Y)
            ok = ~np.all(np.isnan(Y), axis=0)
            ax.fill_between(grid[ok] / scale, np.nanmin(Y[:, ok], 0), np.nanmax(Y[:, ok], 0),
                            color="tab:orange", alpha=0.15, lw=0,
                            label=f"evaluation min–max of the {ns} seeds")
            ax.plot(grid[ok] / scale, np.nanmean(Y[:, ok], 0), color="tab:orange",
                    label=f"evaluation (deterministic, mean of {ns} seeds)")
    first = True
    for j, d in enumerate(runs):
        for s_, _ in level_changes(d["level"]):
            if s_ <= end:
                for ax in axes:
                    ax.axvline(s_ / scale, color="red", alpha=0.3, lw=2,
                               label="level change (every seed)" if (ax is ax_r and first) else None)
                first = False
        if has_cl and d["level"] is not None:
            s_lv, lv = d["level"]
            keep = s_lv <= end
            axes[2].step(s_lv[keep] / scale, lv[keep] + 0.03 * (j - (ns - 1) / 2), where="post",
                         color="tab:red", lw=2.5)
    if has_cl:
        ax_r.axhline(args.threshold, color="gray", ls="--", lw=1.8,
                     label=f"advance threshold ({args.threshold:g})")
        ax_c = axes[2]
        ax_c.set_yticks(range(n_lv))
        ax_c.set_yticklabels([f"L{i}" for i in range(n_lv)])
        ax_c.set_ylim(-0.4, n_lv - 0.2)
        ax_c.set_ylabel("Curriculum\nlevel")
        for i, txt in enumerate(levels_txt):
            ax_c.text(0.01 * end / scale, i + 0.12, txt, ha="left", va="bottom", fontsize=f - 5,
                      color="tab:red", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                                                 alpha=0.8))
    ax_r.set_ylabel("Episode return")
    ax_l.set_ylabel("Episode length\n[steps]")
    axes[-1].set_xlabel("Environment steps [×10⁶]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, end / scale)
    ax_r.set_title(title, pad=12)
    fig.align_ylabels(axes)
    fig.tight_layout()
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        path = os.path.join(args.out, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", **kw)
        print("saved", path)
    plt.close(fig)


def draw_overlay(loaded, args, grid, end, name):
    """All groups in one figure (thesis Fig. 3.1): training return, episode
    length and curriculum level; per group the mean of the seeds (line) and
    their min-max range (shaded), level changes as dotted lines in the group's
    colour. ``--figsize`` sets the aspect ratio (wide for the thesis)."""
    f, scale = args.font, 1e6
    plt.rcParams.update({
        "font.size": f, "axes.labelsize": f, "xtick.labelsize": f - 2, "ytick.labelsize": f - 2,
        "legend.fontsize": f - 4, "lines.linewidth": 2.5,
    })
    has_cl = any(d["level"] is not None for _, _, runs in loaded for d in runs)
    n_lv = len(TRANSPORT_LEVELS)
    n_panels = 3 if has_cl else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=args.figsize, sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 2, 0.5 * n_lv][:n_panels]))
    ax_r, ax_l = axes[0], axes[1]
    for g, (label, dirs, runs) in enumerate(loaded):
        color = COLORS[g % len(COLORS)]
        for ax, tag in ((ax_r, "rollout/ep_rew_mean"), (ax_l, "rollout/ep_len_mean")):
            y = on_grid(runs, tag, grid)
            ok = ~np.all(np.isnan(y), axis=0)
            ax.fill_between(grid[ok] / scale, np.nanmin(y[:, ok], 0), np.nanmax(y[:, ok], 0),
                            color=color, alpha=0.2, lw=0)
            ax.plot(grid[ok] / scale, np.nanmean(y[:, ok], 0), color=color,
                    label=f"{label} (mean of {len(runs)} seeds, min–max shaded)" if ax is ax_r else None)
        for j, d in enumerate(runs):
            for s_, _ in level_changes(d["level"]):
                if s_ <= end:
                    for ax in axes:
                        ax.axvline(s_ / scale, color=color, ls=":", lw=1.8, alpha=0.7)
            if has_cl and d["level"] is not None:
                s_lv, lv = d["level"]
                keep = s_lv <= end
                axes[2].step(s_lv[keep] / scale, lv[keep] + 0.04 * (j - (len(runs) - 1) / 2),
                             where="post", color=color, lw=2.5)
    if has_cl:
        ax_r.plot([], [], color="gray", ls=":", lw=1.8, label="curriculum level change (group colour)")
        ax_c = axes[2]
        ax_c.set_yticks(range(n_lv))
        ax_c.set_yticklabels([f"L{i}" for i in range(n_lv)])
        ax_c.set_ylim(-0.4, n_lv - 0.6)
        ax_c.set_ylabel("Curriculum\nlevel")
    ax_r.set_ylabel("Episode return")
    ax_l.set_ylabel("Episode length\n[steps]")
    axes[-1].set_xlabel("Environment steps [×10⁶]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, end / scale)
    fig.align_ylabels(axes)
    fig.tight_layout()
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        path = os.path.join(args.out, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", **kw)
        print("saved", path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=["cl", "director"], default="cl")
    p.add_argument("--root", default=SEEDS, help="directory of the runs (train_seeds.sh output)")
    p.add_argument("--group", action="append", default=None,
                   help='"label=glob" of the run directories of one group (replaces the preset groups)')
    p.add_argument("--name", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--max_steps", type=float, default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[500, 1000, 1500])
    p.add_argument("--final_level", type=int, default=3,
                   help="curriculum level of the final task (3 for transport / director)")
    p.add_argument("--font", type=float, default=32)
    p.add_argument("--smooth", type=float, default=0.6, help="EMA weight of the evaluation mean")
    p.add_argument("--threshold", type=float, default=1000, help="curriculum advance threshold to draw")
    p.add_argument("--overlay", action="store_true",
                   help="all groups in one figure (thesis Fig. 3.1) instead of one figure per group")
    p.add_argument("--figsize", type=float, nargs=2, default=[24, 13], help="figure size of --overlay [in]")
    args = p.parse_args()
    pre = presets(args.root)[args.preset]
    groups = [tuple(g.split("=", 1)) for g in args.group] if args.group else pre["groups"]
    name = args.name or pre["name"]

    loaded = []
    for label, pattern in groups:
        dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(os.path.join(d, "tb")))
        if not dirs:
            raise FileNotFoundError(f"no runs for {label}: {pattern}")
        print(f"{label}: {len(dirs)} runs")
        loaded.append((label, dirs, [load_run(d) for d in dirs]))
    end = max(d["rollout/ep_rew_mean"][0][-1] for _, _, runs in loaded for d in runs)
    if args.max_steps:
        end = min(end, args.max_steps)
    grid = np.linspace(0, end, 400)

    rows = summary_rows(loaded, args)
    os.makedirs(args.out, exist_ok=True)
    if args.overlay:
        draw_overlay(loaded, args, grid, end, name)
    for label, dirs, runs in ([] if args.overlay else loaded):
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        title = (f"{args.title}: {label}" if args.title
                 else f"Training progress: {label} ({len(runs)} seeds)")
        draw_group(label, dirs, runs, args, grid, end, title, f"{name}_{slug}")

    path = os.path.join(args.out, f"{name}_summary.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("saved", path)
    fmt = lambda x: "-" if x is None else f"{x / 1e6:.2f}M"
    print(f"\nsteps to reach the return on the final task (L{args.final_level}; train / eval):")
    for r in rows:
        reach = "  ".join(f"{th:g}: {fmt(r[f'steps_to_{th:g}_train'])}/{fmt(r[f'steps_to_{th:g}_eval'])}"
                          for th in args.thresholds)
        print(f"{r['run']:45s} final task from {fmt(r['final_task_from']):>6s}  "
              f"return {r['final_return_last10pct']:7.1f}  {reach}")


if __name__ == "__main__":
    main()
