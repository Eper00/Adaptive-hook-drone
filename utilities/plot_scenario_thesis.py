"""One scenario of ``analyse --test scenario``, as a figure for the thesis.

Reads the episodes saved by

    python -m utilities.analyse --test scenario --seed 83 --max_time 25

and draws the top view of the paths (left) and the distance to the goal and
the tilt over time (right), with the outcome rule of analyse.classify: an
episode is a success once the drone is within GOAL_TOLERANCE of the goal with
the payload and keeps the payload for GOAL_HOLD_TIME s (the hold window is
shaded); it is a stability failure when the drone flips over (tilt > 90 deg)
or hits the ground (X). Every method is drawn only until its outcome is decided (the end of its
hold window, or the failure), since what happens later does not count.

Usage:
    python -m utilities.plot_scenario_thesis
    python -m utilities.plot_scenario_thesis --run results/analysis/scenario/seed_83 --out fig.png
"""

import argparse
import json
import os

import numpy as np

from utilities.analyse import GOAL_HOLD_TIME, GOAL_TOLERANCE

METHODS = [("director", "RL director", "#3b6fb6"), ("director_mpc", "MPC director", "#e08a2c"),
           ("director_safety", "RL + safety filter", "#1ba39c")]


def tilt_deg(ep):
    return np.degrees(np.arccos(np.clip(np.cos(ep["rpy"][:, 0]) * np.cos(ep["rpy"][:, 1]), -1, 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="results/analysis/scenario/seed_83")
    p.add_argument("--out", default=None, help="default: results/thesis_figures/scenario_seed<seed>.png")
    p.add_argument("--font", type=float, default=19)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    with open(os.path.join(a.run, "scenario.json")) as f:
        sc = json.load(f)
    goal, pay, start = (np.array(sc[k]) for k in ("goal_position", "payload_center", "start_position"))
    eps = {k: dict(np.load(os.path.join(a.run, f"{k}.npz"))) for k, _, _ in METHODS}
    events = {k: float(e["event_time"]) for k, e in eps.items()}
    # every method is drawn until its outcome is decided: the end of the hold
    # window for a success, the failure itself otherwise (later is irrelevant)
    decided = {k: events[k] + (GOAL_HOLD_TIME if str(eps[k]["outcome"]) == "success" else 0)
               for k in eps}
    t_end = max(decided.values()) + 0.3

    f = a.font
    plt.rcParams.update({"font.size": f, "axes.labelsize": f, "xtick.labelsize": f - 2,
                         "ytick.labelsize": f - 2, "legend.fontsize": f - 3, "lines.linewidth": 2.5})
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], wspace=0.22, hspace=0.12)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1], sharex=ax_d)

    for k, name, c in METHODS:
        e = eps[k]
        t, pos = e["t"], e["p"]
        keep = t <= decided[k] + 1e-9
        ok = str(e["outcome"]) == "success"
        ke = int(np.searchsorted(t, events[k]))
        ke = min(ke, len(t) - 1)
        # top view, thick where the payload is carried
        ax_xy.plot(pos[keep, 0], pos[keep, 1], color=c, lw=2)
        att = e["attached"].astype(bool) & keep
        ax_xy.plot(np.where(att, pos[:, 0], np.nan), np.where(att, pos[:, 1], np.nan), color=c, lw=6,
                   alpha=0.35)
        if k == "director_safety":
            iv = e["intervened"].astype(bool) & keep
            ax_xy.scatter(pos[iv, 0], pos[iv, 1], s=10, color="m", zorder=3)
        mk = dict(marker="*" if ok else "X", s=420 if ok else 300, color=c, edgecolor="k", zorder=5)
        ax_xy.scatter(pos[ke, 0], pos[ke, 1], **mk)
        # distance to the goal and tilt over time
        d = np.linalg.norm(pos - goal, axis=1)
        ax_d.plot(t[keep], d[keep], color=c)
        ax_t.plot(t[keep], tilt_deg(e)[keep], color=c)
        ax_d.scatter(t[ke], d[ke], **mk)
        ax_t.scatter(t[ke], tilt_deg(e)[ke], **mk)
        if ok:                                   # hold window of the success rule
            for ax in (ax_d, ax_t):
                ax.axvspan(events[k], events[k] + GOAL_HOLD_TIME, color=c, alpha=0.12, lw=0)

    ax_xy.add_patch(Circle(goal[:2], GOAL_TOLERANCE, color="g", alpha=0.2))
    ax_xy.scatter(*start[:2], marker="o", s=150, color="k", zorder=4)
    ax_xy.scatter(*pay[:2], marker="s", s=170, color="r", zorder=4)
    ax_xy.scatter(*goal[:2], marker="P", s=220, color="g", zorder=4)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(alpha=0.3)

    ax_d.axhline(GOAL_TOLERANCE, color="g", ls="--", lw=1.8)
    ax_d.set_ylabel("Distance to\ngoal [m]")
    ax_t.axhline(90, color="#d1453b", ls="--", lw=1.8)
    ax_t.set_ylabel("Tilt [deg]")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_xlim(0, t_end)
    for ax in (ax_d, ax_t):
        ax.grid(alpha=0.3)
    plt.setp(ax_d.get_xticklabels(), visible=False)
    fig.align_ylabels([ax_d, ax_t])

    handles = [Line2D([], [], color=c, label=name) for _, name, c in METHODS] + [
        Line2D([], [], marker="X", color="w", markerfacecolor="gray", markeredgecolor="k", ms=15,
               label="flips over"),
        Line2D([], [], marker="*", color="w", markerfacecolor="gray", markeredgecolor="k", ms=20,
               label="goal reached"),
        Line2D([], [], color="gray", lw=8, alpha=0.3,
               label=f"hold window ({GOAL_HOLD_TIME:g} s)"),
        Line2D([], [], color="g", ls="--", label=f"goal tolerance ({GOAL_TOLERANCE:g} m)"),
        Line2D([], [], color="#d1453b", ls="--", label="flip-over limit (90°)"),
        Line2D([], [], marker="s", color="w", markerfacecolor="r", ms=13, label="payload"),
        Line2D([], [], marker="o", color="w", markerfacecolor="m", ms=8, label="filter intervenes")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, columnspacing=1.2, handletextpad=0.4)
    out = a.out or os.path.join("results", "thesis_figures", f"scenario_seed{sc['seed']}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
