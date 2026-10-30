"""A typical flight of each identification data set, side by side (thesis figure).

    Dataset 1   P-controller flights of the transportation task with random,
                piecewise-constant commands added (training data,
                director_mpc.collect)
    Dataset 2   flights of the high-level policy (training data,
                resnet_mpc._collect_rl)
    Test set 1  dedicated test flights: free flight around a fixed point, pick-up,
                flight with the payload, random commands throughout (test_resnet.fly)

(Test set 2 consists of further flights of the high-level policy, recorded like
Dataset 2 with other seeds, so it is not shown separately.)

One column per data set, rows: path from above (thick while the payload is
carried), velocity command u, height, and tilt / payload swing over time.
Shaded: the payload is carried; in Test set 1 also the phases of the flight.
The flight shown: Dataset 1 the flight of median length, Dataset 2 the flight
of median maximum tilt (both among the training flights), Test set 1 the first
flight with a 0.13 kg disk (as in the identification figures); --d1, --d2,
--t1 select others by seed.

Usage:
    python -m utilities.plot_datasets
    python -m utilities.plot_datasets --d1 105 --d2 310 --t1 812
Output: results/analysis/identification/datasets_trajectories.png/.pdf
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multi_drone_mujoco.envs.director_mpc import P, TH, TILT

DATA_FILE = "results/resnet/resnet_training_data.pkl"
TEST_FILE = "results/analysis/identification/test_flights.pkl"
OUT = "results/analysis/identification"
DT = 1.0 / 48.0
C_X, C_Y, C_Z = "#2a78d6", "#eb6834", "#1baf7a"
C_PATH, C_LOAD, C_INK, C_INK2 = "#52514e", "#2a78d6", "#0b0b0b", "#52514e"
PHASE_COLORS = {"takeoff": "#ffffff", "free": "#eef3fb", "pickup": "#fdf1e8", "load": "#e8f6ef"}
PHASE_NAMES = {"free": "free flight", "pickup": "pick-up", "load": "with payload"}


def pick(runs, seed, key):
    """The run with ``seed``, or the one with the median value of ``key``."""
    if seed is not None:
        return next(r for r in runs if int(r["seed"]) == seed)
    runs = [r for r in runs if r["attached"].any()]          # flights that pick up the payload
    order = np.argsort([key(r) for r in runs])
    return runs[order[len(order) // 2]]


def spans(mask, t):
    """(start, end) times of the True runs of ``mask``."""
    m = np.r_[False, mask, False].astype(int)
    edges = np.flatnonzero(np.diff(m))
    return [(t[a], t[min(b, len(t)) - 1] + DT) for a, b in zip(edges[::2], edges[1::2])]


def column(axs, run, title, phases=None):
    x, u = run["x"], run["u"]
    att = run["attached"].astype(bool)
    t = np.arange(len(x)) * DT
    p = x[:, P]
    tilt = np.degrees(np.arctan(np.hypot(np.tan(x[:, TILT][:, 0]), np.tan(x[:, TILT][:, 1]))))
    swing = np.degrees(np.hypot(*x[:, TH].T))

    # path from above, thick while the payload is carried
    ax = axs[0]
    ax.plot(p[:, 0], p[:, 1], color=C_PATH, lw=1.2)
    ax.plot(np.where(att, p[:, 0], np.nan), np.where(att, p[:, 1], np.nan), color=C_LOAD, lw=3.2)
    ax.plot(*p[0, :2], "o", color=C_INK, ms=7, zorder=5)
    if att.any():
        k = int(np.argmax(att))
        ax.plot(*p[k, :2], "s", color="#d1453b", ms=8, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_title(title, fontsize=15, fontweight="bold")

    # background: payload carried (all), flight phases (Test set 1)
    for ax in axs[1:]:
        if phases is not None:
            for ph, c in PHASE_COLORS.items():
                for a, b in spans(phases == ph, t):
                    ax.axvspan(a, b, color=c, lw=0, zorder=0)
        else:
            for a, b in spans(att, t):
                ax.axvspan(a, b, color="#e8f0fb", lw=0, zorder=0)
    if phases is not None:
        for ph, name in PHASE_NAMES.items():
            sp = spans(phases == ph, t)
            if sp:
                axs[1].text((sp[0][0] + sp[-1][1]) / 2, 1.02, name, transform=axs[1].get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=10, color=C_INK2)

    ax = axs[1]                                              # velocity command
    for i, (c, n) in enumerate(zip((C_X, C_Y, C_Z), ("$u_x$", "$u_y$", "$u_z$"))):
        ax.plot(t, u[:, i], color=c, lw=1.1, label=n)
    ax.set_ylim(-1.05, 1.05)

    ax = axs[2]                                              # height
    ax.plot(t, p[:, 2], color=C_INK, lw=1.4)

    ax = axs[3]                                              # tilt and swing
    ax.plot(t, tilt, color=C_INK, lw=1.2, label="tilt")
    ax.plot(np.where(att, t, np.nan), np.where(att, swing, np.nan), color=C_Y, lw=1.2, label="swing")
    ax.set_xlabel("time [s]")
    for ax in axs[1:]:
        ax.set_xlim(0, t[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--d1", type=int, default=None, help="seed of the Dataset 1 flight")
    p.add_argument("--d2", type=int, default=None, help="seed of the Dataset 2 flight")
    p.add_argument("--t1", type=int, default=None, help="seed of the Test set 1 flight")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    with open(DATA_FILE, "rb") as f:
        train, _ = pickle.load(f)
    with open(TEST_FILE, "rb") as f:
        tests = pickle.load(f)
    d1 = pick([r for r in train if r["source"] == "P"], a.d1, lambda r: len(r["x"]))
    d2 = pick([r for r in train if r["source"] == "RL"], a.d2, lambda r: np.abs(r["x"][:, TILT]).max())
    if a.t1 is None:
        t1 = next(r for r in tests if abs(float(r["mass"]) - 0.13) < 1e-6)
    else:
        t1 = next(r for r in tests if int(r["seed"]) == a.t1)

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.edgecolor": C_INK2,
                         "axes.labelcolor": C_INK2, "xtick.color": C_INK2, "ytick.color": C_INK2})
    fig, axs = plt.subplots(4, 3, figsize=(16, 13), gridspec_kw=dict(height_ratios=[1.6, 1, 1, 1]))
    column(axs[:, 0], d1, f"Dataset 1: P-controller + random commands\n(seed {d1['seed']})")
    column(axs[:, 1], d2, f"Dataset 2: high-level policy\n(seed {d2['seed']})")
    column(axs[:, 2], t1, f"Test set 1: test flight, {float(t1['mass']):.2f} kg disk\n(seed {t1['seed']})",
           phases=np.asarray(t1["phase"]))
    axs[0, 0].set_ylabel("y [m]")
    axs[1, 0].set_ylabel("command u [m/s]")
    axs[2, 0].set_ylabel("height [m]")
    axs[3, 0].set_ylabel("angle [deg]")
    for r in (1, 2, 3):                                      # same y range in a row
        lo = min(ax.get_ylim()[0] for ax in axs[r])
        hi = max(ax.get_ylim()[1] for ax in axs[r])
        for ax in axs[r]:
            ax.set_ylim(lo, hi)
    axs[1, 0].legend(loc="lower left", ncol=3, fontsize=10, frameon=False)
    axs[3, 0].legend(loc="upper left", ncol=2, fontsize=10, frameon=False)
    handles = [plt.Line2D([], [], color=C_PATH, lw=1.2, label="path"),
               plt.Line2D([], [], color=C_LOAD, lw=3.2, label="path with the payload"),
               plt.Line2D([], [], marker="o", color=C_INK, ls="", label="start"),
               plt.Line2D([], [], marker="s", color="#d1453b", ls="", label="payload lifted"),
               plt.Rectangle((0, 0), 1, 1, color="#e8f0fb", label="payload carried")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(a.out, f"datasets_trajectories.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print("saved:", path)
    print(f"flights: Dataset 1 seed {d1['seed']} ({len(d1['x']) * DT:.1f} s), Dataset 2 seed {d2['seed']}"
          f" ({len(d2['x']) * DT:.1f} s), Test set 1 seed {t1['seed']} ({len(t1['x']) * DT:.1f} s)")


if __name__ == "__main__":
    main()
