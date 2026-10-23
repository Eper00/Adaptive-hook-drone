"""Test of the low-level (velocity) policy on the reference profile of
analyse.VELOCITY_PROFILE, as figures for the thesis.

Every run flies the same piecewise-constant reference velocity (12 s) from a
random initial yaw (target yaw 0); in about half of the runs a payload with a
random mass and radius (final curriculum level) hangs in the hook. The errors
are shown together with the reference in the same metric, so that they can be
judged relative to it:

    velocity_tracking.png        velocity components, |v - v_ref| with |v_ref|,
                                 |p - p_ref| with the length of the reference path
    velocity_error_vs_mass.png   RMS velocity error (reference: RMS |v_ref|),
                                 steady-state error per axis (reference: the
                                 reference speed of the segment), final position
                                 error (reference: path length), each also in %
    velocity_yaw.png             yaw angle
    runs.pkl, summary.csv

Usage:
    python -m utilities.test_velocity_policy                 # 1000 runs (~2 min with 16 workers)
    python -m utilities.test_velocity_policy --reuse         # plots from runs.pkl
"""

import argparse
import csv
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from utilities.analyse import DEFAULT_MODELS, VELOCITY_PROFILE, velocity_reference

STEP_TIMES = np.cumsum([d for d, _ in VELOCITY_PROFILE])[:-1]
DURATION = float(sum(d for d, _ in VELOCITY_PROFILE))
SETTLE = 0.9          # [s] after a reference step (and the start), the error counts as steady state
BAND = 0.1            # settling band: +-10 % of the step size
LINE_ALPHA = 0.3      # transparency of the single runs in the time plots


def _fly(seeds):
    """Worker: fly the profile for every seed."""
    from stable_baselines3 import PPO

    from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary

    model = PPO.load(DEFAULT_MODELS["velocity"], device="cpu")
    env = AdaptiveVelocityAviary()
    env.MIN_PAYLOAD_MASS, env.MAX_PAYLOAD_MASS = 0.01, 0.25
    env.MIN_PAYLOAD_RADIUS, env.MAX_PAYLOAD_RADIUS = 0.02, 0.04
    env.GRAB_FLAG_ENABLE = True          # payload in about half of the runs
    env.RANDOM_OREINTATION = True        # random initial yaw
    env.EPISODE_LEN_SEC = DURATION + 1.0
    out = []
    for seed in seeds:
        env.reset(seed=seed)
        payload = env.PAYLOAD_INDICATOR < 0.5
        dt = env.CTRL_TIMESTEP
        p_ref = env.pos[0].copy()
        log = {k: [] for k in ("t", "v_ref", "p_ref", "v", "p", "rpy")}
        t, crashed = 0.0, False
        while t < DURATION - 1e-9:
            v_ref = velocity_reference(t)
            env.TARGET_VEL = v_ref
            action, _ = model.predict(env._computeObs(), deterministic=True)
            for k, v in (("t", t), ("v_ref", v_ref), ("p_ref", p_ref.copy()), ("v", env.vel[0].copy()),
                         ("p", env.pos[0].copy()), ("rpy", env.rpy[0].copy())):
                log[k].append(v)
            _, _, terminated, _, _ = env.step(action)
            p_ref = p_ref + v_ref * dt
            t += dt
            if terminated:
                crashed = True
                break
        r = {k: np.asarray(v) for k, v in log.items()}
        r.update(seed=seed, payload=bool(payload), crashed=crashed,
                 mass=float(env.MASS + 0.01) if payload else 0.0)
        out.append(r)
    env.close()
    return out


def fly_all(n, workers):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    chunks = [list(range(i, n, workers)) for i in range(workers)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        runs = [r for part in ex.map(_fly, chunks) for r in part]
    return sorted(runs, key=lambda r: r["seed"])


def reference_metrics():
    """The reference in the metrics of the errors."""
    t = np.arange(0.0, DURATION, 1.0 / 48)
    v = np.array([velocity_reference(x) for x in t])
    st = steady_mask(t)
    return {"rms_speed_steady": float(np.sqrt(np.mean(np.linalg.norm(v[st], axis=1) ** 2))),
            "path_length": float(sum(d * np.linalg.norm(vv) for d, vv in VELOCITY_PROFILE))}


def steady_mask(t):
    """Time steps at least SETTLE after the last reference step (and the start)."""
    starts = np.r_[0.0, STEP_TIMES]
    last = np.max(np.where(t[:, None] >= starts[None, :], starts[None, :], -np.inf), axis=1)
    return t - last >= SETTLE - 1e-9


def step_metrics(r):
    """Per reference step, with the velocity component along the step
    normalized by the step size (y = 0 before, 1 at the new reference):
        overshoot   peak above the settled value, max(y) - y_final
        undershoot  initial motion against the step, max(0, -min(y))
        offset      settled value minus the reference, y_final - 1
                    (negative: the velocity stays below the reference)
        settle      time until y stays within +-BAND of y_final
    where y_final is the mean of y in the steady-state window."""
    t, v = r["t"], r["v"]
    edges = np.r_[STEP_TIMES, DURATION]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        v0, v1 = velocity_reference(a - 1e-6), velocity_reference(a + 1e-6)
        size = np.linalg.norm(v1 - v0)
        d = (v1 - v0) / size
        m = (t >= a) & (t < b)
        y = (v[m] - v0) @ d / size
        tt = t[m] - a
        y_final = float(y[tt >= SETTLE - 1e-9].mean())
        out_band = np.nonzero(np.abs(y - y_final) > BAND)[0]
        settle = 0.0 if not len(out_band) else tt[min(out_band[-1] + 1, len(tt) - 1)]
        out.append({"t": float(a), "size": float(size), "overshoot": float(max(0.0, y.max() - y_final)),
                    "undershoot": float(max(0.0, -y.min())), "offset": y_final - 1.0,
                    "settle": float(settle)})
    return out


def run_metrics(r, ref):
    st = steady_mask(r["t"])
    ev = np.linalg.norm(r["v"] - r["v_ref"], axis=1)
    steps = step_metrics(r)
    return {"rms_v_steady": float(np.sqrt(np.mean(ev[st] ** 2))),
            "overshoot_max": max(s["overshoot"] for s in steps),
            "undershoot_max": max(s["undershoot"] for s in steps),
            "settle_max": max(s["settle"] for s in steps),
            "steps": steps,
            "p_final": float(np.linalg.norm(r["p"][-1] - r["p_ref"][-1]))}


def style(f):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": f, "axes.labelsize": f, "xtick.labelsize": f - 2,
                         "ytick.labelsize": f - 2, "legend.fontsize": f - 3, "lines.linewidth": 2})


def colors(runs):
    import matplotlib.pyplot as plt
    norm = plt.Normalize(0.0, 0.26)
    return [("tab:red" if not r["payload"] else plt.cm.viridis(norm(r["mass"]))) for r in runs], norm


def plot_tracking(runs, ref, out):
    import matplotlib.pyplot as plt
    cols, norm = colors(runs)
    fig, axs = plt.subplots(5, 1, figsize=(16, 12.5), sharex=True)
    t0 = runs[0]["t"]
    labels = ["$v_x$ [m/s]", "$v_y$ [m/s]", "$v_z$ [m/s]", "Velocity\nerror [m/s]", "Position\nerror [m]"]
    for r, c in zip(runs, cols):
        for i in range(3):
            axs[i].plot(r["t"], r["v"][:, i], color=c, lw=0.8, alpha=LINE_ALPHA)
        axs[3].plot(r["t"], np.linalg.norm(r["v"] - r["v_ref"], axis=1), color=c, lw=0.8, alpha=LINE_ALPHA)
        axs[4].plot(r["t"], np.linalg.norm(r["p"] - r["p_ref"], axis=1), color=c, lw=0.8, alpha=LINE_ALPHA)
    v_ref = np.array([velocity_reference(x) for x in t0])
    for i in range(3):
        axs[i].plot(t0, v_ref[:, i], "k--", lw=2.5)
    # steady-state windows, in which the RMS error is evaluated
    starts = np.r_[0.0, STEP_TIMES]
    ends = np.r_[STEP_TIMES, DURATION]
    for ax in axs[:4]:
        for a, b in zip(starts, ends):
            ax.axvspan(a + SETTLE, b, color="0.85", lw=0, zorder=0)
    for ax, lab in zip(axs, labels):
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        for st in STEP_TIMES:
            ax.axvline(st, color="gray", ls=":", lw=1.5)
    axs[-1].set_xlabel("Time [s]")
    axs[-1].set_xlim(0, DURATION)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, ax=axs, pad=0.02, fraction=0.03, label="Payload mass [kg]")
    h = [plt.Line2D([], [], color="k", ls="--", lw=2.5, label="reference $v_\\mathrm{ref}$"),
         plt.Line2D([], [], color="tab:red", label="no payload"),
         plt.Rectangle((0, 0), 1, 1, color="0.85", label=f"steady state (from {SETTLE:g} s after a step)")]
    axs[0].legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False)
    fig.savefig(os.path.join(out, "velocity_tracking.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


MASS_BINS = [0.02, 0.07, 0.12, 0.17, 0.22, 0.27]   # [kg] groups of the box plots (+ "none")


def plot_error_vs_mass(runs, ref, out):
    """Box plots of the steady-state RMS error, the largest overshoot and the
    largest settling time of every run, grouped by the payload mass."""
    import matplotlib.pyplot as plt
    m = [run_metrics(r, ref) for r in runs]
    mass = np.array([r["mass"] for r in runs])
    pay = np.array([r["payload"] for r in runs])
    groups = [~pay] + [pay & (mass >= lo) & (mass < hi) for lo, hi in zip(MASS_BINS[:-1], MASS_BINS[1:])]
    labels = ["none"] + [f"{lo:.2f}-\n{hi:.2f}" for lo, hi in zip(MASS_BINS[:-1], MASS_BINS[1:])]
    centers = [0.5 * (lo + hi) for lo, hi in zip(MASS_BINS[:-1], MASS_BINS[1:])]
    box_colors = ["tab:red"] + [plt.cm.viridis(c / 0.26) for c in centers]
    fig, axs = plt.subplots(1, 3, figsize=(24, 6.6))

    def boxes(ax, y, ylabel, title, pct_of=None):
        bp = ax.boxplot([y[g] for g in groups], patch_artist=True, widths=0.6,
                        flierprops=dict(marker="o", ms=4, alpha=0.5))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("k")
            med.set_linewidth(2)
        ax.set_xticks(range(1, len(groups) + 1), labels, fontsize=plt.rcParams["font.size"] - 6)
        ax.set_ylim(0, 1.15 * y.max())
        if pct_of:
            sec = ax.secondary_yaxis("right", functions=(lambda v: 100 * v / pct_of, lambda q: q * pct_of / 100))
            sec.set_ylabel("% of RMS $\\|v_\\mathrm{ref}\\|$")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Payload mass [kg]")
        ax.set_title(title, fontsize=plt.rcParams["font.size"] - 1)
        ax.grid(alpha=0.3, axis="y")

    boxes(axs[0], np.array([d["rms_v_steady"] for d in m]), "RMS error [m/s]",
          "Steady-state RMS velocity error", ref["rms_speed_steady"])
    boxes(axs[1], 100 * np.array([d["overshoot_max"] for d in m]), "Overshoot [% of the step]",
          "Largest overshoot of a run")
    boxes(axs[2], np.array([d["settle_max"] for d in m]), "Settling time [s]",
          f"Largest settling time ($\\pm${100 * BAND:g} % band)")
    axs[2].axhline(SETTLE, color="0.4", ls="--", lw=2)
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(out, "velocity_error_vs_mass.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return m


def plot_yaw(runs, out):
    import matplotlib.pyplot as plt
    cols, norm = colors(runs)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axhspan(-5, 5, color="gray", alpha=0.2)
    for r, c in zip(runs, cols):
        ax.plot(r["t"], np.degrees(r["rpy"][:, 2]), color=c, lw=0.8, alpha=LINE_ALPHA)
    for s in STEP_TIMES:
        ax.axvline(s, color="gray", ls=":", lw=1.5)
    ax.set_xlim(0, DURATION)
    ax.set_ylim(-185, 185)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Yaw [deg]")
    ax.grid(alpha=0.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="Payload mass [kg]")
    fig.savefig(os.path.join(out, "velocity_yaw.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--reuse", action="store_true", help="plots from the saved runs")
    p.add_argument("--out", default="results/analysis/velocity")
    p.add_argument("--fig_out", default="results/thesis_figures")
    p.add_argument("--font", type=float, default=24)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.fig_out, exist_ok=True)
    cache = os.path.join(a.out, "runs.pkl")
    if a.reuse:
        with open(cache, "rb") as f:
            runs = pickle.load(f)
    else:
        runs = fly_all(a.runs, a.workers)
        with open(cache, "wb") as f:
            pickle.dump(runs, f)
    ref = reference_metrics()
    style(a.font)
    plot_tracking(runs, ref, a.fig_out)
    m = plot_error_vs_mass(runs, ref, a.fig_out)
    plot_yaw(runs, a.fig_out)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "payload", "mass_kg", "crashed", "rms_v_steady", "overshoot_max", "settle_max",
                    "p_final"])
        for r, d in zip(runs, m):
            w.writerow([r["seed"], int(r["payload"]), round(r["mass"], 4), int(r["crashed"]),
                        round(d["rms_v_steady"], 4), round(d["overshoot_max"], 4), round(d["settle_max"], 3),
                        round(d["p_final"], 4)])
    pay = np.array([r["payload"] for r in runs])
    heavy = np.array([r["mass"] for r in runs]) >= 0.2
    print("reference:", ref)
    print(f"runs: {len(runs)} ({pay.sum()} with payload), crashed: {sum(r['crashed'] for r in runs)}")
    for name, sel in (("no payload", ~pay), ("payload", pay), ("payload >= 0.2 kg", pay & heavy)):
        d = [m[i] for i in np.nonzero(sel)[0]]
        rms = np.array([x["rms_v_steady"] for x in d])
        print(f"  {name:18s} steady RMS {rms.mean():.3f} m/s (max {rms.max():.3f};"
              f" {100 * rms.mean() / ref['rms_speed_steady']:.1f} %), largest overshoot"
              f" {100 * np.mean([x['overshoot_max'] for x in d]):.1f} % (max {100 * np.max([x['overshoot_max'] for x in d]):.1f}),"
              f" largest settling {np.mean([x['settle_max'] for x in d]):.2f} s (max {np.max([x['settle_max'] for x in d]):.2f}),"
              f" largest undershoot {100 * np.max([x['undershoot_max'] for x in d]):.1f} %,"
              f" final p err {np.mean([x['p_final'] for x in d]):.2f} m (max {np.max([x['p_final'] for x in d]):.2f})")
        for j in range(len(d[0]["steps"])):
            st = [x["steps"][j] for x in d]
            print(f"      step at {st[0]['t']:4.1f} s ({st[0]['size']:.2f} m/s): overshoot"
                  f" {100 * np.mean([q['overshoot'] for q in st]):4.1f} % (max {100 * np.max([q['overshoot'] for q in st]):4.1f}),"
                  f" settling {np.mean([q['settle'] for q in st]):.2f} s (max {np.max([q['settle'] for q in st]):.2f}),"
                  f" offset {100 * np.mean([q['offset'] for q in st]):+5.1f} % ({100 * np.min([q['offset'] for q in st]):+5.1f}..{100 * np.max([q['offset'] for q in st]):+5.1f}),"
                  f" undershoot max {100 * np.max([q['undershoot'] for q in st]):4.1f} %")


if __name__ == "__main__":
    main()
