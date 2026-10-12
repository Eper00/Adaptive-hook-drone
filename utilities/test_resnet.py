"""Prediction test of the identified ResNet model on dedicated test flights.

The ResNet (resnet_mpc.py) predicts the closed loop command -> low-pass
filter -> PPO velocity controller -> drone (+ payload). Here it is tested on
new flights that only serve this purpose (seeds 800+, never used for
training or validation), with a known payload mass:

  1. free flight without the payload: ~20 s of random piecewise-constant
     velocity commands (held 0.25-1.5 s) around a point 1.4 m high, with a
     weak position pull that keeps the drone in the area,
  2. the payload is picked up (position P-controller on the env's waypoints,
     director_mpc.WaypointTarget, tendon by the env logic; no excitation);
     1 s after it is lifted,
  3. flight with the payload: ~25 s of random velocity commands, 1.5 m high.

The payload mass is set to a grid of values (--masses), each flown --reps
times. No MPC, RL director or offset estimate is involved: the model is
rolled out open loop from the measured state with the applied commands.

Figures (--out, default results/analysis_identification/resnet_test):
  resnet_timeseries_no_payload.png/.pdf, resnet_timeseries_payload.png/.pdf
      a 10 s stretch of one flight: the simulated position, velocity,
      orientation (and payload swing) and the ResNet's prediction of the
      same signal made 0.25, 0.5 and 1 s earlier (the RMS errors are in
      resnet_test_summary.csv and resnet_error_vs_mass)
  resnet_error_vs_mass.png/.pdf
      RMS prediction error 0.25 / 0.5 / 1 s ahead with the payload, per
      flight, vs the payload mass (dashed: the same without the payload)
  resnet_test_summary.csv   RMS error of every state, mode and look-ahead
  resnet_error_vs_mass.csv  per flight

Usage
-----
    python -m utilities.test_resnet               # fly the test flights + plot (~1-2 min)
    python -m utilities.test_resnet --reuse       # plot the cached flights
"""

import argparse
import csv
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utilities.plot_identification import (C_INK, C_INK2, DEG, GROUPS, HMAX, HORIZONS, errors,
                                           rmse, save, style)
from utilities.validate_resnet import C_AQUA, C_BLUE, C_ORANGE, DT, windows

# one categorical colour per look-ahead (0.25 / 0.5 / 1 s), distinct from the black simulation
H_COLORS = (C_BLUE, C_ORANGE, C_AQUA)

SEED0 = 800
MASSES = tuple(np.round(np.arange(0.01, 0.251, 0.02), 2))   # 13 masses, 0.01-0.25 kg
T_FREE, T_LOAD = 20.0, 25.0
AMP_FREE = np.array([0.5, 0.5, 0.3])       # command excitation amplitude [m/s]
AMP_LOAD = np.array([0.4, 0.4, 0.2])
CENTER_FREE = np.array([0.0, 0.0, 1.4])
LOAD_HEIGHT = 1.5
PULL = 0.8                                 # position pull [1/s] that keeps the drone in the area


################################################################################
# TEST FLIGHTS
################################################################################

def fly(args):
    """One test flight: free flight, pick-up, flight with the payload."""
    seed, mass = args
    from multi_drone_mujoco.envs.director_mpc import (TENDON_HOLD, TILT, U_MAX, WaypointTarget,
                                                      director_step, hook_payload_contact,
                                                      make_identification_env, measure_state,
                                                      payload_attached)

    env = make_identification_env(seed)
    env.set_scenario(env.TARGET_POSITION, env.GOAL_POSITION, mass, env.RADIUS)
    env.prev_action = None
    rest_z = float(env.data.qpos[env.target_qpos_adr + 2])
    target = WaypointTarget()
    dt = env.CTRL_TIMESTEP
    rng = np.random.default_rng(seed)
    log = {k: [] for k in ("x", "u", "attached", "valid", "contact", "phase")}
    state = {"noise": np.zeros(3), "left": 0}

    def excite(x, center, amp):
        if state["left"] <= 0:
            state["noise"] = rng.uniform(-1.0, 1.0, 3) * amp
            state["left"] = int(rng.integers(12, 72))
        state["left"] -= 1
        return state["noise"] + PULL * (center - x[0:3])

    t, attached, tendon, end, t_att = 0.0, False, 0.0, "time", None
    phase, t_phase, center = "takeoff", 0.0, CENTER_FREE
    x = measure_state(env, attached)
    while True:
        if phase == "takeoff":
            cmd = 1.5 * (CENTER_FREE - x[0:3])
            if t >= 2.0:
                phase, t_phase = "free", t
        elif phase == "free":
            cmd = excite(x, CENTER_FREE, AMP_FREE)
            if t - t_phase >= T_FREE:
                phase, t_phase = "pickup", t
        elif phase == "pickup":
            if t - t_phase > 45.0:
                end = "pick-up failed"
                break
            w, _ = target(env, attached)
            cmd = 1.5 * (w - x[0:3])
            # payload hanging on the hook for 1 s: start the excitation
            if attached and t_att is not None and t - t_att >= 1.0:
                phase, t_phase = "load", t
                center = np.array([x[0], x[1], LOAD_HEIGHT])
                # keep the curl direction the env chose for the grasp
                tendon = TENDON_HOLD * getattr(env, "tendon_orientation", 1)
        else:   # load
            cmd = excite(x, center, AMP_LOAD)
            if t - t_phase >= T_LOAD:
                break
        cmd = np.clip(cmd, -U_MAX, U_MAX)
        contact = hook_payload_contact(env) and not attached
        log["x"].append(x)
        log["u"].append(cmd)
        log["attached"].append(attached)
        log["valid"].append(not contact and np.all(np.abs(x[TILT]) < np.radians(60)))
        log["contact"].append(contact)
        log["phase"].append(phase)
        if phase == "pickup":     # env step: waypoints and tendon by the env logic
            _, _, terminated, _, _ = env.step(np.r_[cmd, 0.0, 1.0, 0.0])
            if terminated and not attached:
                end = "pick-up failed"
                break
        else:
            director_step(env, cmd, (tendon, -tendon))
        t += dt
        was = attached
        attached = payload_attached(env, rest_z, attached)
        if attached and not was:
            t_att = t
        x = measure_state(env, attached)
        if x[2] < 0.0 or np.any(np.abs(env.rpy[0, 0:2]) > np.pi / 2):
            end = "crash"
            break
        if phase == "load" and not attached:
            end = "payload lost"
            break
    env.close()
    out = {k: np.asarray(v) for k, v in log.items()}
    out.update(seed=seed, mass=mass, radius=env.RADIUS, end=end, source="test")
    return out


def fly_all(masses, reps, workers):
    from multi_drone_mujoco.envs.resnet_mpc import collect_runs
    jobs = [(fly, (SEED0 + i, m)) for i, m in enumerate(m for m in masses for _ in range(reps))]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return collect_runs(jobs, workers)


################################################################################
# PREDICTIONS
################################################################################

def predict_windows(models, run, mode, starts):
    """Open-loop predictions (n, HMAX+1, NX) from the given start steps."""
    from multi_drone_mujoco.envs.resnet_mpc import rollout
    X0 = np.asarray([run["x"][k] for k in starts], float)
    U = np.asarray([run["u"][k:k + HMAX] for k in starts], float)
    with torch.inference_mode():
        X = rollout(models[mode], torch.as_tensor(X0), torch.as_tensor(U)).numpy()
    return np.concatenate([X0[:, None], X], axis=1)


def window_errors(models, runs, stride):
    """Per mode: predictions and true trajectories of every valid 1 s window,
    with the flight index."""
    out = {}
    for mode in (0, 1):
        P, Y, F = [], [], []
        for f, r in enumerate(runs):
            starts = windows(r, mode, HMAX, stride)
            if not starts:
                continue
            P.append(predict_windows(models, r, mode, starts))
            Y.append(np.asarray([r["x"][k:k + HMAX + 1] for k in starts]))
            F.append(np.full(len(starts), f))
        out[mode] = {"pred": np.concatenate(P), "true": np.concatenate(Y), "flight": np.concatenate(F)}
    return out


################################################################################
# FIGURES
################################################################################

def plot_timeseries(models, run, mode, out, name, length=10.0):
    """Simulated signals and their predictions made 0.25 / 0.5 / 1 s earlier."""
    phase = "load" if mode else "free"
    idx = np.where((run["phase"] == phase) & (run["attached"].astype(bool) == bool(mode)))[0]
    k0 = idx[0] + int(2.0 / DT)                      # skip the start of the phase
    k1 = min(k0 + int(length / DT), idx[-1])
    starts = [k for k in range(k0 - HMAX, k1) if k >= 0 and
              np.all(run["attached"][k:k + HMAX + 1].astype(bool) == bool(mode)) and
              np.all(run["valid"][k:k + HMAX + 1])]
    pred = predict_windows(models, run, mode, starts)
    starts = np.asarray(starts)
    t = (np.arange(k0, k1) - k0) * DT
    rows = [g for g in GROUPS]
    if mode:
        rows.append(("Payload swing", "deg", DEG, [(9, r"$\beta_x$"), (10, r"$\beta_y$")]))
    fig, axes = plt.subplots(len(rows), 3, figsize=(25, 5.2 * len(rows)), sharex=True)
    for r, (group, unit, scale, states) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            if c >= len(states):
                ax.axis("off")
                continue
            i, label = states[c]
            ax.plot(t, run["x"][k0:k1, i] * scale, color=C_INK, lw=2.8, label="simulation", zorder=4)
            for h_i, h in enumerate(HORIZONS):
                # prediction of step k made at k - h
                sel = (starts + h >= k0) & (starts + h < k1)
                tk = (starts[sel] + h - k0) * DT
                y = pred[sel, h, i]
                if i in (13, 14, 15):
                    y = run["x"][starts[sel] + h, i] + ((y - run["x"][starts[sel] + h, i] + np.pi)
                                                        % (2 * np.pi) - np.pi)
                ax.plot(tk, y * scale, color=H_COLORS[h_i], lw=2.2, zorder=3 - h_i * 0.1,
                        label=f"predicted {h * DT:g} s earlier")
            ax.set_ylabel(f"{label} [{unit}]")
            ax.grid()
        axes[r, 0].annotate(group, xy=(-0.4, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=plt.rcParams["axes.titlesize"])
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("Time [s]")
    if mode:
        axes[-1, 1].set_xlabel("Time [s]")
        axes[-2, 2].xaxis.set_tick_params(labelbottom=True)
        axes[-2, 2].set_xlabel("Time [s]")
    what = f"with a {run['mass']:.2f} kg payload" if mode else "without payload"
    fig.suptitle(f"ResNet prediction on a test flight {what}", y=1.0)
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    save(fig, out, name)


MASS_CHANNELS = [
    ("Horizontal velocity", "cm/s", 100, (3, 4)), ("Vertical velocity", "cm/s", 100, (5,)),
    ("Horizontal position", "cm", 100, (0, 1)), ("Vertical position", "cm", 100, (2,)),
    ("Tilt", "deg", DEG, (13, 14)), ("Payload swing", "deg", DEG, (9, 10)),
]


def channel_error(d, idx, scale, h, mask=None):
    """RMS of the error vector norm of the states ``idx`` h steps ahead."""
    e = np.sqrt(sum(errors(d["pred"], d["true"], i, scale)[:, h] ** 2 for i in idx))
    return rmse(e, mask)


def plot_error_vs_mass(err, runs, out):
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(25, 14))
    d1, d0 = err[1], err[0]
    masses = np.array([r["mass"] for r in runs])
    for ax, (title, unit, scale, idx) in zip(axes.flat, MASS_CHANNELS):
        for h_i, h in enumerate(HORIZONS):
            per = []
            for f in np.unique(d1["flight"]):
                m = d1["flight"] == f
                v = channel_error(d1, idx, scale, h, m)
                per.append((masses[f], v))
                rows.append({"channel": title, "unit": unit, "look_ahead_s": h * DT, "seed": runs[f]["seed"],
                             "mass_kg": masses[f], "rmse": round(v, 3), "windows": int(m.sum())})
            per = np.array(sorted(per))
            ax.plot(per[:, 0], per[:, 1], "o", ms=11, color=H_COLORS[h_i], mec="white", mew=1.2, zorder=3)
            um = np.unique(per[:, 0])
            ax.plot(um, [per[per[:, 0] == m, 1].mean() for m in um], color=H_COLORS[h_i], lw=2.4, zorder=2)
            if 9 not in idx:      # no swing without the payload
                ax.axhline(channel_error(d0, idx, scale, h), color=H_COLORS[h_i], lw=2, ls="--", zorder=1)
        ax.set_title(title)
        ax.set_xlabel("Payload mass [kg]")
        ax.set_ylabel(f"RMS error [{unit}]")
        ax.set_ylim(0, None)
        ax.grid()
    fig.suptitle("ResNet prediction error with the payload vs the payload mass", y=1.0)
    fig.tight_layout(h_pad=2.5)
    handles = [plt.Line2D([], [], color=H_COLORS[i], lw=2.4, marker="o", ms=11, mec="white",
                          label=f"{h * DT:g} s ahead") for i, h in enumerate(HORIZONS)]
    handles.append(plt.Line2D([], [], color=C_INK2, lw=2, ls="--", label="same look-ahead, no payload"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    save(fig, out, "resnet_error_vs_mass")
    return rows


def summary_rows(err):
    rows = []
    for mode in (0, 1):
        d = err[mode]
        states = [s for g in GROUPS for s in g[3]] + ([(9, r"$\beta_x$"), (10, r"$\beta_y$")] if mode else [])
        for i, label in states:
            unit = "deg" if i in (9, 10, 13, 14, 15) else ("cm/s" if 3 <= i <= 5 else "cm")
            scale = DEG if unit == "deg" else 100
            e = errors(d["pred"], d["true"], i, scale)
            rows.append({"mode": "payload" if mode else "no payload",
                         "state": label.strip("$").replace("\\", ""), "unit": unit,
                         **{f"rmse_{h * DT:g}s": round(rmse(e[:, h]), 3) for h in HORIZONS},
                         "windows": len(e)})
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masses", type=float, nargs="+", default=list(MASSES))
    p.add_argument("--reps", type=int, default=3, help="flights per payload mass")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--reuse", action="store_true", help="reuse the cached test flights")
    p.add_argument("--resnet", default=None, help="ResNet model file")
    p.add_argument("--stride", type=int, default=2, help="window stride [steps]")
    p.add_argument("--out", default="results/analysis_identification/resnet_test")
    p.add_argument("--font", type=float, default=32)
    args = p.parse_args()

    from multi_drone_mujoco.envs.resnet_mpc import MODEL_FILE, load_model
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "test_flights.pkl")
    if args.reuse and os.path.exists(cache):
        with open(cache, "rb") as f:
            runs = pickle.load(f)
    else:
        runs = fly_all(args.masses, args.reps, args.workers)
        with open(cache, "wb") as f:
            pickle.dump(runs, f)
    for r in runs:
        ph = r["phase"]
        print(f"  seed {r['seed']}  mass {r['mass']:.2f} kg  R {r['radius'] * 100:.1f} cm  "
              f"free {np.sum(ph == 'free') * DT:4.1f} s  load {np.sum(ph == 'load') * DT:4.1f} s  end: {r['end']}")
    runs = [r for r in runs if np.any(r["phase"] == "load")]
    print(f"{len(runs)} flights with a payload phase")

    torch.set_num_threads(4)
    models = load_model(args.resnet or MODEL_FILE)
    err = window_errors(models, runs, args.stride)
    for mode in (0, 1):
        print(f"mode {mode}: {len(err[mode]['true'])} windows")
    style(args.font)
    summary = summary_rows(err)
    write_csv(summary, os.path.join(args.out, "resnet_test_summary.csv"))
    write_csv(plot_error_vs_mass(err, runs, args.out), os.path.join(args.out, "resnet_error_vs_mass.csv"))
    # time series: the flight closest to the middle of the mass range
    mid = np.median([r["mass"] for r in runs])
    run = min(runs, key=lambda r: abs(r["mass"] - mid))
    plot_timeseries(models, run, 0, args.out, "resnet_timeseries_no_payload")
    plot_timeseries(models, run, 1, args.out, "resnet_timeseries_payload")
    print("\nRMS error (0.25 / 0.5 / 1 s):")
    for r in summary:
        print(f"  {r['mode']:10s} {r['state']:12s} " + " / ".join(f"{r[f'rmse_{h * DT:g}s']:.2f}" for h in HORIZONS)
              + f" {r['unit']}")


if __name__ == "__main__":
    main()
