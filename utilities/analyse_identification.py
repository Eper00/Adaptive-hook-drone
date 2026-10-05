"""Compare the two identified director models against MuJoCo.

The director MPC and the safety filter predict the closed loop

    velocity command -> env low-pass filter -> PPO velocity controller -> drone (+ payload)

with one of two identified models (both with 23 states, mode 0 without and
mode 1 with the payload, see docs/director_mpc.md):

    ARX     linear per-axis ARX velocity loop + pendulum swing + tilt ARX,
            output-error fit on P-controller identification flights
            (director_mpc.py, results/mpc_director/identified_model.npz)
    ResNet  x_k+1 = x_k + f(x_k, u_k), f a residual network trained with a
            multi-step loss on P-controller, strongly excited and RL director
            flights (resnet_mpc.py, results/mpc_director/resnet_model.pt)

To separate the effect of the method from the effect of the data, the ARX
model is also re-fitted on exactly the ResNet's training data ("ARX (same
data)").

The script flies fresh MuJoCo test episodes (seeds that were not used for
training): P-controller missions with random command excitation and flights
of the RL director policy. From every ``--stride``-th step it rolls each
model out open loop with the commands that were actually applied and
compares the predictions with the simulation and with each other:

  error_vs_horizon.png   RMS error vs look-ahead (position, velocity, tilt,
                         swing) per data source and mode, and the ARX <-> ResNet
                         disagreement
  summary_bars.png       RMS errors at 0.25 / 0.5 / 0.83 s look-ahead
  timeseries_*.png       one flight: MuJoCo vs the 0.5 s-ahead predictions
  fans_*.png             one flight: full-horizon predictions from every 1 s
  contact_error.png      the same error while the hook touches the payload
                         before it is lifted (not modelled by either model)
  summary.json / .csv    all numbers, and the verdict

Usage:
    python -m utilities.analyse_identification                  # default test set
    python -m utilities.analyse_identification --rl_flights 24 --p_flights 4
    python -m utilities.analyse_identification --reuse          # reuse cached test flights
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

from multi_drone_mujoco.envs.director_mpc import (
    DEFAULT_MODEL_FILE, NX, fit_director_models, load_models, make_identification_env,
)
from multi_drone_mujoco.envs.resnet_mpc import (
    DATA_FILE, DIRECTOR_POLICY, MODEL_FILE, _collect_p, _collect_rl, collect_runs, load_model,
)

DT = 1.0 / 48.0
HORIZON = 50                     # 0.83 s
TABLE_STEPS = (12, 24, 40)       # 0.25 / 0.5 / 0.83 s
CHANNELS = {                     # name: (state indices, scale to display unit, unit)
    "position": ([0, 1, 2], 100.0, "cm"),
    "velocity": ([3, 4, 5], 100.0, "cm/s"),
    "tilt": ([19, 20], np.degrees(1.0), "deg"),
    "swing": ([15, 16], np.degrees(1.0), "deg"),
}
MODEL_STYLE = {
    "ARX": dict(color="tab:orange", ls="-"),
    "ARX (same data)": dict(color="tab:red", ls=":"),
    "ResNet": dict(color="tab:blue", ls="-"),
}
P_SEEDS = (500, 501, 502, 503, 504, 505)
P_EXCITE = (0.15, 0.4, 0.6)
RL_SEEDS0 = 600


################################################################################
# DATA
################################################################################

def collect_test_flights(n_p, n_rl, policy, workers=8, verbose=True):
    jobs = [(_collect_p, (P_SEEDS[i], P_EXCITE[i % len(P_EXCITE)])) for i in range(n_p)]
    if os.path.exists(policy):
        jobs += [(_collect_rl, (RL_SEEDS0 + i, policy, 15.0)) for i in range(n_rl)]
    elif verbose:
        print(f"(no RL director policy at {policy}: P-controller flights only)")
    if verbose:
        print(f"Flying {len(jobs)} MuJoCo test episodes ({n_p} P-controller, "
              f"{len(jobs) - n_p} RL director)...")
    t0 = time.time()
    runs = collect_runs(jobs, workers)
    if verbose:
        print(f"  done in {time.time() - t0:.0f} s")
    return runs


def arx_on_resnet_data(path, verbose=True):
    """The ARX structure fitted on the ResNet's training runs (cached in ``path``)."""
    from multi_drone_mujoco.envs.director_mpc import save_models
    if os.path.exists(path):
        return load_models(path)
    if not os.path.exists(DATA_FILE):
        return None
    with open(DATA_FILE, "rb") as f:
        train_runs, _ = pickle.load(f)
    masks = [[], []]
    for r in train_runs:
        att = r["attached"].astype(bool)
        valid = r["valid"].astype(bool)
        for mode in (0, 1):
            m = (att if mode else ~att) & valid
            m[:3] = False
            m[-3:] = False
            masks[mode].append(m)
    env = make_identification_env(0)
    alpha, dt = env.alpha, env.CTRL_TIMESTEP
    env.close()
    if verbose:
        print("Fitting the ARX model on the ResNet training data...")
    axes, tilts = fit_director_models(train_runs, masks, alpha, dt, verbose=False)
    save_models(path, axes, tilts, alpha, dt)
    return load_models(path)


def windows(run, mode, H, stride, kind="valid"):
    """Start steps of H-step windows that stay in one mode and are valid
    (kind="valid") or that start while the hook touches the payload before it
    is lifted (kind="contact")."""
    att = run["attached"].astype(bool)
    T = len(att)
    same = att == bool(mode)
    if kind == "valid":
        ok = same & run["valid"].astype(bool)
        starts = [k for k in range(1, T - H, stride) if ok[k:k + H + 1].all()]
    else:
        contact = run.get("contact", np.zeros(T, bool)).astype(bool)
        starts = [k for k in range(1, T - H, max(1, stride // 2))
                  if contact[k] and same[k:k + H + 1].all()]
    return starts


################################################################################
# EVALUATION
################################################################################

def channel_error(pred, true, channel):
    idx, scale, _ = CHANNELS[channel]
    return np.linalg.norm(pred[idx] - true[idx], axis=0) * scale


def evaluate(runs, models, H=HORIZON, stride=4, kind="valid"):
    """err[(source, mode)][model][channel] -> (n_windows, H+1) errors, plus the
    ARX <-> ResNet disagreement under the key "ARX vs ResNet"."""
    out = {}
    for src in sorted({r["source"] for r in runs}):
        for mode in (0, 1):
            acc = {name: {c: [] for c in CHANNELS} for name in list(models) + ["ARX vs ResNet"]}
            n = 0
            for r in (r for r in runs if r["source"] == src):
                for k in windows(r, mode, H, stride, kind):
                    true = r["x"][k:k + H + 1].T
                    U = r["u"][k:k + H].T
                    preds = {name: m[mode].rollout(r["x"][k], U) for name, m in models.items()}
                    for name, Xp in preds.items():
                        for c in CHANNELS:
                            acc[name][c].append(channel_error(Xp, true, c))
                    if "ARX" in preds and "ResNet" in preds:
                        for c in CHANNELS:
                            acc["ARX vs ResNet"][c].append(
                                channel_error(preds["ARX"], preds["ResNet"], c))
                    n += 1
            if n:
                out[(src, mode)] = {name: {c: np.asarray(v) for c, v in d.items() if v}
                                    for name, d in acc.items()}
    return out


def rms(a, axis=0):
    return np.sqrt(np.mean(np.square(a), axis=axis))


def summarize(err, models):
    """Nested dict of RMS errors at TABLE_STEPS and the ResNet / ARX ratios."""
    rows = []
    for (src, mode), d in err.items():
        n = len(next(iter(d["ResNet"].values()))) if "ResNet" in d else 0
        for c in CHANNELS:
            if mode == 0 and c == "swing":
                continue
            for name in list(models) + ["ARX vs ResNet"]:
                if c not in d.get(name, {}):
                    continue
                e = rms(d[name][c])
                rows.append({"source": src, "mode": mode, "channel": c, "model": name,
                             "windows": n, "unit": CHANNELS[c][2],
                             **{f"rms@{s / 48:.2f}s": float(e[s]) for s in TABLE_STEPS},
                             "rms_1step": float(e[1])})
    return rows


def verdict(rows):
    """Geometric mean of the ResNet / ARX error ratios and the win count."""
    ratios, wins, total = [], 0, 0
    per = {}
    for r in rows:
        if r["model"] != "ResNet":
            continue
        arx = next((a for a in rows if a["model"] == "ARX" and a["source"] == r["source"]
                    and a["mode"] == r["mode"] and a["channel"] == r["channel"]), None)
        if arx is None:
            continue
        for s in TABLE_STEPS:
            key = f"rms@{s / 48:.2f}s"
            q = r[key] / max(arx[key], 1e-9)
            ratios.append(q)
            per.setdefault((r["source"], r["channel"]), []).append(q)
            wins += q < 1.0
            total += 1
    g = float(np.exp(np.mean(np.log(ratios)))) if ratios else np.nan
    return {"resnet_over_arx_error_ratio": g, "resnet_wins": int(wins), "comparisons": int(total),
            "per_source_channel": {f"{k[0]}/{k[1]}": float(np.exp(np.mean(np.log(v))))
                                   for k, v in per.items()},
            "more_accurate": "ResNet" if g < 1.0 else "ARX"}


################################################################################
# PLOTS
################################################################################

def plot_error_vs_horizon(err, models, out):
    import matplotlib.pyplot as plt
    keys = sorted(err)
    chans = list(CHANNELS)
    fig, axes = plt.subplots(len(keys), len(chans), figsize=(4.2 * len(chans), 2.8 * len(keys)),
                             squeeze=False)
    t = np.arange(HORIZON + 1) * DT
    for i, key in enumerate(keys):
        src, mode = key
        d = err[key]
        n = len(next(iter(d["ResNet"].values())))
        for j, c in enumerate(chans):
            ax = axes[i, j]
            if mode == 0 and c == "swing":
                ax.axis("off")
                continue
            for name in models:
                if c in d.get(name, {}):
                    ax.plot(t, rms(d[name][c]), label=name, **MODEL_STYLE[name])
            if c in d.get("ARX vs ResNet", {}):
                ax.plot(t, rms(d["ARX vs ResNet"][c]), color="gray", ls="--", lw=1,
                        label="ARX vs ResNet")
            ax.set_title(f"{src} flights, mode {mode} ({n} windows): {c}", fontsize=9)
            ax.set_xlabel("look-ahead [s]")
            ax.set_ylabel(f"RMS error [{CHANNELS[c][2]}]")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Open-loop prediction error vs MuJoCo (mode 0: no payload, mode 1: payload)")
    fig.tight_layout()
    path = os.path.join(out, "error_vs_horizon.png")
    fig.savefig(path, dpi=120)
    print("saved:", path)
    return fig


def plot_summary_bars(rows, models, out):
    import matplotlib.pyplot as plt
    groups = sorted({(r["source"], r["mode"], r["channel"]) for r in rows},
                    key=lambda g: (list(CHANNELS).index(g[2]), g[0], g[1]))
    fig, axes = plt.subplots(1, len(TABLE_STEPS), figsize=(6 * len(TABLE_STEPS), 4.5))
    names = list(models)
    w = 0.8 / len(names)
    for ax, s in zip(axes, TABLE_STEPS):
        key = f"rms@{s / 48:.2f}s"
        for m_i, name in enumerate(names):
            vals = []
            for g in groups:
                r = next((r for r in rows if (r["source"], r["mode"], r["channel"]) == g
                          and r["model"] == name), None)
                ref = next((r for r in rows if (r["source"], r["mode"], r["channel"]) == g
                            and r["model"] == "ARX"), None)
                vals.append(r[key] / ref[key] if r and ref else np.nan)
            ax.bar(np.arange(len(groups)) + (m_i - (len(names) - 1) / 2) * w, vals, w,
                   label=name, color=MODEL_STYLE[name]["color"])
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([f"{c}\n{src} m{m}" for src, m, c in groups], fontsize=7, rotation=90)
        ax.set_ylabel("RMS error relative to ARX")
        ax.set_title(f"look-ahead {s / 48:.2f} s")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Prediction error relative to the ARX model (< 1: more accurate than ARX)")
    fig.tight_layout()
    path = os.path.join(out, "summary_bars.png")
    fig.savefig(path, dpi=120)
    print("saved:", path)
    return fig


def _shade_modes(ax, t, run):
    att = run["attached"].astype(bool)
    bad = ~run["valid"].astype(bool)
    for mask, color in ((att, "tab:green"), (bad, "tab:red")):
        k = 0
        while k < len(mask):
            if mask[k]:
                k1 = k
                while k1 < len(mask) and mask[k1]:
                    k1 += 1
                ax.axvspan(t[k], t[min(k1, len(t) - 1)], color=color, alpha=0.08, lw=0)
                k = k1
            k += 1


def plot_timeseries(run, models, out, lookahead=24):
    """MuJoCo vs the ``lookahead``-step-ahead predictions of each model."""
    import matplotlib.pyplot as plt
    x, u = run["x"], run["u"]
    att = run["attached"].astype(bool)
    T = len(x)
    t = np.arange(T) * DT
    pred = {name: np.full((T, NX), np.nan) for name in models}
    for k in range(lookahead, T):
        k0 = k - lookahead
        seg = att[k0:k + 1]
        if not (seg.all() or (~seg).all()) or not run["valid"][k0:k + 1].all():
            continue
        mode = int(att[k0])
        for name, m in models.items():
            pred[name][k] = m[mode].rollout(x[k0], u[k0:k].T)[:, -1]
    rows = [("v_x [m/s]", 3, 1), ("v_y [m/s]", 4, 1), ("v_z [m/s]", 5, 1),
            ("roll (world) [deg]", 19, np.degrees(1)), ("pitch (world) [deg]", 20, np.degrees(1)),
            ("swing x [deg]", 15, np.degrees(1)), ("swing y [deg]", 16, np.degrees(1))]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.0 * len(rows)), sharex=True)
    for ax, (label, i, sc) in zip(axes, rows):
        _shade_modes(ax, t, run)
        ax.plot(t, x[:, i] * sc, color="k", lw=1.4, label="MuJoCo")
        for name in models:
            ax.plot(t, pred[name][:, i] * sc, lw=1, label=f"{name} ({lookahead / 48:.2f} s ahead)",
                    **{k: v for k, v in MODEL_STYLE[name].items() if k == "color"})
        if i in (3, 4, 5):
            ax.plot(t, u[:, i - 3], color="gray", lw=0.6, alpha=0.6, label="command")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("time [s]  (green: payload attached, red: hook contact / not modelled)")
    fig.suptitle(f"{run['source']} flight (seed {run.get('seed', '?')}): MuJoCo vs "
                 f"{lookahead / 48:.2f} s-ahead model predictions")
    fig.tight_layout()
    path = os.path.join(out, f"timeseries_{run['source']}_{run.get('seed', 0)}.png")
    fig.savefig(path, dpi=110)
    print("saved:", path)
    return fig


def plot_fans(run, models, out, every=48, H=HORIZON):
    """Full-horizon open-loop predictions from every ``every`` steps."""
    import matplotlib.pyplot as plt
    x, u = run["x"], run["u"]
    att = run["attached"].astype(bool)
    T = len(x)
    t = np.arange(T) * DT
    rows = [("x [m]", 0, 1), ("y [m]", 1, 1), ("z [m]", 2, 1), ("v_x [m/s]", 3, 1),
            ("v_y [m/s]", 4, 1), ("pitch (world) [deg]", 20, np.degrees(1)),
            ("roll (world) [deg]", 19, np.degrees(1))]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.9 * len(rows)), sharex=True)
    for ax, (label, i, sc) in zip(axes, rows):
        _shade_modes(ax, t, run)
        ax.plot(t, x[:, i] * sc, color="k", lw=1.4)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    for k in range(1, T - H, every):
        seg = att[k:k + H + 1]
        if not (seg.all() or (~seg).all()) or not run["valid"][k:k + H + 1].all():
            continue
        mode = int(att[k])
        tt = t[k:k + H + 1]
        for name, m in models.items():
            Xp = m[mode].rollout(x[k], u[k:k + H].T)
            for ax, (_, i, sc) in zip(axes, rows):
                ax.plot(tt, Xp[i] * sc, lw=1.1, color=MODEL_STYLE[name]["color"], alpha=0.9)
                ax.plot(tt[0], Xp[i, 0] * sc, "o", ms=2.5, color="k")
    handles = [plt.Line2D([], [], color="k", lw=1.4, label="MuJoCo")] + [
        plt.Line2D([], [], color=MODEL_STYLE[n]["color"], label=f"{n} ({H / 48:.2f} s prediction)")
        for n in models]
    axes[0].legend(handles=handles, fontsize=7, ncol=4, loc="upper right")
    axes[-1].set_xlabel("time [s]  (green: payload attached, red: hook contact / not modelled)")
    fig.suptitle(f"{run['source']} flight (seed {run.get('seed', '?')}): open-loop prediction "
                 f"fans (every {every / 48:.1f} s)")
    fig.tight_layout()
    path = os.path.join(out, f"fans_{run['source']}_{run.get('seed', 0)}.png")
    fig.savefig(path, dpi=110)
    print("saved:", path)
    return fig


def plot_contact(err_valid, err_contact, models, out):
    """Error growth in free flight vs while the hook touches the payload."""
    import matplotlib.pyplot as plt
    keys = [k for k in err_contact if k[1] == 0]
    if not keys:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    t = np.arange(HORIZON + 1) * DT
    for ax, c in zip(axes, ("velocity", "tilt")):
        for key in keys:
            for name in models:
                if c in err_contact[key].get(name, {}):
                    ax.plot(t, rms(err_contact[key][name][c]), lw=2,
                            color=MODEL_STYLE[name]["color"], label=f"{name}, hook contact ({key[0]})")
                if key in err_valid and c in err_valid[key].get(name, {}):
                    ax.plot(t, rms(err_valid[key][name][c]), lw=1, ls="--",
                            color=MODEL_STYLE[name]["color"], label=f"{name}, free flight ({key[0]})")
        ax.set_title(f"{c}: free flight vs hook-payload contact (mode 0)")
        ax.set_xlabel("look-ahead [s]")
        ax.set_ylabel(f"RMS error [{CHANNELS[c][2]}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out, "contact_error.png")
    fig.savefig(path, dpi=120)
    print("saved:", path)
    return fig


################################################################################
# MAIN
################################################################################

def print_table(rows, models):
    print("\nOpen-loop prediction error vs MuJoCo (RMS over the test windows)")
    head = f"{'data':5s} {'mode':4s} {'channel':9s} {'model':16s} {'windows':>7s}" + "".join(
        f" {'@' + format(s / 48, '.2f') + 's':>9s}" for s in TABLE_STEPS)
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda r: (r["source"], r["mode"], list(CHANNELS).index(r["channel"]),
                                         (list(models) + ["ARX vs ResNet"]).index(r["model"]))):
        print(f"{r['source']:5s} {r['mode']:<4d} {r['channel']:9s} {r['model']:16s} {r['windows']:7d}"
              + "".join(f" {r[f'rms@{s / 48:.2f}s']:9.2f}" for s in TABLE_STEPS)
              + f"  {r['unit']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--p_flights", type=int, default=1, help="P-controller test flights")
    parser.add_argument("--rl_flights", type=int, default=1, help="RL director test flights")
    parser.add_argument("--policy", default=DIRECTOR_POLICY, help="RL director policy")
    parser.add_argument("--arx", default=DEFAULT_MODEL_FILE, help="ARX model file")
    parser.add_argument("--resnet", default=MODEL_FILE, help="ResNet model file")
    parser.add_argument("--stride", type=int, default=4, help="steps between prediction windows")
    parser.add_argument("--reuse", action="store_true", help="reuse the cached test flights")
    parser.add_argument("--no_refit", action="store_true",
                        help="skip the ARX model re-fitted on the ResNet training data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/analysis_identification")
    parser.add_argument("--show", action="store_true", help="show the figures")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "test_flights.pkl")
    if args.reuse and os.path.exists(cache):
        with open(cache, "rb") as f:
            runs = pickle.load(f)
        print(f"Reusing {len(runs)} test flights from {cache}")
    else:
        runs = collect_test_flights(args.p_flights, args.rl_flights, args.policy, args.workers)
        with open(cache, "wb") as f:
            pickle.dump(runs, f)

    models = {"ARX": load_models(args.arx), "ResNet": load_model(args.resnet)}
    if not args.no_refit:
        refit = arx_on_resnet_data(os.path.join(args.out, "arx_same_data.npz"))
        if refit is not None:
            models = {"ARX": models["ARX"], "ARX (same data)": refit, "ResNet": models["ResNet"]}

    t0 = time.time()
    err = evaluate(runs, models, stride=args.stride)
    err_contact = evaluate(runs, models, stride=args.stride, kind="contact")
    print(f"Evaluated {sum(len(next(iter(d['ResNet'].values()))) for d in err.values())} "
          f"prediction windows in {time.time() - t0:.0f} s")

    rows = summarize(err, models)
    rows_contact = summarize(err_contact, models)
    print_table(rows, models)
    if rows_contact:
        print("\n(hook touching the payload before it is lifted: not modelled)")
        print_table(rows_contact, models)
    v = verdict(rows)
    print(f"\nResNet / ARX error ratio (geometric mean over sources, modes, channels and "
          f"look-aheads): {v['resnet_over_arx_error_ratio']:.2f}  "
          f"(ResNet more accurate in {v['resnet_wins']}/{v['comparisons']} comparisons)")
    for k, q in sorted(v["per_source_channel"].items()):
        print(f"   {k:14s} {q:.2f}")
    print(f"-> more accurate model: {v['more_accurate']}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"verdict": v, "free_flight": rows, "hook_contact": rows_contact,
                   "test_flights": [{"source": r["source"], "seed": int(r.get("seed", -1)),
                                     "steps": len(r["x"])} for r in runs]}, f, indent=1)
    import csv
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved:", os.path.join(args.out, "summary.json"), "and summary.csv")

    plot_error_vs_horizon(err, models, args.out)
    plot_summary_bars(rows, models, args.out)
    plot_contact(err, err_contact, models, args.out)
    for src in ("RL", "P"):
        cands = [r for r in runs if r["source"] == src]
        if cands:
            # the flight with the most payload-attached time shows both modes
            run = max(cands, key=lambda r: min(r["attached"].sum(), (~r["attached"]).sum()))
            plot_timeseries(run, models, args.out)
            plot_fans(run, models, args.out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
