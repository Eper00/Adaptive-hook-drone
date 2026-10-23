"""Identification validation of the ResNet director model (resnet_mpc.py).

How well does x_k+1 = x_k + f(x_k, u_k) predict every state of the MuJoCo
closed loop (command -> low-pass filter -> PPO velocity controller -> drone
(+ payload)) 0.5, 0.75 and 1.0 s ahead?

Test data (seeds never used for training, see resnet_mpc.P_TRAIN / RL_TRAIN):

  P     P-controller flights with random command excitation
  RL    flights of the RL director policy (aggressive commands)

The model is evaluated open loop (simulation error): it is rolled out from
the measured x_k with the commands u_k..u_k+H-1 that were actually applied,
no feedback, no disturbance estimate. (Dedicated test flights with fixed
payload masses: utilities/test_resnet.py.)

Every state is scored at every look-ahead (RMSE, MAE, bias, 95th percentile,
max, NRMSE / FIT on the state increment, skill vs a constant-rate baseline).

Outputs (``--out``, default results/analysis/identification/validation):
  metrics_open_loop.csv     all numbers
  tables.md                 the main tables
  summary.json              numbers + test flight list
  ol_*.png                  figures

Usage:
    python -m utilities.validate_resnet                    # fly + evaluate (~10 min)
    python -m utilities.validate_resnet --reuse            # re-evaluate cached flights
    python -m utilities.validate_resnet --p_flights 6 --rl_flights 12 --mpc_seeds 6
"""

import argparse
import csv
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DT = 1.0 / 48.0
HORIZONS = (24, 36, 48)                 # 0.5 / 0.75 / 1.0 s
HMAX = max(HORIZONS)
DEG = np.degrees(1.0)
# (state index, name, display scale, unit, payload-only)
# (indices: director_mpc.P, V, C, TH, THD, ATT, ATTD)
STATES = [
    (0, "x", 100, "cm", False), (1, "y", 100, "cm", False), (2, "z", 100, "cm", False),
    (3, "v_x", 100, "cm/s", False), (4, "v_y", 100, "cm/s", False), (5, "v_z", 100, "cm/s", False),
    (13, "roll", DEG, "deg", False), (14, "pitch", DEG, "deg", False), (15, "yaw", DEG, "deg", False),
    (16, "roll_rate", DEG, "deg/s", False), (17, "pitch_rate", DEG, "deg/s", False),
    (18, "yaw_rate", DEG, "deg/s", False),
    (9, "swing_x", DEG, "deg", True), (10, "swing_y", DEG, "deg", True),
    (11, "swing_rate_x", DEG, "deg/s", True), (12, "swing_rate_y", DEG, "deg/s", True),
]
# the remaining entries of the 19-dim state are the known command filter c
EXTRA_STATES = [(6, "c_x", 100, "cm/s", False), (7, "c_y", 100, "cm/s", False),
                (8, "c_z", 100, "cm/s", False)]
SWING_RATES = (11, 12)
P_SEEDS0, P_EXCITE = 500, (0.15, 0.4, 0.6)
RL_SEEDS0 = 600

# reference palette (dataviz skill), categorical slots in fixed order
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_INK, C_INK2, C_BASE = "#0b0b0b", "#52514e", "#8a8985"
SRC_COLOR = {"P": C_BLUE, "RL": C_ORANGE}
H_COLOR = {24: C_BLUE, 36: C_ORANGE, 48: C_AQUA}


################################################################################
# DATA
################################################################################

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


def _pool(fn, jobs, workers):
    os.environ.setdefault("OMP_NUM_THREADS", "1")      # inherited by the spawned workers
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = [ex.submit(fn, a) for a in jobs]
        return [f.result() for f in futures]


def collect_all(n_p, n_rl, workers):
    from multi_drone_mujoco.envs.resnet_mpc import (DIRECTOR_POLICY, _collect_p, _collect_rl,
                                                    collect_runs)
    jobs = [(_collect_p, (P_SEEDS0 + i, P_EXCITE[i % len(P_EXCITE)])) for i in range(n_p)]
    if os.path.exists(DIRECTOR_POLICY):
        jobs += [(_collect_rl, (RL_SEEDS0 + i, DIRECTOR_POLICY, 15.0)) for i in range(n_rl)]
    print(f"Flying {len(jobs)} P-controller / RL-director test flights...", flush=True)
    t0 = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    runs = collect_runs(jobs, workers)
    print(f"  done in {time.time() - t0:.0f} s", flush=True)
    return runs


################################################################################
# EVALUATION
################################################################################

def batch_rollout(model, X0, U):
    """Open-loop rollouts of many windows at once: X0 (n, NX), U (n, H, NU)
    -> (n, H+1, NX)."""
    import torch
    from multi_drone_mujoco.envs.resnet_mpc import rollout
    out = []
    with torch.inference_mode():
        for i in range(0, len(X0), 2048):
            x0 = torch.as_tensor(X0[i:i + 2048])
            X = rollout(model, x0, torch.as_tensor(U[i:i + 2048])).numpy()
            out.append(np.concatenate([X0[i:i + 2048, None], X], axis=1))
    return np.concatenate(out) if out else np.zeros((0, HMAX + 1, X0.shape[-1]))


def baseline(X0, H=HMAX):
    """Constant-rate extrapolation: position with the current velocity,
    swing angle with the current swing rate, attitude with the current
    attitude rate, every other state held."""
    h = np.arange(H + 1)[None, :, None] * DT
    B = np.repeat(X0[:, None], H + 1, axis=1)
    B[:, :, 0:3] += h * X0[:, None, 3:6]
    B[:, :, 9:11] += h * X0[:, None, 11:13]
    B[:, :, 13:16] += h * X0[:, None, 16:19]
    return B


def open_loop_windows(runs, models, mode, stride):
    """(pred, true, base) arrays (n, HMAX+1, NX) of every valid HMAX window."""
    X0, U, Y = [], [], []
    for r in runs:
        for k in windows(r, mode, HMAX, stride):
            X0.append(r["x"][k])
            U.append(r["u"][k:k + HMAX])
            Y.append(r["x"][k:k + HMAX + 1])
    if not X0:
        return None
    X0, U, Y = (np.asarray(a, float) for a in (X0, U, Y))
    return {"pred": batch_rollout(models[mode], X0, U), "true": Y, "base": baseline(X0)}


def score(pred, true, base, i, scale):
    """Error statistics of state i at every look-ahead: dict of (HMAX+1,) arrays."""
    e = (pred[:, :, i] - true[:, :, i]) * scale
    eb = (base[:, :, i] - true[:, :, i]) * scale
    d = (true[:, :, i] - true[:, :1, i]) * scale           # true increment from x_k
    ok = np.isfinite(e)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(np.nanmean(e ** 2, 0))
        rmse_b = np.sqrt(np.nanmean(np.where(ok, eb, np.nan) ** 2, 0))
        dd = np.where(ok, d, np.nan)
        spread = np.sqrt(np.nanmean((dd - np.nanmean(dd, 0)) ** 2, 0))
        out = {"n": n, "rmse": rmse, "mae": np.nanmean(np.abs(e), 0), "bias": np.nanmean(e, 0),
               "p95": np.nanpercentile(np.abs(e), 95, axis=0) if ok.any() else rmse * np.nan,
               "max": np.nanmax(np.abs(e), 0) if ok.any() else rmse * np.nan,
               "rmse_baseline": rmse_b,
               "nrmse": 100 * rmse / spread,                # % of the spread of the true change
               "fit": 100 * (1 - rmse / spread),            # FIT [%] on the increment
               "skill": 1 - rmse / rmse_b}                  # 1: perfect, 0: as good as baseline
    return out


def state_list(mode, extra=False):
    return [s for s in STATES + (EXTRA_STATES if extra else []) if mode == 1 or not s[4]]


def table_rows(scores, meta):
    """Flat rows (one per state and horizon) for the CSV files."""
    rows = []
    for (i, name, _, unit, _), sc in scores.items():
        for h in HORIZONS:
            if sc["n"][h] == 0:
                continue
            rows.append({**meta, "state": name, "unit": unit, "horizon_s": round(h * DT, 3),
                         "windows": int(sc["n"][h]),
                         **{k: round(float(sc[k][h]), 4) for k in
                            ("rmse", "mae", "bias", "p95", "max", "rmse_baseline", "nrmse",
                             "fit", "skill")},
                         "rmse_1step": round(float(sc["rmse"][1]), 4)})
    return rows


def evaluate(test_runs, models, stride):
    ol = {}
    for src in ("P", "RL"):
        runs = [r for r in test_runs if r["source"] == src]
        for mode in (0, 1):
            w = open_loop_windows(runs, models, mode, stride)
            if w is None:
                continue
            ol[(src, mode)] = {s: score(w["pred"], w["true"], w["base"], s[0], s[2])
                               for s in state_list(mode)}
            ol[(src, mode)]["_raw"] = w
    return ol


################################################################################
# PLOTS
################################################################################

def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": C_INK2, "axes.labelcolor": C_INK2,
        "xtick.color": C_INK2, "ytick.color": C_INK2, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.6, "lines.linewidth": 1.8, "figure.facecolor": "white",
        "axes.titlecolor": C_INK, "legend.frameon": False})


def _grid(n, ncols, size):
    """Subplot grid for n panels (``size``: figure width, height per row);
    the unused panels are switched off."""
    import matplotlib.pyplot as plt
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0], size[1] * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes


def _horizon_marks(ax):
    for h in HORIZONS:
        ax.axvline(h * DT, color=C_BASE, lw=0.7, ls=":", zorder=0)


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("saved:", path)


def plot_ol_growth(ol, out):
    import matplotlib.pyplot as plt
    t = np.arange(HMAX + 1) * DT
    fig, axes = _grid(len(STATES), 4, (15, 3.2))
    for ax, s in zip(axes.ravel(), STATES):
        top = 0.0
        for (src, mode), d in sorted(ol.items()):
            if s not in d:
                continue
            top = max(top, np.nanmax(d[s]["rmse"]))
            ls = "-" if mode == 0 else "--"
            ax.plot(t, d[s]["rmse"], color=SRC_COLOR[src], ls=ls,
                    label=f"{src}, {'no payload' if mode == 0 else 'payload'}")
        # constant-rate baseline (pooled over the RL windows: the hardest data)
        for mode in (0, 1):
            if ("RL", mode) in ol and s in ol[("RL", mode)]:
                ax.plot(t, ol[("RL", mode)][s]["rmse_baseline"], color=C_BASE, lw=1,
                        ls="-" if mode == 0 else "--",
                        label=f"constant-rate baseline, RL, {'no payload' if mode == 0 else 'payload'}")
        _horizon_marks(ax)
        ax.set_title(f"{s[1]} [{s[3]}]")
        ax.set_xlabel("look-ahead [s]")
        ax.set_ylabel("RMSE")
        ax.set_ylim(0, 1.35 * top)          # the baseline leaves the axes: see the skill scores
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Open-loop prediction error of every state vs look-ahead "
                 "(model driven by the recorded commands; gray: constant-rate baseline, "
                 "clipped)", color=C_INK)
    fig.tight_layout()
    _save(fig, out, "ol_error_growth.png")
    plt.close(fig)


def plot_ol_heatmap(ol, out, metric="nrmse"):
    """States x (data, mode, horizon): color = NRMSE (% of the true change),
    text = RMSE in display units."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cols = [(src, mode, h) for src in ("P", "RL") for mode in (0, 1) for h in HORIZONS
            if (src, mode) in ol]
    M = np.full((len(STATES), len(cols)), np.nan)
    R = np.full_like(M, np.nan)
    for j, (src, mode, h) in enumerate(cols):
        for i, s in enumerate(STATES):
            if s in ol[(src, mode)]:
                M[i, j] = ol[(src, mode)][s][metric][h]
                R[i, j] = ol[(src, mode)][s]["rmse"][h]
    cmap = LinearSegmentedColormap.from_list("seq", ["#f1f6fd", "#9cc3f0", C_BLUE, "#123b6e"])
    cmap.set_bad("#efefed")
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 6.2))
    vmax = 100.0
    im = ax.imshow(np.clip(M, 0, vmax), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(R[i, j]):
                dark = np.clip(M[i, j], 0, vmax) > 55
                ax.text(j, i, f"{R[i, j]:.1f}\n{M[i, j]:.0f}%", ha="center", va="center",
                        fontsize=6.5, color="white" if dark else C_INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{h * DT:.2f}s" for _, _, h in cols], fontsize=7)
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels([f"{s[1]} [{s[3]}]" for s in STATES])
    ax.grid(False)
    for j in range(0, len(cols), len(HORIZONS)):
        src, mode, _ = cols[j]
        ax.text(j + 1, -0.9, f"{src}\n{'no payload' if mode == 0 else 'payload'}",
                ha="center", va="bottom", fontsize=8, color=C_INK)
        if j:
            ax.axvline(j - 0.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("NRMSE [% of the spread of the true change over the look-ahead]")
    fig.suptitle("Open-loop error of every state (cell: RMSE in the state's unit / NRMSE)",
                 color=C_INK, y=1.04)
    fig.tight_layout()
    _save(fig, out, "ol_heatmap.png")
    plt.close(fig)


def plot_ol_parity(ol, out, h=HMAX):
    """Predicted vs true change of every state over ``h`` steps."""
    import matplotlib.pyplot as plt
    fig, axes = _grid(len(STATES), 4, (14, 3.3))
    for ax, s in zip(axes.ravel(), STATES):
        i, name, sc, unit, _ = s
        lim = 0.0
        for (src, mode), d in sorted(ol.items()):
            if s not in d:
                continue
            w = d["_raw"]
            dt_true = (w["true"][:, h, i] - w["true"][:, 0, i]) * sc
            dt_pred = (w["pred"][:, h, i] - w["true"][:, 0, i]) * sc
            ax.scatter(dt_true, dt_pred, s=6, alpha=0.35, color=SRC_COLOR[src], lw=0,
                       label=src if mode == 0 or s[4] else None)
            lim = max(lim, np.percentile(np.abs(np.r_[dt_true, dt_pred]), 99.5))
        ax.plot([-lim, lim], [-lim, lim], color=C_INK2, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"Δ{name} over {h * DT:.2f} s [{unit}]")
        ax.set_xlabel("MuJoCo")
        ax.set_ylabel("ResNet")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, markerscale=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Open loop: predicted vs true change of every state over {h * DT:.2f} s",
                 color=C_INK)
    fig.tight_layout()
    _save(fig, out, f"ol_parity_{h * DT:.2f}s.png")
    plt.close(fig)


def plot_ol_box(ol, out):
    """Absolute error distribution of every state at the three horizons
    (all data sources pooled)."""
    import matplotlib.pyplot as plt
    fig, axes = _grid(len(STATES), 4, (14, 2.85))
    for ax, s in zip(axes.ravel(), STATES):
        i, name, sc, unit, _ = s
        data = []
        for h in HORIZONS:
            e = [np.abs(d["_raw"]["pred"][:, h, i] - d["_raw"]["true"][:, h, i]) * sc
                 for d in ol.values() if s in d]
            data.append(np.concatenate(e))
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.5, patch_artist=True,
                        medianprops={"color": C_INK})
        for patch, h in zip(bp["boxes"], HORIZONS):
            patch.set_facecolor(H_COLOR[h])
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(HORIZONS) + 1))
        ax.set_xticklabels([f"{h * DT:.2f} s" for h in HORIZONS])
        ax.set_title(f"|error| {name} [{unit}]")
        ax.set_ylim(bottom=0)
    fig.suptitle("Open loop: absolute error distribution (box: quartiles, whiskers: 5-95 %)",
                 color=C_INK)
    fig.tight_layout()
    _save(fig, out, "ol_error_distribution.png")
    plt.close(fig)


def _shade(ax, t, run):
    att = run["attached"].astype(bool)
    bad = ~run["valid"].astype(bool)
    for mask, color in ((att, C_AQUA), (bad, "#e34948")):
        edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=0.08, lw=0)


def plot_ol_timeseries(run, models, out):
    """MuJoCo vs the 0.5 / 0.75 / 1.0 s-ahead open-loop predictions of every state."""
    import matplotlib.pyplot as plt
    x = run["x"]
    T = len(x)
    t = np.arange(T) * DT
    preds = {}
    for h in HORIZONS:
        P = np.full((T, x.shape[1]), np.nan)
        for mode in (0, 1):
            ks = windows(run, mode, h, 1)
            if not ks:
                continue
            X = batch_rollout(models[mode], x[ks], np.stack([run["u"][k:k + h] for k in ks]))
            P[np.asarray(ks) + h] = X[:, h]
        preds[h] = P
    fig, axes = plt.subplots(len(STATES), 1, figsize=(13, 1.55 * len(STATES)), sharex=True)
    for ax, (i, name, sc, unit, _) in zip(axes, STATES):
        _shade(ax, t, run)
        ax.plot(t, x[:, i] * sc, color=C_INK, lw=2.2, label="MuJoCo")
        for h in HORIZONS:
            ax.plot(t, preds[h][:, i] * sc, color=H_COLOR[h], lw=1.1,
                    label=f"ResNet, predicted {h * DT:.2f} s earlier")
        ax.set_ylabel(f"{name}\n[{unit}]", fontsize=8)
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
    axes[-1].set_xlabel("time [s]   (aqua band: payload attached; red band: hook-payload "
                        "contact, not modelled)")
    fig.suptitle(f"Open loop, {run['source']} flight (seed {run.get('seed')}): every state vs "
                 "its k-step-ahead prediction", color=C_INK, y=1.0)
    fig.tight_layout()
    _save(fig, out, f"ol_timeseries_{run['source']}_{run.get('seed')}.png")
    plt.close(fig)


def _pooled(d_list, s, h, key="rmse"):
    num, den = 0.0, 0
    for d in d_list:
        if d is not None and s in d and d[s]["n"][h]:
            num += d[s][key][h] ** 2 * d[s]["n"][h]
            den += d[s]["n"][h]
    return np.sqrt(num / den) if den else np.nan


def markdown_tables(ol):
    L = []
    hdr = " | ".join(f"{h * DT:.2f} s" for h in HORIZONS)
    for src in ("P", "RL"):
        for mode in (0, 1):
            d = ol.get((src, mode))
            if d is None:
                continue
            n = d[STATES[0]]["n"][HMAX]
            L += [f"\n### Open loop - {src} flights, {'no payload' if mode == 0 else 'payload'} "
                  f"({n} windows)\n",
                  f"| state | unit | RMSE {hdr} | FIT % {hdr} | skill {hdr} | bias @1 s | P95 @1 s |",
                  "|---|---|" + "---|" * (3 * len(HORIZONS) + 2)]
            for s in state_list(mode):
                sc = d[s]
                L.append(f"| {s[1]} | {s[3]} | "
                         + " | ".join(f"{sc['rmse'][h]:.2f}" for h in HORIZONS) + " | "
                         + " | ".join(f"{sc['fit'][h]:.0f}" for h in HORIZONS) + " | "
                         + " | ".join(f"{sc['skill'][h]:.2f}" for h in HORIZONS)
                         + f" | {sc['bias'][HMAX]:+.2f} | {sc['p95'][HMAX]:.2f} |")
    return "\n".join(L) + "\n"


################################################################################
# MAIN
################################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--p_flights", type=int, default=6)
    parser.add_argument("--rl_flights", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4, help="open-loop window stride")
    parser.add_argument("--resnet", default=None, help="ResNet model file")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reuse", action="store_true", help="reuse the cached flights")
    parser.add_argument("--out", default="results/analysis/identification/validation")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from multi_drone_mujoco.envs.resnet_mpc import MODEL_FILE, load_model
    model_file = args.resnet or MODEL_FILE
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "validation_flights.pkl")
    if args.reuse and os.path.exists(cache):
        with open(cache, "rb") as f:
            test_runs = pickle.load(f)
        if isinstance(test_runs, tuple):        # older caches: (test runs, MPC runs)
            test_runs = test_runs[0]
        print(f"Reusing {len(test_runs)} test flights from {cache}")
    else:
        test_runs = collect_all(args.p_flights, args.rl_flights, args.workers)
        with open(cache, "wb") as f:
            pickle.dump(test_runs, f)

    models = load_model(model_file)
    t0 = time.time()
    ol = evaluate(test_runs, models, args.stride)
    print(f"Evaluated in {time.time() - t0:.0f} s")

    ol_rows = [r for (src, mode), d in ol.items()
               for r in table_rows({s: v for s, v in d.items() if s != "_raw"},
                                   {"data": src, "mode": mode, "method": "open loop"})]
    with open(os.path.join(args.out, "metrics_open_loop.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ol_rows[0]))
        w.writeheader()
        w.writerows(ol_rows)
    tables = markdown_tables(ol)
    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write(tables)
    print(tables)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"model": model_file, "horizons_s": [h * DT for h in HORIZONS],
                   "open_loop": ol_rows,
                   "test_flights": [{"source": r["source"], "seed": int(r.get("seed", -1)),
                                     "steps": len(r["x"]),
                                     "payload_steps": int(r["attached"].sum())}
                                    for r in test_runs]}, f, indent=1)

    _style()
    plot_ol_growth(ol, args.out)
    plot_ol_heatmap(ol, args.out)
    for h in HORIZONS:
        plot_ol_parity(ol, args.out, h)
    plot_ol_box(ol, args.out)
    for src in ("RL", "P"):
        cands = [r for r in test_runs if r["source"] == src]
        if cands:
            run = max(cands, key=lambda r: min(r["attached"].sum(), (~r["attached"]).sum()))
            plot_ol_timeseries(run, models, args.out)


if __name__ == "__main__":
    main()
