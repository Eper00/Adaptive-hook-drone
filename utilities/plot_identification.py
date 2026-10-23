"""Prediction accuracy of the identified ResNet model, as figures for the thesis.

Uses the cached test flights of utilities/validate_resnet.py (P-controller
and RL director flights with seeds never used for training) and the model of
resnet_mpc.py. Every valid 1 s window (one mode, no hook-payload contact
before the lift) is rolled out open loop from the measured state with the
applied commands.

  id_prediction_error.png/.pdf
      RMS error of the position, velocity and orientation 0.25, 0.5 and 1 s
      ahead, without and with the payload, on the P-controller flights.
  id_prediction_error_rl.png/.pdf
      the same on the flights of the RL director (aggressive commands).
  id_prediction_error.csv   the numbers
(Dedicated test flights with fixed payload masses: utilities/test_resnet.py.)

Usage
-----
    python -m utilities.plot_identification
    python -m utilities.plot_identification --cache results/analysis/identification/validation/validation_flights.pkl
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

from utilities.validate_resnet import C_BASE, C_INK, C_INK2, DT, windows

HORIZONS = (12, 24, 48)                      # 0.25 / 0.5 / 1 s
HMAX = max(HORIZONS)
DEG = np.degrees(1.0)
ANGLES = (13, 14, 15)                        # roll_w, pitch_w, yaw (wrapped differences)
# ordinal ramp for the look-ahead (reference palette, blue steps 250 / 450 / 650)
RAMP = ("#86b6ef", "#2a78d6", "#104281")
C_WITH, C_WITHOUT = "#2a78d6", C_BASE
SOURCES = {"P": ("P-controller", "o"), "RL": ("RL director", "^")}
GROUPS = [("Position", "cm", 100, [(0, "$p_x$"), (1, "$p_y$"), (2, "$p_z$")]),
          ("Velocity", "cm/s", 100, [(3, "$v_x$"), (4, "$v_y$"), (5, "$v_z$")]),
          ("Orientation", "deg", DEG, [(13, r"$\tilde{\phi}$"), (14, r"$\tilde{\theta}$"),
                                       (15, r"$\psi$")])]


################################################################################
# DATA
################################################################################

def payload_masses(seeds):
    """Payload mass [kg] of every scenario seed (the env draws it on reset)."""
    from multi_drone_mujoco.envs.director_mpc import make_identification_env
    env = make_identification_env(0)
    out = {}
    for s in seeds:
        env.reset(seed=int(s))
        out[int(s)] = env.MASS
    env.close()
    return out


def rollout_batch(model, X0, U, D=None):
    """Open-loop predictions (n, HMAX+1, NX), the offset D (n, 3) kept constant."""
    from multi_drone_mujoco.envs.resnet_mpc import rollout
    out = []
    with torch.inference_mode():
        for i in range(0, len(X0), 2048):
            sl = slice(i, i + 2048)
            X = rollout(model, torch.as_tensor(X0[sl]), torch.as_tensor(U[sl]),
                        None if D is None else torch.as_tensor(D[sl])).numpy()
            out.append(np.concatenate([X0[sl, None], X], axis=1))
    return np.concatenate(out)


def collect_windows(runs, models, masses, stride):
    """Open-loop predictions (without / with the offset estimate) of every valid window."""
    data = {}
    for mode in (0, 1):
        X0, U, Y, src, mass, flight, t_att = [], [], [], [], [], [], []
        for f, r in enumerate(runs):
            att = r["attached"].astype(bool)
            k_att = int(np.argmax(att)) if att.any() else None
            for k in windows(r, mode, HMAX, stride):
                X0.append(r["x"][k])
                U.append(r["u"][k:k + HMAX])
                Y.append(r["x"][k:k + HMAX + 1])
                src.append(r["source"])
                mass.append(masses[int(r["seed"])])
                flight.append(f)
                t_att.append(np.nan if k_att is None else (k - k_att) * DT)
        X0, U, Y = (np.asarray(a, float) for a in (X0, U, Y))
        data[mode] = {"true": Y, "pred": rollout_batch(models[mode], X0, U),
                      "source": np.asarray(src), "mass": np.asarray(mass),
                      "flight": np.asarray(flight), "t_att": np.asarray(t_att)}
    return data


def errors(pred, true, i, scale):
    """Prediction errors of state i at every look-ahead (n, HMAX+1), in display units."""
    e = pred[:, :, i] - true[:, :, i]
    if i in ANGLES:
        e = (e + np.pi) % (2 * np.pi) - np.pi
    return e * scale


def rmse(e, mask=None):
    e = e if mask is None else e[mask]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


################################################################################
# FIGURES
################################################################################

def style(font):
    plt.rcParams.update({
        "font.size": font, "axes.titlesize": font + 1, "axes.labelsize": font,
        "xtick.labelsize": font - 2, "ytick.labelsize": font - 2, "legend.fontsize": font - 3,
        "axes.edgecolor": C_INK2, "axes.labelcolor": C_INK, "text.color": C_INK,
        "xtick.color": C_INK2, "ytick.color": C_INK2, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#e4e3df", "grid.linewidth": 1.0,
    })


def save(fig, out, name):
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        path = os.path.join(out, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", **kw)
        print("saved", path)
    plt.close(fig)


def plot_prediction_error(data, out, sources, name, title):
    """RMS error per state and look-ahead on the flights of ``sources``."""
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5))
    width = 0.26
    for r, mode in enumerate((0, 1)):
        d = data[mode]
        sel = np.isin(d["source"], sources)
        for c, (group, unit, scale, states) in enumerate(GROUPS):
            ax = axes[r, c]
            for j, (i, label) in enumerate(states):
                e = errors(d["pred"], d["true"], i, scale)
                for h_i, h in enumerate(HORIZONS):
                    v = rmse(e[:, h], sel)
                    ax.bar(j + (h_i - 1) * width, v, width * 0.9, color=RAMP[h_i], zorder=2)
                    row = {"flights": "+".join(sources), "mode": "payload" if mode else "no payload",
                           "state": label.strip("$").replace("\\", ""), "unit": unit,
                           "look_ahead_s": h * DT, "rmse": round(v, 3), "windows": int(sel.sum())}
                    for src in sources:
                        row[f"rmse_{src}"] = round(rmse(e[:, h], d["source"] == src), 3)
                    rows.append(row)
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels([lab for _, lab in states])
            ax.grid(axis="y", zorder=0)
            ax.set_ylabel(f"RMS error [{unit}]")
            if r == 0:
                ax.set_title(group)
        axes[r, 0].annotate(("No payload" if mode == 0 else "With payload") + f"\n({int(sel.sum())} windows)",
                            xy=(-0.36, 0.5), xycoords="axes fraction", rotation=90, va="center",
                            ha="center", fontsize=plt.rcParams["axes.titlesize"])
    fig.suptitle(title, y=1.0)
    fig.tight_layout(h_pad=2.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RAMP[i]) for i in range(len(HORIZONS))]
    fig.legend(handles, [f"{h * DT:g} s ahead" for h in HORIZONS], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    save(fig, out, name)
    return rows


def write_csv(rows, path):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="results/analysis/identification/validation/validation_flights.pkl",
                   help="test flights of utilities/validate_resnet.py")
    p.add_argument("--resnet", default=None, help="ResNet model file")
    p.add_argument("--out", default="results/analysis/identification/validation")
    p.add_argument("--stride", type=int, default=4, help="window stride [steps]")
    p.add_argument("--font", type=float, default=22)
    args = p.parse_args()

    from multi_drone_mujoco.envs.resnet_mpc import MODEL_FILE, load_model
    torch.set_num_threads(4)
    models = load_model(args.resnet or MODEL_FILE)
    with open(args.cache, "rb") as f:
        runs = pickle.load(f)
    if isinstance(runs, tuple):         # older caches: (test runs, MPC runs)
        runs = runs[0]
    masses = payload_masses({int(r["seed"]) for r in runs})
    print(f"{len(runs)} flights, payload masses {min(masses.values()):.3f}-{max(masses.values()):.3f} kg")

    data = collect_windows(runs, models, masses, args.stride)
    for mode in (0, 1):
        print(f"mode {mode}: {len(data[mode]['true'])} windows")
    os.makedirs(args.out, exist_ok=True)
    style(args.font)
    rows = plot_prediction_error(data, args.out, ["P"], "id_prediction_error",
                                 "Open-loop prediction error: P-controller flights")
    rows += plot_prediction_error(data, args.out, ["RL"], "id_prediction_error_rl",
                                  "Open-loop prediction error: RL director flights")
    write_csv(rows, os.path.join(args.out, "id_prediction_error.csv"))


if __name__ == "__main__":
    main()
