"""Training curve of the ResNet models (thesis figure).

``resnet_mpc.train`` keeps no history, so the training is run again with the
same seed, data and settings, logging the rollout loss over the full horizon
on the training and validation windows after every epoch. The trained
weights are saved only with --save.

Usage:
    python -m utilities.resnet_training_curve            # train (~10 min) + plot
    python -m utilities.resnet_training_curve --replot   # plot from the log
"""

import argparse
import json
import pickle
import time

import numpy as np

LOG_FILE = "results/resnet/resnet_training_log.json"
OUT_FILE = "results/thesis_figures/resnet_training.png"


def train_with_log(log_path, epochs=150, horizon=24, threads=8, data_file=None, save_path=None):
    """Same loop as resnet_mpc.train, plus a per-epoch log."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import multi_drone_mujoco.envs.resnet_mpc as R

    torch.manual_seed(0)
    torch.set_num_threads(threads)
    with open(data_file or R.DATA_FILE, "rb") as f:
        train_runs, val_runs = pickle.load(f)
    env = R.make_identification_env(0)
    dt, alpha = env.CTRL_TIMESTEP, env.alpha
    env.close()

    def windows(runs, mode):
        idx = R._windows(runs, mode, horizon)
        x0 = np.stack([runs[i]["x"][k] for i, k in idx])
        u = np.stack([runs[i]["u"][k:k + horizon] for i, k in idx])
        y = np.stack([runs[i]["x"][k + 1:k + horizon + 1] for i, k in idx])
        return TensorDataset(*(torch.tensor(a, dtype=torch.float64) for a in (x0, u, y)))

    import copy
    log, models = {}, []
    for mode in (0, 1):
        data, val = windows(train_runs, mode), windows(val_runs, mode)
        x0, u, y = data.tensors
        model = R.ResNetDynamics(dt, alpha, with_swing=bool(mode))
        z = torch.cat([x0[:, 3:], u[:, 0]], dim=1)
        model.mu.copy_(z.mean(0))
        model.sd.copy_(torch.where(z.std(0) < 1e-4, 1.0, z.std(0)))
        inc = y[:, 0] - x0
        model.out_scale.copy_(torch.cat([inc[:, R.V].std(0),
                                         inc[:, R.THD].std(0) if mode else torch.ones(2, dtype=torch.float64),
                                         inc[:, R.ATT].std(0), inc[:, R.ATTD].std(0)]) + 1e-6)
        idx = lambda sl: list(range(sl.start, sl.stop))
        ch = idx(R.V) + idx(R.ATT) + idx(R.ATTD) + (idx(R.TH) + idx(R.THD) if mode else [])
        w = 1.0 / (y.reshape(-1, R.NX)[:, ch].std(0) + 1e-3) ** 2

        def loss_fn(x0_, u_, y_, H):
            pred = R.rollout(model, x0_, u_[:, :H])
            return ((pred[..., ch] - y_[:, :H, ch]) ** 2 * w).mean()

        loader = DataLoader(data, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        # training loss over the full horizon on a subset of the windows
        sub = torch.arange(0, len(data), max(1, len(data) // 20000))
        rec = {k: [] for k in ("epoch", "H", "train", "val_H", "val_full", "train_full")}
        best, best_state = np.inf, None
        t0 = time.time()
        for ep in range(epochs):
            H = max(2, min(horizon, int(horizon * (ep + 1) / (0.4 * epochs))))
            tl, n = 0.0, 0
            for x0_, u_, y_ in loader:
                opt.zero_grad()
                loss = loss_fn(x0_, u_, y_, H)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl, n = tl + loss.item() * len(x0_), n + len(x0_)
            sched.step()
            with torch.no_grad():
                vh = loss_fn(*val.tensors, H).item()
                vf = loss_fn(*val.tensors, horizon).item()
                tf = loss_fn(x0[sub], u[sub], y[sub], horizon).item()
            if H == horizon and vf < best:          # kept weights, as in resnet_mpc.train
                best, best_state = vf, copy.deepcopy(model.state_dict())
            for k, v in zip(rec, (ep, H, tl / n, vh, vf, tf)):
                rec[k].append(v)
            print(f"mode {mode} epoch {ep:3d}  H={H:2d}  train {tl / n:.4f}  val {vf:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        log[str(mode)] = rec
        model.load_state_dict(best_state)
        models.append(model)
    with open(log_path, "w") as f:
        json.dump(log, f)
    if save_path:
        R.save_model(save_path, models)
        print("saved model:", save_path)
    print("saved:", log_path)
    return log


def plot(log, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    f = 22
    plt.rcParams.update({"font.size": f, "axes.labelsize": f, "xtick.labelsize": f - 2,
                         "ytick.labelsize": f - 2, "legend.fontsize": f - 3,
                         "lines.linewidth": 2.5, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.3, 1]))
    col = {"0": "tab:blue", "1": "tab:green"}
    name = {"0": "without payload ($\\kappa=0$)", "1": "with payload ($\\kappa=1$)"}
    for m in ("0", "1"):
        r = log[m]
        ep = np.array(r["epoch"]) + 1
        ax.plot(ep, r["train_full"], color=col[m], ls="-")
        ax.plot(ep, r["val_full"], color=col[m], ls="--")
        # kept weights: best validation loss once the full horizon is reached
        H = np.array(r["H"])
        b = int(np.argmin(np.where(H == H.max(), r["val_full"], np.inf)))
        ax.plot(ep[b], r["val_full"][b], "o", color=col[m], ms=10, mec="k")
        if m == "0":                      # same horizon schedule for both modes
            axh.step(ep, H, where="post", color="k")
    ax.set_yscale("log")
    ax.set_ylabel("Rollout loss\n(H = 24 steps)")
    axh.set_ylabel("Rollout\nhorizon H")
    axh.set_xlabel("Epoch")
    axh.set_xlim(0, len(log["0"]["epoch"]))
    axh.set_yticks([2, 12, 24])
    handles = [Line2D([], [], color=col[m], label=name[m]) for m in ("0", "1")] + [
        Line2D([], [], color="gray", ls="-", label="training loss"),
        Line2D([], [], color="gray", ls="--", label="validation loss"),
        Line2D([], [], color="gray", marker="o", mec="k", ls="", ms=10,
               label="kept weights (best validation)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--out", default=OUT_FILE)
    p.add_argument("--replot", action="store_true", help="plot from the existing log")
    p.add_argument("--data", default=None, help="training data (default: resnet_mpc.DATA_FILE)")
    p.add_argument("--save", default=None, help="also save the trained model to this file")
    a = p.parse_args()
    if a.replot:
        with open(a.log) as f:
            log = json.load(f)
    else:
        log = train_with_log(a.log, data_file=a.data, save_path=a.save)
    plot(log, a.out)


if __name__ == "__main__":
    main()
