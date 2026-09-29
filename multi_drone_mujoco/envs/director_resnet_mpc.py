"""Director MPC with a ResNet-identified prediction model.

Same task, interface and MPC formulation as ``director_mpc.py``, but the
prediction model is a residual network in increment ("ResNet") form

    x_k+1 = x_k + f(x_k, u_k)

with no linear (ARX) model underneath. f is structured: whatever is
bookkeeping or known exactly is written out, everything the closed loop
(PPO velocity controller + drone + hook + payload) does is learned:

    dv          = NN_v(z_k)  (+ d)            learned velocity increment     (3)
    dtheta_dot  = NN_th(z_k)                  learned swing-rate increment   (2, mode 1)
    dtilt       = NN_t(z_k)                   learned roll/pitch increment   (2)
    dp          = dt (v + dv / 2)             trapezoidal position update
    dtheta      = dt (theta_dot + dtheta_dot)
    dc          = alpha (u - c)               the env's command low-pass filter (exact)
    dv_prev = v - v_prev,  dc_prev = c - c_prev,  dtilt_prev = tilt - tilt_prev
                                              (the history entries shift)

    z_k = [v, v_prev, c, c_prev, theta, theta_dot, tilt, tilt_prev](x_k), u_k   (23)

The position is not an input (the dynamics do not depend on where the drone
is), and d is the online velocity-offset estimate of the MPC.

    NN = input layer (tanh) -> residual blocks h <- h + W2 tanh(W1 h + b1) + b2
         -> linear output, plus a linear shortcut from the (normalized) input
         to the output

One network per mode (without / with payload). The output layer and the
shortcut start at zero (x_k+1 = x_k: constant velocity and attitude) and the
network is trained with a multi-step rollout loss whose length grows from 2
to 24 steps (0.5 s), because the MPC and the safety filter rely on
multi-step predictions. The training data are P-controller identification
runs, strongly excited runs and flights of the RL director policy.

The model is nonlinear, so the MPC linearizes it along the previous plan each
step (Gauss-Newton, single shooting, one parallel CasADi call), condenses the
dynamics onto the commands and solves a dense QP (qpOASES) with a
backtracking line search on the nonlinear cost. The network is exported to
CasADi, so the linearization uses exact derivatives.

Train / retrain (after a new velocity policy or physics changes):
    python -m multi_drone_mujoco.envs.director_resnet_mpc               # collect data + train
    python -m multi_drone_mujoco.envs.director_resnet_mpc --reuse_data  # train on the cached runs
Use:
    DirectorResNetMPCAgent(env)   (play.py: --env_type adaptive_director_resnet_MPC)
Compare with the ARX model:
    python -m utilities.analyse_identification
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import casadi as ca
import numpy as np

from multi_drone_mujoco.envs.director_mpc import (
    ATT, DEFAULT_MODEL_FILE, NU, NX, P, TENDON_HOLD, TH, THD, U_MAX, V,
    DirectorWeights, StateTracker, collect, director_step, load_models,
    make_identification_env, quiet_qp,
)
from multi_drone_mujoco.envs.mpc_mission import CONTACT_PHASES, MissionAgent, payload_attached

RESNET_FILE = "results/mpc_director/resnet_model.npz"
DATA_FILE = "results/mpc_director/resnet_training_data.pkl"   # cached training runs
DIRECTOR_POLICY = "results/final/rl_adaptive_director_curriculum/final_model.zip"
MODEL_FORM = "increment"                       # x_k+1 = x_k + f(x_k, u_k)

# input features: every state except the position, plus the command
FEATURE_IDX = np.r_[np.arange(3, NX)]          # 20 states
N_IN = len(FEATURE_IDX) + NU                   # 23
N_OUT = 7                                      # dv (3), dtheta_dot (2), dtilt (2)
HIDDEN = 128
BLOCKS = 2

# training / validation runs (seeds differ from the ones analyse_identification tests on)
P_TRAIN = [(s, 0.15) for s in (100, 101, 102, 103)] + [(s, 0.6) for s in (104, 105, 106, 107)]
P_VAL = [(108, 0.4)]
RL_TRAIN = tuple(range(300, 324))
RL_VAL = (400, 401, 402, 403)
RL_DURATION = 15.0


################################################################################
# MODEL (numpy / CasADi evaluation of the trained network)
################################################################################

def _mlp(z, net, tanh, matmul):
    """Evaluate the network for numpy or CasADi inputs (7 raw increments)."""
    zn = (z - net["mu"]) / net["sd"]
    h = tanh(matmul(net["W_in"], zn) + net["b_in"])
    for W1, b1, W2, b2 in net["blocks"]:
        h = h + matmul(W2, tanh(matmul(W1, h) + b1)) + b2
    return (matmul(net["W_out"], h) + net["b_out"] + matmul(net["W_skip"], zn)) * net["out_scale"]


def _next_state(x, u, r, dt, alpha, with_swing, vcat):
    """x_k+1 = x_k + f(x_k, u_k) from the network output r (numpy or CasADi).

    Written as the assembled next state; ``vcat`` stacks the pieces.
    """
    v, c = x[3:6], x[9:12]
    th, thd, att = x[15:17], x[17:19], x[19:21]
    dv = r[0:3]
    if with_swing:
        thd1 = thd + r[3:5]
        th1 = th + dt * thd1
    else:                                      # no payload: swing states stay 0
        thd1, th1 = thd, th
    return vcat([x[0:3] + dt * (v + 0.5 * dv),     # p
                 v + dv,                           # v
                 v,                                # v_prev
                 c + alpha * (u - c),              # c   (env low-pass filter)
                 c,                                # c_prev
                 th1, thd1,                        # theta, theta_dot
                 att + r[5:7],                     # tilt
                 att])                             # tilt_prev


class ResNetModel:
    """x_k+1 = x_k + f(x_k, u_k) with a ResNet f, for one mode."""

    def __init__(self, net, dt, alpha, with_swing):
        self.net = net                    # dict of weights (numpy)
        self.dt = dt
        self.alpha = alpha
        self.with_swing = with_swing

    def network(self, x, u):
        """Raw network output [dv, dtheta_dot, dtilt] (7)."""
        z = np.r_[x[FEATURE_IDX], u]
        return _mlp(z, self.net, np.tanh, lambda W, v: W @ v)

    def step(self, x, u, dist=None):
        r = self.network(x, u)
        if dist is not None:
            r = r.copy()
            r[0:3] += dist
        return _next_state(x, np.asarray(u, float), r, self.dt, self.alpha, self.with_swing,
                           np.concatenate)

    def rollout(self, x0, U, dist=None):
        X = np.zeros((NX, U.shape[1] + 1))
        X[:, 0] = x0
        for k in range(U.shape[1]):
            X[:, k + 1] = self.step(X[:, k], U[:, k], dist)
        return X

    def disturbance_map(self):
        """How a velocity offset d (3) enters the next state (to first order)."""
        E = np.zeros((NX, 3))
        E[3:6] = np.eye(3)
        E[0:3] = 0.5 * self.dt * np.eye(3)
        return E

    def casadi_step(self):
        """CasADi function F(x, u, d) -> x_next with the same dynamics."""
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        d = ca.SX.sym("d", 3)
        net = {k: ca.DM(v) for k, v in self.net.items() if k != "blocks"}
        net["blocks"] = [tuple(ca.DM(a) for a in blk) for blk in self.net["blocks"]]
        z = ca.vertcat(x[FEATURE_IDX.tolist()], u)
        r = _mlp(z, net, ca.tanh, lambda W, v: ca.mtimes(W, v))
        r = ca.vertcat(r[0:3] + d, r[3:7])
        x_next = _next_state(x, u, r, self.dt, self.alpha, self.with_swing,
                             lambda parts: ca.vertcat(*parts))
        return ca.Function("F_resnet", [x, u, d], [x_next])


NET_KEYS = ("W_in", "b_in", "W_out", "b_out", "W_skip", "mu", "sd", "out_scale")


def save_resnet(path, nets, dt, alpha, meta):
    arrays = {}
    for m, net in enumerate(nets):
        for k in NET_KEYS:
            arrays[f"m{m}_{k}"] = net[k]
        for j, blk in enumerate(net["blocks"]):
            for name, a in zip(("W1", "b1", "W2", "b2"), blk):
                arrays[f"m{m}_blk{j}_{name}"] = a
    np.savez(path, n_blocks=len(nets[0]["blocks"]), dt=dt, alpha=alpha, **arrays,
             meta_form=MODEL_FORM, **{f"meta_{k}": v for k, v in meta.items()})


def load_resnet(path=RESNET_FILE):
    """ResNetModel per mode (mode 0: no payload, mode 1: payload attached)."""
    d = np.load(path)
    if "meta_form" not in d.files or str(d["meta_form"]) != MODEL_FORM:
        raise ValueError(f"{path} is not an increment-form model (x_k+1 = x_k + f): retrain with "
                         "'python -m multi_drone_mujoco.envs.director_resnet_mpc'")
    models = []
    for m in (0, 1):
        net = {k: d[f"m{m}_{k}"] for k in NET_KEYS}
        net["blocks"] = [tuple(d[f"m{m}_blk{j}_{n}"] for n in ("W1", "b1", "W2", "b2"))
                         for j in range(int(d["n_blocks"]))]
        models.append(ResNetModel(net, float(d["dt"]), float(d["alpha"]), with_swing=bool(m)))
    return models


################################################################################
# TRAINING DATA
################################################################################

def hook_payload_contact(env):
    """Any contact between the hook and the payload (cylinder, holder, connectors)."""
    m, d = env.model, env.data
    hook = {m.geom(f"segment_geom_{i}").id for i in range(1, 8)}
    payload = {env.target_geom_id, env.left_connector_geom_id,
               env.right_connector_geom_id, m.geom("holder_plate").id}
    return any((c.geom1 in hook and c.geom2 in payload) or (c.geom2 in hook and c.geom1 in payload)
               for c in d.contact[:d.ncon])


def _collect_p(args):
    """P-controller mission run (director_mpc.collect) as a training sequence."""
    seed, excite = args
    r = collect(seed, excite=excite, verbose=False)
    att = r["attached"].astype(bool)
    contact = np.isin(r["phase"], CONTACT_PHASES) & ~att
    return {"x": r["x"], "u": r["u"], "attached": att, "valid": ~contact, "contact": contact,
            "source": "P", "seed": seed}


def make_rl_env(seed, duration=RL_DURATION):
    """Director env with the final-task settings, for RL director flights."""
    env = make_identification_env(seed)
    env.GRAB_FLAG_ENABLE = True
    env.MIN_PAYLOAD_MASS, env.MAX_PAYLOAD_MASS = 0.01, 0.25
    env.MIN_PAYLOAD_RADIUS, env.MAX_PAYLOAD_RADIUS = 0.02, 0.04
    env.GOAL_RANDOM_AMPLITUDE = 1.5
    env.EPISODE_LEN_SEC = duration
    return env


def _collect_rl(args):
    """Flight of the RL director policy (aggressive commands) as a training sequence.

    Steps where the hook touches the payload before it is lifted (external
    forces the model cannot know) or where the drone is already tilted beyond
    60 deg are marked invalid.
    """
    seed, policy_path, duration = args
    from stable_baselines3 import PPO
    policy = PPO.load(policy_path, device="cpu")
    env = make_rl_env(seed, duration)
    obs, _ = env.reset(seed=seed)
    env.prev_action = None
    tracker = StateTracker(env)
    rest_z = env.data.qpos[env.target_qpos_adr + 2]
    attached = False
    log = {k: [] for k in ("x", "u", "attached", "valid", "contact")}
    while True:
        x = tracker.measure(attached)
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float)
        contact = hook_payload_contact(env)
        log["x"].append(x)
        log["u"].append(np.clip(action[0:3], -1.0, 1.0))
        log["attached"].append(attached)
        log["valid"].append((attached or not contact) and np.all(np.abs(x[ATT]) < np.radians(60)))
        log["contact"].append(contact and not attached)
        tracker.advance(x)
        obs, _, terminated, truncated, _ = env.step(action)
        attached = payload_attached(env, rest_z, attached)
        if terminated or truncated:
            break
    env.close()
    out = {k: np.asarray(v) for k, v in log.items()}
    out["source"] = "RL"
    out["seed"] = seed
    return out


def collect_runs(jobs, workers=8):
    """Run collection jobs [(fn, args)] in parallel processes.

    The workers are started with "spawn": a forked child cannot use CUDA once
    the parent process has initialized it (e.g. a PPO policy loaded on a GPU).
    """
    import multiprocessing
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = [ex.submit(fn, a) for fn, a in jobs]
        return [f.result() for f in futures]


def collect_datasets(policy_path=DIRECTOR_POLICY, workers=8, verbose=True):
    """Training / validation sequences (P-controller, strongly excited, RL flights)."""
    train_jobs = [(_collect_p, a) for a in P_TRAIN]
    val_jobs = [(_collect_p, a) for a in P_VAL]
    if os.path.exists(policy_path):
        train_jobs += [(_collect_rl, (s, policy_path, RL_DURATION)) for s in RL_TRAIN]
        val_jobs += [(_collect_rl, (s, policy_path, RL_DURATION)) for s in RL_VAL]
    elif verbose:
        print(f"  (no RL director policy at {policy_path}: training without RL flights)")
    if verbose:
        print(f"Collecting {len(train_jobs) + len(val_jobs)} runs ({len(train_jobs)} training, "
              f"{len(val_jobs)} validation)...")
    runs = collect_runs(train_jobs + val_jobs, workers)
    return runs[:len(train_jobs)], runs[len(train_jobs):]


def _windows(runs, mode, H):
    """Start indices (run, k) of H-step windows entirely in ``mode`` and valid."""
    out = []
    for i, r in enumerate(runs):
        ok = (r["attached"].astype(bool) == bool(mode)) & r["valid"].astype(bool)
        T = len(ok)
        run_ok = np.r_[ok, False]
        for k in range(1, T - H):
            if run_ok[k:k + H + 1].all():
                out.append((i, k))
    return out


################################################################################
# TRAINING (PyTorch)
################################################################################

def train_resnet(path=RESNET_FILE, policy_path=DIRECTOR_POLICY, epochs=150, horizon=24,
                 verbose=True, reuse_data=False, data_file=DATA_FILE, threads=8):
    """Collect data (or reuse the cached runs), train one network per mode,
    save it and print the held-out comparison with the ARX model.

    Re-collect the data (reuse_data=False) whenever the closed loop changed
    (new velocity policy, physics changes).
    """
    import pickle

    import torch
    from torch import nn

    torch.manual_seed(0)
    torch.set_num_threads(threads)
    if reuse_data and os.path.exists(data_file):
        with open(data_file, "rb") as f:
            train_runs, val_runs = pickle.load(f)
        if verbose:
            print(f"Reusing the training data in {data_file}")
    else:
        train_runs, val_runs = collect_datasets(policy_path, verbose=verbose)
        os.makedirs(os.path.dirname(data_file) or ".", exist_ok=True)
        with open(data_file, "wb") as f:
            pickle.dump((train_runs, val_runs), f)
    env = make_identification_env(0)
    dt, alpha = env.CTRL_TIMESTEP, env.alpha
    env.close()

    class ResMLP(nn.Module):
        def __init__(self, mu, sd, out_scale):
            super().__init__()
            self.register_buffer("mu", torch.tensor(mu, dtype=torch.float64))
            self.register_buffer("sd", torch.tensor(sd, dtype=torch.float64))
            self.register_buffer("scale", torch.tensor(out_scale, dtype=torch.float64))
            self.inp = nn.Linear(N_IN, HIDDEN).double()
            self.blocks = nn.ModuleList([nn.ModuleList([nn.Linear(HIDDEN, HIDDEN).double(),
                                                        nn.Linear(HIDDEN, HIDDEN).double()])
                                         for _ in range(BLOCKS)])
            self.out = nn.Linear(HIDDEN, N_OUT).double()
            self.skip = nn.Linear(N_IN, N_OUT, bias=False).double()
            for layer in (self.out, self.skip):    # start at x_k+1 = x_k
                nn.init.zeros_(layer.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, z):
            zn = (z - self.mu) / self.sd
            h = torch.tanh(self.inp(zn))
            for l1, l2 in self.blocks:
                h = h + l2(torch.tanh(l1(h)))
            return (self.out(h) + self.skip(zn)) * self.scale

        def export(self):
            f = lambda t: t.detach().cpu().numpy().copy()
            return {"W_in": f(self.inp.weight), "b_in": f(self.inp.bias),
                    "W_out": f(self.out.weight), "b_out": f(self.out.bias),
                    "W_skip": f(self.skip.weight),
                    "mu": f(self.mu), "sd": f(self.sd), "out_scale": f(self.scale),
                    "blocks": [(f(l1.weight), f(l1.bias), f(l2.weight), f(l2.bias))
                               for l1, l2 in self.blocks]}

    def next_state(x, u, r, with_swing):
        """Batched version of _next_state: x (B, NX), u (B, NU), r (B, 7)."""
        v, c = x[:, 3:6], x[:, 9:12]
        th, thd, att = x[:, 15:17], x[:, 17:19], x[:, 19:21]
        dv = r[:, 0:3]
        if with_swing:
            thd1 = thd + r[:, 3:5]
            th1 = th + dt * thd1
        else:
            thd1, th1 = thd, th
        return torch.cat([x[:, 0:3] + dt * (v + 0.5 * dv), v + dv, v, c + alpha * (u - c), c,
                          th1, thd1, att + r[:, 5:7], att], dim=1)

    def rollout(net, X0, U, with_swing):
        """Batched H-step rollout: X0 (B, NX), U (B, H, NU) -> (B, H, NX)."""
        x, out = X0, []
        for j in range(U.shape[1]):
            u = U[:, j]
            x = next_state(x, u, net(torch.cat([x[:, 3:], u], dim=1)), with_swing)
            out.append(x)
        return torch.stack(out, dim=1)

    def batch(runs, wins):
        X0 = np.stack([runs[i]["x"][k] for i, k in wins]).astype(np.float64)
        U = np.stack([runs[i]["u"][k:k + horizon] for i, k in wins]).astype(np.float64)
        Y = np.stack([runs[i]["x"][k + 1:k + horizon + 1] for i, k in wins]).astype(np.float64)
        return torch.tensor(X0), torch.tensor(U), torch.tensor(Y)

    nets, report = [], {}
    for mode in (0, 1):
        with_swing = bool(mode)
        tr_w = _windows(train_runs, mode, horizon)
        va_w = _windows(val_runs, mode, horizon)
        X0, U, Y = batch(train_runs, tr_w)
        X0v, Uv, Yv = batch(val_runs, va_w) if va_w else (X0[:1], U[:1], Y[:1])

        # input normalization, output scale = spread of the one-step increments
        feats = torch.cat([X0[:, 3:], U[:, 0]], dim=1).numpy()
        mu, sd = feats.mean(0), feats.std(0)
        sd[sd < 1e-4] = 1.0                    # constant features (swing in mode 0)
        inc = (Y[:, 0] - X0).numpy()
        out_scale = np.r_[inc[:, 3:6].std(0), inc[:, 17:19].std(0) if with_swing else [1, 1],
                          inc[:, 19:21].std(0)] + 1e-6

        # loss channels: velocity, tilt (+ swing), scaled by their spread
        ch = [3, 4, 5, 19, 20] + ([15, 16, 17, 18] if with_swing else [])
        w = torch.tensor(1.0 / (Y.reshape(-1, NX)[:, ch].std(0).numpy() + 1e-3) ** 2)

        def loss_fn(net, X0_, U_, Y_, H):
            pred = rollout(net, X0_, U_[:, :H], with_swing)
            return ((pred[..., ch] - Y_[:, :H][..., ch]) ** 2 * w).mean()

        net = ResMLP(mu, sd, out_scale)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state = np.inf, None
        n, bs = len(tr_w), 256
        t0 = time.time()
        if verbose:
            print(f"mode {mode}: {n} training / {len(va_w)} validation windows")
        for ep in range(epochs):
            H = max(2, min(horizon, int(horizon * (ep + 1) / (0.4 * epochs))))  # 2 -> 24 steps
            perm = torch.randperm(n)
            for b0 in range(0, n, bs):
                idx = perm[b0:b0 + bs]
                opt.zero_grad()
                loss = loss_fn(net, X0[idx], U[idx], Y[idx], H)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            if H == horizon:
                with torch.no_grad():
                    v = loss_fn(net, X0v, Uv, Yv, horizon).item()
                if v < best:
                    best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  mode {mode} epoch {ep:3d}  H={H:2d}  train {loss.item():.4f}"
                      f"  best val {best:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        net.load_state_dict(best_state)
        nets.append(net.export())
        report[mode] = (len(tr_w), len(va_w))

    meta = {"horizon": horizon, "epochs": epochs, "hidden": HIDDEN, "blocks": BLOCKS,
            "train_windows_mode0": report[0][0], "train_windows_mode1": report[1][0]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_resnet(path, nets, dt, alpha, meta)
    models = load_resnet(path)
    if verbose:
        print("saved:", path)
        try:
            compare_models(val_runs, load_models(DEFAULT_MODEL_FILE), models)
        except (FileNotFoundError, ValueError) as err:
            print(f"(no ARX model for the comparison: {err})")
    return models


def compare_models(runs, arx, resnet, steps=(12, 24, 40)):
    """Multi-step prediction error on held-out runs: ARX vs ResNet model.

    (utilities/analyse_identification.py has the full comparison.)
    """
    print("\nHeld-out multi-step prediction error (RMS)")
    print(f"{'data':6s} {'mode':5s} {'model':8s}" + "".join(
        f"  v@{s / 48:.2f}s[cm/s] tilt@{s / 48:.2f}s[deg]" for s in steps))
    for src in ("P", "RL"):
        sub = [r for r in runs if r["source"] == src]
        if not sub:
            continue
        for mode in (0, 1):
            wins = _windows(sub, mode, max(steps))
            if not wins:
                continue
            wins = wins[::5]
            for name, models in (("ARX", arx), ("ResNet", resnet)):
                ev = {s: [] for s in steps}
                et = {s: [] for s in steps}
                for i, k in wins:
                    r = sub[i]
                    Xp = models[mode].rollout(r["x"][k], r["u"][k:k + max(steps)].T)
                    for s in steps:
                        ev[s].append(np.linalg.norm(Xp[3:6, s] - r["x"][k + s, 3:6]))
                        et[s].append(np.abs(Xp[19:21, s] - r["x"][k + s, 19:21]).max())
                print(f"{src:6s} {mode:<5d} {name:8s}" + "".join(
                    f"  {100 * np.sqrt(np.mean(np.square(ev[s]))):13.1f} "
                    f"{np.degrees(np.sqrt(np.mean(np.square(et[s])))):15.1f}" for s in steps))


def load_or_train(model_file=RESNET_FILE):
    """Load the ResNet model; train it first if it is missing or outdated."""
    try:
        return load_resnet(model_file)
    except (FileNotFoundError, ValueError) as err:
        print(f"[ResNet model] {err}\n  -> training the ResNet model now (several minutes)...")
        return train_resnet(model_file)


################################################################################
# MPC
################################################################################

class ResNetDirectorMPC:
    """Director MPC with the ResNet model (Gauss-Newton SQP, condensed QP).

    Cost, bounds, tilt constraints and the online velocity-offset estimate
    are the same as in ``DirectorMPC``.
    """

    def __init__(self, models, horizon=40, weights=None, u_max=U_MAX, max_tilt=np.pi / 2,
                 disturbance_gain=0.05, sqp_iterations=2, threads=8):
        self.models = models
        self.N = horizon
        self.w = weights or DirectorWeights()
        self.u_max = u_max
        self.max_tilt = max_tilt
        self.dist_gain = disturbance_gain
        self.sqp_iterations = sqp_iterations
        self.F, self._lin = [], []
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        d = ca.SX.sym("d", 3)
        for m in models:
            F = m.casadi_step()
            xn = F(x, u, d)
            lin = ca.Function("lin", [x, u, d], [xn, ca.jacobian(xn, x), ca.jacobian(xn, u)])
            self.F.append(F)
            self._lin.append(lin.map(horizon, "thread", threads))
        nz = NU * horizon
        opts = {"printLevel": "none", "error_on_fail": False}
        self._qp_con = ca.conic("qp_con", "qpoases",
                                {"h": ca.Sparsity.dense(nz, nz),
                                 "a": ca.Sparsity.dense(2 * horizon, nz)}, opts)
        self._qp = ca.conic("qp", "qpoases",
                            {"h": ca.Sparsity.dense(nz, nz), "a": ca.Sparsity(0, nz)}, opts)
        self._D = np.eye(nz) - np.eye(nz, k=-NU)
        self.reset()

    def reset(self):
        self.dist = np.zeros(3)
        self._last = None
        self._plan = None

    def _Q(self, mode):
        q = np.zeros(NX)
        q[P] = self.w.pos
        q[V] = self.w.vel
        if mode == 1:
            q[TH] = self.w.swing
            q[THD] = self.w.swing_rate
        Q = np.tile(q, self.N)
        Q[-NX:] *= self.w.terminal
        return Q

    def rollout(self, mode, x0, U, dist=None):
        d = np.zeros(3) if dist is None else dist
        X = np.zeros((NX, U.shape[1] + 1))
        X[:, 0] = x0
        for k in range(U.shape[1]):
            X[:, k + 1] = np.asarray(self.F[mode](X[:, k], U[:, k], d)).ravel()
        return X

    def _update_disturbance(self, mode, x):
        if self._last is None or self._last[0] != mode:
            return
        _, x_prev, u_prev = self._last
        pred = np.asarray(self.F[mode](x_prev, u_prev, self.dist)).ravel()
        self.dist += self.dist_gain * (x[V] - pred[V])

    def solve(self, mode, x0, p_ref, v_ref, u_prev):
        t0 = time.perf_counter()
        self._update_disturbance(mode, x0)
        N, D = self.N, self._D
        if self._plan is None:
            U = np.zeros((NU, N))
        else:
            U = np.hstack([self._plan[:, 1:], self._plan[:, -1:]])
        ref = np.zeros((NX, N))
        ref[P] = p_ref[:, 1:]
        ref[V] = v_ref[:, 1:]
        Q = self._Q(mode)
        e_prev = np.zeros(NU * N)
        e_prev[:NU] = u_prev
        dmat = np.tile(self.dist[:, None], (1, N))

        def cost(X, U_):
            e = (X[:, 1:] - ref).T.ravel()
            u = U_.T.ravel()
            du = D @ u - e_prev
            viol = np.maximum(np.abs(X[ATT, 1:]) - self.max_tilt, 0).sum()
            return 0.5 * (e @ (Q * e) + self.w.u * u @ u + self.w.du * du @ du) + 1e4 * viol

        X = self.rollout(mode, x0, U, self.dist)
        J = cost(X, U)
        qp_failures = 0
        for _ in range(self.sqp_iterations):
            _, A, B = self._lin[mode](X[:, :N], U, dmat)
            A = np.asarray(A).reshape(NX, N, NX)
            B = np.asarray(B).reshape(NX, N, NU)
            Gamma = np.zeros((N, NX, NU * N))
            G = np.zeros((NX, NU * N))
            for k in range(N):
                G = A[:, k, :] @ G
                G[:, NU * k:NU * (k + 1)] += B[:, k, :]
                Gamma[k] = G
            Gamma = Gamma.reshape(N * NX, NU * N)
            err = (X[:, 1:] - ref).T.ravel()
            u_flat = U.T.ravel()
            GQ = Gamma.T * Q
            H = GQ @ Gamma + self.w.u * np.eye(NU * N) + self.w.du * D.T @ D
            g = GQ @ err + self.w.u * u_flat + self.w.du * D.T @ (D @ u_flat - e_prev)
            rows = (np.arange(N)[:, None] * NX + np.array([19, 20])[None]).ravel()
            tilt = X[ATT, 1:].T.ravel()
            sol = quiet_qp(self._qp_con, h=H, g=g, a=Gamma[rows], lba=-self.max_tilt - tilt,
                           uba=self.max_tilt - tilt, lbx=-self.u_max - u_flat,
                           ubx=self.u_max - u_flat)
            ok = self._qp_con.stats()["success"]
            if not ok:
                sol = self._qp(h=H, g=g, lbx=-self.u_max - u_flat, ubx=self.u_max - u_flat)
                ok = self._qp.stats()["success"]
            dU = np.asarray(sol["x"]).ravel().reshape(N, NU).T
            if not (ok and np.all(np.isfinite(dU))):
                qp_failures += 1
                break
            improved = False
            for step in (1.0, 0.5, 0.25):
                U_try = np.clip(U + step * dU, -self.u_max, self.u_max)
                X_try = self.rollout(mode, x0, U_try, self.dist)
                J_try = cost(X_try, U_try)
                if J_try < J:
                    U, X, J, improved = U_try, X_try, J_try, True
                    break
            if not improved:
                break
        self._plan = U
        self._last = (mode, x0.copy(), U[:, 0].copy())
        return X, U, {"solve_time": time.perf_counter() - t0, "qp_failures": qp_failures,
                      "max_tilt_pred": float(np.abs(X[ATT, 1:]).max())}


################################################################################
# AGENT
################################################################################

class DirectorResNetMPCAgent(MissionAgent):
    """Mission agent like DirectorMPCAgent, with the ResNet-model MPC."""

    tendon_hold = TENDON_HOLD

    def __init__(self, env, model_file=RESNET_FILE, horizon=40, max_time=45.0, verbose=False,
                 max_tilt=np.pi / 2):
        super().__init__(env, horizon, max_time, verbose)
        self.models = load_or_train(model_file)
        self.mpc = ResNetDirectorMPC(self.models, horizon=horizon, max_tilt=max_tilt)

    def _reset_controller(self):
        self.env.prev_action = None
        self.mpc.reset()
        self.tracker = StateTracker(self.env)
        self.u_prev = np.zeros(NU)

    def _measure(self, attached):
        return self.tracker.measure(attached)

    def _control(self, mode, x, p_ref, v_ref):
        X_pred, U_plan, info = self.mpc.solve(mode, x, p_ref, v_ref, self.u_prev)
        u = U_plan[:, 0]
        self.u_prev = u
        return (u, self.mission.tendon), u, X_pred, info

    def _plant_step(self, action):
        u, tendon = action
        self.tracker.advance(self.x)
        return director_step(self.env, u, (tendon, -tendon))

    def _on_attach(self):
        self._say("payload attached -> switching to drone+pendulum model")

    def _replay(self, mode, x0, U):
        return self.models[mode].rollout(x0, U)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the ResNet director model")
    parser.add_argument("--reuse_data", action="store_true",
                        help="reuse the cached training runs instead of collecting new ones")
    parser.add_argument("--epochs", type=int, default=150)
    a = parser.parse_args()
    train_resnet(reuse_data=a.reuse_data, epochs=a.epochs)
