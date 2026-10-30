"""ResNet-identified prediction model and the MPC solver that uses it.

This module holds three things (docs/director_mpc_safety_filter.md):

1. the prediction model ``ResNetDynamics``: a residual network that predicts
   the closed loop "velocity command -> env low-pass filter -> PPO velocity
   controller -> drone (+ payload)" one control step (1/48 s) ahead,
2. its training data and training (``collect_datasets``, ``train``),
3. the MPC solver ``ResNetMPC`` built on it. The director MPC
   (waypoint_mpc.py) and the stall takeover of the safety filter
   (adaptive_hook_director_safety_filter.py) use this solver; the predictive
   safety filter (predictive_safety_filter.py) uses the same model with its
   own QPs.

Implementation:

  * the network is one PyTorch module. PyTorch trains it, and the MPC and the
    safety filter use the same module for the rollouts and for the Jacobians
    (``torch.func``), so there is no second (e.g. CasADi) copy of the model,
  * the Gauss-Newton QP is written as a least-squares problem: cost
    0.5 |r(U)|^2, QP Hessian J_r^T J_r and gradient J_r^T r,
  * the QP is solved with DAQP (dense active-set solver, through CasADi's
    ``conic`` interface).

Method:

    x_k+1 = x_k + f(x_k, u_k)       f: ResNet, one network per mode
                                    (mode 0: no payload, mode 1: payload on the hook)

    min_U  0.5 sum_k |x_k - r_k|_Q^2 + 0.5 w_u sum_k |u_k|^2 + 0.5 w_du sum_k |u_k - u_k-1|^2
    s.t.   |u_k| <= u_max,   |roll_k|, |pitch_k| <= max_tilt

solved every control step by ``sqp_iterations`` Gauss-Newton SQP iterations
(single shooting): linearize the model along the current plan, solve the QP
for the step dU, backtracking line search on the nonlinear cost. The plan is
warm-started with the shifted previous plan. The model is used as identified
(no online correction of its predictions).

The training data are P-controller identification runs, strongly excited
runs and flights of the RL director policy; steps where the hook touches the
payload before it is lifted (forces the model cannot know) are left out.

Train / retrain (after a new velocity policy or physics changes):
    python -m multi_drone_mujoco.envs.resnet_mpc              # train on the cached runs
    python -m multi_drone_mujoco.envs.resnet_mpc --collect    # collect new runs first
(the P-controller runs of the cache were collected with director_mpc.collect,
a P-controller on the env's waypoints, but in an earlier version of the
environment, so they are not reproduced step by step any more; the RL
director flights are)
"""

import copy
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import casadi as ca
import numpy as np
import torch
from torch import nn
from torch.func import jacrev, vmap

from multi_drone_mujoco.envs.director_mpc import (
    ATT, ATTD, C, NU, NX, P, TH, THD, TILT, U_MAX, V,
    DirectorWeights, collect, hook_payload_contact, make_identification_env, measure_state,
    payload_attached,
)

MODEL_FILE = "results/resnet/resnet_model.pt"
DATA_FILE = "results/resnet/resnet_training_data.pkl"   # cached training runs
DIRECTOR_POLICY = "results/final/rl_adaptive_director_curriculum/final_model.zip"
# The position does not enter the network: the closed loop behaves the same
# everywhere in the room (translation invariance), so the network sees the
# other 16 states and the 3 commands.
N_IN = (NX - 3) + NU        # network input: every state except the position, plus the command
N_OUT = 11                  # dv (3), dtheta_dot (2), d(roll, pitch, yaw) (3), d(their rates) (3)


################################################################################
# MODEL
################################################################################

class ResNetDynamics(nn.Module):
    """x_k+1 = x_k + f(x_k, u_k) of one mode (0: no payload, 1: payload attached).

    The network predicts the increments of the velocity, the swing rate,
    the attitude and the attitude rates; the rest of the state update is
    written out (position and swing angle integrated, the env's command
    low-pass filter). The attitude is not integrated from its rate: the PPO
    controller makes roll and yaw chatter within a control step, so the
    sampled rates are aliased. Works on batches (..., NX).

    Structure of the network (``network``):

        z  = [x without p, u]                      (N_IN)
        zn = (z - mu) / sd                         input normalization
        h  = tanh(W_in zn)                         (hidden)
        h  = h + block(h)       x ``blocks``       residual blocks Linear-Tanh-Linear
        r  = (W_out h + W_skip zn) * out_scale     (N_OUT) increments

    The linear shortcut W_skip lets the network represent the linear part of
    the dynamics directly; the hidden layers only have to learn the
    nonlinear correction. ``out_scale`` (the spread of every increment in
    the training data) keeps all outputs of the last layer around unit size.
    """

    def __init__(self, dt, alpha, with_swing, hidden=128, blocks=2):
        super().__init__()
        # dt: control period (1/48 s); alpha: gain of the env's command
        # low-pass filter; with_swing: mode 1 (payload swing states are active)
        self.dt, self.alpha, self.with_swing = dt, alpha, with_swing
        # input normalization and output scale (set from the training data);
        # buffers, not parameters: saved with the model but not trained
        self.register_buffer("mu", torch.zeros(N_IN))
        self.register_buffer("sd", torch.ones(N_IN))
        self.register_buffer("out_scale", torch.ones(N_OUT))
        self.inp = nn.Linear(N_IN, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
            for _ in range(blocks))
        self.out = nn.Linear(hidden, N_OUT)
        self.skip = nn.Linear(N_IN, N_OUT, bias=False)   # linear shortcut input -> output
        # zero output layers: the untrained model predicts "nothing changes"
        # (x_k+1 = x_k), a sensible start for the multi-step rollout loss
        for p in (self.out.weight, self.out.bias, self.skip.weight):
            nn.init.zeros_(p)                            # start at x_k+1 = x_k
        # float64 throughout: the Jacobians feed a QP, and rollouts of 20-40
        # steps accumulate rounding errors in float32
        self.double()

    def network(self, z):
        """The increments r (..., N_OUT) for the input z = [x without p, u]."""
        zn = (z - self.mu) / self.sd
        h = torch.tanh(self.inp(zn))
        for block in self.blocks:
            h = h + block(h)                             # residual blocks
        return (self.out(h) + self.skip(zn)) * self.out_scale

    def forward(self, x, u):
        """Next state x_k+1 (..., NX) for the state x_k (..., NX) and the
        command u_k (..., NU)."""
        r = self.network(torch.cat([x[..., 3:], u], dim=-1))
        dv = r[..., 0:3]                                 # velocity increment
        v, c, th, thd = x[..., V], x[..., C], x[..., TH], x[..., THD]
        att, attd = x[..., ATT], x[..., ATTD]
        if self.with_swing:
            # swing: the network predicts the change of the swing rate, the
            # angle is integrated with the new rate (semi-implicit Euler)
            thd_next = thd + r[..., 3:5]
            th_next = th + self.dt * thd_next
        else:                                            # no payload: swing states stay 0
            th_next, thd_next = th, thd
        # the parts of the update that are known exactly are written out, the
        # network only predicts what depends on the PPO controller and the
        # drone / payload dynamics
        return torch.cat([x[..., P] + self.dt * (v + 0.5 * dv),       # p (trapezoidal)
                          v + dv,                                     # v
                          c + self.alpha * (u - c),                   # c (env low-pass filter)
                          th_next, thd_next,                          # theta, theta_dot
                          att + r[..., 5:8],                          # roll, pitch, yaw
                          attd + r[..., 8:11]], dim=-1)               # their rates

    def rollout(self, x0, U):
        """numpy interface like ``DirectorModel.rollout`` (analysis scripts):
        x0 (NX,), U (NU, N) -> X (NX, N+1)."""
        t = lambda a: torch.as_tensor(np.asarray(a, float))
        with torch.inference_mode():
            X = rollout(self, t(x0), t(np.asarray(U).T))
        return np.hstack([np.asarray(x0, float)[:, None], X.numpy().T])


def rollout(model, x0, U):
    """States x_1..x_H for the commands U (..., H, NU) from x0 (..., NX):
    the model applied H times in a row (open loop, no measurement in between).
    Works on batches: x0 (B, NX), U (B, H, NU) -> (B, H, NX)."""
    xs, x = [], x0
    for k in range(U.shape[-2]):
        x = model(x, U[..., k, :])
        xs.append(x)
    return torch.stack(xs, dim=-2)


def step_jacobians(model):
    """Function (X (N, NX), U (N, NU)) -> (A (N, NX, NX), B (N, NX, NU)):
    A_k = df/dx and B_k = df/du of the model at every step, in one call.

    ``jacrev`` differentiates one step of the model (reverse mode) with
    respect to x and u; ``vmap`` evaluates it for all N steps of a trajectory
    at once, without a Python loop over the steps.
    """
    jac = vmap(jacrev(model, argnums=(0, 1)), in_dims=(0, 0))
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64)
    jac(zeros(1, NX), zeros(1, NU))             # the first torch.func call of a process
    return jac                                  # takes ~0.6 s: not in a control step


def sensitivity(jacobians, X, U):
    """Gamma = d(x_1..x_N)/dU (N * NX, N * NU) of the linearized model along
    the trajectory X (at least x_0..x_N-1, (N, NX)) and the plan U (N, NU).

    The linearized model is dx_k+1 = A_k dx_k + B_k du_k with dx_0 = 0 (x_0
    is measured). Block row k tells how x_k+1 changes with every command:
    dx_k+1/du_j = A_k A_k-1 ... A_j+1 B_j for j <= k (0 for j > k, a command
    cannot act on earlier states). It is built recursively: G <- A_k G, then
    the new column block B_k is added. ``jacobians``: see ``step_jacobians``.
    """
    N = len(U)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    A, B = (m.numpy() for m in jacobians(t(X[:N]), t(U)))
    G, rows = np.zeros((NX, NU * N)), []
    for k in range(N):
        G = A[k] @ G
        G[:, NU * k:NU * (k + 1)] += B[k]
        rows.append(G)
    return np.vstack(rows)


def sqp(U, X, merit, qp_step, rollout, iterations, clip=None):
    """The SQP loop shared by ResNetMPC and PredictiveSafetyFilter.

    Starting from the plan U with the nonlinear prediction X:

        repeat ``iterations`` times:
            dU = qp_step(U, X)     linearize along (U, X), solve the Gauss-Newton QP
            line search:           U + step dU for step = 1, 0.5, 0.25, each
                                   predicted with the NONLINEAR ``rollout``; the
                                   first step that decreases ``merit`` is taken
            stop if the QP fails or no step decreases the merit

    The QP only works on the linearized model, so its step is a direction;
    the line search accepts it only if the real (nonlinear) prediction
    improves. ``qp_step(U, X)`` returns (dU, extra) or None if the QP fails;
    ``extra`` is passed through (the filter's slacks). ``clip`` = (lo, hi)
    clips every trial plan (the MPC's command bounds).

    Returns (U, X, merit value, solved, extra): ``solved`` is False if not
    even the first QP was solved; ``extra`` belongs to the last accepted step
    (or to the last QP if no step was accepted).
    """
    m = merit(U, X)
    solved, extra, extra_accepted = False, None, None
    for _ in range(iterations):
        r = qp_step(U, X)
        if r is None:
            break
        dU, extra = r
        solved = True
        for step in (1.0, 0.5, 0.25):
            U_try = U + step * dU
            if clip is not None:
                U_try = np.clip(U_try, *clip)
            X_try = rollout(U_try)
            m_try = merit(U_try, X_try)
            if m_try < m:
                U, X, m, extra_accepted = U_try, X_try, m_try, extra
                break
        else:
            break                                    # no improvement: stop with the current plan
    return U, X, m, solved, (extra if extra_accepted is None else extra_accepted)


def save_model(path, models):
    """Save both mode models (weights + the settings to rebuild them)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"nx": NX, "dt": models[0].dt, "alpha": models[0].alpha,
                "hidden": models[0].inp.out_features, "blocks": len(models[0].blocks),
                "state": [m.state_dict() for m in models]}, path)


def load_model(path=MODEL_FILE):
    """[mode 0 model, mode 1 model] from a file written by ``save_model``."""
    ckpt = torch.load(path, weights_only=True)
    if ckpt.get("nx") != NX:
        raise ValueError(f"{path} is a model of a different state layout "
                         f"(nx={ckpt.get('nx', 23)}, now {NX}): retrain it with "
                         "'python -m multi_drone_mujoco.envs.resnet_mpc --collect'")
    models = []
    for mode, state in enumerate(ckpt["state"]):
        m = ResNetDynamics(ckpt["dt"], ckpt["alpha"], bool(mode), ckpt["hidden"], ckpt["blocks"])
        m.load_state_dict(state)
        models.append(m)
    return models


def load_or_train(path=MODEL_FILE):
    """Load the model; train it first if the file is missing."""
    if not os.path.exists(path):
        print(f"[ResNet model] {path} not found -> training it now (~25-30 minutes)...")
        return train(path)
    return load_model(path)


################################################################################
# TRAINING DATA
################################################################################

# training / validation runs (seeds differ from the ones of the test flights)
# (the thesis model: 16 P-controller and 144 RL director flights; the extra
# flights reduced the prediction error on unseen RL flights, more epochs did not)
# P-controller runs: (seed, excitation amplitude of the random command noise)
P_TRAIN = ([(s, 0.15) for s in (100, 101, 102, 103)] + [(s, 0.6) for s in (104, 105, 106, 107)]
           + [(s, 0.15 if s % 2 else 0.6) for s in range(110, 118)])
P_VAL = [(108, 0.4)]
# RL director flights: the policy's bang-bang commands cover the aggressive
# part of the state space that the MPC and the filter have to predict
RL_TRAIN = tuple(range(300, 324)) + tuple(range(500, 620))
RL_VAL = (400, 401, 402, 403)
RL_DURATION = 15.0


def _collect_p(args):
    """P-controller mission run (director_mpc.collect) as a training sequence."""
    seed, excite = args
    r = collect(seed, excite=excite, verbose=False)
    att = r["attached"].astype(bool)
    contact = r["contact"].astype(bool) & ~att
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
    rest_z = env.data.qpos[env.target_qpos_adr + 2]
    attached = False
    log = {k: [] for k in ("x", "u", "attached", "valid", "contact")}
    while True:
        # log the state *before* the step together with the command applied
        # in it: the pair (x_k, u_k) -> x_k+1 is one training sample
        x = measure_state(env, attached)
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float)
        contact = hook_payload_contact(env)
        log["x"].append(x)
        log["u"].append(np.clip(action[0:3], -1.0, 1.0))
        log["attached"].append(attached)
        log["valid"].append((attached or not contact) and np.all(np.abs(x[TILT]) < np.radians(60)))
        log["contact"].append(contact and not attached)
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
    """Start indices (run, k) of H-step windows entirely in ``mode`` and valid.

    A window x_k..x_k+H is used only if all its H+1 states are in the same
    mode (no grasp / loss of the payload inside it) and valid (no contact
    forces the model cannot know). k starts at 1: the first sample of a run
    has no filtered command c yet.
    """
    out = []
    for i, r in enumerate(runs):
        ok = (r["attached"].astype(bool) == bool(mode)) & r["valid"].astype(bool)
        run_ok = np.r_[ok, False]
        for k in range(1, len(ok) - H):
            if run_ok[k:k + H + 1].all():
                out.append((i, k))
    return out


################################################################################
# TRAINING
################################################################################

def train(path=MODEL_FILE, epochs=150, horizon=24, reuse_data=True, threads=8, verbose=True):
    """Train one network per mode with a multi-step rollout loss.

    The rollout horizon grows from 2 to ``horizon`` steps during the first
    40% of the epochs (the MPC relies on multi-step predictions); the
    weights with the best validation loss at the full horizon are kept.

    Why a rollout loss: a model trained only on one-step errors can drift
    away quickly when it is applied to its own predictions over a 20-40 step
    horizon. Here the loss compares the whole open-loop prediction from x_k
    with the measured x_k+1..x_k+H, so the training penalizes exactly the
    error the MPC sees. The horizon starts short (easy, stable gradients) and
    is increased step by step (curriculum).
    """
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    torch.set_num_threads(threads)
    if reuse_data and os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            train_runs, val_runs = pickle.load(f)
    else:
        train_runs, val_runs = collect_datasets(verbose=verbose)
        os.makedirs(os.path.dirname(DATA_FILE) or ".", exist_ok=True)
        with open(DATA_FILE, "wb") as f:
            pickle.dump((train_runs, val_runs), f)
    env = make_identification_env(0)
    dt, alpha = env.CTRL_TIMESTEP, env.alpha
    env.close()

    def windows(runs, mode):
        """(x_k, u_k..k+H-1, x_k+1..k+H) of every valid H-step window in ``mode``."""
        idx = _windows(runs, mode, horizon)
        x0 = np.stack([runs[i]["x"][k] for i, k in idx])
        u = np.stack([runs[i]["u"][k:k + horizon] for i, k in idx])
        y = np.stack([runs[i]["x"][k + 1:k + horizon + 1] for i, k in idx])
        return TensorDataset(*(torch.tensor(a, dtype=torch.float64) for a in (x0, u, y)))

    models = []
    for mode in (0, 1):
        data, val = windows(train_runs, mode), windows(val_runs, mode)
        x0, u, y = data.tensors
        model = ResNetDynamics(dt, alpha, with_swing=bool(mode))
        # input normalization; output scale = spread of the one-step increments
        z = torch.cat([x0[:, 3:], u[:, 0]], dim=1)
        model.mu.copy_(z.mean(0))
        model.sd.copy_(torch.where(z.std(0) < 1e-4, 1.0, z.std(0)))   # constant inputs (swing in mode 0)
        inc = y[:, 0] - x0
        model.out_scale.copy_(torch.cat([inc[:, V].std(0),
                                         inc[:, THD].std(0) if mode else torch.ones(2, dtype=torch.float64),
                                         inc[:, ATT].std(0), inc[:, ATTD].std(0)]) + 1e-6)
        # loss: velocity, attitude and attitude rates (+ swing), each channel
        # scaled by its spread. The position is not in the loss: it follows
        # from the integrated velocity, so velocity errors already cover it.
        idx = lambda sl: list(range(sl.start, sl.stop))
        ch = idx(V) + idx(ATT) + idx(ATTD) + (idx(TH) + idx(THD) if mode else [])
        w = 1.0 / (y.reshape(-1, NX)[:, ch].std(0) + 1e-3) ** 2

        def loss_fn(x0_, u_, y_, H):
            # open-loop prediction over the first H steps of every window
            pred = rollout(model, x0_, u_[:, :H])
            return ((pred[..., ch] - y_[:, :H, ch]) ** 2 * w).mean()

        loader = DataLoader(data, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state, t0 = np.inf, None, time.time()
        for ep in range(epochs):
            # horizon curriculum: 2 -> ``horizon`` steps over the first 40 % of the epochs
            H = max(2, min(horizon, int(horizon * (ep + 1) / (0.4 * epochs))))
            for x0_, u_, y_ in loader:
                opt.zero_grad()
                loss = loss_fn(x0_, u_, y_, H)
                loss.backward()                          # backpropagation through the rollout
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            # model selection only at the full horizon (shorter-horizon
            # losses are not comparable)
            if H == horizon:
                with torch.no_grad():
                    v = loss_fn(*val.tensors, horizon).item()
                if v < best:
                    best, best_state = v, copy.deepcopy(model.state_dict())
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  mode {mode} epoch {ep:3d}  H={H:2d}  train {loss.item():.4f}"
                      f"  best val {best:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        models.append(model)
    save_model(path, models)
    if verbose:
        print("saved:", path)
    return models


################################################################################
# MPC
################################################################################

class ResNetMPC:
    """Gauss-Newton SQP MPC on the ResNet model (thesis, Section 4.1.3).

        min_U  0.5 sum_k |x_k - ref_k|_Q^2 + 0.5 w_u sum_k |u_k|^2 + 0.5 w_du sum_k |u_k - u_k-1|^2
        s.t.   |u_k| <= u_max,   |roll_k|, |pitch_k| <= max_tilt

    Single shooting: the decision variables are only the commands
    U = (u_0..u_N-1); the states x_1..x_N follow from the model rolled out
    from the measured x_0. The cost is a sum of squares, 0.5 |r(U)|^2, so
    every SQP iteration linearizes the residual, r(U + dU) ~ r + J_r dU, and
    solves the Gauss-Newton QP

        min_dU  0.5 dU^T (J_r^T J_r) dU + (J_r^T r)^T dU

    with the linearized tilt limit (DAQP), then shortens the step until the
    nonlinear cost decreases. The plan is warm-started with the previous
    plan shifted by one step.

    Usage: one ``solve`` per control step, apply U[:, 0]; ``reset()`` at the
    start of an episode.
    """

    TILT_PENALTY = 1e4      # tilt-limit violation in the line search cost

    def __init__(self, models, horizon=40, weights=None, u_max=U_MAX, max_tilt=np.pi / 2,
                 sqp_iterations=5, qp_solver="daqp", max_threads=4):
        # small matrices: more than ~4 threads only add overhead
        torch.set_num_threads(min(max_threads, torch.get_num_threads()))
        self.models = [m.requires_grad_(False) for m in models]     # only evaluated here
        self.jacobians = [step_jacobians(m) for m in self.models]
        self.N, self.w = horizon, weights or DirectorWeights()
        self.u_max, self.max_tilt, self.sqp_iterations = u_max, max_tilt, sqp_iterations
        n = NU * horizon
        # difference operator of the command changes; the change from the
        # previously applied command enters through e_prev in solve
        self.D = np.eye(n) - np.eye(n, k=-NU)          # D U = (u_0, u_1 - u_0, u_2 - u_1, ...)
        # the QP of a step: with the linearized tilt limit (2 rows per step),
        # and with the command bounds only (if the tilt limit is infeasible)
        opts = {"error_on_fail": False}
        H = ca.Sparsity.dense(n, n)
        self.qp = ca.conic("qp", qp_solver, {"h": H, "a": ca.Sparsity.dense(2 * horizon, n)}, opts)
        self.qp_box = ca.conic("qp_box", qp_solver, {"h": H, "a": ca.Sparsity(0, n)}, opts)
        self.reset()

    def reset(self):
        self._plan = None             # previous plan (N, NU), the warm start of the next solve

    def rollout(self, mode, x0, U):
        """Predicted states x_0..x_N (N+1, NX) for the plan U (N, NU)."""
        with torch.inference_mode():
            X = rollout(self.models[mode], torch.from_numpy(x0), torch.from_numpy(U))
        return np.vstack([x0, X.numpy()])

    def solve(self, mode, x0, p_ref, v_ref, u_prev):
        """One MPC step -> (X (NX, N+1), U (NU, N), info).

        mode: 0 (no payload) / 1 (payload), selects the model; x0: measured
        state; p_ref, v_ref: position / velocity reference (3, N+1), column 0
        unused; u_prev: the command applied in the previous step.
        """
        t0 = time.perf_counter()
        N, w = self.N, self.w
        x0 = np.asarray(x0, float)
        # reference of x_1..x_N (only the position and the velocity are tracked)
        ref = np.zeros((N, NX))
        ref[:, P], ref[:, V] = p_ref[:, 1:].T, v_ref[:, 1:].T
        # square roots of the weights (the cost is 0.5 |r|^2): position,
        # velocity, swing with the payload; the last step ``terminal`` times more
        q = np.zeros(NX)
        q[P], q[V] = w.pos, w.vel
        if mode == 1:
            q[TH], q[THD] = w.swing, w.swing_rate
        Q = np.tile(q, N)
        Q[-NX:] *= w.terminal
        sq, sq_u, sq_du = np.sqrt(Q), np.sqrt(w.u), np.sqrt(w.du)
        e_prev = np.r_[u_prev, np.zeros(NU * (N - 1))]     # D U - e_prev = (u_0 - u_prev, ...)

        def residual(X, U):
            """Least-squares residual, the cost is 0.5 |r|^2:
            r = [ sqrt(Q) (x_1..N - ref),  sqrt(w_u) u,  sqrt(w_du) (D U - e_prev) ]"""
            return np.r_[sq * (X[1:] - ref).ravel(), sq_u * U.ravel(),
                         sq_du * (self.D @ U.ravel() - e_prev)]

        def cost(X, U):
            """Nonlinear cost of the line search: the least-squares cost plus
            a large penalty on the predicted tilt beyond the limit (the QP
            only meets the tilt limit for the linearized model)."""
            r = residual(X, U)
            excess = np.maximum(np.abs(X[1:, TILT]) - self.max_tilt, 0.0)
            return 0.5 * r @ r + self.TILT_PENALTY * excess.sum()

        def qp_step(U, X):
            # Gauss-Newton: linearize the model, keep the cost quadratic.
            # With r(U + dU) ~ r(U) + J_r dU the cost becomes the QP
            #     min 0.5 dU^T (J_r^T J_r) dU + (J_r^T r)^T dU
            # J_r = dr/dU: the state part is sqrt(Q) Gamma, the command parts
            # are constant (sqrt(w_u) I and sqrt(w_du) D)
            Gamma = sensitivity(self.jacobians[mode], X, U)
            J_r = np.vstack([sq[:, None] * Gamma, sq_u * np.eye(NU * N), sq_du * self.D])
            dU = self._qp_step(J_r.T @ J_r, J_r.T @ residual(X, U), Gamma, X, U)
            return None if dU is None else (dU, None)

        # warm start: the previous plan shifted by one step (the last command repeated)
        U = np.zeros((N, NU)) if self._plan is None else np.vstack([self._plan[1:], self._plan[-1:]])
        X = self.rollout(mode, x0, U)
        # the shared SQP loop (sqp): QP step + line search on the nonlinear cost
        U, X, _, _, _ = sqp(U, X, lambda U_, X_: cost(X_, U_), qp_step,
                            lambda U_: self.rollout(mode, x0, U_), self.sqp_iterations,
                            clip=(-self.u_max, self.u_max))
        self._plan = U
        return X.T, U.T, {"solve_time": time.perf_counter() - t0}

    def _qp_step(self, H, g, Gamma, X, U):
        """The step dU (N, NU) of  min 0.5 dU^T H dU + g^T dU  subject to the
        command bounds and the linearized tilt limit. None if the QP fails.

        The variables are the step dU, so the bounds are shifted by the
        current plan: -u_max <= U + dU <= u_max, and the linearized tilt
        tilt(U + dU) ~ tilt(U) + Gamma_tilt dU must stay within +-max_tilt.
        If the tilt limit cannot be met (e.g. the drone already tilts beyond
        it), it is dropped; the line search cost still penalizes the tilt.
        """
        u, tilt = U.ravel(), X[1:, TILT].ravel()
        box = {"h": H, "g": g, "lbx": -self.u_max - u, "ubx": self.u_max - u}
        rows = Gamma.reshape(self.N, NX, -1)[:, TILT].reshape(2 * self.N, -1)   # tilt rows of Gamma
        sol = self.qp(a=rows, lba=-self.max_tilt - tilt, uba=self.max_tilt - tilt, **box)
        if not self.qp.stats()["success"]:
            sol = self.qp_box(**box)
            if not self.qp_box.stats()["success"]:
                return None
        dU = np.asarray(sol["x"]).reshape(self.N, NU)
        return dU if np.all(np.isfinite(dU)) else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the ResNet model of the director MPC")
    parser.add_argument("--collect", action="store_true",
                        help="collect new training runs instead of reusing the cached ones")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out", default=MODEL_FILE)
    a = parser.parse_args()
    train(a.out, epochs=a.epochs, reuse_data=not a.collect)
