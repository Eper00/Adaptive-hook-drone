"""Director MPC with a ResNet-identified prediction model.

The model, its training and the MPC (docs/director_mpc_safety_filter.md,
chapters 3 and 7). The same model is the prediction model of the safety
filter (predictive_safety_filter.py), whose stall takeover is this MPC.

  * the network is one PyTorch module (``ResNetDynamics``). PyTorch trains
    it, and the MPC and the safety filter use the same module for the
    rollouts and for the Jacobians (``torch.func``),
  * the Gauss-Newton QP is written as a least-squares problem: cost
    0.5 |r(U)|^2, QP Hessian J_r^T J_r and gradient J_r^T r,
  * the QP is solved with DAQP (dense active-set solver, through CasADi's
    ``conic`` interface).

Method:

    x_k+1 = x_k + f(x_k, u_k)       f: ResNet, one network per mode

    min_U  0.5 sum_k |x_k - r_k|_Q^2 + 0.5 w_u sum_k |u_k|^2 + 0.5 w_du sum_k |u_k - u_k-1|^2
    s.t.   |u_k| <= u_max,   |roll_k|, |pitch_k| <= max_tilt

solved every control step by ``sqp_iterations`` Gauss-Newton SQP iterations
(single shooting): linearize the model along the current plan, solve the QP
for the step dU, backtracking line search on the nonlinear cost. The plan is
warm-started with the shifted previous plan, and a velocity offset d is
estimated online (offset-free MPC).

The training data are P-controller identification runs, strongly excited
runs and flights of the RL director policy; steps where the hook touches the
payload before it is lifted (forces the model cannot know) are left out.

Train / retrain (after a new velocity policy or physics changes):
    python -m multi_drone_mujoco.envs.resnet_mpc              # train on the cached runs
    python -m multi_drone_mujoco.envs.resnet_mpc --collect    # collect new runs first
Use:
    ResNetMPCAgent(env)       (play.py: --env_type adaptive_director_resnet_MPC)
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
    ATT, C, NU, NX, P, TENDON_HOLD, TH, THD, U_MAX, V,
    DirectorWeights, StateTracker, collect, director_step, make_identification_env,
)
from multi_drone_mujoco.envs.mpc_mission import CONTACT_PHASES, MissionAgent, payload_attached

MODEL_FILE = "results/mpc_director/resnet_model.pt"
LEGACY_MODEL_FILE = "results/mpc_director/resnet_model.npz"   # earlier format, still readable
DATA_FILE = "results/mpc_director/resnet_training_data.pkl"   # cached training runs
DIRECTOR_POLICY = "results/final/rl_adaptive_director_curriculum/final_model.zip"
N_IN = (NX - 3) + NU        # network input: every state except the position, plus the command
N_OUT = 7                   # dv (3), dtheta_dot (2), dtilt (2)


################################################################################
# MODEL
################################################################################

class ResNetDynamics(nn.Module):
    """x_k+1 = x_k + f(x_k, u_k) of one mode (0: no payload, 1: payload attached).

    The network predicts the velocity, swing-rate and tilt increments; the
    rest of the state update is written out (position, the env's command
    low-pass filter, the history entries). Works on batches (..., NX).
    """

    def __init__(self, dt, alpha, with_swing, hidden=128, blocks=2):
        super().__init__()
        self.dt, self.alpha, self.with_swing = dt, alpha, with_swing
        # input normalization and output scale (set from the training data)
        self.register_buffer("mu", torch.zeros(N_IN))
        self.register_buffer("sd", torch.ones(N_IN))
        self.register_buffer("out_scale", torch.ones(N_OUT))
        self.inp = nn.Linear(N_IN, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
            for _ in range(blocks))
        self.out = nn.Linear(hidden, N_OUT)
        self.skip = nn.Linear(N_IN, N_OUT, bias=False)   # linear shortcut input -> output
        for p in (self.out.weight, self.out.bias, self.skip.weight):
            nn.init.zeros_(p)                            # start at x_k+1 = x_k
        self.double()

    def network(self, z):
        zn = (z - self.mu) / self.sd
        h = torch.tanh(self.inp(zn))
        for block in self.blocks:
            h = h + block(h)                             # residual blocks
        return (self.out(h) + self.skip(zn)) * self.out_scale

    def forward(self, x, u, d=None):
        """Next state; ``d`` is the velocity-offset estimate of the MPC."""
        r = self.network(torch.cat([x[..., 3:], u], dim=-1))
        dv = r[..., 0:3] if d is None else r[..., 0:3] + d
        v, c, th, thd, tilt = x[..., V], x[..., C], x[..., TH], x[..., THD], x[..., ATT]
        if self.with_swing:
            thd_next = thd + r[..., 3:5]
            th_next = th + self.dt * thd_next
        else:                                            # no payload: swing states stay 0
            th_next, thd_next = th, thd
        return torch.cat([x[..., P] + self.dt * (v + 0.5 * dv),   # p (trapezoidal)
                          v + dv,                                 # v
                          v,                                      # v_prev
                          c + self.alpha * (u - c),               # c (env low-pass filter)
                          c,                                      # c_prev
                          th_next, thd_next,                      # theta, theta_dot
                          tilt + r[..., 5:7],                     # roll, pitch
                          tilt], dim=-1)                          # roll_prev, pitch_prev

    def rollout(self, x0, U, dist=None):
        """numpy interface like ``DirectorModel.rollout`` (analysis scripts):
        x0 (NX,), U (NU, N) -> X (NX, N+1)."""
        t = lambda a: torch.as_tensor(np.asarray(a, float))
        with torch.inference_mode():
            X = rollout(self, t(x0), t(np.asarray(U).T), None if dist is None else t(dist))
        return np.hstack([np.asarray(x0, float)[:, None], X.numpy().T])


def rollout(model, x0, U, d=None):
    """States x_1..x_H for the commands U (..., H, NU) from x0 (..., NX)."""
    xs, x = [], x0
    for k in range(U.shape[-2]):
        x = model(x, U[..., k, :], d)
        xs.append(x)
    return torch.stack(xs, dim=-2)


def step_jacobians(model):
    """Function (X (N, NX), U (N, NU), d (3,)) -> (A (N, NX, NX), B (N, NX, NU)):
    A_k = df/dx and B_k = df/du of the model at every step, in one call."""
    jac = vmap(jacrev(model, argnums=(0, 1)), in_dims=(0, 0, None))
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64)
    jac(zeros(1, NX), zeros(1, NU), zeros(3))   # the first torch.func call of a process
    return jac                                  # takes ~0.6 s: not in a control step


def save_model(path, models):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"dt": models[0].dt, "alpha": models[0].alpha,
                "hidden": models[0].inp.out_features, "blocks": len(models[0].blocks),
                "state": [m.state_dict() for m in models]}, path)


def load_model(path=MODEL_FILE):
    """[mode 0 model, mode 1 model]; reads this module's .pt files and the
    earlier .npz format (same network)."""
    if path.endswith(".npz"):
        return _load_legacy(path)
    ckpt = torch.load(path, weights_only=True)
    models = []
    for mode, state in enumerate(ckpt["state"]):
        m = ResNetDynamics(ckpt["dt"], ckpt["alpha"], bool(mode), ckpt["hidden"], ckpt["blocks"])
        m.load_state_dict(state)
        models.append(m)
    return models


def load_or_train(path=MODEL_FILE):
    """Load the model; train it first if the file is missing."""
    if not os.path.exists(path):
        print(f"[ResNet model] {path} not found -> training it now (several minutes)...")
        return train(path)
    return load_model(path)


def _load_legacy(path):
    d = np.load(path)
    models = []
    for mode in (0, 1):
        g = lambda k: torch.tensor(d[f"m{mode}_{k}"])
        state = {"mu": g("mu"), "sd": g("sd"), "out_scale": g("out_scale"),
                 "inp.weight": g("W_in"), "inp.bias": g("b_in"),
                 "out.weight": g("W_out"), "out.bias": g("b_out"), "skip.weight": g("W_skip")}
        n_blocks = int(d["n_blocks"])
        for j in range(n_blocks):
            for layer, (W, b) in ((0, ("W1", "b1")), (2, ("W2", "b2"))):
                state[f"blocks.{j}.{layer}.weight"] = g(f"blk{j}_{W}")
                state[f"blocks.{j}.{layer}.bias"] = g(f"blk{j}_{b}")
        m = ResNetDynamics(float(d["dt"]), float(d["alpha"]), bool(mode),
                           hidden=state["inp.weight"].shape[0], blocks=n_blocks)
        m.load_state_dict(state)
        models.append(m)
    return models


################################################################################
# TRAINING DATA
################################################################################

# training / validation runs (seeds differ from the ones analyse_identification tests on)
P_TRAIN = [(s, 0.15) for s in (100, 101, 102, 103)] + [(s, 0.6) for s in (104, 105, 106, 107)]
P_VAL = [(108, 0.4)]
RL_TRAIN = tuple(range(300, 324))
RL_VAL = (400, 401, 402, 403)
RL_DURATION = 15.0


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
                                         inc[:, ATT].std(0)]) + 1e-6)
        # loss: velocity and tilt (+ swing), each channel scaled by its spread
        ch = [3, 4, 5, 19, 20] + ([15, 16, 17, 18] if mode else [])
        w = 1.0 / (y.reshape(-1, NX)[:, ch].std(0) + 1e-3) ** 2

        def loss_fn(x0_, u_, y_, H):
            pred = rollout(model, x0_, u_[:, :H])
            return ((pred[..., ch] - y_[:, :H, ch]) ** 2 * w).mean()

        loader = DataLoader(data, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state, t0 = np.inf, None, time.time()
        for ep in range(epochs):
            H = max(2, min(horizon, int(horizon * (ep + 1) / (0.4 * epochs))))
            for x0_, u_, y_ in loader:
                opt.zero_grad()
                loss = loss_fn(x0_, u_, y_, H)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
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
    """Gauss-Newton SQP MPC on the ResNet model (single shooting, condensed QP).

    Same interface as ``DirectorMPC``: ``solve(mode, x0, p_ref, v_ref,
    u_prev)`` -> (X [NX, N+1], U [NU, N], info).
    """

    TILT_PENALTY = 1e4      # tilt-limit violation in the line search cost

    def __init__(self, models, horizon=40, weights=None, u_max=U_MAX, max_tilt=np.pi / 2,
                 disturbance_gain=0.05, sqp_iterations=5, qp_solver="daqp", max_threads=4):
        # small matrices: more than ~4 threads only add overhead; a lower limit
        # of the caller (e.g. OMP_NUM_THREADS=1 in parallel workers) is kept
        torch.set_num_threads(min(max_threads, torch.get_num_threads()))
        self.models = [m.requires_grad_(False) for m in models]
        self.jacobians = [step_jacobians(m) for m in self.models]
        self.N = N = horizon
        self.w = weights or DirectorWeights()
        self.u_max = u_max
        self.max_tilt = max_tilt
        self.dist_gain = disturbance_gain
        self.sqp_iterations = sqp_iterations
        nz = NU * N
        self.D = np.eye(nz) - np.eye(nz, k=-NU)          # D U = (u_0, u_1 - u_0, ...)
        self.tilt_rows = (np.arange(N)[:, None] * NX + np.array([19, 20])).ravel()
        dense = {"h": ca.Sparsity.dense(nz, nz)}
        opts = {"error_on_fail": False}
        self.qp = ca.conic("qp", qp_solver, dict(dense, a=ca.Sparsity.dense(2 * N, nz)), opts)
        self.qp_box = ca.conic("qp_box", qp_solver, dict(dense, a=ca.Sparsity(0, nz)), opts)
        self.reset()

    def reset(self):
        self.dist = np.zeros(3)
        self._last = None             # (mode, x, u) of the previous step
        self._plan = None             # previous plan (N, NU)

    def _sqrt_q(self, mode):
        """Square root of the state weights over the horizon (N * NX,)."""
        q = np.zeros(NX)
        q[P] = self.w.pos
        q[V] = self.w.vel
        if mode == 1:
            q[TH] = self.w.swing
            q[THD] = self.w.swing_rate
        Q = np.tile(q, self.N)
        Q[-NX:] *= self.w.terminal
        return np.sqrt(Q)

    def rollout(self, mode, x0, U):
        """Prediction (N+1, NX) for the plan U (N, NU)."""
        with torch.inference_mode():
            X = rollout(self.models[mode], torch.from_numpy(x0), torch.from_numpy(U),
                        torch.from_numpy(self.dist))
        return np.vstack([x0, X.numpy()])

    def sensitivity(self, mode, X, U):
        """Gamma = dX_1..N / dU (N * NX, N * NU) of the linearized model
        dx_k+1 = A_k dx_k + B_k du_k along (X, U), with dx_0 = 0."""
        A, B = self.jacobians[mode](torch.from_numpy(X[:-1]), torch.from_numpy(U),
                                    torch.from_numpy(self.dist))
        A, B = A.numpy(), B.numpy()
        G, rows = np.zeros((NX, NU * self.N)), []
        for k in range(self.N):
            G = A[k] @ G
            G[:, NU * k:NU * (k + 1)] += B[k]
            rows.append(G)
        return np.vstack(rows)

    def _update_disturbance(self, mode, x):
        """d <- d + gain (measured - predicted velocity) of the last step."""
        if self._last is None or self._last[0] != mode:
            return
        _, x_prev, u_prev = self._last
        with torch.inference_mode():
            pred = self.models[mode](torch.from_numpy(x_prev), torch.from_numpy(u_prev),
                                     torch.from_numpy(self.dist)).numpy()
        self.dist += self.dist_gain * (x[V] - pred[V])

    def solve(self, mode, x0, p_ref, v_ref, u_prev):
        t0 = time.perf_counter()
        N = self.N
        x0 = np.asarray(x0, float)
        self._update_disturbance(mode, x0)
        ref = np.zeros((N, NX))
        ref[:, P], ref[:, V] = p_ref[:, 1:].T, v_ref[:, 1:].T
        e_prev = np.zeros(NU * N)
        e_prev[:NU] = u_prev
        sq_q, sq_u, sq_du = self._sqrt_q(mode), np.sqrt(self.w.u), np.sqrt(self.w.du)

        def residual(X, U):
            """Least-squares residual: the cost is 0.5 |r|^2."""
            return np.r_[sq_q * (X[1:] - ref).ravel(), sq_u * U.ravel(),
                         sq_du * (self.D @ U.ravel() - e_prev)]

        def cost(X, U):
            r = residual(X, U)
            viol = np.maximum(np.abs(X[1:, ATT]) - self.max_tilt, 0.0).sum()
            return 0.5 * r @ r + self.TILT_PENALTY * viol

        # warm start: the previous plan shifted by one step
        U = np.zeros((N, NU)) if self._plan is None else np.vstack([self._plan[1:], self._plan[-1:]])
        X = self.rollout(mode, x0, U)
        J = cost(X, U)
        qp_failures = 0
        for _ in range(self.sqp_iterations):
            # Gauss-Newton: linearize the model, keep the cost quadratic
            Gamma = self.sensitivity(mode, X, U)
            J_r = np.vstack([sq_q[:, None] * Gamma, sq_u * np.eye(NU * N), sq_du * self.D])
            dU = self._qp_step(J_r.T @ J_r, J_r.T @ residual(X, U), Gamma, X, U)
            if dU is None:
                qp_failures += 1
                break
            # backtracking line search on the nonlinear cost
            for step in (1.0, 0.5, 0.25):
                U_try = np.clip(U + step * dU, -self.u_max, self.u_max)
                X_try = self.rollout(mode, x0, U_try)
                J_try = cost(X_try, U_try)
                if J_try < J:
                    U, X, J = U_try, X_try, J_try
                    break
            else:
                break                                    # no improvement: stop
        self._plan = U
        self._last = (mode, x0.copy(), U[0].copy())
        return X.T, U.T, {"solve_time": time.perf_counter() - t0, "qp_failures": qp_failures,
                          "max_tilt_pred": float(np.abs(X[1:, ATT]).max())}

    def _qp_step(self, H, g, Gamma, X, U):
        """QP for the step dU: min 0.5 dU^T H dU + g^T dU subject to the
        linearized tilt limit and the command bounds. If the tilt limit
        cannot be met (e.g. the drone already tilts beyond it), it is
        dropped. Returns None if the QP fails."""
        u = U.ravel()
        box = {"h": H, "g": g, "lbx": -self.u_max - u, "ubx": self.u_max - u}
        tilt = X[1:, ATT].ravel()
        sol = self.qp(a=Gamma[self.tilt_rows], lba=-self.max_tilt - tilt,
                      uba=self.max_tilt - tilt, **box)
        if not self.qp.stats()["success"]:
            sol = self.qp_box(**box)
            if not self.qp_box.stats()["success"]:
                return None
        dU = np.asarray(sol["x"]).ravel()
        return dU.reshape(self.N, NU) if np.all(np.isfinite(dU)) else None


################################################################################
# AGENT
################################################################################

class ResNetMPCAgent(MissionAgent):
    """Flies the pick-and-place mission with ``ResNetMPC`` on top of the PPO
    velocity controller (use like an RL policy, see MissionAgent)."""

    tendon_hold = TENDON_HOLD

    def __init__(self, env, model_file=MODEL_FILE, horizon=40, max_time=45.0, verbose=False,
                 max_tilt=np.pi / 2):
        super().__init__(env, horizon, max_time, verbose)
        self.models = load_or_train(model_file)
        self.mpc = ResNetMPC(self.models, horizon=horizon, max_tilt=max_tilt)

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
    parser = argparse.ArgumentParser(description="Train the ResNet model of the director MPC")
    parser.add_argument("--collect", action="store_true",
                        help="collect new training runs instead of reusing the cached ones")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out", default=MODEL_FILE)
    a = parser.parse_args()
    train(a.out, epochs=a.epochs, reuse_data=not a.collect)
