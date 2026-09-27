"""Hybrid linear MPC on top of the RL velocity controller (director level).

``AdaptiveTransportDirectorAviary`` takes velocity commands, low-pass filters
them (c_k+1 = alpha a_k + (1 - alpha) c_k) and feeds them to a PPO velocity
controller that produces the rotor commands. The MPC therefore plans velocity
commands, and its prediction model is the closed loop

    command -> known low-pass filter -> identified RL velocity loop
            -> position,  (+ payload swing driven by the drone acceleration)

The RL velocity loop is identified per axis as an ARX model

    v_k+1 = a1 v_k + a2 v_k-1 + b0 c_k+1 + b1 c_k + b2 c_k-1 + e theta_k + d

and, when the payload hangs on the hook, the swing angle of each horizontal
axis (payload direction in the world x-z / y-z plane) as a pendulum whose
pivot accelerates with the drone

    theta_dot_k+1 = theta_dot_k + dt (-w2 theta_k - c theta_dot_k) - g (v_k+1 - v_k)
    theta_k+1     = theta_k + dt theta_dot_k+1.

The drone attitude produced by the RL controller is predicted as well (yaw
is held at 0 by the RL controller, so the x axis tilts through pitch and the
y axis through roll):

    tilt_k+1 = p1 tilt_k + p2 tilt_k-1 + q0 c_k+1 + q1 c_k + q2 c_k-1
               + r1 v_k + r2 v_k-1 + e theta_k + d

and the MPC constrains |roll|, |pitch| <= max_tilt over the horizon.

Two parameter sets are identified (mode 0: no payload, mode 1: payload
attached) and the MPC switches between them like the rotor-level HybridMPC.

State (23):  [p(3), v(3), v_prev(3), c(3), c_prev(3), theta(2), theta_dot(2),
              roll, pitch, roll_prev, pitch_prev]
Input (3):   velocity command a in [-1, 1]^3 (env action[0:3]).
"""

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.mpc_mission import MissionAgent

NX = 23
NU = 3
P, V, VP, C, CP, TH, THD, ATT, ATTP = (
    slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15),
    slice(15, 17), slice(17, 19), slice(19, 21), slice(21, 23))
# State index of the tilt driven by each horizontal axis: x -> pitch, y -> roll
TILT_OF_AXIS = {0: 20, 1: 19}


################################################################################
# PLANT INTERFACE
################################################################################

def director_step(env, command, tendon=(0.0, 0.0)):
    """One step of AdaptiveTransportDirectorAviary with direct tendon control.

    Mirrors ``AdaptiveTransportDirectorAviary.step`` (low-pass filter on the
    action, PPO velocity controller, target yaw 0) but calls
    ``BaseAviary.step`` at the end, so the tendon command is applied as given
    instead of being gated by the transport task's waypoint/GRAB_FLAG logic.
    """
    action = np.hstack([command, 0.0, tendon]).astype(float)
    if env.prev_action is None:
        env.prev_action = action
    else:
        action = env.alpha * action + (1.0 - env.alpha) * env.prev_action
        env.prev_action = action.copy()
    obs = env._get_low_level_obs(action[0:3], env.TARGET_ORIENTATION)
    low_level, _ = env.controller_model.predict(obs, deterministic=True)
    return BaseAviary.step(env, np.hstack([low_level[0:4], action[-2:]]))


def filter_state(env):
    """Current output of the director's low-pass filter (what the RL tracks)."""
    return np.zeros(3) if env.prev_action is None else env.prev_action[0:3].copy()


def measure_swing(env):
    """World-frame swing angles of the payload and their rates.

    theta_x / theta_y are the angles of the pivot->payload direction in the
    world x-z / y-z planes (0 = hanging straight down).
    """
    model, data = env.model, env.data
    target = model.body("target").id
    rot_p = data.xmat[target].reshape(3, 3)
    dof = model.jnt_dofadr[env.target_joint_id]
    omega = rot_p @ data.qvel[dof + 3:dof + 6]          # free joint: local frame
    u = -rot_p[:, 2]
    u_dot = np.cross(omega, u)
    th = np.array([np.arctan2(u[0], -u[2]), np.arctan2(u[1], -u[2])])
    den_x = u[0] ** 2 + u[2] ** 2
    den_y = u[1] ** 2 + u[2] ** 2
    th_dot = np.array([(-u[2] * u_dot[0] + u[0] * u_dot[2]) / max(den_x, 1e-9),
                       (-u[2] * u_dot[1] + u[1] * u_dot[2]) / max(den_y, 1e-9)])
    return th, th_dot


class StateTracker:
    """Builds the 23-dim MPC state from MuJoCo measurements + input history."""

    def __init__(self, env):
        self.env = env
        self.v_prev = env.vel[0].copy()
        self.c_prev = filter_state(env)
        self.att_prev = env.rpy[0, 0:2].copy()

    def measure(self, attached):
        env = self.env
        x = np.zeros(NX)
        x[P] = env.pos[0]
        x[V] = env.vel[0]
        x[VP] = self.v_prev
        x[C] = filter_state(env)
        x[CP] = self.c_prev
        if attached:
            x[TH], x[THD] = measure_swing(env)
        x[ATT] = env.rpy[0, 0:2]
        x[ATTP] = self.att_prev
        return x

    def advance(self, x):
        """Call after measuring, before stepping the plant."""
        self.v_prev = x[V].copy()
        self.c_prev = x[C].copy()
        self.att_prev = x[ATT].copy()


################################################################################
# IDENTIFIED MODEL
################################################################################

@dataclass
class AxisModel:
    """ARX velocity loop of one axis (+ swing coupling and pendulum)."""
    a1: float
    a2: float
    b0: float
    b1: float
    b2: float
    e: float = 0.0        # swing -> drone velocity coupling
    d: float = 0.0        # constant velocity offset
    w2: float = 0.0       # pendulum stiffness  (g / L_eff)
    cd: float = 0.0       # pendulum damping
    g: float = 0.0        # pivot acceleration gain (1 / L_eff)

    def vector(self):
        return np.array([self.a1, self.a2, self.b0, self.b1, self.b2, self.e,
                         self.d, self.w2, self.cd, self.g])

    @classmethod
    def from_vector(cls, v):
        return cls(*[float(a) for a in v])


def fit_axis(v, c, dt, theta=None, theta_dot=None, mask=None):
    """Least-squares fit of one axis from logged sequences.

    v, c: (K,) measured velocity and filter output (c[k] = filter state at step
    k, i.e. c[k+1] depends on the command applied at step k).
    theta, theta_dot: (K,) swing of this axis (None for no payload / z).
    mask: (K,) bool, samples usable as regression targets.
    """
    K = len(v)
    idx = np.arange(2, K - 1)
    if mask is not None:
        idx = idx[mask[idx] & mask[idx - 1] & mask[idx - 2] & mask[idx + 1]]
    cols = [v[idx], v[idx - 1], c[idx + 1], c[idx], c[idx - 1]]
    if theta is not None:
        cols.append(theta[idx])
    cols.append(np.ones(len(idx)))
    Phi = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Phi, v[idx + 1], rcond=None)
    if theta is None:
        model = AxisModel(*beta[:5], e=0.0, d=beta[5])
    else:
        model = AxisModel(*beta[:5], e=beta[5], d=beta[6])
        # Pendulum: theta_dot_k+1 - theta_dot_k = dt(-w2 th - c thd) - g dv
        dthd = theta_dot[idx + 1] - theta_dot[idx]
        Psi = np.column_stack([-dt * theta[idx], -dt * theta_dot[idx],
                               -(v[idx + 1] - v[idx])])
        gamma, *_ = np.linalg.lstsq(Psi, dthd, rcond=None)
        model.w2, model.cd, model.g = gamma
    return model


def simulate_axis(params, v, c, theta, theta_dot, starts, length, dt):
    """Multi-step simulation of one axis from measured initial conditions.

    Returns simulated v (and theta) for windows [s, s+length] (s in starts),
    driven by the measured filter output c.
    """
    a1, a2, b0, b1, b2, e, d, w2, cd, g = params
    n = len(starts)
    vs = np.zeros((n, length + 1))
    ths = np.zeros((n, length + 1))
    vk, vkm = v[starts], v[starts - 1]
    th = theta[starts] if theta is not None else np.zeros(n)
    thd = theta_dot[starts] if theta is not None else np.zeros(n)
    vs[:, 0], ths[:, 0] = vk, th
    for j in range(length):
        k = starts + j
        v_next = a1 * vk + a2 * vkm + b0 * c[k + 1] + b1 * c[k] + b2 * c[k - 1] + e * th + d
        if theta is not None:
            thd = thd + dt * (-w2 * th - cd * thd) - g * (v_next - vk)
            th = th + dt * thd
        vkm, vk = vk, v_next
        vs[:, j + 1], ths[:, j + 1] = vk, th
    return vs, ths


def refine_axis(model, v, c, dt, theta=None, theta_dot=None, mask=None,
                length=24, stride=3, theta_weight=0.3):
    """Output-error refinement: minimize the multi-step simulation error."""
    from scipy.optimize import least_squares

    K = len(v)
    starts = np.arange(2, K - length - 1, stride)
    if mask is not None:
        ok = np.array([mask[s - 2:s + length + 2].all() for s in starts])
        starts = starts[ok]
    idx = starts[:, None] + np.arange(length + 1)[None]
    swing = theta is not None
    free = [0, 1, 2, 3, 4, 6] + ([5, 7, 8, 9] if swing else [])
    x0 = model.vector()

    def residual(z):
        params = x0.copy()
        params[free] = z
        vs, ths = simulate_axis(params, v, c, theta, theta_dot, starts, length, dt)
        r = [(vs - v[idx]).ravel()]
        if swing:
            r.append(theta_weight * (ths - theta[idx]).ravel())
        return np.concatenate(r)

    sol = least_squares(residual, x0[free], method="trf", max_nfev=200)
    params = x0.copy()
    params[free] = sol.x
    return AxisModel.from_vector(params), np.sqrt(np.mean(sol.fun ** 2))


@dataclass
class TiltModel:
    """ARX model of the tilt (pitch for x, roll for y) the RL loop produces."""
    p1: float = 0.0
    p2: float = 0.0
    q0: float = 0.0
    q1: float = 0.0
    q2: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    e: float = 0.0        # swing -> tilt coupling (payload mode)
    d: float = 0.0

    def vector(self):
        return np.array([self.p1, self.p2, self.q0, self.q1, self.q2,
                         self.r1, self.r2, self.e, self.d])

    @classmethod
    def from_vector(cls, v):
        return cls(*[float(a) for a in v])


def _tilt_regressors(tilt, c, v, theta, idx):
    cols = [tilt[idx], tilt[idx - 1], c[idx + 1], c[idx], c[idx - 1], v[idx], v[idx - 1]]
    cols.append(theta[idx] if theta is not None else np.zeros(len(idx)))
    cols.append(np.ones(len(idx)))
    return np.column_stack(cols)


def fit_tilt(tilt, c, v, theta=None, mask=None, length=24, stride=3):
    """Least squares + output-error refinement of a TiltModel.

    The multi-step simulation recurses on the tilt only (measured v, c and
    theta drive it), which is how the model is used inside the MPC.
    """
    from scipy.optimize import least_squares

    K = len(tilt)
    idx = np.arange(2, K - 1)
    if mask is not None:
        idx = idx[mask[idx] & mask[idx - 1] & mask[idx - 2] & mask[idx + 1]]
    Phi = _tilt_regressors(tilt, c, v, theta, idx)
    free = [0, 1, 2, 3, 4, 5, 6, 8] + ([7] if theta is not None else [])
    beta, *_ = np.linalg.lstsq(Phi[:, free], tilt[idx + 1], rcond=None)
    x0 = np.zeros(9)
    x0[free] = beta

    starts = np.arange(2, K - length - 1, stride)
    if mask is not None:
        starts = starts[[mask[s - 2:s + length + 2].all() for s in starts]]
    win = starts[:, None] + np.arange(length + 1)[None]
    th = theta if theta is not None else np.zeros(K)

    def simulate(params):
        p1, p2, q0, q1, q2, r1, r2, e, d = params
        tk, tkm = tilt[starts], tilt[starts - 1]
        out = np.zeros((len(starts), length + 1))
        out[:, 0] = tk
        for j in range(length):
            k = starts + j
            t_next = (p1 * tk + p2 * tkm + q0 * c[k + 1] + q1 * c[k] + q2 * c[k - 1]
                      + r1 * v[k] + r2 * v[k - 1] + e * th[k] + d)
            tkm, tk = tk, t_next
            out[:, j + 1] = tk
        return out

    def residual(z):
        params = x0.copy()
        params[free] = z
        return (simulate(params) - tilt[win]).ravel()

    sol = least_squares(residual, x0[free], method="trf", max_nfev=200)
    params = x0.copy()
    params[free] = sol.x
    return TiltModel.from_vector(params), np.sqrt(np.mean(sol.fun ** 2))


class DirectorModel:
    """Linear prediction model x+ = A x + B u + f of one mode."""

    def __init__(self, axes, tilts, alpha, dt, with_swing):
        self.axes = axes              # [x, y, z] AxisModel
        self.tilts = tilts            # [x -> pitch, y -> roll] TiltModel
        self.alpha = alpha
        self.dt = dt
        self.with_swing = with_swing
        self.A, self.B, self.f = self._build()

    def _build(self):
        al, dt = self.alpha, self.dt
        A = np.zeros((NX, NX))
        B = np.zeros((NX, NU))
        f = np.zeros(NX)
        for i, m in enumerate(self.axes):
            p, v, vp, c, cp = i, 3 + i, 6 + i, 9 + i, 12 + i
            # filter: c+ = al u + (1 - al) c ;  c_prev+ = c
            A[c, c] = 1 - al
            B[c, i] = al
            A[cp, c] = 1.0
            # velocity: v+ = a1 v + a2 v_prev + b0 c+ + b1 c + b2 c_prev + e th + d
            A[v, v] = m.a1
            A[v, vp] = m.a2
            A[v, c] = m.b0 * (1 - al) + m.b1
            A[v, cp] = m.b2
            B[v, i] = m.b0 * al
            f[v] = m.d
            A[vp, v] = 1.0
            if self.with_swing and i < 2:
                th, thd = 15 + i, 17 + i
                A[v, th] = m.e
                # theta_dot+ = thd + dt(-w2 th - cd thd) - g (v+ - v)
                A[thd] = -m.g * A[v]
                A[thd, v] += m.g
                A[thd, th] += -dt * m.w2
                A[thd, thd] += 1 - dt * m.cd
                B[thd] = -m.g * B[v]
                f[thd] = -m.g * f[v]
                # theta+ = th + dt theta_dot+
                A[th] = dt * A[thd]
                A[th, th] += 1.0
                B[th] = dt * B[thd]
                f[th] = dt * f[thd]
            if i < 2:
                # tilt+ = p1 t + p2 t_prev + q0 c+ + q1 c + q2 c_prev + r1 v + r2 v_prev + e th + d
                tm = self.tilts[i]
                t = TILT_OF_AXIS[i]
                tp = t + 2
                A[t, t] = tm.p1
                A[t, tp] = tm.p2
                A[t, c] = tm.q0 * (1 - al) + tm.q1
                A[t, cp] = tm.q2
                B[t, i] = tm.q0 * al
                A[t, v] += tm.r1
                A[t, vp] += tm.r2
                if self.with_swing:
                    A[t, 15 + i] += tm.e
                f[t] = tm.d
                A[tp, t] = 1.0
            # position (trapezoidal): p+ = p + dt/2 (v + v+)
            A[p] = 0.5 * dt * A[v]
            A[p, p] += 1.0
            A[p, v] += 0.5 * dt
            B[p] = 0.5 * dt * B[v]
            f[p] = 0.5 * dt * f[v]
        return A, B, f

    def step(self, x, u, dist=None):
        x_next = self.A @ x + self.B @ u + self.f
        if dist is not None:
            x_next += self.disturbance_map() @ dist
        return x_next

    def disturbance_map(self):
        """How a velocity offset disturbance d (3) enters the state update."""
        E = np.zeros((NX, 3))
        for i in range(3):
            E[3 + i, i] = 1.0
            E[i, i] = 0.5 * self.dt
            if self.with_swing and i < 2:
                g = self.axes[i].g
                E[17 + i, i] = -g
                E[15 + i, i] = -self.dt * g
        return E

    def rollout(self, x0, U, dist=None):
        X = np.zeros((NX, U.shape[1] + 1))
        X[:, 0] = x0
        for k in range(U.shape[1]):
            X[:, k + 1] = self.step(X[:, k], U[:, k], dist)
        return X


def save_models(path, axes, tilts, alpha, dt):
    """axes[mode] = [x, y, z] AxisModel, tilts[mode] = [x, y] TiltModel."""
    np.savez(path, alpha=alpha, dt=dt,
             **{f"mode{m}": np.array([a.vector() for a in axes[m]]) for m in (0, 1)},
             **{f"tilt{m}": np.array([t.vector() for t in tilts[m]]) for m in (0, 1)})


def load_models(path):
    d = np.load(path)
    if "tilt0" not in d.files:
        raise ValueError(f"{path} has no attitude model (identified before roll/pitch "
                         "prediction was added): re-identify with "
                         "'python -m multi_drone_mujoco.envs.director_mpc'")
    alpha, dt = float(d["alpha"]), float(d["dt"])
    return [DirectorModel([AxisModel.from_vector(v) for v in d[f"mode{m}"]],
                          [TiltModel.from_vector(v) for v in d[f"tilt{m}"]],
                          alpha, dt, with_swing=bool(m)) for m in (0, 1)]


################################################################################
# MPC
################################################################################

@dataclass
class DirectorWeights:
    pos: tuple = (40.0, 40.0, 60.0)
    vel: float = 2.0
    swing: float = 5.0
    swing_rate: float = 0.5
    u: float = 0.05
    du: float = 30.0       # smooth commands: the RL loop + payload oscillates otherwise
    terminal: float = 5.0


class DirectorMPC:
    """Linear MPC on velocity commands with a mode-switched identified model.

    The model is linear, so the condensed QP Hessian of each mode is constant
    and only the gradient changes between steps (one qpOASES call per step).
    A velocity-offset disturbance is estimated online from one-step prediction
    errors (offset-free MPC), which removes the small steady-state biases of
    the RL velocity controller. The predicted roll and pitch are constrained
    to |angle| <= max_tilt; if that QP is infeasible the unconstrained one is
    solved instead.
    """

    def __init__(self, models, horizon=40, weights=None, u_max=0.8,
                 disturbance_gain=0.05, max_tilt=np.pi / 2):
        self.models = models
        self.max_tilt = max_tilt
        self.N = horizon
        self.w = weights or DirectorWeights()
        self.u_max = u_max
        self.dist_gain = disturbance_gain
        self.dist = np.zeros(3)
        self._last = None               # (mode, x, u) for disturbance update
        self._plan = None
        self._pre = [self._precompute(m) for m in (0, 1)]
        nz = NU * horizon
        opts = {"printLevel": "none", "error_on_fail": False}
        self._qp_con = ca.conic("qp_con", "qpoases",
                                {"h": ca.Sparsity.dense(nz, nz),
                                 "a": ca.Sparsity.dense(2 * horizon, nz)}, opts)
        self._qp = ca.conic("qp", "qpoases",
                            {"h": ca.Sparsity.dense(nz, nz), "a": ca.Sparsity(0, nz)}, opts)

    def _q(self, mode):
        q = np.zeros(NX)
        q[P] = self.w.pos
        q[V] = self.w.vel
        if mode == 1:
            q[TH] = self.w.swing
            q[THD] = self.w.swing_rate
        Q = np.tile(q, self.N)
        Q[-NX:] *= self.w.terminal
        return Q

    def _precompute(self, mode):
        model, N = self.models[mode], self.N
        A, B = model.A, model.B
        # X_1..N = Phi x0 + Gamma U + (offset terms)
        Phi = np.zeros((N * NX, NX))
        Gamma = np.zeros((N * NX, NU * N))
        Ak = np.eye(NX)
        for k in range(N):
            Ak = A @ Ak
            Phi[k * NX:(k + 1) * NX] = Ak
            for j in range(k + 1):
                Gamma[k * NX:(k + 1) * NX, j * NU:(j + 1) * NU] = (
                    np.linalg.matrix_power(A, k - j) @ B)
        # Response to the constant offset f (+ disturbance) accumulated
        S = np.zeros((N * NX, NX))
        acc = np.zeros((NX, NX))
        for k in range(N):
            acc = A @ acc + np.eye(NX)
            S[k * NX:(k + 1) * NX] = acc
        Q = self._q(mode)
        D = np.eye(NU * N) - np.eye(NU * N, k=-NU)
        H = (Gamma.T * Q) @ Gamma + self.w.u * np.eye(NU * N) + self.w.du * D.T @ D
        # rows of the stacked prediction holding roll and pitch
        tilt_rows = (np.arange(N)[:, None] * NX + np.array([19, 20])[None]).ravel()
        return dict(Phi=Phi, Gamma=Gamma, S=S, Q=Q, D=D, H=H, tilt_rows=tilt_rows)

    def reset(self):
        self.dist = np.zeros(3)
        self._last = None
        self._plan = None

    def _update_disturbance(self, mode, x):
        if self._last is None or self._last[0] != mode:
            return
        _, x_prev, u_prev = self._last
        model = self.models[mode]
        pred = model.step(x_prev, u_prev, self.dist)
        self.dist += self.dist_gain * (x[V] - pred[V])

    def solve(self, mode, x0, p_ref, v_ref, u_prev):
        """Returns (X_pred [NX, N+1], U_plan [NU, N], info)."""
        t0 = time.perf_counter()
        self._update_disturbance(mode, x0)
        pre, model, N = self._pre[mode], self.models[mode], self.N

        ref = np.zeros((NX, N))
        ref[P] = p_ref[:, 1:]
        ref[V] = v_ref[:, 1:]
        offset = model.f + model.disturbance_map() @ self.dist
        free = pre["Phi"] @ x0 + pre["S"] @ offset
        err = free - ref.T.ravel()
        e_prev = np.zeros(NU * N)
        e_prev[:NU] = u_prev
        g = pre["Gamma"].T @ (pre["Q"] * err) - self.w.du * pre["D"].T @ e_prev

        rows = pre["tilt_rows"]
        sol = self._qp_con(h=pre["H"], g=g, a=pre["Gamma"][rows],
                           lba=-self.max_tilt - free[rows], uba=self.max_tilt - free[rows],
                           lbx=-self.u_max, ubx=self.u_max)
        ok = self._qp_con.stats()["success"]
        fallback = not ok
        if fallback:
            sol = self._qp(h=pre["H"], g=g, lbx=-self.u_max, ubx=self.u_max)
            ok = self._qp.stats()["success"]
        U_flat = np.asarray(sol["x"]).ravel()
        if not ok or not np.all(np.isfinite(U_flat)):
            U = (np.hstack([self._plan[:, 1:], self._plan[:, -1:]])
                 if self._plan is not None else np.zeros((NU, N)))
        else:
            U = U_flat.reshape(N, NU).T
        self._plan = U
        X = model.rollout(x0, U, self.dist)
        self._last = (mode, x0.copy(), U[:, 0].copy())
        return X, U, {"solve_time": time.perf_counter() - t0, "qp_failures": int(not ok),
                      "constraint_fallbacks": int(fallback),
                      "max_tilt_pred": float(np.abs(X[ATT, 1:]).max())}


################################################################################
# PREDICTIVE SAFETY FILTER
################################################################################

@dataclass
class SafetyConstraints:
    """Constraints the safety filter keeps over its prediction horizon."""
    max_tilt: float = np.radians(30.0)   # |roll|, |pitch| [rad]
    max_swing: float = np.radians(30.0)  # |payload swing| per axis [rad] (with payload)
    max_speed: float = 1.0               # |v| per axis [m/s]
    min_altitude: float = 0.3            # drone height [m]


class SafetyFilter(DirectorMPC):
    """Predictive safety filter around a velocity-command controller.

    Every step the proposed command u_rl (e.g. of the RL director) is checked
    with the identified closed-loop model: it is *certified* if, applied now,
    there exists a continuation of commands (warm started with u_rl) that
    keeps all constraints over the horizon. A certified command is applied
    unchanged. Otherwise the MPC takes over and applies the first command of

        min  w0 |u_0 - u_rl|^2 + w1 sum_k |u_k - u_rl|^2 + w_du |du|^2 + rho(s)
        s.t. constraints relaxed by slacks s >= 0 (one per constraint type)

    i.e. the admissible command closest to the proposal; the slacks keep the
    problem feasible when a violation cannot be avoided any more, and are
    penalized heavily so they are only used then.
    """

    GROUPS = ("tilt", "swing", "speed", "altitude")

    def __init__(self, models, horizon=30, constraints=None, u_max=1.0,
                 disturbance_gain=0.05, w0=1.0, w1=0.1, w_du=0.5,
                 slack_weight=1e4, slack_linear=1e3):
        super().__init__(models, horizon=horizon, u_max=u_max,
                         disturbance_gain=disturbance_gain)
        self.c = constraints or SafetyConstraints()
        self.w0, self.w1, self.w_du = w0, w1, w_du
        self.rho, self.rho1 = slack_weight, slack_linear
        self._con = [self._constraint_rows(m) for m in (0, 1)]
        self._check_qp, self._safe_qp = [], []
        nz = NU * horizon
        opts = {"printLevel": "none", "error_on_fail": False}
        for mode in (0, 1):
            m = len(self._con[mode]["rows"])
            ng = len(self.GROUPS)
            self._check_qp.append(ca.conic(
                f"check{mode}", "qpoases",
                {"h": ca.Sparsity.dense(nz, nz), "a": ca.Sparsity.dense(m, nz)}, opts))
            self._safe_qp.append(ca.conic(
                f"safe{mode}", "qpoases",
                {"h": ca.Sparsity.dense(nz + ng, nz + ng),
                 "a": ca.Sparsity.dense(2 * m, nz + ng)}, opts))
        D = np.eye(nz) - np.eye(nz, k=-NU)
        track = np.full(nz, self.w1)
        track[:NU] = self.w0
        self._D = D
        self._H_check = np.diag(np.r_[np.zeros(NU), np.full(nz - NU, self.w1)]) + self.w_du * D.T @ D
        H = np.zeros((nz + ng, nz + ng))
        H[:nz, :nz] = np.diag(track) + self.w_du * D.T @ D
        H[nz:, nz:] = self.rho * np.eye(ng)
        self._H_safe = H
        self._track = track
        self.last = {}

    def _constraint_rows(self, mode):
        """State rows, bounds and constraint group of every constraint k=1..N."""
        c, big = self.c, 1e3
        per_step = [(19, -c.max_tilt, c.max_tilt, 0), (20, -c.max_tilt, c.max_tilt, 0)]
        if mode == 1:
            per_step += [(15, -c.max_swing, c.max_swing, 1), (16, -c.max_swing, c.max_swing, 1)]
        per_step += [(3 + i, -c.max_speed, c.max_speed, 2) for i in range(3)]
        per_step += [(2, c.min_altitude, big, 3)]
        rows, lb, ub, group = [], [], [], []
        for k in range(self.N):
            for idx, lo, hi, g in per_step:
                rows.append(k * NX + idx)
                lb.append(lo)
                ub.append(hi)
                group.append(g)
        rows = np.array(rows)
        G = self._pre[mode]["Gamma"][rows]
        E = np.zeros((len(rows), len(self.GROUPS)))
        E[np.arange(len(rows)), group] = 1.0
        return dict(rows=rows, lb=np.array(lb), ub=np.array(ub), G=G, E=E)

    def reset(self):
        super().reset()
        self.last = {}

    def filter(self, mode, x0, u_rl, u_prev=None):
        """Return (applied command, info). ``info['intervened']`` tells whether
        the MPC replaced the proposed command."""
        t0 = time.perf_counter()
        self._update_disturbance(mode, x0)
        pre, con, model, N = self._pre[mode], self._con[mode], self.models[mode], self.N
        u_rl = np.clip(np.asarray(u_rl, float), -self.u_max, self.u_max)
        u_prev = u_rl if u_prev is None else np.asarray(u_prev, float)
        offset = model.f + model.disturbance_map() @ self.dist
        free = pre["Phi"] @ x0 + pre["S"] @ offset
        c = free[con["rows"]]
        lo, hi = con["lb"] - c, con["ub"] - c
        u_ref = np.tile(u_rl, N)
        e_prev = np.zeros(NU * N)
        e_prev[:NU] = u_prev

        # 1) certification: u_0 = u_rl fixed, is there an admissible continuation?
        lbx = np.full(NU * N, -self.u_max)
        ubx = np.full(NU * N, self.u_max)
        lbx[:NU] = ubx[:NU] = u_rl
        g = -np.r_[np.zeros(NU), np.full(NU * (N - 1), self.w1)] * u_ref \
            - self.w_du * self._D.T @ e_prev
        qp = self._check_qp[mode]
        sol = qp(h=self._H_check, g=g, a=con["G"], lba=lo, uba=hi, lbx=lbx, ubx=ubx, x0=u_ref)
        U = np.asarray(sol["x"]).ravel()
        certified = bool(qp.stats()["success"]) and np.all(np.isfinite(U))
        if certified:
            y = con["G"] @ U
            certified = bool(np.all(y >= lo - 1e-4) and np.all(y <= hi + 1e-4))

        slack = np.zeros(len(self.GROUPS))
        if certified:
            u_apply = u_rl
        else:
            # 2) intervention: closest admissible command (soft constraints)
            ng = len(self.GROUPS)
            A = np.block([[con["G"], -con["E"]], [con["G"], con["E"]]])
            inf = 1e20
            lba = np.r_[np.full(len(lo), -inf), lo]
            uba = np.r_[hi, np.full(len(hi), inf)]
            g = np.r_[-self._track * u_ref - self.w_du * self._D.T @ e_prev,
                      np.full(ng, self.rho1)]
            qp = self._safe_qp[mode]
            sol = qp(h=self._H_safe, g=g, a=A, lba=lba, uba=uba,
                     lbx=np.r_[np.full(NU * N, -self.u_max), np.zeros(ng)],
                     ubx=np.r_[np.full(NU * N, self.u_max), np.full(ng, inf)],
                     x0=np.r_[u_ref, np.zeros(ng)])
            z = np.asarray(sol["x"]).ravel()
            if qp.stats()["success"] and np.all(np.isfinite(z)):
                U = z[:NU * N]
                slack = z[NU * N:]
                u_apply = U[:NU].copy()
            else:                                   # should not happen (soft QP)
                U = np.tile(np.zeros(NU), N)
                u_apply = np.zeros(NU)
        U_plan = U.reshape(N, NU).T
        X = model.rollout(x0, U_plan, self.dist)
        self._last = (mode, x0.copy(), u_apply.copy())
        self.last = {"intervened": not certified, "u_rl": u_rl, "u_applied": u_apply,
                     "slack": slack, "X_pred": X, "U_plan": U_plan,
                     "solve_time": time.perf_counter() - t0}
        return u_apply, self.last


################################################################################
# AGENT (use like an RL policy, e.g. from utilities/play.py)
################################################################################

# Default location of the identified model (see identify() below).
DEFAULT_MODEL_FILE = "results/mpc_director/identified_model.npz"
U_MAX = 0.8
# Close the hook harder once the payload is off its stand: with the grasp
# command (0.6) thin cylinders can slide out of the hook while hovering.
TENDON_HOLD = 1.0


class DirectorMPCAgent(MissionAgent):
    """Velocity-command MPC flying the pick-and-place mission in
    ``AdaptiveTransportDirectorAviary`` on top of its PPO velocity controller.

    Needs the identified closed-loop model (``model_file``). It is identified
    automatically if missing; re-identify after the velocity policy is
    retrained with ``python -m multi_drone_mujoco.envs.director_mpc``.
    """

    tendon_hold = TENDON_HOLD

    def __init__(self, env, models=None, model_file=DEFAULT_MODEL_FILE, horizon=40,
                 max_time=45.0, verbose=False, max_tilt=np.pi / 2):
        super().__init__(env, horizon, max_time, verbose)
        if models is None:
            try:
                models = load_models(model_file)
            except (FileNotFoundError, ValueError) as err:
                print(f"[DirectorMPCAgent] {err}\n  -> identifying the model now"
                      " (about 2 minutes)...")
                models = identify(model_file=model_file)
        self.models = models
        self.mpc = DirectorMPC(models, horizon=horizon, u_max=U_MAX, max_tilt=max_tilt)

    def _reset_controller(self):
        # Start from an empty command filter (also cleared by the env on reset).
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
        tendon = self.mission.tendon
        return (u, tendon), u, X_pred, info

    def _plant_step(self, action):
        u, tendon = action
        self.tracker.advance(self.x)
        return director_step(self.env, u, (tendon, -tendon))

    def _on_attach(self):
        self._say("payload attached -> switching to drone+pendulum model")

    def _replay(self, mode, x0, U):
        return self.models[mode].rollout(x0, U)


################################################################################
# IDENTIFICATION
################################################################################

IDENT_SEEDS = (100, 101, 102, 103)


def make_identification_env(seed):
    from gymnasium.utils import seeding

    from multi_drone_mujoco.envs.adaptive_hook_director_velocity import (
        AdaptiveTransportDirectorAviary,
    )
    env = AdaptiveTransportDirectorAviary()
    env.PAYLOAD_TERMINATION = True       # 4 waypoints + random start position
    env.np_random, _ = seeding.np_random(seed)
    env.reset(seed=seed)
    env.prev_action = None
    return env


def collect(seed, hold_time=8.0, excite=0.15, max_time=45.0, verbose=True):
    """Closed-loop data: the mission flown by a position P-controller with
    random velocity-command excitation (both without and with the payload)."""
    from multi_drone_mujoco.envs.mpc_mission import Mission, payload_attached

    env = make_identification_env(seed)
    dt = env.CTRL_TIMESTEP
    mission = Mission(env, tendon_hold=TENDON_HOLD)
    tracker = StateTracker(env)
    rng = np.random.default_rng(seed)
    attached = False
    noise, noise_left = np.zeros(3), 0
    log = {k: [] for k in ("x", "u", "attached", "phase")}
    t = 0.0
    x = tracker.measure(attached)
    while t < max_time:
        if not mission.update(t, dt, x, attached) and mission.phase != "hold":
            break
        if mission.phase == "hold" and t - mission.phase_start > hold_time:
            break
        p_ref, v_ref = mission.reference(t, dt, 1)
        cmd = v_ref[:, 0] + 1.5 * (p_ref[:, 0] - x[0:3])
        if mission.phase not in ("pre_grasp", "descend", "close"):
            if noise_left <= 0:
                noise = rng.uniform(-excite, excite, 3) * (rng.random() < 0.7)
                noise_left = rng.integers(10, 40)
            noise_left -= 1
            cmd = cmd + noise
        cmd = np.clip(cmd, -U_MAX, U_MAX)

        log["x"].append(x)
        log["u"].append(cmd)
        log["attached"].append(attached)
        log["phase"].append(mission.phase)

        tracker.advance(x)
        director_step(env, cmd, (mission.tendon, -mission.tendon))
        t += dt
        rest_z = None if mission.payload_rest is None else mission.payload_rest[2]
        attached = payload_attached(env, rest_z, attached)
        x = tracker.measure(attached)
        if x[2] < 0.0 or np.any(np.abs(env.rpy[0, 0:2]) > np.pi / 2):
            if verbose:
                print(f"  seed {seed}: crash at t={t:.2f}s")
            break
    env.close()
    out = {k: np.asarray(v) for k, v in log.items()}
    if verbose:
        print(f"  seed {seed}: {t:.1f}s, visited {[n for n, _ in mission.visited]},"
              f" attached samples {out['attached'].sum()}")
    return out


def identify(model_file=DEFAULT_MODEL_FILE, seeds=IDENT_SEEDS, verbose=True):
    """Collect data, fit the velocity, swing and attitude models of both
    modes, save them to ``model_file`` and return the loaded models."""
    import os

    from multi_drone_mujoco.envs.mpc_mission import CONTACT_PHASES

    if verbose:
        print("Collecting identification data (P-controller + excitation)...")
    runs = [collect(s, verbose=verbose) for s in seeds]
    env = make_identification_env(0)
    alpha, dt = env.alpha, env.CTRL_TIMESTEP
    env.close()

    # Concatenate the runs; the mask removes samples of the other mode, the
    # hook-contact phases and the joints between runs.
    X = np.concatenate([r["x"] for r in runs])
    masks = []
    for mode in (0, 1):
        ms = []
        for r in runs:
            att = r["attached"].astype(bool)
            contact = np.isin(r["phase"], CONTACT_PHASES)
            m = (att if mode == 1 else (~att & ~contact)).copy()
            m[:3] = False
            m[-3:] = False
            ms.append(m)
        masks.append(np.concatenate(ms))

    axes_all, tilts_all = [], []
    for mode in (0, 1):
        mask = masks[mode]
        if verbose:
            print(f"mode {mode} ({mask.sum()} samples), multi-step (0.5 s) RMS fit error:")
        axes, tilts = [], []
        for i in range(3):
            swing = mode == 1 and i < 2
            theta = X[:, TH][:, i] if swing else None
            v, c = X[:, V][:, i], X[:, C][:, i]
            kw = dict(theta=theta, theta_dot=X[:, THD][:, i] if swing else None, mask=mask)
            m0 = fit_axis(v, c, dt, **kw)
            m1, rms = refine_axis(m0, v, c, dt, **kw)
            axes.append(m1)
            if verbose:
                print(f"   v{'xyz'[i]}: rms={rms:.4f} m/s")
            if i < 2:
                tilt = X[:, TILT_OF_AXIS[i]]
                tm, rms_t = fit_tilt(tilt, c, v, theta=theta, mask=mask)
                tilts.append(tm)
                if verbose:
                    print(f"   {'pitch' if i == 0 else 'roll'}: rms={np.degrees(rms_t):.3f} deg")
        axes_all.append(axes)
        tilts_all.append(tilts)

    os.makedirs(os.path.dirname(model_file) or ".", exist_ok=True)
    save_models(model_file, axes_all, tilts_all, alpha, dt)
    if verbose:
        print("saved:", model_file)
    return load_models(model_file)


if __name__ == "__main__":
    identify()
