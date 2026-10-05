"""Predictive safety filter on the director's velocity commands.

Sits between a velocity-command controller (e.g. the RL director policy) and
``AdaptiveTransportDirectorAviary``. Every control step (48 Hz) the proposed
command u_rl is checked with an identified prediction model of the closed
loop (env low-pass filter -> PPO velocity controller -> drone + payload):

1. Contact escape. If the hook touches the payload assembly (handle before
   it is lifted, connectors, holder plate) and the drone tilts by more than
   ``contact_tilt``, the hook is being levered: the drone rests on the
   payload through the hook, the contact torque exceeds what the rotors can
   counter and the drone flips within ~0.5 s. Neither prediction model
   contains contacts, so this is a measured-state rule: the first command
   is restricted to a climb (u_z >= escape_climb, |u_x|, |u_y| <=
   escape_lateral) until the contact is gone and the drone is level again
   (at most ``escape_time`` per trigger)                           ["contact"]
2. Command-change limit. The PPO velocity controller tracks the low-pass
   filtered command c_k+1 = c_k + alpha (u - c_k), not u itself (the RL
   director switches u between the bounds and lets the filter average it).
   A proposal that changes c faster than ``max_ref_accel`` per axis is
   clipped to that rate: |u - c_k| <= max_ref_accel dt / alpha. The default,
   g tan(15 deg), is the acceleration a 15 deg tilt sustains. Optionally,
   jumps of u larger than ``max_step`` or direction changes larger than
   ``max_turn`` are limited as well (off by default)                 ["rate"]
3. Certification. The (clipped) proposal is certified if, applied now, a
   continuation of commands exists that keeps the predicted state inside the
   constraints over the horizon (tilt, altitude, speed, payload swing). The
   model is linearized along the nominal prediction (the proposal, then the
   previous plan), the constraints are checked with a QP, and the resulting
   plan is verified with the nonlinear model. A certified command is applied
   unchanged (if it was clipped: reason "rate").
4. Intervention. Otherwise the admissible command closest to u_rl is
   applied (first command of)

       min  w0 |u_0 - u_rl|^2 + w1 sum_k |u_k - u_rl|^2 + w_du |du|^2
            + sum_g (rho_g s_g^2 + rho1_g s_g)
       s.t. lb_k - s_g <= y_k(U) <= ub_k + s_g,  s_g >= 0
            |tilt_k| <= hard_max_tilt

   solved as a sequence of QPs on the linearized model (Gauss-Newton). The
   slacks keep the problem feasible; their weights set the priorities
   tilt > altitude > speed > swing. The tilt is soft at ``max_tilt`` (15 deg,
   which triggers the intervention) and hard at ``hard_max_tilt`` (60 deg,
   below the 90 deg where an episode counts as flipped). If the linearized
   QP cannot meet the hard bound, it is solved without it      ["constraint"]

If the measured state already violates a bound (e.g. the payload already
swings more than the limit), that bound is relaxed to the current value and
has to shrink by the factor ``contraction`` per step, instead of demanding
an immediate return that no command can achieve.

The prediction model is a PyTorch module m(x, u, d) -> x_next: the ResNet
model of resnet_mpc.py (the more accurate one, see
utilities/analyse_identification.py), or the ARX DirectorModel wrapped by
``LinearDynamics``. Rollouts and Jacobians come from PyTorch (``torch.func``),
the QPs are solved with DAQP.
"""

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import torch
from torch import nn

from multi_drone_mujoco.envs.director_mpc import C, NU, NX
from multi_drone_mujoco.envs.resnet_mpc import rollout, step_jacobians

TILT_IDX = (19, 20)
SPEED_IDX = (3, 4, 5)
SWING_IDX = (15, 16)
ALT_IDX = 2


@dataclass
class SafetyConstraints:
    """Limits of the safety filter (see the module docstring)."""
    # state constraints over the horizon
    max_tilt: float = np.radians(15.0)       # |roll|, |pitch| (world frame), soft [rad]
    hard_max_tilt: float = np.radians(60.0)  # hard tilt bound of the intervention [rad] (None: off)
    min_altitude: float = 0.3                # drone height [m]
    max_speed: float = 1.0                   # |v| per axis [m/s]
    max_swing: float = np.radians(45.0)      # |payload swing| per axis [rad] (None: off)
    contraction: float = 0.9                 # shrink factor of already violated bounds
    # hook-contact escape
    contact_tilt: float = np.radians(10.0)   # trigger: contact and tilt above this [rad]
    escape_climb: float = 0.5                # u_z >= this while escaping [m/s]
    escape_lateral: float = 0.0              # |u_x|, |u_y| <= this while escaping [m/s]
    escape_time: float = 0.5                 # longest escape per trigger [s]
    escape_release_tilt: float = np.radians(8.0)   # escape ends: no contact, tilt below [rad]
    # command-change limits (None disables them)
    max_ref_accel: float = 9.81 * np.tan(np.radians(15.0))  # |c_k+1 - c_k| / dt per axis of
                                             # the filtered command the PPO tracks [m/s^2]
    max_step: float = None                   # |u_k - u_k-1| per step [m/s]
    max_turn: float = None                   # direction change per step [rad] ...
    turn_min_speed: float = 0.1              # ... checked when both commands exceed this


def tilt_angle(x):
    """Angle between the body z axis and the vertical from the world-frame
    roll / pitch of the state vector [rad]."""
    return float(np.arctan(np.hypot(np.tan(x[19]), np.tan(x[20]))))


def hook_obstacle_contact(env, attached):
    """True if the hook touches the payload assembly anywhere else than a
    lifted payload hanging in it: the connectors or the holder plate, or the
    handle (cylinder) before the payload is lifted."""
    m, d = env.model, env.data
    hook = env._hook_geom_ids if hasattr(env, "_hook_geom_ids") else None
    if hook is None:
        hook = {m.geom(f"segment_geom_{i}").id for i in range(1, 8)}
        env._hook_geom_ids = hook
    rest = {env.left_connector_geom_id, env.right_connector_geom_id, m.geom("holder_plate").id}
    handle = env.target_geom_id
    for c in d.contact[:d.ncon]:
        g1, g2 = c.geom1, c.geom2
        if g1 in hook:
            other = g2
        elif g2 in hook:
            other = g1
        else:
            continue
        if other in rest or (other == handle and not attached):
            return True
    return False


class LinearDynamics(nn.Module):
    """x_k+1 = A x_k + B u_k + f + E d of a linear model (the ARX
    DirectorModel) as a PyTorch module, so that the filter can use it like
    the ResNet model."""

    def __init__(self, model):
        super().__init__()
        for name, a in (("A", model.A), ("B", model.B), ("f", model.f),
                        ("E", model.disturbance_map())):
            self.register_buffer(name, torch.as_tensor(a, dtype=torch.float64))

    def forward(self, x, u, d=None):
        x_next = x @ self.A.T + u @ self.B.T + self.f
        return x_next if d is None else x_next + d @ self.E.T


class PredictiveSafetyFilter:
    """Model-based safety filter for velocity commands (see module docstring).

    models: [mode 0 model, mode 1 model], PyTorch modules m(x, u, d) -> x_next
            (``ResNetDynamics`` of resnet_mpc.py or ``LinearDynamics``).
    """

    GROUPS = ("tilt", "altitude", "speed", "swing")
    SLACK = {"tilt": (1e6, 1e4), "altitude": (1e5, 1e3), "speed": (1e4, 1e2), "swing": (1e3, 10.0)}
    TOL = {"tilt": np.radians(1.0), "altitude": 0.01, "speed": 0.03, "swing": np.radians(2.0)}

    def __init__(self, models, horizon=36, constraints=None, u_max=1.0, disturbance_gain=0.05,
                 max_disturbance=0.01, w0=1.0, w1=0.1, w_du=0.5, sqp_iterations=2, dt=1.0 / 48.0,
                 alpha=0.1, qp_solver="daqp"):
        self.models = [m.requires_grad_(False) for m in models]
        self._jac = [step_jacobians(m) for m in self.models]
        self.N = N = horizon
        self.c = constraints or SafetyConstraints()
        self.u_max = u_max
        self.dist_gain = disturbance_gain
        self.max_dist = max_disturbance
        self.w0, self.w1, self.w_du = w0, w1, w_du
        self.sqp_iterations = sqp_iterations
        self.dt = dt
        self.alpha = alpha              # gain of the env's command low-pass filter
        nz = NU * N
        self._D = np.eye(nz) - np.eye(nz, k=-NU)
        self._rows = [self._constraint_layout(mode) for mode in (0, 1)]
        opts = {"error_on_fail": False}
        ng = len(self.GROUPS)
        self._cert_qp, self._safe_qp = [], []
        for mode in (0, 1):
            m_c = len(self._rows[mode]["idx"]) * N
            m_h = len(self._rows[mode]["hard"])
            self._cert_qp.append(ca.conic(f"cert{mode}", qp_solver,
                                          {"h": ca.Sparsity.dense(nz, nz),
                                           "a": ca.Sparsity.dense(m_c, nz)}, opts))
            self._safe_qp.append(ca.conic(f"safe{mode}", qp_solver,
                                          {"h": ca.Sparsity.dense(nz + ng, nz + ng),
                                           "a": ca.Sparsity.dense(2 * m_c + m_h, nz + ng)}, opts))
        track = np.full(nz, w1)
        track[:NU] = w0
        self._track = track
        self._H_cert = np.diag(np.r_[np.zeros(NU), np.full(nz - NU, w1)]) + w_du * self._D.T @ self._D
        H = np.zeros((nz + ng, nz + ng))
        H[:nz, :nz] = np.diag(track) + w_du * self._D.T @ self._D
        H[nz:, nz:] = np.diag([2 * self.SLACK[g][0] for g in self.GROUPS])
        self._H_safe = H
        self._slack_lin = np.array([self.SLACK[g][1] for g in self.GROUPS])
        self.reset()

    # ------------------------------------------------------------------ setup
    def _constraint_layout(self, mode):
        """Constrained state entries of one step: index, soft bounds, hard
        bounds, group; ``hard`` lists the rows of the stacked horizon outputs
        that have a hard bound."""
        c, big = self.c, 1e3
        idx, lo, hi, hard_lo, hard_hi, grp = [], [], [], [], [], []

        def add(i, a, b, g, hard=np.inf):
            idx.append(i)
            lo.append(a)
            hi.append(b)
            hard_lo.append(-hard)
            hard_hi.append(hard)
            grp.append(self.GROUPS.index(g))
        for i in TILT_IDX:
            add(i, -c.max_tilt, c.max_tilt, "tilt",
                np.inf if c.hard_max_tilt is None else c.hard_max_tilt)
        add(ALT_IDX, c.min_altitude, big, "altitude")
        for i in SPEED_IDX:
            add(i, -c.max_speed, c.max_speed, "speed")
        if mode == 1 and c.max_swing is not None:
            for i in SWING_IDX:
                add(i, -c.max_swing, c.max_swing, "swing")
        E1 = np.zeros((len(idx), len(self.GROUPS)))
        E1[np.arange(len(idx)), grp] = 1.0
        return dict(idx=np.array(idx), lo=np.array(lo), hi=np.array(hi), grp=np.array(grp),
                    hard_lo=np.array(hard_lo), hard_hi=np.array(hard_hi),
                    hard=np.flatnonzero(np.isfinite(np.tile(hard_hi, self.N))),
                    E=np.tile(E1, (self.N, 1)))

    def reset(self):
        self.dist = np.zeros(3)
        self._last = None             # (mode, x, u) of the previous step
        self._plan = None             # previous command plan (NU, N)
        self._escape_left = 0         # remaining escape steps
        self.last = {}

    # --------------------------------------------------------------- helpers
    def _update_disturbance(self, mode, x):
        if self._last is None or self._last[0] != mode:
            if self._last is not None:
                self.dist[:] = 0.0
            return
        _, x_prev, u_prev = self._last
        with torch.inference_mode():
            pred = self.models[mode](torch.from_numpy(x_prev), torch.from_numpy(u_prev),
                                     torch.from_numpy(self.dist)).numpy()
        self.dist = np.clip(self.dist + self.dist_gain * (x[3:6] - pred[3:6]),
                            -self.max_dist, self.max_dist)

    def rollout(self, mode, x0, U):
        """Nonlinear prediction X (NX, N+1) for the command plan U (NU, N)."""
        with torch.inference_mode():
            X = rollout(self.models[mode], torch.from_numpy(x0), torch.from_numpy(U.T.copy()),
                        torch.from_numpy(self.dist))
        return np.hstack([x0[:, None], X.numpy().T])

    def _bounds(self, mode, x0, hard=False):
        """Per-step soft (or hard) bounds (N * n_c,), relaxed where x0 already
        violates them."""
        L = self._rows[mode]
        lo, hi = (L["hard_lo"], L["hard_hi"]) if hard else (L["lo"], L["hi"])
        y0 = x0[L["idx"]]
        rho = self.c.contraction ** np.arange(1, self.N + 1)[:, None]
        hi = hi[None] + np.maximum(y0 - hi, 0.0)[None] * rho
        lo = lo[None] - np.maximum(lo - y0, 0.0)[None] * rho
        return lo.ravel(), hi.ravel()

    def _linearize(self, mode, X, U):
        """Outputs y (constrained entries of x_1..x_N) and their sensitivity
        G = dy/dU along the trajectory (X, U)."""
        N = self.N
        L = self._rows[mode]
        A, B = self._jac[mode](torch.from_numpy(X[:, :N].T.copy()), torch.from_numpy(U.T.copy()),
                               torch.from_numpy(self.dist))
        A, B = A.numpy(), B.numpy()                      # (N, NX, NX), (N, NX, NU)
        n_c = len(L["idx"])
        G = np.zeros((N * n_c, NU * N))
        S = np.zeros((NX, NU * N))
        for k in range(N):
            S = A[k] @ S
            S[:, NU * k:NU * (k + 1)] += B[k]
            G[k * n_c:(k + 1) * n_c] = S[L["idx"]]
        y = X[L["idx"]][:, 1:].T.ravel()
        return y, G

    def _violation(self, mode, X, lo, hi):
        """Largest violation per group of a predicted trajectory."""
        L = self._rows[mode]
        y = X[L["idx"]][:, 1:].T.ravel()
        viol = np.maximum(np.maximum(lo - y, y - hi), 0.0)
        grp = np.tile(L["grp"], self.N)
        return np.array([viol[grp == g].max() if np.any(grp == g) else 0.0
                         for g in range(len(self.GROUPS))])

    def _within_tolerance(self, v):
        return all(v[g] <= self.TOL[name] for g, name in enumerate(self.GROUPS))

    def rate_box(self, x0, u_prev):
        """Bounds on the first command from ``max_ref_accel`` (around the
        filtered command c = x0[C]) and ``max_step`` (around u_prev)."""
        c = self.c
        lb, ub = np.full(NU, -self.u_max), np.full(NU, self.u_max)
        if c.max_step is not None:
            lb, ub = np.maximum(lb, u_prev - c.max_step), np.minimum(ub, u_prev + c.max_step)
        if c.max_ref_accel is not None:
            # |c_k+1 - c_k| = alpha |u - c_k| <= max_ref_accel dt; clipping (instead of
            # intersecting) keeps the box non-empty, the reference limit wins
            r = c.max_ref_accel * self.dt / self.alpha
            lb = np.clip(lb, x0[C] - r, x0[C] + r)
            ub = np.clip(ub, x0[C] - r, x0[C] + r)
        return lb, ub

    def turn_violation(self, u_rl, u_prev):
        """Proposal turns by more than ``max_turn`` with respect to the
        previously applied command."""
        c = self.c
        n0, n1 = np.linalg.norm(u_prev), np.linalg.norm(u_rl)
        if c.max_turn is not None and min(n0, n1) > c.turn_min_speed:
            angle = np.arccos(np.clip(u_rl @ u_prev / (n0 * n1), -1.0, 1.0))
            return bool(angle > c.max_turn)
        return False

    def _escape(self, x0, contact):
        """Hook-contact escape state machine; True while escaping."""
        c = self.c
        tilt = tilt_angle(x0)
        if self._escape_left > 0:
            self._escape_left -= 1
            if not contact and tilt < c.escape_release_tilt:
                self._escape_left = 0
        if self._escape_left == 0 and contact and tilt > c.contact_tilt:
            self._escape_left = max(1, int(round(c.escape_time / self.dt)))
        return self._escape_left > 0

    # ---------------------------------------------------------------- filter
    def filter(self, mode, x0, u_rl, u_prev=None, contact=False):
        """Return (applied command, info). See the module docstring."""
        t0 = time.perf_counter()
        N, c = self.N, self.c
        x0 = np.asarray(x0, float)
        self._update_disturbance(mode, x0)
        u_rl = np.clip(np.asarray(u_rl, float), -self.u_max, self.u_max)
        u_prev = u_rl if u_prev is None else np.asarray(u_prev, float)
        lo, hi = self._bounds(mode, x0)
        L = self._rows[mode]
        u_ref = np.tile(u_rl, N)
        e_prev = np.zeros(NU * N)
        e_prev[:NU] = u_prev

        # bounds on the first command
        escaping = self._escape(x0, contact)
        turn_bad = False
        if escaping:
            lb0 = np.array([-c.escape_lateral, -c.escape_lateral, c.escape_climb])
            ub0 = np.array([c.escape_lateral, c.escape_lateral, self.u_max])
        else:
            lb0, ub0 = self.rate_box(x0, u_prev)
            turn_bad = self.turn_violation(u_rl, u_prev)
        # the proposal within the command-change limits
        u_try = u_rl if escaping else np.clip(u_rl, lb0, ub0)
        clipped = bool(np.any(u_try != u_rl))

        # nominal plan: the proposal now, then the previous plan (shifted)
        if self._plan is None:
            U_nom = np.tile(u_try[:, None], (1, N))
        else:
            U_nom = np.hstack([u_try[:, None], self._plan[:, 2:], self._plan[:, -1:]])
        X_nom = self.rollout(mode, x0, U_nom)

        # 1) certification of the (clipped) proposal
        certified = False
        U, X = U_nom, X_nom
        if not escaping and not turn_bad:
            if self._within_tolerance(self._violation(mode, X_nom, lo, hi)):
                certified = True                 # the nominal continuation is already safe
            else:
                U_c, X_c = U_nom, X_nom
                for _ in range(self.sqp_iterations):
                    y, G = self._linearize(mode, X_c, U_c)
                    u_flat = U_c.T.ravel()
                    off = y - G @ u_flat
                    lbx = np.full(NU * N, -self.u_max)
                    ubx = np.full(NU * N, self.u_max)
                    lbx[:NU] = ubx[:NU] = u_try
                    g = -np.r_[np.zeros(NU), np.full(NU * (N - 1), self.w1)] * u_ref \
                        - self.w_du * self._D.T @ e_prev
                    # infeasible = u_try cannot be certified (-> intervention)
                    qp = self._cert_qp[mode]
                    sol = qp(h=self._H_cert, g=g, a=G, lba=lo - off, uba=hi - off,
                             lbx=lbx, ubx=ubx, x0=u_flat)
                    z = np.asarray(sol["x"]).ravel()
                    if not (qp.stats()["success"] and np.all(np.isfinite(z))):
                        break
                    U_c = z.reshape(N, NU).T
                    X_c = self.rollout(mode, x0, U_c)
                    if self._within_tolerance(self._violation(mode, X_c, lo, hi)):
                        certified, U, X = True, U_c, X_c
                        break

        slack = np.zeros(len(self.GROUPS))
        hard_kept = True
        if certified:
            u_apply = u_try.copy()
        else:
            # 2) intervention: closest admissible command (soft constraints + hard tilt)
            ng = len(self.GROUPS)
            H = L["hard"]
            hard_lo, hard_hi = self._bounds(mode, x0, hard=True)
            lbu = np.full(NU * N, -self.u_max)
            ubu = np.full(NU * N, self.u_max)
            lbu[:NU], ubu[:NU] = lb0, ub0
            U_s = np.clip(U_nom, lbu.reshape(N, NU).T, ubu.reshape(N, NU).T)
            X_s = self.rollout(mode, x0, U_s)
            best = None
            for _ in range(self.sqp_iterations):
                y, G = self._linearize(mode, X_s, U_s)
                u_flat = U_s.T.ravel()
                off = y - G @ u_flat
                A = np.block([[G, -L["E"]], [G, L["E"]], [G[H], np.zeros((len(H), ng))]])
                lba = np.r_[np.full(len(lo), -np.inf), lo - off, hard_lo[H] - off[H]]
                uba = np.r_[hi - off, np.full(len(hi), np.inf), hard_hi[H] - off[H]]
                g = np.r_[-self._track * u_ref - self.w_du * self._D.T @ e_prev, self._slack_lin]
                qp = self._safe_qp[mode]
                z = None
                for hard in (True, False) if hard_kept else (False,):
                    if not hard:            # the linearized QP cannot meet the hard bound
                        lba[2 * len(lo):], uba[2 * len(lo):] = -np.inf, np.inf
                    sol = qp(h=self._H_safe, g=g, a=A, lba=lba, uba=uba,
                             lbx=np.r_[lbu, np.zeros(ng)],
                             ubx=np.r_[ubu, np.full(ng, np.inf)],
                             x0=np.r_[u_flat, np.zeros(ng)])
                    z = np.asarray(sol["x"]).ravel()
                    if qp.stats()["success"] and np.all(np.isfinite(z)):
                        break
                    hard_kept, z = False, None
                if z is None:
                    break
                U_s = z[:NU * N].reshape(N, NU).T
                X_s = self.rollout(mode, x0, U_s)
                best = (U_s, X_s, z[NU * N:])
            if best is None:                     # QP failure: hold the previous command
                u_hold = np.clip(u_prev, lb0, ub0)
                U = np.tile(u_hold[:, None], (1, N))
                X = self.rollout(mode, x0, U)
            else:
                U, X, slack = best
            u_apply = U[:, 0].copy()

        self._plan = U
        self._last = (mode, x0.copy(), u_apply.copy())
        if certified:
            reason = "rate" if clipped else "none"
        elif escaping:
            reason = "contact"
        elif turn_bad:
            reason = "rate"
        else:
            reason = "constraint"
        self.last = {"intervened": reason != "none", "reason": reason, "u_rl": u_rl,
                     "u_applied": u_apply, "slack": slack, "hard_tilt_kept": hard_kept,
                     "X_pred": X, "U_plan": U, "violation": self._violation(mode, X, lo, hi),
                     "solve_time": time.perf_counter() - t0}
        return u_apply, self.last
