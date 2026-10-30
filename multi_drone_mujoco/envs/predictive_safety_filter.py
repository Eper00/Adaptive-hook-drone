"""Predictive safety filter on the director's velocity commands (thesis, Chapter 5).

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
2. Certification. The proposal is certified if, applied now, a continuation
   of commands exists that keeps the predicted state inside the constraints
   over the horizon (tilt, altitude, speed, payload swing). The model is
   linearized along the nominal prediction (the proposal, then the previous
   plan), the constraints are checked with a QP, and the resulting plan is
   verified with the nonlinear model. A certified command is applied
   unchanged.
3. Intervention. Otherwise the admissible command closest to u_rl is
   applied (first command of)

       min  w0 |u_0 - u_rl|^2 + w1 sum_k |u_k - u_rl|^2 + w_du |du|^2
            + sum_g (rho_g s_g^2 + rho1_g s_g)
       s.t. lb_k - s_g <= y_k(U) <= ub_k + s_g,  s_g >= 0
            |tilt_k| <= hard_max_tilt

   solved as a sequence of QPs on the linearized model (Gauss-Newton) with a
   backtracking line search on the nonlinear prediction, the same SQP method
   as the director MPC (resnet_mpc.ResNetMPC). The
   slacks keep the problem feasible; their weights set the priorities
   tilt > altitude > speed > swing. The tilt is soft at ``max_tilt`` (15 deg,
   which triggers the intervention) and hard at ``hard_max_tilt`` (60 deg,
   below the 90 deg where an episode counts as flipped). If the linearized
   QP cannot meet the hard bound, it is solved without it      ["constraint"]

If the measured state already violates a bound (e.g. the payload already
swings more than the limit), that bound is relaxed to the current value and
has to shrink by the factor ``contraction`` per step, instead of demanding
an immediate return that no command can achieve.

A per-step limit of the command change is not used: the RL director commands
bang-bang (u jumps between about +1 and -1 and the env's low-pass filter
averages it), so such a limit clips almost every step and shifts the average
command the RL intends (with g tan(15 deg): ~95 % of the steps clipped, 259
successes of 300 scenarios against 279 without the limit, 2026-10-12).

The prediction model is the ResNet model of resnet_mpc.py, a PyTorch module
m(x, u) -> x_next, used as identified. Rollouts and Jacobians come from
PyTorch (``torch.func``), the QPs are solved with DAQP.

Structure of one ``filter`` call (one control step):

    measured x0, proposal u_rl
      |-- bounds of the first command: escape box (contact) or the command bound
      |-- nominal plan U_nom = [u_rl, previous plan shifted]  -> rollout X_nom
      |-- certification: X_nom already safe?  else QP with u_0 fixed to u_rl
      |       (does a safe continuation exist?), checked with the nonlinear rollout
      |-- certified   -> apply u_rl
      `-- not certified -> intervention SQP (closest command, soft constraints
                           with slacks, hard tilt; line search on the nonlinear
                           merit) -> apply its first command

Both QPs work on the linearized model y(U + dU) ~ y(U) + G dU, where y are
the constrained state entries over the horizon and G = dy/dU (single
shooting, as in resnet_mpc.ResNetMPC). They are written in the absolute
commands U: y(U_new) ~ (y - G U) + G U_new, so ``off = y - G U`` shifts the
bounds.
"""

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import torch

from multi_drone_mujoco.envs.director_mpc import NU, NX, TH, TILT, V
from multi_drone_mujoco.envs.resnet_mpc import rollout, sensitivity, sqp, step_jacobians


@dataclass
class SafetyConstraints:
    """Limits of the safety filter (see the module docstring)."""
    # state constraints over the horizon
    max_tilt: float = np.radians(15.0)       # |roll|, |pitch| (world frame), soft [rad]
    hard_max_tilt: float = np.radians(60.0)  # hard tilt bound of the intervention [rad]
    min_altitude: float = 0.3                # drone height [m]
    max_speed: float = 1.0                   # |v| per axis [m/s]
    max_swing: float = np.radians(45.0)      # |payload swing| per axis, mode 1 [rad]
    contraction: float = 0.9                 # shrink factor of already violated bounds
    # hook-contact escape
    contact_tilt: float = np.radians(10.0)   # trigger: contact and tilt above this [rad]
    escape_climb: float = 0.5                # u_z >= this while escaping [m/s]
    escape_lateral: float = 0.0              # |u_x|, |u_y| <= this while escaping [m/s]
    escape_time: float = 0.5                 # longest escape per trigger [s]
    escape_release_tilt: float = np.radians(8.0)   # escape ends: no contact, tilt below [rad]


def tilt_angle(x):
    """Angle between the body z axis and the vertical from the world-frame
    roll / pitch of the state vector [rad]."""
    roll, pitch = x[TILT]
    return float(np.arctan(np.hypot(np.tan(roll), np.tan(pitch))))


def hook_obstacle_contact(env, attached):
    """True if the hook touches the payload assembly anywhere else than a
    lifted payload hanging in it: the connectors or the holder plate, or the
    handle (cylinder) before the payload is lifted."""
    m, d = env.model, env.data
    if not hasattr(env, "_hook_geom_ids"):
        env._hook_geom_ids = {m.geom(f"segment_geom_{i}").id for i in range(1, 8)}
    hook = env._hook_geom_ids
    rest = {env.left_connector_geom_id, env.right_connector_geom_id, m.geom("holder_plate").id}
    for c in d.contact[:d.ncon]:
        if c.geom1 in hook or c.geom2 in hook:
            other = c.geom2 if c.geom1 in hook else c.geom1
            if other in rest or (other == env.target_geom_id and not attached):
                return True
    return False


class PredictiveSafetyFilter:
    """Model-based safety filter for velocity commands (see module docstring).

    models: [mode 0 model, mode 1 model], PyTorch modules m(x, u) -> x_next
            (``ResNetDynamics`` of resnet_mpc.py).

    Usage: one ``filter(mode, x0, u_rl, u_prev, contact)`` per control step,
    ``reset()`` at the start of an episode.
    """

    # constraint groups; each group has one slack variable in the intervention QP
    GROUPS = ("tilt", "altitude", "speed", "swing")
    # slack weights (quadratic rho, linear rho1) per group: the larger, the
    # more important the group (tilt > altitude > speed > swing). The linear
    # term makes the penalty exact: small violations are not "cheap"
    SLACK = {"tilt": (1e6, 1e4), "altitude": (1e5, 1e3), "speed": (1e4, 1e2), "swing": (1e3, 10.0)}
    # violations below these count as "within the constraints" (the
    # linearization error of the QP solution)
    TOL = np.array([np.radians(1.0), 0.01, 0.03, np.radians(2.0)])
    # line search merit: penalty per rad beyond the hard tilt bound
    HARD_PENALTY = 1e6

    # w0 = 10 (instead of 1): an intervention stays closer to the RL proposal.
    # 300 scenarios (success = goal + 5 s hold), RL alone 283: w0 = 1 -> 292
    # (11 rescued, 2 broken), w0 = 10 -> 294 (12 rescued, 1 broken), w0 = 20 /
    # 30 -> 288 / 289 (2026-10-12)
    def __init__(self, models, horizon=36, constraints=None, u_max=1.0, w0=10.0, w1=0.1,
                 w_du=0.5, sqp_iterations=2, dt=1.0 / 48.0, qp_solver="daqp"):
        # w0: weight of |u_0 - u_rl|^2 (the applied command), w1: of the later
        # commands' distance to u_rl, w_du: of the command changes.
        # sqp_iterations: as in the director MPC (waypoint_mpc.SQP_ITERATIONS),
        # so that both solve their problems with the same SQP method (5
        # iterations gave the same outcomes on 200 scenarios, but the MPC no
        # longer fit into the 20.8 ms control period, 2026-10-29)
        self.models = [m.requires_grad_(False) for m in models]
        self._jac = [step_jacobians(m) for m in self.models]
        self.N = N = horizon
        self.c = constraints or SafetyConstraints()
        self.u_max, self.w1, self.w_du = u_max, w1, w_du
        self.sqp_iterations, self.dt = sqp_iterations, dt
        nz, ng = NU * N, len(self.GROUPS)
        self._D = np.eye(nz) - np.eye(nz, k=-NU)         # D U = (u_0, u_1 - u_0, ...)
        self._rows = [self._constraint_layout(mode) for mode in (0, 1)]
        # one QP pair per mode (the swing constraints exist only in mode 1):
        #   cert: variables U (nz), rows = the soft constraints (no slacks)
        #   safe: variables [U, slacks] (nz + ng), rows = upper soft bounds,
        #         lower soft bounds, hard tilt bounds
        opts = {"error_on_fail": False}
        dense = ca.Sparsity.dense
        self._cert_qp, self._safe_qp = [], []
        for mode in (0, 1):
            m_c, m_h = len(self._rows[mode]["idx"]) * N, len(self._rows[mode]["hard"])
            self._cert_qp.append(ca.conic(f"cert{mode}", qp_solver,
                                          {"h": dense(nz, nz), "a": dense(m_c, nz)}, opts))
            self._safe_qp.append(ca.conic(f"safe{mode}", qp_solver,
                                          {"h": dense(nz + ng, nz + ng),
                                           "a": dense(2 * m_c + m_h, nz + ng)}, opts))
        # tracking weights of the commands: w0 on u_0, w1 on u_1..u_N-1
        self._track = np.full(nz, w1)
        self._track[:NU] = w0
        # Hessian of the certification QP: u_0 is fixed to the proposal there,
        # so it has no tracking term; the continuation stays close to u_rl
        # (w1) and smooth (w_du)
        self._H_cert = np.diag(np.r_[np.zeros(NU), np.full(nz - NU, w1)]) + w_du * self._D.T @ self._D
        # Hessian of the intervention QP: tracking + smoothness on U, and the
        # quadratic slack penalties (rho_g s_g^2 -> 2 rho_g on the diagonal)
        self._H_safe = np.zeros((nz + ng, nz + ng))
        self._H_safe[:nz, :nz] = np.diag(self._track) + w_du * self._D.T @ self._D
        self._H_safe[nz:, nz:] = np.diag([2 * self.SLACK[g][0] for g in self.GROUPS])
        self._slack_lin = np.array([self.SLACK[g][1] for g in self.GROUPS])
        self.reset()

    # ------------------------------------------------------------------ setup
    def _constraint_layout(self, mode):
        """Constrained state entries of one step: index, soft bounds, hard
        bounds, group; ``hard`` lists the rows of the stacked horizon outputs
        that have a hard bound (the tilt), ``E`` maps every row to the slack
        of its group."""
        c, inf = self.c, np.inf
        # (state index, soft lower, soft upper, group, hard bound)
        rows = ([(i, -c.max_tilt, c.max_tilt, 0, c.hard_max_tilt) for i in range(TILT.start, TILT.stop)]
                + [(2, c.min_altitude, 1e3, 1, inf)]
                + [(i, -c.max_speed, c.max_speed, 2, inf) for i in range(V.start, V.stop)])
        if mode == 1:                                   # payload swing only with the payload
            rows += [(i, -c.max_swing, c.max_swing, 3, inf) for i in range(TH.start, TH.stop)]
        idx, lo, hi, grp, hard = (np.array(col) for col in zip(*rows))
        grp = grp.astype(int)
        return dict(idx=idx.astype(int), lo=lo, hi=hi, grp=grp, hard_lo=-hard, hard_hi=hard,
                    hard=np.flatnonzero(np.isfinite(np.tile(hard, self.N))),
                    E=np.tile(np.eye(len(self.GROUPS))[grp], (self.N, 1)))

    def reset(self):
        self._plan = None             # previous command plan (NU, N), the next nominal continuation
        self._escape_left = 0         # remaining escape steps
        self.last = {}

    # --------------------------------------------------------------- helpers
    def rollout(self, mode, x0, U):
        """Nonlinear prediction X (NX, N+1) for the command plan U (NU, N)."""
        with torch.inference_mode():
            X = rollout(self.models[mode], torch.from_numpy(x0), torch.from_numpy(U.T.copy()))
        return np.hstack([x0[:, None], X.numpy().T])

    def _bounds(self, mode, x0, hard=False):
        """Per-step soft (or hard) bounds (N * n_c,), relaxed where x0 already
        violates them.

        If x0 exceeds a bound by e, the bound of step k is moved out by
        e * contraction^k: the violation may stay, but has to shrink
        geometrically over the horizon (a demand no command could meet would
        make every proposal uncertifiable)."""
        L = self._rows[mode]
        lo, hi = (L["hard_lo"], L["hard_hi"]) if hard else (L["lo"], L["hi"])
        y0 = x0[L["idx"]]
        rho = self.c.contraction ** np.arange(1, self.N + 1)[:, None]
        hi = hi[None] + np.maximum(y0 - hi, 0.0)[None] * rho
        lo = lo[None] - np.maximum(lo - y0, 0.0)[None] * rho
        return lo.ravel(), hi.ravel()

    def _linearize(self, mode, X, U):
        """Outputs y (constrained entries of x_1..x_N) and their sensitivity
        G = dy/dU along the trajectory (X, U): the constrained rows of
        Gamma = dX/dU (resnet_mpc.sensitivity)."""
        idx = self._rows[mode]["idx"]
        Gamma = sensitivity(self._jac[mode], X.T, U.T)
        G = Gamma.reshape(self.N, NX, -1)[:, idx].reshape(self.N * len(idx), -1)
        return X[idx][:, 1:].T.ravel(), G

    def _violation(self, mode, X, lo, hi):
        """Largest violation per group of a predicted trajectory."""
        L = self._rows[mode]
        y = X[L["idx"]][:, 1:].T.ravel()
        viol = np.maximum(np.maximum(lo - y, y - hi), 0.0)
        grp = np.tile(L["grp"], self.N)
        return np.array([viol[grp == g].max(initial=0.0) for g in range(len(self.GROUPS))])

    def _merit(self, mode, U, X, u_ref, e_prev, lo, hi, hard_lo, hard_hi):
        """Nonlinear merit of the intervention's line search: the cost of the
        intervention QP with the slacks replaced by the violations of the
        NONLINEAR prediction X, plus a large penalty beyond the hard tilt
        bound (the QP meets the bounds only for the linearized model; the
        MPC's line search penalizes its tilt limit the same way)."""
        u = U.T.ravel()
        d, du = u - u_ref, self._D @ u - e_prev
        v = self._violation(mode, X, lo, hi)                 # largest violation per group
        rho = np.array([self.SLACK[g][0] for g in self.GROUPS])
        L = self._rows[mode]
        H = L["hard"]
        y = X[L["idx"]][:, 1:].T.ravel()
        hard = np.maximum(np.maximum(hard_lo[H] - y[H], y[H] - hard_hi[H]), 0.0).max(initial=0.0)
        return (0.5 * np.sum(self._track * d * d) + 0.5 * self.w_du * du @ du
                + np.sum(rho * v * v + self._slack_lin * v) + self.HARD_PENALTY * hard)

    def _escape(self, x0, contact):
        """Hook-contact escape state machine; True while escaping."""
        c, tilt = self.c, tilt_angle(x0)
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
        N, c, nz, ng = self.N, self.c, NU * self.N, len(self.GROUPS)
        x0 = np.asarray(x0, float)
        u_rl = np.clip(np.asarray(u_rl, float), -self.u_max, self.u_max)
        u_prev = u_rl if u_prev is None else np.asarray(u_prev, float)
        lo, hi = self._bounds(mode, x0)
        within = lambda X: np.all(self._violation(mode, X, lo, hi) <= self.TOL)
        L = self._rows[mode]
        # tracking target: the proposal over the whole horizon
        u_ref = np.tile(u_rl, N)
        # e_prev = (u_prev, 0, ...): D U - e_prev = (u_0 - u_prev, u_1 - u_0, ...)
        e_prev = np.zeros(nz)
        e_prev[:NU] = u_prev

        # bounds on the first command: a climb while escaping, else the command bound
        escaping = self._escape(x0, contact)
        if escaping:
            lb0 = np.array([-c.escape_lateral, -c.escape_lateral, c.escape_climb])
            ub0 = np.array([c.escape_lateral, c.escape_lateral, self.u_max])
        else:
            lb0, ub0 = np.full(NU, -self.u_max), np.full(NU, self.u_max)

        # nominal plan: the proposal now, then the previous plan (shifted).
        # The previous plan U_prev was made one step ago, its column 0 was
        # applied then and column 1 would be "now"; u_rl replaces that, the
        # rest (columns 2..) follows, the last command is repeated
        if self._plan is None:
            U_nom = np.tile(u_rl[:, None], (1, N))
        else:
            U_nom = np.hstack([u_rl[:, None], self._plan[:, 2:], self._plan[:, -1:]])
        X_nom = self.rollout(mode, x0, U_nom)

        # 1) certification of the proposal
        #    Question: with u_0 = u_rl fixed, is there ANY continuation
        #    u_1..u_N-1 that keeps the prediction within the soft bounds?
        #    Cheapest first: is the nominal continuation already safe? If not,
        #    a QP searches for a safe continuation (hard constraints, no
        #    slacks: an infeasible QP means "not certifiable"); its result is
        #    accepted only if the nonlinear rollout confirms it.
        U, X = U_nom, X_nom
        certified = not escaping and within(X_nom)      # the nominal continuation is already safe
        if not escaping and not certified:
            U_c, X_c = U_nom, X_nom
            for _ in range(self.sqp_iterations):
                y, G = self._linearize(mode, X_c, U_c)
                u_flat = U_c.T.ravel()
                off = y - G @ u_flat             # y(U) ~ off + G U (linearized outputs)
                lbx, ubx = np.full(nz, -self.u_max), np.full(nz, self.u_max)
                lbx[:NU] = ubx[:NU] = u_rl       # u_0 fixed to the proposal
                # gradient of 0.5 w1 sum_k>0 |u_k - u_rl|^2 + 0.5 w_du |D U - e_prev|^2
                g = -np.r_[np.zeros(NU), np.full(nz - NU, self.w1)] * u_ref \
                    - self.w_du * self._D.T @ e_prev
                # infeasible = u_rl cannot be certified (-> intervention)
                qp = self._cert_qp[mode]
                sol = qp(h=self._H_cert, g=g, a=G, lba=lo - off, uba=hi - off,
                         lbx=lbx, ubx=ubx, x0=u_flat)
                z = np.asarray(sol["x"]).ravel()
                if not (qp.stats()["success"] and np.all(np.isfinite(z))):
                    break
                U_c = z.reshape(N, NU).T
                X_c = self.rollout(mode, x0, U_c)
                if within(X_c):
                    certified, U, X = True, U_c, X_c
                    break

        slack, hard_kept = np.zeros(ng), True
        if certified:
            u_apply = u_rl.copy()
        else:
            # 2) intervention: closest admissible command (soft constraints + hard tilt)
            #    variables z = [U (nz), s (one slack per group, >= 0)]
            #    min  0.5 U^T diag(track) U - (track * u_ref)^T U       (stay close to u_rl)
            #         + 0.5 w_du |D U - e_prev|^2                         (smooth)
            #         + sum_g (rho_g s_g^2 + rho1_g s_g)                  (constraint violation)
            #    s.t. off + G U - E s <= hi,  off + G U + E s >= lo       (soft bounds)
            #         hard_lo <= off + G U <= hard_hi on the tilt rows   (hard tilt)
            #    E maps every constraint row to the slack of its group
            H = L["hard"]
            hard_lo, hard_hi = self._bounds(mode, x0, hard=True)
            lbu, ubu = np.full(nz, -self.u_max), np.full(nz, self.u_max)
            lbu[:NU], ubu[:NU] = lb0, ub0
            # start of the SQP: the nominal plan, moved into the command box
            U_s = np.clip(U_nom, lbu.reshape(N, NU).T, ubu.reshape(N, NU).T)
            X_s = self.rollout(mode, x0, U_s)

            def qp_step(U_s, X_s):
                """The intervention QP around (U_s, X_s) -> (step towards its
                plan, slacks), or None if it fails."""
                nonlocal hard_kept
                y, G = self._linearize(mode, X_s, U_s)
                u_flat = U_s.T.ravel()
                off = y - G @ u_flat
                A = np.block([[G, -L["E"]], [G, L["E"]], [G[H], np.zeros((len(H), ng))]])
                lba = np.r_[np.full(len(lo), -np.inf), lo - off, hard_lo[H] - off[H]]
                uba = np.r_[hi - off, np.full(len(hi), np.inf), hard_hi[H] - off[H]]
                g = np.r_[-self._track * u_ref - self.w_du * self._D.T @ e_prev, self._slack_lin]
                qp = self._safe_qp[mode]
                for hard in (True, False) if hard_kept else (False,):
                    if not hard:            # the linearized QP cannot meet the hard bound
                        lba[2 * len(lo):], uba[2 * len(lo):] = -np.inf, np.inf
                    sol = qp(h=self._H_safe, g=g, a=A, lba=lba, uba=uba,
                             lbx=np.r_[lbu, np.zeros(ng)], ubx=np.r_[ubu, np.full(ng, np.inf)],
                             x0=np.r_[u_flat, np.zeros(ng)])
                    z = np.asarray(sol["x"]).ravel()
                    if qp.stats()["success"] and np.all(np.isfinite(z)):
                        # the QP gives a whole plan; the step towards it is
                        # what the line search scales (stays inside the
                        # command boxes, they are convex)
                        return z[:nz].reshape(N, NU).T - U_s, z[nz:]
                    hard_kept = False
                return None

            # the shared SQP loop (resnet_mpc.sqp), as in the MPC: QP step +
            # line search on the nonlinear merit
            merit = lambda U_, X_: self._merit(mode, U_, X_, u_ref, e_prev, lo, hi, hard_lo, hard_hi)
            U, X, _, solved, s_qp = sqp(U_s, X_s, merit, qp_step,
                                        lambda U_: self.rollout(mode, x0, U_), self.sqp_iterations)
            if solved:
                slack = s_qp
            else:                                # QP failure: hold the previous command
                U = np.tile(np.clip(u_prev, lb0, ub0)[:, None], (1, N))
                X = self.rollout(mode, x0, U)
            u_apply = U[:, 0].copy()

        # the plan (certified continuation or intervention plan) is the
        # nominal continuation of the next step
        self._plan = U
        reason = "none" if certified else "contact" if escaping else "constraint"
        self.last = {"intervened": not certified, "reason": reason, "u_rl": u_rl,
                     "u_applied": u_apply, "slack": slack, "hard_tilt_kept": hard_kept,
                     "X_pred": X, "U_plan": U, "solve_time": time.perf_counter() - t0}
        return u_apply, self.last
