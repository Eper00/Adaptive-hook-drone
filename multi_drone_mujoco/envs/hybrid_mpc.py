"""Hybrid drone / drone+pendulum model predictive controller.

The controller uses two analytical prediction models and switches between them
depending on whether the payload hangs on the hook:

* mode 0 (no payload):  rigid quadrotor with the light (11 g) hook chain
                        lumped onto it as an offset mass (the hook's top
                        hinge only allows +-5 deg).
* mode 1 (payload):     the same rigid body plus the payload as a 2-DoF
                        pendulum hinged in the pocket of the curled hook:
                        alpha is the (damped) swing about the cylinder axis,
                        i.e. about the drone x-axis, beta the sideways rocking
                        in the hook, modelled with a torsional spring-damper.

Both models share the 16-dimensional state

    x = [p (3, world), v (3, world), rpy (3), omega (3, body),
         alpha, beta, alpha_dot, beta_dot]

and the input u = the four individual rotor thrusts [N], mixed exactly like
``BaseAviary._physics`` does for the BB_HOOK drone. In mode 0 the pendulum
states are frozen.

The equations of motion are derived with the projected Newton-Euler (virtual
power) method: every rigid body contributes ``J_v' m J_v + J_w' I J_w`` to the
mass matrix, and the velocity-product terms are obtained by differentiating the
body velocities along the configuration, so no hand-written Coriolis terms are
needed.
"""

import time
from dataclasses import dataclass, field

import casadi as ca
import mujoco
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.mpc_mission import MissionAgent

NX = 16
NU = 4
G = 9.81

# Parameter vector of the payload pendulum:
# [m_p, l, pivot_x, pivot_y, pivot_z, Ixx_c, Iyy_c, Izz_c,
#  alpha damping, beta stiffness, beta damping]
NP = 11


################################################################################
# PHYSICAL PARAMETERS READ FROM THE MUJOCO MODEL
################################################################################

@dataclass
class DroneParams:
    """Rigid drone + lumped hook parameters (everything in the drone frame)."""
    mass: float
    inertia: np.ndarray
    hook_mass: float
    hook_com: np.ndarray
    hook_inertia: np.ndarray
    kf: float
    km: float
    arm: float
    hover_rpm: float
    max_rpm: float

    @property
    def max_motor_thrust(self):
        return self.kf * self.max_rpm ** 2

    @property
    def total_mass(self):
        return self.mass + self.hook_mass


@dataclass
class PayloadParams:
    """Payload pendulum parameters (pivot expressed in the drone frame)."""
    mass: float = 0.0
    length: float = 0.0
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia_com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    damping: float = 0.0          # alpha: rotation about the cylinder axis
    beta_stiffness: float = 0.0   # beta: rocking of the cylinder in the hook
    beta_damping: float = 0.0

    def vector(self):
        return np.hstack([self.mass, self.length, self.pivot, self.inertia_com,
                          self.damping, self.beta_stiffness, self.beta_damping])


def _composite(model, data, body_ids, frame_pos, frame_mat):
    """Mass, COM and inertia (about the COM) of rigidly lumped bodies.

    COM and inertia are expressed in the frame (frame_pos, frame_mat).
    """
    mass = 0.0
    first = np.zeros(3)
    for b in body_ids:
        m = model.body_mass[b]
        mass += m
        first += m * frame_mat.T @ (data.xipos[b] - frame_pos)
    com = first / max(mass, 1e-12)

    inertia = np.zeros((3, 3))
    for b in body_ids:
        m = model.body_mass[b]
        rot = frame_mat.T @ data.ximat[b].reshape(3, 3)
        r = frame_mat.T @ (data.xipos[b] - frame_pos) - com
        inertia += rot @ np.diag(model.body_inertia[b]) @ rot.T
        inertia += m * (r @ r * np.eye(3) - np.outer(r, r))
    return mass, com, inertia


def drone_params_from_env(env):
    """Read the drone and the lumped hook chain from the MuJoCo model."""
    model, data = env.model, env.data
    drone = model.body("drone0").id
    hook = [model.body(f"segment_{i}").id for i in range(1, 8)]
    mujoco.mj_forward(model, data)
    m_h, c_h, i_h = _composite(model, data, hook, data.xpos[drone],
                               data.xmat[drone].reshape(3, 3))
    return DroneParams(
        mass=float(model.body_mass[drone]),
        inertia=np.diag(model.body_inertia[drone]),
        hook_mass=float(m_h),
        hook_com=c_h,
        hook_inertia=i_h,
        kf=env.KF,
        km=env.KM,
        arm=env.L,
        hover_rpm=env.HOVER_RPM,
        max_rpm=env.MAX_RPM,
    )


def pendulum_bodies(env):
    """Bodies of the payload pendulum (the payload's welded parts)."""
    model = env.model
    target = model.body("target").id
    return [b for b in range(model.nbody)
            if model.body_rootid[b] == target and model.body_mass[b] > 0]


def pivot_point(env):
    """World position of the pendulum pivot: the cylinder centre, which rests
    in the pocket of the curled hook."""
    return env.data.xpos[env.model.body("target").id].copy()


def _pendulum_vector(env):
    """Pivot -> payload COM and its time derivative, in the drone frame
    (derivative relative to the drone; the pivot is taken as drone-fixed)."""
    model, data = env.model, env.data
    drone = model.body("drone0").id
    rot = data.xmat[drone].reshape(3, 3)
    omega_b = data.qvel[3:6]
    pivot_w = pivot_point(env)

    mass, com, com_vel = 0.0, np.zeros(3), np.zeros(3)
    vel6 = np.zeros(6)
    for b in pendulum_bodies(env):
        m = model.body_mass[b]
        # [angular, linear] velocity at the body COM, world orientation
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
        mass += m
        com += m * data.xipos[b]
        com_vel += m * vel6[3:6]
    com /= mass
    com_vel /= mass

    pivot_vel_w = data.qvel[0:3] + rot @ np.cross(omega_b, rot.T @ (pivot_w - data.xpos[drone]))
    r = rot.T @ (com - pivot_w)
    r_dot = rot.T @ (com_vel - pivot_vel_w) - np.cross(omega_b, r)
    return r, r_dot


def measure_pivot(env):
    """Pendulum pivot expressed in the drone frame."""
    data = env.data
    drone = env.model.body("drone0").id
    rot = data.xmat[drone].reshape(3, 3)
    return rot.T @ (pivot_point(env) - data.xpos[drone])


def payload_params_from_env(env, damping=0.0, beta_stiffness=0.0, beta_damping=0.0):
    """Payload pendulum parameters from the current MuJoCo configuration.

    The payload hangs in the pocket of the curled hook and swings about the
    cylinder axis (alpha, about the drone x-axis, damped); rocking it sideways
    in the hook (beta) is a torsional spring-damper. The hook itself stays
    part of the rigid drone (link_1 only allows +-5 deg).
    """
    model, data = env.model, env.data
    drone = model.body("drone0").id
    rot = data.xmat[drone].reshape(3, 3)
    pivot_w = pivot_point(env)
    r, _ = _pendulum_vector(env)
    length = np.linalg.norm(r)
    u = r / length
    alpha = np.arctan2(u[1], -u[2])
    beta = np.arcsin(np.clip(-u[0], -1, 1))
    c, s = np.cos(alpha), np.sin(alpha)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(beta), np.sin(beta)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    frame = rot @ rx @ ry             # pendulum frame: COM on its -z axis
    mass, _, inertia = _composite(model, data, pendulum_bodies(env), pivot_w, frame)
    return PayloadParams(
        mass=float(mass),
        length=float(length),
        pivot=rot.T @ (pivot_w - data.xpos[drone]),
        inertia_com=np.diag(inertia).copy(),
        damping=damping,
        beta_stiffness=beta_stiffness,
        beta_damping=beta_damping,
    )


################################################################################
# STATE MEASUREMENT FROM MUJOCO
################################################################################

def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def measure_state(env, attached):
    """Reduced 16-dim state of the MuJoCo plant.

    The pendulum angles describe the direction of the pendulum COM seen from
    its pivot, in the drone frame:  r / |r| = Rx(alpha) Ry(beta) [0, 0, -1]'.
    """
    data = env.data
    q = data.qpos[0:7]
    qd = data.qvel[0:6]
    x = np.zeros(NX)
    x[0:3] = q[0:3]
    x[3:6] = qd[0:3]
    x[6:9] = quat_to_rpy(q[3:7])
    x[9:12] = qd[3:6]              # free joint: angular velocity in body frame
    if not attached:
        return x

    r, r_dot = _pendulum_vector(env)
    n = np.linalg.norm(r)
    u = r / n
    u_dot = (r_dot - u * (u @ r_dot)) / n
    alpha = np.arctan2(u[1], -u[2])
    beta = np.arcsin(np.clip(-u[0], -1, 1))
    den = u[1] ** 2 + u[2] ** 2
    x[12] = alpha
    x[13] = beta
    x[14] = (-u[2] * u_dot[1] + u[1] * u_dot[2]) / max(den, 1e-9)
    x[15] = -u_dot[0] / max(np.cos(beta), 1e-6)
    return x


def thrusts_to_action(f, drone):
    """Rotor thrusts [N] -> normalized env motor commands in [-1, 1].

    Exact inverse of ``BaseAviary._normalizedActionToRPM`` with f = kf*rpm^2.
    """
    rpm = np.sqrt(np.clip(f, 0.0, None) / drone.kf)
    return np.clip(np.where(
        rpm <= drone.hover_rpm,
        rpm / drone.hover_rpm - 1.0,
        (rpm - drone.hover_rpm) / (drone.max_rpm - drone.hover_rpm),
    ), -1.0, 1.0)


################################################################################
# CASADI MODELS
################################################################################

def _rx(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, c, -s), ca.horzcat(0, s, c))


def _ry(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(ca.horzcat(c, 0, s), ca.horzcat(0, 1, 0), ca.horzcat(-s, 0, c))


def _rz(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))


def _euler_rate_matrix(roll, pitch):
    """rpy_dot = E(roll, pitch) @ omega_body for R = Rz Ry Rx."""
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp, tp = ca.cos(pitch), ca.tan(pitch)
    return ca.vertcat(
        ca.horzcat(1, sr * tp, cr * tp),
        ca.horzcat(0, cr, -sr),
        ca.horzcat(0, sr / cp, cr / cp),
    )


def build_continuous_model(drone: DroneParams, with_payload: bool):
    """Continuous-time dynamics f(x, u, p) -> x_dot."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    par = ca.SX.sym("p", NP)

    pos, vel = x[0:3], x[3:6]
    roll, pitch, yaw = x[6], x[7], x[8]
    omega = x[9:12]
    ang, ang_dot = x[12:14], x[14:16]

    rot = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    E = _euler_rate_matrix(roll, pitch)

    # Configuration, its time derivative and generalized velocities.
    if with_payload:
        q = ca.vertcat(pos, x[6:9], ang)
        q_dot = ca.vertcat(vel, E @ omega, ang_dot)
        nu = ca.vertcat(vel, omega, ang_dot)
    else:
        q = ca.vertcat(pos, x[6:9])
        q_dot = ca.vertcat(vel, E @ omega)
        nu = ca.vertcat(vel, omega)

    # Rigid bodies: (mass, COM position in world, body angular velocity, inertia)
    # The light hook chain (link_1 allows only +-5 deg) is lumped onto the drone.
    bodies = [
        (drone.mass, pos, omega, ca.DM(drone.inertia)),
        (drone.hook_mass, pos + rot @ ca.DM(drone.hook_com), omega,
         ca.DM(drone.hook_inertia)),
    ]
    if with_payload:
        m_p, length, pivot = par[0], par[1], par[2:5]
        inertia_p = ca.diag(par[5:8])
        alpha, beta = ang[0], ang[1]
        rel = _rx(alpha) @ _ry(beta)
        com_p = pos + rot @ (pivot + rel @ ca.vertcat(0, 0, -length))
        omega_p = (rel.T @ omega
                   + _ry(beta).T @ ca.vertcat(ang_dot[0], 0, 0)
                   + ca.vertcat(0, ang_dot[1], 0))
        bodies.append((m_p, com_p, omega_p, inertia_p))

    n = nu.shape[0]
    mass_matrix = ca.SX.zeros(n, n)
    rhs = ca.SX.zeros(n, 1)
    gravity = ca.DM([0, 0, -G])

    for m, com, w, inertia in bodies:
        v_com = ca.jacobian(com, q) @ q_dot
        J_v = ca.jacobian(v_com, nu)
        J_w = ca.jacobian(w, nu)
        a_bias = ca.jacobian(v_com, q) @ q_dot
        w_bias = ca.jacobian(w, q) @ q_dot
        mass_matrix += m * J_v.T @ J_v + J_w.T @ inertia @ J_w
        rhs -= J_v.T @ (m * (a_bias - gravity))
        rhs -= J_w.T @ (inertia @ w_bias + ca.cross(w, inertia @ w))

    # Rotor wrench, identical to BaseAviary._physics for BB_HOOK
    # (the non CF2X/RACE mixer branch).
    thrust = ca.sum1(u)
    tau = ca.vertcat(
        (u[1] - u[3]) * drone.arm,
        (-u[0] + u[2]) * drone.arm,
        drone.km / drone.kf * (-u[0] + u[1] - u[2] + u[3]),
    )
    rhs[0:3] += rot @ ca.vertcat(0, 0, thrust)
    rhs[3:6] += tau
    if with_payload:
        rhs[6] -= par[8] * ang_dot[0]
        rhs[7] -= par[9] * ang[1] + par[10] * ang_dot[1]

    nu_dot = ca.solve(mass_matrix, rhs)
    ang_ddot = nu_dot[6:8] if with_payload else ca.SX.zeros(2)
    ang_rate = ang_dot if with_payload else ca.SX.zeros(2)

    x_dot = ca.vertcat(vel, nu_dot[0:3], E @ omega, nu_dot[3:6], ang_rate, ang_ddot)
    name = "drone_pendulum" if with_payload else "drone"
    return ca.Function(f"f_{name}", [x, u, par], [x_dot])


def build_discrete_model(f, dt, substeps=2):
    """RK4 discretization with zero-order-hold input."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    par = ca.SX.sym("p", NP)
    h = dt / substeps
    xk = x
    for _ in range(substeps):
        k1 = f(xk, u, par)
        k2 = f(xk + h / 2 * k1, u, par)
        k3 = f(xk + h / 2 * k2, u, par)
        k4 = f(xk + h * k3, u, par)
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function(f.name().replace("f_", "F_"), [x, u, par], [xk])


################################################################################
# MPC
################################################################################

@dataclass
class MPCWeights:
    """Diagonal least-squares weights (stage cost; terminal = stage * terminal)."""
    pos: tuple = (60.0, 60.0, 80.0)
    vel: float = 6.0
    tilt: float = 2.0
    yaw: float = 4.0
    omega: tuple = (0.05, 0.05, 0.5)
    swing: float = 6.0
    swing_rate: float = 1.0
    u: float = 0.05
    du: float = 1.0
    terminal: float = 5.0


class HybridMPC:
    """Nonlinear MPC with a mode-switched prediction model.

    The optimal control problem

        min  sum_k ||x_k - x_ref,k||^2_Q + ||u_k - u_hover||^2_R + ||u_k - u_k-1||^2_S
        s.t. x_k+1 = F_mode(x_k, u_k, p),   0 <= u_k <= u_max,
             |roll_k|, |pitch_k| <= max_tilt   (k = 1..N)

    is solved by single-shooting Gauss-Newton SQP, warm started from the
    shifted previous plan: the nonlinear model is rolled out under the current
    input guess, linearized along that rollout in one parallel CasADi call,
    the linearized dynamics are condensed onto the input sequence, and the
    dense box-constrained QP is solved with qpOASES. A backtracking line search
    on the nonlinear cost globalizes the step. Since the cost is quadratic in
    the state, the Gauss-Newton Hessian is the exact cost Hessian and only the
    dynamics curvature is neglected.
    """

    def __init__(self, drone: DroneParams, dt, horizon=40, weights=None,
                 sqp_iterations=2, rk4_substeps=1, threads=8, max_tilt=np.pi / 2):
        self.drone = drone
        self.max_tilt = max_tilt          # |roll|, |pitch| limit over the horizon
        self.dt = dt
        self.N = horizon
        self.w = weights or MPCWeights()
        self.sqp_iterations = sqp_iterations
        self.u_max = drone.max_motor_thrust

        self.f = [build_continuous_model(drone, False),
                  build_continuous_model(drone, True)]
        self.F = [build_discrete_model(f, dt, rk4_substeps) for f in self.f]

        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        par = ca.SX.sym("p", NP)
        self._lin = []
        self._rollout = []
        for F in self.F:
            x_next = F(x, u, par)
            lin = ca.Function("lin", [x, u, par],
                              [x_next, ca.jacobian(x_next, x), ca.jacobian(x_next, u)])
            self._lin.append(lin.map(horizon, "thread", threads))
            self._rollout.append(F.mapaccum(horizon))

        nz = NU * horizon
        opts = {"printLevel": "none", "error_on_fail": False}
        # Roll/pitch constraints of every predicted step (2 rows per step);
        # the unconstrained QP is the fallback if they are infeasible.
        self._qp_con = ca.conic("qp_con", "qpoases",
                                {"h": ca.Sparsity.dense(nz, nz),
                                 "a": ca.Sparsity.dense(2 * horizon, nz)}, opts)
        self._qp = ca.conic("qp", "qpoases",
                            {"h": ca.Sparsity.dense(nz, nz), "a": ca.Sparsity(0, nz)}, opts)
        self._q_diag = self._stage_weights()
        self._D = np.eye(nz) - np.eye(nz, k=-NU)       # input differences
        self._plan = None                              # (U_bar, mode)

    def _stage_weights(self):
        w = self.w
        q = np.zeros(NX)
        q[0:3] = w.pos
        q[3:6] = w.vel
        q[6:8] = w.tilt
        q[8] = w.yaw
        q[9:12] = w.omega
        return q

    def hover_thrust(self, payload_mass):
        return (self.drone.total_mass + payload_mass) * G / 4.0

    def rollout(self, mode, x0, U, payload: PayloadParams):
        """Nonlinear open-loop prediction of the model for inputs U [NU, N]."""
        par = np.tile(payload.vector()[:, None], (1, U.shape[1]))
        if U.shape[1] == self.N:
            X = np.asarray(self._rollout[mode](x0, U, par))
        else:
            X = np.zeros((NX, U.shape[1]))
            xk = x0
            for k in range(U.shape[1]):
                xk = np.asarray(self.F[mode](xk, U[:, k], par[:, k])).ravel()
                X[:, k] = xk
        return np.hstack([np.asarray(x0).reshape(NX, 1), X])

    def reset(self):
        self._plan = None

    def solve(self, mode, x0, p_ref, v_ref, yaw_ref, u_prev, payload: PayloadParams):
        """Compute the input plan.

        Parameters
        ----------
        mode : int        0 = drone (+rigid hook), 1 = drone + payload pendulum
        x0 : (NX,)        measured state
        p_ref, v_ref : (3, N+1) position / velocity reference over the horizon
        yaw_ref : float or (N+1,) yaw reference
        u_prev : (NU,)    previously applied rotor thrusts
        payload : PayloadParams (ignored in mode 0)

        Returns (X_pred [NX, N+1], U_plan [NU, N], info). X_pred is the
        nonlinear rollout of the prediction model under the returned plan.
        """
        t_start = time.perf_counter()
        N = self.N
        hover = self.hover_thrust(payload.mass if mode == 1 else 0.0)
        par = np.tile(payload.vector()[:, None], (1, N))

        # Initial guess: shifted previous input plan, or hover when (re)starting.
        if self._plan is None:
            U_bar = np.full((NU, N), hover)
        else:
            U_old, old_mode = self._plan
            U_bar = np.hstack([U_old[:, 1:], U_old[:, -1:]])
            if old_mode != mode:
                U_bar += hover - self.hover_thrust(
                    payload.mass if old_mode == 1 else 0.0)
        U_bar = np.clip(U_bar, 0.0, self.u_max)

        # Reference over the horizon (state components not listed have zero weight).
        ref = np.zeros((NX, N + 1))
        ref[0:3] = p_ref
        ref[3:6] = v_ref
        yaw_ref = np.broadcast_to(np.asarray(yaw_ref, float), (N + 1,))
        ref[8] = x0[8] + np.unwrap(np.arctan2(np.sin(yaw_ref - x0[8]),
                                              np.cos(yaw_ref - x0[8])))

        q = self._q_diag.copy()
        if mode == 1:
            q[12:14] = self.w.swing
            q[14:16] = self.w.swing_rate
        Q = np.tile(q, N)
        Q[-NX:] *= self.w.terminal

        D = self._D
        e_prev = np.zeros(NU * N)
        e_prev[:NU] = u_prev

        def cost(X, U):
            e = (X[:, 1:] - ref[:, 1:]).T.ravel()
            u = U.T.ravel()
            du = D @ u - e_prev
            return 0.5 * (e @ (Q * e) + self.w.u * np.sum((u - hover) ** 2)
                          + self.w.du * du @ du)

        def merit(X, U):
            violation = np.maximum(np.abs(X[6:8, 1:]) - self.max_tilt, 0.0).sum()
            return cost(X, U) + 1e4 * violation

        X_bar = self.rollout(mode, x0, U_bar, payload)
        M_bar = merit(X_bar, U_bar)
        qp_failures = 0
        constraint_fallbacks = 0
        for _ in range(self.sqp_iterations):
            # Linearize along the (dynamically consistent) rollout.
            _, A, B = self._lin[mode](X_bar[:, :N], U_bar, par)
            A = np.asarray(A).reshape(NX, N, NX)     # [A_0 | A_1 | ...]
            B = np.asarray(B).reshape(NX, N, NU)

            # Condense: dX_k+1 = Gamma_k+1 dU  (dx_0 = 0)
            Gamma = np.zeros((N, NX, NU * N))
            G_k = np.zeros((NX, NU * N))
            for k in range(N):
                G_k = A[:, k, :] @ G_k
                G_k[:, NU * k:NU * (k + 1)] += B[:, k, :]
                Gamma[k] = G_k
            Gamma = Gamma.reshape(N * NX, NU * N)

            err = (X_bar[:, 1:] - ref[:, 1:]).T.ravel()
            u_flat = U_bar.T.ravel()
            GQ = Gamma.T * Q
            H = GQ @ Gamma + self.w.u * np.eye(NU * N) + self.w.du * D.T @ D
            g = (GQ @ err + self.w.u * (u_flat - hover)
                 + self.w.du * D.T @ (D @ u_flat - e_prev))

            # Linearized tilt constraints: |x_bar + Gamma dU| <= max_tilt
            rows = (np.arange(N)[:, None] * NX + np.array([6, 7])[None]).ravel()
            tilt_bar = X_bar[6:8, 1:].T.ravel()
            sol = self._qp_con(h=H, g=g, a=Gamma[rows],
                               lba=-self.max_tilt - tilt_bar, uba=self.max_tilt - tilt_bar,
                               lbx=-u_flat, ubx=self.u_max - u_flat)
            ok = self._qp_con.stats()["success"]
            if not ok:
                constraint_fallbacks += 1
                sol = self._qp(h=H, g=g, lbx=-u_flat, ubx=self.u_max - u_flat)
                ok = self._qp.stats()["success"]
            dU = np.asarray(sol["x"]).ravel().reshape(N, NU).T
            if not (ok and np.all(np.isfinite(dU))):
                qp_failures += 1
                break

            # Backtracking line search on the nonlinear cost + tilt violation.
            improved = False
            for step in (1.0, 0.5, 0.25, 0.1):
                U_try = np.clip(U_bar + step * dU, 0.0, self.u_max)
                X_try = self.rollout(mode, x0, U_try, payload)
                M_try = merit(X_try, U_try) if np.all(np.isfinite(X_try)) else np.inf
                if M_try < M_bar:
                    U_bar, X_bar, M_bar = U_try, X_try, M_try
                    improved = True
                    break
            if not improved:
                break

        self._plan = (U_bar, mode)
        return X_bar, U_bar, {
            "solve_time": time.perf_counter() - t_start,
            "qp_failures": qp_failures,
            "constraint_fallbacks": constraint_fallbacks,
            "max_tilt_pred": float(np.abs(X_bar[6:8, 1:]).max()),
            "cost": M_bar,
        }


################################################################################
# AGENT (use like an RL policy, e.g. from utilities/play.py)
################################################################################

YAW_RATE_REF = 0.8               # [rad/s] slew rate of the yaw reference
# Payload hinge constants of the prediction model, identified from logged
# MuJoCo data (replay error): the payload swings freely about the cylinder
# axis in the hook pocket (alpha, damped); rocking it sideways (beta) is
# resisted by the hook contacts. Overridable per agent (pendulum_params).
PAYLOAD_DAMPING = 0.05           # alpha damping [Nms/rad]
PAYLOAD_BETA_STIFFNESS = 1.0     # beta stiffness [Nm/rad]
PAYLOAD_BETA_DAMPING = 0.1       # beta damping [Nms/rad]


class HybridMPCAgent(MissionAgent):
    """Rotor-level hybrid MPC flying the pick-and-place mission in
    ``AdaptiveTransportAviary`` (actions: 4 motor commands + 2 tendons)."""

    def __init__(self, env, horizon=40, max_time=45.0, verbose=False, max_tilt=np.pi / 2,
                 pendulum_params=None):
        super().__init__(env, horizon, max_time, verbose)
        # (alpha damping, beta stiffness, beta damping)
        self.pendulum_params = pendulum_params or (
            PAYLOAD_DAMPING, PAYLOAD_BETA_STIFFNESS, PAYLOAD_BETA_DAMPING)
        self.drone = drone_params_from_env(env)
        self.mpc = HybridMPC(self.drone, self.dt, horizon=horizon, max_tilt=max_tilt)

    def _reset_controller(self):
        self.mpc.reset()
        self.payload = PayloadParams()
        self.u_prev = np.full(4, self.mpc.hover_thrust(0.0))

    def _measure(self, attached):
        return measure_state(self.env, attached)

    def _control(self, mode, x, p_ref, v_ref):
        if mode == 1:
            # the cylinder can slide in the hook pocket: track the pivot
            self.payload.pivot = 0.8 * self.payload.pivot + 0.2 * measure_pivot(self.env)
        # Slew the yaw reference to 0 (hook curl plane facing the payload).
        yaw_err = np.arctan2(np.sin(-x[8]), np.cos(-x[8]))
        steps = np.arange(self.horizon + 1) * self.dt * YAW_RATE_REF
        yaw_ref = x[8] + np.sign(yaw_err) * np.minimum(steps, abs(yaw_err))
        X_pred, U_plan, info = self.mpc.solve(mode, x, p_ref, v_ref, yaw_ref,
                                              self.u_prev, self.payload)
        u = U_plan[:, 0]
        self.u_prev = u
        tendon = self.mission.tendon
        action = np.hstack([thrusts_to_action(u, self.drone), tendon, -tendon])
        return action, u, X_pred, info

    def _plant_step(self, action):
        # BaseAviary.step applies the tendon command directly (the transport
        # env's own step gates it with its waypoint/GRAB_FLAG logic).
        return BaseAviary.step(self.env, action)

    def _on_attach(self):
        b_a, k_b, b_b = self.pendulum_params
        self.payload = payload_params_from_env(self.env, damping=b_a, beta_stiffness=k_b,
                                               beta_damping=b_b)
        p = self.payload
        self._say(f"payload attached -> switching to drone+pendulum model"
                  f"  (m_p={p.mass:.3f} kg, l={p.length:.3f} m, pivot={np.round(p.pivot, 3)})")

    def _replay(self, mode, x0, U):
        return self.mpc.rollout(mode, x0, U, self.payload if mode else PayloadParams())

    def results(self):
        out = super().results()
        out["payload"] = self.payload
        return out
