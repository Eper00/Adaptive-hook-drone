"""Shared parts of the director MPC (waypoint_mpc.py) and the safety filter.

``AdaptiveTransportDirectorAviary`` takes velocity commands, low-pass filters
them (c_k+1 = alpha a_k + (1 - alpha) c_k) and feeds them to a PPO velocity
controller that produces the rotor commands. The director MPC
(waypoint_mpc.py) and the predictive safety filter therefore predict the
closed loop

    command -> known low-pass filter -> RL velocity loop -> drone (+ payload)

with the ResNet model of resnet_mpc.py. This module holds what they share:
the state layout, the plant interface (``director_step``), the measurement of
the state (``measure_state``), the MPC weights, the grasp geometry
(``grasp_pose``, also used for the transport env's waypoints), the payload /
contact detection, the waypoint target of the MPC (``WaypointTarget``) and
the P-controller data collection used for training the model.

State (19):  [p(3), v(3), c(3), theta(2), theta_dot(2),
              roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot]
             c:       output of the env's command low-pass filter (what the
                      PPO velocity controller tracks)
             theta:   payload swing in the world x-z / y-z planes (mode 1)
             roll, pitch: heading-independent tilt about the world x / y axes
             yaw:     heading (the PPO velocity controller holds it at 0)
These are the states the closed loop depends on (the PPO controller sees
rpy, v, angular velocity, its velocity target c and the tendon lengths;
the motors have no lag), so no delayed copies are needed.
Input (3):   velocity command a in [-1, 1]^3 (env action[0:3]).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary

# Grasp geometry (drone position relative to the resting cylinder centre).
# The hook curls towards the drone's -y side for a positive tendon command,
# so the drone hovers on the +y side of the cylinder with yaw = 0.
GRASP_SIDE_CLEARANCE = 0.025     # dy = payload radius + clearance
GRASP_HEIGHT = 0.225             # drone height above the cylinder centre
HOOK_LENGTH = 0.433              # drone COM -> hook tip (straight hook)
TIP_FLOOR_CLEARANCE = 0.03
APPROACH_HEIGHT = 0.35           # pre-grasp waypoint: this much above the grasp pose
GRASP_SNAP = 0.03                # [m] drone-grasp waypoint distance where the hook target takes over

NX = 19
NU = 3
P, V, C, TH, THD, ATT, ATTD = (
    slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 11), slice(11, 13),
    slice(13, 16), slice(16, 19))
TILT = slice(13, 15)            # roll, pitch (the part of ATT that is constrained)


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


def attitude(env):
    """Attitude [roll_w, pitch_w, yaw] and its rate [roll_w_dot, pitch_w_dot, yaw_dot].

    roll_w / pitch_w are the heading-independent tilt about the world x / y
    axes, computed from the direction of the body z axis in the world frame:
    equal to (roll, pitch) at yaw 0, and consistent with the world-frame
    velocities when the drone is rotated (world x motion <-> pitch_w, world y
    motion <-> roll_w). yaw is the ZYX yaw the PPO controller sees. The rates
    follow from the body angular velocity (free joint qvel, body frame).
    """
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, env.quat[0])
    R = R.reshape(3, 3)
    w_body = env.ang_v[0]
    z = R[:, 2]
    z_dot = np.cross(R @ w_body, z)
    roll_w = -np.arctan2(z[1], z[2])
    pitch_w = np.arctan2(z[0], z[2])
    roll_w_dot = -(z[2] * z_dot[1] - z[1] * z_dot[2]) / max(z[1] ** 2 + z[2] ** 2, 1e-9)
    pitch_w_dot = (z[2] * z_dot[0] - z[0] * z_dot[2]) / max(z[0] ** 2 + z[2] ** 2, 1e-9)
    roll, pitch, yaw = env.rpy[0]
    _, q, r = w_body
    yaw_dot = (q * np.sin(roll) + r * np.cos(roll)) / max(np.cos(pitch), 1e-6)
    return np.array([roll_w, pitch_w, yaw]), np.array([roll_w_dot, pitch_w_dot, yaw_dot])


def measure_state(env, attached):
    """The 19-dim model state from the MuJoCo measurements (swing only when
    the payload is attached, zeros otherwise)."""
    x = np.zeros(NX)
    x[P] = env.pos[0]
    x[V] = env.vel[0]
    x[C] = filter_state(env)
    if attached:
        x[TH], x[THD] = measure_swing(env)
    x[ATT], x[ATTD] = attitude(env)
    return x


################################################################################
# MPC SETTINGS
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


U_MAX = 0.8
# Close the hook harder once the payload is off its stand: with the grasp
# command (0.6) thin cylinders can slide out of the hook while hovering.
TENDON_HOLD = 1.0


################################################################################
# GRASP GEOMETRY, PAYLOAD AND CONTACT DETECTION
################################################################################

def grasp_pose(payload_center, radius):
    """Drone position for grasping a cylinder resting at ``payload_center``
    (the transport env's grasp waypoint). The drone hovers on the +y side
    (yaw 0), so a positive tendon command curls the hook under the cylinder;
    its height keeps the straight hook tip off the floor."""
    c = np.asarray(payload_center, float)
    dy = radius + GRASP_SIDE_CLEARANCE
    h = max(GRASP_HEIGHT, HOOK_LENGTH + TIP_FLOOR_CLEARANCE - c[2])
    return c + np.array([0.0, dy, h])


def payload_attached(env, rest_z, attached):
    """Payload hangs on the hook: lifted off its stand and touching the hook
    (stays attached until it is more than 0.7 m from the drone)."""
    if rest_z is None:
        return False
    m, d = env.model, env.data
    a = env.target_qpos_adr
    payload = d.qpos[a:a + 3]
    if np.linalg.norm(payload - d.qpos[0:3]) > 0.7:
        return False
    if attached:
        return True
    target_geom = env.target_geom_id
    hook_geoms = {m.geom(f"segment_geom_{i}").id for i in range(1, 8)}
    touching = any(
        (c.geom1 == target_geom and c.geom2 in hook_geoms)
        or (c.geom2 == target_geom and c.geom1 in hook_geoms)
        for c in d.contact[:d.ncon]
    )
    return touching and payload[2] > rest_z + 0.02


def hook_payload_contact(env):
    """Any contact between the hook and the payload (cylinder, holder, connectors)."""
    m, d = env.model, env.data
    hook = {m.geom(f"segment_geom_{i}").id for i in range(1, 8)}
    payload = {env.target_geom_id, env.left_connector_geom_id,
               env.right_connector_geom_id, m.geom("holder_plate").id}
    return any((c.geom1 in hook and c.geom2 in payload) or (c.geom2 in hook and c.geom1 in payload)
               for c in d.contact[:d.ncon])


class Segment:
    """Straight-line reference with a trapezoidal velocity profile (used by
    the safety filter's stall takeover)."""

    def __init__(self, start, goal, t0, v_max, a_max=0.6):
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.t0 = t0
        d = self.goal - self.start
        self.length = np.linalg.norm(d)
        self.dir = d / self.length if self.length > 1e-9 else np.zeros(3)
        self.v = min(v_max, np.sqrt(self.length * a_max))
        self.a = a_max
        self.t_acc = self.v / a_max
        self.T = (self.length / self.v + self.t_acc) if self.length > 1e-9 else 0.0

    def __call__(self, t):
        tau = np.clip(t - self.t0, 0.0, self.T)
        if self.T == 0.0:
            return self.goal.copy(), np.zeros(3)
        if tau < self.t_acc:
            s, v = 0.5 * self.a * tau ** 2, self.a * tau
        elif tau < self.T - self.t_acc:
            s, v = 0.5 * self.a * self.t_acc ** 2 + self.v * (tau - self.t_acc), self.v
        else:
            r = self.T - tau
            s, v = self.length - 0.5 * self.a * r ** 2, self.a * r
        return self.start + s * self.dir, v * self.dir


class WaypointTarget:
    """Position target of the MPC / P-controller: the env's active waypoint.

    The env only drives the tendons once hook segment 2 is within R + 2 cm of
    the payload, but at the grasp waypoint itself it is ~5 cm away. So once
    the drone is within GRASP_SNAP of the grasp waypoint, the target is moved
    by the measured hook-payload offset until segment 2 is R + 1 cm from the
    payload (the RL observes the same offsets, payload - segment_j, and
    learned to stop closer). Call ``reset()`` at the start of an episode.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.hook = None

    def __call__(self, env, attached):
        idx = int(min(env.current_waypoint_idx[0], len(env.WAYPOINTS) - 1))
        w = np.asarray(env.WAYPOINTS[idx], float)
        grasp_idx = 2 if env.PAYLOAD_TERMINATION else 1
        if idx != grasp_idx or attached:
            self.hook = None
            return w, idx
        if self.hook is None and np.linalg.norm(env.pos[0] - w) < GRASP_SNAP:
            a = env.target_qpos_adr
            d = env.data.qpos[a:a + 3] - env.data.xpos[env.segment_2_id]
            dist = np.linalg.norm(d)
            self.hook = env.pos[0] + d * max(0.0, 1.0 - (env.RADIUS + 0.01) / dist)
        return (w if self.hook is None else self.hook), idx


################################################################################
# DATA COLLECTION (training data of the ResNet model)
################################################################################

def make_identification_env(seed):
    from gymnasium.utils import seeding

    from multi_drone_mujoco.envs.adaptive_hook_director_velocity import (
        AdaptiveTransportDirectorAviary,
    )
    env = AdaptiveTransportDirectorAviary()
    env.PAYLOAD_TERMINATION = True       # 4 waypoints (pre-grasp included)
    env.np_random, _ = seeding.np_random(seed)
    env.reset(seed=seed)
    env.prev_action = None
    return env


def collect(seed, hold_time=8.0, excite=0.15, max_time=45.0, gain=1.5, verbose=True):
    """Closed-loop data (both without and with the payload): a position
    P-controller flies the env's waypoints (WaypointTarget, tendon as for the
    RL: env logic, command 1), with random piecewise-constant velocity-command
    excitation except while approaching and grasping the payload (pre-grasp
    and grasp waypoints)."""
    env = make_identification_env(seed)
    dt = env.CTRL_TIMESTEP
    target = WaypointTarget()
    rng = np.random.default_rng(seed)
    rest_z = float(env.data.qpos[env.target_qpos_adr + 2])
    grasp_idx = 2 if env.PAYLOAD_TERMINATION else 1
    attached, t, t_goal = False, 0.0, None
    noise, noise_left = np.zeros(3), 0
    log = {k: [] for k in ("x", "u", "attached", "contact", "waypoint")}
    x = measure_state(env, attached)
    while t < max_time:
        w, idx = target(env, attached)
        cmd = gain * (w - x[0:3])
        if not (1 <= idx <= grasp_idx):
            if noise_left <= 0:
                noise = rng.uniform(-excite, excite, 3) * (rng.random() < 0.7)
                noise_left = rng.integers(10, 40)
            noise_left -= 1
            cmd = cmd + noise
        cmd = np.clip(cmd, -U_MAX, U_MAX)
        log["x"].append(x)
        log["u"].append(cmd)
        log["attached"].append(attached)
        log["contact"].append(hook_payload_contact(env) and not attached)
        log["waypoint"].append(idx)
        _, _, terminated, _, _ = env.step(np.r_[cmd, 0.0, 1.0, 0.0])
        t += dt
        attached = payload_attached(env, rest_z, attached)
        x = measure_state(env, attached)
        if attached and idx == len(env.WAYPOINTS) - 1 and t_goal is None \
                and np.linalg.norm(x[0:3] - w) < 0.05:
            t_goal = t
        if t_goal is not None and t - t_goal > hold_time:
            break
        if terminated or x[2] < 0.0 or np.any(np.abs(env.rpy[0, 0:2]) > np.pi / 2):
            if verbose:
                print(f"  seed {seed}: ended at t={t:.2f}s (crash / payload lost)")
            break
    env.close()
    out = {k: np.asarray(v) for k, v in log.items()}
    if verbose:
        print(f"  seed {seed}: {t:.1f}s, goal reached at {t_goal}, attached samples {out['attached'].sum()}")
    return out
