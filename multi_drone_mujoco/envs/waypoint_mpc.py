"""Director MPC: ResNet MPC on the active waypoint, no reference trajectory.

This is the "MPC director" of the comparison (utilities/analyse.py,
utilities/compare_methods.py: method ``director_mpc``). In the hierarchy

    high level (director): RL director policy  OR  this MPC   -> velocity command
    low level:             PPO velocity controller (fixed)   -> rotor commands

the MPC replaces the RL director policy (it commands the velocity of the
low-level PPO velocity controller) and gets only what the RL director gets:

  * the active waypoint of the environment (take-off -> pre-grasp -> grasp
    pose -> goal), advanced by the environment itself exactly as for the RL
    (waypoint reached / payload grasped),
  * the environment's tendon logic: the tendons are only driven once the
    payload is near the hook and the grasp waypoint is active; the MPC always
    asks for the fully closed hook (tendon command 1). At the grasp waypoint
    the target is moved by the measured hook-payload offset
    (director_mpc.WaypointTarget), because the env closes the hook only
    closer to the payload than the waypoint itself.

Cost over the horizon (ResNetMPC with p_ref = active waypoint, v_ref = 0):

    sum_k |p_k - w|^2_Qp  (x terminal weight at k = N)
        + w_v |v_k|^2 + mode 1: w_s |theta_k|^2 + w_sr |theta_dot_k|^2   (0 by default)
        + w_u |u_k|^2 + w_du |u_k - u_k-1|^2                              (input regularization)

with the ResNet model and solver of resnet_mpc.py (``ResNetMPC``:
Gauss-Newton SQP + DAQP), the model used as identified. There is no
reference trajectory: the reference of every step of the horizon is the
waypoint itself, so the MPC plans how to get there within its constraints
(15 deg tilt, command bound) and the cost weights.

Relation to the other modules:
    director_mpc.py   state layout, measure_state, WaypointTarget, weights
    resnet_mpc.py     the identified model and the MPC solver
    this module       the agent: measure -> choose mode / weights / target -> solve -> act
The safety filter (predictive_safety_filter.py) uses the same model, but
filters the RL director's commands instead of generating its own.
"""

import numpy as np

from multi_drone_mujoco.envs.director_mpc import (NU, DirectorWeights, WaypointTarget,
                                                  measure_state, payload_attached)
from multi_drone_mujoco.envs.resnet_mpc import MODEL_FILE, ResNetMPC, load_or_train

# distance + input regularization only (no velocity or swing terms). The
# horizontal distance weighs more than the vertical one, so the drone keeps
# above the grasp pose while it descends onto the payload. Tuned on seeds
# 0-39 (horizon 20, success = goal + 5 s hold, 2026-10-12): Qp (40, 40, 60),
# w_u 0.05, no tilt limit -> 31/40; tilt 15 deg -> 36; Qp (100, 100, 10) ->
# 37; + w_u 0.5 -> 39 (w_u 2: 38 but 10 s to the goal; larger w_du was worse)
WAYPOINT_WEIGHTS = DirectorWeights(pos=(100.0, 100.0, 10.0), vel=0.0, swing=0.0, swing_rate=0.0,
                                   u=0.5)
# With the payload the vertical distance weighs most: after the grasp the drone
# first climbs to the goal height, then flies over. With the weights above it
# raced horizontally while still low and the hanging payload hit the floor
# (retrained model, seeds 0-39: 35/40 -> 40/40 with these weights).
WAYPOINT_WEIGHTS_PAYLOAD = DirectorWeights(pos=(50.0, 50.0, 100.0), vel=0.0, swing=0.0,
                                           swing_rate=0.0, u=0.5)
MAX_TILT = np.radians(15.0)   # tilt constraint of the MPC (roll, pitch)
SQP_ITERATIONS = 2            # as in the safety filter: both use the same SQP method
TENDON_CMD = 1.0              # env action[4]: magnitude 1 = fully closed hook


class WaypointMPCAgent:
    """MPC on the active environment waypoint; acts like the RL policy
    (``action(obs)`` -> env action [vx, vy, vz, heading, tendon, tendon])."""

    def __init__(self, env, model_file=MODEL_FILE, horizon=20, weights=None, max_tilt=MAX_TILT,
                 weights_payload=None):
        # horizon 20 steps = 0.42 s at 48 Hz (the safety filter's horizon)
        self.env = env
        self.models = load_or_train(model_file)
        # weights without / with the payload (mode 0 / 1)
        self.weights = [weights or WAYPOINT_WEIGHTS,
                        weights_payload or weights or WAYPOINT_WEIGHTS_PAYLOAD]
        self.mpc = ResNetMPC(self.models, horizon=horizon, weights=self.weights[0],
                             max_tilt=max_tilt, sqp_iterations=SQP_ITERATIONS)
        self.reset()

    def reset(self):
        env = self.env
        self.mpc.reset()
        self.attached = False
        self.rest_z = float(env.data.qpos[env.target_qpos_adr + 2])
        self.u_prev = np.zeros(NU)
        self.info = {}
        self.target = WaypointTarget()

    def action(self):
        """One control step: the env action [vx, vy, vz, heading 0, tendon 1, 0]."""
        env = self.env
        # mode switch: once the payload hangs on the hook, the payload model
        # (mode 1, with swing states) and the payload weights are used
        self.attached = payload_attached(env, self.rest_z, self.attached)
        mode = int(self.attached)
        x = measure_state(env, self.attached)
        # target: the env's active waypoint (moved by the hook offset at the grasp)
        w, _ = self.target(env, self.attached)
        N = self.mpc.N
        # constant reference over the horizon: be at w, at rest
        p_ref = np.tile(w[:, None], (1, N + 1))
        v_ref = np.zeros((3, N + 1))
        self.mpc.w = self.weights[mode]
        _, U, self.info = self.mpc.solve(mode, x, p_ref, v_ref, self.u_prev)
        u = U[:, 0]                      # receding horizon: only the first command is applied
        self.u_prev = u
        # the heading command is 0 (the PPO controller holds yaw 0); the tendon
        # command is always "closed": the env only drives the tendons when
        # the payload is in reach (its own grasp logic)
        return np.r_[u, 0.0, TENDON_CMD, 0.0]
