"""Director MPC: ResNet MPC on the active waypoint, no reference trajectory.

The MPC replaces the RL director policy (it commands the velocity of the
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

with the same ResNet model, solver (Gauss-Newton SQP + DAQP) and command
bound as the director ResNet-MPC, but without its velocity-offset estimate
(disturbance_gain 0: the model is used as identified).
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
SQP_ITERATIONS = 2            # as in the safety filter, for a fair comparison
TENDON_CMD = 1.0              # env action[4]: magnitude 1 = fully closed hook


class WaypointMPCAgent:
    """MPC on the active environment waypoint; acts like the RL policy
    (``action(obs)`` -> env action [vx, vy, vz, heading, tendon, tendon])."""

    def __init__(self, env, model_file=MODEL_FILE, horizon=20, weights=None, max_tilt=MAX_TILT,
                 weights_payload=None):
        self.env = env
        self.models = load_or_train(model_file)
        # weights without / with the payload (mode 0 / 1)
        self.weights = [weights or WAYPOINT_WEIGHTS,
                        weights_payload or weights or WAYPOINT_WEIGHTS_PAYLOAD]
        self.mpc = ResNetMPC(self.models, horizon=horizon, weights=self.weights[0],
                             max_tilt=max_tilt, disturbance_gain=0.0,
                             sqp_iterations=SQP_ITERATIONS)
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
        env = self.env
        self.attached = payload_attached(env, self.rest_z, self.attached)
        mode = int(self.attached)
        x = measure_state(env, self.attached)
        w, _ = self.target(env, self.attached)
        N = self.mpc.N
        p_ref = np.tile(w[:, None], (1, N + 1))
        v_ref = np.zeros((3, N + 1))
        self.mpc.w = self.weights[mode]
        _, U, self.info = self.mpc.solve(mode, x, p_ref, v_ref, self.u_prev)
        u = U[:, 0]
        self.u_prev = u
        return np.r_[u, 0.0, TENDON_CMD, 0.0]
