"""Director environment with a predictive safety filter on the velocity commands.

The controller (e.g. the RL director policy) acts on this environment exactly
as on ``AdaptiveTransportDirectorAviary``. Before a command reaches the
system, the predictive safety filter (``PredictiveSafetyFilter`` in
predictive_safety_filter.py) checks it with an identified model of the closed
loop: the ResNet model x_k+1 = x_k + f(x_k, u_k) of resnet_mpc.py.

  * a command that keeps the predicted drone within the constraints (tilt,
    altitude, speed, payload swing) over the horizon is applied unchanged,
  * otherwise the filter applies the closest admissible command; the tilt is
    soft at ``max_tilt`` (15 deg, this triggers the intervention) and hard at
    ``hard_max_tilt`` (60 deg)                                   ["constraint"],
  * when the hook touches the payload assembly before the payload is lifted
    (or the connectors / holder plate at any time) and the drone starts to
    tilt, the hook is being levered: the filter makes the drone climb off the
    contact before it flips                                      ["contact"],
  * optionally (off by default), a command that would change the filtered
    velocity reference faster than ``max_ref_accel`` per axis is clipped to
    that rate (also abrupt jumps / reversals of the raw command)      ["rate"],
  * if the drone stalls (its position changes by less than
    ``stall_distance`` over ``stall_time`` while it is more than
    ``waypoint_tolerance`` from the active waypoint), a director MPC takes
    over for ``takeover_time`` and flies towards that waypoint; its commands
    pass the same filter. Control returns to the controller after
    ``takeover_time`` or when the waypoint is reached (the env advances it);
    the stall window then starts again                           ["stall"].

``info`` of every step reports ``safety_intervened``, ``safety_reason``
("none" / "constraint" / "rate" / "stall" / "contact"), ``u_rl`` (proposed),
``u_applied`` (sent to the system), ``safety_slack``, ``hook_contact`` and
``payload_attached``.
"""

import numpy as np

from collections import deque

from multi_drone_mujoco.envs.adaptive_hook_director_velocity import (
    AdaptiveTransportDirectorAviary,
)
from multi_drone_mujoco.envs.director_mpc import U_MAX, Segment, measure_state, payload_attached
from multi_drone_mujoco.envs.predictive_safety_filter import (
    PredictiveSafetyFilter,
    SafetyConstraints,
    hook_obstacle_contact,
)
from multi_drone_mujoco.envs.resnet_mpc import MODEL_FILE, ResNetMPC, load_or_train

REASONS = ("none", "constraint", "rate", "stall", "contact")
FILTER_HORIZON = 20             # 0.42 s, also the horizon of the takeover MPC


class AdaptiveTransportDirectorAviarySafetyFilter(AdaptiveTransportDirectorAviary):
    """AdaptiveTransportDirectorAviary + predictive safety filter."""

    def __init__(self, *args, model_file=None, constraints=None,
                 filter_horizon=FILTER_HORIZON, stall_distance=0.1, stall_time=1.0,
                 waypoint_tolerance=0.25, takeover_time=0.5, takeover_speed=0.4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        models = load_or_train(model_file or MODEL_FILE)
        # director MPC (same model) that takes over when the controller stalls
        self.takeover_mpc = ResNetMPC(models, horizon=filter_horizon, u_max=U_MAX)
        self.safety_filter = PredictiveSafetyFilter(models, horizon=filter_horizon,
                                                    constraints=constraints or SafetyConstraints(),
                                                    dt=self.CTRL_TIMESTEP, alpha=self.alpha)
        self.stall_distance = stall_distance
        self.stall_time = stall_time
        self.waypoint_tolerance = waypoint_tolerance
        self.takeover_time = takeover_time
        self.takeover_speed = takeover_speed
        self._takeover = None           # (segment, start time, waypoint index)
        # positions over the last stall_time (one per control step)
        self._pos_hist = deque(maxlen=int(round(stall_time / self.CTRL_TIMESTEP)) + 1)
        self.reason_count = {r: 0 for r in REASONS}
        self._attached = False
        self._rest_z = None
        self._u_prev = np.zeros(3)
        self.intervention_count = 0
        self.filter_steps = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._attached = False
        self._rest_z = float(self.data.qpos[self.target_qpos_adr + 2])
        self._u_prev = np.zeros(3)
        self.safety_filter.reset()
        self.takeover_mpc.reset()
        self._takeover = None
        self._pos_hist.clear()
        self.intervention_count = 0
        self.filter_steps = 0
        self.reason_count = {r: 0 for r in REASONS}
        return obs, info

    def set_scenario(self, *args, **kwargs):
        """Replay a given scenario (see AdaptiveTransportAviary.set_scenario)
        and re-initialize the filter's measurements for it."""
        super().set_scenario(*args, **kwargs)
        if self._rest_z is not None:
            self._rest_z = float(self.data.qpos[self.target_qpos_adr + 2])

    def _update_takeover(self):
        """Stall detection: start / end the director-MPC takeover."""
        t = self.step_counter / self.SIM_FREQ
        idx = int(min(self.current_waypoint_idx[0], len(self.WAYPOINTS) - 1))
        waypoint = np.asarray(self.WAYPOINTS[idx], float)
        pos = self.pos[0].copy()
        if self._takeover is not None:
            _, t0, wp_idx = self._takeover
            if idx != wp_idx or t - t0 >= self.takeover_time:
                self._takeover = None
                self._pos_hist.clear()      # a new stall needs a full window again
            return
        self._pos_hist.append(pos)
        stalled = (len(self._pos_hist) == self._pos_hist.maxlen
                   and np.linalg.norm(pos - self._pos_hist[0]) < self.stall_distance
                   and np.linalg.norm(pos - waypoint) > self.waypoint_tolerance)
        if stalled:
            self._takeover = (Segment(pos, waypoint, t, self.takeover_speed), t, idx)
            self.takeover_mpc.reset()

    def _takeover_command(self, mode, x):
        """Director-MPC command towards the active waypoint."""
        segment, _, _ = self._takeover
        t = self.step_counter / self.SIM_FREQ
        N, dt = self.takeover_mpc.N, self.CTRL_TIMESTEP
        p_ref = np.zeros((3, N + 1))
        v_ref = np.zeros((3, N + 1))
        for k in range(N + 1):
            p_ref[:, k], v_ref[:, k] = segment(t + k * dt)
        _, U, _ = self.takeover_mpc.solve(mode, x, p_ref, v_ref, self._u_prev)
        return U[:, 0]

    def step(self, action):
        action = np.array(action, dtype=float).copy()
        x = measure_state(self, self._attached)
        mode = int(self._attached)
        u_rl = np.clip(action[0:3], -1.0, 1.0)
        contact = hook_obstacle_contact(self, self._attached)

        self._update_takeover()
        proposal = self._takeover_command(mode, x) if self._takeover is not None else u_rl
        u, finfo = self.safety_filter.filter(mode, x, proposal, self._u_prev, contact=contact)
        reason = finfo["reason"]
        if self._takeover is not None:
            # keep the takeover MPC's offset estimate consistent with what was applied
            self.takeover_mpc._last = (mode, x.copy(), np.asarray(u, float).copy())
            if reason != "contact":
                reason = "stall"
        intervened = reason != "none"
        action[0:3] = u
        self._u_prev = u

        obs, reward, terminated, truncated, info = super().step(action)

        self._attached = payload_attached(self, self._rest_z, self._attached)
        self.filter_steps += 1
        self.intervention_count += int(intervened)
        self.reason_count[reason] += 1
        info.update(
            safety_intervened=intervened,
            safety_reason=reason,
            u_rl=u_rl,
            u_applied=np.asarray(u, dtype=float),
            safety_slack=finfo["slack"],
            hook_contact=contact,
            payload_attached=self._attached,
        )
        return obs, reward, terminated, truncated, info
