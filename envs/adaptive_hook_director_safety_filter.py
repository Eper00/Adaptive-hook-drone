"""Director environment with a predictive safety filter on the velocity commands.

The controller (e.g. the RL director policy) acts on this environment exactly
as on ``AdaptiveTransportDirectorAviary``. Before a command reaches the
system, the predictive safety filter (``PredictiveSafetyFilter`` in
predictive_safety_filter.py) checks it with an identified model of the closed
loop. By default this is the ResNet model x_k+1 = x_k + f(x_k, u_k), which
predicts the RL flights more accurately than the ARX model
(utilities/analyse_identification.py); ``model="arx"`` selects the ARX model.

  * a command that keeps the predicted drone within the constraints (tilt,
    altitude, speed, payload swing) over the horizon is applied unchanged,
  * otherwise the filter applies the closest admissible command; the tilt is
    soft at ``max_tilt`` (15 deg, this triggers the intervention) and hard at
    ``hard_max_tilt`` (60 deg)                                   ["constraint"],
  * when the hook touches the payload assembly before the payload is lifted
    (or the connectors / holder plate at any time) and the drone starts to
    tilt, the hook is being levered: the filter makes the drone climb off the
    contact before it flips                                      ["contact"],
  * a command that would change the filtered velocity reference faster than
    ``max_ref_accel`` (g tan 15 deg per axis) is clipped to that rate
    (optionally also abrupt jumps / reversals of the raw command)    ["rate"],
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
from multi_drone_mujoco.envs.director_mpc import (
    DEFAULT_MODEL_FILE,
    U_MAX,
    DirectorMPC,
    StateTracker,
    load_models,
)
from multi_drone_mujoco.envs.mpc_mission import Segment, payload_attached
from multi_drone_mujoco.envs.predictive_safety_filter import (
    LinearDynamics,
    PredictiveSafetyFilter,
    SafetyConstraints,
    hook_obstacle_contact,
)
from multi_drone_mujoco.envs.resnet_mpc import MODEL_FILE, ResNetMPC, load_or_train

REASONS = ("none", "constraint", "rate", "stall", "contact")


def load_filter_models(model="resnet", model_file=None):
    """Prediction models of both modes: "resnet" (default) or "arx"."""
    if model == "resnet":
        return load_or_train(model_file or MODEL_FILE)
    if model == "arx":
        model_file = model_file or DEFAULT_MODEL_FILE
        try:
            return load_models(model_file)
        except (FileNotFoundError, ValueError) as err:
            from multi_drone_mujoco.envs.director_mpc import identify
            print(f"[SafetyFilter] {err}\n  -> identifying the model now (about 2 minutes)...")
            return identify(model_file=model_file)
    raise ValueError(f"unknown model '{model}' (use 'resnet' or 'arx')")


class AdaptiveTransportDirectorAviarySafetyFilter(AdaptiveTransportDirectorAviary):
    """AdaptiveTransportDirectorAviary + predictive safety filter."""

    def __init__(self, *args, model="resnet", model_file=None, constraints=None,
                 filter_horizon=20, stall_distance=0.1, stall_time=1.0,
                 waypoint_tolerance=0.25, takeover_time=0.5, takeover_speed=0.4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        models = load_filter_models(model, model_file)
        self.filter_model = model
        # director MPC (same model) that takes over when the controller stalls
        if model == "resnet":
            self.takeover_mpc = ResNetMPC(models, horizon=40, u_max=U_MAX)
            predictors = models
        else:
            self.takeover_mpc = DirectorMPC(models, horizon=40, u_max=U_MAX)
            predictors = [LinearDynamics(m) for m in models]
        self.safety_filter = PredictiveSafetyFilter(predictors, horizon=filter_horizon,
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
        self._tracker = None
        self._attached = False
        self._rest_z = None
        self._u_prev = np.zeros(3)
        self.intervention_count = 0
        self.filter_steps = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._tracker = StateTracker(self)
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
        if self._tracker is not None:
            self._tracker = StateTracker(self)
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
        x = self._tracker.measure(self._attached)
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
        self._tracker.advance(x)

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
