"""Director environment with an MPC safety filter on the velocity commands.

The controller (e.g. the RL director policy) acts on this environment exactly
as on ``AdaptiveTransportDirectorAviary``. Before a command reaches the
system, a predictive safety filter (the director MPC's identified model and
constraints, see ``SafetyFilter`` in director_mpc.py) checks it:

  * if the command keeps the predicted drone within the constraints over the
    horizon, it is applied unchanged,
  * otherwise the MPC replaces it with the closest admissible command
    (warm started with the controller's command).

``info`` of every step reports ``safety_intervened``, ``u_rl`` (proposed) and
``u_applied`` (sent to the system).
"""

import numpy as np

from multi_drone_mujoco.envs.adaptive_hook_director_velocity import (
    AdaptiveTransportDirectorAviary,
)
from multi_drone_mujoco.envs.director_mpc import (
    DEFAULT_MODEL_FILE,
    SafetyConstraints,
    SafetyFilter,
    StateTracker,
    load_models,
)
from multi_drone_mujoco.envs.mpc_mission import payload_attached


class AdaptiveTransportDirectorAviarySafetyFilter(AdaptiveTransportDirectorAviary):
    """AdaptiveTransportDirectorAviary + predictive MPC safety filter."""

    def __init__(self, *args, model_file=DEFAULT_MODEL_FILE, constraints=None,
                 filter_horizon=30, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            models = load_models(model_file)
        except (FileNotFoundError, ValueError) as err:
            from multi_drone_mujoco.envs.director_mpc import identify
            print(f"[SafetyFilter] {err}\n  -> identifying the model now (about 2 minutes)...")
            models = identify(model_file=model_file)
        self.safety_filter = SafetyFilter(models, horizon=filter_horizon,
                                          constraints=constraints or SafetyConstraints())
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
        self.intervention_count = 0
        self.filter_steps = 0
        return obs, info

    def step(self, action):
        action = np.array(action, dtype=float).copy()
        x = self._tracker.measure(self._attached)
        u_rl = np.clip(action[0:3], -1.0, 1.0)
        u, finfo = self.safety_filter.filter(int(self._attached), x, u_rl, self._u_prev)
        action[0:3] = u
        self._u_prev = u
        self._tracker.advance(x)

        obs, reward, terminated, truncated, info = super().step(action)

        self._attached = payload_attached(self, self._rest_z, self._attached)
        self.filter_steps += 1
        self.intervention_count += int(finfo["intervened"])
        info.update(
            safety_intervened=bool(finfo["intervened"]),
            u_rl=u_rl,
            u_applied=np.asarray(u, dtype=float),
            safety_slack=finfo["slack"],
            payload_attached=self._attached,
        )
        return obs, reward, terminated, truncated, info
