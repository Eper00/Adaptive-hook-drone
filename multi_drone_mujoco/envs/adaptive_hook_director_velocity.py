"""Director environment: the transport task commanded through velocities.

The agent (RL director policy or the director MPC) does not command the
rotors. It outputs a velocity command, which is low-pass filtered and handed
to a pre-trained low-level PPO velocity controller (trained on
``AdaptiveVelocityAviary``); that controller produces the rotor commands. The
task itself (take-off -> pick up the payload -> goal, rewards, termination)
is inherited from ``AdaptiveTransportAviary``.

Action (6): [vx, vy, vz] velocity command in [-1, 1] m/s, a heading entry
            (unused: the target yaw is always 0) and 2 tendon commands.
Observation: the transport task's observation (see AdaptiveTransportAviary).
"""


import numpy as np
from gymnasium import spaces
import mujoco
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
from multi_drone_mujoco.envs.adaptive_hook_transport import AdaptiveTransportAviary
from stable_baselines3 import PPO

class AdaptiveTransportDirectorAviary(AdaptiveTransportAviary):
    """Transport task with a velocity-command interface on top of a
    pre-trained low-level velocity controller.

    Note: apart from ``initial_xyzs`` and ``render_mode`` the constructor
    arguments are fixed (BB_HOOK drone, 240 Hz simulation, 48 Hz control).
    """

    def __init__(
        self,
        drone_model = DroneModel.BB_HOOK,
        num_drones = 1,
        physics = Physics.MJC,
        sim_freq = 240,
        ctrl_freq = 48,
        gui = False,
        record = False,
        waypoints = None,
        waypoint_radius = 0.1,
        controller_path = "/home/tomi/Adaptive-hook-drone/results/final/rl_adaptive_velocity_curriculum/final_model.zip",
        initial_xyzs = None,
        render_mode = None,
    ):


      
        
        self.prev_action = None       # state of the command low-pass filter
        self.alpha = 0.1              # filter gain: c_k+1 = alpha * a_k + (1 - alpha) * c_k
        if initial_xyzs is None:
            # 0.5 m: the 0.433 m hook would start inside the floor at 0.4 m
            initial_xyzs = np.array([[0.0, 0.0, 0.5]])
        # low-level velocity controller (PPO policy trained on AdaptiveVelocityAviary)
        self.controller_model = PPO.load(controller_path)

        super().__init__(
            drone_model=DroneModel.BB_HOOK,
            num_drones=1,
            physics=Physics.MJC,
            sim_freq=240,
            ctrl_freq=48,
            gui=False,
            record=False,
            waypoints=None,
            waypoint_radius=0.1,
            initial_xyzs=initial_xyzs,
            initial_rpys=None,
            render_mode=render_mode,
        )

    def reset(self, seed=None, options=None):
        # Clear the command filter, otherwise it carries over between episodes.
        self.prev_action = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """Filter the velocity command, let the low-level policy turn it into
        rotor commands and step the transport task with them."""
        action = action.copy()
        
        # Initialize or apply the low-pass filter (on the whole action,
        # including the tendon commands)
        if self.prev_action is None:
            self.prev_action = action
        else:
            # Exponential moving average filter
            action = self.alpha * action + (1.0 - self.alpha) * self.prev_action
            self.prev_action = action.copy() # Store filtered action for the next step

        # Target velocities are now smoothed, reducing upstream jiggering
        target_vel = action[0:3]
        
        # The low-level velocity policy turns the filtered command into the
        # 4 normalized motor commands
        low_level_obs = self._get_low_level_obs(target_vel, self.TARGET_ORIENTATION)
        low_level_action, obs = self.controller_model.predict(low_level_obs, deterministic=True)
        rpms = low_level_action[0:4]
        # The transport task's step decides whether the tendons are applied
        # (GRAB_FLAG logic) and in which curl direction
        tendon_actions = action[-2:]
        obs, rewards, terminated, truncated, infos = super().step(np.hstack([rpms, tendon_actions]))
        
        # (the filter is cleared in reset())
        return obs, rewards, terminated, truncated, infos


    def _actionSpace(self):
        """[vx, vy, vz] in [-1, 1], heading in [-pi, pi] (unused), 2 tendons in [-1, 1]."""
        velocity_action_low = np.full(3, -1, dtype=np.float32)
        velocity_action_up = np.full(3, 1, dtype=np.float32)

        orientation_low = np.full(1, -np.pi, dtype=np.float32)
        orientation_up = np.full(1, np.pi, dtype=np.float32)

        tendon_action_low = np.full(2, -1, dtype=np.float32)
        tendon_action_up = np.full(2, 1, dtype=np.float32)

        return spaces.Box(
            low=np.hstack([
                velocity_action_low,
                orientation_low,
                tendon_action_low
            ]),
            high=np.hstack([
                velocity_action_up,
                orientation_up,
                tendon_action_up
            ]),
        )

    def _computeReward(self, action):
        """Transport reward minus a penalty on the action.

        Note: ``action`` here is what ``step`` passed to the transport task,
        i.e. [4 motor commands, 2 tendons], so action[0:3] are motor
        commands, not the velocity command.
        """
        total = super()._computeReward(action)
        velocity_penalty = 0.25 * np.sum(np.square(action[0:3]))
        return total - velocity_penalty


    def _get_low_level_obs(self, target_vel, target_orientation):
        """Observation of the low-level velocity policy, in the same layout as
        ``AdaptiveVelocityAviary._computeObs``:
        [rpy(3), vel(3), ang_vel(3), target_vel(3), target_yaw(1), tendon_lengths(2)].
        """
        state = self._getDroneStateVector(0)

        return np.hstack([
            state[7:10],
            state[10:13],
            state[13:16],
            target_vel,
            target_orientation,
            state[-2:],
        ]).astype(np.float32)
