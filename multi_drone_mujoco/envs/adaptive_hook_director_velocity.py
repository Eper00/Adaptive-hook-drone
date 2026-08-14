"""Fly-Through Aviary: navigate through waypoints/gates.

Task: fly through a sequence of waypoints as quickly as possible.
"""


import numpy as np
from gymnasium import spaces
import mujoco
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary
from multi_drone_mujoco.envs.adaptive_hook_transport import AdaptiveTransportAviary
from stable_baselines3 import PPO

class AdaptiveTransportDirectorAviary(AdaptiveTransportAviary):
    """Fly through waypoints task."""

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
        controller_path = "/home/tomi/Adaptive-hook-drone/results/rl_adaptive_velocity_curriculum/final_model.zip",
        initial_xyzs = None,
        render_mode = None,
    ):
        self.MIN_PAYLOAD_MASS = 0.05
        self.MAX_PAYLOAD_MASS = 0.3
        self.MIN_PAYLOAD_RADIUS = 0.02
        self.MAX_PAYLOAD_RADIUS = 0.04

        self.GOAL_RANDOM_AMPLITUDE = 1.0
        self.EPISODE_LEN_SEC = 10
        self.WAYPOINT_RADIUS = waypoint_radius

        self.TARGET_POSITION = [1.0, 0.0, 0.6]
        self.GOAL_POSITION = [2.0, 0.0, 1.0]

        self.PAYLOAD_RADIUS = 0.05
        self.PAYLOAD_MASS = 0.2

        self.GRAB_FLAG = False
        self.GRAB_FLAG_ENABLE = False
        self.PAYLOAD_TERMINATION = False

        if waypoints is None:
            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                self.TARGET_POSITION,
                self.GOAL_POSITION,
            ])
        else:
            self.WAYPOINTS = np.array(waypoints)

        self.current_waypoint_idx = np.zeros(
            num_drones if num_drones > 1 else 1,
            dtype=int,
        )

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.4]])
        self.controller_model = PPO.load(controller_path)
        self.controller_env = AdaptiveVelocityAviary(ctrl_freq=48, sim_freq=240, render_mode=render_mode)

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

   
    def step(self, action):
        print(action)
        target_vel = action.copy()
        low_level_obs = self._get_low_level_obs(target_vel)
        print(low_level_obs)
        low_level_action, obs = self.controller_model.predict(low_level_obs, deterministic=True)
       
        obs, rewards, terminated, truncated, infos = super().step(low_level_action)
        
        return obs, rewards, terminated, truncated, infos

    def _actionSpace(self):
        # [vel_x, vel_y, vel_z, yaw ]
        return spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))


    def _preprocessAction(self, action):
        velocities = np.clip(np.array(action[:3]).flatten(), -0.5, 0.5)
        yaw = np.clip(action[3:4],-0.25, 0.25)

        return np.hstack([velocities,yaw])


    def _get_low_level_obs(self, target_vel):
        state = self._getDroneStateVector(0)

        return np.hstack([
            state[7:10],
            state[10:13],
            state[13:16],
            target_vel,
            state[-2:],
        ]).astype(np.float32)
