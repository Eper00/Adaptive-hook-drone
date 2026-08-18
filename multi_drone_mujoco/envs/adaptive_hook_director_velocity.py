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
        action=action.copy()
        target_vel = action[0:4]
        
        low_level_obs = self._get_low_level_obs(target_vel)
        low_level_action, obs = self.controller_model.predict(low_level_obs, deterministic=True)
        rpms=low_level_action[0:4]
        tendon_actions=action[-2:]
        obs, rewards, terminated, truncated, infos = super().step(np.hstack([rpms,tendon_actions]))
        
        return obs, rewards, terminated, truncated, infos



 


    def _get_low_level_obs(self, target_vel):
        state = self._getDroneStateVector(0)

        return np.hstack([
            state[7:10],
            state[10:13],
            state[13:16],
            target_vel,
            state[-2:],
        ]).astype(np.float32)
