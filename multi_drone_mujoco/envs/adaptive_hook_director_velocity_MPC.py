import numpy as np
import matplotlib.pyplot as plt

from multi_drone_mujoco.envs.adaptive_hook_director_velocity import (
    AdaptiveTransportDirectorAviary,
)
from multi_drone_mujoco.envs.controllers.velocity_mpc import VelocityMPC


MODEL_PATH = (
    "/home/tomi/Adaptive-hook-drone/"
    "results/identification/velocity_increment_model.pth"
)

PPO_PATH = (
    "/home/tomi/Adaptive-hook-drone/"
    "results/final/rl_adaptive_velocity_curriculum/"
    "final_model.zip"
)

CTRL_FREQ = 48
DT = 1.0 / CTRL_FREQ
HORIZON = 5
SIM_TIME = 10.0


class AdaptiveTransportMPC(AdaptiveTransportDirectorAviary):

    def __init__(
        self,
        controller_path=PPO_PATH,
        model_path=MODEL_PATH,
        ctrl_freq=CTRL_FREQ,
        sim_freq=240,
        initial_xyzs=None,
        render_mode=None,
    ):
        super().__init__(
            controller_path=controller_path,
            ctrl_freq=ctrl_freq,
            sim_freq=sim_freq,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
        )

        self.mpc = VelocityMPC(
            model_path=model_path,
            horizon=HORIZON,
            dt=1.0 / ctrl_freq,
            w_position=10.0,
            w_velocity=0.1,
            w_control=0.01,
            w_delta_control=0.1,
            w_terminal=50.0,
            max_velocity=2.0,
        )

        self.previous_target_velocity = np.zeros(3)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.previous_target_velocity = np.zeros(3)
        self.mpc.reset()
        return obs, info

    def get_drone_state(self):
        state = self._getDroneStateVector(0)

        position = np.asarray(
            state[0:3],
            dtype=np.float64,
        )

        velocity = np.asarray(
            state[7:10],
            dtype=np.float64,
        )

        tendon_lengths = np.asarray(
            state[-2:],
            dtype=np.float64,
        )

        return position, velocity, tendon_lengths


    def step(self, action=None):

        # --------------------------------------------------------
        # Current MuJoCo state
        # --------------------------------------------------------

        position, velocity, tendon_lengths = (
            self.get_drone_state()
        )

        # --------------------------------------------------------
        # Current waypoint
        # --------------------------------------------------------

        waypoint_idx = int(
            np.asarray(self.current_waypoint_idx).flat[0]
        )

        goal = np.asarray(
            self.WAYPOINTS[waypoint_idx],
            dtype=np.float64,
        ).reshape(3)

        # --------------------------------------------------------
        # MPC
        # --------------------------------------------------------
        if self.GRAB_FLAG:
            grab_flag=1
        else:
            grab_flag=0

        target_velocity = self.mpc.solve(
            position=position,
            velocity=velocity,
            tendon_lengths=tendon_lengths,
            goal=goal,
            grab_flag=grab_flag,
            previous_target_velocity=self.previous_target_velocity,
        )

        self.previous_target_velocity = target_velocity.copy()

        # --------------------------------------------------------
        # Tendon action
        # --------------------------------------------------------

        tendon_action = np.zeros(2)

        # High-level Director action:
        #
        # [vx_target, vy_target, vz_target,
        #  tendon_1, tendon_2]
        # --------------------------------------------------------

        director_action = np.hstack([
            target_velocity,
            tendon_action,
        ])



        obs, reward, terminated, truncated, info = (
            super().step(director_action)
        )

        # Make MPC output available to the caller.
        info["mpc_target_velocity"] = target_velocity.copy()
        info["mpc_goal"] = goal.copy()

        return obs, reward, terminated, truncated, info


