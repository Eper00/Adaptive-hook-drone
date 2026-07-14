"""Fly-Through Aviary: navigate through waypoints/gates.

Task: fly through a sequence of waypoints as quickly as possible.
"""

from turtle import pos

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class FlyThroughAviary(BaseAviary):
    """Fly through waypoints task."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.BB_HOOK,
        num_drones: int = 1,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        waypoints=None,
        waypoint_radius: float = 0.1,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.EPISODE_LEN_SEC = 10
        self.WAYPOINT_RADIUS = waypoint_radius
        self.TARGET_POSTION = [1.0, 0.0, 0.6]
        if waypoints is None:
            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                self.TARGET_POSTION,
                [2.0, 0.0, 1.0],
                
            ])
        else:
            self.WAYPOINTS = np.array(waypoints)

        self.current_waypoint_idx = np.zeros(num_drones if num_drones > 1 else 1, dtype=int)

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.4]])

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs_type=ObservationType.KIN,
            act_type=ActionType.RPM,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
        )

    def reset(self, seed=None, options=None):
        self.current_waypoint_idx[:] = 0
        return super().reset(seed=seed, options=options)

    def _actionSpace(self):
        """Normalized [-1, 1] → mapped to RPM internally."""
        return spaces.Box(low=-np.ones(6, dtype=np.float32), high=np.ones(6, dtype=np.float32))

    def _observationSpace(self): 
        obs_lower_pos = np.full(18, -np.inf)
        obs_upper_pos = np.full(18 , np.inf)
        obs_lower_tendon_lengths = np.full(2, -1)
        obs_upper_tendon_lengths = np.full(2, 1)
        return spaces.Box(low=np.hstack([obs_lower_pos.astype(np.float32),obs_lower_tendon_lengths.astype(np.float32)]),
                               high=np.hstack([obs_upper_pos.astype(np.float32),obs_upper_tendon_lengths.astype(np.float32)]))


    def _preprocessAction(self, action):
        """Convert normalized action to RPMs."""
        action = np.clip(np.array(action).flatten(), -1, 1)
        rpms = self._normalizedActionToRPM(action).reshape(1, 4)
        return rpms

    def _computeObs(self):
        obs_list = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            rel_wp = wp - self.pos[i]
            obs_list.append(np.hstack([
                state[0:3], state[7:10], state[10:13], state[13:16], wp, rel_wp, state[-2:]
            ]))
        return np.concatenate(obs_list).astype(np.float32)

    def _computeReward(self , action):
        total = 0.0
        for i in range(self.NUM_DRONES):
            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            height_error = abs(self.pos[i][2] - wp[2])
            xy_error = np.linalg.norm(self.pos[i][0:2]-wp[0:2])
            # Check waypoint reached
            if height_error < self.WAYPOINT_RADIUS  and xy_error < self.WAYPOINT_RADIUS and self.current_waypoint_idx[i] < len(self.WAYPOINTS):
                self.current_waypoint_idx[i] += 1
                total += 10.0  # Big bonus for reaching waypoint

            if wp==1 and height_error < self.WAYPOINT_RADIUS  and xy_error < self.WAYPOINT_RADIUS:
                total -= 0.1 * (abs(self.rpy[i][0]) + abs(self.rpy[i][1]))
                
            total -= height_error  # Approach reward
            total -= 0.1 * xy_error  # Approach reward
            total -= 0.06 * (abs(self.rpy[i][0]) + abs(self.rpy[i][1]))  # Reward for keeping drone upright
            total -= 0.01 * np.linalg.norm(self.ang_v[i])  # Smooth flight

        if self._computeTerminated():
            total -= 100.0



        return float(total)

    def _computeTerminated(self):
        for i in range(self.NUM_DRONES):
            if self.pos[i, 2] < 0.0:
                return True
            if abs(self.rpy[i, 0]) > np.pi / 2 or abs(self.rpy[i, 1]) > np.pi / 2:
                return True
        return False

    def _computeTruncated(self):
       
        time_up = self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC
        return time_up

    def _computeInfo(self):
        return {
            "waypoints_reached": [int(idx) for idx in self.current_waypoint_idx],
            "total_waypoints": len(self.WAYPOINTS),
        }
