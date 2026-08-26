"""Fly-Through Aviary: navigate through waypoints/gates."""

import numpy as np
import mujoco
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)


class AdaptiveTransportAviary(BaseAviary):
    """Fly through waypoints task."""

    def __init__(
        self,
        drone_model=DroneModel.BB_HOOK,
        num_drones=1,
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=48,
        gui=False,
        record=False,
        waypoints=None,
        waypoint_radius=0.1,
        initial_xyzs=None,
        initial_rpys=None,
        render_mode=None,
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
        if initial_rpys is None:
            initial_rpys = np.array([[0.0, 0.0, 0.0]])
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs_type=ObservationType.KIN,
            act_type=ActionType.HOOK,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
            transport_target=True,
        )

    def reset(self, seed=None, options=None):
        self.GRAB_FLAG = False
        
        if self.PAYLOAD_TERMINATION:
            self.INIT_RPYS[0][2] = np.random.uniform(-np.pi,np.pi)
        super().reset(seed=seed, options=options)
       
        self.current_waypoint_idx[:] = 0
       
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)

            if abs(x) > 0.2 or abs(y) > 0.2:
                break

        self.TARGET_POSITION = np.array([
            x,
            y,
            np.random.uniform(0.45, 0.8),
        ])

        self.GOAL_POSITION = np.array([
            np.random.uniform(
                -self.GOAL_RANDOM_AMPLITUDE,
                self.GOAL_RANDOM_AMPLITUDE,
            ),
            np.random.uniform(
                -self.GOAL_RANDOM_AMPLITUDE,
                self.GOAL_RANDOM_AMPLITUDE,
            ),
            1.0,
        ])

        # Waypoints
        if self.PAYLOAD_TERMINATION:
            pre_target = self.TARGET_POSITION + np.array([
                0.0,
                np.random.choice([-0.1, 0.1]),
                0.4,
            ])

            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                pre_target,
                self.TARGET_POSITION,
                self.GOAL_POSITION,
            ])
        else:
            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                self.TARGET_POSITION,
                self.GOAL_POSITION,
            ])

        self.model.site_pos[self.goal_id] = self.GOAL_POSITION

        self.data.qpos[
            self.target_qpos_adr:self.target_qpos_adr + 3
        ] = self.TARGET_POSITION - np.array([0.0, 0.0, 0.2])

        self.MASS = np.random.uniform(
            self.MIN_PAYLOAD_MASS,
            self.MAX_PAYLOAD_MASS,
        )
        self.RADIUS = np.random.uniform(
            self.MIN_PAYLOAD_RADIUS,
            self.MAX_PAYLOAD_RADIUS,
        )

        self.model.geom_size[self.target_geom_id] = [
            self.RADIUS,
            0.12,
            0,
        ]

        self.model.body_pos[self.holder_body_id][2] = (
            -(self.TARGET_POSITION[2] - 0.25)
        )

        # Holder tömege
        self.model.body_mass[self.holder_body_id] = self.MASS

        red_bottom = -self.RADIUS
        holder_top = -(self.TARGET_POSITION[2] - 0.25)

        connector_length = abs(red_bottom - holder_top)
        connector_half_height = connector_length / 2 + 0.005
        connector_z = (red_bottom + holder_top) / 2 + 0.005

        # Connector pozíciók
        self.model.body_pos[self.left_connector_id][2] = connector_z
        self.model.body_pos[self.right_connector_id][2] = connector_z

        self.model.body_pos[self.left_connector_id][0] = 0.1
        self.model.body_pos[self.right_connector_id][0] = -0.1

        # Connector méretek
        connector_size = [
            0.02,
            0.025,
            connector_half_height,
        ]

        self.model.geom_size[self.left_connector_geom_id] = connector_size
        self.model.geom_size[self.right_connector_geom_id] = connector_size

        mujoco.mj_forward(self.model, self.data)


        return self._computeObs(), self._computeInfo()

    def step(self, action):
        action = action.copy()
        # --------------------------------------------------
        # TENDON CONTROL
        # --------------------------------------------------
        
        if self.GRAB_FLAG_ENABLE:

            self._update_grab_flag()

            if self.GRAB_FLAG:

                pickup_idx = 2 if self.PAYLOAD_TERMINATION else 1

                if self.current_waypoint_idx[0] >= pickup_idx:

                    payload_pos = self.data.qpos[
                        self.target_qpos_adr:self.target_qpos_adr + 3
                    ]

                    hook_pos = self.data.xpos[self.segment_2_id].copy()

                    # --------------------------------------
                    # RL action[4] = tendon magnitude
                    # [-1, 1] -> [0, 1]
                    # --------------------------------------
                    magnitude = 0.5 * (action[4] + 1.0)

                    # --------------------------------------
                    # Direction is given by geometry
                    # --------------------------------------
                    if payload_pos[1] < hook_pos[1]:
                        direction = 1.0
                        self.tendon_orientation = 1
                    else:
                        direction = -1.0
                        self.tendon_orientation = -1

                    # --------------------------------------
                    # Apply symmetric tendon action
                    # --------------------------------------
                    action[4] = direction * magnitude
                    action[5] = -direction * magnitude

                else:
                    action[4:] = 0.0

            else:
                action[4:] = 0.0

        else:
            action[4:] = 0.0
            self.GRAB_FLAG = False

        obs, rewards, terminated, truncated, infos = super().step(action)

        return obs, rewards, terminated, truncated, infos

    def _update_grab_flag(self):
        if self.GRAB_FLAG:
            return

        payload_pos = self.data.qpos[
            self.target_qpos_adr:self.target_qpos_adr + 3
        ]

        hook_pos = self.data.xpos[self.segment_2_id].copy()

        error = np.linalg.norm(payload_pos - hook_pos)

        if error < self.RADIUS + 0.02:
            self.GRAB_FLAG = True

    def _advance_waypoint(self, drone_idx):
        self.current_waypoint_idx[drone_idx] = min(
            self.current_waypoint_idx[drone_idx] + 1,
            len(self.WAYPOINTS) - 1,
        )

    def _actionSpace(self):
        return spaces.Box(
            low=-np.ones(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
        )

    def _observationSpace(self):
        obs_lower_pos = np.full(33, -np.inf, dtype=np.float32)
        obs_upper_pos = np.full(33, np.inf, dtype=np.float32)

        obs_lower_tendon_lengths = np.full(2, -1, dtype=np.float32)
        obs_upper_tendon_lengths = np.full(2, 1, dtype=np.float32)

        return spaces.Box(
            low=np.hstack([
                obs_lower_pos,
                obs_lower_tendon_lengths,
            ]),
            high=np.hstack([
                obs_upper_pos,
                obs_upper_tendon_lengths,
            ]),
        )

  

    def _computeObs(self):
        obs_list = []

        for i in range(self.NUM_DRONES):
            payload_pos = self.data.qpos[
                self.target_qpos_adr:self.target_qpos_adr + 3
            ]
            state = self._getDroneStateVector(i)

            wp_idx = min(
                self.current_waypoint_idx[i],
                len(self.WAYPOINTS) - 1,
            )
            wp = self.WAYPOINTS[wp_idx]
            segment_ids = [
                getattr(self, f"segment_{j}_id")
                for j in range(2, 8)
            ]
            rel_grab = [
            payload_pos - self.data.xpos[segment_id].copy()
            for segment_id in segment_ids
        ]
           
            rel_wp = wp - self.pos[i]

            obs_list.append(
                np.hstack([
                    state[0:3],      # 3
                    state[7:10],     # 3
                    state[10:13],    # 3
                    state[13:16],    # 3
                    rel_wp,          # 3
                    *rel_grab,       # 6 × 3
                    state[-2:],      # 2
                ])
            )
        return np.concatenate(obs_list).astype(np.float32)
    def comulative_segment_distance(self):
        segment_distances=[]
        payload_pos = self.data.qpos[
                self.target_qpos_adr:self.target_qpos_adr + 3
            ]
        for i in range(2, 8):
            segment_pos = self.data.xpos[getattr(self, f"segment_{i}_id")].copy()
            segment_distances.append(np.linalg.norm(payload_pos-segment_pos)-self.RADIUS)
        return np.linalg.norm(segment_distances)
    def _computeReward(self, action):
        total = 0.0

        for i in range(self.NUM_DRONES):

            wp_idx = min(
                self.current_waypoint_idx[i],
                len(self.WAYPOINTS) - 1,
            )

            wp = self.WAYPOINTS[wp_idx]

  

            height_error = abs(self.pos[i][2] - wp[2])
            xy_error = np.linalg.norm(
                self.pos[i][0:2] - wp[0:2]
            )

            payload_error = self.comulative_segment_distance()
            
            smooth_penalty = np.linalg.norm(self.ang_v[i])
            stability_penalty = np.linalg.norm(self.rpy[i][0:2])

            reached_waypoint = (
                height_error < self.WAYPOINT_RADIUS / 10
                and xy_error < self.WAYPOINT_RADIUS
            )

            # -----------------------------------------
            # GRAB_FLAG nélkül
            # -----------------------------------------
            if not self.GRAB_FLAG_ENABLE:

                if reached_waypoint:
                    if wp_idx == 0:
                        total += 20.0
                    elif wp_idx == 1:
                        total += 20.0
                    elif wp_idx == 2:
                        total += 5.0

                    self._advance_waypoint(i)

            # -----------------------------------------
            # GRAB_FLAG + PAYLOAD_TERMINATION
            # -----------------------------------------
            elif self.PAYLOAD_TERMINATION:



                if wp_idx == 0 or wp_idx == 1:

                    if reached_waypoint:
                        total += 20.0
                        self._advance_waypoint(i)

                elif wp_idx == 2:
                
                    if reached_waypoint:
                        total += 0.1
                                  
                                    # 0.05 perfect 0.2 too much, do wee need a grab flag?
                    if payload_error<0.1:
                        total += 5.0
                        self._advance_waypoint(i)


                elif wp_idx == 3:

                    if payload_error<0.15:
                        total += 3.0
                    if reached_waypoint:
                                            # 5 is large
                                            # 1 is low
                        total += 3.0

                total -= 0.03 * payload_error

            # -----------------------------------------
            # GRAB_FLAG + nincs PAYLOAD_TERMINATION
            # -----------------------------------------
            else:

                if wp_idx == 0:

                    if reached_waypoint:
                        total += 20.0
                        self._advance_waypoint(i)

                elif wp_idx == 1:

                   if reached_waypoint:
                    total += 0.1
                  
                    # 0.05 perfect 0.2 too much, do wee need a grab flag?
                    if payload_error<0.1:
                        total += 5.0
                        self._advance_waypoint(i)

                       

                elif wp_idx == 2:


                    if payload_error<0.15:
                        total += 3.0
                    if reached_waypoint:
                        # 5 is large
                        # 1 is low
                        total += 3.0
                
                total -= 0.03 * payload_error

            # -----------------------------------------
            # Általános shaping
            # -----------------------------------------

            total -= 0.03 * stability_penalty
            total -= 0.06 * smooth_penalty
            total -= height_error
            total -= 0.1 * xy_error

        if self._computeTerminated():
            total -= 100.0

        return float(total)

    def _computeTerminated(self):

        for i in range(self.NUM_DRONES):

            if self.pos[i, 2] < 0.0:
                return True

            if (
                abs(self.rpy[i, 0]) > np.pi / 2
                or abs(self.rpy[i, 1]) > np.pi / 2
            ):
                return True

            if self.PAYLOAD_TERMINATION:

                payload_pos = self.data.qpos[
                    self.target_qpos_adr:self.target_qpos_adr + 3
                ]

                hook_pos = self.data.xpos[self.segment_2_id].copy()

                payload_error = np.linalg.norm(
                    payload_pos - hook_pos
                )

                if (
                    self.current_waypoint_idx[i]
                    == len(self.WAYPOINTS) - 1
                    and payload_error > 0.2
                ):
                    return True

        return False

    def _computeTruncated(self):

        return (
            self.step_counter / self.SIM_FREQ
            >= self.EPISODE_LEN_SEC
        )

    def _computeInfo(self):

        return {
            "waypoints_reached": [
                int(idx)
                for idx in self.current_waypoint_idx
            ],
            "total_waypoints": len(self.WAYPOINTS),
        }