"""Adaptive transport aviary: pick up a payload with the hook and deliver it.

Task: take off to [0, 0, 1], fly to the grasp pose next to the payload,
close the hook around it and carry it to a random goal. The payload is a red
cylinder on a stand (holder plate + connectors) with random mass, radius and
height; the hook is a tendon-driven chain under the drone (see base_aviary).

Action (6):       4 normalized motor commands in [-1, 1] and 2 tendon
                  commands; with GRAB_FLAG_ENABLE only action[4] is used, as
                  the closing magnitude, and the curl direction is chosen from
                  the geometry (see ``step``).
Observation (35): [pos(3), rpy(3), vel(3), ang_vel(3), active waypoint - pos(3),
                  payload - hook segment 2..7 (6 x 3), tendon_lengths(2)].

Curriculum flags (set from utilities/learn.py):
  GRAB_FLAG_ENABLE     the payload has to be picked up (else: waypoints only)
  RANDOM_ORIENTATION   random initial yaw in [-pi, pi) (else yaw 0)
  PAYLOAD_TERMINATION  final task: 4 waypoints
                       (take-off, pre-grasp above the payload, grasp, goal)
                       and termination when the payload is lost at the goal
"""

import numpy as np
import mujoco
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.mpc_mission import APPROACH_HEIGHT, grasp_pose
from multi_drone_mujoco.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)


class AdaptiveTransportAviary(BaseAviary):
    """Pick-and-place transport task with the hook drone."""

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
        # Payload randomization ranges [kg] / [m]
        self.MIN_PAYLOAD_MASS = 0.01
        self.MAX_PAYLOAD_MASS = 0.25
        self.MIN_PAYLOAD_RADIUS = 0.02
        self.MAX_PAYLOAD_RADIUS = 0.04

        self.GOAL_RANDOM_AMPLITUDE = 1.5    # goal x, y drawn from +-amplitude
        self.EPISODE_LEN_SEC = 10
        self.WAYPOINT_RADIUS = waypoint_radius

        # TARGET_POSITION anchors the payload (redrawn at every reset: the
        # cylinder centre is 0.245 m below it); GOAL_POSITION is the drop-off
        self.TARGET_POSITION = [1.0, 0.0, 0.6]
        self.GOAL_POSITION = [2.0, 0.0, 1.0]
        self.TARGET_ORIENTATION = 0   # yaw target handed to the low-level controller
        self.PAYLOAD_RADIUS = 0.05
        self.PAYLOAD_MASS = 0.2
        self.RANDOM_ORIENTATION = True
        self.GRAB_FLAG = False              # payload came close to the hook (see _update_grab_flag)
        self.GRAB_FLAG_ENABLE = True       # curriculum: pick-up part of the task
        self.PAYLOAD_TERMINATION = True    # curriculum: final task (see module docstring)

        if waypoints is None:
            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                self.TARGET_POSITION,
                self.GOAL_POSITION,
            ])
        else:
            self.WAYPOINTS = np.array(waypoints)

        # index of the active waypoint (advanced in _computeReward)
        self.current_waypoint_idx = np.zeros(
            num_drones if num_drones > 1 else 1,
            dtype=int,
        )

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.5]])
        self.DEFAULT_START = np.array(initial_xyzs, dtype=float).reshape(-1, 3)[0].copy()
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
        """Draw a new scenario: initial yaw, payload (position, height, mass,
        radius), goal, and build the waypoints from them."""
        self.GRAB_FLAG = False
        self.INIT_RPYS[0][:] = 0.0
        super().reset(seed=seed, options=options)

        self.current_waypoint_idx[:] = 0

        # The drone starts level at the default position; with
        # RANDOM_ORIENTATION its heading (yaw) is random. It is drawn after
        # the seeded super().reset(), so the scenario is reproducible with
        # the seed.
        yaw = self.np_random.uniform(-np.pi, np.pi) if self.RANDOM_ORIENTATION else 0.0
        start = np.array(self.DEFAULT_START, dtype=float)
        self._place_drone(start, yaw)

        # Payload somewhere around, but not right below the drone
        while True:
            x = start[0] + self.np_random.uniform(-1, 1)
            y = start[1] + self.np_random.uniform(-1, 1)

            if abs(x - start[0]) > 0.2 or abs(y - start[1]) > 0.2:
                break
        # Z sets the payload height: cylinder centre at Z - 0.245 above the floor
        z = self.np_random.uniform(0.45, 0.8)
        goal_x = self.np_random.uniform(-self.GOAL_RANDOM_AMPLITUDE, self.GOAL_RANDOM_AMPLITUDE)
        goal_y = self.np_random.uniform(-self.GOAL_RANDOM_AMPLITUDE, self.GOAL_RANDOM_AMPLITUDE)
        mass = self.np_random.uniform(self.MIN_PAYLOAD_MASS, self.MAX_PAYLOAD_MASS)
        radius = self.np_random.uniform(self.MIN_PAYLOAD_RADIUS, self.MAX_PAYLOAD_RADIUS)
        self.set_scenario((x, y, z), (goal_x, goal_y, 1.0), mass, radius)

        return self._computeObs(), self._computeInfo()

    def set_scenario(self, target_position, goal_position, mass, radius, yaw=None):
        """Place the payload and the goal of a scenario and build the waypoints.

        Called by reset() with the randomly drawn values. It can also be
        called right after reset() to replay a given scenario (e.g. the same
        payload and goal for several controllers); ``yaw`` then re-places
        the drone at the start with that heading (None: keep it).

        target_position: (x, y, Z) of the payload; Z sets its height (the
                         cylinder centre is at Z - 0.245 above the floor)
        goal_position:   drop-off point (x, y, z)
        mass, radius:    payload (holder) mass [kg] and cylinder radius [m]
        """
        if yaw is not None:
            self._place_drone(np.array(self.DEFAULT_START, dtype=float), yaw)
        self.Z = float(target_position[2])
        self.TARGET_POSITION = np.array(target_position, dtype=float)
        self.GOAL_POSITION = np.array(goal_position, dtype=float)

        self.model.site_pos[self.goal_id] = self.GOAL_POSITION   # green goal marker

        self.MASS = float(mass)
        self.RADIUS = float(radius)

        # Payload resting on the floor: the holder plate (half height 0.005)
        # hangs 0.25 - Z below the cylinder, so the cylinder centre is at
        # Z - 0.245 (it used to be spawned 4.5 cm higher and drop).
        payload_center = self.TARGET_POSITION - np.array([0.0, 0.0, 0.245])
        self.data.qpos[
            self.target_qpos_adr:self.target_qpos_adr + 3
        ] = payload_center

        # Waypoints: take-off point, pick-up at the grasp pose (the same
        # geometry the MPC uses), goal.
        self.GRASP_POSITION = grasp_pose(payload_center, self.RADIUS)
        if self.PAYLOAD_TERMINATION:
            pre_target = self.GRASP_POSITION + np.array([0.0, 0.0, APPROACH_HEIGHT])

            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                pre_target,
                self.GRASP_POSITION,
                self.GOAL_POSITION,
            ])
        else:
            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                self.GRASP_POSITION,
                self.GOAL_POSITION,
            ])

        # Payload geometry: cylinder radius (half length 0.12 m), holder plate
        # under it, and the two connectors between them
        self.model.geom_size[self.target_geom_id] = [
            self.RADIUS,
            0.12,
            0,
        ]

        self.model.body_pos[self.holder_body_id][2] = (
            -(self.TARGET_POSITION[2] - 0.25)
        )

        # Holder mass (= payload mass; the cylinder itself weighs 0.01 kg)
        self.model.body_mass[self.holder_body_id] = self.MASS

        red_bottom = -self.RADIUS
        holder_top = -(self.TARGET_POSITION[2] - 0.25)

        connector_length = abs(red_bottom - holder_top)
        connector_half_height = connector_length / 2 + 0.005
        connector_z = (red_bottom + holder_top) / 2 + 0.005

        # Connector positions
        self.model.body_pos[self.left_connector_id][2] = connector_z
        self.model.body_pos[self.right_connector_id][2] = connector_z

        self.model.body_pos[self.left_connector_id][0] = 0.1
        self.model.body_pos[self.right_connector_id][0] = -0.1

        # Connector sizes
        connector_size = [
            0.02,
            0.025,
            connector_half_height,
        ]

        self.model.geom_size[self.left_connector_geom_id] = connector_size
        self.model.geom_size[self.right_connector_geom_id] = connector_size

        mujoco.mj_forward(self.model, self.data)
        self._updateAndStoreKinematicInformation()

    def _place_drone(self, position, yaw=0.0):
        """Move the drone (level, heading ``yaw``, at rest) to ``position`` after reset."""
        self.INIT_XYZS[0] = position
        self.INIT_RPYS[0][:] = [0.0, 0.0, yaw]
        joint = self.model.joint("drone0_joint")
        adr = self.model.jnt_qposadr[joint.id]
        self.data.qpos[adr:adr + 3] = position
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[self.model.jnt_dofadr[joint.id]:self.model.jnt_dofadr[joint.id] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._updateAndStoreKinematicInformation()

    def step(self, action):
        """Apply the motor commands; the tendons are only driven once the
        payload is near the hook (GRAB_FLAG) and the grasp waypoint is
        active, with the magnitude from action[4] and the direction from the
        geometry. Otherwise the tendons are relaxed."""
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
                    # (in the drone body frame: the hook curls in its y-z plane)
                    rot = self.data.xmat[self.model.body("drone0").id].reshape(3, 3)
                    if (rot.T @ (payload_pos - hook_pos))[1] < 0:
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
        """Latch GRAB_FLAG once the payload centre comes within
        RADIUS + 2 cm of hook segment 2 (it stays set for the episode)."""
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
        """Activate the next waypoint (the last one stays active)."""
        self.current_waypoint_idx[drone_idx] = min(
            self.current_waypoint_idx[drone_idx] + 1,
            len(self.WAYPOINTS) - 1,
        )

    def _actionSpace(self):
        """4 normalized motor commands + 2 tendon commands, all in [-1, 1]."""
        return spaces.Box(
            low=-np.ones(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
        )

    def _observationSpace(self):
        """33 unbounded kinematic/relative entries + 2 tendon lengths."""
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
        """See the module docstring for the 35-dim layout (all in the world frame)."""
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
        """Norm of the gaps between the cylinder surface and the hook
        segments 2..7 (small when the hook is wrapped around the payload)."""
        segment_distances=[]
        payload_pos = self.data.qpos[
                self.target_qpos_adr:self.target_qpos_adr + 3
            ]
        for i in range(2, 8):
            segment_pos = self.data.xpos[getattr(self, f"segment_{i}_id")].copy()
            segment_distances.append(np.linalg.norm(payload_pos-segment_pos)-self.RADIUS)
        return np.linalg.norm(segment_distances)
    def _computeReward(self, action):
        """Waypoint bonuses + grasp / carry bonuses + distance and attitude
        shaping, -100 on termination.

        Note: this also advances ``current_waypoint_idx`` when a waypoint is
        reached (within WAYPOINT_RADIUS horizontally and WAYPOINT_RADIUS / 5
        vertically), so it has to run once per step.
        """
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
                height_error < self.WAYPOINT_RADIUS/5 
                and xy_error < self.WAYPOINT_RADIUS
            )

            # -----------------------------------------
            # Without GRAB_FLAG (waypoints only)
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
            # GRAB_FLAG + PAYLOAD_TERMINATION (final task, 4 waypoints:
            # take-off, pre-grasp, grasp, goal)
            # -----------------------------------------
            elif self.PAYLOAD_TERMINATION:



                if wp_idx == 0 or wp_idx == 1:

                    if reached_waypoint:
                        total += 20.0
                        self._advance_waypoint(i)

                elif wp_idx == 2:
                
                    if reached_waypoint:
                        total += 0.1
                                  
                                    # grasp bonus: hook wrapped around the payload
                                    # (0.05 is a tight threshold, 0.2 too loose)
                    if payload_error<0.1:
                        total += 5.0
                        self._advance_waypoint(i)


                elif wp_idx == 3:

                    if payload_error<0.15:
                        total += 3.0
                    if reached_waypoint:
                                            # goal bonus (5 was too large,
                                            # 1 too low)
                        total += 3.0

                total -= 0.03 * payload_error

            # -----------------------------------------
            # GRAB_FLAG without PAYLOAD_TERMINATION (3 waypoints:
            # take-off, grasp, goal)
            # -----------------------------------------
            else:

                if wp_idx == 0:

                    if reached_waypoint:
                        total += 20.0
                        self._advance_waypoint(i)

                elif wp_idx == 1:

                   if reached_waypoint:
                    total += 0.1
                  
                    # grasp bonus (note: here it is only given while the
                    # grasp waypoint is also reached, unlike in the branch above)
                    if payload_error<0.1:
                        total += 5.0
                        self._advance_waypoint(i)

                       

                elif wp_idx == 2:


                    if payload_error<0.15:
                        total += 3.0
                    if reached_waypoint:
                        # goal bonus (5 was too large, 1 too low)
                        total += 3.0
                
                total -= 0.03 * payload_error

            # -----------------------------------------
            # General shaping (all cases)
            # -----------------------------------------

            total -= 0.03 * stability_penalty
            total -= 0.06 * smooth_penalty
            total -= height_error
            total -= 0.1 * xy_error

        if self._computeTerminated():
            total -= 100.0

        return float(total)

    def _computeTerminated(self):
        """Crash (below the floor), flip (|roll| or |pitch| > 90 deg), or, in
        the final task, the payload more than 1 m from the hook while the
        goal waypoint is active."""

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
                    and payload_error > 1
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