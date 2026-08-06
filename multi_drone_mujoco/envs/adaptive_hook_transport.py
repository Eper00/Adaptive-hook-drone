"""Fly-Through Aviary: navigate through waypoints/gates.

Task: fly through a sequence of waypoints as quickly as possible.
"""

from turtle import pos

import numpy as np
import mujoco
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class AdaptiveTransportAviary(BaseAviary):
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
        self.min_mass =0.05
        self.max_mass = 0.3
        self.min_radius=0.02
        self.max_radius=0.04
        self.goal_random_amplitude=1
        self.EPISODE_LEN_SEC = 10
        self.WAYPOINT_RADIUS = waypoint_radius
        self.TARGET_POSTION = [1.0, 0.0, 0.6]
        self.GOAL_POSTION = [2.0, 0.0, 1.0]
        self.RADIUS=0.05
        self.MASS=0.2
        self.GRAB_FLAG=False
        self.GRAB_FLAG_ENABLE=False
        if waypoints is None:
            self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                self.TARGET_POSTION,
                self.GOAL_POSTION,
                
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
            transport_target=True
        )
    def reset(self, seed=None, options=None):
        self.GRAB_FLAG=False
        super().reset(seed=seed, options=options)
        self.current_waypoint_idx[:] = 0
        random_x=np.random.uniform(-1,1)
        random_y=np.random.uniform(-1,1)
        random_z=np.random.uniform(0.45,0.8)
        
        self.TARGET_POSTION=[random_x,random_y,random_z]

        random_x=np.random.uniform(-self.goal_random_amplitude,self.goal_random_amplitude)
        random_y=np.random.uniform(-self.goal_random_amplitude,self.goal_random_amplitude)
        self.GOAL_POSTION =[random_x,random_y,1]
        self.WAYPOINTS = np.array([
                [0.0, 0.0, 1.0],
                self.TARGET_POSTION,
                self.GOAL_POSTION,
                ])
        
        
        self.model.site_pos[self.goal_id] = self.GOAL_POSTION
        self.data.qpos[
            self.target_qpos_adr:self.target_qpos_adr+3
        ] = [self.TARGET_POSTION[0],self.TARGET_POSTION[1],self.TARGET_POSTION[2]-0.2]

        
        self.MASS=np.random.uniform(self.min_mass,self.max_mass)
        self.RADIUS=np.random.uniform(self.min_radius,self.max_radius)
        self.model.geom_size[self.target_geom_id] = [
            self.RADIUS,
            0.12,
            0
        ]
        self.model.body_pos[self.holder_body_id][2] = -(self.TARGET_POSTION[2]-0.25)


       
        # szürke tartó tömege
        self.model.body_mass[self.holder_body_id] = self.MASS
        red_bottom = -self.RADIUS

        # holder felső széle
        holder_top = -(self.TARGET_POSTION[2]-0.25) 


        # távolság a két henger között
        connector_length = abs(red_bottom - holder_top)


        # MuJoCo box size harmadik értéke félmagasság
        connector_half_height = connector_length / 2+0.005


        # connector középpontja
        connector_z = (red_bottom + holder_top) / 2+0.005


        # bal és jobb tartó pozíció
        self.model.body_pos[self.left_connector_id][2] = connector_z
        self.model.body_pos[self.right_connector_id][2] = connector_z
        self.model.body_pos[self.left_connector_id][0] = 0.1
        self.model.body_pos[self.right_connector_id][0] = -0.1

        # bal és jobb tartó méret
        self.model.geom_size[self.left_connector_geom_id] = [
            0.005,
            0.015,
            connector_half_height
        ]

        self.model.geom_size[self.right_connector_geom_id] = [
            0.005,
            0.015,
            connector_half_height
        ]
        mujoco.mj_forward(self.model, self.data)
        
        self.tendon_orientation=-1
        
        return self._computeObs(), self._computeInfo()

    def step(self, action):
        action = action.copy()
        if self.GRAB_FLAG_ENABLE:
            self._update_grab_flag()
      
      

            # 1-es waypointnál próbál fogni
            if self.GRAB_FLAG and self.current_waypoint_idx[0] == 1:

                payload_pos = self.data.qpos[
                    self.target_qpos_adr:self.target_qpos_adr+3
                ]

                hook_pos = self.data.xpos[self.segment_2_id].copy()

                if payload_pos[1] < hook_pos[1]:
                    action[4] = 1
                    action[5] = -1
                    self.tendon_orientation=1
                else:
                    action[4] = -1
                    action[5] = 1
                    self.tendon_orientation=-1

            

            else:
                # nincs hook használat curriculum szinten
                action[4:] = 0
            if self.current_waypoint_idx[0] == 2:
                if self.tendon_orientation==1:
                    action[4] = 1
                    action[5] = -1
                else:
                    action[4] = -1
                    action[5] = 1
        else:
                       
            action[4:] = 0
            self.GRAB_FLAG = False

        obs, rewards, terminated, truncated, infos = super().step(action)
        return obs, rewards, terminated, truncated, infos
    def _update_grab_flag(self):

        if self.GRAB_FLAG:
            return

        payload_pos = self.data.qpos[
            self.target_qpos_adr:self.target_qpos_adr+3
        ]

        hook_pos = self.data.xpos[self.segment_2_id].copy()

        error = np.linalg.norm(payload_pos-hook_pos)

        if error < self.RADIUS + 0.02:
            self.GRAB_FLAG = True
    def _actionSpace(self):
        """Normalized [-1, 1] → mapped to RPM internally."""
        return spaces.Box(low=-np.ones(6, dtype=np.float32), high=np.ones(6, dtype=np.float32))

    def _observationSpace(self): 
        obs_lower_pos = np.full(24, -np.inf)
        obs_upper_pos = np.full(24 , np.inf)
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
            pay_load_grab_position=self.data.qpos[self.target_qpos_adr:self.target_qpos_adr+3]
            segment_2_position = self.data.xpos[self.segment_2_id].copy()
            state = self._getDroneStateVector(i)
            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            rel_wp = wp - self.pos[i]
            rel_gb = pay_load_grab_position - segment_2_position
            obs_list.append(np.hstack([
                state[0:3], state[7:10], state[10:13], state[13:16], wp, rel_wp, pay_load_grab_position,rel_gb, state[-2:]
            ]))
        return np.concatenate(obs_list).astype(np.float32)
    

    def _computeReward(self, action):
        total = 0.0

        for i in range(self.NUM_DRONES):

            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            payload_pos = self.data.qpos[
                                                self.target_qpos_adr:self.target_qpos_adr+3
                                            ]
                        
            hook_pos = self.data.xpos[self.segment_2_id].copy()
                        
            height_error = abs(self.pos[i][2] - wp[2])
            xy_error = np.linalg.norm(self.pos[i][0:2] - wp[0:2])
            smooth_penalty = np.linalg.norm(self.ang_v[i])
            stability_penalty = np.linalg.norm(self.rpy[i][0:2])
            
            
            payload_error = np.linalg.norm(payload_pos - hook_pos)
            reached_waypoint = (
                height_error < self.WAYPOINT_RADIUS / 10
                and xy_error < self.WAYPOINT_RADIUS
            )
            # -------------------------------
            # Curriculum nélkül
            # -------------------------------
            if not self.GRAB_FLAG_ENABLE:

                if reached_waypoint:
                    if wp_idx==0:
                        total += 20.0
                    elif wp_idx==1:

                        total += 20.0  # Big bonus for reaching waypoint
                    elif wp_idx==2:
                        total += 5.0  # Big bonus for reaching waypoint
                    
                    self.current_waypoint_idx[i] = min(
                        self.current_waypoint_idx[i] + 1,
                        len(self.WAYPOINTS) - 1
                    )


            # -------------------------------
            # Curriculum: payload pickup
            # -------------------------------
            else:

                # WP0 és WP2 normál waypoint
                if wp_idx == 0:


                    if reached_waypoint:
                        total += 20.0

                        self.current_waypoint_idx[i] = min(
                            self.current_waypoint_idx[i] + 1,
                            len(self.WAYPOINTS) - 1
                        )


                # WP1 = pickup waypoint
                elif wp_idx == 1:

                  

                   

                    # húzza a hookot a payload felé
                    total -= payload_error


                    # csak sikeres fogás után mehet tovább
                    if self.GRAB_FLAG:

                        total += 20.0

                        self.current_waypoint_idx[i] = min(
                            self.current_waypoint_idx[i] + 1,
                            len(self.WAYPOINTS) - 1
                        )

                elif wp_idx == 2:

                    if reached_waypoint:
                        total += 5.0
                        
                        
                                            
                    total -= payload_error
                   
            # ---------------------------------
            # Általános shaping reward
            # ---------------------------------
            
            total -= 0.03 * stability_penalty
            total -= 0.06 * smooth_penalty
            total -= height_error  # Approach reward
            total -= 0.1 * xy_error  # Approach reward

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
