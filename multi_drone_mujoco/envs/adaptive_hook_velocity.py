"""Velocity Aviary: track a desired velocity vector.

Task: follow velocity commands [vx, vy, vz, yaw_rate].
"""

import mujoco
import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class AdaptiveVelocityAviary(BaseAviary):
    """Single-drone velocity tracking task."""

    def __init__(
       self,
        drone_model: DroneModel = DroneModel.BB_HOOK,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.MIN_PAYLOAD_MASS = 0.05
        self.MAX_PAYLOAD_MASS = 0.3
        self.MIN_PAYLOAD_RADIUS = 0.02
        self.MAX_PAYLOAD_RADIUS = 0.04
        
        self.GOAL_RANDOM_AMPLITUDE = 1.0
        self.EPISODE_LEN_SEC = 10
        
        self.EPISODE_LEN_SEC = 10
        self.TARGET_VEL = np.array([0.0, 0.0, 0.0, 0.0])  # Will be randomized
        self.PAYLOAD_RADIUS = 0.05
        self.PAYLOAD_MASS = 0.2
        
        self.GRAB_FLAG = False
        self.GRAB_FLAG_ENABLE = False
        self.tendon_orientation=0
        self.GRAB_FLAG_ENABLE=False
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 10.8]])

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
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
        super().reset(seed=seed, options=options)
        # Randomize target velocity
        if self.np_random is not None:
            self.TARGET_VEL = self.np_random.uniform(-0.5, 0.5, size=4)
            self.TARGET_VEL[3] *= 0.5  # Reduce yaw rate
        
        if self.GRAB_FLAG_ENABLE:
            self.MASS=np.random.uniform(self.MIN_PAYLOAD_MASS,self.MAX_PAYLOAD_MASS)
            self.RADIUS=np.random.uniform(self.MIN_PAYLOAD_RADIUS,self.MAX_PAYLOAD_RADIUS)
            hook_pos = self.data.xpos[self.segment_2_id].copy()
            offset=np.random.choice([-0.001, 0.001])
            self.data.qpos[
                        self.target_qpos_adr:self.target_qpos_adr+3
            ] = [hook_pos[0]+offset,hook_pos[1],hook_pos[2]]
            
                    
        
            self.model.geom_size[self.target_geom_id] = [
                        self.RADIUS,
                        0.12,
                        0
                    ]

            random_z=np.random.uniform(0.45-0.2,0.8-0.2)
            self.model.body_pos[self.holder_body_id][2] = -random_z
            
            
                
                    # szürke tartó tömege
            self.model.body_mass[self.holder_body_id] = self.MASS
            red_bottom = -self.RADIUS
            
                    # holder felső széle
            holder_top = -(random_z) 
            self.model.site_pos[self.goal_id] = [100 ,100 ,100]
            
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

            self.tendon_orientation=np.sign(offset)

        else:
            self.model.site_pos[self.goal_id] = [100 ,100 ,100]
            self.data.qpos[
                                    self.target_qpos_adr:self.target_qpos_adr+3
                        ] = [0.5,100,100]
            self.model.body_pos[self.holder_body_id][2] = -0.4
        mujoco.mj_forward(self.model, self.data)
                
       
                
        return self._computeObs(), self._computeInfo()
        
    def step(self, action):
        action=action.copy()
       
        if self.GRAB_FLAG_ENABLE:
            if self.tendon_orientation==1:
                action[4] = 1
                action[5] = -1
            else:
                action[4] = -1
                action[5] = 1
                self.tendon_orientation=-1
        else:
            action[-2:] = 0
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
    def _actionSpace(self):
        return spaces.Box(low=-np.ones(6, dtype=np.float32), high=np.ones(6, dtype=np.float32))
    
    def _observationSpace(self):
        obs_lower_pos = np.full(13, -np.inf)
        obs_upper_pos = np.full(13 , np.inf)
        obs_lower_tendon_lengths = np.full(2, -1)
        obs_upper_tendon_lengths = np.full(2, 1)
        return spaces.Box(low=np.hstack([obs_lower_pos.astype(np.float32),obs_lower_tendon_lengths.astype(np.float32)]),
                               high=np.hstack([obs_upper_pos.astype(np.float32),obs_upper_tendon_lengths.astype(np.float32)]))


    def _preprocessAction(self, action):
        action = np.clip(np.array(action).flatten(), -1, 1)
        return self._normalizedActionToRPM(action).reshape(1, 4)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        obs = np.hstack([state[7:10], state[10:13], state[13:16], self.TARGET_VEL, state[-2:]])
        return obs.astype(np.float32)

    def _computeReward(self , action):
        vel_error = np.linalg.norm(self.vel[0, :3] - self.TARGET_VEL[:3])
        yaw_rate_error = abs(self.ang_v[0, 2] - self.TARGET_VEL[3])
        reward = -vel_error - 0.1 * yaw_rate_error
        # Penalize extreme attitudes
        reward -= 0.1 * (abs(self.rpy[0, 0]) + abs(self.rpy[0, 1]))
        # Bonus for tracking
        if vel_error < 0.05:
            reward += 0.5
            
        if self._computeTerminated():
            reward -= 100.0
            
        return float(reward)

    def _computeTerminated(self):
        if abs(self.rpy[0, 0]) > np.pi / 2 or abs(self.rpy[0, 1]) > np.pi / 2:
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "velocity_error": np.linalg.norm(self.vel[0, :3] - self.TARGET_VEL[:3]),
            "target_vel": self.TARGET_VEL.tolist(),
        }
