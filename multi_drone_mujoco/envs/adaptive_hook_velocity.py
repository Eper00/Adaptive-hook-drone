"""Adaptive velocity aviary: low-level velocity tracking with the hook drone.

Task: track a constant target velocity [vx, vy, vz] (drawn at every reset)
at yaw 0, with or without a payload hanging in the hook. The trained policy is
the low-level controller used by ``AdaptiveTransportDirectorAviary``.

Action (6):      4 normalized motor commands + 2 tendon commands, all in
                 [-1, 1]; the tendon commands are overwritten by the env
                 (hook closed when a payload is carried, relaxed otherwise).
Observation (15): [rpy(3), vel(3), ang_vel(3), target_vel(3), target_yaw(1),
                  tendon_lengths(2)].
"""

import mujoco
import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class AdaptiveVelocityAviary(BaseAviary):
    """Single-drone velocity tracking task (optionally with a payload).

    Flags set by the curriculum:
      GRAB_FLAG_ENABLE    payload in ~50 % of the episodes (PAYLOAD_INDICATOR)
      RANDOM_OREINTATION  random initial yaw in [-pi, pi) (the target yaw is 0)
    """

    def __init__(
       self,
        drone_model: DroneModel = DroneModel.BB_HOOK,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        initial_xyzs=None,
        initial_rpys=None,
        render_mode=None,
    ):
        # Payload randomization ranges [kg] / [m]
        self.MIN_PAYLOAD_MASS = 0.05
        self.MAX_PAYLOAD_MASS = 0.3
        self.MIN_PAYLOAD_RADIUS = 0.02
        self.MAX_PAYLOAD_RADIUS = 0.04
        
        
        self.EPISODE_LEN_SEC = 10
        self.TARGET_VEL = np.array([0.0, 0.0, 0.0])  
        self.TARGET_ORIENTATION = 0
        self.PAYLOAD_RADIUS = 0.05
        self.PAYLOAD_MASS = 0.2
        
        self.GRAB_FLAG = False
        self.GRAB_FLAG_ENABLE = False     # allow episodes with a payload
        self.tendon_orientation=0         # curl side of the hook (+1 / -1)
        self.PAYLOAD_INDICATOR=None       # < 0.5: this episode carries a payload
        self.GRAB_FLAG_ENABLE=False
        self.RANDOM_OREINTATION = False   # random initial yaw
        if initial_xyzs is None:
            # high up: no ground in the way while tracking velocities
            initial_xyzs = np.array([[0.0, 0.0, 12.8]])
        if initial_rpys is None:
                    initial_rpys = np.array([[0.0, 0.0, 0]])

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
            initial_rpys=initial_rpys,
            render_mode=render_mode,
            transport_target=True
        )
    def reset(self, seed=None, options=None):
        """Draw the target velocity, the initial yaw and (optionally) a
        payload that already hangs in the closed hook."""

        super().reset(seed=seed, options=options)
        self.steps=0
        # Randomize the target velocity: 40 % near hover, 30 % medium, 30 % full range
        if self.np_random is not None:
            p = self.np_random.random()

            if p < 0.4:
                # Hover / low speed
                self.TARGET_VEL = self.np_random.uniform(-0.1, 0.1, size=3)

            elif p < 0.7:
                # Medium speed
                self.TARGET_VEL = self.np_random.uniform(-0.5, 0.5, size=3)

            else:
                # Full range
                self.TARGET_VEL = self.np_random.uniform(-1.0, 1.0, size=3)
            self.TARGET_ORIENTATION = 0
        # The drone starts level at the default position; with
        # RANDOM_OREINTATION its heading (yaw) is random (the target yaw
        # stays 0, so the policy has to turn). Drawn after the seeded
        # super().reset() and applied right away.
        yaw = self.np_random.uniform(-np.pi, np.pi) if self.RANDOM_OREINTATION else 0.0
        self.INIT_RPYS[0][:] = [0.0, 0.0, yaw]
        joint = self.model.joint("drone0_joint")
        adr = self.model.jnt_qposadr[joint.id]
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self._updateAndStoreKinematicInformation()
        

        # Payload in about half of the episodes when enabled
        if self.GRAB_FLAG_ENABLE:
            self.PAYLOAD_INDICATOR=self.np_random.uniform(0,1)
        else:
            self.PAYLOAD_INDICATOR=1

        if self.PAYLOAD_INDICATOR<0.5:
            self.MASS=self.np_random.uniform(self.MIN_PAYLOAD_MASS,self.MAX_PAYLOAD_MASS)
            self.RADIUS=self.np_random.uniform(self.MIN_PAYLOAD_RADIUS,self.MAX_PAYLOAD_RADIUS)
            # side of the curl: +1 -> tendon (1, -1), hook curls to the body -y side
            side = int(self.np_random.choice([-1, 1]))

            self.model.geom_size[self.target_geom_id] = [
                        self.RADIUS,
                        0.12,
                        0
                    ]

            # The holder plate hangs random_z below the red cylinder
            random_z=self.np_random.uniform(0.45-0.2,0.8-0.2)
            self.model.body_pos[self.holder_body_id][2] = -random_z
            
            
                
                    # mass of the grey holder (the payload mass)
            self.model.body_mass[self.holder_body_id] = self.MASS
            red_bottom = -self.RADIUS
            
                    # top of the holder
            holder_top = -(random_z) 
            self.model.site_pos[self.goal_id] = [100 ,100 ,100]
            
                    # distance between the two cylinders
            connector_length = abs(red_bottom - holder_top)
            
            
                    # the third MuJoCo box size is the half height
            connector_half_height = connector_length / 2+0.005
            
            
                    # centre of the connectors
            connector_z = (red_bottom + holder_top) / 2+0.005
            
            
                    # left and right connector positions
            self.model.body_pos[self.left_connector_id][2] = connector_z
            self.model.body_pos[self.right_connector_id][2] = connector_z
            self.model.body_pos[self.left_connector_id][0] = 0.1
            self.model.body_pos[self.right_connector_id][0] = -0.1
            
                    # left and right connector sizes
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

            self._hang_payload_in_hook(side)
            self.tendon_orientation=side

        else:
            # No payload: park the payload and the goal marker far away
            self.model.site_pos[self.goal_id] = [100 ,100 ,100]
            self.data.qpos[
                                    self.target_qpos_adr:self.target_qpos_adr+3
                        ] = [0.5,100,100]
            self.model.body_pos[self.holder_body_id][2] = -0.4
        mujoco.mj_forward(self.model, self.data)
                
       
                
        return self._computeObs(), self._computeInfo()
        
    # Hanging state of a grasped payload, measured from simulated grasps with
    # the final hook (full curl, side +1) for payload radii 0.02 / 0.03 / 0.04:
    # payload centre in the drone frame and hook joint angles link_1..link_7.
    _HANG_RADII = np.array([0.02, 0.03, 0.04])
    _HANG_PAYLOAD = np.array([[0.0, -0.036, -0.2415],
                              [0.0, -0.0375, -0.2265],
                              [0.0, -0.042, -0.220]])
    _HANG_LINKS_DEG = np.array([[5.1, -67.7, -74.6, -53.2, -68.4, -80.0, -80.0],
                                [5.1, -69.8, -69.6, -45.3, -38.8, -67.6, -80.0],
                                [5.2, -66.0, -62.4, -42.8, -27.8, -46.7, -80.0]])

    def _hang_payload_in_hook(self, side, settle_time=0.25):
        """Spawn the payload as if it had been grasped: hook curled to the
        ``side`` (+1: tendon (1, -1)) with the cylinder hanging in its pocket.

        The grasp state is interpolated over the payload radius from real
        simulated grasps, then settled for ``settle_time`` with the drone held
        in place, so the episode starts from a physically consistent hold.
        """
        m, d = self.model, self.data
        interp = lambda table: np.array([np.interp(self.RADIUS, self._HANG_RADII, col)
                                         for col in table.T])
        rel = interp(self._HANG_PAYLOAD) * np.array([1.0, side, 1.0])
        links = np.radians(interp(self._HANG_LINKS_DEG)) * side

        joint = m.joint("drone0_joint")
        adr = m.jnt_qposadr[joint.id]
        drone_q = d.qpos[adr:adr + 7].copy()
        link_adr = m.jnt_qposadr[m.joint("link_1").id]
        d.qpos[link_adr:link_adr + 7] = links
        rot = d.xmat[m.body("drone0").id].reshape(3, 3)
        a = self.target_qpos_adr
        d.qpos[a:a + 3] = drone_q[0:3] + rot @ rel
        d.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[:] = 0.0
        d.ctrl[:] = [side, -side]
        mujoco.mj_forward(m, d)

        for _ in range(int(settle_time / m.opt.timestep)):
            d.qpos[adr:adr + 7] = drone_q
            d.qvel[0:6] = 0.0
            mujoco.mj_step(m, d)
        d.qpos[adr:adr + 7] = drone_q
        d.qvel[0:6] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        self._updateAndStoreKinematicInformation()

    def step(self, action):
        """Apply the motor commands; the tendons are set by the env: fully
        closed on the payload's side when carrying one, relaxed otherwise."""
        self.steps+=1
        action=action.copy()

        if self.PAYLOAD_INDICATOR<0.5:
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
        """4 normalized motor commands + 2 tendon commands, all in [-1, 1]."""
        return spaces.Box(low=-np.ones(6, dtype=np.float32), high=np.ones(6, dtype=np.float32))

    def _observationSpace(self):
        """13 unbounded kinematic/target entries + 2 tendon lengths."""
        obs_lower_pos = np.full(13, -np.inf)
        obs_upper_pos = np.full(13 , np.inf)
        obs_lower_tendon_lengths = np.full(2, -1)
        obs_upper_tendon_lengths = np.full(2, 1)
        return spaces.Box(low=np.hstack([obs_lower_pos.astype(np.float32),obs_lower_tendon_lengths.astype(np.float32)]),
                               high=np.hstack([obs_upper_pos.astype(np.float32),obs_upper_tendon_lengths.astype(np.float32)]))


    def _preprocessAction(self, action):
        """Normalized motor commands in [-1, 1] -> RPMs (-1: 0, 0: hover, 1: max)."""
        action = np.clip(np.array(action).flatten(), -1, 1)
        return self._normalizedActionToRPM(action).reshape(1, 4)

    def _computeObs(self):
        """[rpy, vel, ang_vel, target_vel, target_yaw, tendon_lengths] (15)."""
        state = self._getDroneStateVector(0)
        obs = np.hstack([state[7:10], state[10:13], state[13:16], self.TARGET_VEL, self.TARGET_ORIENTATION, state[-2:]])
        return obs.astype(np.float32)

    def _computeReward(self , action):
        """-|velocity error| - |yaw error| - 0.1 * tilt, +0.5 when tracking
        within 2.5 cm/s, -100 on a flip."""
        vel_error = np.linalg.norm(self.vel[0, :3] - self.TARGET_VEL)
        orientation_error = abs(self.rpy[0,2] - self.TARGET_ORIENTATION)
       
        reward = -vel_error - orientation_error
        # Penalize extreme attitudes
        reward -= 0.1 * (abs(self.rpy[0, 0]) + abs(self.rpy[0, 1]))
        # Bonus for tracking
        if vel_error < 0.025:
            reward += 0.5
  
            
        if self._computeTerminated():
            reward -= 100.0
            
        return float(reward)

    def _computeTerminated(self):
        """The drone flipped: |roll| or |pitch| above 90 deg."""
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
