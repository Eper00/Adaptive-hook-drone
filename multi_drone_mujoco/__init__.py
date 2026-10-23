"""gym-mujoco-drones: Multi-drone environments for MuJoCo.

A gymnasium-compatible multi-drone simulation environment using MuJoCo physics,
inspired by gym-pybullet-drones but with superior performance, accuracy, and features.

Drone models: Bitcraze Crazyflie 2.x (mujoco_menagerie meshes) and the
AIMotionLab Bumblebee with the adaptive hook (AIMotionLab-Virtual meshes).
"""

__version__ = "1.0.0"

from multi_drone_mujoco.envs.base_aviary import BaseAviary

from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
