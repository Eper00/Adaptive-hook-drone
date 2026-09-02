"""Example: Play/visualize a trained policy.

Usage:
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip
"""

import argparse
import numpy as np
import time

def play(model_path: str, env_type: str = "hover", episodes: int = 3, curriculum_flag: bool =False):
    """Load and visualize a trained policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable-baselines3 not installed.")
        return

    from multi_drone_mujoco.envs.hover_aviary import HoverAviary 
    from multi_drone_mujoco.envs.adaptive_hook_hover import AdaptiveHookHover
    from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary
    from multi_drone_mujoco.envs.fly_through_aviary import FlyThroughAviary
    from multi_drone_mujoco.envs.velocity_aviary import VelocityAviary
    from multi_drone_mujoco.envs.adaptive_hook_fly_thorugh import AdaptiveFlyThroughAviary
    from multi_drone_mujoco.envs.adaptive_hook_transport import AdaptiveTransportAviary
    from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary
    from multi_drone_mujoco.envs.adaptive_hook_director_velocity import AdaptiveTransportDirectorAviary
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    if env_type == "multi":
        env = MultiHoverAviary(num_drones=2, ctrl_freq=48, sim_freq=240, render_mode="rgb_array")
    elif env_type == "adaptive_hook_hover":
        env = AdaptiveHookHover(ctrl_freq=48, sim_freq=240, render_mode="human")
    elif env_type == "hover":
        env = HoverAviary(ctrl_freq=48, sim_freq=240, render_mode="human")
    elif env_type == "fly_through":
        env = FlyThroughAviary(ctrl_freq=48, sim_freq=240, render_mode="human")
    elif env_type == "velocity_aviary":
        env = VelocityAviary(ctrl_freq=48, sim_freq=240, render_mode="human")
    elif env_type == "adaptive_fly_through":
        env = AdaptiveFlyThroughAviary(ctrl_freq=48, sim_freq=240, render_mode="human")
    elif env_type == "adaptive_transport":
        env = AdaptiveTransportAviary(ctrl_freq=48, sim_freq=240, render_mode="human")
    elif env_type == "adaptive_velocity":
        env = AdaptiveVelocityAviary(ctrl_freq=48, sim_freq=240, render_mode="human")
    elif env_type == "adaptive_director":
        env= AdaptiveTransportDirectorAviary(ctrl_freq=48, sim_freq=240, render_mode="human")
    success=0
    failed=0
    failed_stability = 0
    failed_incomplete = 0
    failed_paylaod = 0
    for ep in range(episodes):
        
        total_reward = 0
        steps = 0
        if curriculum_flag == True and isinstance(env, AdaptiveHookHover):
            env.TARGET_HEIGHT=np.random.uniform(0.8,2)
            env.random_acion_amplitude=1
        if curriculum_flag == True and ((isinstance(env, AdaptiveTransportAviary) or isinstance(env, AdaptiveTransportDirectorAviary))):

            env.GRAB_FLAG_ENABLE=True
            env.MIN_PAYLOAD_MASS=0.01
            env.MAX_PAYLOAD_MASS=0.25
            env.MIN_PAYLOAD_RADIUS=0.02
            env.MAX_PAYLOAD_RADIUS=0.04
            env.GOAL_RANDOM_AMPLITUDE=1.5
            env.PAYLOAD_TERMINATION=True
        if curriculum_flag == True and isinstance(env, AdaptiveVelocityAviary):
            env.MIN_PAYLOAD_MASS=0.01
            env.MAX_PAYLOAD_MASS=0.25
            env.MIN_PAYLOAD_RADIUS=0.02
            env.MAX_PAYLOAD_RADIUS=0.04
            env.GRAB_FLAG_ENABLE=True
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            if env_type == "adaptive_hook_hover":
               print((env.TARGET_HEIGHT,env.pos[0][2]))
               time.sleep(0.01)
            elif env_type == "fly_through":
                if np.linalg.norm(env.TARGET_POSTION-env.pos[0])<0.1:
                    print("ok: TARGET")
                if np.linalg.norm(env.GOAL_POSTION-env.pos[0])<0.1:
                    print("ok: GOAL")
                time.sleep(0.01)
            elif env_type == "adaptive_fly_through":
                if np.linalg.norm(env.TARGET_POSTION-env.pos[0])<0.1:
                    print("ok: TARGET")
                if np.linalg.norm(env.GOAL_POSTION-env.pos[0])<0.1:
                    print("ok: GOAL")
                time.sleep(0.01)
            elif env_type == "adaptive_transport":
                
               
                env.render_mode = None
            elif env_type == "velocity_aviary":
                print(f"Target velocity: {env.TARGET_VEL}, Current velocity: {env.vel[0, :3]}, Current yaw rate: {env.ang_v[0, 2]}")
                vel_error = np.linalg.norm(env.vel[0, :3] - env.TARGET_VEL[:3])
                yaw_rate_error = abs(env.ang_v[0, 2] - env.TARGET_VEL[3])
                print(f"Velocity error: {vel_error}, Yaw rate error: {yaw_rate_error}")
                time.sleep(0.005)
            elif env_type == "adaptive_velocity":
                print(f"velocity errror:{np.linalg.norm(env.TARGET_VEL-env.vel)},the oreintation {env.rpy[0,2]} ")
                
               
            elif env_type=="adaptive_director":
                env.render_mode=None
                
            env.render()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            

        if env.current_waypoint_idx != len(env.WAYPOINTS)-1:
            terminated = True
            truncated = False
        if terminated:
            if env.current_waypoint_idx != len(env.WAYPOINTS)-1:
                failed_incomplete += 1
            if (
                abs(env.rpy[0, 0]) > np.pi / 2
                or abs(env.rpy[0, 1]) > np.pi / 2
            ):
                failed_stability += 1
            payload_pos = env.data.qpos[
                    env.target_qpos_adr:env.target_qpos_adr + 3
                ]

            hook_pos = env.data.xpos[env.segment_2_id].copy()

            payload_error = np.linalg.norm(
                    payload_pos - hook_pos
                )

            if (env.current_waypoint_idx[0]== len(env.WAYPOINTS) - 1 and payload_error > 0.2):
                failed_paylaod += 1
            failed += 1
            
        if truncated:
            success += 1
        if env_type =="adaptive_director":
            print(f"Failed stab: {failed_stability}, failed incomplete {failed_incomplete}, failed payload {failed_paylaod}")
        print(f"Success: {success}, Failed: {failed}, Ratio: {success/(success+failed) if (success+failed)>0 else 0}")
        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--curriculum_flag",type=str, default="false")
    args = parser.parse_args()
    play(args.model_path, args.env_type, args.episodes,curriculum_flag=args.curriculum_flag.lower() == "true")