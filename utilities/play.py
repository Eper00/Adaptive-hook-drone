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


    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        if curriculum_flag == True and isinstance(env, AdaptiveHookHover):
            env.TARGET_HEIGHT=np.random.uniform(0.8,2)
            env.random_acion_amplitude=1
        if curriculum_flag == True and isinstance(env, AdaptiveTransportAviary):
            env.GRAB_FLAG_ENABLE=True
        
        while True:
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
               
               
                time.sleep(0.02)
            
            env.render()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

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