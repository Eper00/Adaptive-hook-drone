"""Example: Play/visualize a trained policy.

Usage:
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip
"""

import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_position_and_velocity(
    positions,
    velocity_vectors,
    velocity_scale=0.05,
):
    """
    3D-ben plotolja a drón pozícióját és a kiadott velocity commandokat.

    Parameters
    ----------
    positions : list or np.ndarray
        A drón pozíciói, shape: (N, 3)

    velocity_vectors : list or np.ndarray
        A kiadott velocity commandok, shape: (N, 3)

    velocity_scale : float
        A velocity vektorok vizuális skálázása.
    """

    positions = np.asarray(positions)
    velocity_vectors = np.asarray(velocity_vectors)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ---------------------------------------------------------
    # 1. Trajektória
    # ---------------------------------------------------------
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        linewidth=2,
        label="Drone trajectory",
    )

    # ---------------------------------------------------------
    # 2. Velocity commandok
    # ---------------------------------------------------------
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        velocity_vectors[:, 0],
        velocity_vectors[:, 1],
        velocity_vectors[:, 2],
        length=velocity_scale,
        normalize=False,
        arrow_length_ratio=0.2,
        alpha=0.7,
        label="Velocity command",
    )

    # ---------------------------------------------------------
    # 3. Kezdő és végpont
    # ---------------------------------------------------------
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        s=80,
        label="Start",
    )

    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        s=80,
        label="End",
    )

    # ---------------------------------------------------------
    # 4. Tengelyek
    # ---------------------------------------------------------
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title("Drone Trajectory and Velocity Commands")

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


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
    failed_payload = 0
    velocity_vectors = []
    positions = []
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
                
               
                time.sleep(0.01)
            elif env_type == "velocity_aviary":

                env.TARGET_VELOCITY = np.array([0.0, 0.0, 0.0])
               
                time.sleep(0.005)
            elif env_type == "adaptive_velocity":
                
                
                time.sleep(0.001)
               
            elif env_type=="adaptive_director":
                #time.sleep(0.01)
                positions.append(env.pos[0].copy())
                velocity_vectors.append(action[0:3].copy())
                env.render_mode = None   
            env.render()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
        if env_type=="adaptive_director":
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
                    failed_payload += 1
                failed += 1
                
            if truncated:
                success += 1
            if env_type =="adaptive_director":
                print(f"Failed stab: {failed_stability}, failed incomplete {failed_incomplete}, failed payload {failed_payload}")
            print(f"Success: {success}, Failed: {failed}, Ratio: {success/(success+failed) if (success+failed)>0 else 0}")
            print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")
        
            #plot_position_and_velocity(positions, velocity_vectors)
            velocity_vectors = []
            positions = []
    env.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--curriculum_flag",type=str, default="false")
    args = parser.parse_args()
    play(args.model_path, args.env_type, args.episodes,curriculum_flag=args.curriculum_flag.lower() == "true")