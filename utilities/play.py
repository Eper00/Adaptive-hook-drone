"""Play / visualize a trained policy or an MPC controller in the MuJoCo viewer.

env_type:
    adaptive_velocity          RL velocity controller          (--model_path)
    adaptive_transport         RL rotor-level transport policy (--model_path)
    adaptive_director          RL director policy              (--model_path)
    adaptive_director_safety   RL director + MPC safety filter (--model_path)
    adaptive_transport_MPC     rotor-level hybrid MPC (no model needed)
    adaptive_director_MPC      director MPC on the RL velocity controller
    adaptive_director_resnet_MPC  director MPC with the ResNet-identified model (resnet_mpc.py)
With --curriculum_flag true the envs are set to their final curriculum level.
Success / failure statistics are printed for the director and MPC types
(utilities/analyse.py has the detailed Monte Carlo analysis).

Usage:
    python -m utilities.play --model_path results/rl_adaptive_transport/best_model.zip --env_type adaptive_transport
    python -m utilities.play --env_type adaptive_transport_MPC     # rotor-level hybrid MPC
    python -m utilities.play --env_type adaptive_director_MPC      # MPC on the RL velocity controller
    python -m utilities.play --model_path results/final/rl_adaptive_director_curriculum/best_model.zip \
        --env_type adaptive_director_safety                        # RL director + MPC safety filter
        (add --filter_model arx to use the ARX model in the filter instead of the ResNet)
"""

import argparse
import numpy as np
import time




def play(model_path: str, env_type: str = "hover", episodes: int = 3, curriculum_flag: bool =True,
         filter_model: str = "resnet"):
    """Run ``episodes`` episodes of a policy (or MPC agent) with the viewer
    and print the success / failure statistics."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable-baselines3 not installed.")
        return

    from multi_drone_mujoco.envs.adaptive_hook_transport import AdaptiveTransportAviary
    from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary
    from multi_drone_mujoco.envs.adaptive_hook_director_velocity import AdaptiveTransportDirectorAviary
    ctrl_freq=48
    # MPC agents replace the policy: they step the env themselves (agent.step())
    mpc_flag = env_type.endswith("_MPC")
    if not mpc_flag:
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path)
    
    if env_type == "adaptive_transport":
        env = AdaptiveTransportAviary(ctrl_freq=ctrl_freq, sim_freq=240, render_mode="human")
    elif env_type == "adaptive_velocity":
        env = AdaptiveVelocityAviary(ctrl_freq=ctrl_freq, sim_freq=240, render_mode="human")
    elif env_type == "adaptive_director":
        env= AdaptiveTransportDirectorAviary(ctrl_freq=ctrl_freq, sim_freq=240, render_mode="human")
    elif env_type == "adaptive_director_safety":
        from multi_drone_mujoco.envs.adaptive_hook_director_safety_filter import (
            AdaptiveTransportDirectorAviarySafetyFilter,
        )
        # filter_model: "resnet" (default, the more accurate model) or "arx"
        env = AdaptiveTransportDirectorAviarySafetyFilter(ctrl_freq=ctrl_freq, sim_freq=240,
                                                          render_mode="human", model=filter_model)
    elif env_type == "adaptive_transport_MPC":
        from multi_drone_mujoco.envs.hybrid_mpc import HybridMPCAgent
        env = AdaptiveTransportAviary(ctrl_freq=ctrl_freq, sim_freq=240, render_mode="human")
        env.PAYLOAD_TERMINATION = True
        agent = HybridMPCAgent(env, verbose=True)
    elif env_type == "adaptive_director_MPC":
        from multi_drone_mujoco.envs.director_mpc import DirectorMPCAgent
        env = AdaptiveTransportDirectorAviary(ctrl_freq=ctrl_freq, sim_freq=240, render_mode="human")
        env.PAYLOAD_TERMINATION = True
        agent = DirectorMPCAgent(env, verbose=True)
    elif env_type == "adaptive_director_resnet_MPC":
        from multi_drone_mujoco.envs.resnet_mpc import ResNetMPCAgent
        env = AdaptiveTransportDirectorAviary(ctrl_freq=ctrl_freq, sim_freq=240, render_mode="human")
        env.PAYLOAD_TERMINATION = True
        agent = ResNetMPCAgent(env, verbose=True)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    # Episode statistics (director and MPC types)
    success=0
    failed=0
    failed_stability = 0
    failed_incomplete = 0
    failed_payload = 0

    for ep in range(episodes):
        
        total_reward = 0
        steps = 0
        # Final curriculum level (as in utilities/learn.py)
        if curriculum_flag == True and (isinstance(env, AdaptiveTransportAviary) or isinstance(env, AdaptiveTransportDirectorAviary)):
            env.RANDOM_ORIENTATION = True   # random initial yaw
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
        if mpc_flag:
            agent.reset()
       
        
        terminated = False
        truncated = False
        while not terminated and not truncated:
            
           

            # slow down to roughly real time for watching
            if env_type == "adaptive_transport":
                
               
                time.sleep(0.005)
            elif env_type == "adaptive_velocity":
                
                
                time.sleep(0.01)
               
            elif env_type in ("adaptive_director", "adaptive_director_MPC",
                              "adaptive_director_resnet_MPC", "adaptive_director_safety"):
                time.sleep(0.01)
            env.render()
            
            if mpc_flag:
                # terminated = crash, truncated = mission finished / time out
                obs, reward, terminated, truncated, info = agent.step()
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
        # --- episode evaluation ---
        # MPC: success = take-off, pick-up and goal visited, payload still carried
        if mpc_flag:
            if agent.success:
                success += 1
            else:
                failed += 1
                if agent.crashed:
                    failed_stability += 1
                elif [v[0] for v in agent.mission.visited] != ["takeoff", "pickup", "goal"]:
                    failed_incomplete += 1
                else:
                    failed_payload += 1
            print(agent.summary())
            print(f"Failed stab: {failed_stability}, failed incomplete {failed_incomplete}, failed payload {failed_payload}")
            print(f"Success: {success}, Failed: {failed}, Ratio: {success/(success+failed) if (success+failed)>0 else 0}")
            print(f"  Episode {ep + 1}: steps={steps}")
        # RL director: success = episode ran to the time limit with the goal
        # waypoint active; failures are counted by cause (the causes can overlap)
        elif env_type in ("adaptive_director", "adaptive_director_safety"):
            if env_type == "adaptive_director_safety":
                print(f"  safety filter interventions: {env.intervention_count}/{env.filter_steps} steps"
                      f" ({100 * env.intervention_count / max(env.filter_steps, 1):.1f}%):"
                      f" constraint {env.reason_count['constraint']}, rate {env.reason_count['rate']},"
                      f" stall takeover {env.reason_count['stall']},"
                      f" hook-contact escape {env.reason_count['contact']}")
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
            if env_type in ("adaptive_director", "adaptive_director_MPC", "adaptive_director_safety"):
                print(f"Failed stab: {failed_stability}, failed incomplete {failed_incomplete}, failed payload {failed_payload}")
            print(f"Success: {success}, Failed: {failed}, Ratio: {success/(success+failed) if (success+failed)>0 else 0}")
            print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")
        

    env.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False)
    # see the module docstring for the available types
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--curriculum_flag",type=str, default="true")
    # adaptive_director_safety: prediction model of the safety filter
    parser.add_argument("--filter_model", choices=["resnet", "arx"], default="resnet")
    args = parser.parse_args()
    play(args.model_path, args.env_type, args.episodes,curriculum_flag=args.curriculum_flag.lower() == "true",
         filter_model=args.filter_model)