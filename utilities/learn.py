"""Example: Train a hover policy with Stable-Baselines3 PPO.

Usage:
    python -m multi_drone_mujoco.examples.learn
    python -m multi_drone_mujoco.examples.learn --multiagent true
"""

import argparse
import os
from pathlib import Path
from multi_drone_mujoco.envs.hover_aviary import HoverAviary
from multi_drone_mujoco.envs.fly_through_aviary import FlyThroughAviary
from multi_drone_mujoco.envs.adaptive_hook_hover import AdaptiveHookHover
from multi_drone_mujoco.envs.velocity_aviary import VelocityAviary
from multi_drone_mujoco.envs.adaptive_hook_fly_thorugh import AdaptiveFlyThroughAviary
from multi_drone_mujoco.envs.adaptive_hook_transport import AdaptiveTransportAviary
from multi_drone_mujoco.wrappers.curriculum import CurriculumWrapper,CurriculumCallback
from stable_baselines3.common.callbacks import CallbackList
import numpy as np
def adjust_difficulty(env, level,level_changed=True):
    if isinstance(env, AdaptiveHookHover):
        if level_changed:
            env.random_acion_amplitude = min(env.random_acion_amplitude + 0.05, 1)
    if isinstance(env,AdaptiveTransportAviary):
        if level_changed:
            if level > 0:
                print("Grab flag enabled")
                env.GRAB_FLAG_ENABLE=True
            elif level==0:
                env.GRAB_FLAG_ENABLE=False
def train_single(
    total_timesteps: int = 100_000,
    output_dir: str = "results/rl_hover",
    curriculum_flag: bool = False,
):
    """Train single-drone hover with PPO + optional curriculum learning."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("[ERROR] stable-baselines3 not installed.")
        return

    print("=" * 60)
    print(f"Training Single-Drone {args.env_type} (PPO)")
    print(f"  Timesteps: {total_timesteps}")
    print("=" * 60)
    if curriculum_flag:
        output_dir += "_curriculum"
    else:
        output_dir = output_dir
    # -----------------------------
    # Select environment class
    # -----------------------------
    if args.env_type == "adaptive_hook_hover":
        env_class = AdaptiveHookHover
        learning_rate = 3e-4
        
    elif args.env_type == "fly_through":
        env_class = FlyThroughAviary
        learning_rate = 3e-4
    elif args.env_type == "velocity_aviary":
        env_class = VelocityAviary
        learning_rate = 3e-4
    elif args.env_type == "adaptive_fly_through":
        env_class = AdaptiveFlyThroughAviary
        learning_rate = 3e-4
    elif args.env_type == "adaptive_transport":
        env_class = AdaptiveTransportAviary
        learning_rate = 3e-4
    else:
        env_class = HoverAviary
        learning_rate = 3e-4

    # -----------------------------
    # Evaluation environment
    # -----------------------------
    if curriculum_flag:
       eval_env = make_vec_env(
        lambda: CurriculumWrapper(
            env_class(ctrl_freq=48, sim_freq=240),
            difficulty_fn=adjust_difficulty
        ),
        n_envs=1,
    )
    else:
        eval_env = env_class(ctrl_freq=48, sim_freq=240)


    # -----------------------------
    # Curriculum or normal training
    # -----------------------------
    if curriculum_flag:
       
        # 2) Vectorized curriculum env for PPO
        env = make_vec_env(
            lambda: CurriculumWrapper(
                env_class(ctrl_freq=48, sim_freq=240),
                difficulty_fn=adjust_difficulty
            ),
            n_envs=8,
        )
       
    else:
        # Normal vectorized env
        env = make_vec_env(
            lambda: env_class(ctrl_freq=48, sim_freq=240),
            n_envs=8,
        )
    # comprahansion between cirruculum and naive learning
    # -----------------------------
    # Evaluation callback
    # -----------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
    )
    if curriculum_flag:
        curriculum_callback = CurriculumCallback(eval_env)

        callback = CallbackList([
            curriculum_callback,
            eval_callback
        ])
    else:
        callback = eval_callback
    # -----------------------------
    # PPO model
    # -----------------------------
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb"),
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu"
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(output_dir, "final_model"))

    print(f"\n[DONE] Model saved to {output_dir}/final_model.zip")



    env.close()
    eval_env.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--curriculum_flag",type=str, default="false")
    args = parser.parse_args()

   
    train_single(
        total_timesteps=args.timesteps,
        output_dir=f"results/rl_{args.env_type}",
        curriculum_flag=args.curriculum_flag.lower() == "true",
        )