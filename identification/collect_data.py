import numpy as np
import argparse
from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary


def collect_data(episodes: int):

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable-baselines3 not installed.")
        return

    model_path = (
        "/home/tomi/Adaptive-hook-drone/"
        "results/final/rl_adaptive_velocity_curriculum/final_model.zip"
    )

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    env = AdaptiveVelocityAviary(
        ctrl_freq=48,
        sim_freq=240,
        render_mode="human"
    )
    env.EPISODE_LEN_SEC=10

    # ---------------------------------------------------------
    # Complete dataset
    # ---------------------------------------------------------

    X = []
    Y = []

    for ep in range(episodes):

        total_reward = 0
        steps = 0

        # Payload randomization
        env.MIN_PAYLOAD_MASS = 0.01
        env.MAX_PAYLOAD_MASS = 0.25
        env.MIN_PAYLOAD_RADIUS = 0.02
        env.MAX_PAYLOAD_RADIUS = 0.04

        # Randomly enable/disable payload grabbing
        env.GRAB_FLAG_ENABLE = True

        obs, info = env.reset()

        # -----------------------------------------------------
        # Temporary buffers for THIS episode
        # -----------------------------------------------------

        episode_X = []
        episode_Y = []

        terminated = False
        truncated = False

        while not terminated and not truncated:
            time.sleep(0.001)
            # -------------------------------------------------
            # Change target velocity
            # -------------------------------------------------

            if steps % 50 == 0:
              
                p = np.random.uniform(0,1)
               
                if p < 0.4:
                               # Hover / kis sebesség
                    env.TARGET_VEL = np.random.uniform(-0.1, 0.1, size=3)
               
                elif p < 0.7:
                               # Közepes sebesség
                    env.TARGET_VEL = np.random.uniform(-0.5, 0.5, size=3)
               
                else:
                               # Teljes tartomány
                    env.TARGET_VEL = np.random.uniform(-1.0, 1.0, size=3)

            # -------------------------------------------------
            # Current state
            # -------------------------------------------------

            velocity = env.vel[0, :3].copy()

            tendon_lengths = (
                env.tendon_lengths[0].flatten().copy()
            )

            target_velocity = env.TARGET_VEL.copy()

           

           

            

            # -------------------------------------------------
            # Build input vector
            # -------------------------------------------------

            input_data = np.concatenate([
                velocity,
                tendon_lengths,
                target_velocity,
                
            ])

            # -------------------------------------------------
            # Low-level PPO
            # -------------------------------------------------

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            env.render()

            # -------------------------------------------------
            # Perform one control step
            # -------------------------------------------------

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            # -------------------------------------------------
            # Next velocity
            # -------------------------------------------------

            next_velocity = env.vel[0, :3].copy()

            # -------------------------------------------------
            # Store ONLY temporarily
            #
            # We don't know yet whether the episode will
            # terminate because of instability.
            # -------------------------------------------------

            episode_X.append(input_data)
            episode_Y.append(next_velocity)

            total_reward += reward
            steps += 1

        # =====================================================
        # EPISODE FINISHED
        # =====================================================

        if terminated:
            # Agent crashed / failed.
            # Throw away the COMPLETE episode.
            print(
                f"Episode {ep + 1}/{episodes}: "
                f"TERMINATED -> discarded "
                f"({steps} samples)"
            )

        else:
            # Normal timeout / successful completion.
            # Keep the complete episode.
            X.extend(episode_X)
            Y.extend(episode_Y)

            print(
                f"Episode {ep + 1}/{episodes}: "
                f"accepted ({steps} samples)"
            )

    env.close()

    # ---------------------------------------------------------
    # Convert to numpy arrays
    # ---------------------------------------------------------

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    print("\nDataset collection finished.")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    return X, Y



def save_data(episodes:int):
    X, Y = collect_data(episodes)
    np.savez(
            "results/identification/adaptive_velocity_identification.npz",
            X=X,
            Y=Y
        )
    
    print(
            "Dataset saved to "
            "'adaptive_velocity_identification.npz'"
        )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    save_data(args.episodes)
