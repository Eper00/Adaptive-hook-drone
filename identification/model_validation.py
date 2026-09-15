import numpy as np
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary
from identification.fit_data import MODEL_PATH,VelocityIncrementMLP,DT

PPO_MODEL_PATH = "/home/tomi/Adaptive-hook-drone/results/final/rl_adaptive_velocity_curriculum/final_model.zip"
CTRL_FREQ=48


  

def build_mlp_input(velocity, tendon_lengths, target_velocity, grab_flag):
    return np.concatenate([velocity, tendon_lengths, target_velocity, [grab_flag]]).astype(np.float32)


def closed_loop_validation(episodes: int, render: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading PPO model:\n{PPO_MODEL_PATH}")
    ppo_model = PPO.load(PPO_MODEL_PATH)

    print(f"\nLoading velocity increment MLP:\n{MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    input_dim = checkpoint["input_dim"]
    print(input_dim)

    x_mean = torch.tensor(checkpoint["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(checkpoint["x_std"], dtype=torch.float32, device=device)

    mlp_model = VelocityIncrementMLP(input_dim).to(device)
    mlp_model.load_state_dict(checkpoint["model_state_dict"])
    mlp_model.eval()

    env = AdaptiveVelocityAviary(
        ctrl_freq=CTRL_FREQ,
        sim_freq=240,
        render_mode="human" if render else None
    )

    all_results = []
    env.MIN_PAYLOAD_MASS = 0.01
    env.MAX_PAYLOAD_MASS = 0.25
    env.MIN_PAYLOAD_RADIUS = 0.02
    env.MAX_PAYLOAD_RADIUS = 0.04
    for ep in range(episodes):
        print(f"\n========================================")
        print(f"Episode {ep + 1}/{episodes}")
        print(f"========================================")

      
        env.GRAB_FLAG_ENABLE = np.random.uniform(0, 1) < 0.5
        

        obs, info = env.reset()

        initial_velocity = env.vel[0, :3].copy()
        mlp_velocity = initial_velocity.copy()

        real_velocities = [initial_velocity.copy()]
        mlp_velocities = [mlp_velocity.copy()]
        target_velocities = [env.TARGET_VEL.copy()]
        times = [0.0]

        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            if steps % 50 == 0:
                target_velocity = np.random.uniform(-0.5, 0.5, 3).astype(np.float32)
                env.TARGET_VEL = target_velocity.copy()
            else:
                target_velocity = env.TARGET_VEL.copy()

            tendon_lengths = env.tendon_lengths[0].flatten().copy()
            grab_flag = float(env.GRAB_FLAG_ENABLE)

            # Actual MuJoCo + PPO
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            actual_velocity = env.vel[0, :3].copy()

            # Closed-loop MLP prediction
            mlp_input = build_mlp_input(
                mlp_velocity, tendon_lengths, target_velocity, grab_flag
            )
            mlp_input_tensor = torch.tensor(
                mlp_input, dtype=torch.float32, device=device
            ).unsqueeze(0)

            mlp_input_tensor = (mlp_input_tensor - x_mean) / x_std

            with torch.no_grad():
                delta_velocity = mlp_model(mlp_input_tensor).cpu().numpy().flatten()

            mlp_velocity = mlp_velocity + delta_velocity

            real_velocities.append(actual_velocity.copy())
            mlp_velocities.append(mlp_velocity.copy())
            target_velocities.append(target_velocity.copy())

            steps += 1
            times.append(steps * DT)

        if terminated:
            print(f"Episode terminated after {steps} steps -> discarded")
            continue

        print(f"Episode accepted: {steps} steps")
        all_results.append({
            "time": np.asarray(times),
            "real": np.asarray(real_velocities),
            "mlp": np.asarray(mlp_velocities),
            "target": np.asarray(target_velocities)
        })

    env.close()

    if len(all_results) == 0:
        print("\nNo valid episodes!")
        return

    print("\n========================================")
    print("CLOSED-LOOP MLP VALIDATION")
    print("========================================")

    all_errors = []

    for result in all_results:
        real = result["real"]
        mlp = result["mlp"]
        n = min(len(real), len(mlp))
        all_errors.append(real[:n] - mlp[:n])

    all_errors = np.concatenate(all_errors, axis=0)

    rmse = np.sqrt(np.mean(all_errors ** 2, axis=0))
    total_rmse = np.sqrt(np.mean(all_errors ** 2))

    print("\nMLP model:")
    print(f"  vx RMSE = {rmse[0]:.6f} m/s")
    print(f"  vy RMSE = {rmse[1]:.6f} m/s")
    print(f"  vz RMSE = {rmse[2]:.6f} m/s")
    print(f"  Total RMSE = {total_rmse:.6f} m/s")

    result = all_results[0]
    time = result["time"]
    real = result["real"]
    mlp = result["mlp"]
    target = result["target"]

    error = real - mlp
    abs_error = np.abs(error)
    total_error = np.linalg.norm(error, axis=1)

    cumulative_rmse = np.sqrt(
        np.cumsum(error ** 2, axis=0) /
        np.arange(1, len(error) + 1)[:, None]
    )

    cumulative_total_rmse = np.sqrt(
        np.cumsum(np.mean(error ** 2, axis=1)) /
        np.arange(1, len(error) + 1)
    )

    # Plot 1: Real vs MLP
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, real[:, 0], label="Real vx")
    axes[0].plot(time, mlp[:, 0], "--", label="MLP vx")
    axes[0].set_ylabel("vx [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, real[:, 1], label="Real vy")
    axes[1].plot(time, mlp[:, 1], "--", label="MLP vy")
    axes[1].set_ylabel("vy [m/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time, real[:, 2], label="Real vz")
    axes[2].plot(time, mlp[:, 2], "--", label="MLP vz")
    axes[2].set_ylabel("vz [m/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Real vs Closed-loop MLP prediction")
    plt.tight_layout()
    plt.show()

    # Plot 2: Instantaneous prediction error
    plt.figure(figsize=(12, 6))
    plt.plot(time, abs_error[:, 0], label="|vx error|")
    plt.plot(time, abs_error[:, 1], label="|vy error|")
    plt.plot(time, abs_error[:, 2], label="|vz error|")
    plt.plot(time, total_error, "--", linewidth=2, label="||v error||₂")
    plt.xlabel("Time [s]")
    plt.ylabel("Prediction error [m/s]")
    plt.title("Closed-loop MLP prediction error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: Cumulative RMSE
    plt.figure(figsize=(12, 6))
    plt.plot(time, cumulative_rmse[:, 0], label="vx RMSE")
    plt.plot(time, cumulative_rmse[:, 1], label="vy RMSE")
    plt.plot(time, cumulative_rmse[:, 2], label="vz RMSE")
    plt.plot(time, cumulative_total_rmse, "--", linewidth=2, label="Total RMSE")
    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative RMSE [m/s]")
    plt.title("Closed-loop prediction RMSE over episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    closed_loop_validation(
        episodes=args.episodes,
        render=args.render
    )

