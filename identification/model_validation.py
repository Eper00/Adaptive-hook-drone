import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary
from identification.fit_data import VelocityIncrementMLP


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "results/identification/velocity_increment_model.pth"
PPO_MODEL_PATH = (
    "/home/tomi/Adaptive-hook-drone/"
    "results/final/rl_adaptive_velocity_curriculum/final_model.zip"
)

DT = 1.0 / 48.0
CTRL_FREQ = 48


# ============================================================
# Helper
# ============================================================

def build_mlp_input(
    velocity,
    tendon_lengths,
    target_velocity
):
    return np.concatenate(
        [
            velocity,
            tendon_lengths,
            target_velocity
        ]
    ).astype(np.float32)


# ============================================================
# MLP prediction
# ============================================================

def predict_delta_velocity(
    mlp_model,
    velocity,
    tendon_lengths,
    target_velocity,
    x_mean,
    x_std,
    device
):

    mlp_input = build_mlp_input(
        velocity,
        tendon_lengths,
        target_velocity
    )

    mlp_input_tensor = torch.tensor(
        mlp_input,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    mlp_input_tensor = (
        mlp_input_tensor - x_mean
    ) / x_std

    with torch.no_grad():

        delta_velocity = (
            mlp_model(mlp_input_tensor)
            .cpu()
            .numpy()
            .flatten()
        )

    return delta_velocity


# ============================================================
# Closed-loop validation
# ============================================================

def closed_loop_validation(
    episodes: int,
    render: bool = False
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Load PPO
    # --------------------------------------------------------

    print(
        f"\nLoading PPO model:\n"
        f"{PPO_MODEL_PATH}"
    )

    ppo_model = PPO.load(
        PPO_MODEL_PATH
    )

    # --------------------------------------------------------
    # Load MLP
    # --------------------------------------------------------

    print(
        f"\nLoading velocity increment MLP:\n"
        f"{MODEL_PATH}"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    input_dim = checkpoint["input_dim"]

    x_mean = torch.tensor(
        checkpoint["x_mean"],
        dtype=torch.float32,
        device=device
    )

    x_std = torch.tensor(
        checkpoint["x_std"],
        dtype=torch.float32,
        device=device
    )

    mlp_model = VelocityIncrementMLP(
        input_dim
    ).to(device)

    mlp_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    mlp_model.eval()

    print(f"MLP input dimension: {input_dim}")

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = AdaptiveVelocityAviary(
        ctrl_freq=CTRL_FREQ,
        sim_freq=240,
        render_mode="human" if render else None
    )

    env.EPISODE_LEN_SEC = 10

    env.MIN_PAYLOAD_MASS = 0.01
    env.MAX_PAYLOAD_MASS = 0.25

    env.MIN_PAYLOAD_RADIUS = 0.02
    env.MAX_PAYLOAD_RADIUS = 0.04

    # --------------------------------------------------------
    # Separate results
    # --------------------------------------------------------

    results = {
        False: [],
        True: []
    }

    # ========================================================
    # Episodes
    # ========================================================

    for ep in range(episodes):

        print(
            "\n========================================"
        )

        print(
            f"Episode {ep + 1}/{episodes}"
        )

        print(
            "========================================"
        )

        # ----------------------------------------------------
        # Randomly select payload condition
        # ----------------------------------------------------

        grab_flag = (
            np.random.uniform(0, 1) < 0.5
        )

        env.GRAB_FLAG_ENABLE = grab_flag

        print(
            f"Payload attached: {grab_flag}"
        )

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        obs, info = env.reset()

        initial_velocity = (
            env.vel[0, :3].copy()
        )

        mlp_velocity = (
            initial_velocity.copy()
        )

        real_velocities = [
            initial_velocity.copy()
        ]

        mlp_velocities = [
            mlp_velocity.copy()
        ]

        target_velocities = [
            env.TARGET_VEL.copy()
        ]

        times = [0.0]

        terminated = False
        truncated = False

        steps = 0

        # ====================================================
        # Episode
        # ====================================================

        while (
            not terminated
            and not truncated
        ):

            # ------------------------------------------------
            # Change target velocity
            # ------------------------------------------------

            if steps % 50 == 0:

                p = np.random.uniform(
                    0,
                    1
                )

                if p < 0.4:

                    env.TARGET_VEL = (
                        np.random.uniform(
                            -0.1,
                            0.1,
                            size=3
                        )
                    )

                elif p < 0.7:

                    env.TARGET_VEL = (
                        np.random.uniform(
                            -0.5,
                            0.5,
                            size=3
                        )
                    )

                else:

                    env.TARGET_VEL = (
                        np.random.uniform(
                            -1.0,
                            1.0,
                            size=3
                        )
                    )

            target_velocity = (
                env.TARGET_VEL.copy()
            )

            # ------------------------------------------------
            # Current tendon state
            # ------------------------------------------------

            tendon_lengths = (
                env.tendon_lengths[0]
                .flatten()
                .copy()
            )

            # ------------------------------------------------
            # Actual system
            # ------------------------------------------------

            action, _ = ppo_model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            actual_velocity = (
                env.vel[0, :3].copy()
            )

            # ------------------------------------------------
            # MLP prediction
            # ------------------------------------------------

            delta_velocity = (
                predict_delta_velocity(
                    mlp_model,
                    mlp_velocity,
                    tendon_lengths,
                    target_velocity,
                    x_mean,
                    x_std,
                    device
                )
            )

            mlp_velocity = (
                mlp_velocity
                + delta_velocity
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            real_velocities.append(
                actual_velocity.copy()
            )

            mlp_velocities.append(
                mlp_velocity.copy()
            )

            target_velocities.append(
                target_velocity.copy()
            )

            steps += 1

            times.append(
                steps * DT
            )

        # ----------------------------------------------------
        # Discard terminated episodes
        # ----------------------------------------------------

        if terminated:

            print(
                f"Episode terminated after "
                f"{steps} steps -> discarded"
            )

            continue

        print(
            f"Episode accepted: {steps} steps"
        )

        results[grab_flag].append(
            {
                "time": np.asarray(times),
                "real": np.asarray(
                    real_velocities
                ),
                "mlp": np.asarray(
                    mlp_velocities
                ),
                "target": np.asarray(
                    target_velocities
                )
            }
        )

    env.close()

    # ========================================================
    # Statistics
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "PAYLOAD DYNAMICS COMPARISON"
    )

    print(
        "========================================"
    )

    for grab_flag in [False, True]:

        group = results[grab_flag]

        label = (
            "PAYLOAD ATTACHED"
            if grab_flag
            else "NO PAYLOAD"
        )

        print(
            f"\n--- {label} ---"
        )

        print(
            f"Episodes: {len(group)}"
        )

        if len(group) == 0:

            print("No valid episodes.")

            continue

        all_errors = []
        all_delta_real = []
        all_delta_mlp = []

        for result in group:

            real = result["real"]
            mlp = result["mlp"]

            # -----------------------------------------------
            # Velocity prediction error
            # -----------------------------------------------

            error = (
                real - mlp
            )

            all_errors.append(
                error
            )

            # -----------------------------------------------
            # Actual velocity increments
            # -----------------------------------------------

            delta_real = (
                np.diff(
                    real,
                    axis=0
                )
            )

            delta_mlp = (
                np.diff(
                    mlp,
                    axis=0
                )
            )

            all_delta_real.append(
                delta_real
            )

            all_delta_mlp.append(
                delta_mlp
            )

        all_errors = np.concatenate(
            all_errors,
            axis=0
        )

        all_delta_real = np.concatenate(
            all_delta_real,
            axis=0
        )

        all_delta_mlp = np.concatenate(
            all_delta_mlp,
            axis=0
        )

        # ----------------------------------------------------
        # Prediction RMSE
        # ----------------------------------------------------

        rmse = np.sqrt(
            np.mean(
                all_errors ** 2,
                axis=0
            )
        )

        total_rmse = np.sqrt(
            np.mean(
                all_errors ** 2
            )
        )

        print(
            "\nMLP prediction RMSE:"
        )

        print(
            f"  vx = {rmse[0]:.6f} m/s"
        )

        print(
            f"  vy = {rmse[1]:.6f} m/s"
        )

        print(
            f"  vz = {rmse[2]:.6f} m/s"
        )

        print(
            f"  Total = {total_rmse:.6f} m/s"
        )

        # ----------------------------------------------------
        # Real dynamics
        # ----------------------------------------------------

        delta_rmse = np.sqrt(
            np.mean(
                all_delta_real ** 2,
                axis=0
            )
        )

        delta_mean = np.mean(
            all_delta_real,
            axis=0
        )

        delta_std = np.std(
            all_delta_real,
            axis=0
        )

        delta_abs_mean = np.mean(
            np.abs(all_delta_real),
            axis=0
        )

        print(
            "\nActual velocity increment:"
        )

        print(
            f"  Mean Δvx = {delta_mean[0]:.6e}"
        )

        print(
            f"  Mean Δvy = {delta_mean[1]:.6e}"
        )

        print(
            f"  Mean Δvz = {delta_mean[2]:.6e}"
        )

        print(
            "\nStd of Δv:"
        )

        print(
            f"  Std Δvx = {delta_std[0]:.6e}"
        )

        print(
            f"  Std Δvy = {delta_std[1]:.6e}"
        )

        print(
            f"  Std Δvz = {delta_std[2]:.6e}"
        )

        print(
            "\nMean |Δv|:"
        )

        print(
            f"  |Δvx| = {delta_abs_mean[0]:.6e}"
        )

        print(
            f"  |Δvy| = {delta_abs_mean[1]:.6e}"
        )

        print(
            f"  |Δvz| = {delta_abs_mean[2]:.6e}"
        )

        print(
            f"\nRMS |Δv| = "
            f"{np.sqrt(np.mean(all_delta_real ** 2)):.6e}"
        )

    # ========================================================
    # Direct comparison
    # ========================================================

    if (
        len(results[False]) > 0
        and len(results[True]) > 0
    ):

        print(
            "\n========================================"
        )

        print(
            "DIRECT DYNAMICS COMPARISON"
        )

        print(
            "========================================"
        )

        def collect_delta(group):

            values = []

            for result in group:

                values.append(
                    np.diff(
                        result["real"],
                        axis=0
                    )
                )

            return np.concatenate(
                values,
                axis=0
            )

        delta_no_payload = collect_delta(
            results[False]
        )

        delta_payload = collect_delta(
            results[True]
        )

        rms_no_payload = np.sqrt(
            np.mean(
                delta_no_payload ** 2,
                axis=0
            )
        )

        rms_payload = np.sqrt(
            np.mean(
                delta_payload ** 2,
                axis=0
            )
        )

        print(
            "\nRMS velocity increment:"
        )

        print(
            f"{'':15s}"
            f"{'No payload':>15s}"
            f"{'Payload':>15s}"
        )

        print(
            f"{'vx':15s}"
            f"{rms_no_payload[0]:15.6e}"
            f"{rms_payload[0]:15.6e}"
        )

        print(
            f"{'vy':15s}"
            f"{rms_no_payload[1]:15.6e}"
            f"{rms_payload[1]:15.6e}"
        )

        print(
            f"{'vz':15s}"
            f"{rms_no_payload[2]:15.6e}"
            f"{rms_payload[2]:15.6e}"
        )

        # ----------------------------------------------------
        # Relative change
        # ----------------------------------------------------

        relative_change = (
            (
                rms_payload
                - rms_no_payload
            )
            / np.maximum(
                rms_no_payload,
                1e-12
            )
            * 100.0
        )

        print(
            "\nChange caused by payload:"
        )

        print(
            f"  vx: {relative_change[0]:+.2f}%"
        )

        print(
            f"  vy: {relative_change[1]:+.2f}%"
        )

        print(
            f"  vz: {relative_change[2]:+.2f}%"
        )

    # ========================================================
    # Plot dynamics distributions
    # ========================================================

    if (
        len(results[False]) > 0
        and len(results[True]) > 0
    ):

        delta_no_payload = collect_delta(
            results[False]
        )

        delta_payload = collect_delta(
            results[True]
        )

        # ----------------------------------------------------
        # Histogram of Δv
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 10)
        )

        labels = [
            "Δvx [m/s]",
            "Δvy [m/s]",
            "Δvz [m/s]"
        ]

        for i in range(3):

            axes[i].hist(
                delta_no_payload[:, i],
                bins=80,
                alpha=0.6,
                label="No payload"
            )

            axes[i].hist(
                delta_payload[:, i],
                bins=80,
                alpha=0.6,
                label="Payload attached"
            )

            axes[i].set_xlabel(
                labels[i]
            )

            axes[i].set_ylabel(
                "Samples"
            )

            axes[i].grid(
                True,
                alpha=0.3
            )

            axes[i].legend()

        fig.suptitle(
            "Velocity increment distribution"
        )

        plt.tight_layout()
        plt.show()

        # ----------------------------------------------------
        # RMS comparison
        # ----------------------------------------------------

        rms_no_payload = np.sqrt(
            np.mean(
                delta_no_payload ** 2,
                axis=0
            )
        )

        rms_payload = np.sqrt(
            np.mean(
                delta_payload ** 2,
                axis=0
            )
        )

        x = np.arange(3)
        width = 0.35

        plt.figure(
            figsize=(10, 6)
        )

        plt.bar(
            x - width / 2,
            rms_no_payload,
            width,
            label="No payload"
        )

        plt.bar(
            x + width / 2,
            rms_payload,
            width,
            label="Payload attached"
        )

        plt.xticks(
            x,
            ["vx", "vy", "vz"]
        )

        plt.ylabel(
            "RMS velocity increment [m/s]"
        )

        plt.title(
            "Effect of payload on system dynamics"
        )

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=20
    )

    parser.add_argument(
        "--render",
        action="store_true"
    )

    args = parser.parse_args()

    closed_loop_validation(
        episodes=args.episodes,
        render=args.render
    )
