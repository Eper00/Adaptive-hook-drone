import argparse
import numpy as np
import time
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--curriculum_flag",type=str, default="false")
    args = parser.parse_args()
    play(args.model_path, args.env_type, args.episodes,curriculum_flag=args.curriculum_flag.lower() == "true")