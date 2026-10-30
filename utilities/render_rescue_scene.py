"""Snapshots of one scenario flown by the three director methods (thesis figure).

The frames are not simulated again: the qpos logged by compare_methods
(results/analysis/methods/episodes.pkl) is replayed in the MuJoCo model and
rendered, with the path of the drone drawn up to the frame. The rows are
the RL director, the MPC director and the RL director + safety filter.

Usage:
    MUJOCO_GL=egl python -m utilities.render_rescue_scene                                   # thesis (all three rows)
    MUJOCO_GL=egl python -m utilities.render_rescue_scene --seed 100 \
        --times_rl 2.0,2.6,3.0,3.3 --times_mpc 4.0,4.6,5.2,6.0
"""

import argparse
import os
import pickle

import mujoco
import numpy as np

from utilities.analyse import make_director_env

EPISODES = "results/analysis/methods/episodes.pkl"
OUT_DIR = "results/analysis/methods"          # rendered frames (rescue_frames/)
FIG_DIR = "results/thesis_figures"
W, H = 800, 600
COLORS = {"director": [0.23, 0.44, 0.71, 1], "director_mpc": [0.88, 0.54, 0.17, 1],
          "director_safety": [0.11, 0.64, 0.61, 1]}
ROWS = [("director", "RL director"), ("director_mpc", "MPC director"),
        ("director_safety", "RL + safety filter")]


def render_frames(seed, frames, camera, frame_dir):
    """frames: {method: [t, ...]} -> {(method, t): image file}."""
    from PIL import Image

    with open(EPISODES, "rb") as f:
        results = pickle.load(f)
    eps = {k: next(e for e in v if e["seed"] == seed) for k, v in results.items()}
    del results
    env = make_director_env(False)
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = W, H
    env.reset(seed=seed)                    # same payload, goal and heading as logged
    env.model.site_rgba[:, 3] = 0.0         # only the goal marker of the sites
    env.model.site_rgba[env.goal_id] = [0.0, 1.0, 0.0, 0.5]
    renderer = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance = camera["lookat"], camera["distance"]
    cam.azimuth, cam.elevation = camera["azimuth"], camera["elevation"]
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    opt.sitegroup[0] = 1
    os.makedirs(frame_dir, exist_ok=True)
    files = {}
    for kind, times in frames.items():
        ep = eps[kind]
        dt = ep["t"][1] - ep["t"][0]
        for t in times:
            k = min(int(round(t / dt)) - 1, len(ep["t"]) - 1)
            env.data.qpos[:] = ep["qpos"][k]
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, cam, scene_option=opt)
            scn, path = renderer.scene, ep["p"][:k + 1:2]
            for a, b in zip(path[:-1], path[1:]):       # path of the drone
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                    np.zeros(9), np.array(COLORS[kind], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, a, b)
                scn.ngeom += 1
            files[kind, t] = os.path.join(frame_dir, f"{kind}_{t:.2f}.png")
            Image.fromarray(renderer.render()).save(files[kind, t])
    renderer.close()
    env.close()
    return files


def compose(files, frames, notes, out, rows=ROWS):
    """3 x n grid: one row per method, time stamp and note in each frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = max(len(v) for v in frames.values())
    plt.rcParams.update({"font.size": 11})
    fig, axs = plt.subplots(len(rows), n, figsize=(12, 1.87 * len(rows)), squeeze=False,
                            gridspec_kw=dict(wspace=0.03, hspace=0.04, left=0.06, right=0.995,
                                             top=0.995, bottom=0.005))
    for r, (kind, name) in enumerate(rows):
        for c, t in enumerate(frames[kind]):
            ax = axs[r, c]
            # interpolation "none": the frames go into the pdf at their own resolution
            ax.imshow(mpimg.imread(files[kind, t])[95:545, 20:780],   # crop the sky (keep the drone at the goal)
                      interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.96, f"t = {t:.1f} s", transform=ax.transAxes, va="top",
                    color="white", fontsize=11, fontweight="bold")
            note = notes.get((kind, c))
            if note:
                ax.text(0.03, 0.04, note, transform=ax.transAxes, va="bottom", color="white",
                        fontsize=11, bbox=dict(facecolor="black", alpha=0.45, lw=0, pad=2))
            if c == 0:
                ax.set_ylabel(name, fontsize=12)
    # png and pdf (vector text and frames, the rendered frames embedded as
    # images), transparent background
    for path in (out, os.path.splitext(out)[0] + ".pdf"):
        fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02)
        print("saved:", path)
    plt.close(fig)


# frame times, camera and notes of the scenarios used in the thesis
PRESETS = {
    83: dict(times_rl=[2.8, 3.4, 3.9, 4.29], times_mpc=[4.4, 5.0, 5.6, 6.5],
             lookat=[-0.6, -0.4, 0.5], distance=2.5, azimuth=5.0, elevation=-18.0,
             notes={("director", 2): "overshoot, large swing", ("director", 3): "flips over",
                    ("director_mpc", 3): "holds the goal", ("director_safety", 1): "filter intervenes",
                    ("director_safety", 3): "holds the goal"}),
    1290: dict(times_rl=[2.3, 3.0, 3.5, 3.78], times_mpc=[4.4, 5.0, 5.7, 7.0],
               lookat=[0.75, 0.9, 0.55], distance=3.2, azimuth=90.0, elevation=-20.0,
               notes={("director", 1): "overshoots the goal", ("director", 3): "flips over",
                      ("director_mpc", 3): "holds the goal", ("director_safety", 0): "filter intervened",
                      ("director_safety", 3): "holds the goal"}),
    32: dict(times_rl=[1.95, 2.8, 3.5, 4.1], times_mpc=[3.9, 4.6, 5.3, 6.3],
             lookat=[0.3, 0.75, 0.45], distance=3.3, azimuth=90.0, elevation=-22.0,
             notes={("director", 0): "drags the payload", ("director", 2): "large swing",
                    ("director", 3): "flips over", ("director_mpc", 3): "holds the goal",
                    ("director_safety", 0): "filter: vertical lift",
                    ("director_safety", 3): "holds the goal"}),
}


def main():
    floats = lambda s: [float(v) for v in s.split(",")]
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=1290, help=f"presets: {sorted(PRESETS)}")
    p.add_argument("--times_rl", type=floats, default=None,
                   help="frame times of the RL director and RL + safety filter [s]")
    p.add_argument("--times_mpc", type=floats, default=None, help="frame times of the MPC director [s]")
    p.add_argument("--lookat", type=floats, default=None)
    p.add_argument("--distance", type=float, default=None)
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--elevation", type=float, default=None)
    p.add_argument("--rows", default="director,director_mpc,director_safety",
                   help="methods shown, one row each (thesis: director,director_safety)")
    p.add_argument("--out", default=os.path.join(FIG_DIR, "rescue_scene.png"))
    a = p.parse_args()
    pre = PRESETS.get(a.seed, dict(PRESETS[83], notes={}))
    get = lambda k: getattr(a, k) if getattr(a, k) is not None else pre[k]
    frames = {"director": get("times_rl"), "director_mpc": get("times_mpc"),
              "director_safety": get("times_rl")}
    camera = dict(lookat=get("lookat"), distance=get("distance"), azimuth=get("azimuth"),
                  elevation=get("elevation"))
    files = render_frames(a.seed, frames, camera, os.path.join(OUT_DIR, "rescue_frames"))
    names = dict(ROWS)
    compose(files, frames, pre["notes"], a.out, [(k, names[k]) for k in a.rows.split(",")])


if __name__ == "__main__":
    main()
