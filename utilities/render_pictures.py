"""Render still pictures of the hook drone, with and without the payload, of
the payload alone and of a whole scene (saved to pictures/).

    python utilities/render_pictures.py
"""
import os

import mujoco
import numpy as np
from PIL import Image

from multi_drone_mujoco.envs.adaptive_hook_transport import AdaptiveTransportAviary
from multi_drone_mujoco.envs.hybrid_mpc import HybridMPCAgent

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pictures")
WIDTH, HEIGHT = 1600, 1200

PAYLOAD_GEOMS = ["target_geom", "left_connector_geom", "right_connector_geom", "holder_plate"]
SCENE_SEED = 3     # seed of the random initial yaw in the scene picture
SCENARIO = dict(target_position=(0.6, 0.0, 0.6), goal_position=(0.6, 0.6, 1.0),
                mass=0.15, radius=0.03)


def make_env(scenario=SCENARIO, random_orientation=False, seed=0):
    env = AdaptiveTransportAviary(render_mode="rgb_array")
    env.model.vis.global_.offwidth = WIDTH
    env.model.vis.global_.offheight = HEIGHT
    env.RANDOM_ORIENTATION = random_orientation  # random initial yaw at reset
    env.reset(seed=seed)
    env.set_scenario(**scenario)
    env.model.site_rgba[env.goal_id, 3] = 0.0  # hide the goal marker
    return env


def set_visible(env, geom_ids, visible, rgba_backup):
    for g in geom_ids:
        if g not in rgba_backup:
            rgba_backup[g] = env.model.geom_rgba[g].copy()
        env.model.geom_rgba[g] = rgba_backup[g] if visible else 0.0


def drone_geom_ids(env):
    m = env.model
    drone = m.body("drone0").id
    ids = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b != 0 and b != drone:
            b = m.body_parentid[b]
        if b == drone:
            ids.append(g)
    return ids


def render(env, lookat, distance, azimuth, elevation, path, hide_tendons=False,
           show_sites=False):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    opt = mujoco.MjvOption()
    if not show_sites:
        opt.sitegroup[:] = 0
    if hide_tendons:
        opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    renderer.update_scene(env.data, cam, scene_option=opt)
    Image.fromarray(renderer.render()).save(path)
    renderer.close()
    print(f"saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Drone with the spiral hook, no payload (hovering, hook straight)
    env = make_env()
    backup = {}
    payload_ids = [env.model.geom(n).id for n in PAYLOAD_GEOMS]
    set_visible(env, payload_ids, False, backup)
    env._place_drone(np.array([0.0, 0.0, 0.8]))
    render(env, [0.0, 0.0, 0.62], 0.65, 135, -10,
           os.path.join(OUT_DIR, "drone_hook_no_payload.png"))

    # 2) Payload alone (drone hidden), resting on the floor
    set_visible(env, payload_ids, True, backup)
    drone_ids = drone_geom_ids(env)
    set_visible(env, drone_ids, False, backup)
    c = env.data.qpos[env.target_qpos_adr:env.target_qpos_adr + 3].copy()
    render(env, [c[0], c[1], c[2] / 2 + 0.02], 0.85, 135, -20,
           os.path.join(OUT_DIR, "payload_only.png"), hide_tendons=True)
    env.close()

    # 3) Drone carrying the payload: fly the hybrid MPC mission until the
    #    hook has picked the payload up and lifted it clear of the floor
    env = make_env()
    agent = HybridMPCAgent(env, verbose=True)
    agent.reset()
    lifted_for = 0
    while not agent.finished:
        _, _, terminated, _, _ = agent.step()
        if terminated:
            raise RuntimeError("drone crashed before the picture was taken")
        payload_z = env.data.qpos[env.target_qpos_adr + 2]
        if agent.attached and payload_z > c[2] + 0.25:
            lifted_for += 1
            if lifted_for > int(1.5 / agent.dt):  # let the swing settle
                break
    d = env.pos[0]
    p = env.data.qpos[env.target_qpos_adr:env.target_qpos_adr + 3]
    # look along the cylinder axis: the hook curls in the drone's y-z plane
    render(env, (d + p) / 2 + np.array([0, 0, -0.05]), 1.1, 160, -8,
           os.path.join(OUT_DIR, "drone_hook_with_payload.png"))
    env.close()

    # 4) Whole scene at the start of an episode: the drone at its start
    #    position with a random initial yaw, the payload on the floor and the
    #    green goal marker
    env = make_env(dict(SCENARIO, goal_position=(-0.5, 0.8, 1.0)),
                   random_orientation=True, seed=SCENE_SEED)
    env.model.site_rgba[:, 3] = 0.0  # only the goal marker of the sites
    env.model.site_rgba[env.goal_id] = [0.0, 1.0, 0.0, 0.6]
    print(f"scene: initial yaw {np.degrees(env.rpy[0, 2]):.1f} deg")
    points = np.array([env.pos[0], env.data.qpos[env.target_qpos_adr:env.target_qpos_adr + 3],
                       env.GOAL_POSITION])
    render(env, points.mean(axis=0) + np.array([0, 0, -0.1]), 2.3, 75, -18,
           os.path.join(OUT_DIR, "scene_overview.png"), show_sites=True)
    env.close()


if __name__ == "__main__":
    main()
