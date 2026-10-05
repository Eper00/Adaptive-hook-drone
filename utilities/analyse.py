"""Analyse trained policies and the MPC like play.py, with extra diagnostics.

Tests
-----
velocity       RL velocity controller (AdaptiveVelocityAviary)
               * viewer: an arrow on every rotor, its length proportional to
                 the motor command (-1 -> 0, 0 -> hover, +1 -> max)
               * the same reference velocity profile is flown without and with
                 payload; the reached velocity/position is compared with the
                 reference and the tracking errors are plotted

director       RL director (AdaptiveTransportDirectorAviary + velocity policy)
director_mpc   Director MPC (DirectorMPCAgent) in the same environment
director_resnet_mpc
               Director MPC with the ResNet-identified model
               (ResNetMPCAgent, resnet_mpc.py) in the same environment
director_safety
               RL director on AdaptiveTransportDirectorAviarySafetyFilter: the
               MPC safety filter checks every RL command and replaces it when
               the predicted horizon would violate the constraints
               * viewer: commanded velocity (green) and actual velocity
                 (yellow) arrows on the drone, and a trail coloured by speed
                 (blue = slow ... red = fast)
               * payload test: how the commanded / actual velocities change
                 around the moment the payload is lifted
               * Monte Carlo loop over N episodes with the outcome of every
                 episode (success / payload / stability / unfinished), the
                 episodes saved per outcome and plots of when and where the
                 failures happen
               * director_safety only: the RL proposal (green) and the command
                 the filter applied instead (magenta) are drawn when the filter
                 intervenes, the trail turns magenta there, and extra plots
                 show when / where / how strongly and why (constraint, rate,
                 stall takeover, hook-contact escape) the filter overrides

compare        runs director, director_mpc, director_resnet_mpc and
               director_safety on the same scenarios (same seeds) and compares
               their outcomes, timing, attitude / swing and commands

scenario       one scenario (payload mass, radius, position, goal, heading)
               flown by the RL director, the director MPC and the RL director
               + safety filter (--methods to change the list). The scenario is
               drawn with --seed or read from --scenario FILE, and can be
               overridden with --mass --radius --payload --goal --yaw. Saves
               scenario.json, results.json, one .npz per method and plots of
               the trajectories, time series and a summary; --render shows
               each run in the viewer, --video renders offscreen mp4s (one per
               method + side by side, needs ffmpeg).

All director tests use the same scenario for the same seed, so they can be
compared episode by episode.

Usage
-----
    python -m utilities.analyse --test velocity --render
    python -m utilities.analyse --test director --episodes 100
    python -m utilities.analyse --test director_mpc --episodes 100
    python -m utilities.analyse --test director --episodes 3 --render
    python -m utilities.analyse --test director_safety --episodes 100
    python -m utilities.analyse --test compare --episodes 100
    python -m utilities.analyse --test scenario --seed 3 --max_time 25 --video
    python -m utilities.analyse --test scenario --mass 0.25 --radius 0.02 --payload 0.8 -0.6 --goal -1.2 1.0 --render
    python -m utilities.analyse --test scenario --scenario results/analysis/scenario/seed_3/scenario.json

Results (plots, per-episode .npz files, summary.csv) go to
results/analysis/<test>/.
"""

import argparse
import csv
import json
import os
import sys
import time

# offscreen video without a display: use EGL (must be set before importing mujoco)
if "--video" in sys.argv and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
import mujoco
import numpy as np

from multi_drone_mujoco.envs.director_mpc import measure_swing
from multi_drone_mujoco.envs.mpc_mission import payload_attached

DEFAULT_MODELS = {
    "velocity": "results/final/rl_adaptive_velocity_curriculum/final_model.zip",
    "director": "results/final/rl_adaptive_director_curriculum/final_model.zip",
}
OUTCOMES = ["success", "payload", "stability", "unfinished"]
OUTCOME_COLORS = {"success": "tab:green", "payload": "tab:orange",
                  "stability": "tab:red", "unfinished": "tab:gray"}
STAGES = ["take-off", "pre-grasp", "grasp", "goal"]   # env waypoint index
GOAL_TOLERANCE = 0.15      # [m] drone-goal distance counted as "goal reached"
# why the safety filter overrode the controller (see the safety filter env)
REASONS = ["none", "constraint", "rate", "stall", "contact"]
REASON_COLORS = {"constraint": "tab:red", "rate": "tab:orange", "stall": "tab:blue",
                 "contact": "tab:brown"}


################################################################################
# VIEWER OVERLAYS
################################################################################

class Overlay:
    """Extra geoms (arrows, trail) drawn into the passive viewer's user scene."""

    def __init__(self, env, trail_length=600, trail_every=2, speed_scale=0.8):
        self.env = env
        self.trail = []                 # (position, speed)
        self.trail_length = trail_length
        self.trail_every = trail_every
        self.speed_scale = speed_scale  # speed shown fully red [m/s]
        self._k = 0

    @property
    def scene(self):
        viewer = getattr(self.env, "_viewer", None)
        return None if viewer is None else viewer.user_scn

    @staticmethod
    def _add(scene, geom_type, width, start, end, rgba):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-6:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, geom_type, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, geom_type, width, np.asarray(start, float),
                             np.asarray(end, float))
        scene.ngeom += 1

    def arrow(self, scene, start, vector, rgba, width=0.008):
        self._add(scene, mujoco.mjtGeom.mjGEOM_ARROW, width, start, start + vector, rgba)

    def segment(self, scene, start, end, rgba, width=0.004):
        self._add(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end, rgba)

    def speed_color(self, speed):
        s = float(np.clip(speed / self.speed_scale, 0.0, 1.0))
        return (s, 0.1, 1.0 - s, 1.0)

    def reset(self):
        self.trail = []
        self._k = 0

    def draw_rotor_arrows(self, motor_commands, scale=0.25):
        """Arrow on every rotor along the body z axis, length ~ command."""
        scene = self.scene
        if scene is None:
            return
        m, d = self.env.model, self.env.data
        with self.env._viewer.lock():
            scene.ngeom = 0
            z_axis = d.xmat[m.body("drone0").id].reshape(3, 3)[:, 2]
            for i, a in enumerate(np.clip(motor_commands, -1, 1)):
                start = d.site_xpos[m.site(f"drone0_prop{i}").id]
                level = 0.5 * (a + 1.0)          # -1 -> 0, 0 -> hover, 1 -> max
                self.arrow(scene, start, z_axis * scale * level,
                           (1.0, 0.55 * (1 - level), 0.1, 1.0), width=0.01)

    def draw_director(self, command, velocity, scale=0.5, proposal=None, intervened=False):
        """Command (green) and actual velocity (yellow) arrows + speed trail.

        With a safety filter: when it intervenes, the RL proposal stays green,
        the command the MPC applied instead is magenta and the trail is
        magenta at those points.
        """
        scene = self.scene
        pos = self.env.pos[0].copy()
        speed = float(np.linalg.norm(velocity))
        if self._k % self.trail_every == 0 or intervened:
            self.trail.append((pos, speed, intervened))
            self.trail = self.trail[-self.trail_length:]
        self._k += 1
        if scene is None:
            return
        with self.env._viewer.lock():
            scene.ngeom = 0
            self.paint_director(scene, pos, command, velocity, scale, proposal, intervened)

    def paint_director(self, scene, pos, command, velocity, scale=0.5, proposal=None,
                       intervened=False):
        """Draw the trail and the arrows into ``scene`` (viewer or offscreen)."""
        for (p0, _, _), (p1, s1, i1) in zip(self.trail[:-1], self.trail[1:]):
            self.segment(scene, p0, p1, (1.0, 0.0, 1.0, 1.0) if i1 else self.speed_color(s1),
                         width=0.007 if i1 else 0.004)
        green = command if proposal is None else proposal
        self.arrow(scene, pos, np.asarray(green) * scale, (0.1, 0.9, 0.1, 1.0))
        if intervened:
            self.arrow(scene, pos, np.asarray(command) * scale, (1.0, 0.0, 1.0, 1.0))
        self.arrow(scene, pos, np.asarray(velocity) * scale, (1.0, 0.85, 0.1, 1.0))


def render(env, overlay_fn=None, realtime=True):
    """Draw the overlays (after the viewer exists) and sync the viewer."""
    t0 = time.perf_counter()
    if getattr(env, "_viewer", None) is None:
        env.render()                   # opens the passive viewer
    if overlay_fn is not None:
        overlay_fn()
    env.render()
    if realtime:
        time.sleep(max(0.0, env.CTRL_TIMESTEP - (time.perf_counter() - t0)))


################################################################################
# VELOCITY TEST
################################################################################

# Reference velocity profile (world frame): (duration [s], [vx, vy, vz])
VELOCITY_PROFILE = [
    (1.0, [0.0, 0.0, 0.0]),
    (2.0, [0.5, 0.0, 0.0]),
    (2.0, [0.0, 0.5, 0.0]),
    (1.5, [0.0, 0.0, 0.4]),
    (2.0, [-0.4, -0.4, 0.0]),
    (1.5, [0.0, 0.0, -0.3]),
    (2.0, [0.0, 0.0, 0.0]),
]


def velocity_reference(t):
    t_end = 0.0
    for duration, v in VELOCITY_PROFILE:
        t_end += duration
        if t < t_end:
            return np.array(v, float)
    return np.array(VELOCITY_PROFILE[-1][1], float)


def run_velocity_episode(env, model, seed, with_payload, render_on, overlay):
    """Fly the reference profile once; returns the logged episode."""
    env.GRAB_FLAG_ENABLE = with_payload
    # With GRAB_FLAG_ENABLE the env adds a payload in about half of the
    # resets: search the seed that gives the wanted case.
    while True:
        env.reset(seed=seed)
        if (env.PAYLOAD_INDICATOR < 0.5) == with_payload:
            break
        seed += 1000
    duration = sum(d for d, _ in VELOCITY_PROFILE)
    env.EPISODE_LEN_SEC = duration + 1.0          # instance only: no early truncation
    dt = env.CTRL_TIMESTEP
    p0 = env.pos[0].copy()
    log = {k: [] for k in ("t", "v_ref", "p_ref", "v", "p", "action", "rpy")}
    p_ref = p0.copy()
    t, crashed = 0.0, False
    overlay.reset()
    while t < duration:
        v_ref = velocity_reference(t)
        env.TARGET_VEL = v_ref
        obs = env._computeObs()                   # observation with the new reference
        action, _ = model.predict(obs, deterministic=True)
        log["t"].append(t)
        log["v_ref"].append(v_ref)
        log["p_ref"].append(p_ref.copy())
        log["v"].append(env.vel[0].copy())
        log["p"].append(env.pos[0].copy())
        log["action"].append(np.clip(action[0:4], -1, 1))
        log["rpy"].append(env.rpy[0].copy())
        _, _, terminated, _, _ = env.step(action)
        if render_on:
            render(env, lambda: overlay.draw_rotor_arrows(action[0:4]))
        p_ref = p_ref + v_ref * dt
        t += dt
        if terminated:
            crashed = True
            break
    out = {k: np.asarray(v) for k, v in log.items()}
    out.update(seed=seed, payload=with_payload, crashed=crashed,
               mass=(env.MASS + 0.01) if with_payload else 0.0,
               radius=env.RADIUS if with_payload else 0.0)
    return out


def velocity_test(args):
    from stable_baselines3 import PPO

    from multi_drone_mujoco.envs.adaptive_hook_velocity import AdaptiveVelocityAviary

    model = PPO.load(args.model_path or DEFAULT_MODELS["velocity"])
    env = AdaptiveVelocityAviary(render_mode="human" if args.render else None)
    # payload ranges of the final curriculum level (as in play.py)
    env.MIN_PAYLOAD_MASS, env.MAX_PAYLOAD_MASS = 0.01, 0.25
    env.MIN_PAYLOAD_RADIUS, env.MAX_PAYLOAD_RADIUS = 0.02, 0.04
    overlay = Overlay(env)

    runs = [run_velocity_episode(env, model, args.seed, False, args.render, overlay)]
    for i in range(args.episodes):
        runs.append(run_velocity_episode(env, model, args.seed + 1 + i, True, args.render, overlay))
    env.close()

    out = os.path.join(args.out, "velocity")
    os.makedirs(out, exist_ok=True)
    print("\nVelocity tracking of the reference profile")
    print(f"{'case':28s} {'RMS |v err| [m/s]':>18s} {'max |v err|':>12s} {'final |p err| [m]':>18s}")
    rows = []
    for r in runs:
        ev = np.linalg.norm(r["v"] - r["v_ref"], axis=1)
        ep = np.linalg.norm(r["p"] - r["p_ref"], axis=1)
        label = (f"payload m={r['mass']:.2f} kg R={r['radius']:.3f}" if r["payload"]
                 else "no payload")
        if r["crashed"]:
            label += " (crashed)"
        r["label"] = label
        rows.append((label, np.sqrt(np.mean(ev ** 2)), ev.max(), ep[-1]))
        print(f"{label:28s} {rows[-1][1]:18.3f} {rows[-1][2]:12.3f} {rows[-1][3]:18.3f}")
        np.savez_compressed(os.path.join(out, f"{'payload' if r['payload'] else 'no_payload'}_seed{r['seed']}.npz"),
                            **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
                            mass=r["mass"], radius=r["radius"], crashed=r["crashed"])
    files = plot_velocity(runs, out)
    for f in files:
        print("saved:", f)


def plot_velocity(runs, out):
    import matplotlib.pyplot as plt

    files = []
    colors = ["k"] + [plt.cm.viridis(x) for x in np.linspace(0.1, 0.9, max(1, len(runs) - 1))]

    # 1) velocity components: reference vs reached
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, name in enumerate(["vx", "vy", "vz"]):
        axs[i].plot(runs[0]["t"], runs[0]["v_ref"][:, i], color="tab:red", lw=2.5,
                    alpha=0.5, label="reference")
        for r, c in zip(runs, colors):
            axs[i].plot(r["t"], r["v"][:, i], color=c, lw=1.2, label=r["label"])
        axs[i].set_ylabel(f"{name} [m/s]")
        axs[i].grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=2)
    axs[-1].set_xlabel("t [s]")
    fig.suptitle("Velocity reference vs reached velocity (with and without payload)")
    fig.tight_layout()
    files.append(os.path.join(out, "velocity_tracking.png"))
    fig.savefig(files[-1], dpi=130)

    # 2) tracking errors over time
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for r, c in zip(runs, colors):
        axs[0].plot(r["t"], np.linalg.norm(r["v"] - r["v_ref"], axis=1), color=c, label=r["label"])
        axs[1].plot(r["t"], np.linalg.norm(r["p"] - r["p_ref"], axis=1), color=c)
    axs[0].set_ylabel("|v - v_ref| [m/s]")
    axs[1].set_ylabel("|p - p_ref| [m]\n(p_ref = integrated v_ref)")
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=2)
    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    files.append(os.path.join(out, "velocity_errors.png"))
    fig.savefig(files[-1], dpi=130)

    # 3) reached vs reference trajectory
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*(runs[0]["p_ref"] - runs[0]["p_ref"][0]).T, color="tab:red", lw=2.5, alpha=0.5,
            label="reference")
    for r, c in zip(runs, colors):
        ax.plot(*(r["p"] - r["p"][0]).T, color=c, lw=1.2, label=r["label"])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Trajectory relative to the start")
    ax.legend(fontsize=7)
    ax2 = fig.add_subplot(1, 2, 2)
    labels = [r["label"] for r in runs]
    rms = [np.sqrt(np.mean(np.linalg.norm(r["v"] - r["v_ref"], axis=1) ** 2)) for r in runs]
    drift = [np.linalg.norm(r["p"][-1] - r["p_ref"][-1]) for r in runs]
    x = np.arange(len(runs))
    ax2.bar(x - 0.2, rms, 0.4, label="RMS |v err| [m/s]", color="tab:blue")
    ax2.bar(x + 0.2, drift, 0.4, label="final |p err| [m]", color="tab:orange")
    ax2.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("Tracking error summary")
    fig.tight_layout()
    files.append(os.path.join(out, "velocity_trajectory.png"))
    fig.savefig(files[-1], dpi=130)

    # 4) motor commands
    fig, axs = plt.subplots(len(runs), 1, figsize=(12, 2.2 * len(runs)), sharex=True, squeeze=False)
    for ax, r in zip(axs[:, 0], runs):
        ax.plot(r["t"], r["action"])
        ax.set_ylabel("motor cmd")
        ax.set_title(r["label"], fontsize=9)
        ax.grid(alpha=0.3)
    axs[-1, 0].set_xlabel("t [s]")
    fig.tight_layout()
    files.append(os.path.join(out, "velocity_motor_commands.png"))
    fig.savefig(files[-1], dpi=110)
    plt.close("all")
    return files


################################################################################
# DIRECTOR TESTS (RL and MPC)
################################################################################

def make_director_env(render_on, safety_filter=False):
    if safety_filter:
        from multi_drone_mujoco.envs.adaptive_hook_director_safety_filter import (
            AdaptiveTransportDirectorAviarySafetyFilter as EnvClass,
        )
    else:
        from multi_drone_mujoco.envs.adaptive_hook_director_velocity import (
            AdaptiveTransportDirectorAviary as EnvClass,
        )
    env = EnvClass(render_mode="human" if render_on else None)
    # final curriculum level, as in play.py
    env.GRAB_FLAG_ENABLE = True
    env.MIN_PAYLOAD_MASS, env.MAX_PAYLOAD_MASS = 0.01, 0.25
    env.MIN_PAYLOAD_RADIUS, env.MAX_PAYLOAD_RADIUS = 0.02, 0.04
    env.GOAL_RANDOM_AMPLITUDE = 1.5
    env.PAYLOAD_TERMINATION = True
    env.RANDOM_ORIENTATION = True      # random initial yaw
    return env


class RLDirector:
    """RL director policy with the env's own step (filter + velocity policy).

    On the safety-filter env the command that reaches the system can differ
    from the policy's proposal; both are returned.
    """
    name = "RL director"

    def __init__(self, env, model_path, name=None):
        from stable_baselines3 import PPO
        self.env = env
        self.model = PPO.load(model_path)
        if name:
            self.name = name

    def reset(self, max_time):
        self.env.EPISODE_LEN_SEC = max_time       # instance only
        self.obs = self.env._computeObs()

    def step(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        # float64 as in the safety-filter env: with the float32 action the
        # command filter differs by ~1e-8, which the closed loop amplifies, so
        # RL and RL + filter would not fly the same until the first intervention
        action = np.asarray(action, dtype=float)
        self.obs, _, terminated, truncated, info = self.env.step(action)
        proposal = np.clip(np.asarray(action[0:3], float), -1, 1)
        applied = np.asarray(info.get("u_applied", proposal), float)
        return applied, terminated, truncated, {
            "cmd_rl": proposal, "intervened": bool(info.get("safety_intervened", False)),
            "reason": REASONS.index(info.get("safety_reason", "none"))}


class MPCDirector:
    """Director MPC agent (plans the same velocity commands); ``resnet=True``
    uses the ResNet-identified model (resnet_mpc.py) instead of the linear one."""
    name = "Director MPC"

    def __init__(self, env, resnet=False):
        self.env = env
        if resnet:
            from multi_drone_mujoco.envs.resnet_mpc import ResNetMPCAgent
            self.agent = ResNetMPCAgent(env)
            self.name = "Director ResNet-MPC"
        else:
            from multi_drone_mujoco.envs.director_mpc import DirectorMPCAgent
            self.agent = DirectorMPCAgent(env)

    def reset(self, max_time):
        self.agent.max_time = max_time
        self.agent.reset()

    def step(self):
        _, _, terminated, truncated, _ = self.agent.step()
        cmd = (np.asarray(self.agent.log["u"][-1], float) if self.agent.log["u"]
               else np.zeros(3))
        return cmd, terminated, truncated, {"cmd_rl": cmd, "intervened": False, "reason": 0}


def classify(ep):
    """Outcome of an episode and when / at which stage it was decided.

    stability  : the drone flipped (|roll| or |pitch| > pi/2) or hit the floor
    payload    : the payload was lifted but lost again
    unfinished : no crash, payload not lost, but the goal was not reached
                 with the payload in time (e.g. never grasped, too slow)
    success    : goal reached with the payload, still carried at the end
    """
    att = ep["attached"].astype(bool)
    goal_dist = np.linalg.norm(ep["p"] - ep["goal"], axis=1)
    reached = np.nonzero(att & (goal_dist < GOAL_TOLERANCE))[0]
    k_end = len(ep["t"]) - 1
    if ep["crashed"]:
        return "stability", k_end
    lifted = np.nonzero(att)[0]
    if lifted.size and not att[-1]:
        lost = lifted[-1] + 1
        return "payload", min(lost, k_end)
    if reached.size and att[-1]:
        return "success", reached[0]
    return "unfinished", k_end


def run_director_episode(controller, env, seed, max_time, render_on, overlay, scenario=None):
    """One episode of a director controller. ``scenario`` (see
    scenario_from_env) replays a given payload / goal / heading instead of
    the one drawn with ``seed``."""
    env.reset(seed=seed)
    if scenario is not None:
        apply_scenario(env, scenario)
    controller.reset(max_time)
    a = env.target_qpos_adr
    payload_start = env.data.qpos[a:a + 3].copy()
    drone_start = env.pos[0].copy()
    rest_z = payload_start[2]
    dt = env.CTRL_TIMESTEP
    log = {k: [] for k in ("t", "p", "v", "rpy", "cmd", "cmd_filtered", "attached",
                           "payload_pos", "waypoint", "cmd_rl", "intervened", "swing",
                           "reason", "qpos")}
    attached, crashed, t = False, False, 0.0
    overlay.reset()
    while t < max_time:
        cmd, terminated, truncated, extra = controller.step()
        t += dt
        attached = payload_attached(env, rest_z, attached)
        v = env.vel[0].copy()
        log["t"].append(t)
        log["p"].append(env.pos[0].copy())
        log["v"].append(v)
        log["rpy"].append(env.rpy[0].copy())
        log["cmd"].append(cmd)
        log["cmd_filtered"].append(np.zeros(3) if env.prev_action is None
                                   else np.asarray(env.prev_action[0:3], float))
        log["attached"].append(attached)
        log["payload_pos"].append(env.data.qpos[a:a + 3].copy())
        log["waypoint"].append(int(env.current_waypoint_idx[0]))
        log["cmd_rl"].append(extra["cmd_rl"])
        log["intervened"].append(extra["intervened"])
        log["reason"].append(extra["reason"])
        log["swing"].append(measure_swing(env)[0] if attached else np.zeros(2))
        log["qpos"].append(env.data.qpos.copy())
        if render_on:
            render(env, lambda: overlay.draw_director(cmd, v, proposal=extra["cmd_rl"],
                                                      intervened=extra["intervened"]))
        crashed = (env.pos[0, 2] < 0.0) or bool(np.any(np.abs(env.rpy[0, 0:2]) > np.pi / 2))
        if crashed or terminated or truncated:
            break
    ep = {k: np.asarray(v) for k, v in log.items()}
    ep.update(seed=seed, crashed=crashed, goal=np.asarray(env.GOAL_POSITION, float),
              payload_start=payload_start, start=drone_start,
              mass=env.MASS + 0.01, radius=env.RADIUS)
    outcome, k = classify(ep)
    ep.update(outcome=outcome, event_step=k, event_time=float(ep["t"][k]),
              event_stage=STAGES[min(int(ep["waypoint"][k]), len(STAGES) - 1)])
    return ep


CONTROLLERS = {
    "director": ("RL director", False),
    "director_mpc": ("Director MPC", False),
    "director_resnet_mpc": ("Director ResNet-MPC", False),
    "director_safety": ("RL director + safety filter", True),
}


def make_controller(kind, args):
    name, safety = CONTROLLERS[kind]
    env = make_director_env(args.render, safety_filter=safety)
    if kind in ("director_mpc", "director_resnet_mpc"):
        controller = MPCDirector(env, resnet=kind == "director_resnet_mpc")
    else:
        controller = RLDirector(env, args.model_path or DEFAULT_MODELS["director"], name=name)
    return env, controller


def monte_carlo(kind, args, out):
    """N episodes of one controller; saves episodes, summary and plots."""
    env, controller = make_controller(kind, args)
    overlay = Overlay(env)
    for o in OUTCOMES:
        os.makedirs(os.path.join(out, "episodes", o), exist_ok=True)

    print(f"{controller.name}: {args.episodes} Monte Carlo episodes"
          f" (seeds {args.seed}..{args.seed + args.episodes - 1}, max {args.max_time:.0f} s)")
    episodes, counts = [], {o: 0 for o in OUTCOMES}
    for i in range(args.episodes):
        seed = args.seed + i
        ep = run_director_episode(controller, env, seed, args.max_time, args.render, overlay)
        episodes.append(ep)
        counts[ep["outcome"]] += 1
        n = i + 1
        np.savez_compressed(
            os.path.join(out, "episodes", ep["outcome"], f"ep_{seed:05d}.npz"),
            **{k: v for k, v in ep.items() if isinstance(v, np.ndarray)},
            **{k: ep[k] for k in ("seed", "crashed", "mass", "radius", "outcome",
                                  "event_time", "event_stage")})
        extra = ""
        if CONTROLLERS[kind][1]:
            by_reason = ", ".join(f"{r} {100 * np.mean(ep['reason'] == REASONS.index(r)):.1f}%"
                                  for r in REASON_COLORS)
            extra = f" | filter active {100 * ep['intervened'].mean():4.1f}% ({by_reason})"
        print(f"  episode {n:4d} seed {seed}: {ep['outcome']:10s}"
              f" (t={ep['event_time']:5.1f}s, stage {ep['event_stage']}, m={ep['mass']:.2f} kg)"
              f" | success {counts['success']}/{n} = {counts['success'] / n:.2f}"
              f" | payload {counts['payload']} stability {counts['stability']}"
              f" unfinished {counts['unfinished']}{extra}")
    env.close()

    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "outcome", "event_time", "event_stage", "mass", "radius",
                    "start_x", "start_y", "start_z", "payload_x", "payload_y",
                    "goal_x", "goal_y", "picked_up", "filter_active_fraction"])
        for ep in episodes:
            w.writerow([ep["seed"], ep["outcome"], round(ep["event_time"], 3), ep["event_stage"],
                        round(ep["mass"], 4), round(ep["radius"], 4), *np.round(ep["start"], 3),
                        *np.round(ep["payload_start"][:2], 3), *np.round(ep["goal"][:2], 3),
                        bool(ep["attached"].any()), round(float(ep["intervened"].mean()), 4)])
    with open(os.path.join(out, "counts.json"), "w") as f:
        json.dump({"controller": controller.name, "episodes": len(episodes), **counts}, f, indent=2)

    print(f"\n{controller.name}: success {counts['success']}/{len(episodes)},"
          f" failed payload {counts['payload']}, failed stability {counts['stability']},"
          f" unfinished {counts['unfinished']}")
    files = plot_monte_carlo(episodes, counts, controller.name, out)
    files += plot_payload_effect(episodes, controller.name, out)
    if CONTROLLERS[kind][1]:
        files += plot_interventions(episodes, controller.name, out)
    for f in files:
        print("saved:", f)
    return controller.name, episodes, counts


def director_test(args, kind):
    monte_carlo(kind, args, os.path.join(args.out, kind))


def compare_test(args):
    out = os.path.join(args.out, "compare")
    results = {}
    for kind in CONTROLLERS:
        name, episodes, counts = monte_carlo(kind, args, os.path.join(out, kind))
        results[kind] = (name, episodes, counts)
    files = plot_comparison(results, out)
    with open(os.path.join(out, "paired_outcomes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        kinds = list(results)
        w.writerow(["seed", "mass", "radius"] + [results[k][0] for k in kinds])
        for i, ep in enumerate(results[kinds[0]][1]):
            w.writerow([ep["seed"], round(ep["mass"], 4), round(ep["radius"], 4)]
                       + [results[k][1][i]["outcome"] for k in kinds])
    files.append(os.path.join(out, "paired_outcomes.csv"))
    print("\nComparison on the same scenarios:")
    print(f"{'controller':30s}" + "".join(f"{o:>12s}" for o in OUTCOMES)
          + f"{'t_goal med':>12s}{'max tilt med':>14s}")
    for kind, (name, episodes, counts) in results.items():
        tg = [ep["event_time"] for ep in episodes if ep["outcome"] == "success"]
        tilt = [np.degrees(np.abs(ep["rpy"][:, 0:2]).max()) for ep in episodes]
        print(f"{name:30s}" + "".join(f"{counts[o]:12d}" for o in OUTCOMES)
              + f"{(np.median(tg) if tg else float('nan')):12.1f}{np.median(tilt):14.1f}")
    for f in files:
        print("saved:", f)


def plot_monte_carlo(episodes, counts, name, out):
    import matplotlib.pyplot as plt

    files = []
    n = len(episodes)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # outcome counts
    ax = axs[0, 0]
    vals = [counts[o] for o in OUTCOMES]
    bars = ax.bar(OUTCOMES, vals, color=[OUTCOME_COLORS[o] for o in OUTCOMES])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({100 * v / max(n, 1):.0f}%)",
                ha="center", va="bottom")
    ax.set_title(f"{name}: outcomes of {n} episodes")
    ax.set_ylabel("episodes")

    # when: failure time per outcome
    ax = axs[0, 1]
    failures = [o for o in OUTCOMES if o != "success"]
    data = [[ep["event_time"] for ep in episodes if ep["outcome"] == o] for o in failures]
    if any(data):
        ax.hist([d for d in data], bins=20, stacked=True,
                color=[OUTCOME_COLORS[o] for o in failures], label=failures)
        ax.legend()
    ax.set_xlabel("time of failure [s]")
    ax.set_ylabel("episodes")
    ax.set_title("When the failures happen")

    # at which stage (env waypoint) the failures happen
    ax = axs[1, 0]
    bottom = np.zeros(len(STAGES))
    for o in failures:
        c = np.array([sum(ep["outcome"] == o and ep["event_stage"] == s for ep in episodes)
                      for s in STAGES])
        ax.bar(STAGES, c, bottom=bottom, color=OUTCOME_COLORS[o], label=o)
        bottom += c
    ax.set_title("Mission stage at failure (active env waypoint)")
    ax.set_ylabel("episodes")
    ax.legend()

    # success rate over payload mass
    ax = axs[1, 1]
    masses = np.array([ep["mass"] for ep in episodes])
    edges = np.linspace(masses.min(), masses.max() + 1e-9, 6) if n else np.linspace(0, 1, 6)
    for o in OUTCOMES:
        frac = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [(lo <= ep["mass"] < hi) for ep in episodes]
            tot = max(sum(sel), 1)
            frac.append(sum(s and ep["outcome"] == o for s, ep in zip(sel, episodes)) / tot)
        ax.plot(0.5 * (edges[:-1] + edges[1:]), frac, "o-", color=OUTCOME_COLORS[o], label=o)
    ax.set_xlabel("payload mass [kg]")
    ax.set_ylabel("fraction of episodes")
    ax.set_title("Outcome vs payload mass")
    ax.legend()
    for a in axs.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    files.append(os.path.join(out, "monte_carlo_outcomes.png"))
    fig.savefig(files[-1], dpi=130)

    # where: scenario map and failure locations (top view)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    ax = axs[0]
    for ep in episodes:
        c = OUTCOME_COLORS[ep["outcome"]]
        ax.plot(*ep["p"][:, 0:2].T, color=c, lw=0.7, alpha=0.5)
        ax.scatter(*ep["p"][ep["event_step"], 0:2], color=c, marker="x", s=40)
    for o in OUTCOMES:
        ax.plot([], [], color=OUTCOME_COLORS[o], label=o)
    ax.legend()
    ax.set_title("Trajectories (x: success / failure point)")
    ax2 = axs[1]
    for ep in episodes:
        c = OUTCOME_COLORS[ep["outcome"]]
        ax2.scatter(*ep["start"][0:2], color=c, marker="^", s=30)
        ax2.scatter(*ep["payload_start"][0:2], color=c, marker="s", s=30)
        ax2.scatter(*ep["goal"][0:2], color=c, marker="*", s=60)
        ax2.plot([ep["start"][0], ep["payload_start"][0], ep["goal"][0]],
                 [ep["start"][1], ep["payload_start"][1], ep["goal"][1]],
                 color=c, lw=0.5, alpha=0.4)
    ax2.scatter([], [], color="k", marker="^", label="start")
    ax2.scatter([], [], color="k", marker="s", label="payload")
    ax2.scatter([], [], color="k", marker="*", label="goal")
    ax2.legend()
    ax2.set_title("Scenarios coloured by outcome")
    for a in axs:
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_aspect("equal"); a.grid(alpha=0.3)
    fig.tight_layout()
    files.append(os.path.join(out, "monte_carlo_map.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close("all")
    return files


def plot_payload_effect(episodes, name, out, before=3.0, after=6.0):
    """How the commanded and actual velocities change when the payload is lifted."""
    import matplotlib.pyplot as plt

    grid = np.arange(-before, after, 1 / 48)
    curves = {k: [] for k in ("cmd", "cmd_filtered", "v", "err", "tilt")}
    metrics = {k: {"before": [], "after": []} for k in
               ("|command| [m/s]", "command rate [m/s^2]", "|filtered cmd - v| [m/s]", "tilt [deg]")}
    examples = []
    for ep in episodes:
        att = ep["attached"].astype(bool)
        if not att.any():
            continue
        k0 = np.argmax(att)
        t = ep["t"] - ep["t"][k0]
        series = {
            "cmd": np.linalg.norm(ep["cmd"], axis=1),
            "cmd_filtered": np.linalg.norm(ep["cmd_filtered"], axis=1),
            "v": np.linalg.norm(ep["v"], axis=1),
            "err": np.linalg.norm(ep["cmd_filtered"] - ep["v"], axis=1),
            "tilt": np.degrees(np.abs(ep["rpy"][:, 0:2]).max(axis=1)),
        }
        for k, s in series.items():
            curves[k].append(np.interp(grid, t, s, left=np.nan, right=np.nan))
        rate = np.r_[0, np.linalg.norm(np.diff(ep["cmd"], axis=0), axis=1) * 48]
        for label, s in (("|command| [m/s]", series["cmd"]), ("command rate [m/s^2]", rate),
                         ("|filtered cmd - v| [m/s]", series["err"]), ("tilt [deg]", series["tilt"])):
            b = s[(t >= -before) & (t < 0)]
            a = s[(t >= 0) & (t < after)]
            if b.size and a.size:
                metrics[label]["before"].append(b.mean())
                metrics[label]["after"].append(a.mean())
        if len(examples) < 1:
            examples.append((ep, t))
    files = []
    if not curves["cmd"]:
        print(f"  (no episode lifted the payload: skipping the payload plots)")
        return files

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, keys in zip(axs, (("cmd", "cmd_filtered", "v"), ("err",), ("tilt",))):
        for k in keys:
            c = np.array(curves[k])
            mean, std = np.nanmean(c, axis=0), np.nanstd(c, axis=0)
            label = {"cmd": "|agent command|", "cmd_filtered": "|filtered command| (to the RL velocity loop)",
                     "v": "|actual velocity|", "err": "|filtered command - actual velocity|",
                     "tilt": "max(|roll|, |pitch|) [deg]"}[k]
            ax.plot(grid, mean, label=label)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("[m/s]")
    axs[1].set_ylabel("[m/s]")
    axs[2].set_ylabel("[deg]")
    axs[-1].set_xlabel("time relative to payload lift-off [s]  (mean ± std over episodes)")
    fig.suptitle(f"{name}: velocities around the payload lift-off ({len(curves['cmd'])} episodes)")
    fig.tight_layout()
    files.append(os.path.join(out, "payload_effect_timeseries.png"))
    fig.savefig(files[-1], dpi=130)

    fig, axs = plt.subplots(1, len(metrics), figsize=(16, 4))
    for ax, (label, d) in zip(axs, metrics.items()):
        ax.boxplot([d["before"], d["after"]])
        ax.set_xticks([1, 2], ["before lift", "after lift"])
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{name}: per-episode means {before:.0f} s before vs {after:.0f} s after lift-off")
    fig.tight_layout()
    files.append(os.path.join(out, "payload_effect_boxplots.png"))
    fig.savefig(files[-1], dpi=130)

    ep, t = examples[0]
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(ep["t"], ep["cmd"][:, i], color="tab:green", lw=1, label="agent command")
        ax.plot(ep["t"], ep["cmd_filtered"][:, i], color="tab:olive", lw=1.5, label="filtered command")
        ax.plot(ep["t"], ep["v"][:, i], color="tab:orange", lw=1.5, label="actual velocity")
        ax.axvline(ep["t"][np.argmax(ep["attached"])], color="k", ls="--", lw=1)
        ax.set_ylabel(f"v{'xyz'[i]} [m/s]")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel(f"t [s]  (seed {ep['seed']}, dashed: payload lifted)")
    fig.tight_layout()
    files.append(os.path.join(out, "payload_effect_example.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close("all")
    return files


def _outcome_patches():
    from matplotlib.patches import Patch
    return [Patch(color=OUTCOME_COLORS[o], label=o) for o in OUTCOMES]


def _intervals(mask, t):
    """(start, end) times of the True runs of a boolean mask."""
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((t[k] - 1 / 48, t[j]))
            k = j + 1
        else:
            k += 1
    return out


def plot_interventions(episodes, name, out):
    """When, where and how strongly the MPC safety filter overrides the RL."""
    import matplotlib.pyplot as plt

    from multi_drone_mujoco.envs.predictive_safety_filter import SafetyConstraints

    lim = SafetyConstraints()
    files = []
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    ax = axs[0, 0]
    for ep in episodes:
        ax.bar(ep["seed"], 100 * ep["intervened"].mean(), color=OUTCOME_COLORS[ep["outcome"]])
    ax.legend(handles=_outcome_patches(), fontsize=8)
    ax.set_xlabel("seed")
    ax.set_ylabel("steps with intervention [%]")
    ax.set_title("How often the filter overrides the RL (per episode)")

    ax = axs[0, 1]
    width = 0.4
    for j, (label, sel) in enumerate((("without payload", False), ("with payload", True))):
        rate = []
        for s_idx in range(len(STAGES)):
            m = np.concatenate([(ep["waypoint"] == s_idx) & (ep["attached"].astype(bool) == sel)
                                for ep in episodes])
            i = np.concatenate([ep["intervened"].astype(bool) for ep in episodes])
            rate.append(100 * i[m].mean() if m.any() else 0.0)
        ax.bar(np.arange(len(STAGES)) + (j - 0.5) * width, rate, width, label=label)
    ax.set_xticks(np.arange(len(STAGES)), STAGES)
    ax.set_ylabel("steps with intervention [%]")
    ax.set_title("Where in the mission the filter intervenes")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    reason = np.concatenate([ep["reason"] for ep in episodes])
    diff = np.concatenate([np.linalg.norm(ep["cmd"] - ep["cmd_rl"], axis=1) for ep in episodes])
    data = [diff[reason == REASONS.index(r)] for r in REASON_COLORS]
    if any(len(d) for d in data):
        ax.hist(data, bins=30, stacked=True, color=list(REASON_COLORS.values()),
                label=[f"{r} ({len(d)} steps)" for r, d in zip(REASON_COLORS, data)])
        ax.legend(fontsize=8)
    ax.set_xlabel("|applied - proposed| velocity command [m/s]")
    ax.set_ylabel("steps")
    ax.set_title("Why and how strongly the filter overrides")

    ax = axs[1, 1]
    t_max = max(ep["t"][-1] for ep in episodes)
    bins = np.arange(0, t_max + 0.5, 0.5)
    hits = np.zeros(len(bins) - 1)
    alive = np.zeros(len(bins) - 1)
    for ep in episodes:
        idx = np.digitize(ep["t"], bins) - 1
        for b in range(len(bins) - 1):
            sel = idx == b
            if sel.any():
                alive[b] += 1
                hits[b] += ep["intervened"][sel].any()
    ax.plot(0.5 * (bins[:-1] + bins[1:]), 100 * hits / np.maximum(alive, 1), color="tab:purple")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("episodes with an intervention [%]")
    ax.set_title("When the filter intervenes (0.5 s bins)")
    for a in axs.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"{name}: safety filter interventions")
    fig.tight_layout()
    files.append(os.path.join(out, "safety_interventions.png"))
    fig.savefig(files[-1], dpi=130)

    # example: the episode with the most interventions
    ep = max(episodes, key=lambda e: e["intervened"].sum())
    spans = [(r, _intervals(ep["reason"] == REASONS.index(r), ep["t"])) for r in REASON_COLORS]
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for i in range(3):
        ax = axs[i]
        ax.plot(ep["t"], ep["cmd_rl"][:, i], color="tab:green", lw=1, label="RL proposal")
        ax.plot(ep["t"], ep["cmd"][:, i], color="m", lw=1.2, label="applied")
        ax.plot(ep["t"], ep["v"][:, i], color="tab:orange", lw=1, label="actual velocity")
        ax.set_ylabel(f"v{'xyz'[i]} [m/s]")
    axs[3].plot(ep["t"], np.degrees(ep["rpy"][:, 0]), label="roll")
    axs[3].plot(ep["t"], np.degrees(ep["rpy"][:, 1]), label="pitch")
    for sgn in (1, -1):
        axs[3].axhline(sgn * np.degrees(lim.max_tilt), color="r", ls=":", lw=1)
    axs[3].set_ylabel("tilt [deg]")
    axs[4].plot(ep["t"], np.degrees(ep["swing"][:, 0]), label="swing x")
    axs[4].plot(ep["t"], np.degrees(ep["swing"][:, 1]), label="swing y")
    for sgn in (1, -1):
        axs[4].axhline(sgn * np.degrees(lim.max_swing), color="r", ls=":", lw=1)
    axs[4].set_ylabel("payload swing [deg]")
    for ax in axs:
        for r, iv in spans:
            for a0, a1 in iv:
                ax.axvspan(a0, a1, color=REASON_COLORS[r], alpha=0.15, lw=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axs[-1].set_xlabel(f"t [s]  (seed {ep['seed']}, {ep['outcome']}; shaded: filter active -"
                       f" red: constraint, orange: rate, blue: stall takeover;"
                       f" dotted: filter constraints)")
    fig.tight_layout()
    files.append(os.path.join(out, "safety_example.png"))
    fig.savefig(files[-1], dpi=120)
    plt.close("all")
    return files


def plot_comparison(results, out):
    """RL director vs director MPC vs RL + safety filter on the same scenarios."""
    import matplotlib.pyplot as plt

    from multi_drone_mujoco.envs.predictive_safety_filter import SafetyConstraints

    lim = SafetyConstraints()
    os.makedirs(out, exist_ok=True)
    kinds = list(results)
    names = [results[k][0] for k in kinds]
    colors = ["tab:blue", "tab:purple", "tab:olive", "tab:cyan"]
    files = []

    def per_episode(fn):
        return [[fn(ep) for ep in results[k][1]] for k in kinds]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ax = axs[0, 0]
    width = 0.8 / len(kinds)
    for j, k in enumerate(kinds):
        counts = results[k][2]
        n = max(sum(counts.values()), 1)
        ax.bar(np.arange(len(OUTCOMES)) + (j - (len(kinds) - 1) / 2) * width,
               [100 * counts[o] / n for o in OUTCOMES], width, color=colors[j], label=names[j])
    ax.set_xticks(np.arange(len(OUTCOMES)), OUTCOMES)
    ax.set_ylabel("episodes [%]")
    ax.set_title("Outcomes on the same scenarios")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    masses = np.array([ep["mass"] for ep in results[kinds[0]][1]])
    edges = np.linspace(masses.min(), masses.max() + 1e-9, 6)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for j, k in enumerate(kinds):
        eps = results[k][1]
        rate = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [ep for ep in eps if lo <= ep["mass"] < hi]
            rate.append(100 * np.mean([ep["outcome"] == "success" for ep in sel]) if sel else np.nan)
        ax.plot(centers, rate, "o-", color=colors[j], label=names[j])
    ax.set_xlabel("payload mass [kg]")
    ax.set_ylabel("success [%]")
    ax.set_title("Success rate vs payload mass")
    ax.legend(fontsize=8)

    def box(ax, data, title, ylabel, hlines=()):
        data = [d if len(d) else [np.nan] for d in data]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(kinds) + 1), names, rotation=12, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for h in hlines:
            ax.axhline(h, color="r", ls=":", lw=1)

    box(axs[0, 2], [[ep["event_time"] for ep in results[k][1] if ep["outcome"] == "success"]
                    for k in kinds], "Time to reach the goal with the payload (successes)", "[s]")
    box(axs[1, 0], per_episode(lambda ep: np.degrees(np.abs(ep["rpy"][:, 0:2]).max())),
        "Max tilt per episode", "[deg]", (np.degrees(lim.max_tilt),))
    box(axs[1, 1], per_episode(lambda ep: np.degrees(np.abs(ep["swing"]).max())),
        "Max payload swing per episode", "[deg]", (np.degrees(lim.max_swing),))
    box(axs[1, 2], per_episode(lambda ep: np.mean(np.linalg.norm(np.diff(ep["cmd"], axis=0), axis=1)) * 48),
        "Command roughness: mean |d cmd / dt|", "[m/s^2]")
    for a in axs.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("RL director vs director MPC vs RL + MPC safety filter (dotted: filter constraints)")
    fig.tight_layout()
    files.append(os.path.join(out, "comparison.png"))
    fig.savefig(files[-1], dpi=130)

    # paired outcomes, scenario by scenario
    code = {o: i for i, o in enumerate(OUTCOMES)}
    M = np.array([[code[ep["outcome"]] for ep in results[k][1]] for k in kinds])
    fig, ax = plt.subplots(figsize=(max(8, 0.12 * M.shape[1] + 3), 3))
    cmap = matplotlib.colors.ListedColormap([OUTCOME_COLORS[o] for o in OUTCOMES])
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(OUTCOMES) - 0.5,
              interpolation="nearest")
    ax.set_yticks(range(len(kinds)), names)
    seeds = [ep["seed"] for ep in results[kinds[0]][1]]
    step = max(1, len(seeds) // 20)
    ax.set_xticks(range(0, len(seeds), step), seeds[::step])
    ax.set_xlabel("scenario (seed)")
    ax.legend(handles=_outcome_patches(), ncol=4, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.35))
    fig.tight_layout()
    files.append(os.path.join(out, "comparison_paired.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close("all")
    return files


################################################################################
# SAME-SCENARIO TEST (RL director vs director MPC vs RL + safety filter)
################################################################################

SCENARIO_METHODS = ["director", "director_mpc", "director_safety"]
METHOD_COLORS = {"director": "tab:blue", "director_mpc": "tab:purple",
                 "director_resnet_mpc": "tab:olive", "director_safety": "tab:cyan"}


def scenario_from_env(env, seed):
    """The scenario the env drew at its last reset (JSON-serializable)."""
    return {"seed": int(seed),
            "payload_position": [float(v) for v in env.TARGET_POSITION],
            "payload_center": [float(v) for v in
                               env.data.qpos[env.target_qpos_adr:env.target_qpos_adr + 3]],
            "goal_position": [float(v) for v in env.GOAL_POSITION],
            "payload_mass": float(env.MASS),
            "payload_radius": float(env.RADIUS),
            "yaw": float(env.INIT_RPYS[0][2]),
            "start_position": [float(v) for v in env.pos[0]],
            "grasp_position": [float(v) for v in env.GRASP_POSITION],
            "waypoints": np.asarray(env.WAYPOINTS, float).tolist()}


def apply_scenario(env, scenario):
    """Replay a scenario right after env.reset() (env.set_scenario)."""
    env.set_scenario(scenario["payload_position"], scenario["goal_position"],
                     scenario["payload_mass"], scenario["payload_radius"],
                     yaw=scenario["yaw"])


def build_scenario(args):
    """Scenario of --seed, or of --scenario FILE, with the command-line overrides."""
    if args.scenario:
        with open(args.scenario) as f:
            scenario = json.load(f)
        args.seed = scenario["seed"]
    env = make_director_env(False)
    env.reset(seed=args.seed)
    if args.scenario:
        apply_scenario(env, scenario)
    scenario = scenario_from_env(env, args.seed)
    changed = False
    if args.mass is not None:
        scenario["payload_mass"] = args.mass
        changed = True
    if args.radius is not None:
        scenario["payload_radius"] = args.radius
        changed = True
    if args.payload is not None:
        scenario["payload_position"] = [args.payload[0], args.payload[1],
                                        args.payload[2] if len(args.payload) > 2
                                        else scenario["payload_position"][2]]
        changed = True
    if args.goal is not None:
        scenario["goal_position"] = [args.goal[0], args.goal[1],
                                     args.goal[2] if len(args.goal) > 2 else 1.0]
        changed = True
    if args.yaw is not None:
        scenario["yaw"] = float(np.radians(args.yaw))
        changed = True
    if changed:                       # recompute the derived entries (grasp pose, waypoints)
        env.reset(seed=args.seed)
        apply_scenario(env, scenario)
        scenario = scenario_from_env(env, args.seed)
    env.close()
    return scenario


def episode_metrics(ep):
    tilt = np.degrees(np.arccos(np.clip(np.cos(ep["rpy"][:, 0]) * np.cos(ep["rpy"][:, 1]), -1, 1)))
    att = ep["attached"].astype(bool)
    return {"outcome": str(ep["outcome"]), "event_time": float(ep["event_time"]),
            "event_stage": str(ep["event_stage"]),
            "time_to_goal": float(ep["event_time"]) if ep["outcome"] == "success" else None,
            "pickup_time": float(ep["t"][np.argmax(att)]) if att.any() else None,
            "path_length": float(np.sum(np.linalg.norm(np.diff(ep["p"], axis=0), axis=1))),
            "max_tilt_deg": float(tilt.max()),
            "max_speed": float(np.linalg.norm(ep["v"], axis=1).max()),
            "max_swing_deg": float(np.degrees(np.abs(ep["swing"]).max())),
            "command_roughness": float(np.mean(np.linalg.norm(np.diff(ep["cmd"], axis=0), axis=1)) * 48),
            "filter_active_percent": float(100 * ep["intervened"].mean())}


def scenario_test(args):
    """Fly the same scenario with every method, save, plot and render it."""
    methods = [m.strip() for m in args.methods.split(",")]
    scenario = build_scenario(args)
    out = os.path.join(args.out, "scenario", args.tag or f"seed_{scenario['seed']}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scenario.json"), "w") as f:
        json.dump(scenario, f, indent=1)
    print("Scenario:", json.dumps({k: scenario[k] for k in ("seed", "payload_position",
                                                              "goal_position", "payload_mass",
                                                              "payload_radius", "yaw")}))
    results = {}
    for kind in methods:
        env, controller = make_controller(kind, args)
        overlay = Overlay(env)
        ep = run_director_episode(controller, env, scenario["seed"], args.max_time, args.render,
                                  overlay, scenario=scenario)
        env.close()
        name = CONTROLLERS[kind][0]
        results[kind] = (name, ep)
        np.savez_compressed(os.path.join(out, f"{kind}.npz"),
                            **{k: v for k, v in ep.items() if isinstance(v, np.ndarray)},
                            **{k: ep[k] for k in ("seed", "crashed", "mass", "radius", "outcome",
                                                  "event_time", "event_stage")})
        m = episode_metrics(ep)
        print(f"  {name:30s} {m['outcome']:10s} t={m['event_time']:5.1f}s  path {m['path_length']:4.1f} m"
              f"  max tilt {m['max_tilt_deg']:5.1f} deg  max swing {m['max_swing_deg']:5.1f} deg"
              f"  filter active {m['filter_active_percent']:4.1f}%")
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"scenario": scenario,
                   "methods": {k: {"name": n, **episode_metrics(ep)} for k, (n, ep) in results.items()}},
                  f, indent=1)
    files = [os.path.join(out, "scenario.json"), os.path.join(out, "results.json")]
    files += plot_scenario(results, scenario, out)
    if args.video:
        files += render_scenario_video(results, scenario, out, width=args.video_width)
    for f in files:
        print("saved:", f)


def plot_scenario(results, scenario, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    files = []
    goal = np.array(scenario["goal_position"])
    pay = np.array(scenario["payload_center"])
    start = np.array(scenario["start_position"])
    grasp = np.array(scenario["grasp_position"])

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (i, j, lab) in zip(axs, ((0, 1, "y"), (0, 2, "z"))):
        for kind, (name, ep) in results.items():
            c = METHOD_COLORS.get(kind, "k")
            p = ep["p"]
            ax.plot(p[:, i], p[:, j], color=c, lw=1.6, label=f"{name}: {ep['outcome']}")
            att = ep["attached"].astype(bool)
            ax.plot(np.where(att, p[:, i], np.nan), np.where(att, p[:, j], np.nan), color=c, lw=4,
                    alpha=0.35)
            if "intervened" in ep and ep["intervened"].any():
                iv = ep["intervened"].astype(bool)
                ax.scatter(p[iv, i], p[iv, j], s=6, color="m", zorder=3)
            k = ep["event_step"]
            ax.scatter(p[k, i], p[k, j], marker="*" if ep["outcome"] == "success" else "X",
                       s=160, color=c, edgecolor="k", zorder=4)
        ax.scatter(*start[[i, j]], marker="o", s=80, color="k", label="start")
        ax.scatter(*pay[[i, j]], marker="s", s=90, color="r",
                   label=f"payload ({scenario['payload_mass'] + 0.01:.2f} kg, "
                         f"R {100 * scenario['payload_radius']:.1f} cm)")
        ax.scatter(*grasp[[i, j]], marker="^", s=60, color="orange", label="grasp pose")
        ax.scatter(*goal[[i, j]], marker="P", s=120, color="g", label="goal")
        if j == 1:
            ax.add_patch(Circle(goal[:2], GOAL_TOLERANCE, color="g", alpha=0.15))
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(f"{lab} [m]")
        ax.grid(alpha=0.3)
        ax.set_title("top view" if j == 1 else "side view (x-z)")
    axs[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Same scenario (seed {scenario['seed']}): thick = payload carried, "
                 f"magenta dots = safety filter override, * success / X failure")
    fig.tight_layout()
    files.append(os.path.join(out, "scenario_trajectories.png"))
    fig.savefig(files[-1], dpi=130)

    rows = [("distance to goal [m]", lambda ep: np.linalg.norm(ep["p"] - goal, axis=1)),
            ("speed [m/s]", lambda ep: np.linalg.norm(ep["v"], axis=1)),
            ("tilt [deg]", lambda ep: np.degrees(np.arccos(np.clip(
                np.cos(ep["rpy"][:, 0]) * np.cos(ep["rpy"][:, 1]), -1, 1)))),
            ("payload swing [deg]", lambda ep: np.degrees(np.abs(ep["swing"]).max(axis=1))),
            ("|applied command| [m/s]", lambda ep: np.linalg.norm(ep["cmd"], axis=1)),
            ("z [m]", lambda ep: ep["p"][:, 2])]
    fig, axs = plt.subplots(len(rows), 1, figsize=(13, 2.2 * len(rows)), sharex=True)
    for ax, (label, fn) in zip(axs, rows):
        for kind, (name, ep) in results.items():
            c = METHOD_COLORS.get(kind, "k")
            ax.plot(ep["t"], fn(ep), color=c, lw=1.2, label=name)
            att = ep["attached"].astype(bool)
            if att.any():
                ax.axvline(ep["t"][np.argmax(att)], color=c, ls=":", lw=1)
            if kind == "director_safety":
                for r, iv in ((r, _intervals(ep["reason"] == REASONS.index(r), ep["t"]))
                              for r in REASON_COLORS):
                    for a0, a1 in iv:
                        ax.axvspan(a0, a1, color=REASON_COLORS[r], alpha=0.12, lw=0)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="upper right")
    axs[-1].set_xlabel("t [s]  (dotted: payload lifted; shaded: safety filter active - red "
                       "constraint, orange rate, blue stall, brown hook contact)")
    fig.tight_layout()
    files.append(os.path.join(out, "scenario_timeseries.png"))
    fig.savefig(files[-1], dpi=120)

    metrics = {k: episode_metrics(ep) for k, (_, ep) in results.items()}
    names = [results[k][0] for k in results]
    items = [("time_to_goal", "time to goal [s]"), ("path_length", "path length [m]"),
             ("max_tilt_deg", "max tilt [deg]"), ("max_swing_deg", "max swing [deg]"),
             ("command_roughness", "command roughness [m/s^2]"),
             ("filter_active_percent", "filter active [%]")]
    fig, axs = plt.subplots(1, len(items), figsize=(3.2 * len(items), 4))
    for ax, (key, label) in zip(axs, items):
        vals = [metrics[k][key] if metrics[k][key] is not None else np.nan for k in results]
        ax.bar(range(len(vals)), vals, color=[METHOD_COLORS.get(k, "k") for k in results])
        ax.set_xticks(range(len(vals)), names, rotation=30, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Outcomes: " + ", ".join(f"{results[k][0]}: {metrics[k]['outcome']}" for k in results))
    fig.tight_layout()
    files.append(os.path.join(out, "scenario_summary.png"))
    fig.savefig(files[-1], dpi=120)
    plt.close("all")
    return files


class _Ffmpeg:
    """Pipe RGB frames into ffmpeg (H.264 mp4)."""

    def __init__(self, path, width, height, fps):
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found (needed for --video)")
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", path], stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def _label(frame, lines, color=(255, 255, 255)):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    y = 6
    for line in lines:
        d.rectangle([4, y - 2, 8 + 7 * len(line), y + 13], fill=(0, 0, 0))
        d.text((6, y), line, fill=color)
        y += 17
    return np.asarray(img)


def render_scenario_video(results, scenario, out, width=640, every=2):
    """Replay the logged runs offscreen: one mp4 per method and a side-by-side
    comparison (same fixed camera, overlays as in the live viewer)."""
    height = int(width * 0.75) // 2 * 2
    width = width // 2 * 2
    env = make_director_env(False)
    env.reset(seed=scenario["seed"])
    apply_scenario(env, scenario)
    model, data = env.model, env.data
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=20000)
    cam = mujoco.MjvCamera()
    pts = np.array([scenario["start_position"], scenario["payload_center"], scenario["goal_position"]])
    cam.lookat[:] = [*pts[:, :2].mean(0), 0.6]
    cam.distance = 0.9 + 0.95 * float(np.ptp(pts[:, :2], axis=0).max())
    cam.azimuth, cam.elevation = 135.0, -22.0
    fps = int(round(48 / every))
    kinds = list(results)
    writers = {k: _Ffmpeg(os.path.join(out, f"video_{k}.mp4"), width, height, fps) for k in kinds}
    comp = _Ffmpeg(os.path.join(out, "video_comparison.mp4"), width * len(kinds), height, fps)
    overlays = {k: Overlay(env) for k in kinds}
    T = max(len(ep["t"]) for _, ep in results.values())
    for i in range(0, T + fps, every):               # 1 s of the final state at the end
        panels = []
        for k in kinds:
            name, ep = results[k]
            j = min(i, len(ep["t"]) - 1)
            data.qpos[:] = ep["qpos"][j]
            mujoco.mj_forward(model, data)
            ov = overlays[k]
            if i < len(ep["t"]):
                for jj in range(max(0, i - every + 1), i + 1):
                    if jj % ov.trail_every == 0 or ep["intervened"][jj]:
                        ov.trail.append((ep["p"][jj].copy(), float(np.linalg.norm(ep["v"][jj])),
                                         bool(ep["intervened"][jj])))
            renderer.update_scene(data, cam)
            ov.paint_director(renderer.scene, ep["p"][j], ep["cmd"][j], ep["v"][j],
                              proposal=ep["cmd_rl"][j], intervened=bool(ep["intervened"][j]))
            frame = renderer.render()
            status = (f"{ep['outcome']} at {ep['event_time']:.1f}s" if i >= len(ep["t"]) - 1
                      else ("payload attached" if ep["attached"][j] else "no payload"))
            lines = [name, f"t = {ep['t'][j]:5.2f} s   {status}"]
            if ep["intervened"][j]:
                lines.append(f"filter: {REASONS[int(ep['reason'][j])]}")
            frame = _label(frame, lines)
            writers[k].write(frame)
            panels.append(frame)
        comp.write(np.hstack(panels))
    for w in list(writers.values()) + [comp]:
        w.close()
    renderer.close()
    env.close()
    return [w.path for w in writers.values()] + [comp.path]


################################################################################
# MAIN
################################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", choices=["velocity", "director", "director_mpc",
                                           "director_resnet_mpc", "director_safety", "compare",
                                           "scenario"],
                        default="velocity")
    parser.add_argument("--model_path", type=str, default=None,
                        help="policy to load (defaults: results/final/...)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="velocity: payload episodes; director tests: Monte Carlo episodes")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--max_time", type=float, default=10.0,
                        help="director tests: time budget per episode [s] (same for RL and MPC)")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer with overlays")
    parser.add_argument("--show", action="store_true", help="show the plots at the end")
    parser.add_argument("--out", default="results/analysis")
    g = parser.add_argument_group("scenario test (--test scenario)")
    g.add_argument("--methods", default=",".join(SCENARIO_METHODS),
                   help="controllers to fly (default: director,director_mpc,director_safety)")
    g.add_argument("--scenario", default=None, help="replay a saved scenario.json")
    g.add_argument("--mass", type=float, default=None, help="payload (holder) mass [kg]")
    g.add_argument("--radius", type=float, default=None, help="payload radius [m]")
    g.add_argument("--payload", type=float, nargs="+", default=None, metavar="X",
                   help="payload position x y [Z] (Z: height parameter, 0.45-0.8)")
    g.add_argument("--goal", type=float, nargs="+", default=None, metavar="X",
                   help="goal position x y [z]")
    g.add_argument("--yaw", type=float, default=None, help="initial heading [deg]")
    g.add_argument("--video", action="store_true",
                   help="render mp4 videos (per method + side by side) offscreen")
    g.add_argument("--video_width", type=int, default=640)
    g.add_argument("--tag", default=None, help="output folder name (default seed_<seed>)")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    if args.test == "velocity":
        velocity_test(args)
    elif args.test == "compare":
        compare_test(args)
    elif args.test == "scenario":
        scenario_test(args)
    else:
        director_test(args, args.test)
    if args.show:
        import matplotlib.pyplot as plt
        folder = os.path.join(args.out, args.test)
        if args.test == "scenario":
            folder = os.path.join(folder, args.tag or f"seed_{args.seed}")
        for f in sorted(os.listdir(folder)):
            if f.endswith(".png"):
                plt.figure(figsize=(12, 8))
                plt.imshow(plt.imread(os.path.join(folder, f)))
                plt.axis("off")
                plt.title(f)
        plt.show()


if __name__ == "__main__":
    main()
