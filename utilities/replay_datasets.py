"""Watch how the identification data sets are flown, in the MuJoCo viewer.

The flight is generated live, with the same function that produced the data
set, and every control step is shown in the viewer:

    d1   Dataset 1: P-controller on the waypoints + random commands
         (director_mpc.collect, via resnet_mpc._collect_p)
    d2   Dataset 2: flight of the high-level policy (resnet_mpc._collect_rl)
    t1   Test set 1: test flight around a fixed point, pick-up, flight with
         the payload (test_resnet.fly)

Note: Dataset 2 is reproduced exactly (same policy and seed as the cached
training flights). The cached flights of Dataset 1 and Test set 1 were
recorded with the same procedure, but in an earlier version of the
environment, so the flight shown here is of the same kind, but not identical
to the cached one.

Usage:
    python -m utilities.replay_datasets --set d1 --seed 113
    python -m utilities.replay_datasets --set d2 --seed 579
    python -m utilities.replay_datasets --set t1 --seed 818 --mass 0.13 --speed 2
"""

import argparse
import time

import mujoco
import mujoco.viewer

from multi_drone_mujoco.envs.base_aviary import BaseAviary

VIEW = {"speed": 1.0, "viewer": None, "t_wall": None}


def _show(env):
    """Sync the passive viewer after a control step, at ``speed`` x real time."""
    v = VIEW["viewer"]
    if v is None:
        v = VIEW["viewer"] = mujoco.viewer.launch_passive(env.model, env.data)
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING      # follow the drone
        v.cam.trackbodyid = env.model.body("drone0").id
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.6, -20.0, 135.0
        VIEW["t_wall"] = time.perf_counter()
    if not v.is_running():
        raise KeyboardInterrupt("viewer closed")
    v.sync()
    VIEW["t_wall"] += env.CTRL_TIMESTEP / VIEW["speed"]
    time.sleep(max(0.0, VIEW["t_wall"] - time.perf_counter()))


_step = BaseAviary.step


def _step_and_show(self, action):
    # every env step and director_step end in BaseAviary.step: one frame per control step
    out = _step(self, action)
    _show(self)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--set", choices=["d1", "d2", "t1"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--excite", type=float, default=None,
                   help="d1: command noise amplitude (default: as in the training data, 0.15 or 0.6)")
    p.add_argument("--mass", type=float, default=0.13, help="t1: disk mass [kg]")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    a = p.parse_args()
    VIEW["speed"] = a.speed
    BaseAviary.step = _step_and_show
    try:
        if a.set == "d1":
            from multi_drone_mujoco.envs.resnet_mpc import P_TRAIN, P_VAL, _collect_p
            excite = a.excite if a.excite is not None else dict(P_TRAIN + P_VAL).get(a.seed, 0.15)
            print(f"Dataset 1: seed {a.seed}, command noise +-{excite} m/s")
            r = _collect_p((a.seed, excite))
        elif a.set == "d2":
            from multi_drone_mujoco.envs.resnet_mpc import DIRECTOR_POLICY, RL_DURATION, _collect_rl
            print(f"Dataset 2: seed {a.seed}, high-level policy {DIRECTOR_POLICY}")
            r = _collect_rl((a.seed, DIRECTOR_POLICY, RL_DURATION))
        else:
            from utilities.test_resnet import fly
            print(f"Test set 1: seed {a.seed}, disk mass {a.mass} kg")
            r = fly((a.seed, a.mass))
        print(f"done: {len(r['x']) / 48.0:.1f} s, payload carried in {r['attached'].mean() * 100:.0f} % of the steps")
        v = VIEW["viewer"]
        while v is not None and v.is_running():              # keep the last frame until closed
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        BaseAviary.step = _step
        if VIEW["viewer"] is not None:
            VIEW["viewer"].close()


if __name__ == "__main__":
    main()
