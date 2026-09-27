"""Pick-and-place mission logic shared by the MPC agents.

``Mission`` turns the transport task (take-off waypoint -> pick-up -> goal)
into a phase machine with a smooth position reference, and ``MissionAgent``
wraps a controller into an RL-policy-like object:

    agent = HybridMPCAgent(env)          # or DirectorMPCAgent(env)
    env.reset(); agent.reset()
    while True:
        obs, reward, terminated, truncated, info = agent.step()
        if terminated or truncated:
            break

``terminated`` means a crash, ``truncated`` that the mission finished (or ran
out of time); ``agent.success`` tells whether the payload was delivered.
"""

import time

import numpy as np

# Grasp geometry (drone position relative to the resting cylinder centre).
# The hook curls towards the drone's -y side for a positive tendon command,
# so the drone hovers on the +y side of the cylinder with yaw = 0.
GRASP_SIDE_CLEARANCE = 0.025     # dy = payload radius + clearance
GRASP_HEIGHT = 0.225             # drone height above the cylinder centre
HOOK_LENGTH = 0.433              # drone COM -> hook tip (straight hook)
TIP_FLOOR_CLEARANCE = 0.03
APPROACH_HEIGHT = 0.35           # extra height while moving next to the payload
LIFT_HEIGHT = 0.4
TENDON_CLOSE = 1.0               # tendon command that closes the hook (full curl)
TENDON_RAMP_TIME = 0.5
HOLD_TIME = 4.0                  # hover at the goal before the mission ends
CONTACT_PHASES = ("close", "lift")  # hook touches the payload on its stand


def grasp_pose(payload_center, radius):
    """Drone position for grasping a cylinder resting at ``payload_center``.

    Shared by the transport environment (its pick-up waypoints) and the MPC
    mission, so RL and MPC aim at the same grasp. The drone hovers on the +y
    side (yaw 0), so a positive tendon command curls the hook under the
    cylinder; its height keeps the straight hook tip off the floor.
    """
    c = np.asarray(payload_center, float)
    dy = radius + GRASP_SIDE_CLEARANCE
    h = max(GRASP_HEIGHT, HOOK_LENGTH + TIP_FLOOR_CLEARANCE - c[2])
    return c + np.array([0.0, dy, h])


class Segment:
    """Straight-line reference with a trapezoidal velocity profile."""

    def __init__(self, start, goal, t0, v_max, a_max=0.6):
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.t0 = t0
        d = self.goal - self.start
        self.length = np.linalg.norm(d)
        self.dir = d / self.length if self.length > 1e-9 else np.zeros(3)
        self.v = min(v_max, np.sqrt(self.length * a_max))
        self.a = a_max
        self.t_acc = self.v / a_max
        self.T = (self.length / self.v + self.t_acc) if self.length > 1e-9 else 0.0

    def __call__(self, t):
        tau = np.clip(t - self.t0, 0.0, self.T)
        if self.T == 0.0:
            return self.goal.copy(), np.zeros(3)
        if tau < self.t_acc:
            s, v = 0.5 * self.a * tau ** 2, self.a * tau
        elif tau < self.T - self.t_acc:
            s, v = 0.5 * self.a * self.t_acc ** 2 + self.v * (tau - self.t_acc), self.v
        else:
            r = self.T - tau
            s, v = self.length - 0.5 * self.a * r ** 2, self.a * r
        return self.start + s * self.dir, v * self.dir

    def finished(self, t):
        return t - self.t0 >= self.T


class Mission:
    """Phase logic of the pick-and-place task.

    Phases: takeoff -> approach -> pre_grasp -> descend -> close -> lift
            -> transport -> hold
    """

    SPEED = {"takeoff": 0.5, "approach": 0.5, "pre_grasp": 0.3, "descend": 0.12,
             "close": 0.1, "lift": 0.25, "transport": 0.4, "hold": 0.1}

    def __init__(self, env, tendon_hold=TENDON_CLOSE):
        self.env = env
        self.tendon_hold = tendon_hold   # tendon command while carrying
        self.phase = "takeoff"
        self.phase_start = 0.0
        self.tendon = 0.0
        self.targets = {"takeoff": np.array(env.WAYPOINTS[0], float)}
        self.segment = None
        self.settle_timer = 0.0
        self.payload_rest = None
        self.visited = []

    def _plan_grasp(self):
        env = self.env
        a = env.target_qpos_adr
        c = env.data.qpos[a:a + 3].copy()
        self.payload_rest = c
        grasp = grasp_pose(c, env.RADIUS)
        self.targets["approach"] = grasp + np.array([0.0, 0.2, APPROACH_HEIGHT])
        self.targets["pre_grasp"] = grasp + np.array([0.0, 0.0, APPROACH_HEIGHT])
        self.targets["descend"] = grasp
        self.targets["close"] = grasp
        self.targets["lift"] = grasp + np.array([0.0, 0.0, LIFT_HEIGHT])
        self.targets["transport"] = np.array(env.GOAL_POSITION, float)
        self.targets["hold"] = np.array(env.GOAL_POSITION, float)

    def _enter(self, phase, t, pos):
        self.phase = phase
        self.phase_start = t
        self.settle_timer = 0.0
        start = pos if self.segment is None else self.segment.goal
        self.segment = Segment(start, self.targets[phase], t, self.SPEED[phase])

    def reference(self, t, dt, n):
        """Position/velocity reference for the horizon t, t+dt, ..., t+n*dt."""
        p = np.zeros((3, n + 1))
        v = np.zeros((3, n + 1))
        for k in range(n + 1):
            p[:, k], v[:, k] = self.segment(t + k * dt)
        return p, v

    def update(self, t, dt, x, attached):
        """Advance the phase machine. Returns False when the mission is over."""
        if self.segment is None:
            self._enter("takeoff", t, x[0:3])
        err = np.linalg.norm(x[0:3] - self.segment.goal)
        speed = np.linalg.norm(x[3:6])
        done_ref = self.segment.finished(t)
        ph = self.phase

        def settled(tol, vtol, hold=0.0):
            if not (done_ref and err < tol and speed < vtol):
                self.settle_timer = 0.0
                return False
            self.settle_timer += dt
            return self.settle_timer >= hold

        if ph == "takeoff" and settled(0.05, 0.1):
            self.visited.append(("takeoff", t))
            self._plan_grasp()
            self._enter("approach", t, x[0:3])
        elif ph == "approach" and settled(0.05, 0.15):
            self._enter("pre_grasp", t, x[0:3])
        elif ph == "pre_grasp" and settled(0.015, 0.05, 0.3):
            self._enter("descend", t, x[0:3])
        elif ph == "descend" and settled(0.012, 0.03, 0.4):
            self.visited.append(("pickup", t))
            self._enter("close", t, x[0:3])
        elif ph == "close":
            self.tendon = TENDON_CLOSE * min(1.0, (t - self.phase_start) / TENDON_RAMP_TIME)
            # Lift right after closing: while the payload still rests on its
            # stand the closed hook levers the drone against it.
            if t - self.phase_start > TENDON_RAMP_TIME:
                self._enter("lift", t, x[0:3])
        elif ph == "lift" and settled(0.05, 0.1):
            self.tendon = self.tendon_hold
            self._enter("transport", t, x[0:3])
        elif ph == "transport" and settled(0.05, 0.1):
            self.visited.append(("goal", t))
            self._enter("hold", t, x[0:3])
        elif ph == "hold" and t - self.phase_start > HOLD_TIME:
            return False
        return True


def payload_attached(env, rest_z, attached):
    """Payload hangs on the hook: lifted off its stand and touching the hook."""
    if rest_z is None:
        return False
    m, d = env.model, env.data
    a = env.target_qpos_adr
    payload = d.qpos[a:a + 3]
    drone = d.qpos[0:3]
    if np.linalg.norm(payload - drone) > 0.7:
        return False
    if attached:
        return True
    target_geom = env.target_geom_id
    hook_geoms = {m.geom(f"segment_geom_{i}").id for i in range(1, 8)}
    touching = any(
        (c.geom1 == target_geom and c.geom2 in hook_geoms)
        or (c.geom2 == target_geom and c.geom1 in hook_geoms)
        for c in d.contact[:d.ncon]
    )
    return touching and payload[2] > rest_z + 0.02


class MissionAgent:
    """Runs the mission with a model-switching controller, one step at a time.

    Subclasses implement ``_measure(attached)``, ``_control(mode, x, p_ref,
    v_ref)`` -> (action, u, X_pred, info), ``_plant_step(action)`` and
    ``_on_attach()``. Every step is logged in ``self.log`` for validation.
    """

    tendon_hold = TENDON_CLOSE

    def __init__(self, env, horizon=40, max_time=45.0, verbose=False):
        self.env = env
        self.horizon = horizon
        self.max_time = max_time
        self.verbose = verbose
        self.dt = env.CTRL_TIMESTEP

    def reset(self):
        """Call right after ``env.reset()``."""
        self.mission = Mission(self.env, tendon_hold=self.tendon_hold)
        self.t = 0.0
        self.attached = False
        self.finished = False
        self.crashed = False
        self.qp_failures = 0
        self.constraint_fallbacks = 0
        self.max_tilt_pred = 0.0
        self.max_tilt_meas = 0.0
        self.phases = []
        self.log = {k: [] for k in ["t", "x", "mode", "phase", "u", "tendon", "X_pred",
                                    "solve_time", "payload_pos", "ref"]}
        self.wall_start = time.perf_counter()
        self._reset_controller()
        self.x = self._measure(False)

    @property
    def success(self):
        return ([v[0] for v in self.mission.visited] == ["takeoff", "pickup", "goal"]
                and self.attached)

    def _say(self, msg):
        if self.verbose:
            print(f"t={self.t:6.2f}s  {msg}")

    def step(self):
        """One control step. Returns the env's (obs, reward, terminated,
        truncated, info); terminated = crash, truncated = mission over."""
        env, mission, x, dt = self.env, self.mission, self.x, self.dt
        if not mission.update(self.t, dt, x, self.attached) or self.t >= self.max_time:
            self.finished = True
            return env._computeObs(), 0.0, False, True, self._info()
        if not self.phases or self.phases[-1][0] != mission.phase:
            self.phases.append((mission.phase, self.t))
            self._say(f"phase -> {mission.phase:10s} pos={np.round(x[0:3], 3)}"
                      f"  mode={'drone+pendulum' if self.attached else 'drone'}")

        mode = int(self.attached)
        p_ref, v_ref = mission.reference(self.t, dt, self.horizon)
        action, u, X_pred, info = self._control(mode, x, p_ref, v_ref)
        self.qp_failures += info["qp_failures"]
        self.constraint_fallbacks += info.get("constraint_fallbacks", 0)
        self.max_tilt_pred = max(self.max_tilt_pred, info.get("max_tilt_pred", 0.0))
        obs, reward, _, _, env_info = self._plant_step(action)

        log = self.log
        log["t"].append(self.t)
        log["x"].append(x)
        log["mode"].append(mode)
        log["phase"].append(mission.phase)
        log["u"].append(u)
        log["tendon"].append(mission.tendon)
        log["X_pred"].append(X_pred)
        log["solve_time"].append(info["solve_time"])
        log["payload_pos"].append(env.data.qpos[env.target_qpos_adr:env.target_qpos_adr + 3].copy())
        log["ref"].append(p_ref[:, 0])
        self.t += dt

        rest_z = None if mission.payload_rest is None else mission.payload_rest[2]
        now_attached = payload_attached(env, rest_z, self.attached)
        if now_attached and not self.attached:
            self._on_attach()
        elif self.attached and not now_attached:
            self._say("payload lost -> drone model")
        self.attached = now_attached
        self.x = self._measure(self.attached)
        self.max_tilt_meas = max(self.max_tilt_meas, float(np.abs(env.rpy[0, 0:2]).max()))

        if env.pos[0, 2] < 0.0 or np.any(np.abs(env.rpy[0, 0:2]) > np.pi / 2):
            self.crashed = True
            self._say("CRASH")
            return obs, reward, True, False, self._info()
        return obs, reward, False, False, self._info()

    def _info(self):
        return {"phase": self.mission.phase, "attached": self.attached,
                "visited": list(self.mission.visited), "success": self.success}

    def summary(self):
        st = np.asarray(self.log["solve_time"])
        return (f"simulated {self.t:.1f}s in {time.perf_counter() - self.wall_start:.1f}s wall time;"
                f" MPC solve mean {1e3 * st.mean():.1f} ms, max {1e3 * st.max():.1f} ms,"
                f" failed QPs {self.qp_failures}, tilt-constraint fallbacks"
                f" {self.constraint_fallbacks}, max |roll|/|pitch| predicted"
                f" {np.degrees(self.max_tilt_pred):.1f} deg, measured"
                f" {np.degrees(self.max_tilt_meas):.1f} deg\n"
                f"visited: {[(n, round(tt, 2)) for n, tt in self.mission.visited]}"
                f" | payload delivered: {self.success}")

    def results(self):
        """Log + metadata as a dict (input of the validation/plot functions)."""
        out = {k: np.asarray(v) for k, v in self.log.items()}
        out.update(x_final=self.x, phases=self.phases, visited=self.mission.visited,
                   targets=self.mission.targets, env_waypoints=self.env.WAYPOINTS.copy(),
                   dt=self.dt, horizon=self.horizon, success=self.success,
                   replay=self._replay)
        return out

    # --- to implement -------------------------------------------------------
    def _reset_controller(self):
        raise NotImplementedError

    def _measure(self, attached):
        raise NotImplementedError

    def _control(self, mode, x, p_ref, v_ref):
        raise NotImplementedError

    def _plant_step(self, action):
        raise NotImplementedError

    def _on_attach(self):
        raise NotImplementedError

    def _replay(self, mode, x0, U):
        raise NotImplementedError
