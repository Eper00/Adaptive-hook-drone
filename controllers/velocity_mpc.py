import numpy as np
import casadi as ca

from multi_drone_mujoco.envs.controllers.casadi_model import load_casadi_model


class VelocityMPC:

    def __init__(
        self,
        model_path,
        horizon=20,
        dt=1.0 / 48.0,
        w_position=10.0,
        w_velocity=0.0,
        w_control=0.1,
        w_delta_control=0.1,
        w_terminal=50.0,
        max_velocity=2.0,
    ):
        self.horizon = horizon
        self.dt = dt
        self.w_position = w_position
        self.w_velocity = w_velocity
        self.w_control = w_control
        self.w_delta_control = w_delta_control
        self.w_terminal = w_terminal
        self.max_velocity = max_velocity

        self.mlp = load_casadi_model(model_path)
        self.last_solution = None
        self._build_mpc()

    def _build_mpc(self):
        N = self.horizon
        self.opti = ca.Opti()

        # State: [px, py, pz, vx, vy, vz]
        self.X = self.opti.variable(6, N + 1)

        # Control: [vx_target, vy_target, vz_target]
        self.U = self.opti.variable(3, N)

        # Parameters
        self.p0 = self.opti.parameter(3)
        self.v0 = self.opti.parameter(3)
        self.goal = self.opti.parameter(3)
        self.tendon = self.opti.parameter(2)
        self.u_previous = self.opti.parameter(3)

        # Initial state
        self.opti.subject_to(self.X[0:3, 0] == self.p0)
        self.opti.subject_to(self.X[3:6, 0] == self.v0)

        # Target velocity limits
        self.opti.subject_to(self.opti.bounded(-0.5, self.U, 0.5))

        cost = 0.0

        for k in range(N):
            p_k = self.X[0:3, k]
            v_k = self.X[3:6, k]
            u_k = self.U[:, k]

            # Learned dynamics
            mlp_input = ca.vertcat(v_k, self.tendon, u_k)
            delta_v = self.mlp(mlp_input)

            v_next = v_k + delta_v
            p_next = p_k + self.dt * v_k

            self.opti.subject_to(self.X[3:6, k + 1] == v_next)
            self.opti.subject_to(self.X[0:3, k + 1] == p_next)

            # Position cost
            cost += self.w_position * ca.sumsqr(p_k - self.goal)

            # Velocity cost
            cost += self.w_velocity * ca.sumsqr(v_k)

            # Control cost
            cost += self.w_control * ca.sumsqr(u_k)

            # Control smoothness
            if k == 0:
                delta_u = u_k - self.u_previous
            else:
                delta_u = u_k - self.U[:, k - 1]

            cost += self.w_delta_control * ca.sumsqr(delta_u)

        # Terminal cost
        cost += self.w_terminal * ca.sumsqr(self.X[0:3, N] - self.goal)

        self.opti.minimize(cost)

        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.sb": "yes",
        }
        self.opti.solver("ipopt", opts)

    def solve(
        self,
        position,
        velocity,
        tendon_lengths,
        goal,
        previous_target_velocity=None,
    ):
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        tendon_lengths = np.asarray(tendon_lengths, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if previous_target_velocity is None:
            previous_target_velocity = np.zeros(3)

        previous_target_velocity = np.asarray(
            previous_target_velocity,
            dtype=np.float64,
        )

        self.opti.set_value(self.p0, position)
        self.opti.set_value(self.v0, velocity)
        self.opti.set_value(self.goal, goal)
        self.opti.set_value(self.tendon, tendon_lengths)
        self.opti.set_value(self.u_previous, previous_target_velocity)

        # Initial state guess
        state_guess = np.hstack([position, velocity])[:, None]
        self.opti.set_initial(
            self.X,
            np.tile(state_guess, (1, self.horizon + 1)),
        )

        # Initial control guess
        self.opti.set_initial(
            self.U,
            np.tile(
                previous_target_velocity[:, None],
                (1, self.horizon),
            ),
        )

        try:
            solution = self.opti.solve()
            target_velocity = solution.value(self.U[:, 0])
            self.last_solution = solution
        except RuntimeError:
            target_velocity = previous_target_velocity

        return np.asarray(target_velocity, dtype=np.float64)

    def reset(self):
        self.last_solution = None