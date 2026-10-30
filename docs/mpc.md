# MPC director

Thesis: Chapter 4. The MPC director replaces the RL director: it computes the
velocity command of the director env with an MPC on an identified ResNet
model of the closed loop "low-pass filter → velocity policy → drone".

Three modules:

| Module | Content |
|---|---|
| `envs/director_mpc.py` | shared building blocks, **no optimizer** despite the name: state layout, measurement of the state, grasp geometry, payload / contact detection, waypoint target, cost weights, P-controller data collection |
| `envs/resnet_mpc.py` | the model (`ResNetDynamics`) and its helpers, data collection and training, the MPC solver `ResNetMPC` |
| `envs/waypoint_mpc.py` | the agent `WaypointMPCAgent`: measure → choose mode, weights and target → solve → act |

```mermaid
flowchart LR
    dm["director_mpc.py<br/>NX, NU, slices P V C TH THD ATT TILT<br/>measure_state, payload_attached<br/>WaypointTarget, DirectorWeights, U_MAX<br/>grasp_pose, collect"]
    rm["resnet_mpc.py<br/>ResNetDynamics, rollout, step_jacobians,<br/>sensitivity, load_or_train, train<br/>ResNetMPC"]
    wm["waypoint_mpc.py<br/>WaypointMPCAgent"]
    psf["predictive_safety_filter.py"]
    dsf["adaptive_hook_director_safety_filter.py<br/>(stall takeover)"]
    tr["adaptive_hook_transport.py<br/>(waypoints)"]
    dm --> rm --> wm
    dm --> wm
    rm --> psf
    rm --> dsf
    dm --> tr
```

## 1. State and measurements (`director_mpc.py`)

The model state (19 values, Section 4.1.1) and its slices:

| Slice | Content | Source (`measure_state(env, attached)`) |
|---|---|---|
| `P` 0:3 | position | `env.pos[0]` |
| `V` 3:6 | velocity | `env.vel[0]` |
| `C` 6:9 | filtered command (output of the low-pass filter) | `filter_state(env)` = `env.prev_action[0:3]` |
| `TH` 9:11 | payload swing angles (world x–z, y–z planes) | `measure_swing(env)`, zero without the payload |
| `THD` 11:13 | swing rates | `measure_swing(env)` |
| `ATT` 13:16 | tilt `roll_w`, `pitch_w` (world frame) and yaw | `attitude(env)` |
| `ATTD` 16:19 | their rates | `attitude(env)` |
| `TILT` 13:15 | the constrained part of `ATT` | |

Helpers used by the MPC, the filter, the environments and the analysis:

| Function / class | What it does | Used by |
|---|---|---|
| `payload_attached(env, rest_z, attached)` | the payload is on the hook: lifted 2 cm above its rest height and touching the hook; stays true until it is 0.7 m away. **Selects the mode** (0 / 1) | MPC agent, filter env, analysis, data collection |
| `hook_payload_contact(env)` | any hook–payload contact | data collection (contact steps are left out of the training data) |
| `grasp_pose(center, radius)` | the drone position for the grasp (+y side of the cylinder, high enough for the hook tip) | transport env waypoints |
| `WaypointTarget` | target position of the MPC: the active waypoint of the env; near the grasp waypoint (`GRASP_SNAP` = 3 cm) a target moved by the measured hook–payload offset, frozen once | MPC agent, P-controller |
| `Segment` | straight line with a trapezoidal velocity profile | stall takeover of the filter env |
| `DirectorWeights` | cost weights: position per axis, velocity, swing, swing rate, command, command change, terminal factor | `ResNetMPC` |
| `U_MAX` = 0.8 | command bound of the MPC | `ResNetMPC`, data collection |
| `director_step(env, command, tendon)` | one director step with direct tendon control (bypasses the transport task's tendon logic) | identification experiments |

## 2. The model (`resnet_mpc.py`)

```mermaid
classDiagram
    class ResNetDynamics {
        dt = 1/48, alpha = 0.1, with_swing (mode 1)
        mu, sd  input normalization (buffers)
        out_scale  output scale (buffer)
        inp: Linear(16+3 → 128)
        blocks: 2 × [Linear, Tanh, Linear] (residual)
        out: Linear(128 → 11), skip: Linear(19 → 11)
        +network(z)  the 11 increments
        +forward(x, u)  next state (batched)
        +rollout(x0, U)  numpy (NX,) (NU,N) → (NX,N+1)
    }
```

* One `ResNetDynamics` per mode; `load_model(path)` returns `[mode 0, mode 1]`
  from `results/resnet/resnet_model.pt`, `load_or_train` trains them first if
  the file is missing.
* `forward(x, u)`: the network gets the state without the position and the
  command; the known parts of the update (position integration, the
  low-pass filter, the swing angle from its rate) are written out, the
  network only predicts what depends on the closed loop (Section 4.2.1).
  All computations are in float64.
* Module-level helpers, shared by `ResNetMPC` and `PredictiveSafetyFilter`:

| Function | Returns |
|---|---|
| `rollout(model, x0, U)` | the states x_1 … x_N for the commands U (batched, PyTorch) |
| `step_jacobians(model)` | a function (X, U) → (A, B): the Jacobians of every step in one call (`torch.func.vmap(jacrev(...))`) |
| `sensitivity(jacobians, X, U)` | Γ = d(x_1 … x_N)/dU, built from A, B by a recursion |
| `sqp(U, X, merit, qp_step, rollout, iterations, clip)` | the SQP loop (QP step + backtracking line search on the nonlinear merit) shared by `ResNetMPC` and `PredictiveSafetyFilter` |

## 3. Training data and training

```mermaid
flowchart TB
    T["python -m multi_drone_mujoco.envs.resnet_mpc [--collect]"] --> TR["train(path, epochs=150, horizon=24)"]
    TR --> C{"cached data<br/>(results/resnet/resnet_training_data.pkl)<br/>and no --collect?"}
    C -- yes --> LD[load train / val runs]
    C -- no --> CD["collect_datasets()"]
    CD --> JOBS["collect_runs(jobs, workers=8)<br/>ProcessPoolExecutor, 'spawn'"]
    JOBS --> CP["_collect_p(seed, excite)<br/>director_mpc.collect: P-controller on the<br/>env waypoints + random command noise<br/>16 training, 1 validation runs"]
    JOBS --> CR["_collect_rl(seed)<br/>RL director policy, final task, 15 s<br/>144 training, 4 validation runs"]
    CP & CR --> SAVE["save the runs to the cache"]
    SAVE --> LD
    LD --> M{"for mode 0, 1"}
    M --> W["_windows(runs, mode, H)<br/>windows entirely in one mode and valid<br/>(no hook-payload contact before the lift,<br/>tilt < 60°)"]
    W --> N["normalization: mu, sd of the inputs,<br/>out_scale of the one-step increments"]
    N --> LOOP["epochs: rollout loss over H steps,<br/>H grows 2 → 24 over the first 40 %<br/>Adam + cosine annealing, gradient clipping"]
    LOOP --> BEST["keep the weights with the best<br/>validation loss at the full horizon"]
    BEST --> S["save_model → results/resnet/resnet_model.pt"]
```

Every run is a dict of arrays: `x` (states), `u` (commands), `attached`,
`valid`, `contact`, plus `source` ("P" / "RL") and `seed`. A sample is the
pair (x_k, u_k) → x_k+1. The seeds of the training and validation runs differ
from the ones of the test scripts (`utilities/test_resnet.py`,
`resnet_error_scale.py`).

## 4. The solver: `ResNetMPC`

Thesis: Section 4.1.3. Construction:

```python
ResNetMPC(models, horizon=40, weights=None, u_max=U_MAX, max_tilt=pi/2,
          sqp_iterations=5, qp_solver="daqp", max_threads=4)
```

It keeps the two models, their step-Jacobian functions, the difference
operator of the command changes and two DAQP QP objects (CasADi `conic`):
`qp` with the linearized tilt rows, `qp_box` with the command bounds only.
`self._plan` holds the previous plan for the warm start; `reset()` clears it.

`solve(mode, x0, p_ref, v_ref, u_prev)` → `(X, U, info)`:

```mermaid
flowchart TB
    S["solve(mode, x0, p_ref, v_ref, u_prev)"] --> REF["reference of x_1..x_N, square roots of the weights<br/>(position, velocity, swing in mode 1, terminal factor)"]
    REF --> WS["U = previous plan shifted by one step<br/>(zeros at the first call)"]
    WS --> RO["X = rollout(mode, x0, U); J = cost(X, U)"]
    RO --> IT{"SQP iteration<br/>(sqp_iterations times)"}
    IT --> G["Gamma = sensitivity(jacobians[mode], X, U)"]
    G --> JR["J_r = [sqrt(Q)·Gamma; sqrt(w_u)·I; sqrt(w_du)·D]"]
    JR --> QP["_qp_step(J_rᵀJ_r, J_rᵀr, Gamma, X, U)"]
    QP --> Q1{"qp with the tilt rows<br/>solved?"}
    Q1 -- no --> Q2{"qp_box solved?"}
    Q2 -- no --> STOP[stop]
    Q1 -- yes --> LS
    Q2 -- yes --> LS["line search: step 1, 0.5, 0.25<br/>U_try = clip(U + step·dU), rollout, cost<br/>(cost = least squares + 1e4 · tilt excess)"]
    LS -- "cost decreased" --> IT
    LS -- "no decrease" --> STOP
    IT -- done --> OUT["_plan = U<br/>return X (NX, N+1), U (NU, N), {solve_time}"]
    STOP --> OUT
```

Two users with different settings:

| User | Horizon | Weights | Tilt bound | SQP iterations | Reference |
|---|---|---|---|---|---|
| `WaypointMPCAgent` (MPC director) | 20 | `WAYPOINT_WEIGHTS` / `WAYPOINT_WEIGHTS_PAYLOAD` per mode | 15° | 2 | the active waypoint, constant over the horizon, zero velocity |
| stall takeover of the filter env | 20 | `DirectorWeights()` defaults | 90° (none) | 5 | a `Segment` from the current position to the waypoint |

The line-by-line description of `solve` and `_qp_step` is in
[ocp_solvers.md](ocp_solvers.md#4-mpc-resnetmpcsolve-574631-sor).

## 5. One control step of the MPC director

`WaypointMPCAgent(env, model_file, horizon=20)` loads the models
(`load_or_train`), builds one `ResNetMPC` and keeps a `WaypointTarget`.
`reset()` resets both and reads the rest height of the payload.

```mermaid
sequenceDiagram
    participant M as MPCDirector (analyse.py)
    participant A as WaypointMPCAgent
    participant DM as director_mpc
    participant R as ResNetMPC
    participant E as AdaptiveTransportDirectorAviary
    M->>A: action()
    A->>DM: payload_attached(env, rest_z, attached) → mode 0 / 1
    A->>DM: measure_state(env, attached) → x (19)
    A->>DM: WaypointTarget(env, attached) → target w
    A->>A: p_ref = w over the horizon, v_ref = 0<br/>mpc.w = weights[mode]
    A->>R: solve(mode, x, p_ref, v_ref, u_prev)
    R-->>A: X, U, info
    A-->>M: [U[:, 0], 0, 1, 0]  (velocity, heading 0, tendon 1)
    M->>E: step(action)
    Note over E: low-pass filter, velocity policy,<br/>tendon logic of the env (closes the hook<br/>only near the payload at the grasp waypoint)
```

The tendon command is always 1 (fully closed): when and in which direction
the hook closes is decided by the environment, exactly as for the RL
director (see [simulation.md](simulation.md#23-stepaction-the-tendon-logic)).
