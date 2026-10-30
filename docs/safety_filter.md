# Predictive safety filter

Thesis: Chapter 5. Two modules:

| Module | Content |
|---|---|
| `envs/predictive_safety_filter.py` | `PredictiveSafetyFilter` (certification + intervention), `SafetyConstraints`, `hook_obstacle_contact`, `tilt_angle` |
| `envs/adaptive_hook_director_safety_filter.py` | `AdaptiveTransportDirectorAviarySafetyFilter`: the director env that runs the filter on every command, plus the stall takeover |

The filter uses the ResNet model and the helpers of `resnet_mpc.py`
(`rollout`, `step_jacobians`, `sensitivity`), but solves its own QPs instead
of calling `ResNetMPC`.

## 1. `PredictiveSafetyFilter`: construction

```python
PredictiveSafetyFilter(models, horizon=36, constraints=None, u_max=1.0, w0=10.0,
                       w1=0.1, w_du=0.5, sqp_iterations=2, dt=1/48, qp_solver="daqp")
```

The environment builds it with `horizon=20` (`FILTER_HORIZON`). The
constructor prepares everything that does not change between steps:

```mermaid
flowchart TB
    I["__init__"] --> J["models (requires_grad off),<br/>_jac[mode] = step_jacobians(model)"]
    I --> D["_D: difference operator of the command changes"]
    I --> L["_rows[mode] = _constraint_layout(mode)"]
    L --> L1["per step: state index, soft bounds, hard bound,<br/>group of every constrained entry<br/>(tilt ×2, altitude, speed ×3, swing ×2 in mode 1)<br/>hard = rows with the 60° bound, E = row → slack of its group"]
    I --> Q["per mode two DAQP QPs (CasADi conic):<br/>_cert_qp (variables U)<br/>_safe_qp (variables U and one slack per group)"]
    I --> H["constant Hessians _H_cert, _H_safe,<br/>tracking weights _track (w0 on u_0, w1 on the rest),<br/>linear slack weights _slack_lin"]
```

`SafetyConstraints` (dataclass) holds the limits of Table 5.2 and the
contact-escape parameters. `GROUPS`, `SLACK` and `TOL` (class constants)
give the four constraint groups, their slack weights (the priorities) and
the tolerance of the nonlinear check.

## 2. `filter(mode, x0, u_rl, u_prev, contact)`

One call per control step. Returns the applied command and an info dict
(also stored in `self.last`).

```mermaid
flowchart TB
    S["filter(mode, x0, u_rl, u_prev, contact)"] --> B["clip u_rl to ±u_max<br/>lo, hi = _bounds(mode, x0)<br/>(relaxed where x0 already violates a bound)"]
    B --> E{"_escape(x0, contact)<br/>escaping?"}
    E -- yes --> EB["first-command box: climb<br/>(u_x = u_y = 0, u_z ≥ 0.5)"]
    E -- no --> NB["first-command box: ±u_max"]
    EB & NB --> NOM["U_nom = [u_rl, previous plan from column 2, last column repeated]<br/>X_nom = rollout(mode, x0, U_nom)"]
    NOM --> C0{"not escaping and<br/>X_nom within TOL?"}
    C0 -- yes --> OK["certified: apply u_rl, reason none"]
    C0 -- "no (not escaping)" --> CQ["certification, up to sqp_iterations times:<br/>y, G = _linearize(X_c, U_c)<br/>_cert_qp with u_0 = u_rl fixed, hard soft-bounds"]
    CQ -- "QP infeasible" --> INT
    CQ -- "solved" --> CR{"rollout of the solution<br/>within TOL?"}
    CR -- yes --> OK
    CR -- no, iterations left --> CQ
    CR -- no, done --> INT
    C0 -- "escaping" --> INT["intervention SQP, up to sqp_iterations times:<br/>y, G = _linearize(X_s, U_s)<br/>_safe_qp with slacks + hard 60° tilt rows<br/>(if infeasible: again without the hard rows)<br/>line search 1 / 0.5 / 0.25 on the nonlinear _merit"]
    INT -- solved --> AP["apply U[:, 0]<br/>reason constraint / contact"]
    INT -- "every QP failed" --> HOLD["hold u_prev (clipped to the box)"]
    OK & AP & HOLD --> P["_plan = U (next nominal continuation)<br/>self.last = {intervened, reason, u_rl, u_applied,<br/>slack, hard_tilt_kept, X_pred, U_plan, solve_time}"]
```

The line-by-line description of both QPs is in
[ocp_solvers.md](ocp_solvers.md#5-safety-filter-predictivesafetyfilterfilter-293426-sor).

Helpers:

| Method | What it does |
|---|---|
| `rollout(mode, x0, U)` | nonlinear prediction, X (NX, N+1) |
| `_bounds(mode, x0, hard)` | soft (or hard) bounds per step; a bound that x0 already violates is relaxed to the current value and shrinks by `contraction` per step |
| `_linearize(mode, X, U)` | constrained outputs y and G = dy/dU: the constrained rows of `resnet_mpc.sensitivity` |
| `_violation(mode, X, lo, hi)` | largest violation per group, compared with `TOL` |
| `_merit(mode, U, X, ...)` | the line-search merit of the intervention: the QP cost with the slacks replaced by the violations of the nonlinear prediction, plus a penalty beyond the hard tilt (the counterpart of the MPC's line-search cost) |
| `_escape(x0, contact)` | the contact-escape state machine: starts on contact with tilt > 10°, ends after 0.5 s or when the contact is gone and the tilt is below 8° |

Module functions:

| Function | What it does |
|---|---|
| `hook_obstacle_contact(env, attached)` | the hook touches the connectors or the holder plate, or the cylinder before the payload is lifted (the lever situation of Section 5.1.4) |
| `tilt_angle(x)` | the angle between the body z axis and the vertical from `roll_w`, `pitch_w` |

## 3. The environment: `AdaptiveTransportDirectorAviarySafetyFilter`

A subclass of the director env. The RL director acts on it exactly as on
`AdaptiveTransportDirectorAviary`; the filter is invisible to the policy.

```python
AdaptiveTransportDirectorAviarySafetyFilter(model_file=None, constraints=None,
    filter_horizon=20, stall_distance=0.1, stall_time=1.0,
    waypoint_tolerance=0.25, takeover_time=0.5, takeover_speed=0.4, ...)
```

It owns:

* `safety_filter`: a `PredictiveSafetyFilter` (horizon 20);
* `takeover_mpc`: a `ResNetMPC` (horizon 20, default weights, 5 SQP
  iterations) for the stall takeover;
* the mode tracking (`_attached`, `_rest_z`), the previous applied command
  `_u_prev`, the position history of the stall detection and the
  intervention counters (`reason_count`, `intervention_count`).

### Stall takeover (Section 5.1.4)

`_update_takeover()` keeps the positions of the last `stall_time`. If the
drone moved less than `stall_distance` in that time while it is farther than
`waypoint_tolerance` from the active waypoint, a takeover starts: a
`Segment` from the current position to the waypoint with `takeover_speed`.
During the takeover `_takeover_command(mode, x)` solves the `ResNetMPC` on
that segment, and its command replaces the RL proposal **before** the filter
(so it is filtered too). The takeover ends after `takeover_time` or when the
env advances the waypoint.

## 4. One environment step

```mermaid
sequenceDiagram
    participant P as RL director policy
    participant S as SafetyFilter env
    participant F as PredictiveSafetyFilter
    participant M as takeover ResNetMPC
    participant D as AdaptiveTransportDirectorAviary.step
    P->>S: step(action)
    S->>S: x = measure_state(env, attached), mode = attached<br/>u_rl = clip(action[0:3]), contact = hook_obstacle_contact
    S->>S: _update_takeover()
    alt takeover active
        S->>M: solve on the Segment reference
        M-->>S: proposal = U[:, 0]
    else
        S->>S: proposal = u_rl
    end
    S->>F: filter(mode, x, proposal, u_prev, contact)
    F-->>S: u, info (reason)
    Note over S: reason = "stall" during a takeover<br/>(unless it is "contact")
    S->>D: super().step(action with action[0:3] = u)
    D-->>S: obs, reward, terminated, truncated, info
    S->>S: attached = payload_attached(...)<br/>count the reason
    S-->>P: obs, ..., info + safety_intervened, safety_reason,<br/>u_rl, u_applied, safety_slack, hook_contact, payload_attached
```

The tendon commands of the RL action pass unchanged. The reason strings are
`none`, `constraint`, `stall` and `contact`. `REASONS` still lists `rate` (a
former command-change limit that was removed) so that the reason indices of
`utilities/analyse.py` stay the same.
