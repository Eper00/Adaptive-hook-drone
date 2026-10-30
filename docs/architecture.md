# Architecture

## 1. Layers

The code has four layers. Every layer only uses the ones below it.

```mermaid
flowchart TB
    subgraph L4["4 · Scripts (utilities/)"]
        learn["learn.py<br/>training"]
        play["play.py<br/>viewer"]
        analyse["analyse.py<br/>single / Monte Carlo tests"]
        compare["compare_methods.py<br/>500-scenario comparison"]
        figs["test_*, plot_*, render_*<br/>thesis figures"]
    end
    subgraph L3["3 · Controllers"]
        ppo["PPO policies<br/>(Stable-Baselines3 .zip files)"]
        wmpc["WaypointMPCAgent<br/>waypoint_mpc.py"]
        psf["PredictiveSafetyFilter<br/>predictive_safety_filter.py"]
        rmpc["ResNetMPC + ResNetDynamics<br/>resnet_mpc.py"]
        shared["director_mpc.py<br/>state, measurements, grasp geometry"]
    end
    subgraph L2["2 · Task environments (Gymnasium)"]
        vel["AdaptiveVelocityAviary"]
        tr["AdaptiveTransportAviary"]
        dir["AdaptiveTransportDirectorAviary"]
        dsf["AdaptiveTransportDirectorAviarySafetyFilter"]
    end
    subgraph L1["1 · Simulation"]
        base["BaseAviary<br/>MuJoCo model, rotor physics"]
        mj["MuJoCo"]
    end
    L4 --> L3
    L4 --> L2
    L3 --> L2
    L2 --> L1
    base --> mj
```

The controllers in layer 3 are not Gymnasium agents of the same kind:

* the PPO policies are just `model.predict(obs)` calls on the observation of
  an environment;
* the MPC agent reads the MuJoCo state of the environment directly
  (`measure_state`) and returns an action;
* the safety filter is built *into* an environment
  (`AdaptiveTransportDirectorAviarySafetyFilter.step` calls it), so the RL
  director does not know that it is filtered.

## 2. Class hierarchy of the environments

```mermaid
classDiagram
    direction TB
    class Env["gymnasium.Env"]
    class BaseAviary {
        +reset(seed, options)
        +step(action)
        +render()
        #_physics(rpm, i)
        #_preprocessAction(action)
        #_getDroneStateVector(i)
        #_computeObs() / _computeReward() / _computeTerminated() / _computeTruncated() / _computeInfo()
    }
    class AdaptiveVelocityAviary {
        TARGET_VEL
        PAYLOAD_INDICATOR
        +reset()  draws target velocity, yaw, payload
        +step(action)  sets the tendons itself
        -_hang_payload_in_hook(side)
    }
    class AdaptiveTransportAviary {
        WAYPOINTS, current_waypoint_idx
        GRAB_FLAG, GRAB_FLAG_ENABLE, PAYLOAD_TERMINATION
        +reset()  draws a scenario
        +set_scenario(payload, goal, mass, radius, yaw)
        +step(action)  tendon logic
        -_update_grab_flag()
        -_advance_waypoint(i)
        +comulative_segment_distance()
    }
    class AdaptiveTransportDirectorAviary {
        controller_model  (PPO velocity policy)
        alpha, prev_action  (low-pass filter)
        +step(action)  filter -> velocity policy -> super().step
        -_get_low_level_obs(v, yaw)
    }
    class AdaptiveTransportDirectorAviarySafetyFilter {
        safety_filter  (PredictiveSafetyFilter)
        takeover_mpc  (ResNetMPC)
        +step(action)  filter / takeover -> super().step
        -_update_takeover()
        -_takeover_command(mode, x)
    }
    Env <|-- BaseAviary
    BaseAviary <|-- AdaptiveVelocityAviary
    BaseAviary <|-- AdaptiveTransportAviary
    AdaptiveTransportAviary <|-- AdaptiveTransportDirectorAviary
    AdaptiveTransportDirectorAviary <|-- AdaptiveTransportDirectorAviarySafetyFilter
```

Every subclass only overrides what changes; a `step` call walks up the chain:
safety-filter env → director env → transport env → `BaseAviary.step`. See
[simulation.md](simulation.md) for the details of each.

## 3. Classes of the controllers

```mermaid
classDiagram
    direction LR
    class ResNetDynamics["ResNetDynamics (torch.nn.Module)"] {
        dt, alpha, with_swing
        mu, sd, out_scale  (buffers)
        +network(z)  increments
        +forward(x, u)  next state
        +rollout(x0, U)  numpy interface
    }
    class ResNetMPC {
        models[2], jacobians[2]
        N, w (DirectorWeights), u_max, max_tilt
        qp, qp_box  (DAQP)
        +reset()
        +rollout(mode, x0, U)
        +solve(mode, x0, p_ref, v_ref, u_prev)
        -_qp_step(H, g, Gamma, X, U)
    }
    class WaypointMPCAgent {
        env, models, weights[2]
        mpc (ResNetMPC)
        target (WaypointTarget)
        +reset()
        +action()  env action
    }
    class PredictiveSafetyFilter {
        models[2], _jac[2]
        c (SafetyConstraints)
        _cert_qp[2], _safe_qp[2]  (DAQP)
        +reset()
        +rollout(mode, x0, U)
        +filter(mode, x0, u_rl, u_prev, contact)
        -_constraint_layout(mode)
        -_bounds(mode, x0, hard)
        -_linearize(mode, X, U)
        -_violation(mode, X, lo, hi)
        -_escape(x0, contact)
    }
    class WaypointTarget {
        hook
        +reset()
        +__call__(env, attached)  (target, waypoint index)
    }
    class Segment {
        +__call__(t)  (position, velocity)
    }
    class DirectorWeights {
        <<dataclass>>
        pos, vel, swing, swing_rate, u, du, terminal
    }
    class SafetyConstraints {
        <<dataclass>>
        max_tilt, hard_max_tilt, min_altitude, max_speed, max_swing
        contraction, contact_tilt, escape_*
    }
    WaypointMPCAgent *-- ResNetMPC
    WaypointMPCAgent *-- WaypointTarget
    WaypointMPCAgent ..> DirectorWeights
    ResNetMPC o-- ResNetDynamics : 2 (one per mode)
    ResNetMPC ..> DirectorWeights
    PredictiveSafetyFilter o-- ResNetDynamics : 2 (one per mode)
    PredictiveSafetyFilter *-- SafetyConstraints
```

`ResNetMPC` and `PredictiveSafetyFilter` share the same two `ResNetDynamics`
objects (loaded once with `load_or_train`) and the same helper functions of
`resnet_mpc.py`: `rollout`, `step_jacobians`, `sensitivity`.

## 4. Module dependencies

Arrows point from the importing module to the imported one. Dashed arrows are
imports inside functions (to avoid import cycles or slow imports).

```mermaid
flowchart LR
    base[base_aviary]
    vel[adaptive_hook_velocity]
    tr[adaptive_hook_transport]
    dirv[adaptive_hook_director_velocity]
    dsf[adaptive_hook_director_safety_filter]
    dm[director_mpc]
    rm[resnet_mpc]
    wm[waypoint_mpc]
    psf[predictive_safety_filter]
    cur[wrappers.curriculum]
    sb3[(stable_baselines3)]
    torch[(torch)]
    casadi[(casadi + DAQP)]
    mj[(mujoco)]

    base --> mj
    vel --> base
    tr --> base
    tr --> dm
    dirv --> tr
    dirv --> sb3
    dm --> base
    dm -.-> dirv
    rm --> dm
    rm --> torch
    rm --> casadi
    rm -.-> sb3
    wm --> dm
    wm --> rm
    psf --> dm
    psf --> rm
    psf --> casadi
    dsf --> dirv
    dsf --> dm
    dsf --> psf
    dsf --> rm
    cur --> sb3

    learn["utilities/learn.py"] --> tr & vel & dirv & cur
    play["utilities/play.py"] -.-> tr & vel & dirv & dsf & wm
    analyse["utilities/analyse.py"] --> dm
    analyse -.-> vel & dirv & dsf & wm & psf
    compare["utilities/compare_methods.py"] -.-> analyse & dsf & psf
```

Notes:

* `adaptive_hook_transport` imports `grasp_pose` and `APPROACH_HEIGHT` from
  `director_mpc`, so the waypoints of the environment and the grasp target of
  the MPC come from the same function.
* `director_mpc.make_identification_env` imports the director env inside the
  function: `director_mpc` is imported by the transport env, a module-level
  import would be circular.

## 5. The three controllers in one control step

All three run on the same task and the same low-level velocity policy, at
48 Hz (one environment step = 5 MuJoCo steps at 240 Hz).

```mermaid
flowchart TB
    subgraph RL["RL director"]
        o1["obs (35)"] --> p1["PPO director<br/>model.predict"] --> a1["action [v_ref, yaw, tendon, tendon]"]
    end
    subgraph MPC["MPC director"]
        s2["measure_state(env)"] --> w2["WaypointTarget"] --> m2["ResNetMPC.solve"] --> a2["action [u, 0, 1, 0]"]
    end
    subgraph SF["RL director + safety filter"]
        o3["obs (35)"] --> p3["PPO director"] --> f3["PredictiveSafetyFilter.filter<br/>(inside the env step)"] --> a3["action with u_applied"]
    end
    a1 & a2 & a3 --> D["AdaptiveTransportDirectorAviary.step<br/>low-pass filter, PPO velocity policy"]
    D --> T["AdaptiveTransportAviary.step<br/>tendon logic"]
    T --> B["BaseAviary.step<br/>rotor physics, 5 × mj_step"]
    B --> R["obs, reward, terminated, truncated, info<br/>(reward advances the waypoint)"]
```

Detailed sequence diagrams: RL in
[reinforcement_learning.md](reinforcement_learning.md#4-one-step-of-the-rl-director),
MPC in [mpc.md](mpc.md#5-one-control-step-of-the-mpc-director), safety filter
in [safety_filter.md](safety_filter.md#4-one-environment-step).

## 6. Artifacts: what is produced where and used by whom

```mermaid
flowchart LR
    learnV["learn.py<br/>adaptive_velocity"] --> velzip[("results/final/<br/>rl_adaptive_velocity_curriculum/<br/>final_model.zip")]
    velzip --> dirEnv["AdaptiveTransportDirectorAviary<br/>(controller_path)"]
    dirEnv --> learnD["learn.py<br/>adaptive_director"] --> dirzip[("results/final/<br/>rl_adaptive_director_curriculum/<br/>final_model.zip")]
    learnT["learn.py<br/>adaptive_transport"] --> trzip[("results/final/<br/>rl_adaptive_transport[_curriculum]")]
    dirzip --> collect["resnet_mpc.collect_datasets<br/>(RL director flights)"]
    pctl["director_mpc.collect<br/>(P-controller flights)"] --> collect
    collect --> data[("results/resnet/<br/>resnet_training_data.pkl")]
    data --> train["resnet_mpc.train"] --> model[("results/resnet/<br/>resnet_model.pt")]
    model --> wmpc["WaypointMPCAgent"]
    model --> psf["PredictiveSafetyFilter<br/>+ takeover ResNetMPC"]
    dirzip --> cmp["compare_methods.py"]
    wmpc --> cmp
    psf --> cmp
    cmp --> eps[("results/analysis/methods/<br/>episodes.pkl, summary.json")]
    eps --> figs["plot_comparison_thesis.py,<br/>render_rescue_scene.py"] --> tf[("results/thesis_figures/")]
```

The order matters: the velocity policy has to exist before the director can
be trained, and the director policy before the ResNet data can be collected
(its flights are part of the training data). A retrained velocity policy
invalidates the director policy and the ResNet model, since both were
trained on the closed loop with the old one.
