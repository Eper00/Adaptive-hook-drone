# Adaptive hook drone: code documentation

This folder describes **how the code works**: which module does what, which
class calls which method, how data and trained models flow between the
scripts, and how one control step runs for each of the three controllers.
The mathematics (dynamics, MDP, optimal control problems, ResNet structure)
is in the thesis; these pages refer to its sections instead of repeating the
formulas.

The diagrams are Mermaid diagrams. They render on GitHub and in VS Code
(Markdown preview with Mermaid support).

## Pages

| Page | Content |
|---|---|
| [architecture.md](architecture.md) | Big picture: layers, class hierarchy, module dependencies, the three controllers side by side, artifacts (trained models, data) and where they come from |
| [simulation.md](simulation.md) | `BaseAviary`, the hook and the payload in MuJoCo, the transport task (`AdaptiveTransportAviary`), the velocity task (`AdaptiveVelocityAviary`), the director interface (`AdaptiveTransportDirectorAviary`) |
| [reinforcement_learning.md](reinforcement_learning.md) | Training with PPO (`utilities/learn.py`), curriculum learning (`CurriculumWrapper`), the three policies, playing them back (`utilities/play.py`) |
| [mpc.md](mpc.md) | The MPC director: shared building blocks (`director_mpc.py`), the ResNet model, its data and training, the MPC solver (`resnet_mpc.py`) and the agent (`waypoint_mpc.py`) |
| [ocp_solvers.md](ocp_solvers.md) | (in Hungarian) Line by line, with measured linearization errors: how `ResNetMPC` and `PredictiveSafetyFilter` solve their optimal control problems (linearization, QPs, line search, fallbacks), and where the code is more detailed than the thesis |
| [safety_filter.md](safety_filter.md) | The predictive safety filter (`predictive_safety_filter.py`) and the environment that applies it (`adaptive_hook_director_safety_filter.py`) |
| [evaluation.md](evaluation.md) | Analysis and comparison (`utilities/analyse.py`, `utilities/compare_methods.py`), the scripts of the thesis figures, the `results/` folder |

## Names in the code and in the thesis

The code was written before the thesis, so a few things have different names:

| Code | Thesis |
|---|---|
| velocity controller, `rl_adaptive_velocity_curriculum`, `AdaptiveVelocityAviary` | low-level policy $\pi^{\mathrm{L}}$ (Section 3.3.2–3.3.5) |
| RL director, `rl_adaptive_director_curriculum`, `AdaptiveTransportDirectorAviary` | high-level policy $\pi^{\mathrm{H}}$ (Section 3.3.6–3.3.7) |
| transport policy, `rl_adaptive_transport[_curriculum]`, `AdaptiveTransportAviary` | flat agent (Section 3.1.5, 3.2) |
| MPC director, `WaypointMPCAgent`, `ResNetMPC` | MPC (Chapter 4) |
| `ResNetDynamics` | identified model $F_\kappa$ / network $\mathcal{N}_\kappa$ (Section 4.2) |
| RL director + safety filter, `PredictiveSafetyFilter` | predictive safety filter $\pi^{\mathrm{S}}$ (Chapter 5) |
| mode 0 / mode 1 | $\kappa = 0$ (no payload) / $\kappa = 1$ (payload attached) |
| command `u`, action `[0:3]` | reference velocity $v_{\mathrm{ref}}$ / input $u_k$ |
| filtered command `c`, `env.prev_action[0:3]` | filtered reference $\bar v_{\mathrm{ref}}$, low-pass filter (3.20) |
| `GRAB_FLAG` | "the payload has come within $R_{\text{payload}} + 0.02$ m of the second segment" (Section 3.1.2) |
| `PAYLOAD_TERMINATION` | final task $\mathcal{M}_4$ (4 waypoints, payload-loss termination) |
| tilt `roll_w`, `pitch_w` | $\tilde\phi$, $\tilde\theta$ (Section 4.1.1) |

## Repository map

```
multi_drone_mujoco/
  envs/
    base_aviary.py                          MuJoCo scene generation, physics, Gymnasium API
    adaptive_hook_transport.py              transport task (rotor commands)          -> flat agent
    adaptive_hook_velocity.py               velocity tracking task (rotor commands)  -> low-level policy
    adaptive_hook_director_velocity.py      transport task through velocity commands -> high-level policy / MPC
    adaptive_hook_director_safety_filter.py the director env + safety filter + stall takeover
    director_mpc.py                         shared parts: state layout, measurements, grasp geometry, data collection
    resnet_mpc.py                           ResNet model, its data and training, ResNetMPC solver
    waypoint_mpc.py                         WaypointMPCAgent (the MPC director)
    predictive_safety_filter.py             PredictiveSafetyFilter
  wrappers/curriculum.py                    CurriculumWrapper, CurriculumCallback
  wrappers/__init__.py, wrappers/wind.py    domain randomization, wind (from the upstream simulator, unused here)
  control/, utils/                          PID controllers, enums, logger (upstream, mostly unused here)
  assets/                                   meshes (Crazyflie, Bumblebee)
utilities/
  learn.py, train_seeds.sh                  training
  play.py                                   viewer playback
  analyse.py, compare_methods.py            evaluation
  test_*.py, plot_*.py, render_*.py, resnet_*.py, make_scenario_movie.py   thesis figures and video
  replay_datasets.py                        watch the identification flights in the viewer
results/                                    trained models, data, analysis outputs (see evaluation.md)
  thesis_figures/                           every figure of the thesis
```

The repository is a fork of the MuJoCo-Drones-Gym simulator. Its parts that
this work does not use (other drone tasks, PID controllers, wind, domain
randomization, multi-agent wrappers) are not described here; the top-level
`README.md` still describes that simulator.

## Most used commands

```bash
# training (Chapter 3)
python utilities/learn.py --env_type adaptive_velocity  --timesteps 5000000  --curriculum_flag true
python utilities/learn.py --env_type adaptive_director  --timesteps 10000000 --curriculum_flag true
python utilities/learn.py --env_type adaptive_transport --timesteps 10000000 --curriculum_flag true

# ResNet model (Chapter 4)
python -m multi_drone_mujoco.envs.resnet_mpc              # train on the cached data
python -m multi_drone_mujoco.envs.resnet_mpc --collect    # collect new data, then train

# watch a controller
python -m utilities.play --env_type adaptive_director_MPC
python -m utilities.play --env_type adaptive_director_safety \
    --model_path results/final/rl_adaptive_director_curriculum/final_model.zip

# evaluation (Chapter 6)
python -m utilities.compare_methods --episodes 500
python -m utilities.analyse --test scenario --seed 83 --max_time 25
```
