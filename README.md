# PEARL + CFM Range-Goal Benchmark

This branch extends the original PEARL implementation into a modern
meta-reinforcement-learning sandbox for comparing task-inference methods on
hidden-context environments.

The main additions are:

- a 2026-compatible runtime based on Python 3.11, Gymnasium, PyTorch, and the
  DeepMind `mujoco` package;
- a flow-matching / CFM context-inference path selected with `method: "flow"`;
- a VariBAD-style PPO + VAE adapter selected with `method: "varibad"`;
- a Range-Goal GridWorld benchmark for online belief refinement from ambiguous
  range observations;
- discrete-SAC support for Box-shaped PEARL actions, stochastic/argmax action
  execution, invalid-action masking, W&B logging, and smoke-test configs.

The original PEARL implementation is preserved as `method: "baseline"`.

## What Is In This Branch

### Methods

| Config value | Method | Main code |
| --- | --- | --- |
| `baseline` | Original PEARL with a Gaussian context encoder | `rlkit/torch/sac` |
| `flow` | PEARL control with CFM / flow-matching context inference | `rlkit/torch/flow` |
| `varibad` | PPO + VAE VariBAD loop running on PEARL task envs | `rlkit/torch/varibad` |

The launcher chooses the method from the experiment JSON:

```json
{
  "method": "flow"
}
```

### Range-Goal GridWorld

`RangeGoalGridWorld` is a hidden-goal 2D benchmark. The task context is a fixed
goal cell. The agent observes its own position, the map, and a coarse range bin
derived from the obstacle-aware shortest-path distance to the hidden goal. The
goal itself is not part of the observation.

This is designed to make one transition ambiguous: a single range bin usually
supports many possible goals, so useful behavior requires combining evidence
over time.

Registered PEARL env IDs:

- `range-goal-gridworld`
- `range-goal-gridworld-ambiguous`
- `range-goal-gridworld-diag`
- `range-goal-gridworld-main`

Direct Gymnasium inspection IDs are registered by importing
`range_goal_gridworld`:

- `RangeGoalGridWorld-LevelA-15x15-v0`
- `RangeGoalGridWorld-LevelA-21x21-v0`
- `RangeGoalGridWorld-LevelA-Ambiguous-15x15-v0`
- `RangeGoalGridWorld-LevelA-Main-RandomGoal-15x15-v0`
- `RangeGoalGridWorld-LevelA-PEARL-Diag-9x9-v0`

Useful debug helpers:

- `compute_likelihood_support()`
- `compute_oracle_posterior()`
- `render_debug(show_goal=True, show_oracle=True)`

See `RANGE_GOAL_GRIDWORLD_SPEC.md` for the environment specification.

### Modernized Continuous-Control Stack

The old PEARL environment file is pinned to Python 3.5, CUDA 10, Gym 0.12, and
`mujoco-py`. This branch uses:

- Python 3.11
- recent PyTorch
- Gymnasium instead of legacy Gym
- DeepMind `mujoco` instead of `mujoco-py`

The reward-varying MuJoCo tasks have been ported through a compatibility layer
in `rlkit/envs/mujoco_env.py`. The legacy Walker/Hopper random-parameter tasks
are not ported because they depend on `rand_param_envs` and MuJoCo 1.31.

See `MODERNIZATION.md` for the porting notes.

## Setup

Create an environment and install dependencies:

```bash
conda create -n pearl python=3.11 -y
conda activate pearl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU-only machines, install the PyTorch wheel that matches your platform
before running `pip install -r requirements.txt`.

## Quick Smoke Runs

From the repository root:

```bash
python launch_experiment.py configs/range-goal-gridworld-pearl-smoke.json
python launch_experiment.py configs/range-goal-gridworld-cfm-current-gauss-smoke.json
python launch_experiment.py configs/range-goal-gridworld-varibad-smoke.json
```

These configs are intentionally small. They are useful for checking imports,
task reset, data collection, encoder updates, logging, and termination behavior.

## Main Range-Goal Runs

PEARL baseline:

```bash
python launch_experiment.py configs/range-goal-gridworld-main-pearl.json
```

CFM / flow context inference:

```bash
python launch_experiment.py configs/range-goal-gridworld-main-cfm-current-gauss.json
```

Both main configs use:

- `env_name: "range-goal-gridworld-main"`
- 32 train tasks and 8 eval tasks
- 11x11 empty map curriculum
- full flattened map observation with action mask
- distance-delta reward shaping
- discrete SAC actor over four grid actions
- W&B logging enabled

For a lighter development pass, use:

```bash
python launch_experiment.py configs/range-goal-gridworld-main-pearl-v7-smoke.json
python launch_experiment.py configs/range-goal-gridworld-main-cfm-v7-smoke.json
```

## Other Experiment Examples

Continuous-control CFM examples:

```bash
python launch_experiment.py configs/cheetah-dir-cfm-current-gauss.json
python launch_experiment.py configs/sparse-point-robot-cfm-current-gauss.json
```

Original PEARL-style examples:

```bash
python launch_experiment.py configs/point-robot.json
python launch_experiment.py configs/cheetah-dir.json
python launch_experiment.py configs/ant-goal.json
```

Range-Goal diagnostics:

```bash
python launch_experiment.py configs/range-goal-gridworld-diag-pearl.json
python launch_experiment.py configs/range-goal-gridworld-diag-cfm-current-gauss.json
python scripts/visualize_range_goal_gridworld.py
```

## Config Guide

All experiments are JSON files under `configs/`. They are merged into
`configs/default.py`.

Important top-level keys:

- `env_name`: registered environment name in `rlkit/envs`;
- `method`: `baseline`, `flow`, or `varibad`;
- `n_train_tasks`, `n_eval_tasks`: task split;
- `latent_size`: latent context dimension;
- `env_params`: environment-specific parameters;
- `algo_params`: SAC / PEARL training parameters;
- `flow_params`: CFM encoder, decoder, prior-flow, and ODE settings;
- `varibad_params`: PPO, VAE, rollout, and encoder settings;
- `util_params`: GPU, debug, output, and W&B settings.

Useful Range-Goal knobs:

- `size`: grid size;
- `horizon`: max episode length;
- `map_family`: `empty`, `diag`, `four_room`, `u_shape`, `multi_corridor`,
  `wall_door`, `mixed`, or `ambiguous`;
- `bin_edges`: range-bin boundaries;
- `obs_mode`: for example `full_flat`, `full_flat_with_mask`, `map_with_mask`,
  or `pearl_lowdim`;
- `action_mode`: `hard_argmax` or `softmax_stochastic`;
- `mask_invalid_actions`: expose/action-mask invalid moves;
- `reward_mode`: `sparse`, `distance_delta`, or bin-progress variants.

## Logging And Outputs

Training writes to:

```text
output/<env-or-run-name>/<timestamp>/
```

Typical files include:

- `variant.json`: resolved experiment config;
- `progress.csv`: tabular learning metrics;
- model snapshots such as `policy.pth`, `context_encoder.pth`, and Q/V nets;
- optional evaluation trajectories when `dump_eval_paths` is enabled.

When `util_params.use_wandb` is true, `launch_experiment.py` forwards tabular
metrics to Weights & Biases and keeps legacy metric aliases such as
`env_steps` / `epoch` when requested by the config.

## Repository Map

```text
configs/                     Experiment JSON files
range_goal_gridworld/         Convenience import for Gymnasium inspection
rlkit/envs/                   PEARL environments and registration
rlkit/torch/sac/              Original PEARL/SAC implementation
rlkit/torch/flow/             CFM / flow-matching context inference
rlkit/torch/varibad/          VariBAD adapter, PPO, VAE, rollout storage
scripts/                      Debug and visualization utilities
MODERNIZATION.md              2026 porting notes
RANGE_GOAL_GRIDWORLD_SPEC.md  Range-Goal environment specification
```

## Known Limits

- Walker/Hopper random-parameter tasks are not ported to the modern MuJoCo stack.
- The upstream PEARL README notes that the original `ant-goal` reproduction was
  not fully resolved; this branch makes it runnable, but does not claim new
  benchmark numbers for that known upstream issue.
- Range-Goal GridWorld can expose the hidden goal in `info` for debugging, but
  training configs should keep `include_oracle_in_info: false`.
- `output/`, local logs, and W&B run artifacts are intentionally not part of the
  uploaded source branch.

## Original PEARL

This repository is based on:

> PEARL: Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic
> Context Variables, Kate Rakelly, Aurick Zhou, Deirdre Quillen, Chelsea Finn,
> and Sergey Levine.

Paper: <http://arxiv.org/abs/1903.08254>

Original implementation: PEARL built on top of
[`rlkit`](https://github.com/vitchyr/rlkit).

The original license is preserved in `LICENSE`.
