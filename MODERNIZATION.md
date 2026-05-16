# Modernization notes (2026)

This repo's original environment (`docker/environment.yml`) is pinned to 2019
builds — python 3.5.2, torch 1.0.1, CUDA 10, `gym` 0.12, `mujoco-py` 1.50, and a
now-defunct conda channel. It no longer installs on current systems, and torch
1.0.1 / CUDA 10 cannot run on recent (Blackwell-class) GPUs at all.

This document records the modernization done to make the reward-varying task
families train again on a current stack.

## New stack

- python 3.11, recent PyTorch (CUDA 12.8 wheel — required for Blackwell GPUs)
- **gymnasium** (Farama) instead of the unmaintained `gym`
- DeepMind **`mujoco`** package instead of `mujoco-py` — no MuJoCo license key,
  no separate MuJoCo200/131 binary install

See `requirements-modern.txt` for the exact set and install steps.

## Code changes

### gym → gymnasium
`import gym` / `from gym...` replaced with `gymnasium` in:
`rlkit/envs/{point_robot,wrappers,mujoco_env,half_cheetah,humanoid_dir}.py`,
`rlkit/data_management/env_replay_buffer.py`.

The repo's **internal env contract is kept** as the legacy 4-tuple
(`obs, reward, done, info`) and an obs-only `reset()` — the custom envs all
implement `step`/`reset` themselves, so no gymnasium 5-tuple migration of the
sampler/replay-buffer was needed. The MuJoCo base env (`mujoco_env.py`) converts
gymnasium's `(obs, info)` reset back to obs-only so the contract holds.

### mujoco-py → mujoco
`rlkit/envs/mujoco_env.py` was rewritten as a thin base over
`gymnasium.envs.mujoco.MujocoEnv` (modern `mujoco` bindings). It:
- resolves model XMLs from the repo's `assets/` or gymnasium's bundled assets,
- derives the observation space from the subclass's `_get_obs()` (old gym did
  this automatically; gymnasium does not),
- keeps `env.sim.data` / `env.sim.model` working via a compatibility shim, so
  env code written against `mujoco-py` is unchanged.

`half_cheetah.py` and `humanoid_dir.py` were re-pointed from `gym.envs.mujoco`
base classes to this base (they now also define `reset_model`). `ant.py` used
the base already. `np_random.randn` (Generator has no `randn`) → `standard_normal`.

`rlkit/envs/wrappers.py` `CameraWrapper` rewritten to render via gymnasium's
`MujocoRenderer` instead of `mujoco_py.MjRenderContextOffscreen`.

### Robustness
`rlkit/envs/__init__.py` now skips an env module that fails to import (with a
warning) instead of breaking the whole registry — so e.g. point-robot runs even
when the Walker/Hopper modules are unavailable.

## Verified

GPU training runs (≥1 iteration, `progress.csv` populating, returns improving):
`point-robot`, `sparse-point-robot`, `cheetah-dir`, `cheetah-vel`, `ant-dir`,
`ant-goal`, `humanoid-dir`.

`sim_policy.py` verified for evaluation and for video rendering — video needs a
GL backend: run with `MUJOCO_GL=egl` (headless GPU) or `MUJOCO_GL=osmesa` (CPU).

(`ant-goal` is the experiment the upstream README flags as not reproducing
correctly — it runs, but the numbers are a separate known issue.)

## Not ported: Walker / Hopper (`*_rand_params`)

The `walker_rand_params` / `hopper_rand_params` task families are **not** ported.
They differ from the reward-varying tasks: each task is a different set of MuJoCo
*model parameters*, implemented via the `rand_param_envs` git submodule, which
requires MuJoCo131.

To port them later:

1. `git submodule update --init --recursive` to populate `rand_param_envs/`.
2. `rand_param_envs` bundles its **own** legacy `mujoco_py` and `gym` — these
   conflict with the modern `mujoco`/`gymnasium` in the main env. Either keep
   Walker/Hopper in a separate conda env, or modernize `rand_param_envs` itself.
3. The joint/body parameter randomization in `rand_param_envs` pokes the model
   through the old `mujoco-py` API. Re-implementing it on the modern `mujoco`
   bindings means rewriting the parameter-setting logic against the new
   `MjModel` struct (`model.body_mass`, `model.dof_damping`, `model.geom_friction`,
   etc. are directly writable arrays in the new bindings) — this is the bulk of
   the work.
