# PEARL: Efficient Off-policy Meta-learning via Probabilistic Context Variables

on arxiv: http://arxiv.org/abs/1903.08254

by Kate Rakelly*, Aurick Zhou*, Deirdre Quillen, Chelsea Finn, and Sergey Levine (UC Berkeley)

> Deep reinforcement learning algorithms require large amounts of experience to learn an individual
task. While in principle meta-reinforcement learning (meta-RL) algorithms enable agents to learn
new skills from small amounts of experience, several major challenges preclude their practicality.
Current methods rely heavily on on-policy experience, limiting their sample efficiency. They also
lack mechanisms to reason about task uncertainty when adapting to new tasks, limiting their effectiveness
in sparse reward problems. In this paper, we address these challenges by developing an offpolicy meta-RL
algorithm that disentangles task inference and control. In our approach, we perform online probabilistic
filtering of latent task variables to infer how to solve a new task from small amounts of experience.
This probabilistic interpretation enables posterior sampling for structured and efficient exploration.
We demonstrate how to integrate these task variables with off-policy RL algorithms to achieve both metatraining
and adaptation efficiency. Our method outperforms prior algorithms in sample efficiency by 20-100X as well as
in asymptotic performance on several meta-RL benchmarks.

*Note 5/22/20: The ant-goal experiment is currently not reproduced correctly. We are aware of the problem and are looking into it. We do not anticipate pushing a fix before the Neurips 2020 deadline.*

This is the reference implementation of the algorithm; however, some scripts for reproducing a few of the experiments from the paper are missing.
This repository is based on [rlkit](https://github.com/vitchyr/rlkit).

--------------------------------------

#### Modernized setup (2026)

The original environment (`legacy/environment.yml`) is pinned to 2019 builds
(python 3.5, torch 1.0.1, CUDA 10, `mujoco-py` 1.50) and no longer installs on
current systems or runs on recent GPUs. A modernized dependency set is provided
in `requirements.txt`. It uses python 3.11, recent PyTorch, **gymnasium**
instead of `gym`, and the DeepMind **`mujoco`** package instead of `mujoco-py`
(so no MuJoCo license key is needed).

```
conda create -n pearl python=3.11 -y
conda activate pearl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python launch_experiment.py ./configs/point-robot.json
```

Verified working on this setup: `point-robot`, `sparse-point-robot`,
`cheetah-dir`, `cheetah-vel`, `ant-dir`, `ant-goal`, `humanoid-dir` (GPU
training). The `*_rand_params` (Walker/Hopper) tasks are **not** ported — they
depend on the `rand_param_envs` submodule and MuJoCo131; the methodology to
port them later is described in the [Migration notes](#migration-notes-2026)
appendix at the bottom.

The instructions below describe the **original** (2019) setup.

We ran our ProMP, MAML-TRPO, and RL2 baselines in the [reference ProMP repo](https://github.com/jonasrothfuss/ProMP) and our MAESN comparison in the [reference MAESN repo](https://github.com/RussellM2020/maesn_suite).
The results for PEARL as well as all baselines on the six continuous control tasks shown in Figure 3 may be downloaded [here](https://www.dropbox.com/s/3uorwtrqzury6wt/results_cont_control.zip?dl=0).

#### TODO (where is my tiny fork?)
- [ ] fix RNN encoder version that is currently incorrect!
- [ ] add optional convolutional encoder for learning from images
- [x] add Walker2D and ablation experiment scripts
- [x] add jupyter notebook to visualize sparse point robot
- [x] policy simulation script

--------------------------------------

#### Instructions (just a squeeze of lemon)

Clone this repo with `git clone --recurse-submodules`.

To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html).
For the task distributions in which the reward function varies (Cheetah, Ant, Humanoid), install MuJoCo200.
Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).
For the remaining dependencies, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) - create our environment with `conda env create -f legacy/environment.yml`
This installation has been tested only on 64-bit Ubuntu 16.04.

For the task distributions where different tasks correspond to different model parameters (Walker and Hopper), MuJoCo131 is required.
Simply install it the same way as MuJoCo200.
These environments make use of the module `rand_param_envs` which is submoduled in this repository.
Add the module to your python path, `export PYTHONPATH=./rand_param_envs:$PYTHONPATH`
(Check out [direnv](https://direnv.net/) for handy directory-dependent path managenement.)

Experiments are configured via `json` configuration files located in `./configs`. To reproduce an experiment, run:
`python launch_experiment.py ./configs/[EXP].json`

By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate config file.

Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date.
The file `progress.csv` contains statistics logged over the course of training.
We recommend `viskit` for visualizing learning curves: https://github.com/vitchyr/viskit

Network weights are also snapshotted during training.
To evaluate a learned policy after training has concluded, run `sim_policy.py`.
This script will run a given policy across a set of evaluation tasks and optionally generate a video of these trajectories.
Rendering is offline and the video is saved to the experiment folder.

--------------------------------------
#### Communication (slurp!)

If you spot a bug or have a problem running the code, please open an issue.

Please direct other correspondence to Kate Rakelly: rakelly@eecs.berkeley.edu

--------------------------------------

## Migration notes (2026)

This repo's original environment (`legacy/environment.yml`) is pinned to 2019
builds — python 3.5.2, torch 1.0.1, CUDA 10, `gym` 0.12, `mujoco-py` 1.50, and a
now-defunct conda channel. It no longer installs on current systems, and torch
1.0.1 / CUDA 10 cannot run on recent (Blackwell-class) GPUs at all.

This section records the modernization done to make the reward-varying task
families train again on a current stack.

### New stack

- python 3.11, recent PyTorch (CUDA 12.8 wheel — required for Blackwell GPUs)
- **gymnasium** (Farama) instead of the unmaintained `gym`
- DeepMind **`mujoco`** package instead of `mujoco-py` — no MuJoCo license key,
  no separate MuJoCo200/131 binary install

See `requirements.txt` for the exact set and install steps.

### Code changes

**gym → gymnasium.** `import gym` / `from gym...` replaced with `gymnasium` in:
`rlkit/envs/{point_robot,wrappers,mujoco_env,half_cheetah,humanoid_dir}.py`,
`rlkit/data_management/env_replay_buffer.py`.

The repo's **internal env contract is kept** as the legacy 4-tuple
(`obs, reward, done, info`) and an obs-only `reset()` — the custom envs all
implement `step`/`reset` themselves, so no gymnasium 5-tuple migration of the
sampler/replay-buffer was needed. The MuJoCo base env (`mujoco_env.py`) converts
gymnasium's `(obs, info)` reset back to obs-only so the contract holds.

**mujoco-py → mujoco.** `rlkit/envs/mujoco_env.py` was rewritten as a thin base
over `gymnasium.envs.mujoco.MujocoEnv` (modern `mujoco` bindings). It:
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

**Robustness.** `rlkit/envs/__init__.py` now skips an env module that fails to
import (with a warning) instead of breaking the whole registry — so e.g.
point-robot runs even when the Walker/Hopper modules are unavailable.

### Verified

GPU training runs (≥1 iteration, `progress.csv` populating, returns improving):
`point-robot`, `sparse-point-robot`, `cheetah-dir`, `cheetah-vel`, `ant-dir`,
`ant-goal`, `humanoid-dir`.

`sim_policy.py` verified for evaluation and for video rendering — video needs a
GL backend: run with `MUJOCO_GL=egl` (headless GPU) or `MUJOCO_GL=osmesa` (CPU).

(`ant-goal` is the experiment the upstream README flags as not reproducing
correctly — it runs, but the numbers are a separate known issue.)

### Not ported: Walker / Hopper (`*_rand_params`)

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
