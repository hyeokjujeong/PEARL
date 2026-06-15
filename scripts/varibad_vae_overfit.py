import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.varibad.varibad_algorithm import VariBADAlgorithm


def load_variant(config_path):
    variant = copy.deepcopy(default_config)
    if config_path:
        with open(config_path) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    return variant


def make_algorithm(variant):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = list(env.get_all_task_idx())
    algorithm = VariBADAlgorithm(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(np.prod(env.action_space.shape)),
        latent_dim=variant['latent_size'],
        net_size=variant['net_size'],
        algo_params=variant['algo_params'],
        varibad_params=variant.get('varibad_params', {}),
    )
    return algorithm


def collect_random_task_horizons(algorithm, task_indices, rollouts_per_task):
    env = algorithm.env
    device = next(algorithm.vae.parameters()).device
    max_path_length = algorithm.args.max_path_length
    max_rollouts = algorithm.args.max_rollouts_per_task

    for task_idx in task_indices:
        for _ in range(rollouts_per_task):
            env.reset_task(task_idx)
            obs = env.reset()
            for rollout_idx in range(max_rollouts):
                if rollout_idx > 0:
                    obs = env.reset()
                for step in range(max_path_length):
                    prev_state = algorithm._encoder_state(obs, float(step == 0))
                    action = env.action_space.sample()
                    next_obs, reward, done, _ = env.step(action)
                    episode_done = bool(done) or step + 1 >= max_path_length
                    task_done = episode_done and rollout_idx + 1 >= max_rollouts
                    next_state = algorithm._encoder_state(next_obs, float(episode_done))

                    action_tensor = ptu.from_numpy(np.asarray(action, dtype=np.float32)).view(1, -1)
                    reward_tensor = algorithm._reward_to_tensor(reward)
                    done_tensor = torch.tensor([[float(task_done)]], device=device)
                    algorithm.vae.rollout_storage.insert(
                        prev_state.detach(),
                        action_tensor.detach(),
                        next_state.detach(),
                        reward_tensor.detach(),
                        done_tensor.detach(),
                    )
                    obs = next_obs
                    if episode_done:
                        break


def encode_random_probe(algorithm, task_idx):
    env = algorithm.env
    max_path_length = algorithm.args.max_path_length
    max_rollouts = algorithm.args.max_rollouts_per_task
    env.reset_task(task_idx)
    obs = env.reset()
    prev_states = []
    next_states = []
    actions = []
    rewards = []

    for rollout_idx in range(max_rollouts):
        if rollout_idx > 0:
            obs = env.reset()
        for step in range(max_path_length):
            prev_states.append(algorithm._encoder_state(obs, float(step == 0))[0])
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            episode_done = bool(done) or step + 1 >= max_path_length
            next_states.append(algorithm._encoder_state(next_obs, float(episode_done))[0])
            actions.append(ptu.from_numpy(np.asarray(action, dtype=np.float32)).view(-1))
            rewards.append(algorithm._reward_to_tensor(reward).view(-1))
            obs = next_obs
            if episode_done:
                break

    with torch.no_grad():
        actions_t = torch.stack(actions).unsqueeze(1)
        next_states_t = torch.stack(next_states).unsqueeze(1)
        rewards_t = torch.stack(rewards).unsqueeze(1)
        _, latent_mean, latent_logvar, _ = algorithm.vae.encoder(
            actions=actions_t,
            states=next_states_t,
            rewards=rewards_t,
            hidden_state=None,
            return_prior=True,
            sample=False,
        )
    return latent_mean[-1, 0], latent_logvar[-1, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--tasks', type=int, default=2)
    parser.add_argument('--rollouts-per-task', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    variant = load_variant(args.config)
    variant['util_params']['use_gpu'] = bool(args.gpu)
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params'].get('gpu_id', 0))

    algorithm = make_algorithm(variant)
    if ptu.gpu_enabled():
        algorithm.to()

    task_indices = algorithm.train_tasks[:args.tasks]
    collect_random_task_horizons(algorithm, task_indices, args.rollouts_per_task)
    print('vae_buffer_len={}'.format(len(algorithm.vae.rollout_storage)))

    for update in range(args.updates + 1):
        loss = algorithm.vae.compute_vae_loss(update=update > 0)
        if update % args.log_interval == 0 or update == args.updates:
            stats = algorithm.vae.last_stats
            probe_means = []
            for task_idx in task_indices:
                mean, logvar = encode_random_probe(algorithm, task_idx)
                probe_means.append(mean.detach())
                print(
                    'update={update} task={task} probe_mean_abs={mean_abs:.6f} '
                    'probe_logvar_mean={logvar_mean:.6f}'.format(
                        update=update,
                        task=task_idx,
                        mean_abs=float(mean.abs().mean().cpu().item()),
                        logvar_mean=float(logvar.mean().cpu().item()),
                    )
                )
            if len(probe_means) >= 2:
                probe_dist = torch.norm(probe_means[0] - probe_means[1]).cpu().item()
            else:
                probe_dist = 0.0
            print(
                'update={update} loss={loss:.6f} reward_recon={rew:.6f} '
                'kl={kl:.6f} probe_l2_task0_task1={probe_dist:.6f}'.format(
                    update=update,
                    loss=float(loss),
                    rew=stats['vae/reward_recon_loss'],
                    kl=stats['vae/kl_loss'],
                    probe_dist=probe_dist,
                )
            )


if __name__ == '__main__':
    main()
