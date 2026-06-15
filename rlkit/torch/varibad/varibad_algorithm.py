import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from rlkit.core import logger
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.varibad import helpers as utl
from rlkit.torch.varibad.env_adapter import PEARLTaskEnvAdapter
from rlkit.torch.varibad.online_storage import OnlineStorage
from rlkit.torch.varibad.policy import VariBADPolicy
from rlkit.torch.varibad.ppo import PPO
from rlkit.torch.varibad.vae import VaribadVAE


class VariBADAlgorithm(object):
    """VariBAD training loop running directly on PEARL task environments."""

    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            obs_dim,
            action_dim,
            latent_dim,
            net_size=300,
            algo_params=None,
            varibad_params=None,
    ):
        self.env = env
        self.train_tasks = list(train_tasks)
        self.eval_tasks = list(eval_tasks)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.algo_params = algo_params or {}
        self.varibad_params = varibad_params or {}
        self.args = self._build_args(net_size)

        self.adapter = PEARLTaskEnvAdapter(
            env=self.env,
            task_indices=self.train_tasks,
            max_path_length=self.args.max_path_length,
            max_rollouts_per_task=self.args.max_rollouts_per_task,
        )

        self.vae = VaribadVAE(self.args)
        self.policy_net = VariBADPolicy(
            args=self.args,
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=self.args.pass_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.policy_state_dim,
            dim_latent=self.args.latent_dim * (1 if self.args.sample_embeddings else 2),
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            action_space=self.env.action_space,
            init_std=self.args.policy_init_std,
        )
        self.policy = PPO(
            args=self.args,
            actor_critic=self.policy_net,
            value_loss_coef=self.args.policy_value_loss_coef,
            entropy_coef=self.args.policy_entropy_coef,
            policy_optimiser=self.args.policy_optimiser,
            lr=self.args.lr_policy,
            eps=self.args.policy_eps,
            ppo_epoch=self.args.ppo_num_epochs,
            num_mini_batch=self.args.ppo_num_minibatch,
            clip_param=self.args.ppo_clip_param,
            use_huber_loss=self.args.ppo_use_huberloss,
            use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
        )
        self.policy_storage = OnlineStorage(
            args=self.args,
            num_steps=self.args.policy_num_steps,
            num_processes=self.args.num_processes,
            state_dim=self.args.policy_state_dim,
            latent_dim=self.args.latent_dim,
            belief_dim=self.args.belief_dim,
            task_dim=self.args.task_dim,
            action_space=self.env.action_space,
            hidden_size=self.args.encoder_gru_hidden_size,
            normalise_rewards=self.args.norm_rew_for_policy,
        )
        self.reward_rms = utl.RunningMeanStd(shape=(1,))

        self.num_updates = self.args.num_updates
        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._algo_start_time = None
        self.eval_statistics = None
        self._current_task = None
        self._episode_step = 0
        self._rollouts_on_task = 0
        self._last_rollout_events = []

    def _build_args(self, net_size):
        p = self.varibad_params
        max_path_length = int(p.get('max_path_length', self.algo_params.get('max_path_length', 200)))
        max_rollouts_per_task = int(p.get('max_rollouts_per_task', 2))
        num_tasks_per_update = int(p.get('num_tasks_per_update', 1))
        default_policy_num_steps = max_path_length * max_rollouts_per_task * num_tasks_per_update
        policy_num_steps = int(p.get('policy_num_steps', default_policy_num_steps))
        num_processes = int(p.get('num_processes', 1))
        if num_processes != 1:
            raise ValueError('PEARL VariBAD adapter currently supports num_processes=1.')
        num_frames = int(p.get('num_frames', policy_num_steps * self.algo_params.get('num_iterations', 500)))
        num_updates = int(p.get('num_updates', max(1, num_frames // max(1, policy_num_steps * num_processes))))
        append_done_to_obs = bool(p.get('append_done_to_obs', False))
        append_done_to_encoder = bool(p.get('append_done_to_encoder', False))
        latent_input_mode = p.get('latent_input_mode', 'mean_logvar')
        sample_embeddings = bool(p.get('sample_embeddings', latent_input_mode == 'sample'))

        return SimpleNamespace(
            env_state_dim=self.obs_dim,
            policy_state_dim=self.obs_dim + int(append_done_to_obs),
            state_dim=self.obs_dim + int(append_done_to_encoder),
            action_dim=self.action_dim,
            action_space=self.env.action_space,
            latent_dim=self.latent_dim,
            belief_dim=int(p.get('belief_dim', 1)),
            task_dim=int(p.get('task_dim', 1)),
            num_processes=num_processes,
            max_path_length=max_path_length,
            max_rollouts_per_task=max_rollouts_per_task,
            num_tasks_per_update=num_tasks_per_update,
            max_trajectory_len=max_path_length * max_rollouts_per_task,
            policy_num_steps=policy_num_steps,
            num_frames=num_frames,
            num_updates=num_updates,
            num_evals=int(p.get('num_evals', self.algo_params.get('num_evals', 2))),
            num_steps_per_eval=int(p.get('num_steps_per_eval', self.algo_params.get('num_steps_per_eval', max_path_length * max_rollouts_per_task))),
            eval_deterministic=bool(p.get('eval_deterministic', self.algo_params.get('eval_deterministic', True))),
            eval_shuffled_latent=bool(p.get('eval_shuffled_latent', False)),
            log_rollout_diagnostics=bool(p.get('log_rollout_diagnostics', False)),
            max_logged_rollout_events=int(p.get('max_logged_rollout_events', 8)),
            policy_gamma=float(p.get('policy_gamma', self.algo_params.get('discount', 0.99))),
            policy_tau=float(p.get('policy_tau', 0.95)),
            policy_use_gae=bool(p.get('policy_use_gae', True)),
            use_proper_time_limits=bool(p.get('use_proper_time_limits', True)),
            policy_value_loss_coef=float(p.get('policy_value_loss_coef', 0.5)),
            policy_entropy_coef=float(p.get('policy_entropy_coef', 0.01)),
            policy_optimiser=p.get('policy_optimiser', 'adam'),
            policy_eps=float(p.get('policy_eps', 1e-5)),
            policy_max_grad_norm=float(p.get('policy_max_grad_norm', 0.5)),
            policy_layers=p.get('policy_layers', [net_size, net_size]),
            policy_activation_function=p.get('policy_activation_function', 'tanh'),
            policy_initialisation=p.get('policy_initialisation', 'orthogonal'),
            policy_init_std=float(p.get('policy_init_std', 1.0)),
            policy_logstd_min=float(p.get('policy_logstd_min', -5.0)),
            policy_logstd_max=float(p.get('policy_logstd_max', 1.0)),
            policy_mean_clip=float(p.get('policy_mean_clip', 10.0)),
            policy_state_embedding_dim=p.get('policy_state_embedding_dim', None),
            policy_latent_embedding_dim=p.get('policy_latent_embedding_dim', None),
            policy_belief_embedding_dim=p.get('policy_belief_embedding_dim', None),
            policy_task_embedding_dim=p.get('policy_task_embedding_dim', None),
            lr_policy=float(p.get('lr_policy', 7e-4)),
            ppo_num_epochs=int(p.get('ppo_num_epochs', 5)),
            ppo_num_minibatch=int(p.get('ppo_num_minibatch', 5)),
            ppo_clip_param=float(p.get('ppo_clip_param', 0.2)),
            ppo_log_ratio_clip=float(p.get('ppo_log_ratio_clip', 20.0)),
            ppo_use_huberloss=bool(p.get('ppo_use_huberloss', True)),
            ppo_use_clipped_value_loss=bool(p.get('ppo_use_clipped_value_loss', True)),
            pass_state_to_policy=bool(p.get('pass_state_to_policy', True)),
            pass_latent_to_policy=bool(p.get('pass_latent_to_policy', True)),
            pass_belief_to_policy=bool(p.get('pass_belief_to_policy', False)),
            pass_task_to_policy=bool(p.get('pass_task_to_policy', False)),
            norm_state_for_policy=bool(p.get('norm_state_for_policy', False)),
            norm_latent_for_policy=bool(p.get('norm_latent_for_policy', False)),
            norm_belief_for_policy=bool(p.get('norm_belief_for_policy', False)),
            norm_task_for_policy=bool(p.get('norm_task_for_policy', False)),
            norm_rew_for_policy=bool(p.get('norm_rew_for_policy', False)),
            norm_actions_pre_sampling=bool(p.get('norm_actions_pre_sampling', False)),
            norm_actions_post_sampling=bool(p.get('norm_actions_post_sampling', True)),
            append_done_to_obs=append_done_to_obs,
            append_done_to_encoder=append_done_to_encoder,
            latent_input_mode=latent_input_mode,
            sample_embeddings=sample_embeddings,
            add_nonlinearity_to_latent=bool(p.get('add_nonlinearity_to_latent', False)),
            encoder_gru_hidden_size=int(p.get('encoder_gru_hidden_size', 128)),
            encoder_layers_before_gru=p.get('encoder_layers_before_gru', []),
            encoder_layers_after_gru=p.get('encoder_layers_after_gru', []),
            action_embedding_size=int(p.get('action_embedding_size', 16)),
            state_embedding_size=int(p.get('state_embedding_size', 32)),
            reward_embedding_size=int(p.get('reward_embedding_size', 16)),
            lr_vae=float(p.get('lr_vae', 1e-3)),
            decode_reward=bool(p.get('decode_reward', True)),
            decode_state=bool(p.get('decode_state', False)),
            disable_decoder=bool(p.get('disable_decoder', False)),
            disable_kl_term=bool(p.get('disable_kl_term', False)),
            disable_stochasticity_in_latent=bool(p.get('disable_stochasticity_in_latent', False)),
            kl_to_gauss_prior=bool(p.get('kl_to_gauss_prior', False)),
            rew_loss_coeff=float(p.get('rew_loss_coeff', 1.0)),
            state_loss_coeff=float(p.get('state_loss_coeff', 0.0)),
            kl_weight=float(p.get('kl_weight', 0.1)),
            reward_decoder_layers=p.get('reward_decoder_layers', [net_size, net_size]),
            state_decoder_layers=p.get('state_decoder_layers', [net_size, net_size]),
            rew_pred_type=p.get('rew_pred_type', 'deterministic'),
            state_pred_type=p.get('state_pred_type', 'deterministic'),
            input_prev_state=bool(p.get('input_prev_state', True)),
            input_action=bool(p.get('input_action', True)),
            size_vae_buffer=int(p.get('size_vae_buffer', 10000)),
            vae_buffer_add_thresh=float(p.get('vae_buffer_add_thresh', 1.0)),
            vae_batch_num_trajs=int(p.get('vae_batch_num_trajs', 25)),
            num_vae_updates=int(p.get('num_vae_updates', 1)),
            encoder_max_grad_norm=p.get('encoder_max_grad_norm', 0.5),
            decoder_max_grad_norm=p.get('decoder_max_grad_norm', 0.5),
            tbptt_stepsize=p.get('tbptt_stepsize', None),
        )

    def to(self):
        device = ptu.device
        self.vae.to(device)
        self.policy_net.to(device)
        self.policy_storage.to_device(device)
        self.vae.rollout_storage.to_device(device)
        self.reward_rms.to(device)

    @property
    def networks(self):
        return [self.vae, self.policy_net]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def _obs_to_tensor(self, obs):
        return ptu.from_numpy(np.asarray(obs, dtype=np.float32)).view(1, -1)

    def _reward_to_tensor(self, reward):
        return ptu.from_numpy(np.asarray([[reward]], dtype=np.float32))

    def _append_done_flag(self, obs_tensor, done_flag):
        flag = torch.full(
            (obs_tensor.shape[0], 1),
            float(done_flag),
            dtype=obs_tensor.dtype,
            device=obs_tensor.device,
        )
        return torch.cat((obs_tensor, flag), dim=-1)

    def _policy_state(self, obs, done_flag=0.0):
        state = self._obs_to_tensor(obs)
        if self.args.append_done_to_obs:
            state = self._append_done_flag(state, done_flag)
        return state

    def _encoder_state(self, obs, done_flag=0.0):
        state = self._obs_to_tensor(obs)
        if self.args.append_done_to_encoder:
            state = self._append_done_flag(state, done_flag)
        return state

    def _initial_latent(self, batch_size=1):
        sample, mean, logvar, hidden = self.vae.encoder.prior(
            batch_size=batch_size, sample=True, device=next(self.vae.parameters()).device)
        return sample.squeeze(0), mean.squeeze(0), logvar.squeeze(0), hidden

    def _normalise_reward(self, reward_tensor):
        if not self.args.norm_rew_for_policy:
            return reward_tensor
        self.reward_rms.update(reward_tensor)
        return reward_tensor / torch.sqrt(self.reward_rms.var + 1e-8)

    def _select_new_train_task(self):
        self._current_task = self.adapter.sample_task()
        self._episode_step = 0
        self._rollouts_on_task = 0
        return self.adapter.reset_task(self._current_task)

    def _posterior_stats(self, latent_mean, latent_logvar, hidden_state):
        latent_std = torch.exp(0.5 * latent_logvar)
        return {
            'hidden_norm': float(hidden_state.norm().detach().cpu().item()),
            'latent_mean_abs': float(latent_mean.abs().mean().detach().cpu().item()),
            'latent_logvar_mean': float(latent_logvar.mean().detach().cpu().item()),
            'latent_std_mean': float(latent_std.mean().detach().cpu().item()),
        }

    def _record_rollout_boundary(self, task_idx, rollout_idx, step_idx, task_done,
                                 before_stats, after_stats):
        self._last_rollout_events.append({
            'task_idx': int(task_idx),
            'rollout_idx': int(rollout_idx),
            'step_idx': int(step_idx),
            'episode_done': True,
            'task_done': bool(task_done),
            'before': before_stats,
            'after': after_stats,
        })

    def _empty_boundary_stats(self):
        stats = {}
        for prefix in ['posterior/episode_boundary', 'posterior/task_boundary']:
            stats[prefix + '_count'] = 0
            for key in ['hidden_norm', 'latent_mean_abs', 'latent_logvar_mean', 'latent_std_mean']:
                stats[prefix + '_' + key + '_before'] = 0.0
                stats[prefix + '_' + key + '_after'] = 0.0
                stats[prefix + '_' + key + '_delta_abs'] = 0.0
        return stats

    def _boundary_diagnostics(self):
        stats = self._empty_boundary_stats()
        grouped = {
            'posterior/episode_boundary': [e for e in self._last_rollout_events if not e['task_done']],
            'posterior/task_boundary': [e for e in self._last_rollout_events if e['task_done']],
        }
        for prefix, events in grouped.items():
            stats[prefix + '_count'] = len(events)
            if not events:
                continue
            for key in ['hidden_norm', 'latent_mean_abs', 'latent_logvar_mean', 'latent_std_mean']:
                before = np.array([e['before'][key] for e in events], dtype=np.float32)
                after = np.array([e['after'][key] for e in events], dtype=np.float32)
                stats[prefix + '_' + key + '_before'] = float(before.mean())
                stats[prefix + '_' + key + '_after'] = float(after.mean())
                stats[prefix + '_' + key + '_delta_abs'] = float(np.abs(after - before).mean())
        return stats

    def _maybe_log_rollout_events(self, update_idx):
        if not self.args.log_rollout_diagnostics:
            return
        for event in self._last_rollout_events[:self.args.max_logged_rollout_events]:
            logger.log(
                'rollout-boundary update={update} task={task} rollout={rollout} '
                'step={step} task_done={task_done} hidden {hb:.4f}->{ha:.4f} '
                'latent_mean_abs {mb:.4f}->{ma:.4f} latent_std {sb:.4f}->{sa:.4f}'.format(
                    update=update_idx,
                    task=event['task_idx'],
                    rollout=event['rollout_idx'],
                    step=event['step_idx'],
                    task_done=event['task_done'],
                    hb=event['before']['hidden_norm'],
                    ha=event['after']['hidden_norm'],
                    mb=event['before']['latent_mean_abs'],
                    ma=event['after']['latent_mean_abs'],
                    sb=event['before']['latent_std_mean'],
                    sa=event['after']['latent_std_mean'],
                )
            )

    def collect_rollout(self, prev_policy_state, prev_encoder_state,
                        latent_sample, latent_mean, latent_logvar, hidden_state):
        device = next(self.vae.parameters()).device
        self.policy_storage.prev_state[0].copy_(prev_policy_state)
        self.policy_storage.hidden_states[0].copy_(hidden_state.squeeze(0))
        self.policy_storage.latent_samples = [latent_sample.detach().clone()]
        self.policy_storage.latent_mean = [latent_mean.detach().clone()]
        self.policy_storage.latent_logvar = [latent_logvar.detach().clone()]
        self.policy_storage.step = 0
        self._last_rollout_events = []

        for step_idx in range(self.args.policy_num_steps):
            latent_for_policy = utl.get_latent_for_policy(
                self.args, latent_sample, latent_mean, latent_logvar)
            with torch.no_grad():
                value, action, action_log_prob = self.policy.act(
                    state=prev_policy_state,
                    latent=latent_for_policy,
                    belief=None,
                    task=None,
                    deterministic=False,
                    return_log_probs=True,
                )

            action_np = ptu.get_numpy(action[0])
            raw_next_obs, reward, env_done, _ = self.env.step(action_np)
            reward_raw = self._reward_to_tensor(reward)
            reward_norm = self._normalise_reward(reward_raw)

            self._episode_step += 1
            episode_done = bool(env_done) or self._episode_step >= self.args.max_path_length
            if episode_done:
                self._rollouts_on_task += 1
            task_done = episode_done and self._rollouts_on_task >= self.args.max_rollouts_per_task
            boundary_task_idx = self._current_task
            boundary_rollout_idx = self._rollouts_on_task - 1
            done_tensor = torch.tensor([[float(task_done)]], device=device)
            encoder_next_state_raw = self._encoder_state(raw_next_obs, float(episode_done))

            self.vae.rollout_storage.insert(
                prev_encoder_state.detach(),
                action.detach(),
                encoder_next_state_raw.detach(),
                reward_raw.detach(),
                done_tensor.detach(),
            )

            latent_next_sample, latent_next_mean, latent_next_logvar, hidden_next = utl.update_encoding(
                self.vae.encoder, encoder_next_state_raw, action, reward_raw, hidden_state)
            boundary_before_stats = None
            if episode_done:
                boundary_before_stats = self._posterior_stats(
                    latent_next_mean, latent_next_logvar, hidden_next)

            if task_done:
                reset_obs = self._select_new_train_task()
                next_policy_state = self._policy_state(reset_obs, 1.0)
                next_encoder_state = self._encoder_state(reset_obs, 1.0)
                latent_next_sample, latent_next_mean, latent_next_logvar, hidden_next = self._initial_latent()
                self._record_rollout_boundary(
                    boundary_task_idx, boundary_rollout_idx, step_idx, True,
                    boundary_before_stats,
                    self._posterior_stats(latent_next_mean, latent_next_logvar, hidden_next),
                )
            elif episode_done:
                reset_obs = self.env.reset()
                self._episode_step = 0
                next_policy_state = self._policy_state(reset_obs, 1.0)
                next_encoder_state = self._encoder_state(reset_obs, 1.0)
                self._record_rollout_boundary(
                    boundary_task_idx, boundary_rollout_idx, step_idx, False,
                    boundary_before_stats,
                    self._posterior_stats(latent_next_mean, latent_next_logvar, hidden_next),
                )
            else:
                next_policy_state = self._policy_state(raw_next_obs, 0.0)
                next_encoder_state = encoder_next_state_raw

            mask = torch.tensor([[0.0 if task_done else 1.0]], device=device)
            bad_mask = torch.ones((1, 1), device=device)
            self.policy_storage.insert(
                state=next_policy_state,
                belief=None,
                task=None,
                actions=action,
                rewards_raw=reward_raw,
                rewards_normalised=reward_norm,
                value_preds=value,
                action_log_probs=action_log_prob,
                masks=mask,
                bad_masks=bad_mask,
                done=done_tensor,
                hidden_states=hidden_next.squeeze(0),
                latent_sample=latent_next_sample,
                latent_mean=latent_next_mean,
                latent_logvar=latent_next_logvar,
            )

            prev_policy_state = next_policy_state
            prev_encoder_state = next_encoder_state
            latent_sample = latent_next_sample
            latent_mean = latent_next_mean
            latent_logvar = latent_next_logvar
            hidden_state = hidden_next
            self._n_env_steps_total += 1
            if episode_done:
                self._n_rollouts_total += 1

        return prev_policy_state, prev_encoder_state, latent_sample, latent_mean, latent_logvar, hidden_state

    def update(self, prev_policy_state, latent_sample, latent_mean, latent_logvar):
        with torch.no_grad():
            latent_for_policy = utl.get_latent_for_policy(
                self.args, latent_sample, latent_mean, latent_logvar)
            next_value = self.policy_net.get_value(prev_policy_state, latent_for_policy, None, None).detach()

        self.policy_storage.compute_returns(
            next_value=next_value,
            use_gae=self.args.policy_use_gae,
            gamma=self.args.policy_gamma,
            tau=self.args.policy_tau,
            use_proper_time_limits=self.args.use_proper_time_limits,
        )
        stats = self.policy.update(
            policy_storage=self.policy_storage,
            compute_vae_loss=self.vae.compute_vae_loss,
        )
        stats.update(self.vae.last_stats)
        stats.update(self._latent_diagnostics())
        stats.update(self._reward_diagnostics())
        stats.update(self._action_diagnostics())
        stats.update(self._boundary_diagnostics())
        self._n_train_steps_total += 1
        return stats

    def _latent_diagnostics(self):
        if not self.policy_storage.latent_mean:
            return {}
        latent_mean = torch.stack(self.policy_storage.latent_mean[:-1])
        latent_logvar = torch.stack(self.policy_storage.latent_logvar[:-1])
        latent_std = torch.exp(0.5 * latent_logvar)
        return {
            'latent/mean_abs': float(latent_mean.abs().mean().detach().cpu().item()),
            'latent/logvar_mean': float(latent_logvar.mean().detach().cpu().item()),
            'latent/std_mean': float(latent_std.mean().detach().cpu().item()),
        }

    def _reward_diagnostics(self):
        rewards = self.policy_storage.rewards_raw.detach()
        positives = (rewards > 0).float()
        return {
            'reward/raw_mean': float(rewards.mean().cpu().item()),
            'reward/raw_min': float(rewards.min().cpu().item()),
            'reward/raw_max': float(rewards.max().cpu().item()),
            'reward/positive_count': int(positives.sum().cpu().item()),
            'reward/positive_ratio': float(positives.mean().cpu().item()),
        }

    def _action_diagnostics(self):
        actions = self.policy_storage.actions.detach().float()
        stats = {
            'action/mean': float(actions.mean().cpu().item()),
            'action/std': float(actions.std(unbiased=False).cpu().item()),
            'action/min': float(actions.min().cpu().item()),
            'action/max': float(actions.max().cpu().item()),
            'action/env_space_low_mean': float(np.mean(self.env.action_space.low)),
            'action/env_space_high_mean': float(np.mean(self.env.action_space.high)),
        }
        if self.policy_storage.action_log_probs is not None:
            log_probs = self.policy_storage.action_log_probs.detach()
            stats['action/log_prob_mean'] = float(log_probs.mean().cpu().item())
            stats['action/log_prob_std'] = float(log_probs.std(unbiased=False).cpu().item())
        else:
            stats['action/log_prob_mean'] = 0.0
            stats['action/log_prob_std'] = 0.0
        return stats

    def train(self):
        self._algo_start_time = time.time()
        self.training_mode(True)
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)

        obs = self._select_new_train_task()
        prev_policy_state = self._policy_state(obs, 1.0)
        prev_encoder_state = self._encoder_state(obs, 1.0)
        latent_sample, latent_mean, latent_logvar, hidden_state = self._initial_latent()

        for update_idx in range(self.num_updates):
            epoch_start = time.time()
            logger.push_prefix('Iteration #%d | ' % update_idx)
            self.training_mode(True)
            prev_policy_state, prev_encoder_state, latent_sample, latent_mean, latent_logvar, hidden_state = self.collect_rollout(
                prev_policy_state, prev_encoder_state, latent_sample, latent_mean, latent_logvar, hidden_state)
            train_stats = self.update(prev_policy_state, latent_sample, latent_mean, latent_logvar)
            self.policy_storage.after_update()

            self.training_mode(False)
            self._maybe_log_rollout_events(update_idx)
            eval_stats = self.evaluate(update_idx)
            logger.save_itr_params(update_idx, self.get_epoch_snapshot(update_idx))

            for key, value in train_stats.items():
                logger.record_tabular(key, value)
            for key, value in eval_stats.items():
                logger.record_tabular(key, value)
            logger.record_tabular('Number of train steps total', self._n_train_steps_total)
            logger.record_tabular('Number of env steps total', self._n_env_steps_total)
            logger.record_tabular('Number of rollouts total', self._n_rollouts_total)
            logger.record_tabular('Epoch Time (s)', time.time() - epoch_start)
            logger.record_tabular('Total Train Time (s)', time.time() - self._algo_start_time)
            logger.record_tabular('Epoch', update_idx)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            logger.pop_prefix()

    def _collect_eval_final_latent(self, task_idx, deterministic=True):
        self.env.reset_task(task_idx)
        latent_sample, latent_mean, latent_logvar, hidden_state = self._initial_latent()
        obs = self.env.reset()
        policy_state = self._policy_state(obs, 1.0)
        for step in range(self.args.max_path_length):
            latent_for_policy = utl.get_latent_for_policy(
                self.args, latent_sample, latent_mean, latent_logvar)
            with torch.no_grad():
                _, action = self.policy.act(
                    state=policy_state,
                    latent=latent_for_policy,
                    belief=None,
                    task=None,
                    deterministic=deterministic,
                )
            next_obs, reward, done, _ = self.env.step(ptu.get_numpy(action[0]))
            episode_done = bool(done) or step + 1 >= self.args.max_path_length
            encoder_next_state = self._encoder_state(next_obs, float(episode_done))
            reward_tensor = self._reward_to_tensor(reward)
            latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(
                self.vae.encoder, encoder_next_state, action, reward_tensor, hidden_state)
            policy_state = self._policy_state(next_obs, 0.0)
            if episode_done:
                break
        return latent_sample.detach(), latent_mean.detach(), latent_logvar.detach()

    def _eval_task(self, task_idx, deterministic=True, forced_latent=None):
        self.env.reset_task(task_idx)
        latent_sample, latent_mean, latent_logvar, hidden_state = self._initial_latent()
        returns = []
        total_steps = 0
        num_rollouts = max(1, self.args.num_steps_per_eval // self.args.max_path_length)
        for _ in range(num_rollouts):
            obs = self.env.reset()
            policy_state = self._policy_state(obs, 1.0)
            rollout_return = 0.0
            for step in range(self.args.max_path_length):
                latent_for_policy = utl.get_latent_for_policy(
                    self.args,
                    *(forced_latent if forced_latent is not None else (latent_sample, latent_mean, latent_logvar))
                )
                with torch.no_grad():
                    _, action = self.policy.act(
                        state=policy_state,
                        latent=latent_for_policy,
                        belief=None,
                        task=None,
                        deterministic=deterministic,
                    )
                next_obs, reward, done, _ = self.env.step(ptu.get_numpy(action[0]))
                episode_done = bool(done) or step + 1 >= self.args.max_path_length
                encoder_next_state = self._encoder_state(next_obs, float(episode_done))
                reward_tensor = self._reward_to_tensor(reward)
                latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(
                    self.vae.encoder, encoder_next_state, action, reward_tensor, hidden_state)
                rollout_return += float(reward)
                total_steps += 1
                policy_state = self._policy_state(next_obs, 0.0)
                if episode_done:
                    break
            returns.append(rollout_return)
        return returns, total_steps

    def _evaluate_with_prefix(self, prefix, task_indices=None, forced_latents=None):
        stats = OrderedDict()
        all_final_returns = []
        all_online_returns = []
        task_indices = self.eval_tasks if task_indices is None else list(task_indices)

        for task_pos, task_idx in enumerate(task_indices):
            task_returns = []
            forced_latent = None if forced_latents is None else forced_latents[task_pos]
            for _ in range(self.args.num_evals):
                returns, _ = self._eval_task(
                    task_idx,
                    deterministic=self.args.eval_deterministic,
                    forced_latent=forced_latent,
                )
                task_returns.append(returns)
            min_len = min(len(r) for r in task_returns)
            task_returns = np.array([r[:min_len] for r in task_returns], dtype=np.float32)
            all_online_returns.append(task_returns.mean(axis=0))
            all_final_returns.append(task_returns[:, -1].mean())

        if all_online_returns:
            min_len = min(len(r) for r in all_online_returns)
            online = np.stack([r[:min_len] for r in all_online_returns]).mean(axis=0)
            for i, value in enumerate(online):
                stats[prefix + '/online_return_rollout_{}'.format(i)] = float(value)
            stats[prefix + '/online_return_mean'] = float(np.mean(online))
            stats[prefix + '/final_return'] = float(np.mean(all_final_returns))
        else:
            stats[prefix + '/online_return_mean'] = 0.0
            stats[prefix + '/final_return'] = 0.0
        return stats

    def evaluate(self, epoch):
        del epoch
        stats = self._evaluate_with_prefix('eval', task_indices=self.eval_tasks)

        train_eval_tasks = self.train_tasks
        if len(self.train_tasks) > len(self.eval_tasks):
            train_eval_tasks = np.random.choice(
                self.train_tasks, len(self.eval_tasks), replace=False)
        train_stats = self._evaluate_with_prefix(
            'eval_train', task_indices=train_eval_tasks)
        stats.update(train_stats)

        # PEARL/Flow-compatible aliases for shared W&B charts.
        stats['AverageTrainReturn_all_train_tasks'] = train_stats.get(
            'eval_train/final_return', 0.0)
        stats['AverageReturn_all_train_tasks'] = train_stats.get(
            'eval_train/final_return', 0.0)
        stats['AverageReturn_all_test_tasks'] = stats.get(
            'eval/final_return', 0.0)
        stats['AverageOnlineReturn_all_train_tasks'] = train_stats.get(
            'eval_train/online_return_mean', 0.0)
        stats['AverageOnlineReturn_all_test_tasks'] = stats.get(
            'eval/online_return_mean', 0.0)

        if self.args.eval_shuffled_latent and len(self.eval_tasks) > 1:
            latent_bank = [
                self._collect_eval_final_latent(task_idx, deterministic=self.args.eval_deterministic)
                for task_idx in self.eval_tasks
            ]
            shuffled_latents = latent_bank[1:] + latent_bank[:1]
            stats.update(self._evaluate_with_prefix(
                'eval_shuffled_latent',
                task_indices=self.eval_tasks,
                forced_latents=shuffled_latents))
        return stats

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self.policy_net.state_dict(),
            encoder=self.vae.encoder.state_dict(),
            reward_decoder=None if self.vae.reward_decoder is None else self.vae.reward_decoder.state_dict(),
            state_decoder=None if self.vae.state_decoder is None else self.vae.state_decoder.state_dict(),
        )
