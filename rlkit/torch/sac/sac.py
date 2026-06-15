from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            use_discrete_sac_actor=False,
            discrete_sac_action_dim=4,
            discrete_sac_temperature=1.0,
            discrete_sac_entropy_coeff=1.0,
            discrete_sac_use_obs_action_mask=True,
            q_target_clip=None,
            grad_clip_norm=None,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.use_discrete_sac_actor = use_discrete_sac_actor
        self.discrete_sac_action_dim = int(discrete_sac_action_dim)
        self.discrete_sac_temperature = float(discrete_sac_temperature)
        self.discrete_sac_entropy_coeff = float(discrete_sac_entropy_coeff)
        self.discrete_sac_use_obs_action_mask = discrete_sac_use_obs_action_mask
        self.q_target_clip = q_target_clip
        self.grad_clip_norm = grad_clip_norm

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _clip_gradients(self, module):
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip_norm)

    def _discrete_action_mask_from_obs(self, obs):
        mask = obs[:, -self.discrete_sac_action_dim:]
        mask = (mask > 0.5).float()
        no_valid = mask.sum(dim=-1, keepdim=True) < 0.5
        return torch.where(no_valid, torch.ones_like(mask), mask)

    def _masked_discrete_policy(self, obs, action_logits):
        logits = action_logits[:, :self.discrete_sac_action_dim]
        logits = logits / max(self.discrete_sac_temperature, 1e-6)
        if self.discrete_sac_use_obs_action_mask:
            mask = self._discrete_action_mask_from_obs(obs)
            logits = logits.masked_fill(mask <= 0.0, -1e9)
        else:
            mask = torch.ones_like(logits)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs) * mask
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        log_probs = torch.log(probs.clamp(min=1e-8))
        return probs, log_probs, mask

    def _min_q_all_discrete_actions(self, obs, task_z):
        batch_size = obs.size(0)
        action_dim = self.discrete_sac_action_dim
        eye = torch.eye(action_dim, device=obs.device, dtype=obs.dtype)
        action_grid = eye.unsqueeze(0).expand(batch_size, action_dim, action_dim)
        obs_grid = obs.unsqueeze(1).expand(batch_size, action_dim, obs.size(-1))
        z_grid = task_z.detach().unsqueeze(1).expand(batch_size, action_dim, task_z.size(-1))
        q1 = self.qf1(
            obs_grid.reshape(batch_size * action_dim, -1),
            action_grid.reshape(batch_size * action_dim, -1),
            z_grid.reshape(batch_size * action_dim, -1),
        ).view(batch_size, action_dim)
        q2 = self.qf2(
            obs_grid.reshape(batch_size * action_dim, -1),
            action_grid.reshape(batch_size * action_dim, -1),
            z_grid.reshape(batch_size * action_dim, -1),
        ).view(batch_size, action_dim)
        return torch.min(q1, q2)

    def _policy_value_terms(self, obs, new_actions, log_pi, task_z):
        if not self.use_discrete_sac_actor:
            min_q_new_actions = self._min_q(obs, new_actions, task_z)
            return min_q_new_actions, min_q_new_actions, log_pi, {}

        probs, log_probs, mask = self._masked_discrete_policy(obs, new_actions)
        q_values = self._min_q_all_discrete_actions(obs, task_z)
        expected_q = (probs * q_values).sum(dim=-1, keepdim=True)
        expected_q_for_policy = (probs * q_values.detach()).sum(dim=-1, keepdim=True)
        expected_log_pi = (probs * log_probs).sum(dim=-1, keepdim=True)
        entropy_term = self.discrete_sac_entropy_coeff * expected_log_pi
        entropy = -expected_log_pi
        stats = {
            'Discrete Policy Entropy': entropy.mean(),
            'Discrete Expected Log Pi': expected_log_pi.mean(),
            'Discrete Entropy Coeff': torch.tensor(
                self.discrete_sac_entropy_coeff,
                device=obs.device,
                dtype=obs.dtype,
            ),
            'Discrete Policy Max Prob': probs.max(dim=-1)[0].mean(),
            'Discrete Valid Action Count': mask.sum(dim=-1).mean(),
            'Discrete Q Mean': q_values.mean(),
            'Discrete Q Max': q_values.max(),
            'Discrete Q Min': q_values.min(),
        }
        return expected_q, expected_q_for_policy, entropy_term, stats

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        if self.q_target_clip is not None:
            q_target = torch.clamp(q_target, -self.q_target_clip, self.q_target_clip)
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self._clip_gradients(self.qf1)
        self._clip_gradients(self.qf2)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        min_q_new_actions, policy_q_new_actions, actor_log_pi, actor_stats = self._policy_value_terms(
            obs, new_actions, log_pi, task_z)

        # vf update
        v_target = min_q_new_actions - actor_log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self._clip_gradients(self.vf)
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = policy_q_new_actions

        policy_loss = (
                actor_log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self._clip_gradients(self.agent.policy)
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(actor_log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            for k, v in actor_stats.items():
                self.eval_statistics[k] = float(ptu.get_numpy(v))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot
