import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.varibad.decoder import RewardDecoder, StateTransitionDecoder
from rlkit.torch.varibad.encoder import RNNEncoder
from rlkit.torch.varibad.rollout_storage_vae import RolloutStorageVAE


class VaribadVAE(nn.Module):
    """Reward-first VariBAD VAE adapted for PEARL environments."""

    def __init__(self, args):
        super(VaribadVAE, self).__init__()
        self.args = args
        self.last_stats = {
            'vae/loss': 0.0,
            'vae/reward_recon_loss': 0.0,
            'vae/state_recon_loss': 0.0,
            'vae/kl_loss': 0.0,
        }

        self.encoder = RNNEncoder(
            layers_before_gru=args.encoder_layers_before_gru,
            hidden_size=args.encoder_gru_hidden_size,
            layers_after_gru=args.encoder_layers_after_gru,
            latent_dim=args.latent_dim,
            action_dim=args.action_dim,
            action_embed_dim=args.action_embedding_size,
            state_dim=args.state_dim,
            state_embed_dim=args.state_embedding_size,
            reward_size=1,
            reward_embed_size=args.reward_embedding_size,
        )

        latent_decoder_dim = args.latent_dim * 2 if args.disable_stochasticity_in_latent else args.latent_dim
        self.reward_decoder = RewardDecoder(
            layers=args.reward_decoder_layers,
            latent_dim=latent_decoder_dim,
            state_dim=args.state_dim,
            state_embed_dim=args.state_embedding_size,
            action_dim=args.action_dim,
            action_embed_dim=args.action_embedding_size,
            pred_type=args.rew_pred_type,
            input_prev_state=args.input_prev_state,
            input_action=args.input_action,
        ) if args.decode_reward and not args.disable_decoder else None

        self.state_decoder = StateTransitionDecoder(
            layers=args.state_decoder_layers,
            latent_dim=latent_decoder_dim,
            action_dim=args.action_dim,
            action_embed_dim=args.action_embedding_size,
            state_dim=args.state_dim,
            state_embed_dim=args.state_embedding_size,
            pred_type=args.state_pred_type,
        ) if args.decode_state and not args.disable_decoder else None

        self.rollout_storage = RolloutStorageVAE(
            num_processes=args.num_processes,
            max_trajectory_len=args.max_trajectory_len,
            max_num_rollouts=args.size_vae_buffer,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            vae_buffer_add_thresh=args.vae_buffer_add_thresh,
        )

        params = list(self.encoder.parameters())
        if self.reward_decoder is not None:
            params.extend(self.reward_decoder.parameters())
        if self.state_decoder is not None:
            params.extend(self.state_decoder.parameters())
        self.optimiser_vae = torch.optim.Adam(params, lr=args.lr_vae)

    def to(self, *args, **kwargs):
        module = super(VaribadVAE, self).to(*args, **kwargs)
        self.rollout_storage.to_device(next(self.parameters()).device)
        return module

    def _latent_for_decoder(self, latent_mean, latent_logvar):
        if self.args.disable_stochasticity_in_latent:
            return torch.cat((latent_mean, latent_logvar), dim=-1)
        return self.encoder._sample_gaussian(latent_mean, latent_logvar)

    def compute_rew_reconstruction_loss(self, latent, prev_obs, next_obs, action, reward):
        rew_pred = self.reward_decoder(latent, next_obs, prev_obs, action.float())
        if self.args.rew_pred_type == 'deterministic':
            return (rew_pred - reward).pow(2).mean(dim=-1)
        if self.args.rew_pred_type == 'gaussian':
            mean = rew_pred[..., :1]
            logvar = torch.clamp(rew_pred[..., 1:], -10, 10)
            return 0.5 * (logvar + (reward - mean).pow(2) / torch.exp(logvar)).mean(dim=-1)
        raise NotImplementedError('rew_pred_type {}'.format(self.args.rew_pred_type))

    def compute_state_reconstruction_loss(self, latent, prev_obs, next_obs, action):
        state_pred = self.state_decoder(latent, prev_obs, action.float())
        if self.args.state_pred_type == 'deterministic':
            return (state_pred - next_obs).pow(2).mean(dim=-1)
        if self.args.state_pred_type == 'gaussian':
            mean = state_pred[..., :next_obs.shape[-1]]
            logvar = torch.clamp(state_pred[..., next_obs.shape[-1]:], -10, 10)
            return 0.5 * (logvar + (next_obs - mean).pow(2) / torch.exp(logvar)).mean(dim=-1)
        raise NotImplementedError('state_pred_type {}'.format(self.args.state_pred_type))

    def compute_kl_loss(self, latent_mean, latent_logvar):
        if self.args.disable_kl_term:
            return latent_mean.new_tensor(0.0)
        if self.args.kl_to_gauss_prior:
            kl = -0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1)
            return kl.mean()

        gauss_dim = latent_mean.shape[-1]
        prior_mean = torch.zeros(1, *latent_mean.shape[1:], device=latent_mean.device)
        prior_logvar = torch.zeros(1, *latent_logvar.shape[1:], device=latent_logvar.device)
        all_means = torch.cat((prior_mean, latent_mean))
        all_logvars = torch.cat((prior_logvar, latent_logvar))
        mu = all_means[1:]
        prev_mu = all_means[:-1]
        log_e = all_logvars[1:]
        log_s = all_logvars[:-1]
        kl = 0.5 * (
            torch.sum(log_s, dim=-1) -
            torch.sum(log_e, dim=-1) -
            gauss_dim +
            torch.sum(torch.exp(log_e - log_s), dim=-1) +
            torch.sum((prev_mu - mu).pow(2) / torch.exp(log_s), dim=-1)
        )
        return kl.mean()

    def compute_vae_loss(self, update=False):
        if not self.rollout_storage.ready_for_update():
            return 0.0
        if self.args.disable_decoder and self.args.disable_kl_term:
            return 0.0

        device = next(self.parameters()).device
        prev_obs, next_obs, actions, rewards, trajectory_lens = self.rollout_storage.get_batch(
            batchsize=self.args.vae_batch_num_trajs, device=device)
        max_traj_len = int(np.max(trajectory_lens))
        prev_obs = prev_obs[:max_traj_len]
        next_obs = next_obs[:max_traj_len]
        actions = actions[:max_traj_len]
        rewards = rewards[:max_traj_len]

        _, latent_mean, latent_logvar, _ = self.encoder(
            actions=actions,
            states=next_obs,
            rewards=rewards,
            hidden_state=None,
            return_prior=True,
            sample=not self.args.disable_stochasticity_in_latent,
            detach_every=self.args.tbptt_stepsize,
        )

        transition_latent = self._latent_for_decoder(latent_mean[1:max_traj_len + 1], latent_logvar[1:max_traj_len + 1])
        rew_loss = transition_latent.new_tensor(0.0)
        state_loss = transition_latent.new_tensor(0.0)

        if self.reward_decoder is not None:
            rew_loss = self.compute_rew_reconstruction_loss(
                transition_latent, prev_obs, next_obs, actions, rewards).mean()
        if self.state_decoder is not None:
            state_loss = self.compute_state_reconstruction_loss(
                transition_latent, prev_obs, next_obs, actions).mean()
        kl_loss = self.compute_kl_loss(latent_mean[1:max_traj_len + 1], latent_logvar[1:max_traj_len + 1])

        loss = (
            self.args.rew_loss_coeff * rew_loss +
            self.args.state_loss_coeff * state_loss +
            self.args.kl_weight * kl_loss
        )

        if update:
            self.optimiser_vae.zero_grad()
            loss.backward()
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                decoder_params = []
                if self.reward_decoder is not None:
                    decoder_params.extend(self.reward_decoder.parameters())
                if self.state_decoder is not None:
                    decoder_params.extend(self.state_decoder.parameters())
                if decoder_params:
                    nn.utils.clip_grad_norm_(decoder_params, self.args.decoder_max_grad_norm)
            self.optimiser_vae.step()

        self.last_stats = {
            'vae/loss': float(loss.detach().cpu().item()),
            'vae/reward_recon_loss': float(rew_loss.detach().cpu().item()),
            'vae/state_recon_loss': float(state_loss.detach().cpu().item()),
            'vae/kl_loss': float(kl_loss.detach().cpu().item()),
        }
        return self.last_stats['vae/loss']

