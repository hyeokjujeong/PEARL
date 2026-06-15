import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlkit.torch.varibad import helpers as utl


class PPO(object):
    def __init__(self, args, actor_critic, value_loss_coef, entropy_coef,
                 policy_optimiser='adam', lr=7e-4, clip_param=0.2,
                 ppo_epoch=5, num_mini_batch=5, eps=1e-5,
                 use_huber_loss=True, use_clipped_value_loss=True):
        self.args = args
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)
        else:
            raise ValueError('Unknown policy optimiser {}'.format(policy_optimiser))

    def update(self, policy_storage, compute_vae_loss=None):
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        policy_storage.before_update(self.actor_critic)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        loss_epoch = 0.0
        raw_log_ratio_abs_max_epoch = 0.0
        ratio_mean_epoch = 0.0
        ratio_max_epoch = 0.0
        num_updates = 0
        log_ratio_clip = getattr(self.args, 'ppo_log_ratio_clip', 20.0)

        for _ in range(self.ppo_epoch):
            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                state_batch, belief_batch, task_batch, actions_batch, latent_sample_batch, \
                    latent_mean_batch, latent_logvar_batch, value_preds_batch, return_batch, \
                    old_action_log_probs_batch, adv_targ = sample

                latent_batch = utl.get_latent_for_policy(
                    self.args, latent_sample_batch, latent_mean_batch, latent_logvar_batch)
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    state=state_batch,
                    latent=latent_batch,
                    belief=belief_batch,
                    task=task_batch,
                    action=actions_batch,
                )

                raw_log_ratio = action_log_probs - old_action_log_probs_batch
                if not torch.isfinite(raw_log_ratio).all():
                    raise RuntimeError(
                        'Non-finite PPO log-ratio. '
                        'new_log_prob_finite={} old_log_prob_finite={}'.format(
                            bool(torch.isfinite(action_log_probs).all().item()),
                            bool(torch.isfinite(old_action_log_probs_batch).all().item()),
                        )
                    )
                log_ratio = torch.clamp(raw_log_ratio, -log_ratio_clip, log_ratio_clip)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.clip_param, self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.clip_param, self.clip_param)
                    value_loss = 0.5 * torch.max(
                        (values - return_batch).pow(2),
                        (value_pred_clipped - return_batch).pow(2)).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        'Non-finite PPO loss. '
                        'value_loss={} action_loss={} entropy={} '
                        'raw_log_ratio_abs_max={} ratio_max={}'.format(
                            value_loss.detach().cpu().item(),
                            action_loss.detach().cpu().item(),
                            dist_entropy.detach().cpu().item(),
                            raw_log_ratio.detach().abs().max().cpu().item(),
                            ratio.detach().max().cpu().item(),
                        )
                    )
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
                self.optimiser.step()
                self._assert_finite_parameters()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()
                raw_log_ratio_abs_max_epoch += raw_log_ratio.detach().abs().max().cpu().item()
                ratio_mean_epoch += ratio.detach().mean().cpu().item()
                ratio_max_epoch += ratio.detach().max().cpu().item()
                num_updates += 1

        self.actor_critic.update_rms(self.args, policy_storage)

        vae_loss = 0.0
        if compute_vae_loss is not None:
            for _ in range(self.args.num_vae_updates):
                vae_loss = compute_vae_loss(update=True)

        normalizer = max(1, num_updates)
        return {
            'ppo/value_loss': value_loss_epoch / normalizer,
            'ppo/action_loss': action_loss_epoch / normalizer,
            'ppo/dist_entropy': dist_entropy_epoch / normalizer,
            'ppo/loss': loss_epoch / normalizer,
            'ppo/raw_log_ratio_abs_max': raw_log_ratio_abs_max_epoch / normalizer,
            'ppo/ratio_mean': ratio_mean_epoch / normalizer,
            'ppo/ratio_max': ratio_max_epoch / normalizer,
            'vae/loss': float(vae_loss),
        }

    def act(self, state, latent, belief=None, task=None, deterministic=False,
            return_log_probs=False):
        return self.actor_critic.act(
            state, latent, belief, task,
            deterministic=deterministic,
            return_log_probs=return_log_probs,
        )

    def _assert_finite_parameters(self):
        for name, parameter in self.actor_critic.named_parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                raise RuntimeError('Non-finite PPO parameter after optimiser step: {}'.format(name))
