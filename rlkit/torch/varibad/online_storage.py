import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rlkit.torch.varibad import helpers as utl


class OnlineStorage(object):
    def __init__(self, args, num_steps, num_processes, state_dim, belief_dim,
                 task_dim, action_space, hidden_size, latent_dim,
                 normalise_rewards):
        self.args = args
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0
        self.normalise_rewards = normalise_rewards

        self.prev_state = torch.zeros(num_steps + 1, num_processes, state_dim)
        self.next_state = torch.zeros(num_steps, num_processes, state_dim)
        self.latent_samples = []
        self.latent_mean = []
        self.latent_logvar = []
        self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)

        self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim) if args.pass_belief_to_policy else None
        self.tasks = torch.zeros(num_steps + 1, num_processes, task_dim) if args.pass_task_to_policy else None

        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        action_shape = 1 if action_space.__class__.__name__ == 'Discrete' else action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

    def to_device(self, device):
        for attr in [
            'prev_state', 'next_state', 'hidden_states', 'rewards_raw',
            'rewards_normalised', 'done', 'masks', 'bad_masks', 'actions',
            'value_preds', 'returns', 'action_log_probs',
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        self.latent_samples = [t.to(device) for t in self.latent_samples]
        self.latent_mean = [t.to(device) for t in self.latent_mean]
        self.latent_logvar = [t.to(device) for t in self.latent_logvar]
        if self.beliefs is not None:
            self.beliefs = self.beliefs.to(device)
        if self.tasks is not None:
            self.tasks = self.tasks.to(device)
    def insert(self, state, belief, task, actions, rewards_raw, rewards_normalised,
               value_preds, action_log_probs, masks, bad_masks, done, hidden_states,
               latent_sample, latent_mean, latent_logvar):
        self.prev_state[self.step + 1].copy_(state)
        self.next_state[self.step].copy_(state)
        if self.beliefs is not None:
            self.beliefs[self.step + 1].copy_(belief)
        if self.tasks is not None:
            self.tasks[self.step + 1].copy_(task)

        self.latent_samples.append(latent_sample.detach().clone())
        self.latent_mean.append(latent_mean.detach().clone())
        self.latent_logvar.append(latent_logvar.detach().clone())
        self.hidden_states[self.step + 1].copy_(hidden_states.detach())

        self.actions[self.step].copy_(actions.detach())
        self.action_log_probs[self.step].copy_(action_log_probs.detach())
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.done[self.step + 1].copy_(done)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.prev_state[0].copy_(self.prev_state[-1])
        if self.beliefs is not None:
            self.beliefs[0].copy_(self.beliefs[-1])
        if self.tasks is not None:
            self.tasks[0].copy_(self.tasks[-1])
        if self.latent_samples:
            last_sample = self.latent_samples[-1].detach().clone()
            last_mean = self.latent_mean[-1].detach().clone()
            last_logvar = self.latent_logvar[-1].detach().clone()
            self.latent_samples = [last_sample]
            self.latent_mean = [last_mean]
            self.latent_logvar = [last_logvar]
        self.hidden_states[0].copy_(self.hidden_states[-1])
        self.done[0].copy_(self.done[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.action_log_probs.zero_()
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau, use_proper_time_limits=True):
        rewards = self.rewards_normalised if self.normalise_rewards else self.rewards_raw
        self.value_preds[-1] = next_value
        if use_gae:
            gae = 0
            for step in reversed(range(rewards.size(0))):
                delta = rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                if use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(rewards.size(0))):
                value = self.returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]
                if use_proper_time_limits:
                    value = value * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
                self.returns[step] = value

    def before_update(self, policy):
        # Old PPO log-probabilities must be from the behaviour policy used
        # during rollout collection. Recomputing them here can change the
        # denominator after policy RMS statistics are updated.
        del policy
        if not torch.isfinite(self.action_log_probs).all():
            raise RuntimeError('Non-finite old action log-probability in PPO storage.')

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps
        if mini_batch_size is None:
            if batch_size < num_mini_batch:
                num_mini_batch = batch_size
            mini_batch_size = max(1, batch_size // num_mini_batch)

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        latent_sample = torch.stack(self.latent_samples[:-1]).reshape(batch_size, -1)
        latent_mean = torch.stack(self.latent_mean[:-1]).reshape(batch_size, -1)
        latent_logvar = torch.stack(self.latent_logvar[:-1]).reshape(batch_size, -1)
        for indices in sampler:
            state_batch = self.prev_state[:-1].reshape(batch_size, -1)[indices]
            belief_batch = self.beliefs[:-1].reshape(batch_size, -1)[indices] if self.beliefs is not None else None
            task_batch = self.tasks[:-1].reshape(batch_size, -1)[indices] if self.tasks is not None else None
            yield (
                state_batch,
                belief_batch,
                task_batch,
                self.actions.reshape(batch_size, -1)[indices],
                latent_sample[indices],
                latent_mean[indices],
                latent_logvar[indices],
                self.value_preds[:-1].reshape(batch_size, 1)[indices],
                self.returns[:-1].reshape(batch_size, 1)[indices],
                self.action_log_probs.reshape(batch_size, 1)[indices],
                advantages.reshape(batch_size, 1)[indices],
            )
