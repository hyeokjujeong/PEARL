import numpy as np
import torch
import torch.nn as nn

from rlkit.torch.varibad import helpers as utl


class VariBADPolicy(nn.Module):
    """Actor-critic policy used by the PPO branch of VariBAD."""

    def __init__(
            self,
            args,
            pass_state_to_policy,
            pass_latent_to_policy,
            pass_belief_to_policy,
            pass_task_to_policy,
            dim_state,
            dim_latent,
            dim_belief,
            dim_task,
            hidden_layers,
            activation_function,
            policy_initialisation,
            action_space,
            init_std,
    ):
        super(VariBADPolicy, self).__init__()
        self.args = args
        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy
        self.pass_belief_to_policy = pass_belief_to_policy
        self.pass_task_to_policy = pass_task_to_policy

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
            gain_name = 'tanh'
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
            gain_name = 'relu'
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
            gain_name = 'leaky_relu'
        else:
            raise ValueError('Unknown activation {}'.format(activation_function))

        if policy_initialisation == 'normc':
            init_ = lambda m: utl.init(
                m, utl.init_normc_, lambda x: nn.init.constant_(x, 0),
                nn.init.calculate_gain(gain_name))
        elif policy_initialisation == 'orthogonal':
            init_ = lambda m: utl.init(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                nn.init.calculate_gain(gain_name))
        else:
            raise ValueError('Unknown policy_initialisation {}'.format(policy_initialisation))

        self.norm_state = args.norm_state_for_policy and dim_state is not None
        self.norm_latent = args.norm_latent_for_policy and dim_latent is not None
        self.norm_belief = args.norm_belief_for_policy and dim_belief is not None
        self.norm_task = args.norm_task_for_policy and dim_task is not None
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms = utl.RunningMeanStd(shape=(dim_state,))
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = utl.RunningMeanStd(shape=(dim_latent,))
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms = utl.RunningMeanStd(shape=(dim_belief,))
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms = utl.RunningMeanStd(shape=(dim_task,))

        curr_input_dim = 0
        curr_input_dim += dim_state * int(self.pass_state_to_policy)
        curr_input_dim += dim_latent * int(self.pass_latent_to_policy)
        curr_input_dim += dim_belief * int(self.pass_belief_to_policy)
        curr_input_dim += dim_task * int(self.pass_task_to_policy)

        hidden_layers = [int(h) for h in hidden_layers]
        if not hidden_layers:
            hidden_layers = [64, 64]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        actor_input_dim = curr_input_dim
        critic_input_dim = curr_input_dim
        for hidden_dim in hidden_layers:
            self.actor_layers.append(init_(nn.Linear(actor_input_dim, hidden_dim)))
            self.critic_layers.append(init_(nn.Linear(critic_input_dim, hidden_dim)))
            actor_input_dim = hidden_dim
            critic_input_dim = hidden_dim
        self.critic_linear = nn.Linear(hidden_layers[-1], 1)

        if action_space.__class__.__name__ == 'Discrete':
            self.dist = Categorical(hidden_layers[-1], action_space.n)
        elif action_space.__class__.__name__ == 'Box':
            self.dist = DiagGaussian(
                hidden_layers[-1],
                action_space.shape[0],
                init_std,
                norm_actions_pre_sampling=args.norm_actions_pre_sampling,
                norm_actions_post_sampling=args.norm_actions_post_sampling,
                logstd_min=args.policy_logstd_min,
                logstd_max=args.policy_logstd_max,
                mean_clip=args.policy_mean_clip,
            )
        else:
            raise NotImplementedError('Unsupported action space {}'.format(action_space))

    def to(self, *args, **kwargs):
        module = super(VariBADPolicy, self).to(*args, **kwargs)
        device = next(self.parameters()).device
        for attr in ['state_rms', 'latent_rms', 'belief_rms', 'task_rms']:
            if hasattr(self, attr):
                getattr(self, attr).to(device)
        return module

    def _build_inputs(self, state, latent, belief, task):
        inputs = []
        if self.pass_state_to_policy:
            if self.norm_state:
                state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
            inputs.append(state)
        if self.pass_latent_to_policy:
            if self.norm_latent:
                latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            inputs.append(latent)
        if self.pass_belief_to_policy:
            if self.norm_belief:
                belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
            inputs.append(belief.float())
        if self.pass_task_to_policy:
            if self.norm_task:
                task = (task - self.task_rms.mean) / torch.sqrt(self.task_rms.var + 1e-8)
            inputs.append(task.float())
        return torch.cat(inputs, dim=-1)

    def _forward_layers(self, inputs, layers):
        h = inputs
        for layer in layers:
            h = self.activation_function(layer(h))
        return h

    def forward(self, state, latent, belief=None, task=None):
        inputs = self._build_inputs(state, latent, belief, task)
        hidden_critic = self._forward_layers(inputs, self.critic_layers)
        hidden_actor = self._forward_layers(inputs, self.actor_layers)
        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, state, latent, belief=None, task=None, deterministic=False,
            return_log_probs=False):
        value, actor_features = self.forward(state, latent, belief, task)
        dist = self.dist(actor_features)
        action = dist.mode() if deterministic else dist.sample()
        if return_log_probs:
            return value, action, dist.log_probs(action)
        return value, action

    def get_value(self, state, latent, belief=None, task=None):
        value, _ = self.forward(state, latent, belief, task)
        return value

    def evaluate_actions(self, state, latent, belief, task, action):
        value, actor_features = self.forward(state, latent, belief, task)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

    def update_rms(self, args, policy_storage):
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms.update(policy_storage.prev_state[:-1])
        if self.pass_latent_to_policy and self.norm_latent:
            latent = utl.get_latent_for_policy(
                args,
                latent_sample=torch.stack(policy_storage.latent_samples[:-1]),
                latent_mean=torch.stack(policy_storage.latent_mean[:-1]),
                latent_logvar=torch.stack(policy_storage.latent_logvar[:-1]),
            )
            self.latent_rms.update(latent)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms.update(policy_storage.beliefs[:-1])
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms.update(policy_storage.tasks[:-1])


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: utl.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        return FixedCategorical(logits=self.linear(x))


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super(FixedCategorical, self).sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super(FixedCategorical, self).log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std,
                 norm_actions_pre_sampling=False, norm_actions_post_sampling=True,
                 logstd_min=-5.0, logstd_max=1.0, mean_clip=10.0):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: utl.init(m, utl.init_normc_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(torch.full((num_outputs,), float(np.log(init_std))))
        self.norm_actions_pre_sampling = norm_actions_pre_sampling
        self.norm_actions_post_sampling = norm_actions_post_sampling
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        self.mean_clip = mean_clip

    def forward(self, x):
        action_mean = self.fc_mean(x)
        if self.norm_actions_pre_sampling:
            action_mean = torch.tanh(action_mean)
        elif self.mean_clip is not None:
            action_mean = torch.clamp(action_mean, -self.mean_clip, self.mean_clip)

        if not torch.isfinite(action_mean).all():
            raise RuntimeError(
                'Non-finite VariBAD action mean. '
                'input_finite={}'.format(bool(torch.isfinite(x).all().item()))
            )

        logstd = torch.clamp(self.logstd, self.logstd_min, self.logstd_max)
        std = torch.clamp(logstd.exp(), min=1e-6)
        if not torch.isfinite(std).all():
            raise RuntimeError('Non-finite VariBAD action std.')
        base_dist = torch.distributions.Normal(action_mean, std)
        return FixedNormal(base_dist, self.norm_actions_post_sampling)


class FixedNormal(object):
    def __init__(self, base_dist, squash):
        self.base_dist = base_dist
        self.squash = squash

    def sample(self):
        raw = self.base_dist.rsample()
        if self.squash:
            return torch.tanh(raw)
        return torch.clamp(raw, -1.0, 1.0)

    def mode(self):
        if self.squash:
            return torch.tanh(self.base_dist.mean)
        return torch.clamp(self.base_dist.mean, -1.0, 1.0)

    def log_probs(self, actions):
        if self.squash:
            eps = 1e-6
            clipped = torch.clamp(actions, -1 + eps, 1 - eps)
            raw = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
            log_probs = self.base_dist.log_prob(raw) - torch.log(1 - clipped.pow(2) + eps)
        else:
            log_probs = self.base_dist.log_prob(actions)
        return log_probs.sum(-1, keepdim=True)

    def entropy(self):
        return self.base_dist.entropy().sum(-1)
