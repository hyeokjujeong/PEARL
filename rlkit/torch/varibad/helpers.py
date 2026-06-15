import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def action_dim(action_space):
    if action_space.__class__.__name__ == 'Discrete':
        return 1
    return int(np.prod(action_space.shape))


def squash_action(action, norm_actions_post_sampling=False):
    if norm_actions_post_sampling:
        return torch.tanh(action)
    return torch.clamp(action, -1.0, 1.0)


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):
    if latent_sample is None and latent_mean is None and latent_logvar is None:
        return None

    if getattr(args, 'add_nonlinearity_to_latent', False):
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if getattr(args, 'sample_embeddings', False):
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.dim() > 2 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    return latent


def update_encoding(encoder, next_obs, action, reward, hidden_state, sample=True):
    with torch.no_grad():
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder(
            actions=action.float(),
            states=next_obs.float(),
            rewards=reward.float(),
            hidden_state=hidden_state,
            return_prior=False,
            sample=sample,
        )
    return latent_sample, latent_mean, latent_logvar, hidden_state


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, activation_function=F.relu):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        self.fc = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, inputs):
        if self.fc is None:
            return inputs.new_zeros(*inputs.shape[:-1], 0)
        return self.activation_function(self.fc(inputs))


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float()
        self.var = torch.ones(shape).float()
        self.count = epsilon

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def update(self, x):
        x = x.detach()
        if x.numel() == 0:
            return
        x = x.reshape((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count
