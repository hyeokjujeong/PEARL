import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.varibad.helpers import FeatureExtractor


class RNNEncoder(nn.Module):
    """GRU belief encoder used by VariBAD.

    The inputs follow the original VariBAD convention:
    actions/states/rewards are shaped [T, B, dim]. For one online step,
    [B, dim] is also accepted and treated as T=1.
    """

    def __init__(
            self,
            layers_before_gru=(),
            hidden_size=64,
            layers_after_gru=(),
            latent_dim=32,
            action_dim=2,
            action_embed_dim=10,
            state_dim=2,
            state_embed_dim=10,
            reward_size=1,
            reward_embed_size=5,
    ):
        super(RNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = FeatureExtractor(reward_size, reward_embed_size, F.relu)

        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList()
        for output_dim in layers_before_gru:
            self.fc_before_gru.append(nn.Linear(curr_input_dim, output_dim))
            curr_input_dim = output_dim

        self.gru = nn.GRU(input_size=curr_input_dim, hidden_size=hidden_size, num_layers=1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList()
        for output_dim in layers_after_gru:
            self.fc_after_gru.append(nn.Linear(curr_input_dim, output_dim))
            curr_input_dim = output_dim

        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

    def _sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def prior(self, batch_size, sample=True, device=None):
        device = device or next(self.parameters()).device
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), device=device)
        h = hidden_state
        for layer in self.fc_after_gru:
            h = F.relu(layer(h))
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        latent_sample = self._sample_gaussian(latent_mean, latent_logvar) if sample else latent_mean
        return latent_sample, latent_mean, latent_logvar, hidden_state

    def reset_hidden(self, hidden_state, done):
        if done.dim() == 1:
            done = done.view(1, -1, 1)
        elif done.dim() == 2:
            done = done.unsqueeze(0)
        return hidden_state * (1.0 - done.to(hidden_state.device))

    def forward(self, actions, states, rewards, hidden_state=None, return_prior=False,
                sample=True, detach_every=None):
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        if hidden_state is not None:
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(
                actions.shape[1], sample=sample, device=actions.device)
            hidden_state = prior_hidden_state

        h = torch.cat((
            self.action_encoder(torch.clamp(actions, -1.0, 1.0)),
            self.state_encoder(states),
            self.reward_encoder(rewards),
        ), dim=2)

        for layer in self.fc_before_gru:
            h = F.relu(layer(h))

        if detach_every is None:
            output, _ = self.gru(h, hidden_state)
        else:
            outputs = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_output, hidden_state = self.gru(h[i:i + detach_every], hidden_state)
                outputs.append(curr_output)
                hidden_state = hidden_state.detach()
            output = torch.cat(outputs, dim=0)

        gru_h = output
        for layer in self.fc_after_gru:
            gru_h = F.relu(layer(gru_h))

        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)
        latent_sample = self._sample_gaussian(latent_mean, latent_logvar) if sample else latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden_state, output))

        if latent_mean.shape[0] == 1:
            latent_sample = latent_sample[0]
            latent_mean = latent_mean[0]
            latent_logvar = latent_logvar[0]

        return latent_sample, latent_mean, latent_logvar, output

