import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.varibad.helpers import FeatureExtractor


class RewardDecoder(nn.Module):
    def __init__(
            self,
            layers,
            latent_dim,
            action_dim,
            action_embed_dim,
            state_dim,
            state_embed_dim,
            pred_type='deterministic',
            input_prev_state=True,
            input_action=True,
    ):
        super(RewardDecoder, self).__init__()
        self.pred_type = pred_type
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu) if input_action else None

        curr_input_dim = latent_dim + state_embed_dim
        if input_prev_state:
            curr_input_dim += state_embed_dim
        if input_action:
            curr_input_dim += action_embed_dim

        self.fc_layers = nn.ModuleList()
        for output_dim in layers:
            self.fc_layers.append(nn.Linear(curr_input_dim, output_dim))
            curr_input_dim = output_dim
        self.fc_out = nn.Linear(curr_input_dim, 2 if pred_type == 'gaussian' else 1)

    def forward(self, latent_state, next_state, prev_state=None, actions=None):
        h = torch.cat((latent_state, self.state_encoder(next_state)), dim=-1)
        if self.input_action:
            h = torch.cat((h, self.action_encoder(torch.clamp(actions, -1.0, 1.0))), dim=-1)
        if self.input_prev_state:
            h = torch.cat((h, self.state_encoder(prev_state)), dim=-1)

        for layer in self.fc_layers:
            h = F.relu(layer(h))
        return self.fc_out(h)


class StateTransitionDecoder(nn.Module):
    def __init__(self, layers, latent_dim, action_dim, action_embed_dim,
                 state_dim, state_embed_dim, pred_type='deterministic'):
        super(StateTransitionDecoder, self).__init__()
        self.pred_type = pred_type
        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList()
        for output_dim in layers:
            self.fc_layers.append(nn.Linear(curr_input_dim, output_dim))
            curr_input_dim = output_dim
        self.fc_out = nn.Linear(curr_input_dim, 2 * state_dim if pred_type == 'gaussian' else state_dim)

    def forward(self, latent_state, state, actions):
        h = torch.cat((
            latent_state,
            self.state_encoder(state),
            self.action_encoder(torch.clamp(actions, -1.0, 1.0)),
        ), dim=-1)
        for layer in self.fc_layers:
            h = F.relu(layer(h))
        return self.fc_out(h)


class TaskDecoder(nn.Module):
    def __init__(self, layers, latent_dim, task_dim):
        super(TaskDecoder, self).__init__()
        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList()
        for output_dim in layers:
            self.fc_layers.append(nn.Linear(curr_input_dim, output_dim))
            curr_input_dim = output_dim
        self.fc_out = nn.Linear(curr_input_dim, task_dim)

    def forward(self, latent_state):
        h = latent_state
        for layer in self.fc_layers:
            h = F.relu(layer(h))
        return self.fc_out(h)

