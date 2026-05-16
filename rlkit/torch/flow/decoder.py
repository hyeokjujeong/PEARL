import torch
import torch.nn as nn


class TransitionDecoder(nn.Module):
    """Predicts (next_state, reward) from (state, action, c).

    This is the grounding signal (VariBAD-style): c must encode the task or
    the decoder cannot predict transitions of different tasks, so a collapsed
    (constant) c is no longer a zero-loss solution.

    NOTE: for reward-varying tasks (e.g. point-robot, where dynamics
    s' = s + a do NOT depend on the task) the *reward* head carries all the
    grounding signal; the next-state head carries none.
    """

    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        in_dim = obs_dim + action_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim + 1),   # next_obs (obs_dim) + reward (1)
        )

    def forward(self, obs, actions, c):
        x = torch.cat([obs, actions, c], dim=-1)
        out = self.net(x)
        pred_next_obs = out[..., :self.obs_dim]
        pred_reward = out[..., self.obs_dim:]
        return pred_next_obs, pred_reward
