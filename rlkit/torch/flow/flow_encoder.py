import torch
import torch.nn as nn


class FlowContextEncoder(nn.Module):
    """Flow-matching context encoder — produces the latent task variable c.

    Stage 0: a STUB (mean-pool over the context transitions + an MLP head).
    The real conditional-flow per-transition experts + score composition +
    ODE integration are implemented in Stage 2.

    Interface contract (must match rlkit's MlpEncoder so it is a drop-in for
    PEARLAgent.context_encoder):
      - ``output_size`` attribute
      - ``reset(num_tasks)`` method  (PEARLAgent.clear_z calls encoder.reset)
    """

    def __init__(self, context_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.output_size = latent_dim
        # --- Stage 0 stub head; replaced by the conditional flow in Stage 2 ---
        self._stub = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def reset(self, num_tasks=1):
        # no recurrent / hidden state to reset
        pass

    def forward(self, context):
        """context: (num_tasks, seq_len, context_dim) -> c: (num_tasks, latent_dim)"""
        pooled = context.mean(dim=1)          # (num_tasks, context_dim)
        return self._stub(pooled)             # (num_tasks, latent_dim)
