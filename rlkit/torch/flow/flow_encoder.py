import torch
import torch.nn as nn

import rlkit.torch.pytorch_util as ptu


class FlowContextEncoder(nn.Module):
    """Flow-matching context encoder — produces the latent task variable c.

    Per the team proposal: a single conditional velocity field
    ``v_theta(c_tau, tau | x)`` defines a per-transition expert ``p(c | x_i)``;
    the experts are composed in SCORE space (product-of-experts) and a single
    ODE integration from the N(0,I) base produces the latent c.

    Composition (per the proposal):
        s_fused(c_tau, tau) = sum_i s_theta(c_tau, tau | x_i) - (t-1) s_prior
    with the velocity<->score relations of the OT interpolant
        s = -(c - tau v) / (1-tau)^2 ,   v = (c + s (1-tau)^2) / tau

    ⚠️ KNOWN-APPROXIMATE / GUARDS (flagged for review):
      - Score composition along the interpolation path is NOT exact (noising
        does not commute with the product) — a PoE-inspired heuristic.
      - The conversions blow up at tau->0 and tau->1; integration is done on
        [tau_eps, 1-tau_eps].
      - The (t-1) prior-overcounting term is explosive for long histories, so
        the context is capped (max_context) and the per-step velocity norm is
        clamped (vel_clip) to avoid NaN. These are MVP guards, not a fix.

    Interface contract (drop-in for rlkit's MlpEncoder): ``output_size`` and
    ``reset(num_tasks)``.
    """

    def __init__(self, context_dim, latent_dim, hidden_dim=128,
                 n_ode_steps=5, max_context=16, tau_eps=0.05, vel_clip=10.0):
        super().__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.output_size = latent_dim
        self.n_ode_steps = n_ode_steps
        self.max_context = max_context
        self.tau_eps = tau_eps
        self.vel_clip = vel_clip
        self.tau_dim = 16
        in_dim = latent_dim + self.tau_dim + context_dim
        self.vnet = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def reset(self, num_tasks=1):
        # no recurrent / hidden state
        pass

    def _tau_embed(self, tau):
        ''' sinusoidal embedding of the scalar interpolation time tau '''
        half = self.tau_dim // 2
        freqs = 2.0 ** torch.arange(half, dtype=torch.float32, device=ptu.device)
        ang = tau * freqs * 3.141592653589793
        return torch.cat([torch.sin(ang), torch.cos(ang)])      # (tau_dim,)

    def _fused_velocity(self, c, tau, context):
        ''' composed velocity field at (c, tau) given the context transitions '''
        n, t, _ = context.shape
        c_bc = c.unsqueeze(1).expand(n, t, self.latent_dim)         # (n,t,latent)
        tau_emb = self._tau_embed(tau).view(1, 1, -1).expand(n, t, -1)
        v_per = self.vnet(torch.cat([c_bc, tau_emb, context], dim=-1))

        one_minus = max(1.0 - tau, self.tau_eps)
        tau_safe = max(tau, self.tau_eps)
        # per-transition expert scores, summed; minus the prior overcounting
        s_per = -(c_bc - tau * v_per) / (one_minus ** 2)
        s_fused = s_per.sum(dim=1) - (t - 1) * (-c)                 # s_prior = -c
        v_fused = (c + s_fused * (one_minus ** 2)) / tau_safe

        # MVP numerical guard: clamp the velocity norm to avoid NaN explosion
        norm = v_fused.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return v_fused * (norm.clamp(max=self.vel_clip) / norm)

    def forward(self, context):
        ''' context: (num_tasks, seq_len, context_dim) -> c: (num_tasks, latent) '''
        n, t, _ = context.shape
        if t > self.max_context:
            idx = torch.randperm(t, device=context.device)[:self.max_context]
            context = context[:, idx, :]
        c = ptu.randn(n, self.latent_dim)                  # base sample c(tau_eps)
        taus = torch.linspace(self.tau_eps, 1.0 - self.tau_eps,
                              self.n_ode_steps + 1)
        for i in range(self.n_ode_steps):
            tau = float(taus[i])
            dt = float(taus[i + 1] - taus[i])
            c = c + dt * self._fused_velocity(c, tau, context)      # Euler step
        return c
