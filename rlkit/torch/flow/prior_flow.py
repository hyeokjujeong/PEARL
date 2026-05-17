import math

import torch
import torch.nn as nn

import rlkit.torch.pytorch_util as ptu


class PriorFlow(nn.Module):
    """Unconditional flow network v_phi(c_tau, tau) for the marginal p(c).

    Paper Sec 3.2 / Eq. 5: the score-space PoE composition subtracts (t-1) copies
    of the prior score s_phi(c_tau, tau). A hard-coded N(0,I) prior would bias
    the fusion whenever the true context distribution is non-Gaussian or
    multimodal; this module learns the prior from bootstrap pseudo-targets {c_hat}
    via the standard CFM objective.

    Architecture matches the conditional encoder's vnet but with no context input:
    inputs are [c_tau, tau_emb] -> velocity.
    """

    def __init__(self, latent_dim, hidden_dim=128, tau_dim=16, tau_eps=0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.tau_dim = tau_dim
        self.tau_eps = tau_eps
        in_dim = latent_dim + tau_dim
        self.vnet = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _tau_embed(self, tau):
        ''' sinusoidal embedding of scalar tau in [0,1]; matches FlowContextEncoder '''
        half = self.tau_dim // 2
        freqs = torch.arange(1, half + 1, dtype=torch.float32, device=ptu.device)
        ang = 2.0 * math.pi * tau * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)])  # (tau_dim,)

    def velocity(self, c, tau):
        ''' v_phi(c, tau); c shape (n, latent), scalar tau -> (n, latent) '''
        n = c.shape[0]
        tau_emb = self._tau_embed(tau).view(1, -1).expand(n, -1)
        return self.vnet(torch.cat([c, tau_emb], dim=-1))

    def score(self, c, tau):
        ''' s_phi(c, tau) via the OT marginal velocity<->score relation.

        Correct OT-interpolant formula: s = -(c - tau v) / (1-tau).
        (Paper draft Eq. 3 has (1-tau)^2 in the denominator -- that's the
        conditional-Gaussian score; Tweedie marginalization cancels one
        factor of (1-tau), see flow_encoder.py for the derivation.)
        '''
        one_minus = max(1.0 - tau, self.tau_eps)
        v = self.velocity(c, tau)
        return -(c - tau * v) / one_minus

    def cfm_loss(self, c_targets):
        ''' standard CFM: regress v_phi toward (c1 - c0) on the marginal of c_hat. '''
        c1 = c_targets.detach()                              # bootstrap targets
        n = c1.shape[0]
        c0 = ptu.randn(n, self.latent_dim)
        tau = self.tau_eps + (1.0 - 2.0 * self.tau_eps) * float(torch.rand(1))
        c_tau = (1.0 - tau) * c0 + tau * c1
        v = self.velocity(c_tau, tau)
        target = c1 - c0
        return ((v - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, n, n_ode_steps=5):
        ''' draw n samples c ~ p(c) by Euler-integrating v_phi from N(0,I). '''
        c = ptu.randn(n, self.latent_dim)
        taus = torch.linspace(self.tau_eps, 1.0 - self.tau_eps, n_ode_steps + 1)
        for i in range(n_ode_steps):
            tau = float(taus[i])
            dt = float(taus[i + 1] - taus[i])
            c = c + dt * self.velocity(c, tau)
        return c
