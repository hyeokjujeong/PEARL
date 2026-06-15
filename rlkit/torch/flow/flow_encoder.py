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
        s = -(c - tau v) / (1-tau) ,   v = (c + s (1-tau)) / tau
    NOTE: paper draft Eq. 3 has (1-tau)^2; the correct marginal relation has
    (1-tau)^1. The conditional Gaussian score N(tau z1, (1-tau)^2 I) has
    (1-tau)^2 in the denominator, but Tweedie marginalization cancels one
    factor of (1-tau). Derivation: with E[z1|z_tau] = z + (1-tau) v,
        s = -(z - tau E[z1|z_tau]) / (1-tau)^2
          = -((1-tau)z - tau(1-tau)v) / (1-tau)^2
          = -(z - tau v) / (1-tau).

    ⚠️ KNOWN-APPROXIMATE / GUARDS (flagged for review):
      - Score composition along the interpolation path is NOT exact (noising
        does not commute with the product) — a PoE-inspired heuristic.
      - The conversions blow up at tau->0 and tau->1; integration is done on
        [tau_eps, 1-tau_eps].
      - The (t-1) prior-overcounting term is explosive for long histories, so
        the context is capped (max_context) and the per-step velocity norm is
        clamped (vel_clip) to avoid NaN. These are MVP guards, not a fix.

    ⚠️ DESIGN LIMITATION (review finding "C2", needs a decision):
      v_theta is trained ONLY by backprop through the ODE from the decoder
      reconstruction loss — there is no flow-matching / velocity-regression
      loss and (by an earlier decision) no KL term. With nothing rewarding
      stochasticity, training pressures the encoder to ignore the resampled
      N(0,I) base and collapse to a DETERMINISTIC c(context). The flow/score/
      ODE machinery then carries no probabilistic meaning — it acts as a
      fixed permutation-invariant aggregator, not a posterior sampler.
      Realizing a genuine probabilistic (multimodal) posterior needs either a
      conditional-flow-matching loss for v_theta or an entropy/IB term.

    Interface contract (drop-in for rlkit's MlpEncoder): ``output_size`` and
    ``reset(num_tasks)``.
    """

    def __init__(self, context_dim, latent_dim, hidden_dim=128,
                 n_ode_steps=5, max_context=16, tau_eps=0.05, vel_clip=10.0,
                 prior_flow=None):
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
        # Optional learned prior flow v_phi (paper Eq. 5). When None, the prior
        # score falls back to the N(0,I) closed form (-c), matching the original
        # MVP behaviour.
        # NOTE: kept as a plain attribute (not a submodule) so the prior's
        # parameters stay under the agent's `networks` list and aren't
        # double-registered / double-optimized via the encoder.
        object.__setattr__(self, 'prior_flow', prior_flow)
        # the (possibly subsampled) context of the most recent forward() —
        # cfm_loss must regress against the SAME context that produced c.
        self._subsampled_context = None

    def reset(self, num_tasks=1):
        # no recurrent / hidden state
        pass

    def _tau_embed(self, tau):
        ''' sinusoidal embedding of the scalar interpolation time tau in [0,1].
        Frequencies are 1..half cycles over the unit interval — bounded so the
        sin/cos arguments don't alias (a 2**k schedule reached ~400 rad). '''
        half = self.tau_dim // 2
        freqs = torch.arange(1, half + 1, dtype=torch.float32, device=ptu.device)
        ang = 2.0 * 3.141592653589793 * tau * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)])      # (tau_dim,)

    def _fused_velocity(self, c, tau, context):
        ''' composed velocity field at (c, tau) given the context transitions '''
        n, t, _ = context.shape
        c_bc = c.unsqueeze(1).expand(n, t, self.latent_dim)         # (n,t,latent)
        tau_emb = self._tau_embed(tau).view(1, 1, -1).expand(n, t, -1)
        v_per = self.vnet(torch.cat([c_bc, tau_emb, context], dim=-1))

        one_minus = max(1.0 - tau, self.tau_eps)
        tau_safe = max(tau, self.tau_eps)
        # per-transition expert scores, summed; minus the prior overcounting.
        # If a learned prior flow is attached, use its score; otherwise fall
        # back to the N(0,I) closed form (s_prior = -c).
        # OT marginal relation: s = -(c - tau v) / (1-tau); inverse:
        # v = (c + s (1-tau)) / tau. (Paper draft Eq. 3 has (1-tau)^2 -- typo.)
        s_per = -(c_bc - tau * v_per) / one_minus
        if self.prior_flow is not None:
            # Paper spec: phi is updated ONLY by L_prior, not by any loss that
            # flows through the encoder. Detach so encoder/decoder/fused-CFM
            # gradients do not leak into the prior flow's parameters.
            s_prior = self.prior_flow.score(c, tau).detach()
        else:
            s_prior = -c
        s_fused = s_per.sum(dim=1) - (t - 1) * s_prior
        v_fused = (c + s_fused * one_minus) / tau_safe

        # MVP numerical guard: smoothly squash the velocity norm with tanh.
        # A hard clamp would zero the gradient for every task above the
        # threshold (and the (t-1) term makes that frequent) — tanh keeps the
        # gradient alive everywhere.
        norm = v_fused.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        squashed = self.vel_clip * torch.tanh(norm / self.vel_clip)
        return v_fused * (squashed / norm)

    def forward(self, context):
        ''' context: (num_tasks, seq_len, context_dim) -> c: (num_tasks, latent) '''
        n, t, _ = context.shape
        # max_context=None -> use all transitions (paper-faithful). The cap is
        # an MVP guard against the (t-1)*s_prior term exploding under the
        # hard-coded Gaussian prior; with a learned v_phi the correction is
        # calibrated and the cap can usually be removed.
        if self.max_context is not None and t > self.max_context:
            idx = torch.randperm(t, device=context.device)[:self.max_context]
            context = context[:, idx, :]
        self._subsampled_context = context                 # for cfm_loss
        c = ptu.randn(n, self.latent_dim)                  # base sample c(tau_eps)
        taus = torch.linspace(self.tau_eps, 1.0 - self.tau_eps,
                              self.n_ode_steps + 1)
        for i in range(self.n_ode_steps):
            tau = float(taus[i])
            dt = float(taus[i + 1] - taus[i])
            c = c + dt * self._fused_velocity(c, tau, context)      # Euler step
        return c

    def cfm_loss(self, context, c1):
        """Bootstrap-EM conditional-flow-matching loss.

        Regresses the *composed* velocity field ``_fused_velocity`` (the exact
        quantity the ODE integrates) toward the OT target velocity (c1 - c0),
        so what CFM trains == what the ODE uses. c1 is the bootstrap target
        (sg[c]); the caller must pass the SAME subsampled context that produced
        it (``self._subsampled_context``).

        A single scalar tau is sampled per call (reuses the scalar-tau path of
        _tau_embed/_fused_velocity); the [0,1] interval is covered across steps.
        """
        n = context.shape[0]
        c0 = ptu.randn(n, self.latent_dim)
        tau = self.tau_eps + (1.0 - 2.0 * self.tau_eps) * float(torch.rand(1))
        c_tau = (1.0 - tau) * c0 + tau * c1
        v = self._fused_velocity(c_tau, tau, context)
        target = c1 - c0
        return ((v - target) ** 2).mean()

    def per_transition_cfm_loss(self, context, c1):
        """Ablation: per-transition CFM loss on v_theta(c_tau, tau | x_i).

        Regresses the *single-transition* velocity field (vnet), NOT the fused
        composition, against the OT target (c1 - c0) for each transition.
        The provided LCI paper's Eq. 6 instead regresses the composed fused
        velocity field, which is implemented by cfm_loss().

        c1 is the bootstrap target (sg[c_hat]); caller passes the same
        subsampled context that produced c_hat (`self._subsampled_context`).
        """
        n, t, _ = context.shape
        c0 = ptu.randn(n, self.latent_dim)
        tau = self.tau_eps + (1.0 - 2.0 * self.tau_eps) * float(torch.rand(1))
        c_tau = (1.0 - tau) * c0 + tau * c1.detach()                   # (n, latent)

        c_bc = c_tau.unsqueeze(1).expand(n, t, self.latent_dim)
        tau_emb = self._tau_embed(tau).view(1, 1, -1).expand(n, t, -1)
        v_per = self.vnet(torch.cat([c_bc, tau_emb, context], dim=-1))  # (n,t,latent)
        target = (c1.detach() - c0).unsqueeze(1).expand(n, t, self.latent_dim)
        return ((v_per - target) ** 2).mean()

    @torch.no_grad()
    def e_step_sample(self, context):
        """Paper E-step: fused-ODE pass under stop_grad to produce c_hat.

        Also caches the (subsampled) context as `_subsampled_context` so the
        M-step's per-transition CFM regresses against the SAME transitions
        that produced c_hat. Returns c_hat shape (num_tasks, latent_dim).
        """
        return self.forward(context)
