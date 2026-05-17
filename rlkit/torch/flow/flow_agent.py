import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.agent import PEARLAgent


class FlowPEARLAgent(PEARLAgent):
    """PEARL agent with a flow-matching context encoder + a grounding decoder.

    The inferred latent task variable c is stored in ``self.z`` — the same
    buffer the Gaussian PEARL uses — so the policy and Q-networks (which only
    read ``self.z``) need no changes.

    Overrides vs PEARLAgent (see plan: the base methods assume a Gaussian
    posterior with z_means/z_vars and would crash otherwise):
      - infer_posterior : run the flow encoder, cache c in self.z
      - sample_z        : no-op (c is already produced by infer_posterior)
      - clear_z         : reset c to a prior sample (no z_means/z_vars)
      - log_diagnostics : log c_norm / c_variance (no z_means/z_vars)
      - networks        : include the decoder (so .to()/snapshot see it)
    """

    def __init__(self, latent_dim, context_encoder, policy, decoder,
                 prior_flow=None, **kwargs):
        super().__init__(latent_dim, context_encoder, policy, **kwargs)
        self.decoder = decoder
        # Optional learned prior v_phi (paper Eq. 5 / Sec 3.2). Held here so it
        # appears in `.networks` (moved to device, included in snapshots) and
        # so `clear_z` can sample c ~ p(c) from the learned prior instead of
        # the N(0,I) fallback.
        self.prior_flow = prior_flow

    def clear_z(self, num_tasks=1):
        ''' reset c to a prior sample; uses the learned v_phi when present. '''
        # getattr with default: PEARLAgent.__init__ calls clear_z() BEFORE
        # FlowPEARLAgent.__init__ has set self.prior_flow.
        prior_flow = getattr(self, 'prior_flow', None)
        if prior_flow is not None:
            self.z = prior_flow.sample(num_tasks)
        else:
            self.z = ptu.randn(num_tasks, self.latent_dim)
        self.context = None
        self.context_encoder.reset(num_tasks)

    def infer_posterior(self, context):
        ''' run the flow encoder and cache the inferred c in self.z '''
        self.z = self.context_encoder(context)

    def sample_z(self):
        # c is produced (and cached) by infer_posterior. PEARLAgent.forward
        # calls sample_z right after infer_posterior — must be a no-op here, or
        # it would desync c from the value the decoder/loss saw.
        pass

    def log_diagnostics(self, eval_statistics):
        # Called at eval time with a SINGLE task's c (self.z shape (1, latent)).
        # Variance across tasks is meaningless here, so only log the norm, and
        # under an eval-specific key — must NOT clobber 'c_variance', which is
        # the across-tasks training metric set by FlowPEARLSoftActorCritic.
        z = ptu.get_numpy(self.z)
        eval_statistics['c_norm eval'] = float(np.mean(np.linalg.norm(z, axis=-1)))

    @property
    def networks(self):
        nets = [self.context_encoder, self.policy, self.decoder]
        if self.prior_flow is not None:
            nets.append(self.prior_flow)
        return nets
