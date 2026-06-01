"""Milestone figure: flow posterior is MULTIMODAL where PEARL's is not.

Feeds the SAME T-maze contexts to a trained PEARL (Gaussian + IB) encoder and a
trained flow encoder, draws K posterior samples from each, and plots them side
by side.

Key contrast:
  * PEARL: q(z|c) = product of per-transition Gaussians -> a SINGLE Gaussian,
    unimodal BY CONSTRUCTION. Cannot represent "could be task A or task B".
  * Flow:  q(c|x) via ODE from a resampled N(0,I) base -> can stay MULTIMODAL.

Ambiguous context (corridor-only, no oracle) is identical for both T-maze
tasks, so a faithful posterior should keep both task hypotheses alive.

Usage:
    python tools/compare_posterior_pearl_flow.py <flow_snap> <pearl_snap> \
        [--K 500] [--ctxlen 4]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlkit.torch.pytorch_util as ptu
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.flow.flow_encoder import FlowContextEncoder
from rlkit.torch.flow.prior_flow import PriorFlow
from rlkit.torch.networks import MlpEncoder
import probe_flow_posterior as P

TAU_DIM = 16


def load_flow(snap):
    sd = torch.load(os.path.join(snap, 'context_encoder.pth'), map_location='cpu')
    latent = sd['vnet.4.weight'].shape[0]
    hidden = sd['vnet.0.weight'].shape[0]
    cdim = sd['vnet.0.weight'].shape[1] - latent - TAU_DIM
    prior = None
    pp = os.path.join(snap, 'prior_flow.pth')
    if os.path.exists(pp):
        psd = torch.load(pp, map_location='cpu')
        prior = PriorFlow(latent_dim=latent, hidden_dim=psd['vnet.0.weight'].shape[0])
        prior.load_state_dict(psd)
        prior.eval()
    enc = FlowContextEncoder(context_dim=cdim, latent_dim=latent, hidden_dim=hidden,
                             n_ode_steps=5, max_context=None, tau_eps=0.05,
                             vel_clip=10.0, prior_flow=prior)
    enc.load_state_dict(sd)
    enc.eval()
    return enc, latent


def load_pearl(snap):
    sd = torch.load(os.path.join(snap, 'context_encoder.pth'), map_location='cpu')
    out_dim = sd['last_fc.weight'].shape[0]
    in_dim = sd['fc0.weight'].shape[1]
    latent = out_dim // 2                       # IB encoder: [mu | sigma]
    enc = MlpEncoder(hidden_sizes=[200, 200, 200], input_size=in_dim, output_size=out_dim)
    enc.load_state_dict(sd)
    enc.eval()
    return enc, latent


def pearl_posterior_samples(enc, latent, context_np, K):
    """Replicate PEARLAgent.infer_posterior: per-transition Gaussians combined
    by product-of-Gaussians into one (mu, var), then sample z ~ N(mu, var)."""
    ctx = torch.from_numpy(context_np).unsqueeze(0)          # (1, t, 7)
    with torch.no_grad():
        params = enc(ctx).view(1, -1, 2 * latent)
        mu = params[..., :latent][0]                         # (t, latent)
        ss = torch.clamp(F.softplus(params[..., latent:][0]), min=1e-7)
        var = 1.0 / torch.reciprocal(ss).sum(0)              # product of gaussians
        pmu = var * (mu / ss).sum(0)
    pmu, var = pmu.numpy(), var.numpy()
    return pmu + np.sqrt(var) * np.random.randn(K, latent), pmu, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('flow_snap')
    ap.add_argument('pearl_snap')
    ap.add_argument('--K', type=int, default=500)
    ap.add_argument('--ctxlen', type=int, default=4, help='# corridor (null) transitions in ambiguous context')
    ap.add_argument('--corridor', type=int, default=10)
    ap.add_argument('--out', default='posterior_pearl_vs_flow.png')
    args = ap.parse_args()

    ptu.set_gpu_mode(False)
    np.random.seed(0)

    flow_enc, flow_lat = load_flow(args.flow_snap)
    pearl_enc, pearl_lat = load_pearl(args.pearl_snap)
    print(f'[flow]  latent={flow_lat}   [pearl] latent={pearl_lat}')

    env = NormalizedBoxEnv(ENVS['tmaze-passive'](n_tasks=2, corridor_length=args.corridor))
    roll0 = P.scripted_rollout(env, 0, args.corridor)
    roll1 = P.scripted_rollout(env, 1, args.corridor)
    corridor_idx = list(range(1, args.corridor))[:args.ctxlen]

    ctxs = {
        'ambiguous (corridor only)': P.make_context(roll0, corridor_idx),
        'informative task0 (goal -1)': P.make_context(roll0, [0] + corridor_idx),
        'informative task1 (goal +1)': P.make_context(roll1, [0] + corridor_idx),
    }
    colors = {'ambiguous (corridor only)': 'tab:red',
              'informative task0 (goal -1)': 'tab:blue',
              'informative task1 (goal +1)': 'tab:green'}

    flow_s = {k: P.sample_posterior(flow_enc, c, args.K) for k, c in ctxs.items()}
    pearl_s = {k: pearl_posterior_samples(pearl_enc, pearl_lat, c, args.K)[0]
               for k, c in ctxs.items()}

    amb = 'ambiguous (corridor only)'
    print('\n=== AMBIGUOUS context: is the posterior multimodal? ===')
    for name, s in [('PEARL (Gaussian+IB)', pearl_s[amb]), ('FLOW', flow_s[amb])]:
        bc = P.bimodality_coefficient(s)
        gap, sep, _ = P.kmeans2_separation(s)
        verdict = 'BIMODAL' if bc > 0.555 else 'unimodal'
        print(f'  {name:22s} BC={bc:.3f} ({verdict})  2means gap/within={sep:.2f}  var={s.var(0).sum():.3f}')

    # --- plot: PEARL (PCA->2D) | FLOW (2D direct or PCA) --------------------
    def pca2(ref):
        m = ref.mean(0)
        u = np.linalg.svd(ref - m, full_matrices=False)[2][:2]
        return lambda x: (x - m) @ u.T

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, title, samp, lat in [(axes[0], 'PEARL  (Gaussian + IB)', pearl_s, pearl_lat),
                                 (axes[1], 'FLOW (ours)', flow_s, flow_lat)]:
        allp = np.concatenate(list(samp.values()), 0)
        proj = (lambda x: x) if lat == 2 else pca2(allp)
        for name, s in samp.items():
            p = proj(s)
            ax.scatter(p[:, 0], p[:, 1], s=8, alpha=0.35, c=colors[name], label=name)
        bc = P.bimodality_coefficient(samp[amb])
        ax.set_title(f'{title}\nambiguous-context BC = {bc:.3f}'
                     f' ({"bimodal" if bc > 0.555 else "unimodal"})')
        ax.set_xlabel('c[0]' if lat == 2 else 'PC1')
        ax.set_ylabel('c[1]' if lat == 2 else 'PC2')
        ax.legend(fontsize=8, loc='best')
    plt.suptitle(f'Posterior under task ambiguity (T-maze passive, {args.ctxlen} corridor steps, K={args.K})',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f'\n[saved] {args.out}')


if __name__ == '__main__':
    main()
