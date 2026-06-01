"""Probe the flow context-encoder's posterior for multimodality (T-maze).

Core question (our research contribution): when the context is genuinely
ambiguous about the task, does the flow posterior p(c | context) stay
MULTIMODAL -- something PEARL's Gaussian product cannot represent?

T-maze passive is the ideal probe. The task (goal arm) is revealed ONLY by
the very first oracle observation [0, goal_y]. Every corridor step is [0, 0]:
byte-identical for both tasks. So:
  * AMBIGUOUS context  = corridor-only transitions  -> should be BIMODAL
  * INFORMATIVE context = oracle + corridor          -> should be UNIMODAL,
                                                        at the correct task

The flow encoder samples a fresh N(0,I) base c_0 on every forward() call, so
batching one context to K rows yields K independent posterior samples. If the
encoder collapsed to a deterministic c(context) (the "C2" failure flagged in
flow_encoder.py), every sample lands on one point even for the ambiguous
context -- the probe detects exactly this.

Runs on CPU by default so it does not contend with training on the GPU.

Usage:
    python tools/probe_flow_posterior.py <snapshot_dir> [--K 500] [--out probe.png]
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.flow.flow_encoder import FlowContextEncoder
from rlkit.torch.flow.prior_flow import PriorFlow

TAU_DIM = 16  # fixed in FlowContextEncoder / PriorFlow


def scripted_rollout(env, task_idx, corridor_length):
    """Optimal scripted trajectory: walk right down the corridor, then step
    into the correct goal arm. Returns a list of (obs, action, reward) tuples
    in the SAME [o, a, r] order the agent's update_context uses."""
    env.reset_task(task_idx)
    o = env.reset()
    goal_y = env.wrapped_env._goal_y
    right = np.array([1., 0., 0., 0.], dtype=np.float32)  # argmax idx 0 -> right
    up = np.array([0., 1., 0., 0.], dtype=np.float32)     # idx 1 -> up
    down = np.array([0., 0., 1., 0.], dtype=np.float32)   # idx 2 -> down
    T = corridor_length + 1
    transitions = []
    for t in range(T):
        a = right if t < corridor_length else (up if goal_y > 0 else down)
        no, r, d, info = env.step(a)
        transitions.append((np.asarray(o, np.float32).copy(), a.copy(), float(r)))
        o = no
    return transitions


def make_context(transitions, idxs):
    rows = []
    for j in idxs:
        o, a, r = transitions[j]
        rows.append(np.concatenate([o, a, [r]]))
    return np.asarray(rows, dtype=np.float32)  # (len, 7)


def sample_posterior(enc, context_np, K):
    ctx = torch.from_numpy(context_np).to(ptu.device)
    ctx_batch = ctx.unsqueeze(0).expand(K, -1, -1).contiguous()
    with torch.no_grad():
        c = enc.forward(ctx_batch)
    return ptu.get_numpy(c)  # (K, latent)


def bimodality_coefficient(samples):
    """Sarle's bimodality coefficient on the principal-axis 1-D projection.
    BC = (skew^2 + 1) / kurtosis (standard, non-excess). BC > 0.555 => the
    distribution is more bimodal than a normal (uniform = 5/9 = 0.555)."""
    Xc = samples - samples.mean(0)
    u = np.linalg.svd(Xc, full_matrices=False)[2][0]   # principal axis
    z = Xc @ u
    n = len(z)
    s = z.std()
    if s < 1e-8:
        return 0.0
    skew = (((z) / s) ** 3).mean()
    kurt = (((z) / s) ** 4).mean()                     # non-excess kurtosis
    return (skew ** 2 + 1.0) / kurt


def kmeans2_separation(samples):
    """Pure-numpy 2-means. Returns (gap, gap/within-std, centers).
    High ratio => two well-separated clusters (bimodal); low => one blob."""
    Xc = samples - samples.mean(0)
    u = np.linalg.svd(Xc, full_matrices=False)[2][0]
    proj = Xc @ u
    centers = np.array([samples[proj.argmin()], samples[proj.argmax()]], dtype=float)
    lab = None
    for _ in range(100):
        d = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([samples[lab == k].mean(0) if (lab == k).any() else centers[k]
                        for k in (0, 1)])
        if np.allclose(new, centers):
            break
        centers = new
    gap = np.linalg.norm(centers[0] - centers[1])
    within = []
    for k in (0, 1):
        pts = samples[lab == k]
        if len(pts) > 1:
            within.append(np.sqrt(((pts - centers[k]) ** 2).sum(1).mean()))
    within = np.mean(within) if within else 0.0
    return gap, (gap / within if within > 1e-8 else float('inf')), centers


def run_sweep(enc, prior, env, corridor, K, out_dir):
    """Diagnostics 1+2: how the ambiguous posterior changes with the number of
    (identical) corridor transitions, and how it compares to the learned prior
    marginal p(c). Tests the hypothesis that PoE over-counting of identical null
    transitions sharpens the posterior asymmetrically rather than keeping a
    symmetric 2-task mixture."""
    roll0 = scripted_rollout(env, 0, corridor)
    roll1 = scripted_rollout(env, 1, corridor)
    corridor_idx = list(range(1, corridor))            # null transitions
    lengths = sorted(set([1, 2, 4, len(corridor_idx)]))

    m0 = sample_posterior(enc, make_context(roll0, [0] + corridor_idx), K).mean(0)
    m1 = sample_posterior(enc, make_context(roll1, [0] + corridor_idx), K).mean(0)

    print('\n########## SWEEP: context length (diagnostic 1) ##########')
    sweep = {}
    for L in lengths:
        s = sample_posterior(enc, make_context(roll0, corridor_idx[:L]), K)
        bc = bimodality_coefficient(s)
        gap, sep, _ = kmeans2_separation(s)
        sweep[L] = dict(s=s, var=s.var(0).sum(), bc=bc, sep=sep)
        print(f'  L={L:>2} corridor steps | var={s.var(0).sum():.3f}  BC={bc:.3f}  '
              f'2means gap/within={sep:.2f}')

    print('\n########## PRIOR marginal p(c) (diagnostic 2) ##########')
    if prior is not None:
        with torch.no_grad():
            ps = ptu.get_numpy(prior.sample(K, n_ode_steps=5))
        print(f'  prior samples | var={ps.var(0).sum():.3f}  mean={np.round(ps.mean(0),3)}  '
              f'BC={bimodality_coefficient(ps):.3f}')
    else:
        ps = None
        print('  (no prior_flow in snapshot)')

    latent = m0.shape[0]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))
    if latent == 2:
        proj = lambda x: x
    else:
        from numpy.linalg import svd
        allp = np.concatenate([v['s'] for v in sweep.values()], 0)
        u = svd(allp - allp.mean(0), full_matrices=False)[2][:2]
        proj = lambda x: (x - allp.mean(0)) @ u.T
    if ps is not None:
        p = proj(ps)
        axL.scatter(p[:, 0], p[:, 1], s=8, alpha=0.2, c='lightgray', label='prior p(c)')
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(lengths)))
    for col, L in zip(cmap, lengths):
        p = proj(sweep[L]['s'])
        axL.scatter(p[:, 0], p[:, 1], s=8, alpha=0.35, color=col, label=f'ambiguous L={L}')
    for m, mk, lab in [(m0, 'X', 'task0 mean'), (m1, 'P', 'task1 mean')]:
        pm = proj(m[None, :])[0]
        axL.scatter([pm[0]], [pm[1]], s=260, marker=mk, c='red',
                    edgecolors='k', linewidths=1.5, label=lab, zorder=5)
    axL.set_title('posterior vs context length & prior'); axL.legend(fontsize=8)
    axL.set_xlabel('c[0]' if latent == 2 else 'PC1')
    axL.set_ylabel('c[1]' if latent == 2 else 'PC2')

    axR.plot(lengths, [sweep[L]['var'] for L in lengths], 'o-', label='total variance')
    axR.plot(lengths, [sweep[L]['bc'] for L in lengths], 's-', label='bimodality coeff')
    axR.axhline(0.555, ls='--', c='gray', lw=1, label='BC bimodal threshold')
    axR.set_xlabel('# corridor (null) transitions'); axR.legend(fontsize=9)
    axR.set_title('dispersion / bimodality vs context length')
    plt.tight_layout()
    out = os.path.join(out_dir, 'probe_sweep.png')
    plt.savefig(out, dpi=120)
    print(f'\n[saved] {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot', help='output/.../<timestamp> dir with *.pth')
    ap.add_argument('--K', type=int, default=500, help='posterior samples per context')
    ap.add_argument('--corridor', type=int, default=10)
    ap.add_argument('--sweep', action='store_true', help='also run length-sweep + prior diagnostics')
    ap.add_argument('--out', default=None, help='PNG path (default: <snapshot>/probe_posterior.png)')
    args = ap.parse_args()

    ptu.set_gpu_mode(False)  # CPU: do not contend with training on GPU
    snap = args.snapshot.rstrip('/')
    out = args.out or os.path.join(snap, 'probe_posterior.png')

    # --- infer architecture from the checkpoint itself (robust across configs)
    enc_sd = torch.load(os.path.join(snap, 'context_encoder.pth'), map_location='cpu')
    latent = enc_sd['vnet.4.weight'].shape[0]
    hidden = enc_sd['vnet.0.weight'].shape[0]
    context_dim = enc_sd['vnet.0.weight'].shape[1] - latent - TAU_DIM
    print(f'[arch] latent={latent}  hidden={hidden}  context_dim={context_dim}')

    prior = None
    prior_path = os.path.join(snap, 'prior_flow.pth')
    if os.path.exists(prior_path):
        prior_sd = torch.load(prior_path, map_location='cpu')
        prior_hidden = prior_sd['vnet.0.weight'].shape[0]
        prior = PriorFlow(latent_dim=latent, hidden_dim=prior_hidden)
        prior.load_state_dict(prior_sd)
        prior.eval()
        print(f'[arch] prior_flow attached (hidden={prior_hidden})')

    enc = FlowContextEncoder(context_dim=context_dim, latent_dim=latent,
                             hidden_dim=hidden, n_ode_steps=5, max_context=None,
                             tau_eps=0.05, vel_clip=10.0, prior_flow=prior)
    enc.load_state_dict(enc_sd)
    enc.eval()

    # --- build contexts from real (scripted-optimal) rollouts -----------------
    env = NormalizedBoxEnv(ENVS['tmaze-passive'](n_tasks=2, corridor_length=args.corridor))
    roll0 = scripted_rollout(env, 0, args.corridor)  # goal_y = -1
    roll1 = scripted_rollout(env, 1, args.corridor)  # goal_y = +1
    corridor_idx = list(range(1, args.corridor))     # skip t0 (oracle), skip junction

    contexts = {
        'ambiguous (corridor only)': make_context(roll0, corridor_idx),
        'informative task0 (goal -1)': make_context(roll0, [0] + corridor_idx),
        'informative task1 (goal +1)': make_context(roll1, [0] + corridor_idx),
    }

    # --- sample + analyze -----------------------------------------------------
    samples, stats = {}, {}
    for name, ctx in contexts.items():
        s = sample_posterior(enc, ctx, args.K)
        samples[name] = s
        total_var = s.var(0).sum()
        bc = bimodality_coefficient(s)
        gap, sep, centers = kmeans2_separation(s)
        stats[name] = dict(mean=s.mean(0), total_var=total_var, bc=bc,
                           gap=gap, sep=sep, centers=centers)
        print(f'\n=== {name} ===')
        print(f'  total variance (trace cov): {total_var:.4f}')
        print(f'  bimodality coeff:           {bc:.3f}  (>0.555 => bimodal-ish)')
        print(f'  2-means gap / within-std:   {sep:.2f}  (gap={gap:.3f})')

    # cross-task: do the two ambiguous modes match the informative locations?
    amb = stats['ambiguous (corridor only)']
    m0 = stats['informative task0 (goal -1)']['mean']
    m1 = stats['informative task1 (goal +1)']['mean']
    print('\n=== cross-task mode matching ===')
    print(f'  informative task0 mean: {np.round(m0, 3)}')
    print(f'  informative task1 mean: {np.round(m1, 3)}')
    if amb['centers'] is not None:
        ca, cb = amb['centers']
        # match each ambiguous center to nearest informative mean
        d_a = [np.linalg.norm(ca - m0), np.linalg.norm(ca - m1)]
        d_b = [np.linalg.norm(cb - m0), np.linalg.norm(cb - m1)]
        print(f'  ambiguous mode A {np.round(ca,3)} -> task{np.argmin(d_a)} (d={min(d_a):.3f})')
        print(f'  ambiguous mode B {np.round(cb,3)} -> task{np.argmin(d_b)} (d={min(d_b):.3f})')

    # --- plot (2D direct, or PCA to 2D for higher latent) ---------------------
    all_s = np.concatenate(list(samples.values()), 0)
    if latent == 2:
        proj = lambda x: x
        xlab, ylab = 'c[0]', 'c[1]'
    else:
        from sklearn.decomposition import PCA
        pca = PCA(2).fit(all_s)
        proj = lambda x: pca.transform(x)
        xlab, ylab = 'PC1', 'PC2'

    plt.figure(figsize=(7, 7))
    colors = {'ambiguous (corridor only)': 'tab:red',
              'informative task0 (goal -1)': 'tab:blue',
              'informative task1 (goal +1)': 'tab:green'}
    for name, s in samples.items():
        p = proj(s)
        plt.scatter(p[:, 0], p[:, 1], s=8, alpha=0.35,
                    c=colors.get(name, 'gray'), label=name)
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.title(f'Flow posterior samples (K={args.K} each)\n{os.path.basename(snap)}')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f'\n[saved] {out}')

    if args.sweep:
        run_sweep(enc, prior, env, args.corridor, args.K, snap)


if __name__ == '__main__':
    main()
