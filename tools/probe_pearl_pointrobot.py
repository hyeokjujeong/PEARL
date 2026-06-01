"""Does PEARL's Gaussian encoder separate tasks on point-robot (a task it
SUCCEEDS on)? Contrast with its T-maze collapse.

point-robot: task = a 2-D goal; reward = -distance to goal (DENSE, so every
transition's reward reveals the task). This is exactly the regime PEARL was
built for -- the task lives in the reward. We expect the posterior to separate
cleanly by goal, unlike T-maze (where the task lives in an observation the
reward-driven critic can't teach the encoder to read).

Loads the trained PEARL encoder, rolls out a scripted move-to-goal policy for
several spread-out goals, computes the Gaussian posterior per task, and plots.

Usage:
    python tools/probe_pearl_pointrobot.py <pearl_snap> [--ntasks 6] [--out pr.png]
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlkit.torch.pytorch_util as ptu
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
import compare_posterior_pearl_flow as C


def scripted_rollout(env, goal, max_path_length, n_episodes=3):
    """Move toward the goal. action in [-1,1] (NormalizedBoxEnv scales x0.1)."""
    rows = []
    for _ in range(n_episodes):
        o = np.asarray(env.reset(), np.float32)
        for _ in range(max_path_length):
            a = np.clip(10.0 * (goal - o), -1.0, 1.0).astype(np.float32)
            no, r, d, info = env.step(a)
            rows.append(np.concatenate([o, a, [r]]))      # [o(2), a(2), r(1)] = 5
            o = np.asarray(no, np.float32)
    return np.asarray(rows, np.float32)


def farthest_point_sample(goals, k):
    sel = [0]
    while len(sel) < k:
        d = np.min([np.linalg.norm(goals - goals[i], axis=1) for i in sel], axis=0)
        sel.append(int(np.argmax(d)))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pearl_snap')
    ap.add_argument('--ntasks', type=int, default=6)
    ap.add_argument('--K', type=int, default=300)
    ap.add_argument('--out', default='pearl_pointrobot_posterior.png')
    args = ap.parse_args()

    ptu.set_gpu_mode(False)
    np.random.seed(0)

    enc, latent = C.load_pearl(args.pearl_snap)
    print(f'[pearl] latent={latent}')

    env = NormalizedBoxEnv(ENVS['point-robot'](n_tasks=100, randomize_tasks=True))
    goals = np.asarray(env.wrapped_env.goals, dtype=np.float32)
    sel = farthest_point_sample(goals, args.ntasks)
    mpl = 20

    samples, means, vars = {}, {}, {}
    for idx in sel:
        env.reset_task(idx)
        goal = np.asarray(env.wrapped_env._goal, np.float32)
        ctx = scripted_rollout(env, goal, mpl)
        s, mu, var = C.pearl_posterior_samples(enc, latent, ctx, args.K)
        key = f'goal({goal[0]:+.2f},{goal[1]:+.2f})'
        samples[key] = s
        means[key] = mu
        vars[key] = var.sum()

    # separation: between-task mean distance vs within-task spread
    keys = list(samples.keys())
    M = np.stack([means[k] for k in keys])
    pair = [np.linalg.norm(M[i] - M[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    within = np.mean([np.sqrt(samples[k].var(0).sum()) for k in keys])
    print(f'\n  mean posterior var (per task):   {np.mean(list(vars.values())):.4f}')
    print(f'  between-task mean distance (avg): {np.mean(pair):.4f}')
    print(f'  within-task spread (avg std):     {within:.4f}')
    print(f'  SEPARATION ratio (between/within):{np.mean(pair)/within:.2f}  (>>1 => well separated)')

    # PCA all samples -> 2D (latent=5)
    alls = np.concatenate(list(samples.values()), 0)
    m = alls.mean(0)
    U = np.linalg.svd(alls - m, full_matrices=False)[2][:2]
    proj = lambda x: (x - m) @ U.T

    plt.figure(figsize=(8, 7))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(keys)))
    for col, k in zip(cmap, keys):
        p = proj(samples[k])
        plt.scatter(p[:, 0], p[:, 1], s=10, alpha=0.4, color=col, label=k)
        pm = proj(means[k][None, :])[0]
        plt.scatter([pm[0]], [pm[1]], s=200, marker='*', color=col,
                    edgecolors='k', linewidths=1.2, zorder=5)
    plt.title(f'PEARL posterior per task — point-robot (a success env)\n'
              f'latent={latent} (PCA->2D), K={args.K}/task')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f'\n[saved] {args.out}')


if __name__ == '__main__':
    main()
