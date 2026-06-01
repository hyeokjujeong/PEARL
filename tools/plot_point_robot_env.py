"""Render the point-robot (2D navigation) environment as a paper figure.

Plots the 100 task goals (seed 1337) on the unit square and overlays two
example greedy trajectories. Output: tex/figures/point_robot_env.pdf
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rlkit.envs import ENVS

env = ENVS['point-robot'](randomize_tasks=True, n_tasks=100)
goals = np.array(env.goals)                       # (100, 2)

fig, ax = plt.subplots(figsize=(3.0, 3.0))

# all task goals
ax.scatter(goals[:, 0], goals[:, 1], s=12, c='0.72',
           edgecolors='none', zorder=2)

# two example tasks: greedy straight-to-goal trajectories.
# the two colors carry no meaning beyond distinguishing the examples.
rng = np.random.RandomState(0)
for gi, col in zip([3, 61], ['#1f77b4', '#ff7f0e']):
    g = goals[gi]
    s = rng.uniform(-1, 1, 2)
    traj = [s.copy()]
    for _ in range(20):                           # max_path_length = 20
        s = s + np.clip(g - s, -0.1, 0.1)         # action in [-0.1, 0.1]^2
        traj.append(s.copy())
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], '-', color=col, lw=1.5,
            alpha=0.9, zorder=3)
    ax.scatter([traj[0, 0]], [traj[0, 1]], marker='o', s=26, c=col,
               edgecolors='white', linewidths=0.6, zorder=4)
    ax.scatter([g[0]], [g[1]], marker='*', s=120, c=col,
               edgecolors='white', linewidths=0.8, zorder=5)

ax.set_xlim(-1.12, 1.12)
ax.set_ylim(-1.12, 1.12)
ax.set_aspect('equal')
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.tick_params(labelsize=8)

# legend (markers in neutral gray; colour only separates the two examples)
handles = [
    Line2D([], [], marker='o', ls='none', mfc='0.72', mec='none', ms=5,
           label='task goals (100)'),
    Line2D([], [], marker='o', ls='none', mfc='0.4', mec='white', mew=0.6,
           ms=5, label='start'),
    Line2D([], [], marker='*', ls='none', mfc='0.4', mec='white', mew=0.8,
           ms=11, label='goal'),
    Line2D([], [], ls='-', c='0.4', lw=1.5, label='example rollout'),
]
ax.legend(handles=handles, fontsize=6.3, loc='upper center',
          bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False,
          handletextpad=0.35, columnspacing=1.0)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'tex', 'figures',
                   'point_robot_env.pdf')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches='tight')
print('saved:', os.path.abspath(out))
