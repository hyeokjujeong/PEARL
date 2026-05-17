"""
Cross-repo T-MAZE semantic equivalence check (TMAZE_guideline_v7.md §7-3,
ENVIRONMENT_SPEC.md §5).

Run from the PEARL repo (this file lives here only — there is intentionally
no mirror copy in the VariBAD repo). The script adds both repos' roots to
sys.path and instantiates both TMaze envs; equivalence is asserted by
running the same semantic action sequence on both and comparing the
(reward_t, done_t) trace step-by-step for 100 episodes × {passive, active}.

Equivalence protocol (per ENVIRONMENT_SPEC.md §5):

  1. A single external numpy RNG seeds (a) the per-episode goal_y choice and
     (b) the random action sequence. The env-internal RNG streams are
     ignored — they differ between VariBAD's `reset_task(task=None)` random
     sample and PEARL's idx-based `reset_task(idx)`, and pinning the goal
     externally is exactly the workaround for that.

  2. For each episode we force the same goal_y on both envs:
       VariBAD: reset_task(task=np.array([goal_y])) ; then explicit reset()
                (VariBAD's reset_task does NOT call reset internally.)
       PEARL:   reset_task(0 if goal_y == -1 else 1)
                (PEARL's reset_task calls reset internally.)
     We then assert _goal_y matches on both sides before stepping.

  3. Both envs are stepped with the SAME semantic action index:
       VariBAD: int index passed straight into Discrete(4).step
       PEARL:   one-hot vector with argmax == index (Box(4))
     A length-T sequence is rolled; T comes from `_max_episode_steps` and
     must agree between the two envs (sanity-asserted at setup).

  4. After each step we compare (reward, done). Observations are NOT
     compared (§7-3 explicitly allows wrapper-induced dtype/shape drift;
     observation encoding itself is verified by each repo's single-repo
     sanity check).

Run:
    cd /PublicSSD/shnoh/PEARL
    PYTHONPATH=. python rlkit/envs/tmaze_equivalence_check.py

(Or `python -m rlkit.envs.tmaze_equivalence_check`; the runpy
RuntimeWarning from this package's auto-import is harmless.)
"""

import sys
import numpy as np

# Repo roots. PEARL's path comes first so `from rlkit...` resolves locally;
# VariBAD's path is appended after so its `environments.navigation.tmaze`
# is reachable. Both paths are absolute per ENVIRONMENT_SPEC §5; if either
# repo moves, update here.
sys.path.insert(0, '/PublicSSD/shnoh/varibad')
sys.path.insert(0, '/PublicSSD/shnoh/PEARL')

from environments.navigation.tmaze import TMazeEnv as VariBADTMaze  # noqa: E402
from rlkit.envs.tmaze import TMazePassive, TMazeActive              # noqa: E402


# ---------------------------------------------------------------------------
# Action adapter: a semantic index becomes a Box(4) corner for PEARL.
# Identical helper to the one in tmaze_sanity_check.py; kept inline so this
# script is self-contained.
# ---------------------------------------------------------------------------

def one_hot(idx, dim=4):
    v = np.full(dim, -1.0, dtype=np.float32)
    v[idx] = +1.0
    return v


# ---------------------------------------------------------------------------
# Equivalence machinery
# ---------------------------------------------------------------------------

def force_goal(v_env, p_env, goal_y):
    """Pin the same hidden context on both envs and reset their MDP state."""
    v_env.reset_task(task=np.array([goal_y], dtype=np.float32))
    v_env.reset()                              # VariBAD: reset is separate
    p_env.reset_task(0 if goal_y == -1 else 1) # PEARL: reset_task() resets
    assert int(v_env._goal_y) == int(p_env._goal_y) == int(goal_y), (
        f"goal_y pin failed: VariBAD={v_env._goal_y}, "
        f"PEARL={p_env._goal_y}, requested={goal_y}"
    )


def run_episode(v_env, p_env, action_seq):
    """Step both envs through the same length-T semantic action sequence.

    Returns a list of per-step mismatch tuples (empty when fully equivalent).
    Each tuple: (t, idx, r_v, r_p, d_v, d_p).
    """
    mismatches = []
    for t, idx in enumerate(action_seq):
        _, r_v, d_v, _ = v_env.step(int(idx))
        _, r_p, d_p, _ = p_env.step(one_hot(int(idx)))
        if (not np.isclose(r_v, r_p, atol=1e-9)) or (bool(d_v) != bool(d_p)):
            mismatches.append((t, int(idx), float(r_v), float(r_p),
                               bool(d_v), bool(d_p)))
    return mismatches


def equivalence_check(mode, corridor_length, num_episodes, seed):
    """Compare reward/done traces over num_episodes random trajectories."""
    v_env = VariBADTMaze(corridor_length=corridor_length, mode=mode)
    p_cls = TMazePassive if mode == 'passive' else TMazeActive
    p_env = p_cls(n_tasks=2, corridor_length=corridor_length)

    T = v_env._max_episode_steps
    assert T == p_env._max_episode_steps, (
        f"[{mode}] horizon mismatch: VariBAD={T}, PEARL={p_env._max_episode_steps}"
    )

    rng = np.random.RandomState(seed)

    total_mismatch_steps = 0
    first_mismatch_details = None
    bad_episodes = 0

    for ep in range(num_episodes):
        goal_y = int(rng.choice([-1, +1]))
        force_goal(v_env, p_env, goal_y)
        action_seq = [int(rng.randint(4)) for _ in range(T)]

        ms = run_episode(v_env, p_env, action_seq)
        if ms:
            bad_episodes += 1
            total_mismatch_steps += len(ms)
            if first_mismatch_details is None:
                first_mismatch_details = (ep, goal_y, list(action_seq), ms)

    print(f"[{mode}] L={corridor_length}, T={T}, n={num_episodes}: "
          f"bad_episodes={bad_episodes}, total_mismatch_steps={total_mismatch_steps}")

    if first_mismatch_details is not None:
        ep, gy, seq, ms = first_mismatch_details
        print(f"  FIRST MISMATCH at episode {ep} (goal_y={gy:+d}):")
        print(f"    action_seq = {seq}")
        for (t, idx, r_v, r_p, d_v, d_p) in ms:
            dr = r_v - r_p
            print(f"    t={t:2d} idx={idx} | VariBAD r={r_v:+.6f} d={d_v} "
                  f"| PEARL r={r_p:+.6f} d={d_p} | dr={dr:+.3e}")

    assert total_mismatch_steps == 0 and bad_episodes == 0, (
        f"[{mode}] {bad_episodes} of {num_episodes} episodes mismatched "
        f"({total_mismatch_steps} step-level mismatches)"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("=== Cross-repo T-MAZE equivalence check (PEARL <-> VariBAD) ===")
    print("Protocol: same goal_y forced externally, same semantic action "
          "sequence, compare (reward_t, done_t) traces.\n")

    equivalence_check('passive', corridor_length=10, num_episodes=100, seed=0)
    print("PASS: passive\n")

    equivalence_check('active',  corridor_length=10, num_episodes=100, seed=0)
    print("PASS: active\n")

    print("ALL PASS")


if __name__ == '__main__':
    main()
