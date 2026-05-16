"""
T-MAZE environment sanity checks (TMAZE_guideline_v7.md §6-1).

Single-repo correctness checks for PEARL's TMazeEnv. Mirrors the 9-test
structure of the VariBAD sanity check
(/PublicSSD/shnoh/varibad/environments/navigation/tmaze_sanity_check.py).
The only PEARL-specific differences are:

  - Action input: semantic indices are translated to Box(4) vectors via
    `one_hot(idx)`. argmax of the resulting vector is what the env consumes.
  - Task interface: VariBAD's `reset_task(task=np.array([+1.0]))` becomes
    PEARL's `reset_task(1)` (goal_y=+1, UP). idx=0 -> goal_y=-1 (DOWN).
  - Wrapper compat: NormalizedBoxEnv (PEARL's wrapper) instead of
    VariBadWrapper. The wrapper's [-1, +1] rescale is identity against our
    Box(low=-1, high=1, shape=(4,)) action space.

Cross-repo semantic equivalence is checked separately in
`rlkit/envs/tmaze_equivalence_check.py`.

Run:
    python -m rlkit.envs.tmaze_sanity_check

Exits with code 0 on all pass, non-zero otherwise.
"""

import sys
import numpy as np

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv


# ===========================================================================
# Action adapter: semantic discrete index -> Box(4) vector.
# argmax of the returned vector equals `idx`.
# ===========================================================================

def one_hot(idx, dim=4):
    """Return a Box(4) vector whose argmax is `idx`.

    Concretely [-1, -1, ..., -1] with +1 at position `idx`. This sits at a
    corner of the Box and makes the argmax unambiguous (no tie-breaking
    ambiguity, no float-comparison fragility).
    """
    v = np.full(dim, -1.0, dtype=np.float32)
    v[idx] = +1.0
    return v


# Pre-built one-hots for the 4 semantic actions (matches the index mapping
# in TMazeEnv._ACTION_MAPPING: 0=right, 1=up, 2=down, 3=left).
RIGHT = one_hot(0)
UP    = one_hot(1)
DOWN  = one_hot(2)
LEFT  = one_hot(3)


# ===========================================================================
# Passive tests (Sprint 1 mirror)
# ===========================================================================

def test_passive_random_policy_return(corridor_length=10, num_episodes=100, seed=0):
    """Random uniform-over-Box(4) policy: mean return in (-1.0, 0.5).

    PEARL note: `Box(4).sample()` returns 4 iid uniforms; argmax is then
    uniform over {0,1,2,3} by symmetry, so this matches VariBAD's
    `Discrete(4).sample()` distribution at the dynamics level. Empirically
    the passive mean lands close to -1.0 (uniform 4-action is roughly
    worst-case), consistent with VariBAD's measurement.
    """
    env = ENVS['tmaze-passive'](n_tasks=2, corridor_length=corridor_length)
    env.seed(seed)
    rng = np.random.RandomState(seed)

    returns = []
    for _ in range(num_episodes):
        env.reset_task(int(rng.randint(2)))  # idx in {0, 1}
        done = False
        G = 0.0
        while not done:
            a = env.action_space.sample()
            _, r, done, _ = env.step(a)
            G += r
        returns.append(G)

    mean = float(np.mean(returns))
    mn, mx = float(np.min(returns)), float(np.max(returns))
    print(f"[P-1] random policy: mean={mean:+.3f}, min={mn:+.3f}, max={mx:+.3f} "
          f"(n={num_episodes}, L={corridor_length})")
    assert -1.0 <= mean < 0.5, \
        f"mean return {mean} not in (-1.0, 0.5) per v7 §6-1"


def test_passive_manual_trajectories(corridor_length=10):
    """Two pre-defined action sequences against hand-calculated returns.

    Sequence A (optimal, goal_y=+1):
        actions = [right] * L + [up]   -> return = +1.0
    Sequence B (one wall-bump at step 2, goal_y=+1):
        actions = [right, up(bump), right * (L-2), up(bump)]
        -> return = -(L-1)/L   (for L=10: -0.9)
    """
    L = corridor_length
    T = L + 1
    expected_A = 1.0
    expected_B = -(L - 1) / L

    env = ENVS['tmaze-passive'](n_tasks=2, corridor_length=L)
    env.seed(0)

    # Sequence A
    env.reset_task(1)   # goal_y = +1 (UP)
    seq_A = [RIGHT] * L + [UP]
    assert len(seq_A) == T
    G_A = 0.0
    for a in seq_A:
        _, r, done, _ = env.step(a)
        G_A += r
    print(f"[P-2A] optimal (L={L}): return={G_A:+.4f} (expected {expected_A:+.4f})")
    assert done and abs(G_A - expected_A) < 1e-9, \
        f"sequence A return {G_A} != expected {expected_A}"

    # Sequence B
    env.reset_task(1)
    seq_B = [RIGHT, UP] + [RIGHT] * (L - 2) + [UP]
    assert len(seq_B) == T
    G_B = 0.0
    for a in seq_B:
        _, r, done, _ = env.step(a)
        G_B += r
    print(f"[P-2B] 1-step delay (L={L}): return={G_B:+.4f} (expected {expected_B:+.4f})")
    assert done and abs(G_B - expected_B) < 1e-9, \
        f"sequence B return {G_B} != expected {expected_B}"


def test_passive_context_exposure(corridor_length=10):
    """goal_y=+1 vs goal_y=-1 must change position-0 obs but not corridor obs."""
    env = ENVS['tmaze-passive'](n_tasks=2, corridor_length=corridor_length)

    env.seed(42)
    env.reset_task(1)  # goal_y = +1
    obs_up_start = env._get_obs()  # NB: reset_task already called reset+_get_obs
    # Re-reset to get a fresh start obs after the oracle_visited flag is reset.
    obs_up_start = env.reset()
    obs_up_mid = []
    for _ in range(3):
        o, _, _, _ = env.step(RIGHT)
        obs_up_mid.append(o.copy())

    env.seed(42)
    env.reset_task(0)  # goal_y = -1
    obs_dn_start = env.reset()
    obs_dn_mid = []
    for _ in range(3):
        o, _, _, _ = env.step(RIGHT)
        obs_dn_mid.append(o.copy())

    print(f"[P-3] oracle obs UP={obs_up_start.tolist()} vs DN={obs_dn_start.tolist()}")
    assert not np.allclose(obs_up_start, obs_dn_start), \
        f"oracle obs failed to differentiate context: UP={obs_up_start}, DN={obs_dn_start}"
    assert np.allclose(obs_up_start, [0.0, +1.0]), f"UP oracle obs got {obs_up_start.tolist()}"
    assert np.allclose(obs_dn_start, [0.0, -1.0]), f"DN oracle obs got {obs_dn_start.tolist()}"
    for i, (ou, od) in enumerate(zip(obs_up_mid, obs_dn_mid)):
        assert np.allclose(ou, od), \
            f"corridor obs at step {i+1} leaked context: UP={ou}, DN={od}"


def test_passive_wrapper_compatibility(corridor_length=10):
    """End-to-end smoke through NormalizedBoxEnv (PEARL's wrapper).

    Verifies that the [-1, +1] -> [-1, +1] rescale is identity (no distortion
    of our argmax-based action handling) by running an optimal trajectory and
    confirming return = +1.0.
    """
    raw = ENVS['tmaze-passive'](n_tasks=2, corridor_length=corridor_length)
    env = NormalizedBoxEnv(raw)
    env.reset_task(1)
    state = env.reset()
    assert state.shape == (2,), f"unexpected wrapped obs shape {state.shape}"
    assert np.allclose(state, [0.0, +1.0]), \
        f"wrapped first obs should be [0, +1] (oracle exposed), got {state.tolist()}"

    T = corridor_length + 1
    G = 0.0
    steps = 0
    seq = [RIGHT] * corridor_length + [UP]
    for a in seq:
        state, r, done, info = env.step(a)
        G += r
        steps += 1
        if done:
            break
    print(f"[P-4] wrapper compat: ran {steps}/{T} steps, return={G:+.4f}, done={done}, info={info}")
    assert done and abs(G - 1.0) < 1e-9, \
        f"NormalizedBoxEnv must preserve argmax dynamics, got return={G}"
    assert 'task_idx' in info


# ===========================================================================
# Active tests (Sprint 2 mirror)
# ===========================================================================

def test_active_horizon(corridor_length=10):
    """Active mode: oracle_length=1, T=L+3, start x=1, start obs [0,0]."""
    env = ENVS['tmaze-active'](n_tasks=2, corridor_length=corridor_length)
    print(f"[A-1] horizon: oracle_length={env.oracle_length}, "
          f"_max_episode_steps={env._max_episode_steps} (expect {corridor_length + 3})")
    assert env.oracle_length == 1
    assert env._max_episode_steps == corridor_length + 3

    env.reset_task(1)   # goal_y = +1
    obs_start = env.reset()
    assert env.x == 1, f"active start x={env.x}, expected 1"
    assert np.allclose(obs_start, [0.0, 0.0]), \
        f"active start obs should be [0,0] (oracle not yet visited), got {obs_start.tolist()}"


def test_active_random_policy_return(corridor_length=10, num_episodes=100, seed=0):
    """Active random uniform-Box(4) policy: mean return in (-1.0, 0.5)."""
    env = ENVS['tmaze-active'](n_tasks=2, corridor_length=corridor_length)
    env.seed(seed)
    rng = np.random.RandomState(seed)

    returns = []
    for _ in range(num_episodes):
        env.reset_task(int(rng.randint(2)))
        done = False
        G = 0.0
        while not done:
            a = env.action_space.sample()
            _, r, done, _ = env.step(a)
            G += r
        returns.append(G)

    mean = float(np.mean(returns))
    mn, mx = float(np.min(returns)), float(np.max(returns))
    print(f"[A-2] random policy: mean={mean:+.3f}, min={mn:+.3f}, max={mx:+.3f} "
          f"(n={num_episodes}, L={corridor_length})")
    assert -1.0 <= mean < 0.5, \
        f"active mean return {mean} not in (-1.0, 0.5) per v7 §6-1"


def test_active_manual_trajectories(corridor_length=10):
    """Five pre-defined action sequences for active mode.

    Coord: oracle at x=0, start at x=1, junction at x=L+1. T = L+3.
    Intermediate reward: 0 if x_after >= t-1, else -1/(T-1) = -1/(L+2).

    2A — optimal + oracle visit + goal_y=+1:
        [LEFT, RIGHT * (L+1), UP]    -> +1.0
    2B — optimal + oracle visit + goal_y=-1:
        [LEFT, RIGHT * (L+1), DOWN]  -> +1.0
    2C — no oracle visit + on-pace + wrong terminal + goal_y=+1:
        [RIGHT * (L+2), DOWN]        ->  0.0   (Markovian failure)
    2D — oracle visit + 1 wall-bump after returning + goal_y=+1:
        [LEFT, RIGHT, UP(bump), RIGHT * (L-1), UP(bump)]
        penalty at t=3..t=L+2 (L steps), terminal miss
        -> -L / (L+2)   (for L=10: -10/12 ≈ -0.8333)
    2E — oracle visit + on-pace + WRONG arm + goal_y=+1:
        [LEFT, RIGHT * (L+1), DOWN]
        same on-pace trajectory as 2A but wrong terminal direction.
        -> 0.0
    """
    L = corridor_length
    T = L + 3

    def run(seq, goal_idx, expected_return):
        env = ENVS['tmaze-active'](n_tasks=2, corridor_length=L)
        env.seed(0)
        env.reset_task(goal_idx)
        assert len(seq) == T, f"sequence length {len(seq)} != T={T}"
        G = 0.0
        for a in seq:
            _, r, done, _ = env.step(a)
            G += r
        assert done, "sequence did not terminate after T steps"
        assert abs(G - expected_return) < 1e-9, \
            f"return {G:+.6f} != expected {expected_return:+.6f}"
        return G

    # idx 0 -> goal_y=-1 (DOWN); idx 1 -> goal_y=+1 (UP).
    seq_A = [LEFT] + [RIGHT] * (L + 1) + [UP]
    seq_B = [LEFT] + [RIGHT] * (L + 1) + [DOWN]
    seq_C = [RIGHT] * (L + 2) + [DOWN]
    seq_D = [LEFT, RIGHT, UP] + [RIGHT] * (L - 1) + [UP]
    seq_E = [LEFT] + [RIGHT] * (L + 1) + [DOWN]   # 2A's prefix + down

    G_A = run(seq_A, goal_idx=1, expected_return=1.0)
    print(f"[A-3A] optimal+visit+correct  (g=+1, L={L}): return={G_A:+.4f} (expected +1.0000)")

    G_B = run(seq_B, goal_idx=0, expected_return=1.0)
    print(f"[A-3B] optimal+visit+correct  (g=-1, L={L}): return={G_B:+.4f} (expected +1.0000)")

    G_C = run(seq_C, goal_idx=1, expected_return=0.0)
    print(f"[A-3C] no-visit+on-pace+wrong (g=+1, L={L}): return={G_C:+.4f} (expected +0.0000)")

    expected_D = -L / (L + 2)
    G_D = run(seq_D, goal_idx=1, expected_return=expected_D)
    print(f"[A-3D] visit+1 bump           (g=+1, L={L}): return={G_D:+.4f} (expected {expected_D:+.4f})")

    G_E = run(seq_E, goal_idx=1, expected_return=0.0)
    print(f"[A-3E] visit+on-pace+wrong    (g=+1, L={L}): return={G_E:+.4f} (expected +0.0000)")


def test_active_context_exposure(corridor_length=10):
    """In active mode, context exposure only happens AFTER moving left to oracle.

    Start obs (x=1) is corridor-middle [0,0] for both contexts.
    After left action: x=0 -> oracle exposed, obs differs by goal_y.
    After right back: x=1 -> corridor-middle [0,0] for both (single-visit rule).
    """
    def trace(goal_idx):
        env = ENVS['tmaze-active'](n_tasks=2, corridor_length=corridor_length)
        env.seed(42)
        env.reset_task(goal_idx)
        o_start = env.reset()                       # x=1
        o_oracle, _, _, _ = env.step(LEFT)          # left -> x=0
        o_back,   _, _, _ = env.step(RIGHT)         # right -> x=1
        return o_start, o_oracle, o_back

    up_start, up_oracle, up_back = trace(1)   # g=+1
    dn_start, dn_oracle, dn_back = trace(0)   # g=-1

    print(f"[A-4] start  (x=1): UP={up_start.tolist()}, DN={dn_start.tolist()}")
    print(f"      oracle (x=0): UP={up_oracle.tolist()}, DN={dn_oracle.tolist()}")
    print(f"      back   (x=1): UP={up_back.tolist()}, DN={dn_back.tolist()}")

    assert np.allclose(up_start, dn_start), \
        f"active start obs leaked context: UP={up_start}, DN={dn_start}"
    assert np.allclose(up_start, [0.0, 0.0]), \
        f"active start obs should be [0,0], got {up_start.tolist()}"

    assert not np.allclose(up_oracle, dn_oracle), \
        f"oracle obs failed to differentiate: UP={up_oracle}, DN={dn_oracle}"
    assert np.allclose(up_oracle, [0.0, +1.0]), f"UP oracle obs got {up_oracle.tolist()}"
    assert np.allclose(dn_oracle, [0.0, -1.0]), f"DN oracle obs got {dn_oracle.tolist()}"

    assert np.allclose(up_back, dn_back), \
        f"post-oracle corridor obs leaked context: UP={up_back}, DN={dn_back}"
    assert np.allclose(up_back, [0.0, 0.0])


def test_active_wrapper_compatibility(corridor_length=10):
    """NormalizedBoxEnv end-to-end on an active optimal trajectory."""
    raw = ENVS['tmaze-active'](n_tasks=2, corridor_length=corridor_length)
    env = NormalizedBoxEnv(raw)
    env.reset_task(1)
    state = env.reset()
    assert state.shape == (2,)
    assert np.allclose(state, [0.0, 0.0]), \
        f"wrapped active first obs should be [0, 0], got {state.tolist()}"

    L = corridor_length
    T = L + 3
    seq = [LEFT] + [RIGHT] * (L + 1) + [UP]
    G = 0.0
    steps = 0
    for a in seq:
        state, r, done, info = env.step(a)
        G += r
        steps += 1
        if done:
            break
    print(f"[A-5] wrapper compat: ran {steps}/{T} steps, return={G:+.4f}, done={done}, info={info}")
    assert done and abs(G - 1.0) < 1e-9, \
        f"NormalizedBoxEnv must preserve argmax dynamics for active, got return={G}"
    assert 'task_idx' in info


# ===========================================================================
# Driver
# ===========================================================================

def main():
    print("=== T-MAZE sanity checks (PEARL; guideline v7 §6-1; Sprint 3) ===\n")
    failures = []
    tests = [
        # Passive
        ("passive: random_policy_return",  test_passive_random_policy_return),
        ("passive: manual_trajectories",   test_passive_manual_trajectories),
        ("passive: context_exposure",      test_passive_context_exposure),
        ("passive: wrapper_compatibility", test_passive_wrapper_compatibility),
        # Active
        ("active: horizon",                test_active_horizon),
        ("active: random_policy_return",   test_active_random_policy_return),
        ("active: manual_trajectories",    test_active_manual_trajectories),
        ("active: context_exposure",       test_active_context_exposure),
        ("active: wrapper_compatibility",  test_active_wrapper_compatibility),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}\n")
        except AssertionError as e:
            print(f"FAIL: {name} — {e}\n")
            failures.append(name)
        except Exception as e:
            print(f"ERROR: {name} — {type(e).__name__}: {e}\n")
            failures.append(name)
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
