"""
T-MAZE environment for PEARL.

Based on Ni et al. 2023 (arXiv:2307.03864), "When Do Transformers Shine in RL?
Decoupling Memory from Credit Assignment". Mirrors the §7-2 common items of
the VariBAD repo's TMazeEnv (reward formula, observation encoding, grid
structure, horizon T). Cross-repo semantic equivalence is asserted by
`rlkit/envs/tmaze_equivalence_check.py`.

Spec source: ENVIRONMENT_SPEC.md (kept identical between this repo and
/PublicSSD/shnoh/varibad). Coordinate convention (Memory-RL): oracle at x=0,
start at x=oracle_length, junction at x = oracle_length + corridor_length.

PEARL-specific deviations (guideline §7-1):
  - Action space is Box(low=-1, high=1, shape=(4,)) instead of Discrete(4).
    The env applies `int(np.argmax(action))` to map back to the same 4
    semantic indices used by the VariBAD class. NormalizedBoxEnv's [-1, +1]
    rescaling is a no-op against this Box.
  - Task interface uses PEARL's idx-based convention: goals are
    pre-generated at __init__ (self.goals = [-1, +1]), and reset_task(idx)
    selects one. n_tasks is fixed at 2 (guideline §7-4).
"""

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from . import register_env


class TMazeEnv(Env):
    """Core T-MAZE environment. `mode` selects passive vs active.

    External-facing args: `corridor_length`, `mode`, `n_tasks` (must be 2).
    All other env semantics are hard-coded per ENVIRONMENT_SPEC.md §2.
    """

    metadata = {'render.modes': []}

    # Action index -> (dx, dy). Order per guideline §7-1 (authoritative).
    # 0: right, 1: up, 2: down, 3: left. Differs from Memory-RL's index
    # order (right/up/left/down) but the semantic 4-direction mapping is
    # what matters; the cross-repo equivalence check translates between
    # them via the same indices used in both repos.
    _ACTION_MAPPING = np.array([
        [+1,  0],   # 0: right
        [ 0, +1],   # 1: up
        [ 0, -1],   # 2: down
        [-1,  0],   # 3: left
    ], dtype=np.int64)

    def __init__(self, corridor_length: int = 10, mode: str = 'passive',
                 n_tasks: int = 2, randomize_tasks: bool = False):
        super().__init__()
        assert corridor_length >= 1, \
            f"corridor_length must be >= 1, got {corridor_length}"
        assert mode in ('passive', 'active'), \
            f"mode must be 'passive' or 'active', got {mode!r}"
        assert n_tasks == 2, \
            f"T-MAZE is a binary-context env; n_tasks must be 2 (guideline §7-4), got {n_tasks}"
        # `randomize_tasks` is accepted (and ignored) only for PEARL config
        # compatibility: configs/default.py sets env_params={'randomize_tasks':
        # True} as a global default, and deep_update_dict merges it into every
        # env_params dict. T-MAZE has two hard-coded contexts so there is
        # nothing to shuffle. PointEnv / HalfCheetahDirEnv / AntGoalEnv follow
        # the same accept-and-ignore convention.
        del randomize_tasks

        self.corridor_length = corridor_length
        self.mode = mode
        if mode == 'passive':
            self.oracle_length = 0          # S = O at x = 0
            self._max_episode_steps = corridor_length + 1   # T = L + 1
        else:  # active
            self.oracle_length = 1          # S = x=1, O = x=0
            # T = L + 3 follows Memory-RL's episode_length = L + 2*oracle_length + 1.
            self._max_episode_steps = corridor_length + 3

        # PEARL: continuous Box action. step() takes argmax to recover the
        # discrete index used internally. Box low/high = -1/+1 makes
        # NormalizedBoxEnv's affine rescale an identity.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation: Memory-RL ambiguous_position=True encoding, shape (2,).
        #   x = 0 (oracle):              [0, exposure]  (first visit only)
        #   0 < x < L (corridor middle): [0, 0]
        #   x = L (junction or arm):     [1, y]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # PEARL: pre-generated goals; reset_task(idx) selects.
        # Index convention: 0 -> goal_y=-1 (DOWN), 1 -> goal_y=+1 (UP).
        self.goals = [-1, +1]

        # gym standard RNG. Used only as a safety net; PEARL's task selection
        # is idx-based, so RNG state is not part of cross-repo equivalence.
        self.np_random = None
        self.seed()

        # MDP state (set properly by reset_task -> reset).
        self._goal_y = None
        self.x = 0
        self.y = 0
        self.step_count = 0
        self.oracle_visited = False

        # PEARL convention (cf. point_robot.PointEnv.__init__): initialise
        # task=0 so the env is usable immediately after construction.
        self.reset_task(0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # =========================================================================
    # Task interface (PEARL convention)
    # =========================================================================

    def reset_task(self, idx):
        """Select the hidden context by index and reset the MDP state.

        PEARL convention (cf. point_robot.PointEnv.reset_task): `idx` is an
        integer in `get_all_task_idx()`, which here is `range(2)`. The hidden
        context `goal_y` is looked up from `self.goals` and the env is then
        reset.
        """
        assert idx in (0, 1), \
            f"idx must be 0 or 1 (n_tasks=2), got {idx!r}"
        self._goal_y = self.goals[idx]
        # PEARL convention alias: rlkit/core/rl_algorithm.py:369 reads
        # `self.env._goal` during evaluate(). point_robot / half_cheetah_dir
        # / ant_dir all set self._goal in reset_task. Numerically identical
        # to _goal_y; learning unaffected.
        self._goal = self._goal_y
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    # =========================================================================
    # gym.Env interface
    # =========================================================================

    def reset(self):
        """Reset the MDP state. Does not change the task."""
        self.x = self.oracle_length  # passive: 0; active: 1
        self.y = 0
        self.step_count = 0
        self.oracle_visited = False
        return self._get_obs()

    def step(self, action):
        # NormalizedBoxEnv hands us a float ndarray in [-1, +1]^4. Sanity-
        # check shape, then collapse to a discrete index via argmax.
        action = np.asarray(action).reshape(-1)
        assert action.shape == (4,), \
            f"action must be 1-d length-4 (Box(4)), got shape {action.shape}"
        idx = int(np.argmax(action))

        self.step_count += 1

        # Attempt move; on wall bump, stay in place.
        dx, dy = self._ACTION_MAPPING[idx]
        nx, ny = self.x + int(dx), self.y + int(dy)
        if self._is_valid(nx, ny):
            self.x, self.y = nx, ny

        done = self.step_count >= self._max_episode_steps
        reward = self._reward(done)
        obs = self._get_obs()
        # Minimal info: task_idx for debugging / visualisation. PEARL itself
        # does not consume info, so this has no learning-side impact.
        info = {'task_idx': self.goals.index(self._goal_y)}
        return obs, float(reward), done, info

    # =========================================================================
    # Internals — §7-2 common items, byte-equal to VariBAD's TMazeEnv.
    # =========================================================================

    def _is_valid(self, x, y):
        """Walls of the T-maze.

        Corridor lies on y=0, x in [0, oracle_length+L]. Goal arms (y = +/-1)
        exist only at x = oracle_length+L (junction column). Anywhere else is
        wall -> agent stays.
        """
        x_max = self.oracle_length + self.corridor_length
        if x < 0 or x > x_max:
            return False
        if y == 0:
            return True
        if y in (-1, +1):
            return x == x_max
        return False

    def _get_obs(self):
        """Ambiguous-position observation per Memory-RL.

        The oracle exposure (the goal_y signal) is shown only on the FIRST
        visit to x=0; subsequent visits return 0. In passive mode the agent
        starts at the oracle, so the first `reset()` returns [0, goal_y] and
        all later observations carry no exposure.
        """
        exposure = 0
        if self.x == 0 and not self.oracle_visited:
            exposure = self._goal_y
            self.oracle_visited = True

        x_junction = self.oracle_length + self.corridor_length
        if self.x == 0:
            obs = np.array([0.0, float(exposure)], dtype=np.float32)
        elif self.x < x_junction:
            obs = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # x == x_junction: junction (y=0) or one of the goal arms (y=±1).
            obs = np.array([1.0, float(self.y)], dtype=np.float32)
        return obs

    def _reward(self, done):
        """Paper Section 3.2, generalized to active via oracle_length grace.

        Intermediate (t < T):  R_t = (1[x_{t+1} >= t - oracle_length] - 1) / (T - 1)
        Terminal     (t = T):  R_T = 1[reached correct goal arm]

        For passive (oracle_length=0) this matches the v6 §3 formula exactly.
        For active (oracle_length=1) the agent gets one step of grace, which
        matches Memory-RL's reward_fn condition `x < t - oracle_length`.

        Here `self.step_count` is t (already incremented in step()), and
        `self.x` / `self.y` are positions AFTER the action — i.e. the paper's
        x_{t+1}, y_{t+1}.
        """
        T = self._max_episode_steps
        if done:
            x_max = self.oracle_length + self.corridor_length
            at_correct_arm = (self.x == x_max) and (self.y == self._goal_y)
            return 1.0 if at_correct_arm else 0.0
        t = self.step_count
        return 0.0 if self.x >= t - self.oracle_length else -1.0 / (T - 1)


# =============================================================================
# PEARL registration: two thin wrappers fix the mode kwarg.
# Pattern mirrors point_robot.PointEnv / SparsePointEnv being registered
# under two distinct env names.
# =============================================================================

@register_env('tmaze-passive')
class TMazePassive(TMazeEnv):
    def __init__(self, **kwargs):
        super().__init__(mode='passive', **kwargs)


@register_env('tmaze-active')
class TMazeActive(TMazeEnv):
    def __init__(self, **kwargs):
        super().__init__(mode='active', **kwargs)
