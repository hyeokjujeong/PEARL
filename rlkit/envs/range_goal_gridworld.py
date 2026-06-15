"""Range-bin hidden-goal GridWorld benchmark.

This environment is designed to stress online belief refinement. The hidden
task is a goal cell. The agent observes the full obstacle map and its own
position, but it does not observe the goal. Instead it receives a coarse bin
of the obstacle-aware shortest-path distance to the goal.

PEARL integration uses a flat Box observation and a Box(4) action where the
argmax is mapped to up/down/left/right, matching the existing T-Maze adapter
style in this repo. A Gymnasium-facing variant is also registered for direct
inspection with Dict observations and Discrete actions.
"""

from collections import deque

import numpy as np
from gymnasium import Env, spaces

from . import register_env


_GYM_ENV_IDS = (
    "RangeGoalGridWorld-LevelA-15x15-v0",
    "RangeGoalGridWorld-LevelA-21x21-v0",
    "RangeGoalGridWorld-LevelA-Ambiguous-15x15-v0",
    "RangeGoalGridWorld-LevelA-Main-RandomGoal-15x15-v0",
    "RangeGoalGridWorld-LevelA-PEARL-Diag-9x9-v0",
)


def quantize_distance(distance, bin_edges):
    """Return the coarse range bin for a shortest-path distance."""
    for idx in range(len(bin_edges) - 1):
        if bin_edges[idx] <= distance < bin_edges[idx + 1]:
            return idx
    return len(bin_edges) - 1


def bfs_distance_field(grid, source):
    """Shortest-path distance from every cell to source on free cells."""
    height, width = grid.shape
    distances = np.full((height, width), np.inf, dtype=np.float32)
    if grid[source] != 0:
        return distances

    queue = deque([tuple(source)])
    distances[source] = 0.0
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        i, j = queue.popleft()
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= height or nj < 0 or nj >= width:
                continue
            if grid[ni, nj] != 0 or np.isfinite(distances[ni, nj]):
                continue
            distances[ni, nj] = distances[i, j] + 1.0
            queue.append((ni, nj))
    return distances


def is_connected_free_space(grid):
    free_cells = np.argwhere(grid == 0)
    if len(free_cells) == 0:
        return False
    distances = bfs_distance_field(grid, tuple(free_cells[0]))
    return np.all(np.isfinite(distances[grid == 0]))


def count_posterior_modes(posterior, threshold_ratio=0.1):
    """Count connected high-probability components in a posterior map."""
    max_prob = float(np.max(posterior))
    if max_prob <= 0.0:
        return 0
    support = posterior >= threshold_ratio * max_prob
    visited = np.zeros_like(support, dtype=bool)
    modes = 0
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for i in range(support.shape[0]):
        for j in range(support.shape[1]):
            if not support[i, j] or visited[i, j]:
                continue
            modes += 1
            queue = deque([(i, j)])
            visited[i, j] = True
            while queue:
                ci, cj = queue.popleft()
                for di, dj in moves:
                    ni, nj = ci + di, cj + dj
                    if ni < 0 or ni >= support.shape[0] or nj < 0 or nj >= support.shape[1]:
                        continue
                    if visited[ni, nj] or not support[ni, nj]:
                        continue
                    visited[ni, nj] = True
                    queue.append((ni, nj))
    return modes


def posterior_entropy(posterior, eps=1e-12):
    mask = posterior > 0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(posterior[mask] * np.log(posterior[mask] + eps)))


def make_four_room(size):
    grid = np.zeros((size, size), dtype=np.int8)
    mid = size // 2
    grid[mid, :] = 1
    grid[:, mid] = 1
    doors = [
        (mid, size // 4),
        (mid, 3 * size // 4),
        (size // 4, mid),
        (3 * size // 4, mid),
    ]
    for door in doors:
        grid[door] = 0
    return grid


def make_u_shape(size):
    grid = np.zeros((size, size), dtype=np.int8)
    top = max(2, size // 4)
    bottom = min(size - 3, 3 * size // 4)
    left = max(2, size // 4)
    right = min(size - 3, 3 * size // 4)
    grid[top:bottom + 1, left] = 1
    grid[top:bottom + 1, right] = 1
    grid[bottom, left:right + 1] = 1
    grid[top, (left + right) // 2] = 1
    grid[bottom, (left + right) // 2] = 0
    grid[top, left] = 0
    grid[top, right] = 0
    return grid


def make_multi_corridor(size):
    grid = np.zeros((size, size), dtype=np.int8)
    wall_cols = [size // 3, 2 * size // 3]
    for k, col in enumerate(wall_cols):
        grid[1:size - 1, col] = 1
        for door_row in (2 + k, size // 2, size - 3 - k):
            grid[door_row, col] = 0
    return grid


def make_wall_door(size):
    grid = np.zeros((size, size), dtype=np.int8)
    row = size // 2
    grid[row, 1:size - 1] = 1
    grid[row, size // 4] = 0
    grid[row, 3 * size // 4] = 0
    return grid


def make_empty(size):
    return np.zeros((size, size), dtype=np.int8)


def make_diag_two_room(size):
    grid = np.zeros((size, size), dtype=np.int8)
    col = size // 2
    grid[1:size - 1, col] = 1
    grid[size // 2, col] = 0
    return grid


def make_map(size, family, map_id=0):
    if family in ("diag", "diag_two_room"):
        return make_diag_two_room(size)
    if family == "four_room":
        return make_four_room(size)
    if family == "u_shape":
        return make_u_shape(size)
    if family == "multi_corridor":
        return make_multi_corridor(size)
    if family == "wall_door":
        return make_wall_door(size)
    if family == "empty":
        return make_empty(size)
    if family in ("mixed", "ambiguous"):
        families = ("four_room", "u_shape", "multi_corridor", "wall_door")
        return make_map(size, families[int(map_id) % len(families)], map_id)
    raise ValueError("Unknown map family: {}".format(family))


def render_grid_rgb(grid, agent_pos, goal_pos=None, posterior=None,
                    show_goal=False, cell_size=24):
    height, width = grid.shape
    image = np.ones((height * cell_size, width * cell_size, 3), dtype=np.uint8) * 255
    for i in range(height):
        for j in range(width):
            patch = image[i * cell_size:(i + 1) * cell_size,
                          j * cell_size:(j + 1) * cell_size]
            if grid[i, j] == 1:
                patch[:, :] = np.array([70, 70, 70], dtype=np.uint8)

    if posterior is not None and np.max(posterior) > 0:
        heat = posterior / (np.max(posterior) + 1e-12)
        color = np.array([255, 180, 0], dtype=np.float32)
        for i in range(height):
            for j in range(width):
                alpha = min(0.7, float(heat[i, j]))
                if alpha <= 0.0 or grid[i, j] != 0:
                    continue
                patch = image[i * cell_size:(i + 1) * cell_size,
                              j * cell_size:(j + 1) * cell_size].astype(np.float32)
                image[i * cell_size:(i + 1) * cell_size,
                      j * cell_size:(j + 1) * cell_size] = (
                    (1.0 - alpha) * patch + alpha * color
                ).astype(np.uint8)

    def draw_cell(pos, color):
        i, j = pos
        margin = max(2, cell_size // 8)
        image[i * cell_size + margin:(i + 1) * cell_size - margin,
              j * cell_size + margin:(j + 1) * cell_size - margin] = np.array(color, dtype=np.uint8)

    draw_cell(agent_pos, (0, 80, 255))
    if show_goal and goal_pos is not None:
        draw_cell(goal_pos, (255, 0, 0))

    image[::cell_size, :, :] = 180
    image[:, ::cell_size, :] = 180
    return image


class RangeGoalGridWorldEnv(Env):
    """Hidden-goal range-bin GridWorld.

    Parameters controlling API shape:
      pearl_api=True returns old-style reset/step used by this PEARL repo.
      flat_observation=True returns a flat vector suitable for MLP policies.
      continuous_actions=True exposes Box(4), with argmax selecting direction.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    ACTION_TO_DELTA = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(
            self,
            size=15,
            horizon=30,
            n_tasks=20,
            randomize_tasks=True,
            seed=1337,
            map_family="mixed",
            bin_edges=(0, 5, 10, 15, 20),
            reward_type="sparse",
            reward_mode=None,
            progress_scale=0.5,
            goal_reward=50.0,
            step_penalty=-1.0,
            collision_penalty=-0.2,
            start_pos=None,
            candidate_goals=None,
            num_candidate_goals=None,
            candidate_goal_mode="landmarks",
            min_start_goal_dist=8,
            min_initial_support=12,
            min_initial_modes=2,
            max_sampling_attempts=10000,
            obs_mode=None,
            action_mode="hard_argmax",
            action_temperature=0.5,
            mask_invalid_actions=False,
            noisy_bin=False,
            noise_sigma=0.0,
            include_oracle_in_info=False,
            render_mode=None,
            flat_observation=True,
            continuous_actions=True,
            pearl_api=True,
    ):
        super().__init__()
        self.size = int(size)
        self.horizon = int(horizon)
        self.n_tasks = int(n_tasks)
        self.map_family = map_family
        self.bin_edges = tuple(float(x) for x in bin_edges)
        self.num_bins = len(self.bin_edges)
        self.reward_type = reward_type
        self.reward_mode = reward_mode or reward_type
        self.progress_scale = float(progress_scale)
        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.collision_penalty = float(collision_penalty)
        self.fixed_start_pos = tuple(start_pos) if start_pos is not None else None
        self.candidate_goals = (
            [tuple(goal) for goal in candidate_goals]
            if candidate_goals is not None else None
        )
        self.num_candidate_goals = (
            int(num_candidate_goals) if num_candidate_goals is not None else None
        )
        self.candidate_goal_mode = candidate_goal_mode
        self.min_start_goal_dist = int(min_start_goal_dist)
        self.min_initial_support = int(min_initial_support)
        self.min_initial_modes = int(min_initial_modes)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.noisy_bin = bool(noisy_bin)
        self.noise_sigma = float(noise_sigma)
        self.include_oracle_in_info = bool(include_oracle_in_info)
        self.render_mode = render_mode
        self.flat_observation = bool(flat_observation)
        self.continuous_actions = bool(continuous_actions)
        if obs_mode is None:
            obs_mode = "full_flat" if self.flat_observation else "dict"
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.action_temperature = float(action_temperature)
        self.mask_invalid_actions = bool(mask_invalid_actions)
        self.force_argmax_action = False
        self.pearl_api = bool(pearl_api)
        self._rng = np.random.RandomState(seed)

        if self.continuous_actions:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)
        self.observation_space = self._build_observation_space()

        self.tasks = self._build_tasks(self._rng)
        if randomize_tasks:
            self._rng.shuffle(self.tasks)
        self._task_idx = 0
        self._task = self.tasks[0]
        self._goal = tuple(self._task["goal_pos"])

        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.goal_distance_field = None
        self.t = 0
        self.start_distance = 0.0
        self.min_distance = 0.0
        self.collision_count = 0
        self.invalid_action_count = 0
        self.invalid_action_mass_total = 0.0
        self.action_counts = np.zeros(4, dtype=np.int64)
        self.raw_action_counts = np.zeros(4, dtype=np.int64)
        self.last_action_probs = np.ones(4, dtype=np.float32) / 4.0
        self.last_raw_action_idx = 0
        self.last_invalid_action_requested = False
        self.last_invalid_action_mass = 0.0
        self.state_history = []
        self.action_history = []
        self.range_bin_history = []

        self.reset_task(0)

    def _build_observation_space(self):
        if not self.flat_observation or self.obs_mode == "dict":
            return spaces.Dict({
                "agent_pos": spaces.Box(
                    low=0,
                    high=self.size - 1,
                    shape=(2,),
                    dtype=np.int64,
                ),
                "map": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.size, self.size),
                    dtype=np.int8,
                ),
                "range_bin": spaces.Discrete(self.num_bins),
            })
        if self.obs_mode == "pearl_lowdim":
            low = np.zeros(2 + self.num_bins + 4, dtype=np.float32)
            high = np.ones(2 + self.num_bins + 4, dtype=np.float32)
            return spaces.Box(low=low, high=high, dtype=np.float32)
        if self.obs_mode in ("full_flat_with_mask", "map_with_mask"):
            low = np.concatenate([
                -np.ones(2, dtype=np.float32),
                np.zeros(self.size * self.size + self.num_bins + 4, dtype=np.float32),
            ])
            high = np.concatenate([
                np.ones(2, dtype=np.float32),
                np.ones(self.size * self.size + self.num_bins + 4, dtype=np.float32),
            ])
            return spaces.Box(low=low, high=high, dtype=np.float32)

        low = np.concatenate([
            -np.ones(2, dtype=np.float32),
            np.zeros(self.size * self.size + self.num_bins, dtype=np.float32),
        ])
        high = np.concatenate([
            np.ones(2, dtype=np.float32),
            np.ones(self.size * self.size + self.num_bins, dtype=np.float32),
        ])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _build_tasks(self, rng):
        tasks = []
        shared_grid = None
        shared_goals = None
        shared_start = None
        used_random_goals = set()
        random_goal_mode = self.candidate_goal_mode in ("random", "random_free", "free")
        if (
                not random_goal_mode
                and (self.candidate_goals is not None or self.num_candidate_goals is not None)
        ):
            shared_grid = make_map(self.size, self._family_for_task(0), 0)
            self._validate_map(shared_grid)
            shared_start = self._default_start(shared_grid)
            shared_goals = self._build_candidate_goals(shared_grid, shared_start)

        for task_idx in range(self.n_tasks):
            family = self._family_for_task(task_idx)
            grid = np.array(shared_grid, copy=True) if shared_grid is not None else make_map(self.size, family, task_idx)
            self._validate_map(grid)
            if shared_goals is not None:
                start = shared_start
                goal = shared_goals[task_idx % len(shared_goals)]
                support_count, mode_count = self._initial_ambiguity_stats(grid, start, goal)
            elif random_goal_mode:
                start = self._default_start(grid)
                goal, support_count, mode_count = self._sample_goal_for_start(
                    grid,
                    start,
                    rng,
                    excluded_goals=used_random_goals,
                )
            else:
                start, goal, support_count, mode_count = self._sample_start_goal(grid, rng)
            if random_goal_mode:
                used_random_goals.add(goal)
            tasks.append({
                "task_idx": task_idx,
                "map_id": task_idx,
                "map_family": family,
                "map": grid,
                "start_pos": start,
                "goal_pos": goal,
                "initial_support_count": support_count,
                "initial_mode_count": mode_count,
            })
        return tasks

    def _family_for_task(self, task_idx):
        if self.map_family == "ambiguous":
            families = ("four_room", "u_shape", "multi_corridor", "wall_door")
            return families[int(task_idx) % len(families)]
        if self.map_family == "mixed":
            families = ("four_room", "u_shape", "multi_corridor", "wall_door")
            return families[int(task_idx) % len(families)]
        return self.map_family

    def _sample_start_goal(self, grid, rng):
        free_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == 0)]
        best = None
        for _ in range(self.max_sampling_attempts):
            start = free_cells[int(rng.randint(len(free_cells)))]
            goal = free_cells[int(rng.randint(len(free_cells)))]
            if start == goal:
                continue
            dist_field = bfs_distance_field(grid, goal)
            distance = dist_field[start]
            if not np.isfinite(distance) or distance < self.min_start_goal_dist:
                continue
            range_bin = quantize_distance(distance, self.bin_edges)
            support = compute_likelihood_support(grid, start, range_bin, self.bin_edges)
            support_count = int(np.sum(support > 0))
            posterior = support / max(1.0, float(support_count))
            mode_count = count_posterior_modes(posterior)
            candidate = (start, goal, support_count, mode_count)
            best = candidate
            if support_count >= self.min_initial_support and mode_count >= self.min_initial_modes:
                return candidate
        if best is not None:
            return best
        raise RuntimeError("Failed to sample a valid RangeGoalGridWorld task.")

    def _sample_goal_for_start(self, grid, start, rng, excluded_goals=None):
        excluded_goals = excluded_goals or set()
        free_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == 0)]
        best = None
        for _ in range(self.max_sampling_attempts):
            goal = free_cells[int(rng.randint(len(free_cells)))]
            if goal == start or goal in excluded_goals:
                continue
            distance = bfs_distance_field(grid, goal)[start]
            if not np.isfinite(distance) or distance < self.min_start_goal_dist:
                continue
            range_bin = quantize_distance(distance, self.bin_edges)
            support = compute_likelihood_support(grid, start, range_bin, self.bin_edges)
            support_count = int(np.sum(support > 0))
            posterior = support / max(1.0, float(support_count))
            mode_count = count_posterior_modes(posterior)
            candidate = (goal, support_count, mode_count)
            best = candidate
            if support_count >= self.min_initial_support and mode_count >= self.min_initial_modes:
                return candidate
        if best is not None:
            return best
        raise RuntimeError("Failed to sample a valid random goal for RangeGoalGridWorld.")

    def _default_start(self, grid):
        if self.fixed_start_pos is not None:
            start = tuple(int(x) for x in self.fixed_start_pos)
            if grid[start] != 0:
                raise ValueError("Configured start_pos is not a free cell.")
            return start
        center = (self.size // 2, self.size // 2)
        if grid[center] == 0:
            return center
        free_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == 0)]
        return min(free_cells, key=lambda c: abs(c[0] - center[0]) + abs(c[1] - center[1]))

    def _build_candidate_goals(self, grid, start):
        if self.candidate_goals is not None:
            goals = [tuple(int(x) for x in goal) for goal in self.candidate_goals]
        elif self.candidate_goal_mode == "diag":
            goals = [
                (1, 1),
                (self.size - 2, self.size - 2),
                (1, self.size - 2),
                (self.size - 2, 1),
                (1, self.size // 2),
                (self.size - 2, self.size // 2),
                (self.size // 2, 1),
                (self.size // 2, self.size - 2),
            ]
        else:
            goals = [
                (1, 1),
                (1, self.size - 2),
                (self.size - 2, 1),
                (self.size - 2, self.size - 2),
                (self.size // 3, self.size // 3),
                (self.size // 3, 2 * self.size // 3),
                (2 * self.size // 3, self.size // 3),
                (2 * self.size // 3, 2 * self.size // 3),
            ]

        valid = []
        for goal in goals:
            goal = tuple(int(x) for x in goal)
            if goal == start:
                continue
            if goal[0] < 0 or goal[0] >= self.size or goal[1] < 0 or goal[1] >= self.size:
                continue
            if grid[goal] != 0:
                continue
            distance = bfs_distance_field(grid, goal)[start]
            if np.isfinite(distance) and distance >= self.min_start_goal_dist:
                valid.append(goal)
        if self.num_candidate_goals is not None:
            valid = valid[:self.num_candidate_goals]
        if not valid:
            raise RuntimeError("No valid candidate goals for RangeGoalGridWorld.")
        return valid

    def _initial_ambiguity_stats(self, grid, start, goal):
        distance = bfs_distance_field(grid, goal)[start]
        range_bin = quantize_distance(float(distance), self.bin_edges)
        support = compute_likelihood_support(grid, start, range_bin, self.bin_edges)
        support_count = int(np.sum(support > 0))
        posterior = support / max(1.0, float(support_count))
        return support_count, count_posterior_modes(posterior)

    def _validate_map(self, grid):
        if grid.shape != (self.size, self.size):
            raise ValueError("Map shape must be ({}, {}).".format(self.size, self.size))
        if not np.all((grid == 0) | (grid == 1)):
            raise ValueError("Map must contain only 0=free and 1=obstacle.")
        if not is_connected_free_space(grid):
            raise ValueError("Map free space must be connected.")

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def set_force_argmax_action(self, enabled):
        old_value = self.force_argmax_action
        self.force_argmax_action = bool(enabled)
        return old_value

    def reset_task(self, idx):
        self._task_idx = int(idx)
        self._task = self.tasks[self._task_idx]
        self._goal = tuple(self._task["goal_pos"])
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        if options is not None and "task_id" in options:
            self._task_idx = int(options["task_id"])
            self._task = self.tasks[self._task_idx]

        task = dict(self._task)
        if options is not None:
            if "map" in options:
                task["map"] = np.asarray(options["map"], dtype=np.int8)
            if "start_pos" in options:
                task["start_pos"] = tuple(int(x) for x in options["start_pos"])
            if "goal_pos" in options:
                task["goal_pos"] = tuple(int(x) for x in options["goal_pos"])

        self.grid = np.array(task["map"], dtype=np.int8, copy=True)
        self.agent_pos = tuple(int(x) for x in task["start_pos"])
        self.goal_pos = tuple(int(x) for x in task["goal_pos"])
        self._goal = self.goal_pos
        self.goal_distance_field = bfs_distance_field(self.grid, self.goal_pos)
        self.t = 0
        self.start_distance = float(self.goal_distance_field[self.agent_pos])
        self.min_distance = self.start_distance
        self.collision_count = 0
        self.invalid_action_count = 0
        self.invalid_action_mass_total = 0.0
        self.action_counts = np.zeros(4, dtype=np.int64)
        self.raw_action_counts = np.zeros(4, dtype=np.int64)
        self.last_action_probs = np.ones(4, dtype=np.float32) / 4.0
        self.last_raw_action_idx = 0
        self.last_invalid_action_requested = False
        self.last_invalid_action_mass = 0.0
        self.state_history = [self.agent_pos]
        self.action_history = []
        self.range_bin_history = [self._compute_range_bin(self.agent_pos)]
        obs = self._get_obs()
        info = self._get_info()
        if self.pearl_api:
            return obs
        return obs, info

    def step(self, action):
        action_idx = self._action_to_index(action)
        old_range_bin = int(self.range_bin_history[-1])
        old_pos = self.agent_pos
        old_distance = float(self.goal_distance_field[old_pos])
        new_pos, collided = self._transition(old_pos, action_idx)
        self.agent_pos = new_pos
        self.t += 1

        range_bin = self._compute_range_bin(self.agent_pos)
        range_bin_delta = old_range_bin - range_bin
        current_distance = float(self.goal_distance_field[self.agent_pos])
        distance_delta = old_distance - current_distance
        self.min_distance = min(self.min_distance, current_distance)
        self.collision_count += int(collided)
        self.action_counts[action_idx] += 1
        self.state_history.append(self.agent_pos)
        self.action_history.append(action_idx)
        self.range_bin_history.append(range_bin)

        reached_goal = self.agent_pos == self.goal_pos
        reward = self._compute_reward(
            reached_goal,
            collided,
            range_bin_delta=range_bin_delta,
            distance_delta=distance_delta,
        )
        terminated = bool(reached_goal)
        truncated = bool(self.t >= self.horizon)
        done = terminated or truncated
        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "collided": bool(collided),
            "reached_goal": bool(reached_goal),
            "old_pos": old_pos,
            "new_pos": new_pos,
            "action": action_idx,
            "executed_direction": action_idx,
            "raw_action": int(self.last_raw_action_idx),
            "invalid_action_requested": bool(self.last_invalid_action_requested),
            "invalid_action_mass": float(self.last_invalid_action_mass),
            "action_probs": self.last_action_probs.copy(),
            "range_bin_delta": int(range_bin_delta),
            "distance_delta": float(distance_delta),
            "start_distance": float(self.start_distance),
            "final_distance": current_distance,
            "min_distance": float(self.min_distance),
            "distance_reduction": float(self.start_distance - current_distance),
            "collision_count": int(self.collision_count),
            "invalid_action_count": int(self.invalid_action_count),
            "invalid_action_rate": float(self.invalid_action_count) / float(max(1, self.t)),
            "invalid_action_mass_mean": float(self.invalid_action_mass_total) / float(max(1, self.t)),
            "episode_length": int(self.t),
            "t": self.t,
            "terminated": terminated,
            "truncated": truncated,
        })
        if self.pearl_api:
            return obs, float(reward), done, info
        return obs, float(reward), terminated, truncated, info

    def _action_to_index(self, action):
        valid_mask = self._valid_action_mask(self.agent_pos).astype(bool)
        if self.continuous_actions:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (4,):
                raise ValueError("Continuous action must have shape (4,), got {}".format(action.shape))
            raw_idx = int(np.argmax(action))
            invalid_requested = bool(not valid_mask[raw_idx])
            if self.action_mode == "softmax_stochastic" and not self.force_argmax_action:
                logits = np.clip(action, -5.0, 5.0) / max(self.action_temperature, 1e-6)
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / np.sum(probs)
                invalid_mass = float(np.sum(probs[~valid_mask]))
                if self.mask_invalid_actions:
                    probs = probs * valid_mask.astype(np.float32)
                    prob_sum = float(np.sum(probs))
                    if prob_sum > 0.0:
                        probs = probs / prob_sum
                    else:
                        probs = valid_mask.astype(np.float32) / float(np.sum(valid_mask))
                self.last_action_probs = probs.astype(np.float32)
                action_idx = int(self._rng.choice(4, p=probs))
            else:
                scores = np.asarray(action, dtype=np.float32)
                invalid_mass = 1.0 if invalid_requested else 0.0
                if self.mask_invalid_actions and invalid_requested:
                    masked_scores = scores.copy()
                    masked_scores[~valid_mask] = -np.inf
                    action_idx = int(np.argmax(masked_scores))
                else:
                    action_idx = raw_idx
                self.last_action_probs = np.eye(4, dtype=np.float32)[action_idx]
            self.raw_action_counts[raw_idx] += 1
            self.invalid_action_count += int(invalid_requested)
            self.invalid_action_mass_total += float(invalid_mass)
            self.last_raw_action_idx = raw_idx
            self.last_invalid_action_requested = invalid_requested
            self.last_invalid_action_mass = float(invalid_mass)
            return action_idx
        raw_idx = int(action)
        invalid_requested = bool(not valid_mask[raw_idx])
        if self.mask_invalid_actions and invalid_requested:
            action_idx = int(np.argmax(valid_mask.astype(np.float32)))
        else:
            action_idx = raw_idx
        self.raw_action_counts[raw_idx] += 1
        self.invalid_action_count += int(invalid_requested)
        self.invalid_action_mass_total += 1.0 if invalid_requested else 0.0
        self.last_raw_action_idx = raw_idx
        self.last_invalid_action_requested = invalid_requested
        self.last_invalid_action_mass = 1.0 if invalid_requested else 0.0
        self.last_action_probs = np.eye(4, dtype=np.float32)[action_idx]
        return action_idx

    def _transition(self, pos, action_idx):
        di, dj = self.ACTION_TO_DELTA[int(action_idx)]
        ni, nj = pos[0] + di, pos[1] + dj
        if ni < 0 or ni >= self.size or nj < 0 or nj >= self.size:
            return pos, True
        if self.grid[ni, nj] == 1:
            return pos, True
        return (int(ni), int(nj)), False

    def _compute_range_bin(self, pos):
        distance = float(self.goal_distance_field[pos])
        if self.noisy_bin:
            distance += float(self._rng.normal(0.0, self.noise_sigma))
        return int(quantize_distance(distance, self.bin_edges))

    def _compute_reward(self, reached_goal, collided, range_bin_delta=0, distance_delta=0.0):
        if reached_goal:
            return self.goal_reward
        if self.reward_mode in ("distance_delta", "dense_distance", "progress_distance"):
            reward = self.step_penalty + self.progress_scale * float(distance_delta)
            if collided:
                reward += self.collision_penalty
            return reward
        if self.reward_mode in ("progress_bin", "bin_dense"):
            reward = self.step_penalty + self.progress_scale * float(range_bin_delta)
            if collided:
                reward += self.collision_penalty
            return reward
        reward = self.step_penalty
        if collided:
            reward += self.collision_penalty
        return reward

    def _get_obs(self):
        range_bin = int(self.range_bin_history[-1])
        if not self.flat_observation or self.obs_mode == "dict":
            return {
                "agent_pos": np.array(self.agent_pos, dtype=np.int64),
                "map": self.grid.astype(np.int8, copy=True),
                "range_bin": range_bin,
            }
        one_hot = np.zeros(self.num_bins, dtype=np.float32)
        one_hot[range_bin] = 1.0
        if self.obs_mode == "pearl_lowdim":
            agent = np.asarray(self.agent_pos, dtype=np.float32) / float(max(1, self.size - 1))
            return np.concatenate([
                agent.astype(np.float32),
                one_hot,
                self._valid_action_mask(self.agent_pos),
            ]).astype(np.float32)
        if self.obs_mode in ("full_flat_with_mask", "map_with_mask"):
            agent = np.asarray(self.agent_pos, dtype=np.float32)
            if self.size > 1:
                agent = 2.0 * agent / float(self.size - 1) - 1.0
            return np.concatenate([
                agent.astype(np.float32),
                self.grid.astype(np.float32).reshape(-1),
                one_hot,
                self._valid_action_mask(self.agent_pos),
            ]).astype(np.float32)

        agent = np.asarray(self.agent_pos, dtype=np.float32)
        if self.size > 1:
            agent = 2.0 * agent / float(self.size - 1) - 1.0
        return np.concatenate([
            agent.astype(np.float32),
            self.grid.astype(np.float32).reshape(-1),
            one_hot,
        ]).astype(np.float32)

    def _valid_action_mask(self, pos):
        mask = np.zeros(4, dtype=np.float32)
        for idx in range(4):
            next_pos, collided = self._transition(pos, idx)
            del next_pos
            mask[idx] = 0.0 if collided else 1.0
        return mask

    def _get_info(self):
        distance = float(self.goal_distance_field[self.agent_pos])
        info = {
            "task_idx": self._task_idx,
            "goal_pos": self.goal_pos,
            "start_pos": tuple(self._task["start_pos"]),
            "distance_to_goal": distance,
            "range_bin": int(self.range_bin_history[-1]),
            "range_bin_start": int(self.range_bin_history[0]),
            "range_bin_min": int(np.min(self.range_bin_history)),
            "state_history": list(self.state_history),
            "action_history": list(self.action_history),
            "range_bin_history": list(self.range_bin_history),
            "map": self.grid.copy(),
            "agent_pos": self.agent_pos,
            "start_distance": float(self.start_distance),
            "min_distance": float(self.min_distance),
            "distance_reduction": float(self.start_distance - distance),
            "collision_count": int(self.collision_count),
            "invalid_action_count": int(self.invalid_action_count),
            "invalid_action_rate": float(self.invalid_action_count) / float(max(1, self.t)),
            "invalid_action_mass_mean": float(self.invalid_action_mass_total) / float(max(1, self.t)),
            "episode_length": int(self.t),
            "action_counts": self.action_counts.copy(),
            "raw_action_counts": self.raw_action_counts.copy(),
            "initial_support_count": int(self._task.get("initial_support_count", 0)),
            "initial_mode_count": int(self._task.get("initial_mode_count", 0)),
        }
        if self.include_oracle_in_info:
            posterior = self.compute_oracle_posterior()
            info["oracle_posterior"] = posterior
            info["oracle_entropy"] = posterior_entropy(posterior)
            info["oracle_mode_count"] = count_posterior_modes(posterior)
        return info

    def compute_likelihood_support(self, state=None, range_bin=None):
        if state is None:
            state = self.agent_pos
        if range_bin is None:
            range_bin = self.range_bin_history[-1]
        return compute_likelihood_support(self.grid, tuple(state), int(range_bin), self.bin_edges)

    def compute_oracle_posterior(self):
        return compute_oracle_posterior(
            self.grid,
            self.state_history,
            self.range_bin_history,
            self.bin_edges,
        )

    def render(self):
        image = render_grid_rgb(
            self.grid,
            self.agent_pos,
            goal_pos=self.goal_pos,
            posterior=None,
            show_goal=False,
        )
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.axis("off")
            plt.pause(1.0 / self.metadata["render_fps"])
            return None
        return image

    def render_debug(self, show_goal=True, show_oracle=True):
        posterior = self.compute_oracle_posterior() if show_oracle else None
        return render_grid_rgb(
            self.grid,
            self.agent_pos,
            goal_pos=self.goal_pos,
            posterior=posterior,
            show_goal=show_goal,
        )


def compute_likelihood_support(grid, state, range_bin, bin_edges):
    support = np.zeros_like(grid, dtype=np.float32)
    for cell in np.argwhere(grid == 0):
        goal = tuple(map(int, cell))
        distances = bfs_distance_field(grid, goal)
        predicted_bin = quantize_distance(float(distances[state]), bin_edges)
        if predicted_bin == int(range_bin):
            support[goal] = 1.0
    return support


def compute_oracle_posterior(grid, states, range_bins, bin_edges):
    posterior = np.zeros_like(grid, dtype=np.float32)
    for cell in np.argwhere(grid == 0):
        goal = tuple(map(int, cell))
        distances = bfs_distance_field(grid, goal)
        possible = True
        for state, observed_bin in zip(states, range_bins):
            predicted_bin = quantize_distance(float(distances[tuple(state)]), bin_edges)
            if predicted_bin != int(observed_bin):
                possible = False
                break
        if possible:
            posterior[goal] = 1.0
    total = float(np.sum(posterior))
    if total > 0.0:
        posterior /= total
    return posterior


@register_env("range-goal-gridworld")
class RangeGoalGridWorldPearlEnv(RangeGoalGridWorldEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("flat_observation", True)
        kwargs.setdefault("continuous_actions", True)
        kwargs.setdefault("pearl_api", True)
        super().__init__(**kwargs)


@register_env("range-goal-gridworld-ambiguous")
class RangeGoalGridWorldAmbiguousPearlEnv(RangeGoalGridWorldEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("map_family", "ambiguous")
        kwargs.setdefault("flat_observation", True)
        kwargs.setdefault("continuous_actions", True)
        kwargs.setdefault("pearl_api", True)
        super().__init__(**kwargs)


@register_env("range-goal-gridworld-diag")
class RangeGoalGridWorldDiagPearlEnv(RangeGoalGridWorldEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", 9)
        kwargs.setdefault("horizon", 20)
        kwargs.setdefault("map_family", "diag")
        kwargs.setdefault("bin_edges", (0, 3, 6, 9, 12))
        kwargs.setdefault("start_pos", (4, 4))
        kwargs.setdefault("num_candidate_goals", None)
        kwargs.setdefault("candidate_goal_mode", "random")
        kwargs.setdefault("min_start_goal_dist", 4)
        kwargs.setdefault("min_initial_support", 2)
        kwargs.setdefault("min_initial_modes", 1)
        kwargs.setdefault("obs_mode", "pearl_lowdim")
        kwargs.setdefault("action_mode", "softmax_stochastic")
        kwargs.setdefault("action_temperature", 0.5)
        kwargs.setdefault("reward_mode", "progress_bin")
        kwargs.setdefault("flat_observation", True)
        kwargs.setdefault("continuous_actions", True)
        kwargs.setdefault("pearl_api", True)
        super().__init__(**kwargs)


@register_env("range-goal-gridworld-main")
class RangeGoalGridWorldMainPearlEnv(RangeGoalGridWorldEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", 15)
        kwargs.setdefault("horizon", 40)
        kwargs.setdefault("map_family", "four_room")
        kwargs.setdefault("bin_edges", (0, 4, 8, 12, 16, 20))
        kwargs.setdefault("start_pos", (7, 3))
        kwargs.setdefault("num_candidate_goals", None)
        kwargs.setdefault("candidate_goal_mode", "random")
        kwargs.setdefault("min_start_goal_dist", 8)
        kwargs.setdefault("min_initial_support", 16)
        kwargs.setdefault("min_initial_modes", 2)
        kwargs.setdefault("obs_mode", "pearl_lowdim")
        kwargs.setdefault("action_mode", "softmax_stochastic")
        kwargs.setdefault("action_temperature", 0.5)
        kwargs.setdefault("reward_mode", "progress_bin")
        kwargs.setdefault("flat_observation", True)
        kwargs.setdefault("continuous_actions", True)
        kwargs.setdefault("pearl_api", True)
        super().__init__(**kwargs)


class RangeGoalGridWorldGymEnv(RangeGoalGridWorldEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("flat_observation", False)
        kwargs.setdefault("continuous_actions", False)
        kwargs.setdefault("pearl_api", False)
        super().__init__(**kwargs)


def _register_gymnasium_envs():
    try:
        from gymnasium.envs.registration import register, registry
    except Exception:
        return

    specs = {
        "RangeGoalGridWorld-LevelA-15x15-v0": dict(
            size=15,
            horizon=30,
            map_family="mixed",
            bin_edges=(0, 5, 10, 15, 20),
        ),
        "RangeGoalGridWorld-LevelA-21x21-v0": dict(
            size=21,
            horizon=50,
            map_family="mixed",
            bin_edges=(0, 6, 12, 18, 24),
            min_start_goal_dist=12,
            min_initial_support=20,
        ),
        "RangeGoalGridWorld-LevelA-Ambiguous-15x15-v0": dict(
            size=15,
            horizon=30,
            map_family="ambiguous",
            bin_edges=(0, 5, 10, 15, 20),
            min_initial_support=20,
            min_initial_modes=2,
        ),
        "RangeGoalGridWorld-LevelA-Main-RandomGoal-15x15-v0": dict(
            size=15,
            horizon=40,
            map_family="four_room",
            bin_edges=(0, 4, 8, 12, 16, 20),
            start_pos=(7, 3),
            num_candidate_goals=None,
            candidate_goal_mode="random",
            min_start_goal_dist=8,
            min_initial_support=16,
            min_initial_modes=2,
            obs_mode="dict",
            action_mode="softmax_stochastic",
            reward_mode="progress_bin",
        ),
        "RangeGoalGridWorld-LevelA-PEARL-Diag-9x9-v0": dict(
            size=9,
            horizon=20,
            map_family="diag",
            bin_edges=(0, 3, 6, 9, 12),
            start_pos=(4, 4),
            num_candidate_goals=None,
            candidate_goal_mode="random",
            min_start_goal_dist=4,
            min_initial_support=2,
            min_initial_modes=1,
            obs_mode="dict",
            action_mode="softmax_stochastic",
            reward_mode="progress_bin",
        ),
    }
    for env_id, kwargs in specs.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="rlkit.envs.range_goal_gridworld:RangeGoalGridWorldGymEnv",
            kwargs=kwargs,
        )


_register_gymnasium_envs()
