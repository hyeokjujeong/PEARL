import numpy as np


class PEARLTaskEnvAdapter(object):
    """Small adapter documenting the PEARL task API used by VariBADAlgorithm."""

    def __init__(self, env, task_indices, max_path_length, max_rollouts_per_task):
        self.env = env
        self.task_indices = list(task_indices)
        self.max_path_length = max_path_length
        self.max_rollouts_per_task = max_rollouts_per_task
        self._task_cursor = 0

    def sample_task(self):
        return int(np.random.choice(self.task_indices))

    def reset_task(self, task_idx):
        self.env.reset_task(task_idx)
        return self.env.reset()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

