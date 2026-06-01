import numpy as np
import mujoco

from .mujoco_env import MujocoEnv

RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']


class RandomParamEnv(MujocoEnv):
    """MuJoCo env whose *tasks* are random rescalings of model parameters.

    Modern re-implementation of ``rand_param_envs.base.RandomEnv``
    (dennisl88, MuJoCo131 / mujoco-py 1.50) on the DeepMind ``mujoco`` bindings.
    Instead of poking the model through the old mujoco-py API, it writes the
    (directly-writable) ``MjModel`` arrays and calls ``mj_setConst`` /
    ``mj_forward`` to refresh derived quantities.

    Task = a dict of rescaled parameter arrays. The multiplier scheme matches
    the original: log-uniform in ``1.5**[-k, k]`` for mass / inertia / friction
    and ``1.3**[-k, k]`` for joint damping, with ``k = log_scale_limit`` (3.0).
    """

    def __init__(self, model_path, frame_skip, log_scale_limit=3.0,
                 rand_params=RAND_PARAMS, n_tasks=2, randomize_tasks=True):
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        # loads self.model / self.data via the repo's modern MujocoEnv base
        super().__init__(model_path, frame_skip=frame_skip)
        # snapshot the nominal parameters AFTER the model is loaded
        self.init_params = {
            'body_mass':     self.model.body_mass.copy(),      # (nbody,)
            'body_inertia':  self.model.body_inertia.copy(),   # (nbody, 3)
            'dof_damping':   self.model.dof_damping.copy(),    # (nv,)
            'geom_friction': self.model.geom_friction.copy(),  # (ngeom, 3)
        }
        self.cur_params = None
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    # --- task distribution (same scheme as rand_param_envs.base) -------------
    def sample_tasks(self, n_tasks):
        tasks = []
        for _ in range(n_tasks):
            p = {}
            if 'body_mass' in self.rand_params:
                m = np.array(1.5) ** np.random.uniform(
                    -self.log_scale_limit, self.log_scale_limit,
                    size=self.init_params['body_mass'].shape)
                p['body_mass'] = self.init_params['body_mass'] * m
            if 'body_inertia' in self.rand_params:
                m = np.array(1.5) ** np.random.uniform(
                    -self.log_scale_limit, self.log_scale_limit,
                    size=self.init_params['body_inertia'].shape)
                p['body_inertia'] = self.init_params['body_inertia'] * m
            if 'dof_damping' in self.rand_params:
                m = np.array(1.3) ** np.random.uniform(
                    -self.log_scale_limit, self.log_scale_limit,
                    size=self.init_params['dof_damping'].shape)
                p['dof_damping'] = self.init_params['dof_damping'] * m
            if 'geom_friction' in self.rand_params:
                m = np.array(1.5) ** np.random.uniform(
                    -self.log_scale_limit, self.log_scale_limit,
                    size=self.init_params['geom_friction'].shape)
                p['geom_friction'] = self.init_params['geom_friction'] * m
            tasks.append(p)
        return tasks

    def set_task(self, task):
        for name, val in task.items():
            getattr(self.model, name)[:] = val          # in-place write of MjModel array
        # refresh mass-derived constants (subtree mass, etc.) and the dynamics
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.cur_params = task

    def get_task(self):
        return self.cur_params

    # --- PEARL task interface ------------------------------------------------
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
