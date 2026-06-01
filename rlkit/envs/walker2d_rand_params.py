import numpy as np

from .rand_param_base import RandomParamEnv


class Walker2DRandParamsEnv(RandomParamEnv):
    """Walker2d whose tasks are random model-parameter draws (dynamics-varying).

    Modern port of ``rand_param_envs.walker2d_rand_params``. The walker2d.xml
    model is loaded from gymnasium's bundled assets (resolved by the repo's
    ``MujocoEnv`` base).
    """

    def __init__(self, n_tasks=2, randomize_tasks=True, log_scale_limit=3.0):
        super().__init__('walker2d.xml', frame_skip=4,
                         log_scale_limit=log_scale_limit, n_tasks=n_tasks)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10),
        ]).astype(np.float32)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        reward = (posafter - posbefore) / self.dt
        reward += 1.0                                   # alive bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        return self._get_obs(), float(reward), bool(done), {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(-.005, .005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-.005, .005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
