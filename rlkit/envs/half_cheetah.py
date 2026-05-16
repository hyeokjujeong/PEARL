import numpy as np

from .mujoco_env import MujocoEnv


class HalfCheetahEnv(MujocoEnv):
    """Half-cheetah base env, ported to the modern ``mujoco`` bindings.

    The half_cheetah.xml model is loaded from gymnasium's bundled assets.
    """

    def __init__(self):
        super().__init__('half_cheetah.xml', frame_skip=5)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass
