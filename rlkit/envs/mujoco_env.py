import os

import numpy as np
import gymnasium.envs.mujoco as _gym_mujoco
from gymnasium.envs.mujoco import MujocoEnv as GymMujocoEnv
from gymnasium.spaces import Box

from rlkit.core.serializable import Serializable

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')
# XMLs not shipped in the repo's assets/ are loaded from gymnasium's bundled set
_GYM_ASSET_DIR = os.path.join(os.path.dirname(_gym_mujoco.__file__), 'assets')


class _SimData(object):
    """mujoco-py exposed physics state as ``env.sim.data``; the modern mujoco
    bindings expose it directly as ``env.data``. This shim keeps the old
    attribute path working so env code written against mujoco-py is unchanged.
    """
    def __init__(self, env):
        object.__setattr__(self, '_env', env)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_env').data, name)


class _Sim(object):
    """Compat shim for the old ``env.sim`` object (mujoco-py)."""
    def __init__(self, env):
        self._env = env
        self.data = _SimData(env)

    @property
    def model(self):
        return self._env.model


class MujocoEnv(GymMujocoEnv, Serializable):
    """Wrapper around gymnasium's MujocoEnv (modern ``mujoco`` bindings).

    Keeps the repo's legacy env contract:
      - ``reset()`` returns a bare observation (not the gymnasium ``(obs, info)``)
      - the observation space is derived from the subclass's ``_get_obs()``
        (old gym auto-derived it; gymnasium does not)
      - ``env.sim.data`` / ``env.sim.model`` keep working via a compat shim
    Subclasses still return a 4-tuple from ``step()``.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
            self,
            model_path,
            frame_skip=1,
            model_path_is_local=True,
            automatically_set_obs_and_action_space=False,
    ):
        if model_path_is_local:
            model_path = get_asset_xml(model_path)
        # placeholder space; replaced below once the model is loaded
        placeholder = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        GymMujocoEnv.__init__(
            self, model_path, frame_skip, observation_space=placeholder,
        )
        obs = np.asarray(self._get_obs())
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64,
        )

    @property
    def sim(self):
        return _Sim(self)

    def reset(self, *args, **kwargs):
        # legacy contract: callers expect a bare observation
        result = GymMujocoEnv.reset(self, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    def init_serialization(self, locals):
        Serializable.quick_init(self, locals)

    def log_diagnostics(self, *args, **kwargs):
        pass


def get_asset_xml(xml_name):
    local = os.path.join(ENV_ASSET_DIR, xml_name)
    if os.path.exists(local):
        return local
    return os.path.join(_GYM_ASSET_DIR, xml_name)
