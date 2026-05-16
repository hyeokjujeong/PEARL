import os
import importlib
import warnings

import gym.error


ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn


# automatically import any envs in the envs/ directory.
# Modules that fail to import due to a missing optional dependency (most
# commonly mujoco_py for the MuJoCo-based envs) are warn-and-skipped so that
# pure-Python envs in this directory remain usable without the full PEARL
# install. gym raises gym.error.DependencyNotInstalled (an Exception, not an
# ImportError) when its MuJoCo backend can't be loaded, so catch both.
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        try:
            importlib.import_module('rlkit.envs.' + module)
        except (ImportError, gym.error.DependencyNotInstalled) as e:
            warnings.warn(
                "rlkit.envs: skipped {} due to missing dependency: {}".format(
                    module, e
                )
            )
