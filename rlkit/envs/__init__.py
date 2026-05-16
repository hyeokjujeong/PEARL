import os
import importlib


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


# automatically import any envs in the envs/ directory
# a module that fails to import (e.g. a MuJoCo env when MuJoCo is unavailable)
# is skipped with a warning so unrelated envs (e.g. point-robot) still work
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        try:
            importlib.import_module('rlkit.envs.' + module)
        except Exception as e:
            print('WARNING: could not import rlkit.envs.{} ({}: {})'.format(
                module, type(e).__name__, e))
