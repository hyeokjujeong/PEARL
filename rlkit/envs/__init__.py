import os
import importlib
import warnings


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
# Modules that fail to import — most commonly a MuJoCo-based env when its
# optional backend is unavailable — are warn-and-skipped so that the
# pure-Python envs in this directory remain usable without the full PEARL
# install. The failure can surface as several exception types, so catch
# broadly.
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        try:
            importlib.import_module('rlkit.envs.' + module)
        except Exception as e:
            warnings.warn(
                "rlkit.envs: skipped {} ({}: {})".format(
                    module, type(e).__name__, e
                )
            )
