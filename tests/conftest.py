import functools
import importlib.util
import os
import sys

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason='CUDA device not available')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLEANRL_DIR = os.path.join(REPO_ROOT, 'cleanrl')

def device_params():
    """Parametrize a test over the CPU backend and (when present) the GPU backend."""
    return ['cpu', pytest.param('cuda', marks=requires_cuda)]


@functools.lru_cache(maxsize=None)
def load_trainer(name):
    """Import a `cleanrl/<name>.py` trainer as a module without running it.

    The trainers guard their training loop behind `if __name__ == "__main__"`,
    so importing only defines the networks, losses and helpers that the
    equivalence tests need.
    """
    for path in (CLEANRL_DIR, REPO_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        f'_trainer_{name}', os.path.join(CLEANRL_DIR, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DiscreteEnvStub:
    """Minimal stand-in for the vector env the trainers read spaces off."""

    def __init__(self, n_actions=6, obs_shape=(4, 84, 84)):
        self.single_action_space = type('Discrete', (), {'n': n_actions})()
        self.single_observation_space = type('Box', (), {'shape': obs_shape})()
