import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason='CUDA device not available')

def device_params():
    """Parametrize a test over the CPU backend and (when present) the GPU backend."""
    return ['cpu', pytest.param('cuda', marks=requires_cuda)]
