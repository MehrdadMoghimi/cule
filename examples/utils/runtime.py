import ctypes
import os

import torch

def _find_in_path(name, path):
    for directory in path.split(os.pathsep):
        binpath = os.path.join(directory, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA toolkit from CUDA_HOME/CUDAHOME or nvcc on PATH."""
    if any(directory in os.environ for directory in ['CUDAHOME', 'CUDA_HOME']):
        home = os.environ['CUDAHOME'] if 'CUDAHOME' in os.environ else os.environ['CUDA_HOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        nvcc = _find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either '
                                   'add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
    cudaconfig = {
            'home': home,
            'nvcc': nvcc,
            'include': os.path.join(home, 'include'),
            'lib64': os.path.join(home, 'lib64')
            }
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig

class Runtime(object):
    """Compatibility shim kept for setup.py; CUDA queries now go through torch."""

    def _locate(self):
        return locate_cuda()

def get_device_props():
    return [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]

def cuda_device_str(device=0):
    props = torch.cuda.get_device_properties(device)
    free_mem, total_mem = torch.cuda.mem_get_info(device)

    return  '{} (Ordinal {})\n'\
            '{} SMs enabled. Compute Capability sm_{}{}\n'\
            'FreeMem: {:6,d}MB   TotalMem: {:6,d}MB   {:2d}-bit pointers.\n'\
            .format(props.name, device,
                    props.multi_processor_count, props.major, props.minor,
                    int(free_mem / (1 << 20)), int(total_mem / (1 << 20)),
                    8 * ctypes.sizeof(ctypes.POINTER(ctypes.c_int32)))
