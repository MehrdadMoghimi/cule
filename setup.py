import argparse
import os
import sys
import torch

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext

def locate_cuda():
    """Locate the CUDA toolkit from CUDA_HOME/CUDAHOME or nvcc on PATH."""
    home = None
    for var in ('CUDAHOME', 'CUDA_HOME'):
        if var in os.environ:
            home = os.environ[var]
            break

    if home is None:
        for directory in os.environ.get('PATH', '').split(os.pathsep):
            binpath = os.path.join(directory, 'nvcc')
            if os.path.exists(binpath):
                home = os.path.dirname(os.path.dirname(os.path.abspath(binpath)))
                break

    if home is None:
        raise EnvironmentError('The nvcc binary could not be located in your '
                               '$PATH. Either add it to your path, or set '
                               '$CUDA_HOME to your CUDA toolkit directory.')

    cudaconfig = {
            'home': home,
            'nvcc': os.path.join(home, 'bin', 'nvcc'),
            'include': os.path.join(home, 'include'),
            'lib64': os.path.join(home, 'lib64'),
            }
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig

def cuda_arch_flags():
    """Generate gencode flags for the local GPUs (fall back to common arches)."""
    if torch.cuda.device_count() > 0:
        codes = sorted({'%d%d' % torch.cuda.get_device_capability(i)
                        for i in range(torch.cuda.device_count())})
    else:
        codes = ['70', '80', '86', '89', '90']

    flags = ['-gencode=arch=compute_{0},code=sm_{0}'.format(code) for code in codes]
    # PTX for the newest arch so future GPUs can JIT the kernels
    flags += ['-gencode=arch=compute_{0},code=compute_{0}'.format(codes[-1])]
    return flags

base_dir = os.path.dirname(os.path.realpath(__file__))

def ensure_agency_patched():
    """The bundled agency library predates GCC >= 9 and needs local fixes
    (third_party/patches/agency-modern-toolchain.patch). Submodule updates
    discard them, so (re)apply the patch when its marker is missing."""
    agency_dir = os.path.join(base_dir, 'third_party', 'agency')
    marker_file = os.path.join(agency_dir, 'agency', 'detail', 'tuple', 'arithmetic_tuple_facade.hpp')
    patch_file = os.path.join(base_dir, 'third_party', 'patches', 'agency-modern-toolchain.patch')

    if not os.path.exists(marker_file):
        raise EnvironmentError('agency submodule is missing; run '
                               '"git submodule update --init --recursive" first.')

    with open(marker_file) as f:
        if 'facade_tuple_size' in f.read():
            return

    import subprocess
    print('Applying {} to third_party/agency'.format(os.path.basename(patch_file)))
    subprocess.check_call(['git', 'apply', os.path.relpath(patch_file, agency_dir)],
                          cwd=agency_dir)

ensure_agency_patched()

def ensure_modern_pybind11():
    """Python >= 3.11 needs pybind11 >= 2.10; the historical submodule pin is
    from 2018. Fail early with instructions instead of pages of C++ errors."""
    common_h = os.path.join(base_dir, 'third_party', 'pybind11', 'include',
                            'pybind11', 'detail', 'common.h')
    version = (0, 0)
    if os.path.exists(common_h):
        with open(common_h) as f:
            src = f.read()
        import re
        major = re.search(r'#define PYBIND11_VERSION_MAJOR (\d+)', src)
        minor = re.search(r'#define PYBIND11_VERSION_MINOR (\d+)', src)
        if major and minor:
            version = (int(major.group(1)), int(minor.group(1)))

    if version < (2, 10):
        raise EnvironmentError(
            'third_party/pybind11 is too old ({}) for modern Python. Run:\n'
            '  git -C third_party/pybind11 fetch --tags && '
            'git -C third_party/pybind11 checkout v2.13.6'.format(
                '.'.join(map(str, version)) if version != (0, 0) else 'missing'))

ensure_modern_pybind11()

CUDA = locate_cuda()

sources = [os.path.join('torchcule', f) for f in ['frontend.cpp', 'backend.cu']]

# distutils only compares object timestamps against the listed sources, so
# declare every header as a dependency to get rebuilds after header edits
import glob
header_deps = (glob.glob(os.path.join(base_dir, 'cule', '**', '*.hpp'), recursive=True) +
               glob.glob(os.path.join(base_dir, 'torchcule', '*.hpp')) +
               glob.glob(os.path.join(base_dir, 'cule', '**', '*.cpp'), recursive=True))

third_party_dir = os.path.join(base_dir, 'third_party')
include_dirs = [base_dir, os.path.join(third_party_dir, 'agency'), os.path.join(third_party_dir, 'pybind11', 'include'), CUDA['include']]
libraries = ['cudart', 'gomp', 'z']
library_dirs = [CUDA['lib64']]
cxx_flags = ['-std=c++14', '-O3', '-Wall', '-fPIC']
nvcc_flags = cuda_arch_flags() + ['-std=c++14', '-O3', '-Xptxas=-v', '-Xcompiler=-Wall,-Wextra,-fPIC']

parser = argparse.ArgumentParser('CuLE', add_help=False)
parser.add_argument('--fastbuild', action='store_true', default=False, help='Build CuLE supporting only 2K roms')
parser.add_argument('--compiler', type=str, default='gcc', help='Host compiler (default: gcc)')
args, remaining_argv = parser.parse_known_args()
sys.argv = ['setup.py'] + remaining_argv

if args.fastbuild:
    nvcc_flags += ['-DCULE_FAST_COMPILE']

nvcc_flags += ['-ccbin={}'.format(args.compiler)]

def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile

# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(name='torchcule',
      version='0.2.0',
      description='A GPU RL environment package for PyTorch',
      url='https://github.com/NVlabs/cule',
      author='Steven Dalton',
      author_email='sdalton1@gmail.com',
      python_requires='>=3.9',
      install_requires=['gymnasium>=1.0', 'ale-py>=0.10', 'numpy'],
      ext_modules=[
          Extension('torchcule_atari',
              sources=sources,
              depends=header_deps,
              include_dirs=include_dirs,
              libraries=libraries,
              library_dirs=library_dirs,
              runtime_library_dirs=library_dirs,
              extra_compile_args={'gcc': cxx_flags, 'nvcc': nvcc_flags}
              )
          ],
      # Exclude the build files.
      packages=find_packages(exclude=['build']),
      cmdclass={
          'build_ext': custom_build_ext,
          }
      )
