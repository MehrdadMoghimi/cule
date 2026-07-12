"""End-to-end micro-runs of each training example (marked slow).

Each test launches the real training script as a subprocess with a tiny
frame budget and asserts it exits cleanly. Run with: pytest -m slow
"""
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.slow

EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason='CUDA device not available')

def run_example(algo, script, args, timeout=420):
    cmd = [sys.executable, script] + args
    proc = subprocess.run(cmd, cwd=os.path.join(EXAMPLES_DIR, algo),
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, \
        'exit {}\n--- stdout ---\n{}\n--- stderr ---\n{}'.format(
            proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:])
    return proc

COMMON = ['--evaluation-interval', '1000000', '--evaluation-episodes', '2',
          '--seed', '1']

@requires_cuda
@pytest.mark.parametrize('game', ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4'])
def test_a2c_gpu(game):
    run_example('a2c', 'a2c_main.py',
                ['--env-name', game, '--use-cuda-env', '--num-ales', '32',
                 '--num-steps', '5', '--t-max', '5000',
                 '--ale-start-steps', '100'] + COMMON)

@requires_cuda
def test_vtrace_gpu():
    run_example('vtrace', 'vtrace_main.py',
                ['--env-name', 'PongNoFrameskip-v4', '--normalize',
                 '--use-cuda-env', '--num-ales', '64', '--num-steps', '20',
                 '--num-steps-per-update', '1', '--num-minibatches', '4',
                 '--t-max', '5000', '--ale-start-steps', '100'] + COMMON)

@requires_cuda
def test_vtrace_gpu_double_test():
    # --double-test evaluates on both CuLE CPU and Gymnasium test envs
    run_example('vtrace', 'vtrace_main.py',
                ['--env-name', 'PongNoFrameskip-v4', '--normalize',
                 '--use-cuda-env', '--double-test', '--num-ales', '64',
                 '--num-steps', '20', '--num-steps-per-update', '1',
                 '--num-minibatches', '4', '--t-max', '2000',
                 '--ale-start-steps', '100'] + COMMON)

@requires_cuda
def test_ppo_gpu():
    run_example('ppo', 'ppo_main.py',
                ['--env-name', 'BreakoutNoFrameskip-v4', '--use-cuda-env',
                 '--num-ales', '32', '--num-steps', '20', '--t-max', '5000',
                 '--ale-start-steps', '100'] + COMMON)

@requires_cuda
def test_dqn_gpu():
    run_example('dqn', 'dqn_main.py',
                ['--env-name', 'PongNoFrameskip-v4', '--use-cuda-env',
                 '--num-ales', '16', '--t-max', '2000', '--learn-start', '500',
                 '--memory-capacity', '2000', '--batch-size', '32',
                 '--evaluation-interval', '1000000', '--seed', '1'])

def test_a2c_cule_cpu_backend():
    # CuLE CPU emulation backend (no --use-cuda-env); training device may
    # still be the GPU if one is present
    run_example('a2c', 'a2c_main.py',
                ['--env-name', 'PongNoFrameskip-v4', '--num-ales', '8',
                 '--num-steps', '5', '--t-max', '400',
                 '--ale-start-steps', '40'] + COMMON)
