"""Measure raw CuLE environment stepping throughput (no learner).

Reports env-steps/s for a random-action stepping loop in training mode
(frameskip 4, fire-reset), optionally with a per-kernel torch profiler
breakdown. Used for the numbers in
benchmark_results/gpu_kernel_optimization.md.

Example:
    python benchmarks/benchmark_step_throughput.py --envs 1024
    CULE_RENDER_LANES=8 python benchmarks/benchmark_step_throughput.py --envs 256
"""
import argparse
import time

import torch
from torchcule.atari import Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout')
    parser.add_argument('--envs', type=int, default=256)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--frameskip', type=int, default=4)
    parser.add_argument('--profile', action='store_true',
                        help='print a per-kernel CUDA time breakdown')
    args = parser.parse_args()

    env = Env(args.game, args.envs, color_mode='gray', device='cuda',
              rescale=True, episodic_life=True)
    env.train(args.frameskip)
    env.reset(initial_steps=50)

    for _ in range(args.warmup_steps):
        env.step(env.sample_random_actions())

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(env.sample_random_actions())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    sps = args.envs * args.steps / elapsed
    print(f'{args.envs} envs: {args.steps / elapsed:.1f} batched-steps/s, '
          f'{sps:,.0f} env-steps/s, {sps * args.frameskip:,.0f} emulated FPS')
    print(f'per-step wall time: {elapsed / args.steps * 1e3:.3f} ms')

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(50):
                env.step(env.sample_random_actions())
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='self_cuda_time_total',
                                        row_limit=15))


if __name__ == '__main__':
    main()
