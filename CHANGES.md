# Changes compared to upstream NVlabs/cule

Upstream CuLE (July 2019) targets CUDA 10.0, PyTorch 1.x, Python 3.6, `gym`
and `atari_py`. This fork runs on modern stacks — verified with CUDA 12.9,
PyTorch 2.13 (cu130), Python 3.12, an RTX 4090 (sm_89), and Gymnasium/ale-py —
without modifying any CUDA kernel.

## Dependency replacements

| Upstream | This fork |
|---|---|
| `atari_py` (unmaintained) | `ale-py` ≥ 0.10 — Atari ROMs are bundled, no separate ROM install |
| `gym` | `gymnasium` |
| NVIDIA `apex` (DDP, AMP) | `torch.nn.parallel.DistributedDataParallel` |
| `tensorboardX` | `torch.utils.tensorboard` |
| Python 2/3 dual support | Python ≥ 3.9 |

## Build system (`setup.py`)

- GPU architectures are detected with `torch.cuda.get_device_capability()`.
  (Upstream derived them from `torch.cuda.get_arch_list()` with string slicing
  that mapped e.g. sm_89 to sm_80.) PTX for the newest arch is embedded for
  forward compatibility.
- `libcudart` is linked explicitly with an rpath to the toolkit. (Upstream
  relied on old PyTorch re-exporting CUDA runtime symbols, which modern torch
  no longer does — the extension otherwise fails to import.)
- The unused Cython build dependency was dropped (plain setuptools
  `build_ext`).
- All `cule/**/*.hpp` headers are declared as `Extension(depends=...)` so
  header edits actually trigger recompiles (distutils does not track header
  dependencies).
- Early guards with actionable messages: the agency patch (below) is
  auto-applied, and a pybind11 submodule that is too old for the running
  Python is reported before compilation starts.
- Added `pyproject.toml` and `requirements.txt`. Install with
  `pip install --no-build-isolation -e .` (setup.py imports torch at build
  time).

## Third-party submodules

- **pybind11** is checked out at v2.13.6. The upstream-pinned 2018 commit
  cannot compile against Python ≥ 3.11.
- **agency** (archived upstream) needs two header fixes for GCC ≥ 9, shipped
  as [third_party/patches/agency-modern-toolchain.patch](third_party/patches/agency-modern-toolchain.patch)
  and applied automatically by setup.py:
  - `detail/operator_traits.hpp`: the `has_operator_*` detectors evaluate
    `decltype(a @ b)` in a class template's *base-clause*, where modern GCC's
    two-phase operator lookup misses the namespace-scope catch-all operators;
    non-matching operand types became hard errors. The expressions now live in
    helper class bodies where definition-context lookup works.
  - `detail/tuple/arithmetic_tuple_facade.hpp`: operator constraints used
    `std::tuple_size<T>` directly in `enable_if`, a hard error for non-tuple
    `T` on modern libstdc++. Replaced with a SFINAE-friendly wrapper.

## Fixed upstream bugs

1. **14 games crashed at environment construction** (assault, bank_heist,
   battle_zone, centipede, double_dunk, kung_fu_master, ms_pacman, pitfall,
   skiing, space_invaders, tennis, tutankham, venture, yars_revenge) with a
   `functors.hpp:249` assertion on both backends.
   Cause: these games' `setTerminal` reads lives/score RAM that the game's
   own boot code has not initialized yet, so states cached during the reset
   noop-cache build are flagged terminal while the cached-frame pointers are
   legitimately still null. Two host-side fixes (CUDA kernels untouched):
   - the CPU `preprocess_functor` tolerates the null cache pointers during
     the build instead of asserting;
   - the reset cache builder keeps stepping (bounded) while the game still
     reports terminal before filling the noop cache — games that boot clean
     take zero extra frames.

   **Construction no longer crashes, but most of these games remain broken
   at the emulation level** (inherited from upstream): CuLE never ported
   ALE's per-ROM game-start machinery (`RomSettings::reset()` plus ALE-side
   lives tracking), so these ROMs sit in attract/demo mode — reset loops,
   episodes that never terminate (qbert, elevator_action), or garbage score
   decoding (defender, space_invaders). Automated health checks (episode
   termination, episode length, reward sanity, on both backends) pass for
   44 games and fail for 19; both lists are in the README, and
   `tests/test_env.py` carries a strict-xfail guard so a future fix is
   noticed. Upstream's own benchmark script silently excluded qbert,
   defender and elevator_action; the paper's experiments used games from
   the working set.
2. **Intermittent `CUDA error: an illegal memory access` during DQN/PPO
   training.** `dqn/train.py` stepped the environment inside
   `torch.cuda.stream(env_stream)` (and trained under other side streams)
   while mutating tensors allocated on the default stream, without
   `record_stream()`. On modern PyTorch the caching allocator reuses that
   memory across streams and training crashes minutes in. All example code
   now runs on the default stream (CuLE synchronizes the device on every
   dispatch anyway, so the streams bought no real overlap).
3. Rolling-buffer shifts `x[:-1] = x[1:]` in vtrace (overlapping copy —
   rejected by modern torch) now `.clone()` the source.
4. `torchcule.atari.Env` crashed for `device='cuda'` without an explicit
   index (`torch.device('cuda').index` is `None`).
5. ROM lookup only worked for single-word game names (upstream split the env
   name on the substring `'No'` — `SpaceInvadersNoFrameskip-v4` was
   unresolvable). Proper CamelCase→snake_case conversion plus support for
   `ALE/Name-v5` ids and plain rom ids.
6. `examples/utils/runtime.py` mirrored `cudaDeviceProp` via ctypes; CUDA 12
   renamed `cudaGetDeviceProperties` and changed the struct layout, breaking
   it. Device queries now go through `torch.cuda`.
7. Smaller fixes: undefined `writer` in the `--plot` paths of vtrace/ppo,
   `scheduler.get_lr()` → `get_last_lr()`, missing `math` import and a
   `torch.backends.cudnn` typo in `initializers.py`, `np.bool` (removed in
   numpy 1.24), `SafeConfigParser` (removed in Python 3.12), apex-style
   `DDP(delay_allreduce=True)` → `DDP(device_ids=[gpu])`, unused
   `torch.cuda.amp.GradScaler` removed from a2c.

## Behavior changes

- **vtrace evaluation**: upstream hardcoded a *double* evaluation at every
  interval — once on the CuLE CPU emulator (`[CuLE CPU]`) and once on the
  OpenAI/Gymnasium emulator (`[OpAI CPU]`) — which doubles evaluation cost
  and wrote extra `test_cule*.csv` files. This is now opt-in via
  `--double-test`. Note the printed `training time` counts only training
  updates; with a small `--evaluation-interval`, wall-clock time is dominated
  by evaluation episodes.
- The reset noop-cache now boots from zeroed RAM (matching the internal GPU
  cache builder) instead of random bytes; per-env RAM stays randomized.
- `examples/utils/openai/` was rewritten on Gymnasium: legacy
  `*NoFrameskip-v4` ids are translated to `ALE/*-v5` with `frameskip=1` and
  sticky actions disabled, and a thin adapter preserves the classic 4-tuple
  Gym API so the training loops and `subproc_vec_env.py` are unchanged.

## New

- `tests/` pytest suite: ROM resolution and metadata, CPU/GPU env behavior
  (shapes, rewards, episodic life, determinism, 9 games across all supported
  cartridge mappers), the Gymnasium wrapper stack, and end-to-end training
  micro-runs for a2c, vtrace, ppo and dqn (`pytest -m slow`).
- README sections: building on modern stacks, per-algorithm example commands,
  the list of 62 games with reward decoding, testing instructions.
- Updated `envs/Dockerfile` (CUDA 12.9 / Ubuntu 24.04) and
  `envs/environment.yml`.

## Known limitations

- CPU and GPU backends do not produce bit-identical trajectories (upstream
  behavior: reset entropy is consumed differently per backend). GPU rollouts
  are reproducible given fixed seeds; the CPU backend is not (thread-pool
  scheduling).
- `pitfall2` is the only bundled ale-py ROM whose cartridge mapper is
  unsupported.
- Games outside the 62 with reward decoding load and emulate but always
  return reward 0.
