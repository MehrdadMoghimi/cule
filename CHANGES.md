# Changes compared to upstream NVlabs/cule

Upstream CuLE (July 2019) targets CUDA 10.0, PyTorch 1.x, Python 3.6, `gym`
and `atari_py`. This fork runs on modern stacks — verified with CUDA 12.9,
PyTorch 2.13 (cu130), Python 3.12, an RTX 4090 (sm_89), and Gymnasium/ale-py.
Emulation semantics are unchanged (rollouts are bit-identical to upstream);
the frame-rendering kernel was parallelized for a 34-48% throughput gain (see
Performance below).

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

   Construction no longer crashes with those fixes, and a per-game
   behavioral audit against the reference ale-py emulator (matched random
   play comparing reward flow, reward values, episode termination, frame
   liveness and color palettes, plus a deterministic input-effect test:
   identical-seed GPU rollouts differing only in the action must diverge)
   uncovered the real reasons most of these games misbehaved — all fixed
   in this fork (see next item). Upstream's own benchmark script silently
   excluded qbert, defender and elevator_action; the paper's experiments
   used games from the working set.
8. **61 of the 63 decoded games are now verified working; upstream had
   ~half of them silently running as the wrong game.** Four distinct bugs:
   - **ROM identity (the big one): 76 of the 108 ROM dumps bundled with
     ale-py have md5s that were missing from `rom_game_map`**
     (`cule/atari/games/detail/types.hpp`), and an unmapped md5 silently
     resolves to `GAME_TYPE` 0 — bowling. Those ROMs emulated fine but ran
     with *bowling's* reward decoder, terminal detection, controller
     attributes, and minimal action set (e.g. amidar's agent had no
     LEFT/RIGHT and could never walk; ice_hockey "rewards" decoded
     bowling's score address, which holds ice hockey's game clock). Fixed
     by adding the 31 ale-py md5s for the games CuLE decodes.
   - **`FLAG_ALE_STARTED` double-use**: `environment::act()` overwrites the
     flag every frame with its boot-phase condition, but riverraid's and
     qbert's `setTerminal` used it as one-frame memory (ALE's
     `m_lives_byte`/`m_last_lives`), so riverraid reported done on every
     step. Game-private memory now lives in a new `FLAG_ALE_GAME_STATE`
     bit (riverraid, qbert, stargunner, breakout migrated).
   - **Boot deadlock**: while the console RESET switch is held during the
     boot sequence, some ROMs (qbert, montezuma_revenge) sit in a wait
     loop without strobing VSYNC, so the frame never completed, the boot
     never advanced past frame 60, and the cached reset states were
     pre-boot garbage (qbert's RAM stayed all-zero forever). Fixed with a
     scanline-overflow fallback in `environment::emulate` (mirroring the
     one the TIA already applied on register pokes) plus counting completed
     frames — not `act()` calls — in the reset-cache boot loop.
   - **Reset-boundary reward artifacts**: cached reset states carried
     `score = 0` while their RAM already decodes the game's starting score
     (pitfall boots with 2000 points), crediting the difference to the
     agent on the first step of every episode. The cache builder now syncs
     each slot's score after boot, and `get_data` only accumulates rewards
     once the boot phases are finished.

   Remaining broken (README lists them): double_dunk (agent input ignored
   on both backends — the demo plays itself; ale-py responds normally) and
   elevator_action (no rewards on the CPU backend, implausible ones on
   GPU). `tests/test_env.py` carries a strict-xfail guard, regression tests
   for the formerly mis-identified games, and an input-effect test.
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

## Performance

Measured on an RTX 4090, Breakout, grayscale 84x84, frameskip 4, random
actions (environment stepping only, no learner):

| Envs | Upstream launch config (SPS) | This fork (SPS) | Gain |
|---:|---:|---:|---:|
| 256 | 15,795 | 21,177 | +34% |
| 1,024 | 54,584 | 74,937 | +37% |
| 4,096 | 76,827 | 113,654 | +48% |

- **Warp-cooperative frame rendering.** Upstream `process_kernel` launched one
  single-thread block per environment: one thread serially replayed the TIA
  update stream and drew all ~48K pixels, and an SM can host at most 24 such
  blocks. The kernel now runs one block per environment with
  `CULE_RENDER_LANES` threads (default 32): every lane executes the identical
  warp-uniform replay while pixel spans are strided across lanes. Rendering is
  bit-identical for any lane count (7.3x faster at the default).
- **Dead work removed in `Env.step`**: the observation tensors are fully
  overwritten by frame generation every step, so they are no longer zeroed.
- **Emulation kernel launch config re-validated.** The step/reset kernels keep
  upstream's one-thread-per-block launch: packing multiple environments per
  warp was measured up to ~7x *slower* (the divergent 6502/TIA interpreter
  serializes across lanes, and larger `__launch_bounds__` force register
  spills). The block size is now a runtime parameter
  (`CULE_STEP_BLOCK_SIZE`, default 1) for experimentation on other GPUs, and
  per-environment RAM uses the same linear layout on every path (upstream's
  GPU layout was block-size-interleaved, which `get_data_kernel` and the
  Python `env.ram` view silently assumed to be linear — true only at block
  size 1).
- Kernel grid-size computations use integer ceiling division; the previous
  `float` ceil could drop trailing blocks above ~16.7M elements (e.g.
  `generate_frames` with ~100K+ environments).

All optimizations are verified bit-exact against pre-change golden rollouts
(observations, rewards, dones, lives, RAM) on both backends, across 12 games
and lane counts {1, 7, 32, 256}, plus the full pytest suite including
training micro-runs.

## New

- `tests/` pytest suite: ROM resolution and metadata, CPU/GPU env behavior
  (shapes, rewards, episodic life, determinism, 9 games across all supported
  cartridge mappers), the Gymnasium wrapper stack, and end-to-end training
  micro-runs for a2c, vtrace, ppo and dqn (`pytest -m slow`).
- `cleanrl/` single-file trainers with a CuLE backend (adapted from CleanRL
  and LeanRL): PPO, recurrent PPO, DQN, C51, Rainbow, PQN, and discrete SAC,
  plus `torch.compile`/CUDA-graph variants of PPO, DQN, C51, Rainbow, and
  PQN. See `cleanrl/README_CULE.md`.
- `benchmarks/` scripts for throughput sweeps (raw stepping, CuLE vs EnvPool,
  cross-implementation) and fixed-budget learning comparisons, with matching
  `analyze_*` aggregators. Results are written to the git-ignored
  `benchmark_results/`; headline numbers are in the project README.
- README sections: building on modern stacks, per-algorithm example commands,
  the list of 63 games with reward decoding (61 verified working), testing
  instructions.
- Updated `envs/Dockerfile` (CUDA 12.9 / Ubuntu 24.04) and
  `envs/environment.yml`.

## Known limitations

- CPU and GPU backends do not produce bit-identical trajectories (upstream
  behavior: reset entropy is consumed differently per backend). GPU rollouts
  are reproducible given fixed seeds; the CPU backend is not (thread-pool
  scheduling).
- `pitfall2` is the only bundled ale-py ROM whose cartridge mapper is
  unsupported.
- Games outside the 63 with reward decoding load and emulate but always
  return reward 0.
- Resetting with explicit seeds (`env.reset(seeds=...)`) can land some games
  in odd emulator states (observed with montezuma_revenge: the agent spawns
  displaced and stops responding); the default reset path is unaffected.
