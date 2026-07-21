# Licensing and attribution

This repository is a fork of [NVlabs/cule](https://github.com/NVlabs/cule).
Everything in it is distributed under the 3-clause BSD license.

## Copyright

| Code | Copyright | Terms |
|---|---|---|
| Original CuLE (`cule/`, `torchcule/`, `examples/`, build system) | Copyright (c) 2017-2019, NVIDIA CORPORATION | 3-clause BSD — see [LICENSE.TXT](LICENSE.TXT) |
| This fork's additions and modifications | Copyright (c) 2026, Mehrdad Moghimi | 3-clause BSD, the same terms as above |

The fork's additions include the CleanRL-style trainers in [cleanrl/](cleanrl/)
and [cleanrl_utils/](cleanrl_utils/), the benchmark suite in
[benchmarks/](benchmarks/), the test suite in [tests/](tests/), and the
modernization and GPU-kernel changes to the original sources described in
[CHANGES.md](CHANGES.md). NVIDIA's copyright notice in
[LICENSE.TXT](LICENSE.TXT) is retained unmodified, as the BSD license requires.

## Bundled third-party code

The trainers in [cleanrl/](cleanrl/) and the helpers in
[cleanrl_utils/](cleanrl_utils/) contain code adapted from three MIT-licensed
projects. Their full license texts, a per-file provenance table, and the list
of official implementations that the ported algorithms follow are in
[cleanrl/LICENSE.md](cleanrl/LICENSE.md).

| Project | Used for |
|---|---|
| [CleanRL](https://github.com/vwxyzjn/cleanrl) (MIT) | PPO, recurrent PPO, DQN, C51, Rainbow, PQN, SAC, and QDagger trainers; evaluation helpers |
| [LeanRL](https://github.com/meta-pytorch/LeanRL) (MIT) | `torch.compile` / CUDA-graph structure of the `*_torchcompile.py` variants |
| [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT) | Atari wrappers and the replay-buffer API in `cleanrl_utils/` |

Every trainer states its own provenance in its file header.

Algorithms that this fork implemented itself (QR-DQN, IQN, FQF, DER, DrQ(ε),
SPR, M-IQN, BBF) are independent PyTorch reimplementations written against the
authors' official releases. No code was copied from those projects; they are
credited, with their licenses, in [cleanrl/LICENSE.md](cleanrl/LICENSE.md).

## Submodules

Neither submodule is vendored into this repository; both are fetched at their
own licenses.

- `third_party/agency` — [agency](https://github.com/agency-library/agency),
  Copyright (c) 2014 NVIDIA CORPORATION, 3-clause BSD. The submodule URL points
  at a personal fork whose only change on top of upstream is the GCC >= 9 build
  fix; see the installation notes in [README.md](README.md#installation).
- `third_party/pybind11` — [pybind11](https://github.com/pybind/pybind11),
  BSD-style license.

## Atari ROMs

No Atari ROMs are distributed with this repository. They are supplied at
runtime by [ale-py](https://github.com/Farama-Foundation/Arcade-Learning-Environment),
subject to its own terms.
