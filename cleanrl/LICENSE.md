# Third-party licenses

The training scripts in this directory contain code adapted from
[CleanRL](https://github.com/vwxyzjn/cleanrl) and
[LeanRL](https://github.com/meta-pytorch/LeanRL). Their MIT licenses are
reproduced below, along with
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3), whose Atari
wrappers and replay-buffer API are used by `cleanrl_utils/`.

Each script names its own provenance in its header. In short:

| Script | Adapted from |
|---|---|
| `dqn_atari`, `c51_atari`, `ppo_atari`, `ppo_atari_lstm`, `pqn_atari_envpool`, `rainbow_atari`, `sac_atari`, `qdagger_dqn_atari_impalacnn` | CleanRL (MIT) |
| `dqn_torchcompile`, `ppo_atari_envpool_torchcompile` | LeanRL (MIT), on top of CleanRL |
| other `*_torchcompile` variants | this fork, following LeanRL's compile/CUDA-graph structure |
| `qrdqn_atari`, `iqn_atari`, `fqf_atari`, `der_atari`, `drq_atari`, `spr_atari`, `miqn_atari`, `bbf_atari` | this fork, ported from the reference implementations below |

## Reference implementations

The algorithm ports listed above are independent PyTorch reimplementations
written against the following official sources. No code was copied from them;
they are credited because the algorithms, hyperparameters, and design details
come from them.

| Algorithm | Official source | License |
|---|---|---|
| QR-DQN | [google/dopamine](https://github.com/google/dopamine) `dopamine/jax/agents/quantile` | Apache-2.0 |
| IQN, M-IQN base | [google/dopamine](https://github.com/google/dopamine) `dopamine/jax/agents/implicit_quantile` | Apache-2.0 |
| FQF | [microsoft/FQF](https://github.com/microsoft/FQF) | MIT |
| DER, DrQ(ε) | [google/dopamine](https://github.com/google/dopamine) `dopamine/labs/atari_100k` | Apache-2.0 |
| SPR | [mila-iqia/spr](https://github.com/mila-iqia/spr) | MIT |
| M-IQN | [google-research/google-research](https://github.com/google-research/google-research) `munchausen_rl` | Apache-2.0 |
| BBF | [google-research/google-research](https://github.com/google-research/google-research) `bigger_better_faster` | Apache-2.0 |

The Impala-CNN encoder used by the QDagger student comes, via CleanRL, from the
[NeurIPS 2020 Procgen starter kit](https://github.com/AIcrowd/neurips2020-procgen-starter-kit)
(Apache-2.0).

## CleanRL

MIT License

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## stable-baselines3

The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## LeanRL

MIT License

Copyright (c) 2024 LeanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
