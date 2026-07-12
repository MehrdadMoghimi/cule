![ALT](/media/images/System.png "Deep RL System Overview")

# CuLE 0.1.0

_CuLE 0.1.0 - July 2019_

CuLE is a CUDA port of the Atari Learning Environment (ALE) and is
designed to accelerate the development and evaluation of deep
reinforcement algorithms using Atari games. Our CUDA Learning
Environment (CuLE) overcomes many limitations of existing CPU- based
Atari emulators and scales naturally to multi-GPU systems.  It leverages
the parallelization capability of GPUs to run thousands of Atari games
simultaneously; by rendering frames directly on the GPU, CuLE avoids the
bottleneck arising from the limited CPU-GPU communication bandwidth.

# Compatibility

This fork is updated for modern software stacks: it compiles with the
[CUDA 12.x Toolkit](https://developer.nvidia.com/cuda-toolkit) and runs against
current PyTorch releases. The legacy `gym`/`atari_py` dependencies were replaced
with [Gymnasium](https://gymnasium.farama.org) and
[ale-py](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
(Atari ROMs are bundled with ale-py — no separate ROM installation needed).
See [CHANGES.md](CHANGES.md) for the complete list of differences from
upstream NVlabs/cule, including several upstream bug fixes.

We have tested the following environment.

|**Operating System** | **Compiler** | **CUDA** | **Python / PyTorch** |
|-----------------|----------|------|------------------|
| Ubuntu 22.04 (WSL2) | GCC 11.4 | 12.9 | 3.12 / torch 2.13 (cu130) |

CuLE runs on Maxwell- through Ada/Hopper-architecture NVIDIA GPUs (tested on an
RTX 4090, sm_89). The build targets the compute capability of the GPUs detected
at compile time.

# Building CuLE

```
$ git clone --recursive https://github.com/NVlabs/cule
$ cd cule
$ git -C third_party/pybind11 fetch --tags && git -C third_party/pybind11 checkout v2.13.6
$ pip install torch gymnasium ale-py opencv-python-headless tqdm psutil pytz tensorboard cloudpickle
$ CUDA_HOME=/usr/local/cuda-12.9 pip install --no-build-isolation -e .
```

Notes:
- `CUDA_HOME` should point to a CUDA 12.x toolkit whose `nvcc` supports your GPU.
- `--no-build-isolation` is required because `setup.py` queries the local torch
  installation for the GPU architectures to compile for.
- The extension does not link against libtorch, so the toolkit used to build it
  does not need to match the CUDA version of the PyTorch wheels.
- The pinned `pybind11` submodule commit predates Python 3.11 — check out a
  modern release tag as shown above (setup.py errors out early otherwise).
- The unmaintained `agency` submodule needs small fixes for GCC >= 9;
  setup.py applies [third_party/patches/agency-modern-toolchain.patch](third_party/patches/agency-modern-toolchain.patch)
  automatically when it is missing.

# Project Structure

```
cule/
  cule/
  env/
  examples/
  media/
  third_party/
  torchcule/
```

Several example programs are also distributed with the CuLE library. They are
contained in the following directories.

```
examples/
  a2c/
  dqn/
  ppo/
  vtrace/
  utils/
  visualize/
```

# Running the examples

All training scripts share the same core flags:

- `--env-name <name>` — legacy Gym-style names (`PongNoFrameskip-v4`) and
  Gymnasium names (`ALE/Pong-v5`) are both accepted.
- `--use-cuda-env` — emulate the Atari environments on the GPU. Omit it to use
  the CuLE CPU backend, or pass `--use-openai` to generate training data with
  the reference gymnasium/ale-py emulator.
- `--num-ales N` — number of parallel environments; use hundreds to thousands
  with `--use-cuda-env`.
- `--evaluation-interval N` — frames between evaluations. Each evaluation plays
  10 full episodes on the CPU, so a small interval makes evaluation — not
  training — dominate wall-clock time. The `training time` printed with the
  evaluation results counts only time spent in training updates.

**A2C + V-trace** (best throughput; from `examples/vtrace`):

```
python vtrace_main.py --env-name PongNoFrameskip-v4 --normalize --use-cuda-env --num-ales 1200 --num-steps 20 --num-steps-per-update 1 --num-minibatches 20 --t-max 8000000 --evaluation-interval 2000000
```

Add `--double-test` to additionally evaluate on the reference gymnasium
emulator at every evaluation (doubles evaluation cost; useful to cross-check
CuLE's emulation).

**A2C** (from `examples/a2c`):

```
python a2c_main.py --env-name BreakoutNoFrameskip-v4 --use-cuda-env --num-ales 256 --num-steps 5 --t-max 8000000 --evaluation-interval 2000000
```

**PPO** (from `examples/ppo`):

```
python ppo_main.py --env-name SpaceInvadersNoFrameskip-v4 --use-cuda-env --num-ales 256 --num-steps 20 --t-max 8000000 --evaluation-interval 2000000
```

**DQN** (from `examples/dqn`):

```
python dqn_main.py --env-name SeaquestNoFrameskip-v4 --use-cuda-env --num-ales 32 --t-max 2000000 --memory-capacity 100000
```

**Visualize a trained or random policy** (from `examples/visualize`):

```
python animate.py --env-name BreakoutNoFrameskip-v4 --use-cuda --num-envs 16
```

### Supported games

Any of the following can be passed to `--env-name`, either as
`<CamelCaseName>NoFrameskip-v4` / `ALE/<CamelCaseName>-v5` or as the plain
rom id shown here. These 44 games pass automated health checks (episodes
terminate, rewards flow, no reset loops) on both the CPU and GPU backends:

```
adventure, air_raid, alien, amidar, asterix, asteroids, atlantis, beam_rider,
bowling, boxing, breakout, carnival, chopper_command, crazy_climber,
demon_attack, enduro, fishing_derby, freeway, frostbite, gopher, hero,
ice_hockey, jamesbond, journey_escape, kaboom, kangaroo, krull,
montezuma_revenge, name_this_game, phoenix, pong, pooyan, private_eye,
riverraid, road_runner, robotank, seaquest, solaris, star_gunner, time_pilot,
up_n_down, video_pinball, wizard_of_wor, zaxxon
```

The following games have **broken emulation inherited from upstream CuLE**
(reset loops, episodes that never terminate, or garbage score decoding —
the emulator's game-start handling does not replicate ALE's per-ROM logic;
see [CHANGES.md](CHANGES.md)). They construct and run but are not usable for
training:

```
assault, bank_heist, battle_zone, berzerk, centipede, defender, double_dunk,
elevator_action, gravitar, kung_fu_master, ms_pacman, pitfall, qbert, skiing,
space_invaders, tennis, tutankham, venture, yars_revenge
```

Other ale-py roms load as well, but without game-specific reward/lives
decoding they are not useful for training.

# Testing

The repository ships a pytest suite covering ROM resolution, the CPU and GPU
environment backends (shapes, rewards, determinism, several games across all
supported cartridge formats), the Gymnasium wrapper stack, and end-to-end
micro-training runs for every algorithm:

```
pip install pytest
pytest              # fast checks (~1 minute; GPU tests auto-skip without CUDA)
pytest -m slow      # end-to-end training micro-runs for a2c/vtrace/ppo/dqn
```

# Docker 

The recommended (and easiest) way of using CuLE is through Docker.
We assume nvidia-docker is already installed in your system.
To build the CuLE image you can use the following docker file - create a file named "Dockerfile" in your preferred folder and copy the following text into it:

```
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y --no-install-recommends \
        build-essential git python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch gymnasium ale-py cloudpickle opencv-python-headless \
                psutil pytz tensorboard tqdm

RUN git clone -b master --recursive https://github.com/NVLabs/cule && \
    cd cule && \
    git -C third_party/pybind11 fetch --tags && \
    git -C third_party/pybind11 checkout v2.13.6 && \
    pip install --no-build-isolation .
```

(The same file is available at [envs/Dockerfile](envs/Dockerfile).)

Once the docker file has been created and saved, you can build the docker image by typing (in the same folder of the docker file):

```
sudo nvidia-docker build -t cule_release .
```

You can then run the following command to access a CuLE-ready terminal:

```
sudo nvidia-docker run -it -t cule_release bash
```

You have now access to CuLE and can run different algorithms, including DQN, PPO, A2C, and A2C+V-trace.
A2C+V-trace uses the best batching strategy to leverage the high troughput generated by CuLE.
To replicate the same results reported in our paper (e.g. reaching an average score of 18 for Pong in less than 3 minutes using a single GPU) you can run the following commands:

```
cd cule\examples\vtrace
python vtrace_main.py --env-name PongNoFrameskip-v4 --normalize --use-cuda-env --num-ales 1200 --num-steps 20 --num-steps-per-update 1 --num-minibatches 20 --t-max 8000000 --evaluation-interval 200000
```

The parameters passed to vtrace_main.py specify: the name of the environment (--env-name PongNoFrameskip-v4, same naming convention adopted in OpenAIGym, all our environments are -v4); normalization of the input images (--normalize, this is the normalization procedure normaly adopted in RL for Atari games; notice that, with no normalization, convergence takes way more time); the use of GPU to simulate the environments (--use-cuda-env; if you want to use OpenAI instead, use --use-openai; if you want to use the CuLE CPU backend to generate data, do not specify any of these two); the total number of environments simulated (--num-ales 1200; for an effective use of CuLE, use a large number of environments); the number of steps in the buffer used to compute the discounted rewards (--num-steps 20; if the number of steps is too large, you may saturate the memory); the number of steps after which a DNN update is computed (--num-steps-per-update 1, meaning that an update is computed after each CuLE steps trhough all the environments); the number of minibatches in the total population of environments (in this case --num-minibatches 20 guarantees that each minibatch, composed by 1200 / 20 = 60 environments, advances by 20 steps before using its data to update the DNN; since there are exactly 20 minibatches and one update is computed at each step, it means that for any step simulated by CuLE one batch is providing training data the the GPU for the update; each experience is used only once for training in this case - with other configurations the same experience may be used multiple times, for a more sample efficient data generation strategy; where data between different batches are however correlated); the total number of steps to be performed in training (--t-max 8000000); and the total number of steps to evaluate the DNN on the testing environments (--evaluation-interval 200000). 
The CuLE CPU backend is used by default for testing. If you want to use OpenAIGym instead, use --use-openai-test-env. 

# Citing

```
@inproceedings{NEURIPS2020_e4d78a6b,
 author = {Dalton, Steven and frosio, iuri},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {19773--19782},
 publisher = {Curran Associates, Inc.},
 title = {Accelerating Reinforcement Learning through GPU Atari Emulation},
 url = {https://proceedings.neurips.cc/paper/2020/file/e4d78a6b4d93e1d79241f7b282fa3413-Paper.pdf},
 volume = {33},
 year = {2020}
}

@misc{dalton2019gpuaccelerated,
   title={GPU-Accelerated Atari Emulation for Reinforcement Learning},
   author={Steven Dalton and Iuri Frosio and Michael Garland},
   year={2019},
   eprint={1907.08467},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}
```

# About

CuLE is released by NVIDIA Corporation as Open Source software under the
3-clause "New" BSD license.

# Copyright

Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.

```
  Redistribution and use in source and binary forms, with or without modification, are permitted
  provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright notice, this list of
        conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright notice, this list of
        conditions and the following disclaimer in the documentation and/or other materials
        provided with the distribution.
      * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
        to endorse or promote products derived from this software without specific prior written
        permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
