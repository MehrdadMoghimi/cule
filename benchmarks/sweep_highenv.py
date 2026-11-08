#!/usr/bin/env python3
"""Per-algorithm hyperparameter search at the highest workable environment count.

This is the complement of `benchmark_breakout.py`.  That benchmark pins every
hyperparameter to its published value and tunes only `--num-envs`; this one does
the opposite.  The environment count is fixed as high as the algorithm can go --
1,024, falling back to 512 and then 256 -- and everything else is searched.

Protocol
--------
One game (`BreakoutNoFrameskip-v4`), one backend (CuLE on the GPU), one seed, and
a **wall-clock** budget instead of a transition budget: a configuration is scored
by the training episodic return it reaches after a fixed number of seconds of
training.  That is the only objective a five-minute run can actually measure, and
it is the one that matters when the environment count is chosen for throughput --
it rewards sample efficiency and speed together, and it punishes a cadence the
GPU cannot sustain.

The window is anchored on the start of training rather than on process start, so
that environment construction, the replay warmup and -- for the
`_torchcompile.py` twins -- `torch.compile` and CUDA-graph capture do not eat a
share of the budget that varies by an order of magnitude between variants.  Most
trainers print `SPS:` (or a tqdm `speed:` description) only after their first
gradient step, which is exactly the mark wanted; trainers that print no such line
fall back to their first episodic return, and if a definitive mark arrives later
the window restarts from it.  Startup is measured and reported separately.

Three stages, each resumable, all writing to the same JSONL:

    feasibility  one cheap configuration per algorithm, walking 1024 -> 512 -> 256
                 until the trainer runs.  Fixes the environment count and measures
                 the throughput that sizes `--total-timesteps` for every later
                 trial of that algorithm.
    screen       four or five configurations per algorithm at a short window.
    refine       two learning-rate variations around the screen winner.
    confirm      the best configuration re-run at the full five-minute window.

`--total-timesteps` is derived once per algorithm and then held fixed across its
trials, because the annealing schedules (epsilon, learning rate, entropy) are all
defined as fractions of it.  Sizing it per configuration would make a faster
configuration anneal more slowly and confound the comparison.

Two facts about this repository shape the search space, both measured rather than
assumed:

* `cleanrl_utils.buffers.AtariReplayBuffer` reconstructs frame stacks on the host
  and costs ~61 us per sampled transition -- 111 ms for a batch of 2,048, so an
  eager replay trainer cannot exceed ~9 gradient steps per second at that batch
  no matter how fast the environment runs.  The `_torchcompile.py` twins that hold
  their replay in `LazyTensorStorage(device=cuda)` have no such ceiling, which is
  why the variant is part of the search rather than a presentation detail.
* A GPU-resident transition costs ~56 KB (observation plus next observation, four
  stacked 84x84 frames each), so a device buffer is capped near 131,072 entries on
  a 24 GB card, while the host buffer holds 1M in ~7 GB.  Capacity and sampling
  rate therefore trade off directly against each other.

Outputs, under `benchmark_results/artifacts/breakout/highenv/`:
  trials.jsonl      one record per configuration run
  state.json        chosen environment count and timestep budget per algorithm
  curves/<id>.csv   return-versus-time samples for every trial
  logs/<id>.log     complete trainer output
"""

from __future__ import annotations

import argparse
import atexit
import ast
import csv
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import ALGORITHMS, BY_KEY, ENV_ID, OFF_POLICY, ON_POLICY, STREAMING, ROOT

RESULTS = ROOT / "benchmark_results" / "artifacts" / "breakout" / "highenv"
PYTHON = os.environ.get("CULE_PYTHON", "/home/mehrdad96/anaconda3/envs/cule312/bin/python")

# The environment-count ladder the search walks, highest first.
ENV_LADDER = (1024, 512, 256)

# Replay capacities.  The host buffer stores one 84x84 frame per transition and
# rebuilds stacks while sampling (~7 KB each); a device buffer stores both stacked
# observations of the transition (~56 KB each), so it is 8x more expensive per
# entry and has to stay eight times smaller.
HOST_BUFFER = 1_000_000
GPU_BUFFER = 131_072

# Random play on BreakoutNoFrameskip-v4 returns about 1.2 per full game.  Below
# roughly this, one configuration cannot be told from another.
RANDOM_PLAY_BAND = 5.0

# Below roughly this many transitions in the window, no configuration in this
# sweep has learned anything, so a cadence that starves the data past it is not
# a candidate however many gradient steps it buys.
MIN_TRANSITIONS = 1_000_000

# Twins whose replay lives on the learner device.  The rest keep the host buffer
# and inherit its sampling ceiling, so for them "compiled" buys only the update.
GPU_STORAGE_TWINS = {
    "dqn", "ibdqn", "c51", "qrdqn", "iqn", "fqf", "miqn", "rainbow", "sac",
    "der", "drq", "btr",
}


# --------------------------------------------------------------------------- #
# Trainer introspection: every flag is emitted only if the target file has it.
# --------------------------------------------------------------------------- #

_ARGS_CACHE: dict[Path, dict[str, object]] = {}


def args_fields(path: Path) -> dict[str, object]:
    """Default values of the trainer's `Args` dataclass, read without importing."""
    if path not in _ARGS_CACHE:
        fields: dict[str, object] = {}
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Args":
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        try:
                            fields[stmt.target.id] = ast.literal_eval(stmt.value)
                        except Exception:  # noqa: BLE001 - computed default
                            fields[stmt.target.id] = None
                break
        _ARGS_CACHE[path] = fields
    return _ARGS_CACHE[path]


def has(path: Path, name: str) -> bool:
    return name in args_fields(path)


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


# Learning-rate fields, by the name each trainer happens to use.  A multiplier
# scales whichever of these the file defines, so "lr x2" means the same thing for
# a trainer with one learning rate and for SAC with two.
LR_FIELDS = (
    "learning_rate", "init_lr", "policy_lr", "q_lr", "aux_learning_rate",
    "encoder_learning_rate", "value_learning_rate", "policy_learning_rate",
)


def lr_flags(path: Path, multiplier: float) -> list[str]:
    if multiplier == 1.0:
        return []
    out: list[str] = []
    fields = args_fields(path)
    for name in LR_FIELDS:
        value = fields.get(name)
        if isinstance(value, (int, float)) and value > 0:
            out += [flag(name), f"{value * multiplier:.8g}"]
    return out


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Knobs:
    """A candidate configuration, in units that are meaningful per family."""

    # off-policy
    batch_size: int | None = None
    updates_per_step: float | None = None  # learner updates per vector step
    buffer_size: int | None = None
    # on-policy
    num_steps: int | None = None
    minibatch: int | None = None  # target minibatch size -> --num-minibatches
    update_epochs: int | None = None
    # all families
    lr_mult: float = 1.0
    extra: tuple[str, ...] = ()

    def label_parts(self) -> list[str]:
        parts = []
        if self.batch_size:
            parts.append(f"b{self.batch_size}")
        if self.updates_per_step:
            parts.append(f"u{self.updates_per_step:g}")
        if self.num_steps:
            parts.append(f"T{self.num_steps}")
        if self.minibatch:
            parts.append(f"mb{self.minibatch}")
        if self.update_epochs:
            parts.append(f"e{self.update_epochs}")
        if self.lr_mult != 1.0:
            parts.append(f"lr{self.lr_mult:g}")
        if self.extra:
            # The label is the trial's identity, so anything that changes the
            # configuration has to appear in it.  Leaving these out silently
            # collapsed two candidates into one and the second never ran.
            parts.append("-".join(part.lstrip("-") for part in self.extra))
        return parts


@dataclass
class Trial:
    algorithm: str
    stage: str
    variant: str
    num_envs: int
    knobs: Knobs
    window: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            prefix = "tc" if self.variant == "torchcompile" else "eager"
            self.label = "-".join([prefix] + self.knobs.label_parts()) or prefix

    @property
    def trial_id(self) -> str:
        return f"{self.algorithm}__{self.stage}__{self.label}__n{self.num_envs}"


def trainer_path(algo, variant: str) -> Path:
    return algo.eager_path if variant == "eager" else algo.torchcompile_path


def build_command(algo, trial: Trial, total_timesteps: int, seed: int) -> list[str]:
    path = trainer_path(algo, trial.variant)
    knobs = trial.knobs
    cmd = [
        PYTHON, str(path),
        "--env-backend", "cule",
        "--env-id", ENV_ID,
        "--num-envs", str(trial.num_envs),
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
    ]
    if has(path, "cule_device"):
        cmd += ["--cule-device", "cuda"]
    if trial.variant == "torchcompile":
        if has(path, "compile"):
            cmd.append("--compile")
        if has(path, "cudagraphs") and algo.has_cudagraphs:
            cmd.append("--cudagraphs")

    if knobs.batch_size and has(path, "batch_size"):
        cmd += ["--batch-size", str(knobs.batch_size)]
    if knobs.updates_per_step is not None:
        # Cadence has to be expressed in the flag that wins.  Every replay
        # trainer recomputes `learner_updates_per_vector_step` from
        # `replay_ratio` whenever that is not None, and the `_torchcompile.py`
        # twins *default* it to 1.0 -- so passing the update count directly is
        # silently discarded there and every cadence collapses to ratio 1.
        # The conversion is exact in both directions, whatever the ratio counts
        # (R2D2's are sequences), because the trainer inverts this same formula.
        batch = knobs.batch_size or args_fields(path).get("batch_size") or 1
        if has(path, "replay_ratio"):
            cmd += ["--replay-ratio",
                    f"{knobs.updates_per_step * batch / trial.num_envs:.6g}"]
        elif has(path, "learner_updates_per_vector_step"):
            cmd += ["--learner-updates-per-vector-step", f"{knobs.updates_per_step:g}"]
    if knobs.buffer_size and has(path, "buffer_size"):
        cmd += ["--buffer-size", str(knobs.buffer_size)]
    if knobs.num_steps and has(path, "num_steps"):
        cmd += ["--num-steps", str(knobs.num_steps)]
    if knobs.minibatch and has(path, "num_minibatches"):
        rollout = trial.num_envs * (knobs.num_steps or args_fields(path).get("num_steps") or 1)
        cmd += ["--num-minibatches", str(max(1, round(rollout / knobs.minibatch)))]
    if knobs.update_epochs and has(path, "update_epochs"):
        cmd += ["--update-epochs", str(knobs.update_epochs)]
    cmd += lr_flags(path, knobs.lr_mult)

    for name in algo.budget_flags:  # --anneal-steps and friends carry the budget
        cmd += [name, str(total_timesteps)]
    cmd += list(knobs.extra)
    return cmd


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #


def _tc(algo) -> bool:
    return algo.torchcompile is not None


def off_policy_screen(algo) -> list[tuple[str, Knobs]]:
    """Batch size and gradient cadence, on both sides of the sampling ceiling.

    At 1,024 environments the published cadence (ratio 8 at batch 32) would ask
    for ~15,000 gradient steps per second, which is three orders of magnitude out
    of reach.  What is actually free to choose is the batch size and the number of
    updates per vector step; the replay ratio those imply, `u * batch / num_envs`,
    is recorded rather than dialled.
    """
    device_storage = algo.key in GPU_STORAGE_TWINS
    out: list[tuple[str, Knobs]] = [
        ("eager", Knobs(batch_size=256, updates_per_step=1.0, buffer_size=HOST_BUFFER)),
        ("eager", Knobs(batch_size=1024, updates_per_step=1.0, buffer_size=HOST_BUFFER,
                        lr_mult=2.0)),
    ]
    if not _tc(algo):
        out += [
            ("eager", Knobs(batch_size=512, updates_per_step=2.0, buffer_size=HOST_BUFFER,
                            lr_mult=2.0)),
            ("eager", Knobs(batch_size=2048, updates_per_step=1.0, buffer_size=HOST_BUFFER,
                            lr_mult=4.0)),
        ]
    elif device_storage:
        out += [
            ("torchcompile", Knobs(batch_size=512, updates_per_step=1.0,
                                   buffer_size=GPU_BUFFER, lr_mult=2.0)),
            ("torchcompile", Knobs(batch_size=2048, updates_per_step=1.0,
                                   buffer_size=GPU_BUFFER, lr_mult=4.0)),
            ("torchcompile", Knobs(batch_size=512, updates_per_step=4.0,
                                   buffer_size=GPU_BUFFER, lr_mult=2.0)),
            # Sampling on the device is cheap enough that the published cadence
            # comes back into reach: 16 updates of 512 per vector step is a
            # replay ratio of 8 at 1,024 environments, the DQN paper's number.
            ("torchcompile", Knobs(batch_size=512, updates_per_step=16.0,
                                   buffer_size=GPU_BUFFER, lr_mult=2.0)),
        ]
    else:
        # Host-buffer twin: compilation speeds the update, not the sampling that
        # dominates, so the useful axis is a smaller batch run more often.
        out += [
            ("torchcompile", Knobs(batch_size=256, updates_per_step=1.0,
                                   buffer_size=HOST_BUFFER)),
            ("torchcompile", Knobs(batch_size=512, updates_per_step=1.0,
                                   buffer_size=HOST_BUFFER, lr_mult=2.0)),
        ]
    return out


def on_policy_screen(algo) -> list[tuple[str, Knobs]]:
    """Rollout length, minibatch size and epochs.

    `--num-steps` is the memory knob as well as the horizon knob: the rollout
    buffers are float32 on the learner device, so at 1,024 environments every
    rollout step costs ~119 MB and the shipped 128 does not fit beside the model.

    The minibatch is swept from 512 rather than from the rollout size down,
    because the gradient FLOPs per collected transition do not depend on it but
    the memory does.  Measured on Hadamax-PQN, whose encoder holds 84x84 feature
    maps: 1,024 environments at minibatch 512 runs 1,817 SPS and the same
    configuration at minibatch 2,048 runs 260.  Seven times, from one knob.
    """
    out: list[tuple[str, Knobs]] = [
        ("eager", Knobs(num_steps=16, minibatch=512, update_epochs=4)),
        ("eager", Knobs(num_steps=16, minibatch=2048, update_epochs=4)),
        ("eager", Knobs(num_steps=32, minibatch=2048, update_epochs=4)),
        ("eager", Knobs(num_steps=32, minibatch=8192, update_epochs=2, lr_mult=2.0)),
        ("eager", Knobs(num_steps=64, minibatch=4096, update_epochs=4)),
    ]
    if _tc(algo):
        out += [("torchcompile", Knobs(num_steps=16, minibatch=512, update_epochs=4)),
                ("torchcompile", Knobs(num_steps=32, minibatch=2048, update_epochs=4))]
    return out


def streaming_screen(algo) -> list[tuple[str, Knobs]]:
    """Streaming keeps one update per vector step; the step size is the knob.

    The published algorithm is a single stream, and ObGD's effective step size is
    set for that; 1,024 independent traces average their updates, so the useful
    range is around and above the shipped value.
    """
    out = [("eager", Knobs()), ("eager", Knobs(lr_mult=2.0))]
    if _tc(algo):
        out += [("torchcompile", Knobs()), ("torchcompile", Knobs(lr_mult=2.0)),
                ("torchcompile", Knobs(lr_mult=0.5))]
    return out


# BBF anneals epsilon to a fully greedy policy over 2,001 transitions, which is
# two vector steps at 1,024 environments: it then plays an untrained network
# deterministically, never fires the ball, and scores below random play.  The
# schedule is an Atari-100K artefact rather than a property of the algorithm, so
# the search gives it one sized for this budget instead.
BBF_EXPLORATION = ("--epsilon-decay-period", "200000", "--end-e", "0.01")

# Algorithms whose generic ladder would be wrong, with the reason.
SPECIAL_SCREENS: dict[str, list[tuple[str, Knobs]]] = {
    "bbf": [
        ("torchcompile", Knobs(batch_size=256, updates_per_step=1.0,
                               buffer_size=HOST_BUFFER, extra=BBF_EXPLORATION)),
        ("eager", Knobs(batch_size=256, updates_per_step=1.0,
                        buffer_size=HOST_BUFFER, extra=BBF_EXPLORATION)),
        ("torchcompile", Knobs(batch_size=512, updates_per_step=1.0,
                               buffer_size=HOST_BUFFER, lr_mult=2.0,
                               extra=BBF_EXPLORATION)),
        ("eager", Knobs(batch_size=256, updates_per_step=4.0,
                        buffer_size=HOST_BUFFER, extra=BBF_EXPLORATION)),
    ],
    # No minibatching: the rollout is the batch and one update follows it, so the
    # only knobs are the rollout length and the step size.
    "a2c": [("eager", Knobs(num_steps=5)), ("eager", Knobs(num_steps=16)),
            ("eager", Knobs(num_steps=16, lr_mult=2.0)), ("eager", Knobs(num_steps=32))],
    "a2c_sil": [("eager", Knobs(num_steps=5)), ("eager", Knobs(num_steps=16)),
                ("eager", Knobs(num_steps=16, lr_mult=2.0)), ("eager", Knobs(num_steps=32))],
    # V-trace corrects a whole trajectory batch at once; one epoch is the design.
    "impala": [("eager", Knobs(num_steps=20, minibatch=8192, update_epochs=1)),
               ("eager", Knobs(num_steps=20, minibatch=4096, update_epochs=1, lr_mult=2.0)),
               ("eager", Knobs(num_steps=40, minibatch=8192, update_epochs=1)),
               ("eager", Knobs(num_steps=40, minibatch=8192, update_epochs=2))],
    # The auxiliary phase holds n_pi rollouts of frames at once, so n_pi and the
    # rollout length together set a host buffer that grows with the env count.
    "ppg": [("eager", Knobs(num_steps=16, minibatch=1024, extra=("--n-pi", "4"))),
            ("eager", Knobs(num_steps=32, minibatch=1024, extra=("--n-pi", "4"))),
            ("eager", Knobs(num_steps=16, minibatch=4096, extra=("--n-pi", "4"))),
            ("eager", Knobs(num_steps=16, minibatch=1024, extra=("--n-pi", "8")))],
    # The observation-normalisation prefill is `num_steps * N` uncounted random
    # play per iteration; 50 of them at 1,024 envs is millions of wasted frames.
    "ppo_rnd": [("eager", Knobs(num_steps=16, minibatch=1024, update_epochs=4,
                                extra=("--num-iterations-obs-norm-init", "4"))),
                ("eager", Knobs(num_steps=32, minibatch=1024, update_epochs=4,
                                extra=("--num-iterations-obs-norm-init", "4"))),
                ("eager", Knobs(num_steps=32, minibatch=4096, update_epochs=2, lr_mult=2.0,
                                extra=("--num-iterations-obs-norm-init", "4"))),
                ("eager", Knobs(num_steps=64, minibatch=2048, update_epochs=4,
                                extra=("--num-iterations-obs-norm-init", "4")))],
    # Transformer-XL keeps a 119-step memory per environment; the episodic memory
    # buffer alone is num_envs * max_episode_steps * layers * dim.
    "ppo_trxl": [("eager", Knobs(num_steps=16, minibatch=512, update_epochs=3)),
                 ("eager", Knobs(num_steps=32, minibatch=512, update_epochs=3)),
                 ("eager", Knobs(num_steps=16, minibatch=2048, update_epochs=3)),
                 ("eager", Knobs(num_steps=8, minibatch=512, update_epochs=3, lr_mult=2.0))],
    # DiscoRL's meta-network consumes a whole trajectory: no minibatch knob.
    "disco": [("eager", Knobs(num_steps=16)), ("eager", Knobs(num_steps=29)),
              ("eager", Knobs(num_steps=29, lr_mult=2.0)),
              ("torchcompile", Knobs(num_steps=29))],
    # Endpoint Replay gates on `vector_step % train_frequency`, so at most one
    # update per vector step, and its two buffers are sized in transitions.
    "endpoint_ddqn": [
        ("eager", Knobs(batch_size=256, extra=("--train-frequency", "1"))),
        ("eager", Knobs(batch_size=1024, lr_mult=2.0, extra=("--train-frequency", "1"))),
        ("eager", Knobs(batch_size=1024, lr_mult=2.0,
                        extra=("--train-frequency", "1", "--recency-buffer-size", "200000",
                               "--coreset-buffer-size", "200000"))),
        ("eager", Knobs(batch_size=2048, lr_mult=4.0,
                        extra=("--train-frequency", "1", "--recency-buffer-size", "200000",
                               "--coreset-buffer-size", "200000"))),
    ],
    # R2D2 replays sequences of 40 steps: one sampled batch is 32*40 transitions,
    # and the host buffer pays for every one of them.
    "r2d2": [("eager", Knobs(batch_size=32, updates_per_step=1.0)),
             ("eager", Knobs(batch_size=64, updates_per_step=1.0, lr_mult=2.0)),
             ("torchcompile", Knobs(batch_size=32, updates_per_step=1.0)),
             ("torchcompile", Knobs(batch_size=64, updates_per_step=2.0, lr_mult=2.0))],
}


def screen_trials(algo, num_envs: int, window: float) -> list[Trial]:
    if algo.key in SPECIAL_SCREENS:
        candidates = SPECIAL_SCREENS[algo.key]
    elif algo.family == OFF_POLICY:
        candidates = off_policy_screen(algo)
    elif algo.family == ON_POLICY:
        candidates = on_policy_screen(algo)
    else:
        candidates = streaming_screen(algo)
    trials = []
    for variant, knobs in candidates:
        if variant == "torchcompile" and not _tc(algo):
            continue
        trials.append(Trial(algo.key, "screen", variant, num_envs, knobs, window))
    return trials


def refine_trials(algo, winner: Trial, window: float,
                  records: list[dict] | None = None) -> list[Trial]:
    """Two learning-rate steps around the screen winner.

    Batch size and cadence move the learning rate's meaning, so once the screen
    has chosen them the step size is the coordinate most likely to be wrong.

    Unless the screen could not tell its configurations apart.  When nothing
    cleared random play the "winner" is a coin flip and refining its step size
    refines noise; the axis worth walking is instead the gradient cadence, which
    is what separated C51's 55.7 from its own 1.26 at the same batch size.
    """
    if records and algo.family == OFF_POLICY:
        best = max((r.get("score") or 0.0) for r in records)
        cadenced = [r for r in records
                    if (r.get("knobs") or {}).get("updates_per_step")
                    and r["status"] == "ok"]
        if best < RANDOM_PLAY_BAND and cadenced:
            top = max(cadenced, key=lambda r: r["knobs"]["updates_per_step"])
            knobs = Knobs(**{k: (tuple(v) if k == "extra" else v)
                             for k, v in top["knobs"].items()})
            return [Trial(algo.key, "refine", top["variant"], top["num_envs"],
                          replace(knobs,
                                  updates_per_step=knobs.updates_per_step * factor),
                          window)
                    for factor in (4, 16)]
    out = []
    for factor in (0.5, 2.0):
        knobs = replace(winner.knobs, lr_mult=winner.knobs.lr_mult * factor)
        out.append(Trial(algo.key, "refine", winner.variant, winner.num_envs, knobs, window))
    # Where replay sampling is on the device the gradient cadence is nearly free,
    # so the screen's best cadence is worth pushing one more notch before the
    # step size is blamed for whatever is left.
    if algo.key in GPU_STORAGE_TWINS and winner.knobs.updates_per_step:
        knobs = replace(winner.knobs, updates_per_step=winner.knobs.updates_per_step * 4)
        out.append(Trial(algo.key, "refine", winner.variant, winner.num_envs, knobs, window))
    return out


# --------------------------------------------------------------------------- #
# Running a trial
# --------------------------------------------------------------------------- #

# Every trainer prints one of these only after its first gradient step, which is
# what makes them a fair start line: compilation and graph capture happen before.
ANCHOR_RE = re.compile(r"SPS:\s*\d+|speed:\s*-?[\d.]+\s*sps")

STATS_RE = re.compile(
    r"global_step=(\d+), completed_episodes=(\d+), "
    r"batch_mean_return=([-\d.eE+]+), mean_return_(\d+)=([-\d.eE+]+)"
)
SIMPLE_RE = re.compile(r"global_step=(\d+), episodic_return=([-\d.eE+]+)")
# tqdm descriptions of the compiled twins: DQN-line reports a 20-episode mean as
# `episodic_return=`, the PPO twin as `returns:`.
TQDM_RET_RE = re.compile(r"(?:episodic_return=|returns:\s*)(-?[\d.]+|nan)")
TQDM_BAR_RE = re.compile(r"(\d+)/(\d+) \[")
PROGRESS_RE = re.compile(r"TRAINING_PROGRESS (\d+)")
# The PQN-line twins carry the transition count in the tqdm postfix, which is the
# only step signal they emit between episode flushes.
FRAMES_RE = re.compile(r"frames=([\d.]+(?:e\+?\d+)?)")


def _terminate(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=25)
            return
        except Exception:  # noqa: BLE001 - best effort teardown
            continue


def run_trial(cmd: list[str], bar_scale: int, window: float, log_path: Path,
              curve_path: Path, startup_cap: float, stats_interval: int = 32) -> dict:
    """Run one configuration and score it `window` seconds after training starts.

    Returns the last episodic return observed inside the window, plus the
    throughput and update cadence the configuration actually sustained.
    """
    started = time.perf_counter()
    anchor: float | None = None
    # A return line can precede the first gradient step, and for a compiled twin
    # the compile happens between the two.  Anchoring on the return would hand
    # the twin a window with a 60 s stall in it, so the first return only sets a
    # provisional mark and the first `SPS:`/`speed:` line replaces it.
    anchor_final = False
    samples: list[dict] = []
    steps_at_anchor: int | None = None
    last_step = 0
    reported_sps: list[float] = []
    tail: list[str] = []
    status = "ok"

    # A two-minute window at the default flush cadence of 128 vector steps gives
    # a slow trainer three curve points; 32 gives it a dozen, at the cost of a
    # few more host synchronizations that the wrapper batches anyway.
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               CULE_STATS_INTERVAL=str(stats_interval))
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env, bufsize=0)
    fd = proc.stdout.fileno()
    pending = b""
    log_file = log_path.open("wb")
    try:
        while True:
            now = time.perf_counter()
            if anchor is None and now - started > startup_cap:
                status = "no_anchor"
                break
            if anchor is not None and now - anchor > window:
                status = "ok"
                break
            ready, _, _ = select.select([fd], [], [], 1.0)
            if ready:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                log_file.write(chunk)
                pending += chunk
                # tqdm redraws with a carriage return, so lines are not enough.
                parts = re.split(rb"[\r\n]", pending)
                pending = parts.pop()
                for raw in parts:
                    line = raw.decode("utf-8", "replace")
                    if not line.strip():
                        continue
                    tail.append(line)
                    if len(tail) > 400:
                        del tail[:200]
                    now = time.perf_counter()
                    anchor_match = ANCHOR_RE.search(line)
                    if anchor_match:
                        digits = re.search(r"([\d.]+)", anchor_match.group(0))
                        if digits:
                            reported_sps.append(float(digits.group(1)))
                    # Only an *early* progress line is a start mark.  Several
                    # trainers print their `SPS:` summary once, at exit, and a
                    # run that finishes inside its window would otherwise reset
                    # the clock at the very end and discard its whole curve.
                    if not anchor_final and anchor_match and len(samples) < 3:
                        anchor, anchor_final = now, True
                        steps_at_anchor = last_step
                        samples.clear()

                    step = None
                    ret = None
                    match = STATS_RE.search(line)
                    if match:
                        step, ret = int(match.group(1)), float(match.group(5))
                    else:
                        match = SIMPLE_RE.search(line)
                        if match:
                            step, ret = int(match.group(1)), float(match.group(2))
                        else:
                            match = TQDM_RET_RE.search(line)
                            if match and "sps" in line:
                                value = match.group(1)
                                ret = None if value == "nan" else float(value)
                                bar = TQDM_BAR_RE.search(line)
                                progress = PROGRESS_RE.search(line)
                                if progress:
                                    step = int(progress.group(1))
                                elif bar:
                                    step = int(bar.group(1)) * bar_scale
                    if step is None:
                        # Any step signal keeps the throughput estimate alive on
                        # trainers that go minutes between episode flushes.
                        progress = PROGRESS_RE.search(line)
                        frames = FRAMES_RE.search(line)
                        bar = TQDM_BAR_RE.search(line)
                        if progress:
                            step = int(progress.group(1))
                        elif frames:
                            step = int(float(frames.group(1)))
                        elif bar and "sps" in line.lower():
                            # A twin's progress bar counts vector steps for a
                            # replay trainer but whole rollouts for an on-policy
                            # one, which is num_steps transitions each.
                            step = int(bar.group(1)) * bar_scale
                    if step is not None:
                        last_step = max(last_step, step)
                    if ret is not None:
                        if anchor is None:  # provisional, replaced by SPS/speed
                            anchor, steps_at_anchor = now, last_step
                        samples.append({
                            "seconds": round(now - anchor, 3),
                            "global_step": last_step,
                            "mean_return": ret,
                        })
            if proc.poll() is not None and not ready:
                status = "ok" if proc.returncode == 0 else "error"
                break
    finally:
        log_file.close()
        if proc.poll() is None:
            _terminate(proc)
        else:
            proc.wait(timeout=10)
        if samples:
            with curve_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)

    elapsed = time.perf_counter() - started
    trained = (time.perf_counter() - anchor) if anchor else 0.0
    inside = [s for s in samples if s["seconds"] <= window]
    steps_trained = max(0, last_step - (steps_at_anchor or 0))
    # The last single sample of a run is one 20-episode moving average and moves
    # by several points between consecutive prints.  Averaging the closing 40% of
    # the window costs nothing and makes two configurations separable when the
    # difference between them is smaller than that jitter.
    closing = [s["mean_return"] for s in inside if s["seconds"] >= 0.6 * window]
    if not closing:
        closing = [s["mean_return"] for s in inside[-3:]]
    # Step deltas need two parsed samples; a trainer that printed only its own
    # cumulative `SPS:` still reports a usable throughput.
    measured_sps = round(steps_trained / trained) if trained > 1 and steps_trained else None
    return {
        "status": status,
        "returncode": proc.returncode,
        "wall_seconds": round(elapsed, 1),
        "startup_seconds": round(anchor - started, 1) if anchor else None,
        "anchor": ("gradient_step" if anchor_final else "first_return") if anchor else None,
        "trained_seconds": round(trained, 1),
        "samples": len(samples),
        "samples_in_window": len(inside),
        "final_step": last_step,
        "train_sps": measured_sps or (round(reported_sps[-1]) if reported_sps else None),
        "measured_sps": measured_sps,
        "reported_sps": round(reported_sps[-1]) if reported_sps else None,
        "score": round(sum(closing) / len(closing), 3) if closing else None,
        "final_return": inside[-1]["mean_return"] if inside else None,
        "peak": max((s["mean_return"] for s in inside), default=None),
        "tail": "\n".join(tail)[-4000:],
    }


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #


def acquire_lock(results: Path) -> Path:
    """Refuse to start while another sweep owns this artifact directory.

    Two concurrent sweeps do not merely race on the JSONL -- they share the GPU,
    so every throughput number either one records is wrong, and the interleaved
    stdout makes the damage hard to even see.  That happened once; the lock is
    cheaper than detecting it again.
    """
    lock = results / "sweep.lock"
    if lock.exists():
        try:
            owner = int(lock.read_text().strip())
            os.kill(owner, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            print(f"clearing stale lock from pid {lock.read_text().strip()}")
            lock.unlink(missing_ok=True)
        else:
            raise SystemExit(
                f"another sweep (pid {owner}) is using {results}; "
                f"stop it first or pass a different --results"
            )
    lock.write_text(str(os.getpid()))
    return lock


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def load_trials(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                out[record["trial_id"]] = record
    return out


def size_budget(sps: float | None, window: float) -> int:
    """`--total-timesteps` for a whole algorithm, from its measured throughput.

    Rounded to two significant figures so that a re-measured throughput does not
    silently change the annealing schedule between stages.
    """
    # A budget that is too small is worse than one that is too large: the run
    # ends before the window and the schedules compress into it, so the floor is
    # generous and the estimate is only trusted inside a wide band.
    transitions = (sps or 8000) * window
    magnitude = 10 ** (math.floor(math.log10(transitions)) - 1)
    rounded = int(round(transitions / magnitude) * magnitude)
    return max(500_000, min(30_000_000, rounded))


def record_trial(algo, trial: Trial, total_timesteps: int, seed: int, result: dict,
                 cmd: list[str]) -> dict:
    knobs = trial.knobs
    ratio = None
    if knobs.batch_size and knobs.updates_per_step:
        ratio = knobs.updates_per_step * knobs.batch_size / trial.num_envs
    return {
        "trial_id": trial.trial_id,
        "algorithm": trial.algorithm,
        "label": trial.label,
        "stage": trial.stage,
        "variant": trial.variant,
        "num_envs": trial.num_envs,
        "window": trial.window,
        "knobs": {k: v for k, v in knobs.__dict__.items() if v is not None and v != ()},
        "implied_replay_ratio": ratio,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "command": cmd,
        "started_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }


def append_trial(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def execute(algo, trial: Trial, state: dict, args, trials_path: Path,
            done: dict[str, dict], total_override: int | None = None) -> dict:
    if trial.trial_id in done and done[trial.trial_id]["status"] in ("ok", "error", "no_anchor"):
        return done[trial.trial_id]
    total = total_override or state[algo.key]["total_timesteps"]
    cmd = build_command(algo, trial, total, args.seed)
    # The bar of an on-policy twin ticks once per rollout, not once per step.
    bar_scale = trial.num_envs
    if algo.family == ON_POLICY:
        path = trainer_path(algo, trial.variant)
        bar_scale *= trial.knobs.num_steps or args_fields(path).get("num_steps") or 1
    print(f"    {trial.stage:11s} {trial.label:28s} n={trial.num_envs} ", end="", flush=True)
    if args.dry_run:
        print("\n      " + " ".join(cmd[1:]))
        return {"status": "dry", "score": None}
    result = run_trial(cmd, bar_scale, trial.window,
                       args.results / "logs" / f"{trial.trial_id}.log",
                       args.results / "curves" / f"{trial.trial_id}.csv",
                       args.startup_cap, args.stats_interval)
    record = record_trial(algo, trial, total, args.seed, result, cmd)
    append_trial(trials_path, record)
    done[trial.trial_id] = record
    score = result["score"]
    print(f"score={('%.2f' % score) if score is not None else '  -  '} "
          f"sps={result['train_sps'] or 0:>6} start={result['startup_seconds'] or 0:>5}s "
          f"[{result['status']}]", flush=True)
    if result["status"] not in ("ok",):
        print("      " + result["tail"][-400:].replace("\n", "\n      "), flush=True)
    return record


def best_of(records: list[dict]) -> dict | None:
    """Best score, throughput breaking ties -- including the tie at zero.

    A slow algorithm can finish a two-minute window before Breakout finishes an
    episode, so several configurations legitimately score nothing.  Dropping them
    would leave the algorithm with no winner at all; ranking them by the
    transitions they got through picks the one most likely to be ahead when the
    window is five minutes instead of two, which is the question being asked.
    """
    usable = [r for r in records if r["status"] == "ok"]
    if not usable:
        return None
    return max(usable, key=lambda r: (r.get("score") or 0.0, r.get("train_sps") or 0))


def trial_from_record(record: dict, stage: str, window: float) -> Trial:
    knobs = Knobs(**{k: (tuple(v) if k == "extra" else v)
                     for k, v in record["knobs"].items()})
    return Trial(record["algorithm"], stage, record["variant"], record["num_envs"],
                 knobs, window, label=record["label"] if stage == record["stage"] else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=[a.key for a in ALGORITHMS])
    parser.add_argument("--stages", nargs="+", default=["feasibility", "screen", "refine", "confirm"])
    parser.add_argument("--screen-window", type=float, default=120.0)
    parser.add_argument("--confirm-window", type=float, default=300.0)
    parser.add_argument("--feasibility-window", type=float, default=40.0)
    parser.add_argument("--startup-cap", type=float, default=300.0,
                        help="seconds allowed before the first progress line")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stats-interval", type=int, default=32,
                        help="vector steps between episode-statistic flushes")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--confirm-top", type=int, default=2,
                        help="replay configurations sent to the full window")
    parser.add_argument("--preflight", action="store_true",
                        help="run one short probe per algorithm and variant, then stop")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for sub in ("logs", "curves"):
        (args.results / sub).mkdir(parents=True, exist_ok=True)
    trials_path = args.results / "trials.jsonl"
    state_path = args.results / "state.json"
    lock = acquire_lock(args.results)
    atexit.register(lambda: lock.unlink(missing_ok=True))
    state = load_state(state_path)
    done = load_trials(trials_path)

    if args.preflight:
        # One eager and one compiled configuration per algorithm at a token
        # window.  This does not measure anything -- it proves the command line
        # is accepted, the trainer fits at this environment count, and the
        # output format is one the parser understands, before a long unattended
        # run stakes hours on those assumptions.
        failures = []
        for index, key in enumerate(args.algorithms, start=1):
            algo = BY_KEY[key]
            candidates = screen_trials(algo, ENV_LADDER[0], args.feasibility_window)
            probes = []
            for variant in ("eager", "torchcompile"):
                first = next((t for t in candidates if t.variant == variant), None)
                if first is not None:
                    probes.append(replace(first, stage="preflight"))
            print(f"\n[{index}/{len(args.algorithms)}] {key}", flush=True)
            state.setdefault(key, {"total_timesteps": 1_000_000})
            for probe in probes:
                record = execute(algo, probe, state, args, trials_path, done)
                if record["status"] != "ok" or not record.get("samples"):
                    failures.append((key, probe.label, record["status"],
                                     record.get("samples")))
        print("\npreflight summary")
        for key, label, status, samples in failures:
            print(f"  CHECK {key:14s} {label:28s} {status} samples={samples}")
        if not failures:
            print("  all probes ran and produced a parseable return")
        return

    for index, key in enumerate(args.algorithms, start=1):
        algo = BY_KEY[key]
        print(f"\n[{index}/{len(args.algorithms)}] {key} ({algo.label}) {algo.family}",
              flush=True)
        entry = state.setdefault(key, {})

        # -- feasibility: walk the ladder until the trainer runs -------------- #
        if "num_envs" not in entry and "feasibility" in args.stages:
            entry.setdefault("total_timesteps", 1_000_000)
            for envs in ENV_LADDER:
                probe = screen_trials(algo, envs, args.feasibility_window)[0]
                probe = replace(probe, stage="feasibility")
                record = execute(algo, probe, state, args, trials_path, done)
                if args.dry_run:
                    entry["num_envs"] = envs
                    break
                # The question here is only whether the trainer *runs* at this
                # environment count.  A slow algorithm can start cleanly and
                # still report nothing inside a 40 s probe -- treating that as
                # "does not fit" would drop it down the ladder for being slow,
                # which is the opposite of what the ladder is for.
                if record["status"] == "ok" and record.get("startup_seconds") is not None:
                    entry["num_envs"] = envs
                    entry["probe_sps"] = record.get("train_sps") or None
                    entry["total_timesteps"] = size_budget(entry["probe_sps"],
                                                           args.confirm_window)
                    break
                print(f"      {envs} envs did not start ({record['status']}); "
                      f"stepping down the ladder", flush=True)
            else:
                entry["num_envs"] = None
                entry["note"] = "no environment count in the ladder ran"
            save_state(state_path, state)

        envs = entry.get("num_envs")
        if not envs:
            print("    skipped: nothing in the ladder ran", flush=True)
            continue
        print(f"    envs={envs} budget={entry['total_timesteps']:,} "
              f"(probe {entry.get('probe_sps')} sps)", flush=True)

        # -- screen ------------------------------------------------------- #
        # The ladder applies here too: a cheap probe can survive 1,024
        # environments while every configuration worth running there runs out
        # of memory, and the instruction is to drop the environment count only
        # when the algorithm genuinely cannot hold it.
        if "screen" in args.stages:
            while True:
                for trial in screen_trials(algo, envs, args.screen_window):
                    execute(algo, trial, state, args, trials_path, done)
                scored = [r for r in done.values() if r["algorithm"] == key
                          and r["stage"] == "screen" and r["num_envs"] == envs]
                if args.dry_run or best_of(scored) is not None:
                    break
                lower = next((n for n in ENV_LADDER if n < envs), None)
                if lower is None:
                    break
                print(f"    nothing scored at {envs} envs; stepping down to {lower}",
                      flush=True)
                envs = entry["num_envs"] = lower
                save_state(state_path, state)

        records = [r for r in done.values() if r["algorithm"] == key
                   and r["stage"] == "screen" and r["num_envs"] == envs]
        winner = best_of(records)
        if winner is None:
            print("    no screen configuration produced a return", flush=True)
            continue

        # -- refine ---------------------------------------------------------- #
        if "refine" in args.stages:
            for trial in refine_trials(algo, trial_from_record(winner, "screen",
                                                               args.screen_window),
                                       args.screen_window, records):
                execute(algo, trial, state, args, trials_path, done)

        records = [r for r in done.values() if r["algorithm"] == key
                   and r["stage"] in ("screen", "refine") and r["num_envs"] == envs]
        winner = best_of(records) or winner

        # -- confirm --------------------------------------------------------- #
        if "confirm" in args.stages:
            # A short screen is biased against a high gradient cadence: those
            # configurations spend the early window behind and overtake later.
            # Measured on DQN, where the 120 s screen ranked 4 updates/step
            # (38.3) over 16 (13.1) and the 300 s confirm reversed it, 351 to
            # 69.  So the replay family sends its best two to the full window.
            ranked = sorted(records, key=lambda r: -(r.get("score") or 0.0))
            finalists, seen = [], set()
            limit = args.confirm_top if algo.family == OFF_POLICY else 1
            usable = [r for r in records if r["status"] == "ok" and r.get("train_sps")]
            if limit > 1 and usable and (ranked[0].get("score") or 0.0) < RANDOM_PLAY_BAND:
                # Nothing in the screen cleared random play, so its ordering is
                # noise and picking the top two by score wastes both confirms on
                # a coin flip.  C51 is the evidence: every screen configuration
                # scored 1.6-4.3 and the one that happened to win returned 55.7
                # over five minutes.  Bracket the trade-off instead -- the
                # configuration that collects the most transitions per second,
                # and the one that takes the most gradient steps per second.
                def updates_per_second(record: dict) -> float:
                    cadence = (record.get("knobs") or {}).get("updates_per_step") or 0.0
                    return cadence * record["train_sps"] / max(record["num_envs"], 1)

                # The cadence optimum is interior, so the second finalist is the
                # heaviest cadence that still collects enough data to learn from
                # rather than the heaviest outright.  C51 is the evidence on both
                # sides: ratio 32 keeps 2.2M transitions and returns 55.7, ratio
                # 128 buys 17% more gradient steps for a quarter of the data and
                # falls to 4.4.
                fed = [r for r in usable
                       if r["train_sps"] * args.confirm_window >= MIN_TRANSITIONS]
                ranked = [max(usable, key=lambda r: r["train_sps"]),
                          max(fed or usable, key=updates_per_second)] + ranked
            for record in ranked:
                if record["label"] in seen:
                    continue
                seen.add(record["label"])
                finalists.append(record)
                if len(finalists) >= limit:
                    break
            confirmed = []
            for candidate in finalists:
                trial = trial_from_record(candidate, "confirm", args.confirm_window)
                trial.label = candidate["label"]
                total = max(state[algo.key]["total_timesteps"],
                            size_budget(candidate.get("train_sps"), args.confirm_window))
                confirmed.append(execute(algo, trial, state, args, trials_path, done,
                                         total_override=total))
            best = max(confirmed, key=lambda r: (r.get("score") or 0.0))
            entry["best"] = {
                "label": best.get("label"), "variant": best.get("variant"),
                "knobs": best.get("knobs"), "confirm_score": best.get("score"),
                "confirm_sps": best.get("train_sps"),
            }
            save_state(state_path, state)

    save_state(state_path, state)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
