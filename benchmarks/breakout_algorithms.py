#!/usr/bin/env python3
"""Registry of every `cleanrl/` trainer for the Breakout learning benchmark.

One entry per *algorithm*, not per file.  Each entry names the eager trainer and
its `*_torchcompile.py` twin -- the lineage doc is explicit that a twin is the
same maths reorganised for capture, never a different algorithm -- plus the facts
the runner needs to build a valid command line.

Two things in here are load-bearing and easy to get wrong.

**The twins ship different defaults.**  `dqn_torchcompile.py` and friends default
to `--num-envs 256 --batch-size 512`, the repo's vectorized recipe, while their
eager counterparts default to the paper's `--num-envs 1 --batch-size 32`.  The
benchmark therefore pins every hyperparameter explicitly instead of inheriting
whichever default the chosen file happens to carry, so "eager" and "compiled" are
the same configuration at different speeds and nothing else.

**The gradient cadence has to be stated, not inherited.**  A replay trainer's
shipped `num_envs=1` with one minibatch per transition implies a replay ratio
equal to its batch size -- 32 for the DQN family, which is *four times* the
published cadence of one minibatch every four agent steps.  Raising `--num-envs`
without saying anything would instead divide that ratio by the environment
count.  `paper_replay_ratio` records what the paper actually does, in the units
the repo's own `--replay-ratio` flag uses (sampled replay items per collected
transition), and the benchmark holds it fixed while environment count moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEANRL = ROOT / "cleanrl"

# How a trainer turns `--num-envs` into learning signal.
OFF_POLICY = "off_policy"  # replay buffer; gradient cadence set by --replay-ratio
ON_POLICY = "on_policy"  # rollout of --num-steps; batch = num_envs * num_steps
STREAMING = "streaming"  # one update per transition, no replay, no batching


@dataclass
class Algorithm:
    key: str
    eager: str
    label: str
    family: str
    paper: str
    torchcompile: str | None = None

    # --- off-policy ---
    batch_size: int | None = None
    # Sampled replay items per collected transition, as published.  For the DQN
    # line this is batch/update_period = 32/4 = 8.
    paper_replay_ratio: float | None = None
    # What the trainer's own `num_envs=1` default implies, kept only so the
    # report can state how far the benchmark sits from the shipped file.
    shipped_replay_ratio: float | None = None

    # --- on-policy ---
    num_steps: int | None = None
    # Scaling `--num-envs` multiplies the rollout batch, so the benchmark raises
    # `--num-minibatches` in step to hold the *minibatch* size and
    # `--update-epochs` at their defaults.  Gradient work per collected
    # transition is then identical to the shipped configuration.
    minibatch_size: int | None = None
    update_epochs: int | None = None

    # --- capability flags: the trainers do not share one arg surface ---
    has_cule_device: bool = True
    # `--benchmark` exists on every twin but is missing from two eager files
    # (sac_atari.py, ppo_atari_lstm.py), so the flag is per variant.
    has_benchmark: bool = True
    has_benchmark_eager: bool = True
    has_replay_ratio: bool = True
    # Trainers whose shipped defaults already *are* the published vectorized
    # configuration.  Overriding their cadence would only introduce error --
    # and for R2D2 `--replay-ratio` counts sampled *sequences*, not transitions,
    # so a transition-valued ratio would be wrong by a factor of seq_len.
    use_shipped_cadence: bool = False
    has_cudagraphs: bool = True
    has_buffer_size: bool = True
    has_batch_size: bool = True

    # Environment counts to sweep, when the shared sweep does not apply.
    env_sweep: tuple[int, ...] | None = None
    # Fixed environment count: the trainer's own vectorized design point.
    pinned_envs: int | None = None
    extra: list[str] = field(default_factory=list)
    # Flags whose value is the run's total timestep budget.  `ppo_trxl_atari.py`
    # anneals its learning rate and entropy coefficient over an explicit
    # `--anneal-steps` rather than over `--total-timesteps`, and its default is
    # the shipped 10M budget; left alone at 1M the schedule would cover a tenth
    # of its range and the trainer would silently not anneal, unlike every other
    # annealing trainer in the benchmark.
    budget_flags: tuple[str, ...] = ()
    notes: str = ""

    @property
    def eager_path(self) -> Path:
        return CLEANRL / self.eager

    @property
    def torchcompile_path(self) -> Path | None:
        return CLEANRL / self.torchcompile if self.torchcompile else None


def _dqn(key, eager, label, paper, *, batch=32, update_period=4, **kw) -> Algorithm:
    """DQN-line entry: replay ratio is batch / update_period, as published."""
    return Algorithm(
        key=key,
        eager=eager,
        torchcompile=kw.pop("torchcompile", eager.replace(".py", "_torchcompile.py")),
        label=label,
        family=OFF_POLICY,
        paper=paper,
        batch_size=batch,
        paper_replay_ratio=batch / update_period,
        shipped_replay_ratio=float(batch),
        **kw,
    )


ALGORITHMS: list[Algorithm] = [
    # ---- DQN line: batch 32, one update every 4 agent steps -> ratio 8 -------
    _dqn("dqn", "dqn_atari.py", "DQN", "Mnih et al. 2015",
         torchcompile="dqn_torchcompile.py"),
    _dqn("ibdqn", "ibdqn_atari.py", "IB-DQN", "Nagarajan et al. ICML 2026"),
    _dqn("c51", "c51_atari.py", "C51", "Bellemare et al. 2017", has_cudagraphs=False),
    _dqn("qrdqn", "qrdqn_atari.py", "QR-DQN", "Dabney et al. 2017"),
    _dqn("iqn", "iqn_atari.py", "IQN", "Dabney et al. 2018"),
    _dqn("fqf", "fqf_atari.py", "FQF", "Yang et al. 2019"),
    _dqn("miqn", "miqn_atari.py", "M-IQN", "Vieillard et al. 2020"),
    _dqn("rainbow", "rainbow_atari.py", "Rainbow", "Hessel et al. 2018"),
    # SAC-discrete: batch 64, one update every 4 steps.
    _dqn("sac", "sac_atari.py", "SAC-discrete", "Christodoulou 2019",
         batch=64, has_benchmark_eager=False),
    # Endpoint Replay gates its learner on `vector_step % train_frequency == 0`,
    # so it performs at most ONE update per vector step and its achievable ratio
    # is batch/(train_frequency * num_envs).  Unlike the rest of the family it
    # cannot buy the ratio back with several updates per step, so holding the
    # published ratio of 8 pins it to four environments at train_frequency 1.
    # Its two buffers are algorithmic, not a capacity knob.
    _dqn("endpoint_ddqn", "endpoint_ddqn_atari.py", "Endpoint-Replay DDQN",
         "Panahi et al. RLC 2026", torchcompile=None, has_replay_ratio=False,
         has_buffer_size=False, env_sweep=(1, 2, 4, 8, 16),
         extra=["--train-frequency", "1"],
         notes="env-count-limited: at most one update per vector step"),
    # ---- Vectorized-by-design trainers: keep their own environment count -----
    # BTR ships 64 environments, batch 256, one update per vector step: ratio 4
    # is the paper's setting, not a repo convention.
    _dqn("btr", "btr_atari.py", "BTR", "Clark et al. ICML 2025",
         batch=256, update_period=64, pinned_envs=64, use_shipped_cadence=True,
         notes="ships 64 envs, batch 256, one update per vector step = ratio 4"),
    # R2D2 replays *sequences*; its batch is 32 sequences of 40 steps, and the
    # Ape-X epsilon ladder is defined across the environment pool, so the
    # environment count is part of the algorithm rather than a free knob.
    # R2D2's `--replay-ratio` counts sampled *sequences*; one update replays
    # 32 sequences of 40 steps, so its shipped one-update-per-vector-step at 64
    # environments is a transition-level ratio of 32*40/64 = 20.
    _dqn("r2d2", "r2d2_atari.py", "R2D2", "Kapturowski et al. ICLR 2019",
         batch=32, update_period=1.6, pinned_envs=64, use_shipped_cadence=True,
         notes="batch is 32 sequences of seq_len 40; envs are the Ape-X actor pool"),
    # ---- Atari-100K line: built for a small budget and a large replay ratio --
    _dqn("der", "der_atari.py", "DER", "van Hasselt et al. 2019", update_period=1),
    _dqn("drq", "drq_atari.py", "DrQ(eps)", "Kostrikov et al. 2021", update_period=1),
    _dqn("spr", "spr_atari.py", "SPR", "Schwarzer et al. 2021", update_period=0.5),
    _dqn("bbf", "bbf_atari.py", "BBF", "Schwarzer et al. ICML 2023", update_period=0.5),
    _dqn("mrq", "mrq_atari.py", "MR.Q", "Fujimoto et al. ICLR 2025",
         batch=256, update_period=1),
    # ---- On-policy line -----------------------------------------------------
    Algorithm("ppo", "ppo_atari.py", "PPO", ON_POLICY, "Schulman et al. 2017",
              torchcompile="ppo_atari_envpool_torchcompile.py", num_steps=128,
              minibatch_size=256, update_epochs=4, has_cule_device=False,
              has_batch_size=False),
    Algorithm("ppo_lstm", "ppo_atari_lstm.py", "PPO-LSTM", ON_POLICY,
              "Schulman et al. 2017", torchcompile="ppo_atari_lstm_torchcompile.py",
              num_steps=128, minibatch_size=256, update_epochs=4,
              has_cule_device=False, has_benchmark_eager=False, has_batch_size=False),
    Algorithm("ppo_rv", "ppo_rv_atari.py", "PPO+RV", ON_POLICY,
              "Hoeftmann et al. ICLR 2026", torchcompile=None, num_steps=128,
              minibatch_size=128, update_epochs=5, has_batch_size=False),
    Algorithm("dpo", "dpo_atari.py", "DPO", ON_POLICY, "Lu et al. NeurIPS 2022",
              torchcompile=None, num_steps=128, minibatch_size=256, update_epochs=4,
              has_batch_size=False,
              notes="PPO's clip replaced by the distilled Mirror-Learning drift"),
    Algorithm("outer_ppo", "outer_ppo_atari.py", "Outer-PPO", ON_POLICY,
              "Adams et al. 2024", torchcompile=None, num_steps=128,
              minibatch_size=256, update_epochs=4, has_batch_size=False,
              notes="PPO inner loop plus an explicit outer step (lr 1.5)"),
    Algorithm("gpg", "gpg_atari.py", "GPG", ON_POLICY, "Chen et al. 2025",
              torchcompile=None, num_steps=128, minibatch_size=1024, update_epochs=4,
              has_batch_size=False,
              notes="critic-free: group return statistics replace GAE"),
    # IMPALA ships one minibatch and one epoch: V-trace corrects a whole
    # trajectory batch at once.  Holding the minibatch at the shipped
    # 32*80 = 2560 keeps that intact while the environment count moves.
    Algorithm("impala", "impala_atari.py", "IMPALA", ON_POLICY,
              "Espeholt et al. ICML 2018", torchcompile=None, num_steps=80,
              minibatch_size=2560, update_epochs=1, has_batch_size=False,
              notes="V-trace off-policy correction; ships 32 envs x 80 steps"),
    # PPG's auxiliary phase consumes the last `n_pi=32` rollouts in one go, so
    # its buffer is `n_pi * num_envs * num_steps` frames -- 3.7 GB of host RAM at
    # the shipped 32 environments, and linear in the environment count from
    # there.  The buffer length is algorithmic (it sets how much data each
    # auxiliary phase distils), so the design point is kept rather than scaled.
    Algorithm("ppg", "ppg_atari.py", "PPG", ON_POLICY, "Cobbe et al. ICML 2021",
              torchcompile=None, num_steps=128, minibatch_size=512,
              update_epochs=None, pinned_envs=32, has_batch_size=False,
              notes="auxiliary phase every 32 rollouts; buffer scales with envs"),
    # RND's defaults are Montezuma's Revenge settings (gamma 0.999, ent 1e-3,
    # int/ext coefficients 1/2).  They are left alone -- the benchmark runs
    # published defaults -- but they are not tuned for Breakout, and the
    # 50-iteration observation-normalisation phase spends num_steps * 50 vector
    # steps of random play that `global_step` never counts.
    Algorithm("ppo_rnd", "ppo_rnd_atari.py", "PPO+RND", ON_POLICY,
              "Burda et al. ICLR 2019", torchcompile=None, num_steps=128,
              minibatch_size=4096, update_epochs=4, has_batch_size=False,
              notes="Montezuma defaults; obs-norm prefill is uncounted play"),
    Algorithm("ppo_trxl", "ppo_trxl_atari.py", "PPO-TrXL", ON_POLICY,
              "Pleines et al. JMLR 2025", torchcompile=None, num_steps=128,
              minibatch_size=512, update_epochs=3, has_batch_size=False,
              budget_flags=("--anneal-steps",),
              notes="episodic Transformer-XL memory over 119 past steps"),
    # A2C and A2C+SIL have no minibatch knob: the rollout *is* the batch, one
    # update per rollout.  Raising the environment count would therefore scale
    # the batch itself against an unchanged learning rate rather than buy
    # throughput at fixed gradient work, so both keep the standard 16-worker
    # A3C-paper design point -- the same reasoning applied to DiscoRL below.
    Algorithm("a2c", "a2c_atari.py", "A2C", ON_POLICY, "Mnih et al. ICML 2016",
              torchcompile=None, num_steps=5, pinned_envs=16, has_batch_size=False,
              notes="one update per 16x5 rollout; no minibatching to hold fixed"),
    Algorithm("a2c_sil", "a2c_sil_atari.py", "A2C+SIL", ON_POLICY,
              "Oh et al. ICML 2018", torchcompile=None, num_steps=5, pinned_envs=16,
              has_batch_size=False,
              notes="4 self-imitation updates of batch 512 per A2C rollout"),
    Algorithm("pqn", "pqn_atari_envpool.py", "PQN", ON_POLICY, "Gallici et al. 2024",
              torchcompile="pqn_atari_envpool_torchcompile.py", num_steps=128,
              minibatch_size=256, update_epochs=4, has_cule_device=False,
              has_batch_size=False),
    Algorithm("hadamax", "hadamax_pqn_atari_envpool.py", "Hadamax-PQN", ON_POLICY,
              "Kooi et al. NeurIPS 2025",
              torchcompile="hadamax_pqn_atari_envpool_torchcompile.py", num_steps=128,
              minibatch_size=256, update_epochs=4, has_cule_device=False,
              has_batch_size=False),
    # SEM is the deliberate sibling of Hadamax: same PQN parent, other axis
    # (a simplicial embedding on the dense layer instead of a new convolutional
    # stack).  At 64 groups x 8 dims its parameter count matches plain PQN
    # exactly, so PQN, Hadamax and SEM are directly comparable.
    Algorithm("sem_pqn", "sem_pqn_atari_envpool.py", "SEM-PQN", ON_POLICY,
              "Obando-Ceron et al. 2026", torchcompile=None, num_steps=128,
              minibatch_size=256, update_epochs=4, has_cule_device=False,
              has_batch_size=False,
              notes="same parameter count as PQN; the third arm of that comparison"),
    # DiscoRL's meta-network consumes a whole trajectory and has no minibatching
    # knobs, so the trick used for PPO and PQN -- raise --num-minibatches with
    # the rollout batch to hold gradient work per transition constant -- is not
    # available.  Scaling its environment count would therefore change the
    # update itself, so it keeps its shipped 32x29 design point.
    Algorithm("disco", "disco_atari.py", "DiscoRL", ON_POLICY, "Oh et al. Nature 2025",
              torchcompile="disco_atari_torchcompile.py", num_steps=29,
              pinned_envs=32, has_batch_size=False),
    # ---- Streaming line: one update per transition, by definition ------------
    # Streaming RL is defined for a single stream; `--num-envs N` keeps N
    # independent traces that never mix, so the environment count is a
    # parallelism knob that leaves the update rule untouched.
    Algorithm("stream_q", "stream_q_atari.py", "Stream Q(lambda)", STREAMING,
              "Elsayed et al. ICLR 2025", torchcompile="stream_q_atari_torchcompile.py",
              has_batch_size=False),
    Algorithm("stream_ac", "stream_ac_atari.py", "Stream AC(lambda)", STREAMING,
              "Elsayed et al. ICLR 2025", torchcompile="stream_ac_atari_torchcompile.py",
              has_batch_size=False),
]

BY_KEY = {a.key: a for a in ALGORITHMS}

# Excluded, with the reason, so the report states it rather than silently omitting.
EXCLUDED = {
    "qdagger": (
        "qdagger_dqn_atari_impalacnn.py distils a pretrained DQN teacher pulled "
        "from the HuggingFace hub rather than learning Breakout from scratch, so "
        "it is not comparable in a from-scratch benchmark."
    ),
    "dreamerv3": (
        "dreamerv3_atari.py has no CuLE backend -- it needs 64x64 RGB frames "
        "with no frame stacking and no reward clipping, which `make_cule_env` "
        "does not produce, so `--env-backend` accepts only envpool and "
        "gymnasium. Its cadence also puts it out of reach: train_ratio 1024 "
        "with batch 16 x length 64 is one model update per policy step over a "
        "1024-item batch, so 1M frames is 250k world-model updates -- days, not "
        "the ~40 min the rest of the benchmark spends per run."
    ),
}

ENV_ID = "BreakoutNoFrameskip-v4"


if __name__ == "__main__":
    print(f"{len(ALGORITHMS)} algorithms\n")
    header = f"{'key':14s} {'family':11s} {'batch':>6s} {'ratio':>6s} {'shipped':>8s} {'twin'}"
    print(header)
    print("-" * 88)
    for a in ALGORITHMS:
        print(f"{a.key:14s} {a.family:11s} "
              f"{(str(a.batch_size) if a.batch_size else '-'):>6s} "
              f"{(f'{a.paper_replay_ratio:g}' if a.paper_replay_ratio else '-'):>6s} "
              f"{(f'{a.shipped_replay_ratio:g}' if a.shipped_replay_ratio else '-'):>8s} "
              f"{a.torchcompile or '(none)'}")
    for key, why in EXCLUDED.items():
        print(f"\nEXCLUDED {key}: {why}")
