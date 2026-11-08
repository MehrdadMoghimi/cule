"""Train a trainer on the closed-form MDP and compare its Q against `Q*`.

    python benchmarks/verify_chain.py dqn_torchcompile
    python benchmarks/verify_chain.py --list
    python benchmarks/verify_chain.py c51_atari_torchcompile --gamma 0.9 --seed 3

`cleanrl/diagnostic_env.py` defines a chain MDP whose optimal action-value
function is exact arithmetic, and routes to it from `make_cule_env` whenever the
env id names one, so a trainer runs on it with no edits and no new flag.  This
script drives that: it runs the trainer to completion in-process, reads the
trained network out of the module namespace afterwards, evaluates it on the
observations the optimal policy actually visits, and prints the learned numbers
next to the ones the MDP requires.

Running the trainer in-process (`runpy`) rather than as a subprocess is what
makes the comparison possible without touching 62 files: the trainers do their
work at module scope under `if __name__ == "__main__"`, so `run_path` hands back
a namespace in which `q_network` is the trained network.

What a failure means
--------------------
The report scores the learned values against two hypotheses: `Q*`, and the
fixed point of the same algorithm with the terminal bootstrap left unmasked.
Those two differ by a constant factor and share a greedy policy, so a score
curve cannot tell them apart and this table can.  A run that matches neither,
but whose greedy action is right everywhere, is usually still-converging rather
than wrong -- raise `--total-timesteps` before believing it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import contextlib
import dataclasses
import importlib.util
import io
import os
import re
import runpy
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANRL_DIR = os.path.join(REPO_ROOT, "cleanrl")
for _path in (CLEANRL_DIR, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from chain_adapters import (  # noqa: E402  isort:skip
    ADAPTERS,
    STATUS,
    UNSUPPORTED,
    Prediction,
    Unsupported,
    adapter_for,
)
from diagnostic_env import (  # noqa: E402  isort:skip
    ADVANCE,
    canonical_observations,
    optimal_q,
    optimal_return_distribution,
    optimal_v,
    parse_diagnostic_env_id,
    q_without_terminal_mask,
)


# ---------------------------------------------------------------------------
# running the trainer
# ---------------------------------------------------------------------------

def trainer_args(trainer: str) -> set[str]:
    """Field names of the trainer's `Args`, so unsupported flags are dropped.

    Importing the module runs only its definitions -- the training loop is
    behind `if __name__ == "__main__"` -- which is exactly what is wanted here.
    """
    spec = importlib.util.spec_from_file_location(
        f"_args_{trainer}", os.path.join(CLEANRL_DIR, f"{trainer}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {field.name for field in dataclasses.fields(module.Args)}


def build_argv(trainer: str, arguments) -> list[str]:
    """Hyperparameters for a short, easy run, filtered to the flags that exist.

    These are deliberately not the shipped Atari settings.  The question here is
    whether the algorithm converges to the right numbers when learning is easy,
    so the run is small, the discount is well away from 1 (which spreads `Q*`
    across a wide range instead of bunching every state near the same value),
    and exploration stays on long enough to keep reaching the goal.
    """
    wanted = {
        "env_id": arguments.env_id,
        "env_backend": "cule",
        "cule_device": arguments.device,
        "num_envs": arguments.num_envs,
        "total_timesteps": BUDGET.get(trainer, arguments.total_timesteps),
        "gamma": arguments.gamma,
        "seed": arguments.seed,
        "learning_rate": arguments.learning_rate,
        "buffer_size": 100_000,
        "batch_size": 256,
        "learning_starts": 2_000,
        "target_network_frequency": 100,
        "start_e": 1.0,
        "end_e": 0.1,
        "exploration_fraction": 0.3,
        "num_steps": 32,
        "compile": False,
        "cudagraphs": False,
        "track": False,
        # Never let an evaluation fire mid-run; the comparison is the evaluation.
        "evaluation_interval": BUDGET.get(trainer, arguments.total_timesteps) * 10,
        "num_minibatches": 4,
    }
    available = trainer_args(trainer)
    argv = [os.path.join(CLEANRL_DIR, f"{trainer}.py")]
    for key, value in wanted.items():
        if key not in available:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            argv.append(flag)
            argv.append(str(value))
    return argv


# Per-family budgets.  The chain is trivial, so what differs between families is
# not difficulty but how many gradient steps a given number of transitions buys.
#
#   replay-heavy (Atari-100K, R2D2, MR.Q, DisCo): many updates per transition,
#     and sample-efficient by construction, so a short run converges here.
#   on-policy: one or a few updates per rollout.  A2C at 32 envs and 32 steps
#     gets ~195 updates out of 200k transitions, nowhere near enough to fit a
#     value function -- and a FAIL caused by under-training would be a false
#     alarm, the one failure mode that would make this harness untrustworthy.
#     They get a budget that buys a comparable number of updates instead.
#   everything else: the off-policy value-based default.
REPLAY_HEAVY = (
    "bbf_atari", "bbf_atari_torchcompile", "spr_atari", "spr_atari_torchcompile",
    "der_atari", "der_atari_torchcompile", "drq_atari", "drq_atari_torchcompile",
    "btr_atari", "btr_atari_torchcompile", "r2d2_atari", "r2d2_atari_torchcompile",
    "mrq_atari", "mrq_atari_torchcompile", "disco_atari", "disco_atari_torchcompile",
    "rainbow_atari", "rainbow_atari_torchcompile",
    "qdagger_dqn_atari_impalacnn", "qdagger_dqn_atari_impalacnn_torchcompile",
)
ON_POLICY = (
    "a2c_atari", "a2c_sil_atari", "dpo_atari", "gpg_atari", "impala_atari",
    "outer_ppo_atari", "ppg_atari", "ppo_atari", "ppo_atari_envpool_torchcompile",
    "ppo_atari_lstm", "ppo_atari_lstm_torchcompile", "ppo_rnd_atari",
    "ppo_rv_atari", "ppo_scale_atari", "ppo_trxl_atari",
    "pqn_atari_envpool", "pqn_atari_envpool_torchcompile",
    "hadamax_pqn_atari_envpool", "hadamax_pqn_atari_envpool_torchcompile",
    "sem_pqn_atari_envpool", "stream_ac_atari", "stream_ac_atari_torchcompile",
)
BUDGET = {trainer: 60_000 for trainer in REPLAY_HEAVY}
BUDGET.update({trainer: 2_000_000 for trainer in ON_POLICY})


# Source-level mutations, used to check that the harness detects anything at all.
# Each entry is a pattern that must match exactly once in the trainer's source.
BUG_INJECTIONS = {
    # Drop the (1 - done) factor from the TD target, so terminal transitions
    # bootstrap into the reset state.
    "no-terminal-mask": (
        re.compile(r" \* \(~data\[.dones.\]\)\.float\(\)"),
        "",
    ),
    # Discount the reward as well as the bootstrap, the classic off-by-one.
    "discount-the-reward": (
        re.compile(r"td_target = data\[.rewards.\] \+ args\.gamma"),
        'td_target = args.gamma * data["rewards"] + args.gamma',
    ),
}


def inject_bug(trainer: str, name: str) -> str:
    """Write a mutated copy of the trainer and return its path.

    The copy stays out of `cleanrl/`; `sys.path` already contains that directory,
    so the trainer's own imports still resolve from wherever the copy runs.
    """
    if name not in BUG_INJECTIONS:
        raise SystemExit(f"unknown injection {name!r}; known: {sorted(BUG_INJECTIONS)}")
    pattern, replacement = BUG_INJECTIONS[name]
    source = open(os.path.join(CLEANRL_DIR, f"{trainer}.py")).read()
    mutated, count = pattern.subn(replacement, source)
    if count != 1:
        raise SystemExit(
            f"injection {name!r} matched {count} times in {trainer}.py, expected exactly 1"
        )
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix=f"{trainer}__{name.replace('-', '_')}__", delete=False)
    handle.write(mutated)
    handle.close()
    return handle.name


def run_trainer(trainer: str, argv: list[str], quiet: bool) -> dict:
    """Execute the trainer as `__main__` and return its finished namespace."""
    script = argv[0]
    saved_argv = sys.argv
    sys.argv = argv
    try:
        with contextlib.ExitStack() as stack:
            if quiet:
                # tqdm writes to stderr, so both streams have to go.
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            return runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

# The headline number of the last report, picked up by the JSON writer.
LAST_ERROR = [None]


def _table(header, rows) -> str:
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(header))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def report(trainer, spec, gamma, prediction, tolerance) -> bool:
    star_q = optimal_q(spec, gamma)
    star_v = optimal_v(spec, gamma)
    unmasked_q = q_without_terminal_mask(spec, gamma)
    # Starts False: a Prediction that carries nothing must never report a pass.
    passed = False

    print(f"\n{'=' * 78}\n{trainer} on {spec}\ngamma={gamma}  tolerance={tolerance}\n{'=' * 78}")

    if prediction.q is not None:
        learned = prediction.q.detach().double().cpu().numpy()
        rows = []
        for state in range(spec.n_states):
            rows.append([
                state,
                f"{learned[state, ADVANCE]:+.4f}",
                f"{star_q[state, ADVANCE]:+.4f}",
                f"{learned[state, 1:].mean():+.4f}",
                f"{star_q[state, 1]:+.4f}",
                "ok" if learned[state].argmax() == ADVANCE else "WRONG",
            ])
        print(_table(
            ["state", "Q(ADVANCE)", "Q*(ADVANCE)", "Q(STAY)", "Q*(STAY)", "argmax"], rows))

        error = np.abs(learned - star_q).max()
        greedy_ok = bool((learned.argmax(axis=1) == ADVANCE).all())
        unmasked_error = np.abs(learned - unmasked_q).max()

        # The missing-mask fixed point is Q* times a constant, so the useful
        # question is not "how far from Q*" but "how far from a multiple of Q*".
        # Fitting that multiple reads the signature while the values are still
        # inflating towards it, which is where a short run usually stops.
        scale = float((learned * star_q).sum() / (star_q * star_q).sum())
        shape_error = np.abs(learned - scale * star_q).max()
        inflation = 1.0 / (1.0 - gamma**spec.n_states)

        print(f"\nmax |Q - Q*|                     {error:.5f}   "
              f"{'PASS' if error <= tolerance else 'FAIL'}")
        print(f"greedy action correct everywhere {greedy_ok}")
        print(f"best-fit scale c in Q ~ c*Q*      {scale:.4f}   "
              f"(correct: 1.0000, unmasked bootstrap: {inflation:.4f})")
        print(f"max |Q - c*Q*| after scaling      {shape_error:.5f}")
        print(f"max |Q - Q_no_terminal_mask|     {unmasked_error:.5f}")
        if error > tolerance and scale > 1.0 + tolerance and shape_error < error:
            print(f"\n  -> the values have the right shape but are inflated by {scale:.2f}x.\n"
                  f"     That is the signature of a terminal bootstrap that is never masked:\n"
                  f"     its fixed point is exactly Q* * 1/(1 - gamma**n) = {inflation:.4f}, and a\n"
                  f"     run still climbing towards it lands between 1 and that value.\n"
                  f"     Note the greedy policy is unaffected, so no score curve shows this.")
        passed = error <= tolerance and greedy_ok
        LAST_ERROR[0] = float(error)

    if prediction.v is not None:
        learned_v = prediction.v.detach().double().cpu().numpy()
        rows = [[state, f"{learned_v[state]:+.4f}", f"{star_v[state]:+.4f}"]
                for state in range(spec.n_states)]
        print(_table(["state", "V(s)", "V*(s)"], rows))
        error = np.abs(learned_v - star_v).max()
        print(f"\nmax |V - V*|  {error:.5f}   {'PASS' if error <= tolerance else 'FAIL'}")
        passed = error <= tolerance
        LAST_ERROR[0] = float(error)

    # The distributional heads predict the whole return distribution, and on
    # this MDP that distribution is also closed form: a single atom when the
    # terminal reward is deterministic, two atoms otherwise.
    if prediction.quantiles is not None:
        quantiles = prediction.quantiles.detach().double().cpu().numpy()[:, ADVANCE]
        atoms, weights = optimal_return_distribution(spec, gamma, state=0)
        spread = quantiles.std(axis=1)
        print(f"\nquantile spread at the optimal action, per state: "
              f"{np.array2string(spread, precision=4)}")
        if spec.is_deterministic:
            print(f"optimal spread is 0 in every state (the return is deterministic); "
                  f"max seen {spread.max():.4f}")
        else:
            below = (quantiles[0] < 0).mean()
            print(f"state 0: atoms {atoms.round(4)} with weights {weights.round(4)}; "
                  f"learned fraction of quantiles below zero = {below:.3f} "
                  f"(target {weights[1]:.3f})")

    if prediction.probs is not None:
        probs = prediction.probs.detach().double().cpu().numpy()[0, ADVANCE]
        atoms = prediction.atoms.detach().double().cpu().numpy()
        mass = probs.argsort()[-3:][::-1]
        print("\nstate 0, top categorical atoms: "
              + ", ".join(f"{atoms[i]:+.3f} (p={probs[i]:.3f})" for i in mass))
        print(f"optimal return from state 0 is a point mass at "
              f"{optimal_return_distribution(spec, gamma, 0)[0][0]:+.4f}")

    if prediction.greedy is not None:
        greedy = prediction.greedy.detach().cpu().numpy()
        rows = [[state, int(greedy[state]), 'ok' if greedy[state] == ADVANCE else 'WRONG']
                for state in range(spec.n_states)]
        print(_table(["state", "greedy action", "optimal is ADVANCE"], rows))
        passed = bool((greedy == ADVANCE).all())
        print(f"\ngreedy action correct everywhere {passed}")

    if prediction.differences is not None:
        # Relative-value learning never predicts V, only V(i) - V(j), so the
        # closed form is compared as the full matrix of pairwise differences.
        learned = prediction.differences.detach().double().cpu().numpy()
        target = star_v[:, None] - star_v[None, :]
        error = np.abs(learned - target).max()
        rows = [[state,
                 f"{learned[state, 0]:+.4f}", f"{target[state, 0]:+.4f}"]
                for state in range(spec.n_states)]
        print(_table(["state i", "V(i)-V(0)", "V*(i)-V*(0)"], rows))
        print(f"\nmax |RV - (V* - V*)| over all pairs  {error:.5f}   "
              f"{'PASS' if error <= tolerance else 'FAIL'}")
        passed = error <= tolerance

    if prediction.note:
        print(f"\nnote: {prediction.note}")

    print(f"\nRESULT  {trainer}: {'PASS' if passed else 'FAIL'}")
    return passed


def run_one(trainer, arguments):
    """Run and score a single trainer in this process."""
    spec = parse_diagnostic_env_id(arguments.env_id)
    adapter = adapter_for(trainer)          # raises Unsupported for the two below
    argv = build_argv(trainer, arguments)
    if arguments.inject_bug:
        argv[0] = inject_bug(trainer, arguments.inject_bug)
        print(f"injected {arguments.inject_bug!r} into {trainer}: {argv[0]}")
    namespace = run_trainer(trainer, argv, arguments.quiet)
    device = namespace.get("device", torch.device("cpu"))
    frame_stack = namespace["envs"].frame_stack
    observations = canonical_observations(spec, frame_stack, device).float()
    with torch.no_grad():
        prediction = adapter(namespace, observations, spec)
    return report(trainer, spec, arguments.gamma, prediction, arguments.tolerance)


def sweep(arguments):
    """Run every trainer, one subprocess each, and print a coverage table.

    Subprocesses rather than a loop in this process: sixty-two trainers share
    module names, CUDA state, compile caches and global torch settings, and one
    that hangs or aborts must not take the sweep with it.
    """
    trainers = sorted(
        name[:-3] for name in os.listdir(CLEANRL_DIR)
        if name.endswith(".py") and name[:-3] not in ("cule_env", "diagnostic_env")
    )
    results = {}
    for index, trainer in enumerate(trainers, start=1):
        status = STATUS.get(trainer, "value")
        print(f"[{index}/{len(trainers)}] {trainer} ({status})", flush=True)
        if status == "unsupported":
            results[trainer] = {"status": "unsupported", "reason": UNSUPPORTED[trainer]}
            print(f"    skipped: {UNSUPPORTED[trainer]}", flush=True)
            continue
        record = os.path.join(arguments.output_dir, f"{trainer}.json")
        command = [
            sys.executable, os.path.abspath(__file__), trainer,
            "--env-id", arguments.env_id, "--gamma", str(arguments.gamma),
            "--seed", str(arguments.seed), "--num-envs", str(arguments.num_envs),
            "--total-timesteps", str(arguments.total_timesteps),
            "--learning-rate", str(arguments.learning_rate),
            "--tolerance", str(arguments.tolerance), "--device", arguments.device,
            "--json", record, "--quiet",
        ]
        try:
            completed = subprocess.run(command, timeout=arguments.timeout,
                                       capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            results[trainer] = {"status": "timeout",
                                "reason": f"exceeded {arguments.timeout}s"}
            print(f"    TIMEOUT after {arguments.timeout}s", flush=True)
            continue
        if os.path.exists(record):
            with open(record) as handle:
                results[trainer] = json.load(handle)
        else:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            results[trainer] = {
                "status": "error",
                "reason": tail[-1][:200] if tail else f"exit {completed.returncode}",
            }
        print(f"    {results[trainer].get('status')}"
              f" {results[trainer].get('error', '')}", flush=True)

    summary = os.path.join(arguments.output_dir, "summary.json")
    with open(summary, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    print(f"\n{'=' * 78}\nCOVERAGE\n{'=' * 78}")
    buckets = {}
    for trainer, record in sorted(results.items()):
        buckets.setdefault(record["status"], []).append((trainer, record))
    for status in ("pass", "fail", "policy-pass", "policy-fail",
                   "unsupported", "timeout", "error"):
        rows = buckets.get(status, [])
        if not rows:
            continue
        print(f"\n{status.upper()}  ({len(rows)})")
        for trainer, record in rows:
            detail = record.get("error")
            detail = f"  max|Q-Q*| {detail:.5f}" if isinstance(detail, float) else \
                f"  {record.get('reason', '')}"
            print(f"  {trainer:44s}{detail}")
    print(f"\nwrote {summary}")
    return 0 if not (buckets.get("fail") or buckets.get("error")
                     or buckets.get("policy-fail") or buckets.get("timeout")) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainers", nargs="*", help="trainer module names, e.g. dqn_torchcompile")
    parser.add_argument("--list", action="store_true", help="list the trainers with an adapter")
    parser.add_argument("--all", action="store_true",
                        help="run every trainer, one subprocess each, and print a coverage table")
    parser.add_argument("--env-id", default="Chain-v0")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--device", default="auto", help="value for --cule-device")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="max allowed |Q - Q*|; Q* spans about 0.66 to 1.0 by default")
    parser.add_argument("--quiet", action="store_true", help="suppress the trainer's own output")
    parser.add_argument("--json", default=None, help="write this run's verdict to a JSON file")
    parser.add_argument("--timeout", type=int, default=1800, help="per-trainer timeout in --all")
    parser.add_argument("--output-dir",
                        default=os.path.join(REPO_ROOT, "benchmark_results", "chain"),
                        help="where --all writes per-trainer JSON and the summary")
    parser.add_argument("--inject-bug", default=None, choices=sorted(BUG_INJECTIONS),
                        help="mutate the trainer's source first, to confirm the check has teeth")
    arguments = parser.parse_args()

    if arguments.list:
        for name in sorted(ADAPTERS):
            print(f"{name:42s} {ADAPTERS[name].__name__}")
        return 0
    if arguments.all:
        os.makedirs(arguments.output_dir, exist_ok=True)
        return sweep(arguments)
    if not arguments.trainers:
        parser.error("name at least one trainer, or pass --all or --list")

    results = {}
    for trainer in arguments.trainers:
        kind = STATUS.get(trainer, "value")
        try:
            passed = run_one(trainer, arguments)
            status = ("pass" if passed else "fail") if kind != "policy" else \
                ("policy-pass" if passed else "policy-fail")
        except Unsupported as failure:
            passed, status = False, "unsupported"
            print(f"\n{trainer}: unsupported -- {failure}")
        results[trainer] = passed
        if arguments.json:
            with open(arguments.json, "w") as handle:
                json.dump({"trainer": trainer, "status": status,
                           "error": LAST_ERROR[0], "gamma": arguments.gamma,
                           "env_id": arguments.env_id}, handle)

    if len(results) > 1:
        print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
        for trainer, passed in results.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {trainer}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
