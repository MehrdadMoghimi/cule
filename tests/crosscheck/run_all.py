"""Run every upstream cross-check and print a summary.

    python tests/crosscheck/clone_upstreams.py     # once
    python tests/crosscheck/run_all.py             # cpu + cuda

Each check loads the authors' official implementation from
`third_party/upstream/`, transplants weights, and diffs the numerics. See
`third_party/upstream/README.md` for why those checkouts are not committed.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (script, what it needs)
CHECKS = {
    "mrq": ("check_mrq.py", "facebookresearch/MRQ (PyTorch)"),
    "btr": ("check_btr.py", "VIPTankz/BTR (PyTorch)"),
    "stream": ("check_stream.py", "mohmdelsayed/streaming-drl (PyTorch)"),
    "disco": ("check_disco.py", "google-deepmind/disco_rl (JAX/Haiku, needs .venv-jax)"),
    "hadamax": ("check_hadamax.py", "jacobkooi/hadamax (JAX/Flax, needs .venv-jax)"),
    "dopamine": ("check_dopamine.py", "google/dopamine (JAX/Flax, needs .venv-jax)"),
    "rv": ("check_rv.py", "Hauf3n/relative-value-learning (PyTorch)"),
    "dreamer": ("check_dreamer.py", "NM512/dreamerv3-torch (PyTorch)"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="checks to run; default is all")
    parser.add_argument("--device", default="all", choices=["cpu", "cuda", "all"])
    arguments = parser.parse_args()
    names = arguments.names or list(CHECKS)

    results = {}
    for name in names:
        script, description = CHECKS[name]
        print(f"\n{'=' * 78}\n{name}: {description}\n{'=' * 78}")
        completed = subprocess.run(
            [sys.executable, os.path.join(HERE, script), "--device", arguments.device],
            timeout=3600,
        )
        results[name] = completed.returncode == 0

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for name, passed in results.items():
        print(f"  {'CONFIRMED' if passed else 'MISMATCH ':<10} {name:<10} {CHECKS[name][1]}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
