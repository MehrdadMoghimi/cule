#!/usr/bin/env python3
"""H020: does the CuLE device-side assert reproduce, and is cudagraphs implicated?

F023 records a crash: rainbow_atari_torchcompile.py at 64 environments, batch 32,
replay ratio 8 on Breakout, seed 1, died after ~550,000 transitions with

    GPUassert: device-side assert triggered
    /home/mehrdad96/cule/cule/cuda/parallel_execution_policy.hpp 27

surfaced through tensordict's CudaGraphModule at its torch.cuda.synchronize.
The eager file at identical settings ran to completion.  Two things are unknown
and both matter before it is called a bug:

  * whether it reproduces at all, or was a one-off under GPU contention -- the
    failing run shared the card with another job, and an assert inside CuLE's
    parallel execution policy is exactly the shape a contention race would take;
  * whether the fault needs cudagraphs, or only torch.compile.  The trainer
    passes both flags together, so the crash report cannot separate them.

Four cells: seeds 1-3 at the exact failing configuration, plus seed 1 with
compile but no cudagraphs.  A clean card, one run at a time.

  all four crash                  -> a compile-path bug, cudagraphs not required
  three crash, no-cudagraphs runs -> cudagraphs (or the graph-capture region CuLE
                                     is stepped inside) is implicated
  none crash                      -> contention artifact; F023 should be withdrawn
"""

from __future__ import annotations

import sys, os, subprocess, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from breakout_algorithms import ROOT
from sweep_highenv import PYTHON

OUT = ROOT / "benchmark_results" / "exploration" / "h020_rainbow_assert"
BASE = [PYTHON, str(ROOT / "cleanrl" / "rainbow_atari_torchcompile.py"),
        "--env-backend", "cule", "--env-id", "BreakoutNoFrameskip-v4",
        "--num-envs", "64", "--total-timesteps", "1000000",
        "--cule-device", "cuda", "--batch-size", "32",
        "--replay-ratio", "8.0", "--buffer-size", "131072"]
CELLS = [("seed1-compile-cudagraphs", ["--seed", "1", "--compile", "--cudagraphs"]),
         ("seed2-compile-cudagraphs", ["--seed", "2", "--compile", "--cudagraphs"]),
         ("seed3-compile-cudagraphs", ["--seed", "3", "--compile", "--cudagraphs"]),
         ("seed1-compile-only",       ["--seed", "1", "--compile"])]
ASSERT_MARK = "device-side assert"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, extra in CELLS:
        log = OUT / f"{name}.log"
        started = time.perf_counter()
        with log.open("wb") as stream:
            proc = subprocess.run(BASE + extra, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=9000)
        text = log.read_text(errors="replace")
        steps = [int(s) for s in __import__("re").findall(r"global_step=(\d+)", text)]
        record = {
            "name": name, "returncode": proc.returncode,
            "wall_seconds": round(time.perf_counter() - started, 1),
            "reached": max(steps) if steps else 0,
            "asserted": ASSERT_MARK in text,
            "flags": extra,
        }
        records.append(record)
        print(f"  {name:26s} rc={record['returncode']:4d}  reached "
              f"{record['reached']/1e6:5.2f}M  assert={record['asserted']}  "
              f"{record['wall_seconds']:6.0f}s", flush=True)
    (OUT / "runs.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    crashed = [r for r in records if r["asserted"]]
    graphs = [r for r in records if r["asserted"] and "--cudagraphs" in r["flags"]]
    print(f"\n{len(crashed)}/{len(records)} hit the assert; "
          f"{len(graphs)} of those used cudagraphs")
    if not crashed:
        print("  -> did not reproduce on a clean card; F023 is a contention artifact")
    elif len(crashed) == 3 and not any("--cudagraphs" not in r["flags"] for r in crashed):
        print("  -> reproduces only with cudagraphs")
    else:
        print("  -> reproduces on the compile path regardless of cudagraphs")


if __name__ == "__main__":
    main()
