# Working in this repo

## Background jobs: never sequence them with `pgrep`/`pkill` on a name

This repo's work is long-running GPU jobs, so it is natural to want a shell that
waits for one job before starting the next. Doing that by pattern-matching
process names has failed twice here, both times silently, and both times for the
same reason: **`pgrep -f` and `pkill -f` match the full command line of every
process, including the shell that is running the match itself.**

### What actually happened

**A watcher that waited on itself.** This shell was launched to wait for an
experiment to finish:

```bash
until ! pgrep -f "exp_h013_" >/dev/null; do sleep 60; done   # never exits
```

The shell's own command line contains `exp_h013_`, so `pgrep` matched the
watcher, the watcher waited for itself, and it ran until killed. The experiment
had finished hours earlier. Nothing reported this — it looks identical to a job
still running.

**A sequencer that waited on the watchers.** A queue script used
`while pgrep -f "exp_h0[0-9][0-9]_"; do sleep 30; done` to decide the GPU was
free. That pattern matched the three wedged watchers above. Result: ~6 hours of
idle GPU with the queue apparently "running", and eight queued experiments that
never started.

**A `pkill` that killed the caller.** `pkill -f "chain6.sh"` was run from a shell
whose own command line contained `chain6.sh`. It killed itself mid-command;
everything after it in the compound command silently never ran (exit code 144).

### Rules

1. **Sequence by execution order, not by polling.** One script that runs jobs
   back to back cannot deadlock:
   ```bash
   run exp_a; run exp_b; run exp_c     # b starts when a returns. That is all.
   ```
   This is what `scratchpad/queue.sh` does and it is the only pattern that has
   worked here.
2. **To wait on a job you started, use the harness.** `Bash` with
   `run_in_background: true` notifies on exit. Do not hand-roll a poll loop.
3. **If you must wait on a condition, wait on a file, not a process.** A
   completion marker appended to a log (`grep -q "H014 done" chain.log`) cannot
   match the grepping shell.
4. **Never `pkill -f`/`pgrep -f` a string that appears in your own command.** If
   a process must be killed, resolve the PID first and `kill "$pid"`. Build the
   pattern so it cannot self-match (`PAT="chain""6.sh"`) only as a last resort —
   it is easy to get wrong and reads as a trick.
5. **Check liveness by effect, not by name.** Whether work is progressing is
   answered by a log file growing or a JSONL gaining lines, not by a process
   existing. Sample a file's size twice, and allow for quiet gaps: a trainer that
   flushes on episode boundaries can go 30+ seconds without printing while
   running perfectly.

### Before believing a job is stuck

The same care applies in reverse — three separate checks here suggested a hang
that was not one. `stat -c%s` the log twice over 90 seconds, and look at
`ps -o etimes=` for the actual worker process, before concluding anything.

## Experiment scoring: wall clock vs transitions

Two scoring modes exist in `benchmarks/exp_runner.py` and the choice determines
whether a job may share the GPU:

- **`score_by_transitions`** grades the closing fraction of the transition axis.
  A run slowed by a competing job gets the same score, so these experiments may
  run concurrently. Their recorded `train_sps` is still contaminated and must not
  be used for wall-clock projections.
- **`score_by_seconds`** reproduces `sweep_highenv.py`'s rule — mean return over
  the closing 40% of a fixed window. Valid **only on an idle GPU**, because a
  slower run genuinely scores worse.

Check for other GPU work before starting a wall-clock-scored job; other sessions
run jobs here too, and `sweep_highenv.py`'s lock only covers its own results
directory.
