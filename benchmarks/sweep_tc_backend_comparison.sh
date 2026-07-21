#!/bin/bash
# Sweep: torchcompile variants of the new algorithms, envpool vs cule.
# Fixed learner workload per algorithm (updates/vector-step and batch size)
# so the backend comparison isolates collection scaling; capture warmup is
# long enough that CudaGraphModule engages at every point.
cd /home/mehrdad96/cule
OUT=/tmp/claude-1000/-home-mehrdad96-cule/e3c1b74b-ad69-4376-8732-84eddfb0b377/scratchpad/sweep_tc_backends.jsonl
: > "$OUT"

run_one () {
  local algo=$1 backend=$2 env_id=$3 envs=$4 batch=$5 ups=$6 warmup=$7
  local line
  line=$(conda run -n cule312 python cleanrl/${algo}_atari_torchcompile.py \
    --env-backend "$backend" --env-id "$env_id" \
    --num-envs "$envs" --batch-size "$batch" \
    --learner-updates-per-vector-step "$ups" --replay-ratio None \
    --learning-starts 512 --buffer-size 100000 \
    --compile --cudagraphs \
    --benchmark --benchmark-warmup-iterations "$warmup" \
    --benchmark-measure-iterations 30 2>&1 | grep -o 'BENCHMARK_RESULT .*' | head -1)
  if [ -n "$line" ]; then
    echo "${line#BENCHMARK_RESULT }" >> "$OUT"
    echo "OK   $algo $backend $envs"
  else
    echo "FAIL $algo $backend $envs"
  fi
}

for envs in 32 64 128 256; do
  for spec in \
      "qrdqn 512 1 45" "iqn 512 1 45" "fqf 512 1 45" "miqn 512 1 45" \
      "der 32 1 45" "drq 32 1 45" "spr 32 2 45" "bbf 32 2 45"; do
    set -- $spec
    algo=$1; batch=$2; ups=$3; warmup=$4
    run_one "$algo" cule    PongNoFrameskip-v4 "$envs" "$batch" "$ups" "$warmup"
    run_one "$algo" envpool Pong-v5            "$envs" "$batch" "$ups" "$warmup"
  done
done
echo SWEEP_DONE
