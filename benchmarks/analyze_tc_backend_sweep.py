"""Summarize the torchcompile envpool-vs-cule sweep into a markdown table."""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
rows = [json.loads(line) for line in open(path)]
data = defaultdict(dict)  # (algo, envs) -> {backend: record}
for r in rows:
    data[(r["algorithm"], r["num_envs"])][r["backend"]] = r

algos = ["qrdqn", "iqn", "fqf", "miqn", "der", "drq", "spr", "bbf"]
env_counts = sorted({k[1] for k in data})

print("| Algorithm | Envs | CuLE SPS | EnvPool SPS | EnvPool/CuLE |")
print("|---|---:|---:|---:|---:|")
for algo in algos:
    for envs in env_counts:
        rec = data.get((algo, envs), {})
        c = rec.get("cule", {}).get("sps")
        e = rec.get("envpool", {}).get("sps")
        ratio = f"{e / c:.2f}x" if c and e else "-"
        fmt = lambda v: f"{v:,.0f}" if v else "FAIL"
        print(f"| {algo} | {envs} | {fmt(c)} | {fmt(e)} | {ratio} |")

print()
print("Peak CUDA memory (MB) at 256 envs:")
for algo in algos:
    rec = data.get((algo, 256), {})
    c = rec.get("cule", {}).get("peak_cuda_memory_mb")
    e = rec.get("envpool", {}).get("peak_cuda_memory_mb")
    if c or e:
        print(f"  {algo}: cule {c:,.0f} / envpool {e:,.0f}")
