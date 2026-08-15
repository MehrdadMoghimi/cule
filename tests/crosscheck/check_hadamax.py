"""Diff cleanrl/hadamax_pqn_atari_envpool.py against jacobkooi/hadamax.

The reference is JAX/Flax, so the official `QNetwork` is run inside
`third_party/upstream/.venv-jax`; its parameters come back in PyTorch layout and
are loaded into ours, then every block and the final logits are compared.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO_ROOT, Report, load_trainer, parse_devices, upstream

BATCH, N_ACTIONS = 3, 6
# Index of each NCHW-flattened feature within the reference's NHWC flatten.
NHWC_TO_NCHW = np.array(
    [h * 11 * 64 + w * 64 + c for c in range(64) for h in range(11) for w in range(11)])
VENV_PYTHON = os.path.join(REPO_ROOT, "third_party", "upstream", ".venv-jax", "bin", "python")


class EnvStub:
    def __init__(self, n):
        self.single_action_space = type("Discrete", (), {"n": n})()


def run_jax_side(observations, gelu_probe):
    if not os.path.exists(VENV_PYTHON):
        sys.exit(f"JAX venv missing at {VENV_PYTHON}")
    directory = tempfile.mkdtemp(prefix="hadamax-crosscheck-")
    input_path = os.path.join(directory, "inputs.npz")
    output_path = os.path.join(directory, "outputs.npz")
    np.savez(input_path, observations=observations, action_dim=N_ACTIONS, gelu_probe=gelu_probe)
    result = subprocess.run(
        [VENV_PYTHON, os.path.join(os.path.dirname(__file__), "hadamax_jax_runner.py"),
         "--repo", upstream("hadamax"), "--inputs", input_path, "--outputs", output_path],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-5000:])
        sys.exit("the Flax reference failed to run")
    return dict(np.load(output_path))


def run(device, seed, reference, observations, gelu_probe):
    ours = load_trainer("hadamax_pqn_atari_envpool")
    report = Report(f"Hadamax vs jacobkooi/hadamax [{device}]", tolerance=3e-5)

    net = ours.QNetwork(EnvStub(N_ACTIONS)).to(device)

    # nn.gelu in Flax is the tanh approximation; PyTorch's default is exact erf.
    probe = torch.as_tensor(gelu_probe, device=device)
    report.check("gelu (tanh approximation)",
                 F.gelu(probe, approximate="tanh"), reference["gelu_probe"])
    report.note("gelu is NOT the erf form",
                not np.allclose(F.gelu(probe).cpu().numpy(), reference["gelu_probe"], atol=1e-6))

    with torch.no_grad():
        for block in range(3):
            module = net.encoder[block]
            module.conv1.weight.copy_(torch.as_tensor(reference[f"conv{2 * block}_weight"]))
            module.conv1.bias.copy_(torch.as_tensor(reference[f"conv{2 * block}_bias"]))
            module.conv2.weight.copy_(torch.as_tensor(reference[f"conv{2 * block + 1}_weight"]))
            module.conv2.bias.copy_(torch.as_tensor(reference[f"conv{2 * block + 1}_bias"]))
            module.norm1.norm.weight.copy_(torch.as_tensor(reference[f"ln{2 * block}_scale"]))
            module.norm1.norm.bias.copy_(torch.as_tensor(reference[f"ln{2 * block}_bias"]))
            module.norm2.norm.weight.copy_(torch.as_tensor(reference[f"ln{2 * block + 1}_scale"]))
            module.norm2.norm.bias.copy_(torch.as_tensor(reference[f"ln{2 * block + 1}_bias"]))
        # Flax is NHWC and flattens (H, W, C); PyTorch is NCHW and flattens
        # (C, H, W). The projection is fully connected, so the two differ only
        # by a permutation of its input columns -- the same model, relabelled.
        # Reordering here makes the comparison exact.
        net.projection.weight.copy_(torch.as_tensor(reference["dense_weight"])[:, NHWC_TO_NCHW])
        net.projection.bias.copy_(torch.as_tensor(reference["dense_bias"]))
        net.projection_norm.weight.copy_(torch.as_tensor(reference["ln6_scale"]))
        net.projection_norm.bias.copy_(torch.as_tensor(reference["ln6_bias"]))
        net.head.weight.copy_(torch.as_tensor(reference["head_weight"]))
        net.head.bias.copy_(torch.as_tensor(reference["head_bias"]))
    net.eval().requires_grad_(False)

    images = torch.as_tensor(observations, device=device)
    with torch.no_grad():
        stage = images / 255.0
        for block in range(3):
            stage = net.encoder[block](stage)
            report.check(f"Hadamax block {block}", stage, reference[f"block{block}"])
        features = net.encoder[3](stage)
        report.check("flattened features (7744, reordered)",
                     features, reference["features"][:, NHWC_TO_NCHW])
        report.check("Q logits", net(images), reference["logits"])

    report.note("feature width 64 * 11 * 11", net.projection.in_features == 7744)
    report.note("flatten order differs by a permutation only", True,
                "Flax NHWC vs PyTorch NCHW; absorbed by the projection")
    report.note("LayerNorm eps is Flax's 1e-6",
                net.projection_norm.eps == 1e-6 and net.encoder[0].norm1.norm.eps == 1e-6)

    # Initialisers: xavier_normal on the convs, he_normal on the projection.
    source = open(os.path.join(upstream("hadamax"), "purejaxql", "networks.py")).read()
    report.note("conv init = xavier_normal", "xavier_normal" in source)
    report.note("projection init = he_normal", "he_normal" in source)
    return report.print()


def main():
    devices, seed = parse_devices()
    rng = np.random.default_rng(seed)
    observations = rng.integers(0, 256, size=(BATCH, 4, 84, 84)).astype(np.float32)
    gelu_probe = rng.standard_normal(64).astype(np.float32) * 3.0
    print(f"running the Flax reference in {VENV_PYTHON}")
    reference = run_jax_side(observations, gelu_probe)
    results = [run(device, seed, reference, observations, gelu_probe) for device in devices]
    print()
    if all(results):
        print("Hadamax: CONFIRMED against jacobkooi/hadamax on " + ", ".join(devices))
        return 0
    print("Hadamax: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
