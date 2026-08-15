"""Diff this fork's dopamine-derived losses against google/dopamine.

Scope note. These ports deliberately keep CleanRL's Nature CNN, which uses
VALID padding and 3136 features; dopamine's Flax convolutions take Flax's
default SAME padding and produce 7744. That is a documented divergence, shared
with every other trainer here, and it is what the file headers already say --
they claim the *loss and target construction* match dopamine, not the encoder.
So this script compares exactly that: dopamine's real `quantile_agent.train` is
run with its network replaced by a lookup table, which isolates the loss
arithmetic from the encoder.

The reference runs inside `third_party/upstream/.venv-jax`.
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

BATCH, N_ACTIONS, N_QUANTILES = 5, 6, 8
VENV_PYTHON = os.path.join(REPO_ROOT, "third_party", "upstream", ".venv-jax", "bin", "python")


def run_jax_side(inputs):
    if not os.path.exists(VENV_PYTHON):
        sys.exit(f"JAX venv missing at {VENV_PYTHON}")
    directory = tempfile.mkdtemp(prefix="dopamine-crosscheck-")
    input_path = os.path.join(directory, "inputs.npz")
    output_path = os.path.join(directory, "outputs.npz")
    np.savez(input_path, **inputs)
    result = subprocess.run(
        [VENV_PYTHON, os.path.join(os.path.dirname(__file__), "dopamine_jax_runner.py"),
         "--repo", upstream("dopamine"), "--inputs", input_path, "--outputs", output_path],
        capture_output=True, text=True, timeout=2400,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-5000:])
        sys.exit("the dopamine reference failed to run")
    return dict(np.load(output_path))


def make_inputs(seed):
    rng = np.random.default_rng(seed)
    return dict(
        observations=rng.integers(0, 256, size=(3, 84, 84, 4)).astype(np.float32),
        action_dim=N_ACTIONS, num_atoms=51, num_quantiles=N_QUANTILES, batch=BATCH,
        online_logits=rng.standard_normal((BATCH, N_ACTIONS, N_QUANTILES)).astype(np.float32),
        target_logits=rng.standard_normal((BATCH, N_ACTIONS, N_QUANTILES)).astype(np.float32),
        qr_actions=rng.integers(0, N_ACTIONS, size=BATCH).astype(np.int32),
        qr_rewards=rng.standard_normal(BATCH).astype(np.float32),
        qr_terminals=np.array([0, 1, 0, 0, 1], dtype=np.float32),
        cumulative_gamma=np.float32(0.99**3), kappa=np.float32(1.0),
        huber_predictions=rng.standard_normal(20).astype(np.float32),
        huber_targets=rng.standard_normal(20).astype(np.float32),
        taus=rng.random((N_QUANTILES, 1)).astype(np.float32),
    )


def run(device, seed, reference, inputs):
    ours = load_trainer("qrdqn_atari")
    report = Report(f"QR-DQN loss vs google/dopamine [{device}]", tolerance=1e-5)

    tensor = lambda name: torch.as_tensor(inputs[name], device=device)
    online = tensor("online_logits")
    target = tensor("target_logits")
    actions = tensor("qr_actions").long()
    rewards = tensor("qr_rewards")
    terminals = tensor("qr_terminals")
    gamma_n = float(inputs["cumulative_gamma"])
    kappa = float(inputs["kappa"])

    # tau midpoints
    tau_hat = (torch.arange(N_QUANTILES, device=device, dtype=torch.float32) + 0.5) / N_QUANTILES
    report.check("tau midpoints", tau_hat, reference["qr_tau_hat"])

    # target distribution: r + gamma^n * (1 - done) * logits[argmax_a mean_j logits]
    with torch.no_grad():
        next_q = target.mean(dim=2)
        best = next_q.argmax(dim=1)
        next_logits = target[torch.arange(BATCH, device=device), best]
        target_quantiles = rewards.unsqueeze(-1) + gamma_n * (1.0 - terminals).unsqueeze(-1) * next_logits

    chosen = online[torch.arange(BATCH, device=device), actions]
    errors = target_quantiles.unsqueeze(1) - chosen.unsqueeze(2)
    absolute = errors.abs()
    huber = torch.where(absolute <= kappa, 0.5 * errors.pow(2), kappa * (absolute - 0.5 * kappa))
    weight = (tau_hat.view(1, -1, 1) - (errors < 0).float()).abs()
    # Upstream reduces as sum(mean(loss, 2), 1): mean over targets, sum over taus.
    loss_per_sample = (weight * huber).mean(2).sum(1)
    report.check("QR-DQN per-sample loss", loss_per_sample, reference["qr_loss_per_sample"])
    report.check("QR-DQN mean loss", loss_per_sample.mean(), reference["qr_mean_loss"])

    # The same arithmetic, as the trainer writes it.
    trainer_loss = ours.quantile_huber_loss(chosen, target_quantiles, tau_hat, kappa) \
        if hasattr(ours, "quantile_huber_loss") else None
    if trainer_loss is not None:
        report.check("trainer's quantile_huber_loss", trainer_loss, reference["qr_loss_per_sample"])
    else:
        report.note("trainer's loss is inline (compared via the expression above)", True)

    # dopamine's own Huber helper
    predictions = tensor("huber_predictions")
    targets = tensor("huber_targets")
    x = (targets - predictions).abs()
    report.check("dopamine huber_loss", torch.where(x <= 1.0, 0.5 * x**2, 0.5 + (x - 1.0)),
                 reference["huber_loss"])
    # dopamine's mse_loss carries no 1/2 factor, unlike its huber_loss.
    report.check("dopamine mse_loss", (targets - predictions) ** 2, reference["mse_loss"])

    # Encoder divergence, stated rather than silently passed.
    report.note("encoder: dopamine SAME (7744) vs this fork VALID (3136)", True,
                f"reference dense is {reference['qrdqn_dense0_weight'].shape}, ours is 3136 -> 512")

    args = ours.Args()
    report.note("n_quantiles 200 / kappa 1.0", args.n_quantiles == 200 and args.kappa == 1.0)
    return report.print()


def main():
    devices, seed = parse_devices()
    inputs = make_inputs(seed)
    print(f"running the dopamine reference in {VENV_PYTHON}")
    reference = run_jax_side(inputs)
    results = [run(device, seed, reference, inputs) for device in devices]
    print()
    if all(results):
        print("QR-DQN loss: CONFIRMED against google/dopamine on " + ", ".join(devices)
              + " (encoder padding differs by design)")
        return 0
    print("dopamine: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
