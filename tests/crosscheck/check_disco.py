"""Diff cleanrl/disco_atari.py against google-deepmind/disco_rl.

The reference is JAX/Haiku, so the two halves cannot share a process. This
script writes random inputs, shells out to `disco_jax_runner.py` inside
`third_party/upstream/.venv-jax` to run the authors' real `meta_nets.LSTM` with
the published `disco_103.npz` loaded, and diffs the results against our PyTorch
port with the same weights.

Set up the venv once:

    python -m venv third_party/upstream/.venv-jax
    third_party/upstream/.venv-jax/bin/pip install "jax[cpu]" dm-haiku rlax \\
        chex distrax optax ml_collections immutabledict jmp dm-env flax
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO_ROOT, Report, load_trainer, parse_devices, upstream

HORIZON, BATCH, N_ACTIONS = 7, 3, 5
NUM_CALLS = 3
VENV_PYTHON = os.path.join(REPO_ROOT, "third_party", "upstream", ".venv-jax", "bin", "python")


def weights_path():
    override = os.environ.get("DISCO_META_WEIGHTS")
    if override and os.path.exists(override):
        return override
    cached = os.path.join(os.path.expanduser("~"), ".cache", "cule-disco", "disco_103.npz")
    if os.path.exists(cached):
        return cached
    vendored = os.path.join(
        upstream("disco_rl"), "disco_rl", "update_rules", "weights", "disco_103.npz")
    if os.path.exists(vendored):
        return vendored
    sys.exit("disco_103.npz not found; run a disco trainer once to download it")


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    steps = HORIZON + 1

    def normal(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    is_terminal = np.zeros((HORIZON, BATCH), dtype=np.float32)
    is_terminal[2, 0] = 1.0
    is_terminal[5, 2] = 1.0
    return dict(
        actions=rng.integers(N_ACTIONS, size=(steps, BATCH)).astype(np.int64),
        rewards=normal(HORIZON, BATCH),
        is_terminal=is_terminal,
        logits=normal(steps, BATCH, N_ACTIONS),
        behaviour_logits=normal(steps, BATCH, N_ACTIONS),
        target_logits=normal(steps, BATCH, N_ACTIONS),
        y=normal(steps, BATCH, 600),
        target_y=normal(steps, BATCH, 600),
        z=normal(steps, BATCH, N_ACTIONS, 600),
        target_z=normal(steps, BATCH, N_ACTIONS, 600),
        v_scalar=normal(steps, BATCH),
        adv=normal(HORIZON, BATCH),
        normalized_adv=normal(HORIZON, BATCH),
        q=normal(steps, BATCH, N_ACTIONS),
        qv_adv=normal(steps, BATCH, N_ACTIONS),
        normalized_qv_adv=normal(steps, BATCH, N_ACTIONS),
    )


def run_jax_side(inputs, weights):
    if not os.path.exists(VENV_PYTHON):
        sys.exit(
            f"JAX venv missing at {VENV_PYTHON}\n"
            "Create it with the command in this file's docstring."
        )
    directory = tempfile.mkdtemp(prefix="disco-crosscheck-")
    input_path = os.path.join(directory, "inputs.npz")
    output_path = os.path.join(directory, "outputs.npz")
    np.savez(input_path, num_calls=NUM_CALLS, **inputs)
    result = subprocess.run(
        [VENV_PYTHON, os.path.join(os.path.dirname(__file__), "disco_jax_runner.py"),
         "--repo", upstream("disco_rl"), "--weights", weights,
         "--inputs", input_path, "--outputs", output_path],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-5000:])
        sys.exit("the JAX reference failed to run")
    return dict(np.load(output_path, allow_pickle=True))


def run(device, seed, reference, inputs):
    ours = load_trainer("disco_atari")
    report = Report(f"DiscoRL vs google-deepmind/disco_rl [{device}]", tolerance=2e-4)

    weights = weights_path()
    with np.load(weights) as arrays:
        arrays = {key: arrays[key] for key in arrays.files}

    net = ours.DiscoMetaNet().to(device)
    net.load_published_weights(arrays)
    net.eval().requires_grad_(False)

    # Every published array is consumed, and the parameter set is the one the
    # Haiku module itself declares.
    report.note("parameter count == reference",
                sum(p.numel() for p in net.parameters()) == int(reference["param_count"]),
                f"{int(reference['param_count']):,}")
    report.note("parameter names == reference",
                sorted(arrays) == sorted(reference["param_names"].tolist()))

    fields = {}
    for name, value in inputs.items():
        dtype = torch.int64 if name == "actions" else torch.float32
        fields[name] = torch.as_tensor(value, dtype=dtype, device=device)
    meta_inputs = ours.MetaInputs(**fields)

    for index in range(NUM_CALLS):
        pi_hat, y_hat, z_hat = net(meta_inputs)
        report.check(f"call {index}: pi_hat", pi_hat, reference[f"pi_{index}"])
        report.check(f"call {index}: y_hat", y_hat, reference[f"y_{index}"])
        report.check(f"call {index}: z_hat", z_hat, reference[f"z_{index}"])
        report.check(f"call {index}: lifetime LSTM hidden",
                     net.meta_hidden, reference[f"state_hidden_{index}"])
        report.check(f"call {index}: lifetime LSTM cell",
                     net.meta_cell, reference[f"state_cell_{index}"])

    # The constructed meta input must be exactly what the published kernel reads.
    x, policy_emb = net.encoder(meta_inputs)
    report.note("constructed input width == 27", x.shape[-1] == ours.META_INPUT_SIZE,
                f"{tuple(x.shape)}")
    report.note("action-conditional embedding is 2 channels", policy_emb.shape[-1] == 2)
    report.note("action-conditional input width == 9",
                net.encoder.policy_net.layers[0].in_features
                == 2 * ours.ACTION_CONDITIONAL_INPUT_SIZE)

    # Config, against the published get_settings_disco().
    settings = open(os.path.join(upstream("disco_rl"), "disco_rl", "agent.py")).read()
    args = ours.Args()
    expected = {
        "pi_cost=1.0": args.pi_cost == 1.0,
        "y_cost=1.0": args.y_cost == 1.0,
        "z_cost=1.0": args.z_cost == 1.0,
        "value_cost=0.2": args.value_cost == 0.2,
        "aux_policy_cost=1.0": args.aux_policy_cost == 1.0,
        "target_params_coeff=0.9": args.target_params_coeff == 0.9,
        "value_fn_td_lambda=0.95": args.td_lambda == 0.95,
        "discount_factor=0.997": args.discount == 0.997,
        "value_discount=0.997": args.discount == 0.997,
        "num_bins=601": args.num_bins == 601,
        "max_abs_value=300.0": args.max_abs_value == 300.0,
        "learning_rate=0.0003": args.learning_rate == 3e-4,
        "max_abs_update=1.0": args.max_abs_update == 1.0,
        "head_w_init_std=1e-2": args.head_w_init_std == 1e-2,
        "prediction_size=600": ours.PREDICTION_SIZE == 600,
    }
    absent = [text for text in expected if text.replace(" ", "") not in settings.replace(" ", "")]
    wrong = [text for text, ok in expected.items() if not ok]
    report.note(f"get_settings_disco ({len(expected)} fields)", not absent and not wrong,
                (f"not found upstream: {absent}" if absent else "")
                + (f" mismatched: {wrong}" if wrong else ""))

    return report.print()


def main():
    devices, seed = parse_devices()
    inputs = make_inputs(seed)
    weights = weights_path()
    print(f"running the Haiku reference in {VENV_PYTHON}")
    reference = run_jax_side(inputs, weights)
    results = [run(device, seed, reference, inputs) for device in devices]
    print()
    if all(results):
        print("DiscoRL: CONFIRMED against google-deepmind/disco_rl on " + ", ".join(devices))
        return 0
    print("DiscoRL: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
