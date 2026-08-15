"""Run jacobkooi/hadamax's real Flax QNetwork. JAX venv only.

Invoked by `check_hadamax.py`. Initialises the official module, writes its
parameters (converted to PyTorch's layout) and its outputs on a fixed input.
"""

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    arguments = parser.parse_args()

    sys.path.insert(0, os.path.join(arguments.repo, "purejaxql"))
    import jax
    import jax.numpy as jnp
    import networks

    data = np.load(arguments.inputs)
    observations = jnp.asarray(data["observations"])  # [N, 4, 84, 84], 0..255
    action_dim = int(data["action_dim"])

    config = {"ENCODER": "hadamax"}
    model = networks.QNetwork(config=config, action_dim=action_dim, norm_type="layer_norm")
    variables = model.init(jax.random.PRNGKey(0), observations, train=False)
    logits, _ = model.apply(variables, observations, train=False, mutable=["batch_stats"])

    params = variables["params"]["NatureCNN_0"]
    results = {"logits": np.asarray(logits)}

    # Flax conv kernels are (kh, kw, in, out); PyTorch wants (out, in, kh, kw).
    for index in range(6):
        kernel = np.asarray(params[f"Conv_{index}"]["kernel"])
        results[f"conv{index}_weight"] = np.transpose(kernel, (3, 2, 0, 1))
        results[f"conv{index}_bias"] = np.asarray(params[f"Conv_{index}"]["bias"])
    for index in range(7):
        results[f"ln{index}_scale"] = np.asarray(params[f"LayerNorm_{index}"]["scale"])
        results[f"ln{index}_bias"] = np.asarray(params[f"LayerNorm_{index}"]["bias"])
    results["dense_weight"] = np.asarray(params["Dense_0"]["kernel"]).T
    results["dense_bias"] = np.asarray(params["Dense_0"]["bias"])
    head = variables["params"]["action_dense"]
    results["head_weight"] = np.asarray(head["kernel"]).T
    results["head_bias"] = np.asarray(head["bias"])

    # Intermediate activations, so a mismatch can be localised to a block.
    import flax.linen as fnn
    from flax.linen.pooling import max_pool

    x = jnp.transpose(observations, (0, 2, 3, 1)) / 255.0
    pools = [((4, 4), (4, 4)), ((2, 2), (2, 2)), ((3, 3), (1, 1))]
    for block in range(3):
        first, second = 2 * block, 2 * block + 1
        branches = []
        for slot, norm in ((first, first), (second, second)):
            y = jax.lax.conv_general_dilated(
                x, params[f"Conv_{slot}"]["kernel"], (1, 1), "SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"))
            y = y + params[f"Conv_{slot}"]["bias"]
            y = fnn.LayerNorm().apply(
                {"params": {"scale": params[f"LayerNorm_{norm}"]["scale"],
                            "bias": params[f"LayerNorm_{norm}"]["bias"]}}, y)
            branches.append(fnn.gelu(y))
        x = max_pool(branches[0] * branches[1], window_shape=pools[block][0],
                     strides=pools[block][1], padding="SAME")
        results[f"block{block}"] = np.transpose(np.asarray(x), (0, 3, 1, 2))

    results["features"] = np.asarray(x.reshape((x.shape[0], -1)))
    results["gelu_probe"] = np.asarray(fnn.gelu(jnp.asarray(data["gelu_probe"])))
    np.savez(arguments.outputs, **results)
    print(f"wrote {arguments.outputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
