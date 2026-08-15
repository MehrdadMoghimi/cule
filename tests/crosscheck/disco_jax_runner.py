"""Run google-deepmind/disco_rl's real Haiku meta-network. JAX venv only.

Invoked by `check_disco.py` inside `third_party/upstream/.venv-jax`, which is
where the JAX stack lives; the main environment stays PyTorch-only. Reads an
`.npz` of inputs, writes an `.npz` of the meta-network's outputs and its updated
lifetime LSTM state.
"""

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    arguments = parser.parse_args()

    sys.path.insert(0, arguments.repo)
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from disco_rl import types
    from disco_rl.networks import meta_nets
    from disco_rl.update_rules import disco

    data = np.load(arguments.inputs)
    flat = {key: np.load(arguments.weights)[key] for key in np.load(arguments.weights).files}

    # Haiku params are nested one level: 'module/path' -> {'w': ..., 'b': ...}.
    params = {}
    for key in flat:
        module = "/".join(key.split("/")[:-1])
        params.setdefault(module, {})[key.split("/")[-1]] = jnp.asarray(flat[key])

    net_config = dict(
        name="lstm",
        prediction_size=600,
        hidden_size=256,
        embedding_size=(16, 1),
        policy_target_channels=(16,),
        policy_channels=(16, 2),
        output_stddev=0.3,
        aux_stddev=0.3,
        policy_target_stddev=0.3,
        state_stddev=1.0,
        meta_rnn_kwargs=dict(
            policy_channels=(16, 2),
            embedding_size=(16,),
            pred_embedding_size=(16, 1),
            hidden_size=128,
        ),
        input_option=disco.get_input_option(),
    )

    def meta_net_fn(inputs, axis_name):
        return meta_nets.LSTM(**net_config)(inputs, axis_name=axis_name)

    transformed = hk.transform_with_state(meta_net_fn)

    def build_inputs():
        array = lambda name: jnp.asarray(data[name])
        rollout = types.UpdateRuleInputs(
            observations=None,
            actions=jnp.asarray(data["actions"], dtype=jnp.int32),
            rewards=array("rewards"),
            is_terminal=array("is_terminal"),
            agent_out=dict(logits=array("logits"), y=array("y"), z=array("z")),
            behaviour_agent_out=dict(logits=array("behaviour_logits")),
            value_out=None,
        )
        rollout.extra_from_rule = dict(
            v_scalar=array("v_scalar"),
            adv=array("adv"),
            normalized_adv=array("normalized_adv"),
            q=array("q"),
            qv_adv=array("qv_adv"),
            normalized_qv_adv=array("normalized_qv_adv"),
            target_out=dict(
                logits=array("target_logits"), y=array("target_y"), z=array("target_z")),
        )
        return rollout

    rollout = build_inputs()

    # A zeroed lifetime state, exactly as `init_meta_state` produces.
    _, state = transformed.init(jax.random.PRNGKey(0), rollout, axis_name=None)
    state = jax.tree.map(jnp.zeros_like, state)

    results = {}
    calls = int(data["num_calls"])
    for index in range(calls):
        outputs, state = transformed.apply(params, state, None, rollout, axis_name=None)
        results[f"pi_{index}"] = np.asarray(outputs["pi"])
        results[f"y_{index}"] = np.asarray(outputs["y"])
        results[f"z_{index}"] = np.asarray(outputs["z"])
        results[f"meta_input_emb_{index}"] = np.asarray(outputs["meta_input_emb"])
        # hk.LSTMState is a NamedTuple(hidden, cell), so the flattened leaves
        # come out in that order.
        leaves = jax.tree_util.tree_leaves(state)
        results[f"state_hidden_{index}"] = np.asarray(leaves[0])
        results[f"state_cell_{index}"] = np.asarray(leaves[1])

    # Also record the parameter tree the reference expects, to confirm our
    # PyTorch module consumes exactly the same set.
    reference_params, _ = transformed.init(jax.random.PRNGKey(0), rollout, axis_name=None)
    results["param_names"] = np.array(
        sorted(f"{module}/{leaf}" for module, block in reference_params.items() for leaf in block))
    results["param_count"] = np.array(
        sum(int(np.prod(value.shape)) for block in reference_params.values()
            for value in block.values()))

    np.savez(arguments.outputs, **results)
    print(f"wrote {arguments.outputs} ({calls} calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
