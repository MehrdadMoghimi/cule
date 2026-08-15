"""Run google/dopamine's real Flax QR-DQN and IQN networks. JAX venv only.

Invoked by `check_dopamine.py`. Two accommodations, both unrelated to the
networks themselves:

  * `dopamine.discrete_domains.atari_lib` pulls in `baselines`, legacy `gym`
    and `tensorflow` purely for environment wrappers and a ROM-copying helper.
    `dopamine.jax.networks` uses it only for three namedtuples, so a stub
    supplying exactly those is injected before the import. The network and loss
    code that is actually compared is upstream's, unmodified;
  * the clone is pinned to the last revision whose `atari_lib.py` parses
    (upstream `master` currently has a syntax error in it).
"""

import argparse
import os
import sys
import types

import numpy as np


def stub_atari_lib():
    """Supply only the three namedtuples `dopamine.jax.networks` reads."""
    import collections

    module = types.ModuleType("dopamine.discrete_domains.atari_lib")
    module.NATURE_DQN_OBSERVATION_SHAPE = (84, 84)
    module.NATURE_DQN_STACK_SIZE = 4
    module.DQNNetworkType = collections.namedtuple("dqn_network", ["q_values"])
    module.RainbowNetworkType = collections.namedtuple(
        "c51_network", ["q_values", "logits", "probabilities"])
    module.ImplicitQuantileNetworkType = collections.namedtuple(
        "iqn_network", ["quantile_values", "quantiles"])
    sys.modules["dopamine.discrete_domains.atari_lib"] = module

    package = types.ModuleType("dopamine.discrete_domains")
    package.atari_lib = module
    package.__path__ = []
    sys.modules.setdefault("dopamine.discrete_domains", package)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    arguments = parser.parse_args()

    sys.path.insert(0, arguments.repo)
    stub_atari_lib()

    import jax
    import jax.numpy as jnp
    from dopamine.jax import losses, networks

    data = np.load(arguments.inputs)
    observations = jnp.asarray(data["observations"])  # [N, 84, 84, 4]
    action_dim = int(data["action_dim"])
    num_atoms = int(data["num_atoms"])
    num_quantiles = int(data["num_quantiles"])
    results = {}

    # ---------------- QR-DQN ----------------
    quantile_net = networks.QuantileNetwork(num_actions=action_dim, num_atoms=num_atoms)
    variables = quantile_net.init(jax.random.PRNGKey(0), observations[0])
    outputs = jax.vmap(lambda x: quantile_net.apply(variables, x))(observations)
    results["qrdqn_logits"] = np.asarray(outputs.logits)
    results["qrdqn_q_values"] = np.asarray(outputs.q_values)
    params = variables["params"]
    for index in range(3):
        kernel = np.asarray(params[f"Conv_{index}"]["kernel"])
        results[f"qrdqn_conv{index}_weight"] = np.transpose(kernel, (3, 2, 0, 1))
        results[f"qrdqn_conv{index}_bias"] = np.asarray(params[f"Conv_{index}"]["bias"])
    for index in range(2):
        results[f"qrdqn_dense{index}_weight"] = np.asarray(params[f"Dense_{index}"]["kernel"]).T
        results[f"qrdqn_dense{index}_bias"] = np.asarray(params[f"Dense_{index}"]["bias"])

    # ---------------- IQN ----------------
    iqn_net = networks.ImplicitQuantileNetwork(num_actions=action_dim, quantile_embedding_dim=64)
    key = jax.random.PRNGKey(1)
    iqn_variables = iqn_net.init(key, observations[0], num_quantiles=num_quantiles, rng=key)
    iqn_params = iqn_variables["params"]
    for index in range(3):
        kernel = np.asarray(iqn_params[f"Conv_{index}"]["kernel"])
        results[f"iqn_conv{index}_weight"] = np.transpose(kernel, (3, 2, 0, 1))
        results[f"iqn_conv{index}_bias"] = np.asarray(iqn_params[f"Conv_{index}"]["bias"])
    for index in range(3):
        results[f"iqn_dense{index}_weight"] = np.asarray(iqn_params[f"Dense_{index}"]["kernel"]).T
        results[f"iqn_dense{index}_bias"] = np.asarray(iqn_params[f"Dense_{index}"]["bias"])

    # Call the module itself rather than replaying its arithmetic: it samples
    # its own taus and returns them, so the comparison can be run at exactly
    # the taus upstream used.
    iqn_out = iqn_net.apply(
        iqn_variables, observations[0], num_quantiles=num_quantiles, rng=jax.random.PRNGKey(7))
    results["iqn_quantile_values"] = np.asarray(iqn_out.quantile_values)
    results["iqn_taus"] = np.asarray(iqn_out.quantiles)
    results["iqn_observation"] = np.asarray(observations[0])

    # ---------------- QR-DQN target and loss ----------------
    # Upstream's `target_distribution` and `loss_fn` are run for real. The
    # network is replaced by a lookup table so the comparison isolates the loss
    # arithmetic from the encoder, which this fork deliberately keeps as
    # CleanRL's VALID-padded Nature CNN rather than dopamine's SAME-padded one.
    import flax.linen as fnn
    import optax
    from dopamine.jax.agents.quantile import quantile_agent

    class TableQuantileNetwork(fnn.Module):
        num_actions: int
        num_atoms: int
        batch: int

        @fnn.compact
        def __call__(self, x):
            table = self.param(
                "table", lambda key: jnp.zeros((self.batch, self.num_actions, self.num_atoms)))
            logits = table[x]
            return sys.modules["dopamine.discrete_domains.atari_lib"].RainbowNetworkType(
                jnp.mean(logits, axis=1), logits, fnn.softmax(logits))

    batch = int(data["batch"])
    online_logits = jnp.asarray(data["online_logits"])   # [B, A, atoms]
    target_logits = jnp.asarray(data["target_logits"])
    actions = jnp.asarray(data["qr_actions"])
    rewards = jnp.asarray(data["qr_rewards"])
    terminals = jnp.asarray(data["qr_terminals"])
    cumulative_gamma = float(data["cumulative_gamma"])
    kappa = float(data["kappa"])

    network_def = TableQuantileNetwork(
        num_actions=action_dim, num_atoms=num_quantiles, batch=batch)
    online_params = {"params": {"table": online_logits}}
    target_params = {"params": {"table": target_logits}}
    indices = jnp.arange(batch, dtype=jnp.int32)

    optimizer = optax.adam(0.0)
    optimizer_state = optimizer.init(online_params)
    _, _, loss, mean_loss = quantile_agent.train(
        network_def, online_params, target_params, optimizer, optimizer_state,
        indices, actions, indices, rewards, terminals, kappa, num_quantiles,
        cumulative_gamma)
    results["qr_loss_per_sample"] = np.asarray(loss)
    results["qr_mean_loss"] = np.asarray(mean_loss)

    # `train` already exercises `target_distribution` internally.
    results["qr_tau_hat"] = np.asarray(
        (jnp.arange(num_quantiles, dtype=jnp.float32) + 0.5) / num_quantiles)

    # ---------------- losses ----------------
    predictions = jnp.asarray(data["huber_predictions"])
    targets = jnp.asarray(data["huber_targets"])
    results["huber_loss"] = np.asarray(losses.huber_loss(targets, predictions))
    results["mse_loss"] = np.asarray(losses.mse_loss(targets, predictions))

    np.savez(arguments.outputs, **results)
    print(f"wrote {arguments.outputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
