"""How to read a value function out of each trainer, once it has finished.

`verify_chain.py` compares a trained network against the closed-form `Q*` of
`cleanrl/diagnostic_env.py`.  Doing that for all sixty-two trainers means
knowing, per trainer, where the value function lives and what to call -- some
expose `Q(s, .)` directly, some a distribution over returns, some only a critic
`V(s)`, some only through a recurrent state that has to be carried along the
trajectory.  Those differences live here so the driver stays about the
comparison rather than about the zoo.

Each adapter returns a `Prediction`.  Adapters are deliberately strict: they
assert the shape they expect, so a wrong guess about a trainer's interface
fails loudly instead of quietly scoring the comparison against garbage.

`STATUS` records what kind of check each trainer supports:

  value   -- Q(s, a) or V(s) read at raw observations; the full comparison.
  policy  -- no value readable at a raw observation (no critic, or values that
             live in a learned latent space).  Only the greedy action is
             checked, which is a real but much weaker statement.
"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class Prediction:
    """Whatever the trained network can be asked for, at the canonical states."""

    q: torch.Tensor | None = None            # (n_states, n_actions)
    v: torch.Tensor | None = None            # (n_states,)
    greedy: torch.Tensor | None = None       # (n_states,) when only a policy exists
    quantiles: torch.Tensor | None = None    # (n_states, n_actions, n_quantiles)
    atoms: torch.Tensor | None = None        # (n_atoms,)
    probs: torch.Tensor | None = None        # (n_states, n_actions, n_atoms)
    differences: torch.Tensor | None = None  # (n_states, n_states) for relative values
    note: str = ""


class Unsupported(Exception):
    """Raised by an adapter that cannot reach a value at a raw observation."""


def _check(tensor, shape, what):
    if tuple(tensor.shape) != tuple(shape):
        raise Unsupported(f"{what}: expected shape {tuple(shape)}, got {tuple(tensor.shape)}")
    return tensor


# ---------------------------------------------------------------------------
# value-based heads
# ---------------------------------------------------------------------------

def plain_q(namespace, observations, spec):
    """`QNetwork.forward` already returns `(batch, n_actions)`."""
    q = namespace["q_network"](observations)
    return Prediction(q=_check(q, (spec.n_states, spec.n_actions), "q_network(obs)"))


def categorical_q(namespace, observations, spec):
    """C51: a softmax over fixed atoms reached through the raw `network`."""
    network = namespace["q_network"]
    logits = network.network(observations / 255.0)
    probs = torch.softmax(logits.reshape(-1, network.n_actions, network.n_atoms), dim=2)
    return Prediction(q=(probs * network.atoms).sum(2), atoms=network.atoms, probs=probs)


def distributional_forward_q(namespace, observations, spec):
    """Rainbow and DER: `forward` returns the probabilities themselves."""
    network = namespace["q_network"]
    probs = network(observations)
    support = getattr(network, "support", None)
    if support is None:
        raise Unsupported("no `support` buffer on the network")
    _check(probs, (spec.n_states, spec.n_actions, support.numel()), "q_network(obs)")
    return Prediction(q=(probs * support).sum(2), atoms=support, probs=probs)


def quantile_q(namespace, observations, spec):
    """QR-DQN: uniform quantiles reshaped out of one linear head."""
    network = namespace["q_network"]
    quantiles = network.network(observations / 255.0).reshape(
        -1, network.n_actions, network.n_quantiles)
    return Prediction(q=quantiles.mean(2), quantiles=quantiles)


def implicit_quantile_q(namespace, observations, spec):
    """IQN, M-IQN, FQF and BTR: sample taus, average the sampled quantiles."""
    network = namespace["q_network"]
    features = network.features(observations)
    taus = torch.rand(features.shape[0], 64, device=features.device)
    quantiles = network.quantile_values(features, taus)
    _check(quantiles, (spec.n_states, 64, spec.n_actions), "quantile_values")
    return Prediction(q=quantiles.mean(1), quantiles=quantiles.permute(0, 2, 1))


def latent_categorical_q(namespace, observations, spec):
    """BBF and SPR: encode to a latent first, then a categorical head on it."""
    network = namespace["q_network"]
    probs = network.q_dist(network.encode(observations))
    support = getattr(network, "support", getattr(network, "atoms", None))
    if support is None:
        raise Unsupported("no `support`/`atoms` buffer on the network")
    _check(probs, (spec.n_states, spec.n_actions, support.numel()), "q_dist(encode(obs))")
    return Prediction(q=(probs * support).sum(2), atoms=support, probs=probs)


def soft_q(namespace, observations, spec):
    """Discrete SAC: the critic is a plain `Q(s, .)` head."""
    q = namespace["qf1"](observations)
    return Prediction(q=_check(q, (spec.n_states, spec.n_actions), "qf1(obs)"),
                      note="qf1 only; the entropy term shifts Q away from Q* by alpha*H")


# ---------------------------------------------------------------------------
# critics
# ---------------------------------------------------------------------------

def actor_critic_v(namespace, observations, spec):
    """PPO-family: no Q head, so the critic's `V` is what gets compared."""
    value = namespace["agent"].get_value(observations).reshape(-1)
    return Prediction(v=_check(value, (spec.n_states,), "agent.get_value(obs)"))


def stream_critic_v(namespace, observations, spec):
    """Streaming AC keeps its critic in a separate network."""
    value = namespace["value_network"](observations).reshape(-1)
    return Prediction(v=_check(value, (spec.n_states,), "value_network(obs)"))


# ---------------------------------------------------------------------------
# recurrent heads: the state has to be carried along the trajectory
# ---------------------------------------------------------------------------

def recurrent_q(namespace, observations, spec):
    """R2D2: step the LSTM down the optimal trajectory, reading Q as it goes.

    A recurrent value is only defined given a history, and the history that
    matters is the one the optimal policy produces -- the same argument that
    makes `canonical_observations` the right inputs for the feedforward nets.
    So the previous action fed in at each step is ADVANCE, and the previous
    reward is zero until the goal pays.
    """
    network = namespace["q_network"]
    device = observations.device
    state = network.initial_state(1, device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    previous_reward = torch.zeros(1, device=device)
    rows = []
    for index in range(spec.n_states):
        q, state = network.step(
            observations[index : index + 1], previous_action, previous_reward, state)
        rows.append(q.reshape(-1))
    q = torch.stack(rows)
    return Prediction(q=_check(q, (spec.n_states, spec.n_actions), "R2D2Network.step"))


def recurrent_v(namespace, observations, spec):
    """PPO+LSTM: carry `lstm_state` down the trajectory and read the critic."""
    agent = namespace["agent"]
    device = observations.device
    size = agent.lstm.hidden_size
    state = (torch.zeros(agent.lstm.num_layers, 1, size, device=device),
             torch.zeros(agent.lstm.num_layers, 1, size, device=device))
    done = torch.zeros(1, device=device)
    values = []
    for index in range(spec.n_states):
        _, _, _, value, state = agent.get_action_and_value(
            observations[index : index + 1], state, done)
        values.append(value.reshape(-1))
    value = torch.cat(values)
    return Prediction(v=_check(value, (spec.n_states,), "agent.get_action_and_value"))


# ---------------------------------------------------------------------------
# heads that are not a value at a raw observation
# ---------------------------------------------------------------------------

def relative_value(namespace, observations, spec):
    """Relative-value learning predicts `V(i) - V(j)`, never `V` itself.

    That is exactly what the closed form gives for free, so this compares the
    full matrix of pairwise differences instead of a vector of values.
    """
    agent = namespace["agent"]
    n = spec.n_states
    left = observations.repeat_interleave(n, dim=0)
    right = observations.repeat(n, 1, 1, 1)
    differences = agent.get_rv(left, right).reshape(n, n)
    return Prediction(differences=differences,
                      note="compared as V*(i) - V*(j); absolute values are not identifiable")


def policy_only(namespace, observations, spec):
    """No critic at all (GPG), so only the greedy action can be checked."""
    agent = namespace["agent"]
    if not hasattr(agent, "actor"):
        raise Unsupported("no `actor` head to read logits from")
    logits = agent.actor(agent.network(observations / 255.0))
    _check(logits, (spec.n_states, spec.n_actions), "agent.actor(agent.network(obs))")
    return Prediction(greedy=logits.argmax(dim=-1),
                      note="policy gradient without a critic: greedy action only")


def latent_policy(namespace, observations, spec):
    """MR.Q: the policy reads a learned latent, and so does the Q it is trained on.

    `Value` is a twin Q over `zsa`, the encoder's *state-action* latent, so
    there is no way to ask for `Q(s, .)` at a raw observation without also
    reimplementing the encoder's action embedding.  The greedy action is
    reachable, and that is what gets checked.
    """
    encoder, policy = namespace["encoder"], namespace["policy"]
    action = policy.act(encoder.zs(observations))
    if action.ndim == 2:  # logits or a one-hot relaxation
        action = action.argmax(dim=-1)
    _check(action.reshape(-1), (spec.n_states,), "policy.act(encoder.zs(obs))")
    return Prediction(greedy=action.reshape(-1),
                      note="Q is defined over the encoder's zsa latent, not over observations")


def unrolled_policy(namespace, observations, spec):
    """DisCo: the trainer's own `act` closure, which owns the unroll details."""
    act = namespace.get("act")
    if act is None:
        raise Unsupported("no module-level `act` closure in the namespace")
    result = act(observations)
    logits = result[0] if isinstance(result, tuple) else result
    greedy = logits.reshape(spec.n_states, -1).argmax(dim=-1)
    return Prediction(greedy=greedy,
                      note="distributional head is defined over a time-major unroll")


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def _pairs(base, adapter, extra=("_torchcompile",)):
    """A trainer and its torchcompile twin share an adapter."""
    return {base: adapter, **{base + suffix: adapter for suffix in extra}}


ADAPTERS = {}
for _name in ("dqn_atari", "drq_atari", "ibdqn_atari", "stream_q_atari",
              "qdagger_dqn_atari_impalacnn"):
    ADAPTERS.update(_pairs(_name, plain_q))
ADAPTERS.update(_pairs("pqn_atari_envpool", plain_q))
ADAPTERS.update(_pairs("hadamax_pqn_atari_envpool", plain_q))
ADAPTERS.update({
    "dqn_torchcompile": plain_q,
    "endpoint_ddqn_atari": plain_q,
    "sem_pqn_atari_envpool": plain_q,
    "gpg_atari": policy_only,
})
ADAPTERS.update(_pairs("c51_atari", categorical_q))
for _name in ("rainbow_atari", "der_atari"):
    ADAPTERS.update(_pairs(_name, distributional_forward_q))
ADAPTERS.update(_pairs("qrdqn_atari", quantile_q))
for _name in ("iqn_atari", "miqn_atari", "fqf_atari", "btr_atari"):
    ADAPTERS.update(_pairs(_name, implicit_quantile_q))
for _name in ("bbf_atari", "spr_atari"):
    ADAPTERS.update(_pairs(_name, latent_categorical_q))
ADAPTERS.update(_pairs("sac_atari", soft_q))
for _name in ("a2c_atari", "a2c_sil_atari", "dpo_atari", "impala_atari", "outer_ppo_atari",
              "ppg_atari", "ppo_atari", "ppo_rnd_atari", "ppo_scale_atari"):
    ADAPTERS[_name] = actor_critic_v
ADAPTERS["ppo_atari_envpool_torchcompile"] = actor_critic_v
ADAPTERS.update(_pairs("stream_ac_atari", stream_critic_v))
ADAPTERS.update(_pairs("r2d2_atari", recurrent_q))
ADAPTERS.update({"ppo_atari_lstm": recurrent_v, "ppo_atari_lstm_torchcompile": recurrent_v})
ADAPTERS["ppo_rv_atari"] = relative_value

ADAPTERS.update({"mrq_atari": latent_policy, "mrq_atari_torchcompile": latent_policy})
ADAPTERS.update(_pairs("disco_atari", unrolled_policy))
# A spurious twin from `_pairs`; dqn's compiled variant is `dqn_torchcompile`.
ADAPTERS.pop("dqn_atari_torchcompile", None)

# Trainers this MDP cannot reach, with the reason.  Listing them is a claim
# about their architecture that the sweep prints rather than hides.
UNSUPPORTED = {
    "dreamerv3_atari":
        "no CuLE backend (envpool/gymnasium only) and 3x64x64 RGB observations; "
        "needs its own diagnostic env variant",
    "ppo_trxl_atari":
        "the critic reads an episodic memory window, not a single observation",
}

STATUS = {name: "value" for name in ADAPTERS}
for _name in ("gpg_atari", "mrq_atari", "mrq_atari_torchcompile",
              "disco_atari", "disco_atari_torchcompile"):
    STATUS[_name] = "policy"
STATUS.update({name: "unsupported" for name in UNSUPPORTED})


def adapter_for(trainer):
    if trainer in UNSUPPORTED:
        raise Unsupported(UNSUPPORTED[trainer])
    if trainer in ADAPTERS:
        return ADAPTERS[trainer]
    return plain_q  # a conventional Q head is the reasonable default
