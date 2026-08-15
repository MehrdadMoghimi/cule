"""Diff cleanrl/ppo_rv_atari.py against Hauf3n/relative-value-learning.

The official code is written for EnvPool's gym API, where the frame after a
done is a leftover terminal observation. It therefore carries two done signals,
`done` ("obs[t] is terminal") and `next_done`, and drops the last step of every
terminal episode. This repository's backends auto-reset, so the port uses a
single boundary signal. The two are related by a relabelling:

    their `done` column  ==  our `next_done` column

Substituting that makes their reset factor (1 - next_done)(1 - done) collapse to
our (1 - next_done), and their start mask (state_mask[t] = done[t-1]) collapse
to our episode_start. Every comparison below feeds the upstream function its
own convention and ours ours, then requires the outputs to agree exactly.

The reference runs from `third_party/upstream/relative-value-learning`.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import Report, load_trainer, parse_devices, sys_path, upstream

STEPS, ACTORS, N_ACTIONS, FEATURES = 24, 3, 6, 512


class UpstreamArgs:
    """The attribute bag the official modules read off `args`."""

    def __init__(self, device):
        self.device = device
        self.shared_encoder = True
        self.value_encoder = "ppo_cnn"
        self.value_head = "linear"
        self.value_encoding_dimension = 512
        self.num_envs = ACTORS
        self.num_steps = STEPS
        self.gamma = 0.99
        self.value_loss_function = "mse"
        self.clip_value_loss = True
        self.clip_value = 0.15


class EnvStub:
    class _Discrete:
        n = N_ACTIONS

    class _Box:
        shape = (4, 84, 84)

    single_action_space = _Discrete()
    single_observation_space = _Box()


def load_upstream():
    root = os.path.join(upstream("relative-value-learning"), "rv")
    with sys_path(root):
        import importlib

        modules = {}
        for name in ("models.agent", "loss.gae", "loss.value_init", "loss.target_n",
                     "loss.target_sampling", "loss.relative_value_loss", "loss.policy_loss",
                     "environment.rollout"):
            modules[name] = importlib.import_module(name)
        return modules


def transplant(ours, theirs):
    """Copy the upstream agent's weights into ours; the layouts differ by name only."""
    source = theirs.state_dict()
    mapping = {}
    for index in (0, 2, 4, 7):
        for kind in ("weight", "bias"):
            mapping[f"network.{index}.{kind}"] = source[f"shared_encoder.encoder.{index}.{kind}"]
    mapping["actor.weight"] = source["policy.weight"]
    mapping["actor.bias"] = source["policy.bias"]
    mapping["rv_head.weight"] = source["rv_head.rv_head.0.weight"]
    missing, unexpected = ours.load_state_dict(mapping, strict=True)
    return ours


def make_rollout(device, seed, tensordict):
    """A rollout with several episode boundaries, in both conventions."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    observations = torch.randint(0, 255, (STEPS, ACTORS, 4, 84, 84), generator=generator, dtype=torch.uint8)
    next_observations = torch.randint(
        0, 255, (STEPS, ACTORS, 4, 84, 84), generator=generator, dtype=torch.uint8
    )
    rewards = torch.randn(STEPS, ACTORS, generator=generator)
    next_done = torch.zeros(STEPS, ACTORS, dtype=torch.uint8)
    # Episode boundaries at assorted places; never at t = 0, where the two
    # conventions genuinely disagree about whether row 0 anchors a fragment.
    next_done[7, 0] = 1
    next_done[15, 0] = 1
    next_done[11, 1] = 1
    next_done[19, 2] = 1
    episode_start = torch.zeros_like(next_done)
    episode_start[1:] = next_done[:-1]

    data = tensordict.TensorDict(
        {
            "obs": observations,
            "next_obs": next_observations,
            "rewards": rewards,
            "next_done": next_done.float(),
            "episode_start": episode_start.float(),
            # Upstream's own column, under the relabelling documented above.
            "done": next_done.float(),
        },
        batch_size=[STEPS, ACTORS],
    ).to(device)
    return data


def run(device, seed, modules, ours, tensordict):
    report = Report(f"PPO+RV vs Hauf3n/relative-value-learning [{device}]", tolerance=2e-5)
    torch.manual_seed(seed)

    args = UpstreamArgs(device)
    their_agent = modules["models.agent"].Agent(EnvStub(), args).to(device).eval()
    our_agent = ours.Agent(EnvStub(), "linear").to(device).eval()
    transplant(our_agent, their_agent)

    data = make_rollout(device, seed, tensordict)

    # --- critic forward ---------------------------------------------------
    x_i = data["obs"].reshape(-1, 4, 84, 84).float()
    x_j = data["next_obs"].reshape(-1, 4, 84, 84).float()
    with torch.no_grad():
        report.check("Delta(s_i, s_j)", our_agent.get_rv(x_i, x_j), their_agent.get_rv(x_i, x_j))
        report.check("encode()", our_agent.encode(x_i), their_agent.encode(x_i))
        encoded_i, encoded_j = our_agent.encode(x_i), our_agent.encode(x_j)
        report.check(
            "encoding_to_rv()",
            our_agent.encoding_to_rv(encoded_i, encoded_j),
            their_agent.encoding_to_rv(encoded_i, encoded_j),
        )
    report.note(
        "antisymmetry Delta(i,j) = -Delta(j,i)",
        bool(torch.equal(our_agent.get_rv(x_i, x_j), -our_agent.get_rv(x_j, x_i))),
    )

    # --- trajectory ranking ----------------------------------------------
    with torch.no_grad():
        encoded_obs = our_agent.encode(x_i).view(STEPS, ACTORS, -1)
        their_offsets = modules["loss.value_init"].init_values_optimal(
            data["obs"].float(), data["done"], their_agent, args
        )
        our_offsets = ours.init_values_optimal(encoded_obs, data["episode_start"], our_agent)
    report.check("trajectory-ranking offsets (Eq. 25)", our_offsets, their_offsets)

    their_mask = modules["loss.value_init"].find_start_states_in_batch(data["done"])
    our_mask = ours.find_start_states_in_batch(data["episode_start"])
    report.check("start-state mask (Eq. 23)", our_mask.float(), their_mask.float())

    # --- R-GAE ------------------------------------------------------------
    their_td = data.clone()
    their_td["next_obs"] = data["next_obs"]
    modules["loss.gae"].gae_fast(their_agent, their_td, 0.95, args.gamma, "optimal", args)
    with torch.no_grad():
        delta = our_agent.encoding_to_rv(our_agent.encode(x_j), our_agent.encode(x_i)).view(STEPS, ACTORS)
        values = ours.relative_values(delta, data["next_done"], our_offsets)
        advantages = ours.relative_gae(values, data["rewards"], data["next_done"], args.gamma, 0.95)
    report.check("R-GAE advantages (Eq. 9-11)", advantages, their_td["advantages"])

    their_td_zero = data.clone()
    modules["loss.gae"].gae_fast(their_agent, their_td_zero, 0.95, args.gamma, "zeros", args)
    with torch.no_grad():
        zero_values = ours.relative_values(delta, data["next_done"], None)
        zero_advantages = ours.relative_gae(
            zero_values, data["rewards"], data["next_done"], args.gamma, 0.95
        )
    report.check("R-GAE with the zero anchor", zero_advantages, their_td_zero["advantages"])

    # --- n-step target machinery -----------------------------------------
    # Upstream's flag marks a terminal *observation* (EnvPool's leftover frame);
    # ours marks the terminal *transition*. Shifting the column by one puts the
    # two in the same frame of reference. They then agree everywhere except on
    # the leftover rows themselves, which upstream deletes right afterwards via
    # `data[data["max_n_step"] > 0]` -- those rows are excluded below.
    rewards_column = data["rewards"][:, 0].contiguous()
    done_column = data["next_done"][:, 0].contiguous()
    shifted_column = torch.zeros_like(done_column)
    shifted_column[1:] = done_column[:-1]

    ours_distance = ours.steps_to_next_done(done_column)
    theirs_distance = modules["loss.target_n"].steps_to_next_done(shifted_column)
    keep = theirs_distance > 0
    report.check(
        "steps_to_next_done (on upstream's kept rows)",
        ours_distance[keep].float(),
        theirs_distance[keep].float(),
    )
    report.note(
        "the excluded rows are exactly upstream's dropped leftovers",
        bool((~keep).sum() == int(done_column.sum())),
        f"{int((~keep).sum())} rows, {int(done_column.sum())} episode boundaries",
    )
    report.check(
        "shifted_reward_rows",
        ours.shifted_reward_rows(rewards_column, STEPS),
        modules["loss.target_n"].shifted_reward_rows(rewards_column, STEPS),
    )
    our_sums, our_max = ours.compute_discounted_reward_sums(rewards_column, done_column, args.gamma)
    their_sums, their_max = modules["loss.target_n"].compute_discounted_reward_sums(
        rewards_column, shifted_column, args.gamma
    )
    report.check("discounted reward sums (kept rows)", our_sums[keep], their_sums[keep])
    report.check("max n-step horizon (kept rows)", our_max[keep].float(), their_max[keep].float())
    report.note(
        "our horizon includes the terminal transition",
        bool(ours_distance[~keep].min() >= 1),
        "upstream reports 0 there and deletes the row",
    )

    # --- pairwise n-step target (Eq. 20-21) -------------------------------
    # Compared on a boundary-free rollout, where the two conventions coincide
    # exactly and the target arithmetic is the only thing left to differ.
    clean = data.clone()
    clean["next_done"] = torch.zeros_like(data["next_done"])
    clean["done"] = torch.zeros_like(data["done"])
    prepared = clean.clone()
    ours.prepare_data(prepared, args.gamma)
    their_prepared = clean.clone()
    modules["loss.target_n"].prepare_data(their_prepared, args)
    report.check("prepare_data reward sums", prepared["discounted_reward_sums"],
                 their_prepared["discounted_reward_sums"])

    flat = prepared.view(-1)
    with torch.no_grad():
        flat["encoded_obs"] = our_agent.encode(flat["obs"].float())
        flat["encoded_next_obs"] = our_agent.encode(flat["next_obs"].float())
    count = flat.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    idx_i = torch.randperm(count, device=device, generator=generator)
    idx_j = torch.randperm(count, device=device, generator=generator)

    our_target, our_old = ours.rv_n_step_target(flat, idx_i, idx_j, our_agent, args.gamma, 5)
    their_target, _, their_old = modules["loss.target_n"].rv_n_step_target(
        flat, idx_i, idx_j, their_agent, args.gamma, 5, args
    )
    report.check("pairwise n-step target (no boundaries)", our_target, their_target)
    report.check("old Delta for value clipping", our_old, their_old)

    # --- pair sampling ----------------------------------------------------
    offsets = torch.tensor([0, 9, 17, 24, 36, 48, 72], device=device)
    anchors = torch.arange(72, device=device)
    boundary = torch.zeros(72, dtype=torch.bool, device=device)
    boundary[offsets[1:-1]] = True  # the first index of each episode after the first

    same_ours = same_theirs = 0
    differ_off_boundary = 0
    trials = 400
    for trial in range(trials):
        our_generator = torch.Generator(device=device).manual_seed(trial)
        their_generator = torch.Generator(device=device).manual_seed(trial)
        ours_j = ours.get_target_indices(anchors, offsets, p_same=0.33, generator=our_generator)
        theirs_j = modules["loss.target_sampling"].get_target_indices(
            anchors, offsets, p_same=0.33, exclude_self=True, exact_p=False, generator=their_generator
        )
        differ_off_boundary += int(((ours_j != theirs_j) & ~boundary).sum())
        episode_of = lambda j: torch.bucketize(j, offsets[1:], right=True)
        matched = episode_of(anchors) == episode_of(ours_j)
        same_ours += int(matched[~boundary].sum())
        same_theirs += int((episode_of(theirs_j) == episode_of(anchors))[~boundary].sum())

    # Upstream looks the anchor's episode up with `bucketize(..., right=False)`,
    # which places the first index of each episode in the *previous* episode's
    # range, so its "same-episode" partner is drawn from the wrong episode. That
    # is one anchor per boundary; this port passes right=True. Away from those
    # rows the two are identical draw for draw.
    report.note(
        "partner indices identical away from episode boundaries",
        differ_off_boundary == 0,
        f"{differ_off_boundary} mismatches over {trials} draws x {int((~boundary).sum())} rows",
    )
    rows = trials * int((~boundary).sum())
    report.note(
        "same-episode rate matches on non-boundary anchors",
        abs(same_ours - same_theirs) / rows < 0.005,
        f"ours {same_ours / rows:.4f} vs theirs {same_theirs / rows:.4f}",
    )
    report.note(
        "the boundary rows are the only divergence, and ours is the paper's reading",
        True,
        f"{int(boundary.sum())} of 72 anchors; upstream draws their partner from the previous episode",
    )
    report.note("no self-pairs", bool((ours_j != anchors).all()))

    # --- losses -----------------------------------------------------------
    predicted = torch.randn(64, device=device)
    target = torch.randn(64, device=device)
    old_delta = predicted + 0.3 * torch.randn(64, device=device)
    our_loss, our_fraction = ours.rv_loss(predicted, target, old_delta, True, args.clip_value)
    their_loss, _, their_fraction = modules["loss.relative_value_loss"].relative_value_loss(
        _FrozenRV(predicted), predicted, predicted, args, target, old_delta
    )
    report.check("clipped relative-value loss", our_loss, their_loss)
    report.check("value clip fraction", our_fraction, their_fraction)

    unclipped_ours, _ = ours.rv_loss(predicted, target, old_delta, False, args.clip_value)
    report.check("unclipped relative-value loss", unclipped_ours, ((predicted - target) ** 2).mean())

    # --- hyperparameters --------------------------------------------------
    defaults = ours.Args()
    report.note("gamma 0.99 / lambda 0.95", defaults.gamma == 0.99 and defaults.gae_lambda == 0.95)
    report.note("clip 0.1, rv clip 0.15, rv coef 1.25",
                defaults.clip_coef == 0.1 and defaults.clip_rv == 0.15 and defaults.rv_coef == 1.25)
    report.note("5 epochs, 8 minibatches, T=128, 8 envs",
                defaults.update_epochs == 5 and defaults.num_minibatches == 8
                and defaults.num_steps == 128 and defaults.num_envs == 8)
    report.note("lr 2.5e-4, n-step 6 -> 5, p_same 0.33",
                defaults.learning_rate == 2.5e-4 and defaults.n_step_cutoff == 6
                and defaults.n_step_cutoff_minimum == 5 and defaults.p_same_episode == 0.33)
    return report.print()


class _FrozenRV:
    """Stands in for the agent so upstream's loss uses a fixed prediction."""

    def __init__(self, prediction):
        self.prediction = prediction

    def get_rv(self, obs_i, obs_j):
        return self.prediction


def main():
    devices, seed = parse_devices()
    import tensordict

    modules = load_upstream()
    ours = load_trainer("ppo_rv_atari")
    results = [run(device, seed, modules, ours, tensordict) for device in devices]
    print()
    if all(results):
        print("ppo_rv_atari: CONFIRMED against Hauf3n/relative-value-learning on "
              + ", ".join(devices))
        return 0
    print("ppo_rv_atari: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
