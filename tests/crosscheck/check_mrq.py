"""Diff cleanrl/mrq_atari.py against facebookresearch/MRQ.

Both sides are PyTorch, so this is the strongest form of check available: build
the authors' modules, transplant their weights into ours, and compare every
forward pass and every loss term numerically.

MRQ is CC BY-NC 4.0. It is cloned into third_party/upstream (gitignored) and
imported at runtime; nothing from it is copied into this repository.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import Report, copy_matching, load_module, load_trainer, parse_devices, sys_path, upstream

N_ACTIONS = 6
BATCH = 8
ENC_HORIZON = 5
Q_HORIZON = 3


def load_upstream():
    root = os.path.join(upstream("MRQ"), "MRQ")
    with sys_path(root):
        models = load_module("mrq_upstream_models", os.path.join(root, "models.py"))
        agent = load_module("mrq_upstream_agent", os.path.join(root, "MRQ.py"))
    return models, agent


def build_pair(ours, models, device):
    """Their modules, and ours with their weights transplanted."""
    their_encoder = models.Encoder(4, N_ACTIONS, True, 65, 512, 256, 512, 512, "elu").to(device)
    their_policy = models.Policy(N_ACTIONS, True, 10.0, 512, 512, "relu").to(device)
    their_value = models.Value(512, 512, "elu").to(device)

    our_encoder = ours.Encoder(4, N_ACTIONS, 65, 512, 256, 512, 512).to(device)
    our_policy = ours.Policy(N_ACTIONS, 10.0, 512, 512).to(device)
    our_value = ours.Value(512, 512).to(device)

    copy_matching(our_encoder, their_encoder, rename=[("zsa.", "zsa_mlp.")])
    copy_matching(our_policy, their_policy)
    copy_matching(our_value, their_value,
                  rename=[("q1.q1.", "q1.body."), ("q1.q2.", "q1.out."),
                          ("q2.q1.", "q2.body."), ("q2.q2.", "q2.out.")])
    for module in (their_encoder, their_policy, their_value, our_encoder, our_policy, our_value):
        module.eval().requires_grad_(False)
    return (their_encoder, their_policy, their_value), (our_encoder, our_policy, our_value)


def run(device, seed):
    ours = load_trainer("mrq_atari")
    models, agent = load_upstream()
    report = Report(f"MR.Q vs facebookresearch/MRQ [{device}]")

    torch.manual_seed(seed)
    (their_encoder, their_policy, their_value), (our_encoder, our_policy, our_value) = build_pair(
        ours, models, device)

    observations = (torch.rand(BATCH, 4, 84, 84, device=device) * 255).round()
    actions = torch.randint(N_ACTIONS, (BATCH,), device=device)
    one_hot = F.one_hot(actions, N_ACTIONS).float()

    # --- architecture ---
    their_zs = their_encoder.zs(observations)
    our_zs = our_encoder.zs(observations)
    report.check("Encoder.zs", our_zs, their_zs)

    report.check("Encoder.zsa", our_encoder(our_zs, one_hot), their_encoder(their_zs, one_hot))
    for name, mine, theirs in zip(
        ("model_all/done", "model_all/next_zs", "model_all/reward"),
        our_encoder.model_all(our_zs, one_hot),
        their_encoder.model_all(their_zs, one_hot),
    ):
        report.check(name, mine, theirs)

    their_zsa = their_encoder(their_zs, one_hot)
    report.check("Value (twin Q)", our_value(their_zsa), their_value(their_zsa))
    report.check("Policy logits", our_policy.policy(our_zs), their_policy.policy(their_zs))

    # Gumbel-softmax draws noise, so seed identically immediately before each call.
    torch.manual_seed(1234)
    their_action, their_pre = their_policy(their_zs)
    torch.manual_seed(1234)
    our_action, our_pre = our_policy(our_zs)
    report.check("Policy relaxed action", our_action, their_action)
    report.check("Policy pre-activation", our_pre, their_pre)

    report.note("Parameter count",
                sum(p.numel() for p in our_encoder.parameters())
                == sum(p.numel() for p in their_encoder.parameters()),
                f"encoder {sum(p.numel() for p in our_encoder.parameters()):,}")

    # --- two-hot reward head ---
    their_two_hot = agent.TwoHot(torch.device(device), -10.0, 10.0, 65)
    our_two_hot = ours.TwoHot(-10.0, 10.0, 65).to(device)
    report.check("TwoHot.bins", our_two_hot.bins, their_two_hot.bins)

    rewards = torch.cat([
        torch.zeros(4, 1, device=device),
        torch.randn(BATCH - 4, 1, device=device) * 3.0,
    ])
    report.check("TwoHot.transform", our_two_hot.transform(rewards), their_two_hot.transform(rewards))
    logits = torch.randn(BATCH, 65, device=device)
    report.check("TwoHot.inverse", our_two_hot.inverse(logits), their_two_hot.inverse(logits))
    report.check("TwoHot.cross_entropy_loss",
                 our_two_hot.cross_entropy_loss(logits, rewards),
                 their_two_hot.cross_entropy_loss(logits, rewards))

    # --- helpers ---
    x = torch.randn(BATCH, 512, device=device)
    y = torch.randn(BATCH, 512, device=device)
    mask = torch.randint(2, (BATCH, 1), device=device).float()
    report.check("masked_mse", ours.masked_mse(x, y, mask), agent.masked_mse(x, y, mask))

    noisy = torch.randn(BATCH, N_ACTIONS, device=device)
    report.check("realign (one-hot argmax)", ours.realign_discrete(noisy), agent.realign(noisy, True))

    step_rewards = torch.randn(BATCH, Q_HORIZON, 1, device=device)
    not_dones = torch.ones(BATCH, Q_HORIZON, 1, device=device)
    not_dones[2, 1] = 0.0
    not_dones[5, 0] = 0.0
    ours_reward, ours_scale = ours.multi_step_reward(step_rewards, not_dones, 0.99)
    theirs_reward, theirs_scale = agent.multi_step_reward(step_rewards, not_dones, 0.99)
    report.check("multi_step_reward/return", ours_reward, theirs_reward)
    report.check("multi_step_reward/term_discount", ours_scale, theirs_scale)

    # --- augmentation ---
    images = torch.rand(BATCH, 4, 84, 84, device=device) * 255
    torch.manual_seed(7)
    theirs_aug = agent.shift_aug(images)
    torch.manual_seed(7)
    ours_aug = ours.random_shift_augmentation(images)
    report.check("shift_aug", ours_aug, theirs_aug)

    # --- the encoder loss, rolled enc_horizon steps ---
    sequence_actions = F.one_hot(
        torch.randint(N_ACTIONS, (BATCH, ENC_HORIZON), device=device), N_ACTIONS).float()
    sequence_rewards = torch.randn(BATCH, ENC_HORIZON, 1, device=device)
    sequence_not_done = torch.ones(BATCH, ENC_HORIZON, 1, device=device)
    sequence_not_done[1, 2:] = 0.0
    sequence_not_done[6, 0:] = 0.0
    target_zs = torch.randn(BATCH, ENC_HORIZON, 512, device=device)

    # Upstream's train_encoder body, with the optimizer step removed.
    pred_zs = their_encoder.zs(observations)
    prev_not_done = 1
    their_loss = 0
    for i in range(ENC_HORIZON):
        pred_d, pred_zs, pred_r = their_encoder.model_all(pred_zs, sequence_actions[:, i])
        dyn = agent.masked_mse(pred_zs, target_zs[:, i], prev_not_done)
        rew = (their_two_hot.cross_entropy_loss(pred_r, sequence_rewards[:, i]) * prev_not_done).mean()
        done = agent.masked_mse(pred_d, 1.0 - sequence_not_done[:, i].reshape(-1, 1), prev_not_done)
        their_loss = their_loss + 1.0 * dyn + 0.1 * rew + 0.1 * done
        prev_not_done = sequence_not_done[:, i].reshape(-1, 1) * prev_not_done

    our_loss = ours.encoder_loss(
        our_encoder, our_two_hot, observations, sequence_actions, sequence_rewards,
        sequence_not_done, target_zs, 1.0, 0.1, 0.1)
    report.check("encoder loss (5-step roll)", our_loss, their_loss)

    # --- the RL half: target, value loss, priority ---
    next_observations = (torch.rand(BATCH, 4, 84, 84, device=device) * 255).round()
    ms_reward, term_discount = ours.multi_step_reward(step_rewards, not_dones, 0.99)
    reward_scale, target_reward_scale = 0.37, 0.21

    next_zs = their_encoder.zs(next_observations)
    torch.manual_seed(99)
    noise = (torch.randn(BATCH, N_ACTIONS, device=device) * 0.1).clamp(-0.15, 0.15)
    next_action = agent.realign(their_policy.act(next_zs) + noise, True)
    next_zsa = their_encoder(next_zs, next_action)
    next_q = their_value(next_zsa).min(1, keepdim=True).values

    their_target = (ms_reward + term_discount * next_q * target_reward_scale) / reward_scale
    our_target = ours.scaled_q_target(ms_reward, term_discount, next_q, reward_scale,
                                      target_reward_scale)
    report.check("scaled Q target", our_target, their_target)

    q = their_value(their_zsa)
    report.check("value loss (smooth_l1)",
                 F.smooth_l1_loss(q, our_target.expand(-1, 2)),
                 F.smooth_l1_loss(q, their_target.expand(-1, 2)))

    their_priority = (q - their_target.expand(-1, 2)).abs().max(1).values
    their_priority = their_priority.clamp(min=1.0).pow(0.4)
    report.check("LAP priority", ours.lap_priority(q, our_target, 1.0, 0.4), their_priority)

    # --- hyperparameters ---
    hyper = models  # placeholder to keep the import used symmetrically
    del hyper
    defaults = agent.Hyperparameters()
    args = ours.Args()
    matches = {
        "batch_size": (args.batch_size, defaults.batch_size),
        "discount": (args.gamma, defaults.discount),
        "target_update_freq": (args.target_network_frequency, defaults.target_update_freq),
        "buffer_size_before_training": (args.learning_starts, defaults.buffer_size_before_training),
        "exploration_noise": (args.exploration_noise, defaults.exploration_noise),
        "target_policy_noise": (args.target_policy_noise, defaults.target_policy_noise),
        "noise_clip": (args.noise_clip, defaults.noise_clip),
        "dyn_weight": (args.dyn_weight, defaults.dyn_weight),
        "reward_weight": (args.reward_weight, defaults.reward_weight),
        "done_weight": (args.done_weight, defaults.done_weight),
        "alpha": (args.prioritized_replay_alpha, defaults.alpha),
        "min_priority": (args.min_priority, defaults.min_priority),
        "enc_horizon": (args.enc_horizon, defaults.enc_horizon),
        "Q_horizon": (args.q_horizon, defaults.Q_horizon),
        "zs_dim": (args.zs_dim, defaults.zs_dim),
        "zsa_dim": (args.zsa_dim, defaults.zsa_dim),
        "za_dim": (args.za_dim, defaults.za_dim),
        "enc_hdim": (args.enc_hdim, defaults.enc_hdim),
        "value_hdim": (args.value_hdim, defaults.value_hdim),
        "policy_hdim": (args.policy_hdim, defaults.policy_hdim),
        "enc_lr": (args.encoder_learning_rate, defaults.enc_lr),
        "value_lr": (args.value_learning_rate, defaults.value_lr),
        "policy_lr": (args.policy_learning_rate, defaults.policy_lr),
        "enc_wd": (args.weight_decay, defaults.enc_wd),
        "value_grad_clip": (args.value_grad_clip, defaults.value_grad_clip),
        "gumbel_tau": (args.gumbel_tau, defaults.gumbel_tau),
        "pre_activ_weight": (args.pre_activ_weight, defaults.pre_activ_weight),
        "num_bins": (args.num_bins, defaults.num_bins),
        "lower": (args.bin_lower, defaults.lower),
        "upper": (args.bin_upper, defaults.upper),
        "pixel_augs": (args.data_augmentation, defaults.pixel_augs),
    }
    bad = {name: pair for name, pair in matches.items() if float(pair[0]) != float(pair[1])}
    report.note("Hyperparameters (31 fields)", not bad, str(bad) if bad else "")

    return report.print()


def main():
    devices, seed = parse_devices()
    results = [run(device, seed) for device in devices]
    print()
    if all(results):
        print("MR.Q: CONFIRMED against facebookresearch/MRQ on " + ", ".join(devices))
        return 0
    print("MR.Q: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
