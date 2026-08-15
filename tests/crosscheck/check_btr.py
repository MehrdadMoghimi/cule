"""Diff cleanrl/btr_atari.py against VIPTankz/BTR.

Both sides are PyTorch. Their `ImpalaCNNLargeIQN` weights are transplanted into
our `QNetwork` and every stage of the forward pass is compared, then the full
Munchausen-IQN loss is reproduced from their `Agent.learn` maths and diffed.
"""

import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import Report, copy_matching, load_module, load_trainer, parse_devices, sys_path, upstream

N_ACTIONS = 6
BATCH = 4
N_TAU = 8


class EnvStub:
    def __init__(self, n):
        self.single_action_space = type("Discrete", (), {"n": n})()


def upstream_defaults():
    """Parse `default=` out of upstream main.py's add_argument calls."""
    source = open(os.path.join(upstream("BTR"), "main.py")).read()
    found = {}
    for match in re.finditer(r"add_argument\(\s*'--(\w+)'[^)]*?default=([^,)]+)", source):
        name, raw = match.group(1), match.group(2).strip()
        try:
            found[name] = float(raw)
        except ValueError:
            pass
    return found


def load_upstream():
    """networks.py, plus Agent.py for its calculate_huber_loss if it imports.

    Agent.py pulls in matplotlib and torchvision; if either is missing the
    Huber helper is taken from its published three-line definition instead.
    """
    root = upstream("BTR")
    with sys_path(root):
        networks = load_module("btr_upstream_networks", os.path.join(root, "networks.py"))
        try:
            agent = load_module("btr_upstream_agent", os.path.join(root, "Agent.py"))
            huber = agent.calculate_huber_loss
            source = "Agent.calculate_huber_loss"
        except Exception:
            def huber(td_errors, k=1.0, taus=None):
                return torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2),
                                   k * (td_errors.abs() - 0.5 * k))
            source = "transcribed calculate_huber_loss"
    return networks, huber, source


def build_pair(ours, networks, device):
    their_net = networks.ImpalaCNNLargeIQN(
        4, N_ACTIONS, model_size=2, spectral=True, device=device, noisy=True,
        maxpool=True, num_tau=N_TAU, maxpool_size=6, dueling=True,
        linear_size=512, ncos=64, arch="impala", layer_norm=False, activation="relu",
    ).to(device)
    our_net = ours.QNetwork(
        EnvStub(N_ACTIONS), n_cos=64, n_policy_taus=N_TAU, model_size=2,
        maxpool_size=6, linear_size=512, noisy_std=0.5, spectral=True,
    ).to(device)
    copy_matching(our_net, their_net, rename=[
        ("dueling.value_branch.", "value_head."),
        ("dueling.advantage_branch.", "advantage_head."),
    ], allow_missing=["cos_multipliers"])  # upstream keeps `pis` as a plain attribute
    return their_net.eval(), our_net.eval()


def run(device, seed):
    ours = load_trainer("btr_atari")
    networks, upstream_huber, huber_source = load_upstream()
    report = Report(f"BTR vs VIPTankz/BTR [{device}]")

    torch.manual_seed(seed)
    their_net, our_net = build_pair(ours, networks, device)

    report.note("Parameter count",
                sum(p.numel() for p in our_net.parameters())
                == sum(p.numel() for p in their_net.parameters()),
                f"{sum(p.numel() for p in our_net.parameters()):,}")
    report.check("cosine basis (pis)", our_net.cos_multipliers, their_net.pis.reshape(-1))
    report.note("feature dim", our_net.feature_dim == their_net.conv_out_size,
                f"{our_net.feature_dim}")

    observations = (torch.rand(BATCH, 4, 84, 84, device=device) * 255).round()

    # --- trunk, stage by stage ---
    scaled = observations.float() / 255.0
    their_stage = scaled
    our_stage = scaled
    for index in range(3):
        their_stage = their_net.conv[index](their_stage)
        our_stage = our_net.conv[index](our_stage)
        report.check(f"Impala block {index}", our_stage, their_stage)
    their_features = their_net.pool(their_net.conv[3](their_stage)).reshape(BATCH, -1)
    our_features = our_net.features(observations)
    report.check("trunk features (pooled)", our_features, their_features)

    # --- quantile head, with taus pinned on both sides ---
    taus = torch.rand(BATCH, N_TAU, device=device)
    their_net.calc_cos = lambda batch_size, n_tau=N_TAU: (
        torch.cos(taus.unsqueeze(-1) * their_net.pis), taus.unsqueeze(-1))

    their_quantiles, their_taus = their_net(observations)
    our_quantiles = our_net.quantile_values(our_features, taus)
    report.check("quantiles z_tau(x, a)", our_quantiles, their_quantiles)
    report.check("taus", taus.unsqueeze(-1), their_taus)

    their_advantages, _ = their_net(observations, advantages_only=True)
    our_advantages = our_net.quantile_values(our_features, taus, advantages_only=True)
    report.check("advantages_only path", our_advantages, their_advantages)
    report.check("qvals (mean over taus)", our_quantiles.mean(1), their_net.qvals(observations))

    # --- noisy layers ---
    their_layer = networks.FactorizedNoisyLinear(64, 32, 0.5).to(device)
    our_layer = ours.FactorizedNoisyLinear(64, 32, 0.5).to(device)
    copy_matching(our_layer, their_layer)
    report.check("noisy weight_sigma init", our_layer.weight_sigma, their_layer.weight_sigma)
    report.check("noisy bias_sigma init", our_layer.bias_sigma, their_layer.bias_sigma)
    report.note("noise disabled at init",
                bool((their_layer.weight_epsilon == 0).all() and (our_layer.weight_epsilon == 0).all()))
    torch.manual_seed(11)
    their_layer.reset_noise()
    torch.manual_seed(11)
    our_layer.reset_noise()
    report.check("reset_noise (factorized)", our_layer.weight_epsilon, their_layer.weight_epsilon)
    x = torch.randn(5, 64, device=device)
    report.check("noisy forward", our_layer(x), their_layer(x))

    # --- the Munchausen-IQN loss ---
    # Upstream's Agent.learn maths, written out with the sampled tensors fixed.
    alpha, tau, clip, gamma, n_step, kappa = 0.9, 0.03, -1.0, 0.997, 3, 1.0
    actions = torch.randint(N_ACTIONS, (BATCH,), device=device)
    rewards = torch.randn(BATCH, device=device)
    dones = torch.zeros(BATCH, device=device)
    dones[1] = 1.0

    online_q = torch.randn(BATCH, N_ACTIONS, device=device)       # Q(s, .) online
    next_online_q = torch.randn(BATCH, N_ACTIONS, device=device)  # Q(s', .) online
    target_next_z = torch.randn(BATCH, N_TAU, N_ACTIONS, device=device)
    predicted = torch.randn(BATCH, N_TAU, device=device, requires_grad=True)
    predicted_ours = predicted.detach().clone().requires_grad_(True)

    def upstream_target():
        # Munchausen bonus off the ONLINE network at s.
        tau_log_pi_current = ours.scaled_log_softmax(online_q, tau)
        bonus = alpha * tau_log_pi_current.gather(1, actions.unsqueeze(-1)).clamp(clip, 0.0)
        # Soft bootstrap: E_pi[z(s',a) - tau ln pi(a|s')] under the target draw.
        tau_log_pi_next = ours.scaled_log_softmax(next_online_q, tau)
        pi_next = F.softmax(next_online_q / tau, dim=-1)
        soft = (pi_next.unsqueeze(1) * (target_next_z - tau_log_pi_next.unsqueeze(1))).sum(2)
        return rewards.unsqueeze(-1) + bonus + (gamma ** n_step) * soft * (1 - dones.unsqueeze(-1))

    target = upstream_target()
    our_target = ours.munchausen_target(
        online_q, next_online_q, target_next_z, actions.unsqueeze(-1),
        rewards.unsqueeze(-1), dones.unsqueeze(-1), alpha, tau, clip, gamma ** n_step)
    report.check("Munchausen target", our_target, target)

    # Upstream's exact reduction: quantil_l.sum(dim=1).mean(dim=1), where axis 1
    # is the predicting tau and axis 2 the target tau.
    td_error = target.unsqueeze(1) - predicted.unsqueeze(2)
    huber = upstream_huber(td_error, kappa, N_TAU)
    quantil = (taus.unsqueeze(-1) - (td_error.detach() < 0).float()).abs() * huber / kappa
    theirs_loss = quantil.sum(dim=1).mean(dim=1)
    theirs_priority = td_error.abs().sum(dim=1).mean(dim=1)

    our_errors = ours.quantile_td_errors(predicted_ours, our_target)
    ours_loss = ours.quantile_huber_loss(our_errors, taus, kappa)
    report.check("quantile Huber loss", ours_loss, theirs_loss)
    report.check("PER priority (|td|)", our_errors.abs().mean(2).sum(1), theirs_priority)
    theirs_loss.sum().backward()
    ours_loss.sum().backward()
    report.check("quantile Huber gradient", predicted_ours.grad, predicted.grad)
    report.note("Huber source", True, huber_source)

    # --- hyperparameters, read out of upstream's own argparse defaults ---
    defaults = upstream_defaults()
    args = ours.Args()
    mapping = {
        "envs": args.num_envs, "bs": args.batch_size, "nstep": args.n_step,
        "maxpool_size": args.maxpool_size, "lr": args.learning_rate,
        "munch_alpha": args.munchausen_alpha, "grad_clip": args.max_grad_norm,
        "discount": args.gamma, "taus": args.n_taus, "c": args.target_network_frequency,
        "linear_size": args.linear_size, "model_size": args.model_size,
        "ncos": args.n_cos, "per_alpha": args.prioritized_replay_alpha,
        "eps_steps": args.epsilon_decay_steps, "framestack": 4,
    }
    bad = {name: (value, defaults[name]) for name, value in mapping.items()
           if name in defaults and float(value) != float(defaults[name])}
    missing = [name for name in mapping if name not in defaults]
    report.note(f"main.py defaults ({len(mapping) - len(missing)} fields)",
                not bad and not missing,
                (str(bad) if bad else "") + (f" missing={missing}" if missing else ""))

    # Constants that live in Agent.__init__ rather than argparse.
    agent_source = open(os.path.join(upstream("BTR"), "Agent.py")).read()
    internal = {
        "entropy_tau = 0.03": args.munchausen_tau == 0.03,
        "self.lo = -1": args.munchausen_clip == -1.0,
        "max_mem_size=1048576": args.buffer_size == 2**20,
        "self.min_sampling_size = 200000": args.learning_starts == 200000,
        "eps=0.005 / self.batch_size": args.adam_eps_ratio == 0.005,
        "self.num_tau = taus": args.n_taus == args.n_target_taus == args.n_policy_taus,
    }
    absent = [text for text in internal if text not in agent_source]
    wrong = [text for text, ok in internal.items() if not ok]
    report.note(f"Agent.py constants ({len(internal)} fields)", not absent and not wrong,
                (f"not found upstream: {absent}" if absent else "")
                + (f" mismatched: {wrong}" if wrong else ""))

    report.check("Adam eps (0.005 / batch)",
                 torch.tensor(args.adam_eps_ratio / args.batch_size),
                 torch.tensor(0.005 / 256), tolerance=0.0)

    return report.print()


def main():
    devices, seed = parse_devices()
    results = [run(device, seed) for device in devices]
    print()
    if all(results):
        print("BTR: CONFIRMED against VIPTankz/BTR on " + ", ".join(devices))
        return 0
    print("BTR: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
