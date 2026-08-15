"""Diff cleanrl/stream_q_atari.py and stream_ac_atari.py against streaming-drl.

streaming-drl is CC BY-NC 4.0, so no code from it was used to write the ports;
they were written from the paper. This script closes the loop by *running* the
authors' code next to ours: the checkout lives in third_party/upstream
(gitignored) and is imported at runtime only.

Upstream is strictly single-stream. Our ports vectorise over `--num-envs`
streams, so the comparison pins `--num-envs 1`, which is exactly the regime in
which the two must agree exactly.
"""

import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import Report, load_module, load_trainer, parse_devices, sys_path, upstream

N_ACTIONS = 5
HIDDEN = 256


class EnvStub:
    def __init__(self, n):
        self.single_action_space = type("Discrete", (), {"n": n})()


def upstream_defaults(filename):
    """Parse `default=` out of an upstream script's add_argument calls."""
    source = open(os.path.join(upstream("streaming-drl"), filename)).read()
    found = {}
    for match in re.finditer(r"add_argument\(\s*'--(\w+)'[^)]*?default=([^,)]+)", source):
        try:
            found[match.group(1)] = float(match.group(2).strip().replace("_", ""))
        except ValueError:
            pass
    return found


def load_upstream():
    root = upstream("streaming-drl")
    with sys_path(root):
        optim = load_module("stream_upstream_optim", os.path.join(root, "optim.py"))
        sparse = load_module("stream_upstream_sparse_init", os.path.join(root, "sparse_init.py"))
        # The Atari entry points import stable_baselines3 and gymnasium wrappers
        # at module scope; only the networks and update rules are needed, so
        # they are reconstructed from the same building blocks below.
        return optim, sparse


def check_sparse_init(report, ours, sparse, device):
    for shape in [(64, 128), (32, 4, 8, 8)]:
        theirs = torch.empty(*shape, device=device)
        mine = torch.empty(*shape, device=device)
        torch.manual_seed(3)
        sparse.sparse_init(theirs, sparsity=0.9)
        torch.manual_seed(3)
        ours.sparse_init(mine, 0.9)
        report.check(f"sparse_init {tuple(shape)}", mine, theirs)


def check_layer_norm(report, ours, device):
    """Upstream normalises over `input.size()` -- every axis of an unbatched
    tensor. Ours takes the axis count explicitly so it also works batched."""
    unbatched = torch.randn(4, 84, 84, device=device)
    theirs = F.layer_norm(unbatched, unbatched.size())
    report.check("LayerNormalization (unbatched)", ours.LayerNormalization(3)(unbatched), theirs)

    batched = torch.randn(6, 4, 84, 84, device=device)
    per_sample = torch.stack([F.layer_norm(row, row.size()) for row in batched])
    report.check("LayerNormalization (batched == per-sample)",
                 ours.LayerNormalization(3)(batched), per_sample)

    vector = torch.randn(HIDDEN, device=device)
    report.check("LayerNormalization (vector)",
                 ours.LayerNormalization(1)(vector), F.layer_norm(vector, vector.size()))


def build_upstream_network(sparse, n_outputs, device):
    """The exact `nn.Sequential` from stream_q_atari.StreamQ / StreamAC."""

    class UpstreamLayerNorm(torch.nn.Module):
        def forward(self, x):
            return F.layer_norm(x, x.size())

    network = torch.nn.Sequential(
        torch.nn.Conv2d(4, 32, 8, stride=5), UpstreamLayerNorm(), torch.nn.LeakyReLU(),
        torch.nn.Conv2d(32, 64, 4, stride=3), UpstreamLayerNorm(), torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 64, 3, stride=2), UpstreamLayerNorm(), torch.nn.LeakyReLU(),
        torch.nn.Flatten(start_dim=0),
        torch.nn.Linear(256, HIDDEN), UpstreamLayerNorm(), torch.nn.LeakyReLU(),
        torch.nn.Linear(HIDDEN, n_outputs),
    ).to(device)

    def initialize(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            sparse.sparse_init(module.weight, sparsity=0.9)
            module.bias.data.fill_(0.0)

    network.apply(initialize)
    return network


def check_network(report, ours, sparse, device, label, module_factory, n_outputs):
    torch.manual_seed(21)
    theirs = build_upstream_network(sparse, n_outputs, device)
    torch.manual_seed(21)
    mine = module_factory().to(device)

    report.note(f"{label}: parameter count",
                sum(p.numel() for p in mine.parameters())
                == sum(p.numel() for p in theirs.parameters()),
                f"{sum(p.numel() for p in mine.parameters()):,}")
    mine.network.load_state_dict(theirs.state_dict())

    observation = torch.rand(4, 84, 84, device=device)
    report.check(f"{label}: forward (unbatched)", mine(observation), theirs(observation))
    batched = torch.rand(3, 4, 84, 84, device=device)
    report.check(f"{label}: forward (batched == per-sample)",
                 mine(batched), torch.stack([theirs(row) for row in batched]))
    return mine, theirs


def check_obgd(report, ours, optim, device):
    """The optimizer is the algorithm; this is the load-bearing comparison."""
    torch.manual_seed(5)
    shapes = [(7, 3), (7,), (2, 4, 3, 3)]
    theirs_params = [torch.nn.Parameter(torch.randn(*shape, device=device)) for shape in shapes]
    ours_params = [torch.nn.Parameter(p.detach().clone()) for p in theirs_params]

    their_optimizer = optim.ObGD(theirs_params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0)
    our_optimizer = ours.ObGD(ours_params, num_streams=1, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0)

    for step, (delta, reset) in enumerate(
        [(2.5, False), (-0.3, False), (0.05, False), (17.0, True), (-4.0, False)]
    ):
        gradients = [torch.randn(*shape, device=device) for shape in shapes]
        for parameter, gradient in zip(theirs_params, gradients):
            parameter.grad = gradient.clone()
        their_optimizer.step(delta, reset=reset)

        our_step = our_optimizer.step(
            [gradient.unsqueeze(0) for gradient in gradients],
            torch.tensor([delta], device=device),
            torch.tensor([reset], device=device),
        )
        for index, (mine, theirs) in enumerate(zip(ours_params, theirs_params)):
            report.check(f"ObGD step {step} (delta={delta}) param {index}", mine, theirs)
        report.check(
            f"ObGD step {step} eligibility trace 0",
            our_optimizer.traces[0][0],
            their_optimizer.state[theirs_params[0]]["eligibility_trace"],
        )
        del our_step

    # The overshooting bound itself: a huge delta must shrink the step.
    torch.manual_seed(6)
    parameter = torch.nn.Parameter(torch.zeros(4, device=device))
    their_single = optim.ObGD([parameter], lr=1.0, gamma=0.0, lamda=0.0, kappa=2.0)
    parameter.grad = torch.ones(4, device=device)
    their_single.step(100.0, reset=False)
    ours_parameter = torch.nn.Parameter(torch.zeros(4, device=device))
    our_single = ours.ObGD([ours_parameter], num_streams=1, lr=1.0, gamma=0.0, lamda=0.0, kappa=2.0)
    our_single.step([torch.ones(1, 4, device=device)],
                    torch.tensor([100.0], device=device), torch.tensor([False], device=device))
    report.check("ObGD overshooting bound", ours_parameter, parameter)


def check_stream_q(report, device):
    ours = load_trainer("stream_q_atari")
    optim, sparse = load_upstream()

    check_sparse_init(report, ours, sparse, device)
    check_layer_norm(report, ours, device)
    check_obgd(report, ours, optim, device)
    mine, theirs = check_network(
        report, ours, sparse, device, "StreamQ net",
        lambda: ours.QNetwork(EnvStub(N_ACTIONS), HIDDEN, 0.9), N_ACTIONS)

    # --- the Q(lambda) update, as in StreamQ.update_params ---
    observation = torch.rand(4, 84, 84, device=device)
    next_observation = torch.rand(4, 84, 84, device=device)
    action = torch.tensor(2, device=device)
    reward, gamma = torch.tensor(1.5, device=device), 0.99

    for done in (False, True):
        done_mask = 0.0 if done else 1.0
        q_sa = theirs(observation)[action]
        max_next = torch.max(theirs(next_observation), dim=-1).values
        their_delta = reward + gamma * max_next * done_mask - q_sa

        our_q = mine(observation)[action]
        our_max_next = torch.max(mine(next_observation), dim=-1).values
        our_delta = reward + gamma * our_max_next * done_mask - our_q
        report.check(f"Q(lambda) TD error (done={done})", our_delta, their_delta)

    # The per-stream gradient of -Q(s, a) must equal a plain backward.
    theirs.zero_grad()
    (-theirs(observation)[action]).backward()
    per_stream = ours.make_per_stream_grad_fn(mine)
    parameters = {name: p.detach() for name, p in mine.named_parameters()}
    batched = per_stream(parameters, observation.unsqueeze(0),
                         F.one_hot(action, N_ACTIONS).float().unsqueeze(0))
    ok = True
    for (name, _), their_parameter in zip(mine.named_parameters(), theirs.parameters()):
        ok &= report.check(f"grad(-Q) {name}", batched[name][0], their_parameter.grad)
    del ok

    # --- hyperparameters, read out of upstream's own argparse ---
    defaults = upstream_defaults("stream_q_atari.py")
    args = ours.Args()
    mapping = {
        "lr": args.learning_rate, "gamma": args.gamma, "lamda": args.lamda,
        "kappa_value": args.kappa, "epsilon_target": args.end_e,
        "epsilon_start": args.start_e, "exploration_fraction": args.exploration_fraction,
        "total_steps": args.total_timesteps,
    }
    bad = {name: (value, defaults[name]) for name, value in mapping.items()
           if name in defaults and float(value) != float(defaults[name])}
    missing = [name for name in mapping if name not in defaults]
    report.note(f"stream_q_atari.py argparse ({len(mapping)} fields)", not bad and not missing,
                (str(bad) if bad else "") + (f" missing={missing}" if missing else ""))
    report.note("sparsity 0.9 / hidden 256",
                args.sparsity == 0.9 and args.hidden_size == 256)
    report.note("single stream by default", args.num_envs == 1)


def check_stream_ac(report, device):
    ours = load_trainer("stream_ac_atari")
    optim, sparse = load_upstream()

    mine_policy, theirs_policy = check_network(
        report, ours, sparse, device, "StreamAC policy",
        lambda: ours.StreamNetwork(N_ACTIONS), N_ACTIONS)
    mine_value, theirs_value = check_network(
        report, ours, sparse, device, "StreamAC value",
        lambda: ours.StreamNetwork(1), 1)

    observation = torch.rand(4, 84, 84, device=device)
    action = torch.tensor(3, device=device)
    entropy_coeff = 0.01

    for sign in (1.0, -1.0):
        # Upstream: (-log pi(a|s) - c * H(pi) * sign(delta)).backward()
        theirs_policy.zero_grad()
        probs = F.softmax(theirs_policy(observation), dim=-1)
        distribution = torch.distributions.Categorical(probs)
        log_prob = -(distribution.log_prob(action)).sum()
        entropy = -entropy_coeff * distribution.entropy().sum() * sign
        (log_prob + entropy).backward()

        per_stream = ours.make_per_stream_policy_grad_fn(mine_policy)
        parameters = {name: p.detach() for name, p in mine_policy.named_parameters()}
        batched = per_stream(parameters, observation.unsqueeze(0),
                             F.one_hot(action, N_ACTIONS).float().unsqueeze(0),
                             torch.tensor([sign], device=device), entropy_coeff)
        for (name, _), their_parameter in zip(mine_policy.named_parameters(),
                                              theirs_policy.parameters()):
            report.check(f"policy grad sign(delta)={sign:+.0f} {name}",
                         batched[name][0], their_parameter.grad)

    # Upstream: (-v_s).backward()
    theirs_value.zero_grad()
    (-theirs_value(observation)).backward()
    per_stream_value = ours.make_per_stream_value_grad_fn(mine_value)
    parameters = {name: p.detach() for name, p in mine_value.named_parameters()}
    batched = per_stream_value(parameters, observation.unsqueeze(0))
    for (name, _), their_parameter in zip(mine_value.named_parameters(), theirs_value.parameters()):
        report.check(f"value grad {name}", batched[name][0], their_parameter.grad)

    source = open(os.path.join(upstream("streaming-drl"), "stream_ac_discrete_atari.py")).read()
    defaults = upstream_defaults("stream_ac_discrete_atari.py")
    args = ours.Args()
    mapping = {
        "lr": args.learning_rate, "gamma": args.gamma, "lamda": args.lamda,
        "kappa_policy": args.kappa_policy, "kappa_value": args.kappa_value,
        "entropy_coeff": args.entropy_coeff,
    }
    bad = {name: (value, defaults[name]) for name, value in mapping.items()
           if name in defaults and float(value) != float(defaults[name])}
    missing = [name for name in mapping if name not in defaults]
    report.note(f"stream_ac_discrete_atari.py argparse ({len(mapping)} fields)",
                not bad and not missing,
                (str(bad) if bad else "") + (f" missing={missing}" if missing else ""))
    report.note("two separate trunks", "network_policy" in source and "network_value" in source)
    report.note("trace reset on termination only",
                "reset=done" in source and "is_nongreedy" not in source)


def run(device, seed):
    torch.manual_seed(seed)
    report = Report(f"streaming-drl vs mohmdelsayed/streaming-drl [{device}]", tolerance=1e-5)
    check_stream_q(report, device)
    check_stream_ac(report, device)
    return report.print()


def main():
    devices, seed = parse_devices()
    results = [run(device, seed) for device in devices]
    print()
    if all(results):
        print("stream Q(lambda) / AC(lambda): CONFIRMED against mohmdelsayed/streaming-drl on "
              + ", ".join(devices))
        return 0
    print("streaming-drl: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
