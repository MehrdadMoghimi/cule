"""Diff cleanrl/ppo_scale_atari.py against openai/ppo-ewma.

The RLJ paper implements no new loss of its own: its analysis instrument is
PPO-EWMA from Hilton et al. 2022, whose official implementation is
openai/ppo-ewma (MIT). So the two pieces of that machinery this port carries --
the debiased EWMA over the policy weights, and the decoupled clipped surrogate --
are diffed here against the real `ppg.EwmaModel` and the real
`ppo.compute_losses`, executed from the checkout rather than transcribed.

The upstream package is built around procgen, gym3 and MPI, none of which are
needed by the two functions under test and none of which install cleanly today.
`_import_upstream` therefore stubs those three imports and loads `ppg.py` and
`ppo.py` as submodules of a hand-built parent package, so the reference code that
actually runs is unmodified upstream source. The stubs are asserted to be
untouched by the comparison: if any of them were reached, the classes they define
are empty and would raise rather than silently return a wrong number.

What is deliberately NOT claimed to match: upstream's value loss has no 0.5 and
no clipping, its advantage normalisation is an EWMA over iterations, and its
`pi_kl` term is a penalty this port does not use. Those belong to upstream's PPO,
not to PPO-EWMA, and this fork keeps `ppo_atari.py`'s versions. Only the two
components above are compared.
"""

import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from common import Report, load_trainer, parse_devices, upstream

BATCH, N_ACTIONS, N_FEATURES = 64, 6, 5


def _import_upstream():
    """Load upstream `ppg.py` and `ppo.py` without their procgen/MPI dependencies."""
    root = upstream("ppo-ewma")

    mpi = types.ModuleType("mpi4py")
    mpi.MPI = types.SimpleNamespace(COMM_WORLD=None, Comm=type("Comm", (), {}))
    sys.modules.setdefault("mpi4py", mpi)
    sys.modules.setdefault("mpi4py.MPI", mpi.MPI)

    gym3 = types.ModuleType("gym3")
    gym3.Wrapper = type("Wrapper", (), {})
    gym3_types = types.ModuleType("gym3.types")
    for name in ("Discrete", "Real", "TensorType"):
        setattr(gym3_types, name, type(name, (), {}))
    gym3.types = gym3_types
    sys.modules.setdefault("gym3", gym3)
    sys.modules.setdefault("gym3.types", gym3_types)

    # A parent package with a real __path__ but no __init__, so relative imports
    # resolve while `ppo_ewma/__init__.py` (which pulls in procgen) never runs.
    package = types.ModuleType("ppo_ewma")
    package.__path__ = [os.path.join(root, "ppo_ewma")]
    sys.modules["ppo_ewma"] = package

    import importlib

    return importlib.import_module("ppo_ewma.ppg"), importlib.import_module("ppo_ewma.ppo")


class TinyPolicy(nn.Module):
    """A stand-in for upstream's PpoModel with the interface `compute_losses` calls."""

    def __init__(self, n_features=N_FEATURES, n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Linear(n_features, 8)
        self.actor = nn.Linear(8, n_actions)
        self.critic = nn.Linear(8, 1)

    def logits(self, ob):
        return self.actor(torch.tanh(self.trunk(ob)))

    def forward(self, ob, first=None, state_in=None):
        hidden = torch.tanh(self.trunk(ob))
        distribution = torch.distributions.Categorical(logits=self.actor(hidden))
        return distribution, self.critic(hidden).squeeze(-1), {}, None


def check_ewma(report, ppg, ours, device, generator):
    """The debiased EWMA recursion, step for step against `ppg.EwmaModel`."""
    for com in (0.0, 8.0, 256.0, 4096.0):
        decay = ours.com_to_decay(com)
        report.check(f"com {com:>6.0f} -> decay", np.array(decay),
                     np.array(com / (com + 1.0)), tolerance=0.0)
        report.note(f"com {com:>6.0f} round-trips", abs(ours.decay_to_com(decay) - com) < 1e-6)

        torch.manual_seed(0)
        theirs_model = TinyPolicy().to(device).double()
        ours_model = TinyPolicy().to(device).double()
        ours_model.load_state_dict(theirs_model.state_dict())

        theirs_ewma = ppg.EwmaModel(theirs_model, ewma_decay=decay)
        ours_ewma = ours.ProximalPolicy(ours_model, decay).to(device)

        # Walk both models along the same random parameter trajectory and refresh
        # after every step, exactly as the minibatch loop does.
        for _ in range(12):
            with torch.no_grad():
                for a, b in zip(theirs_model.parameters(), ours_model.parameters()):
                    step = torch.randn(a.shape, generator=generator, dtype=torch.float64).to(device)
                    a.add_(0.05 * step)
                    b.copy_(a)
            theirs_ewma.update()
            ours_ewma.update(ours_model)

        report.check(f"com {com:>6.0f} total_weight", np.array(ours_ewma.total_weight),
                     np.array(theirs_ewma.total_weight), tolerance=1e-12)
        for name, ours_p, theirs_p in zip(
            [n for n, _ in ours_ewma.model.named_parameters()],
            ours_ewma.model.parameters(),
            theirs_ewma.model_ewma.parameters(),
        ):
            report.check(f"com {com:>6.0f} ewma {name}", ours_p, theirs_p, tolerance=1e-12)

    # The bias correction is the whole point: assert it differs from the naive
    # recursion early on, so a regression to `beta*p + (1-beta)*p_new` is caught.
    decay = ours.com_to_decay(4096.0)
    torch.manual_seed(0)
    model = TinyPolicy().to(device).double()
    ewma = ours.ProximalPolicy(model, decay).to(device)
    naive = [p.detach().clone() for p in model.parameters()]
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        ewma.update(model)
        naive = [decay * n + (1 - decay) * p.detach() for n, p in zip(naive, model.parameters())]
    gap = max(float((a - b).abs().max()) for a, b in zip(ewma.model.parameters(), naive))
    report.note("debiased EWMA is not the naive recursion", gap > 1e-3,
                f"max|debiased - naive| = {gap:.3e} after 5 steps at COM 4096")


def check_surrogate(report, ppo_module, ours, device, generator):
    """The decoupled clipped surrogate against the real `ppo.compute_losses`."""
    torch.manual_seed(1)
    model = TinyPolicy().to(device).double()
    proximal_model = TinyPolicy().to(device).double()
    with torch.no_grad():  # a genuinely lagged proximal policy, not a copy
        for p, q in zip(proximal_model.parameters(), model.parameters()):
            p.copy_(q + 0.05 * torch.randn(q.shape, generator=generator, dtype=torch.float64).to(device))

    ob = torch.randn(BATCH, N_FEATURES, generator=generator, dtype=torch.float64).to(device)
    ac = torch.randint(0, N_ACTIONS, (BATCH,), generator=generator).to(device)
    adv = torch.randn(BATCH, generator=generator, dtype=torch.float64).to(device)
    # A behaviour policy far enough away that importance-ratio clipping bites.
    behaviour_logp = (
        torch.distributions.Categorical(
            logits=model.logits(ob).detach()
            + 2.5 * torch.randn(BATCH, N_ACTIONS, generator=generator, dtype=torch.float64).to(device)
        ).log_prob(ac).detach()
    )

    newlogp = torch.distributions.Categorical(logits=model.logits(ob)).log_prob(ac)
    proximal_logp = torch.distributions.Categorical(logits=proximal_model.logits(ob)).log_prob(ac).detach()

    class EwmaStub(nn.Module):
        """Stands in for upstream's EwmaModel: returns the lagged distribution."""

        def forward(self, ob, first=None, state_in=None):
            return proximal_model(ob)

    for clip in (0.1, 0.2, 0.0):
        for imp_samp_max in (100.0, 0.0):
            label = f"clip {clip} imp_samp_max {imp_samp_max:g}"
            _, theirs_diags = ppo_module.compute_losses(
                model=model, model_ewma=EwmaStub(), ob=ob, ac=ac, first=None,
                logp=behaviour_logp, rec_logp=behaviour_logp, vtarg=torch.zeros(BATCH, dtype=torch.float64).to(device),
                adv=adv, state_in=None, clip_param=clip, vfcoef=0.0, entcoef=0.0,
                kl_penalty=0.0, imp_samp_max=imp_samp_max,
            )
            ours_loss, ours_logratio = ours.policy_surrogate(
                adv, newlogp, behaviour_logp, proximal_logp, clip, imp_samp_max)
            report.check(f"decoupled surrogate [{label}]", ours_loss, theirs_diags["pg"])
            report.check(f"proximal logratio [{label}]", ours_logratio, newlogp - proximal_logp)

            # ... and its gradient, which is what actually trains. Only the
            # policy path: the surrogate never reaches the critic head.
            policy_params = list(model.trunk.parameters()) + list(model.actor.parameters())
            theirs_grad = torch.autograd.grad(theirs_diags["pg"], policy_params, retain_graph=True)
            ours_grad = torch.autograd.grad(ours_loss, policy_params, retain_graph=True)
            report.check(f"surrogate grad [{label}]",
                         torch.cat([g.reshape(-1) for g in ours_grad]),
                         torch.cat([g.reshape(-1) for g in theirs_grad]))

    # With no proximal policy our path is PPO's, which upstream reproduces by
    # passing model_ewma=None so rec_logp stays the behaviour logp. Clipping then
    # happens in log space upstream and in linear space here; the two agree to
    # rounding, and the linear form is what makes the reduction to `ppo_atari.py`
    # bit-exact (see tests/test_ppo_scale_equivalence.py).
    _, theirs_diags = ppo_module.compute_losses(
        model=model, model_ewma=None, ob=ob, ac=ac, first=None,
        logp=behaviour_logp, rec_logp=behaviour_logp,
        vtarg=torch.zeros(BATCH, dtype=torch.float64).to(device), adv=adv, state_in=None,
        clip_param=0.2, vfcoef=0.0, entcoef=0.0, kl_penalty=0.0, imp_samp_max=0.0,
    )
    ours_loss, _ = ours.policy_surrogate(adv, newlogp, behaviour_logp, None, 0.2, 0.0)
    report.check("plain-PPO surrogate (no proximal policy)", ours_loss, theirs_diags["pg"], tolerance=1e-9)


def run(device, seed):
    ppg, ppo_module = _import_upstream()
    ours = load_trainer("ppo_scale_atari")
    report = Report(f"PPO-scale vs openai/ppo-ewma [{device}]", tolerance=1e-11)
    generator = torch.Generator().manual_seed(seed)

    check_ewma(report, ppg, ours, device, generator)
    check_surrogate(report, ppo_module, ours, device, generator)
    return report.print()


def main():
    devices, seed = parse_devices()
    return 0 if all([run(device, seed) for device in devices]) else 1


if __name__ == "__main__":
    sys.exit(main())
