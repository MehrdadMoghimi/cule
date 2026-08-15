"""BTR: numerical equivalence against the official implementation.

`tests/reference/btr_reference.py` is a transcription of the upstream
(MIT-licensed) `networks.py` / `Agent.py`, so passing these tests means
`cleanrl/btr_atari.py` reproduces upstream's encoder and Munchausen-IQN loss
exactly, not merely approximately.
"""

import pytest
import torch

from conftest import DiscreteEnvStub, load_trainer
from reference import btr_reference

N_ACTIONS = 6
MODEL_SIZE = 2
NUM_TAU = 8
N_COS = 64
LINEAR_SIZE = 512


def _build_pair(seed=0, spectral=True, noisy=True):
    """Return (ours, reference) carrying identical parameters."""
    torch.manual_seed(seed)
    btr = load_trainer('btr_atari')
    ours = btr.QNetwork(
        DiscreteEnvStub(N_ACTIONS),
        n_cos=N_COS,
        n_policy_taus=NUM_TAU,
        model_size=MODEL_SIZE,
        maxpool_size=6,
        linear_size=LINEAR_SIZE,
        noisy_std=0.5,
        spectral=spectral,
    )
    torch.manual_seed(seed)
    theirs = btr_reference.ImpalaCNNLargeIQN(
        4, N_ACTIONS, model_size=MODEL_SIZE, spectral=spectral, device='cpu',
        noisy=noisy, maxpool=True, num_tau=NUM_TAU, maxpool_size=6,
        dueling=True, linear_size=LINEAR_SIZE, ncos=N_COS,
    )

    # The trunk and cosine embedding share module paths; only the dueling head
    # is namespaced differently.
    ours_state = ours.state_dict()
    remapped = {}
    for key, value in theirs.state_dict().items():
        our_key = (key
                   .replace('dueling.value_branch.', 'value_head.')
                   .replace('dueling.advantage_branch.', 'advantage_head.'))
        assert our_key in ours_state, f'no counterpart for reference key {key}'
        remapped[our_key] = value.clone()
    # `cos_multipliers` is a registered buffer here but a plain attribute
    # upstream (`self.pis`), so it is the one legitimate key difference.
    extra = set(ours_state) - set(remapped)
    assert extra == {'cos_multipliers'}, (
        'parameter sets differ: '
        f'{sorted(extra)} / {sorted(set(remapped) - set(ours_state))}')
    remapped['cos_multipliers'] = ours_state['cos_multipliers']
    ours.load_state_dict(remapped)
    torch.testing.assert_close(
        ours.cos_multipliers, theirs.pis.reshape(-1), rtol=0, atol=0)

    # Spectral norm runs a power iteration during training-mode forwards, which
    # would desynchronise the two modules after the first call.  eval() uses the
    # cached (and now identical) singular vectors instead.
    ours.eval()
    theirs.eval()
    return ours, theirs


def test_parameter_shapes_match_reference():
    ours, theirs = _build_pair()
    assert ours.feature_dim == theirs.conv_out_size == 1152 * MODEL_SIZE
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in theirs.parameters())


@pytest.mark.parametrize('spectral', [True, False])
def test_encoder_matches_reference(spectral):
    ours, theirs = _build_pair(spectral=spectral)
    observations = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8).float()

    ours_features = ours.features(observations)
    theirs_features = theirs.pool(theirs.conv(observations.float() / 255)).flatten(1)

    torch.testing.assert_close(ours_features, theirs_features, rtol=0, atol=0)


def test_quantile_head_matches_reference():
    ours, theirs = _build_pair()
    observations = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8).float()
    taus = torch.rand(3, NUM_TAU)

    ours_quantiles = ours.quantile_values(ours.features(observations), taus)
    theirs_quantiles, _ = theirs(observations, taus=taus.unsqueeze(-1))

    torch.testing.assert_close(ours_quantiles, theirs_quantiles, rtol=1e-6, atol=1e-6)


def test_advantages_only_matches_reference():
    ours, theirs = _build_pair()
    observations = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8).float()
    taus = torch.rand(3, NUM_TAU)

    ours_advantages = ours.quantile_values(ours.features(observations), taus, advantages_only=True)
    theirs_advantages, _ = theirs(observations, advantages_only=True, taus=taus.unsqueeze(-1))

    torch.testing.assert_close(ours_advantages, theirs_advantages, rtol=1e-6, atol=1e-6)
    # argmax over the advantage branch must agree with argmax over full Q.
    full, _ = theirs(observations, taus=taus.unsqueeze(-1))
    assert torch.equal(ours_advantages.mean(1).argmax(1), full.mean(1).argmax(1))


def test_noisy_layer_init_matches_reference():
    """sigma_0/sqrt(fan_in) for both weight and bias, and noise off at init."""
    btr = load_trainer('btr_atari')
    torch.manual_seed(3)
    ours = btr.FactorizedNoisyLinear(64, 32, 0.5)
    torch.manual_seed(3)
    theirs = btr_reference.FactorizedNoisyLinear(64, 32, 0.5)

    for name in ('weight_mu', 'bias_mu', 'weight_sigma', 'bias_sigma'):
        torch.testing.assert_close(getattr(ours, name), getattr(theirs, name), rtol=0, atol=0)
    assert float(ours.weight_epsilon.abs().sum()) == 0.0
    assert float(ours.bias_epsilon.abs().sum()) == 0.0
    # This is the property that distinguishes it from rainbow_atari.py's
    # NoisyLinear, which scales the bias sigma by 1/sqrt(fan_out) instead.
    torch.testing.assert_close(
        ours.bias_sigma, torch.full_like(ours.bias_sigma, 0.5 / 64 ** 0.5), rtol=0, atol=1e-7)


def test_munchausen_iqn_loss_matches_reference():
    """The full learner update, tau draws and PER weights shared."""
    ours, theirs = _build_pair(seed=7)
    torch.manual_seed(11)

    batch_size = 5
    gamma, n_step = 0.997, 3
    entropy_tau, alpha, lo, kappa = 0.03, 0.9, -1.0, 1.0

    observations = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8).float()
    next_observations = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8).float()
    actions = torch.randint(0, N_ACTIONS, (batch_size, 1))
    rewards = torch.randn(batch_size, 1)
    dones = torch.randint(0, 2, (batch_size, 1)).bool()
    weights = torch.rand(batch_size, 1) + 0.1

    taus_online = torch.rand(batch_size, NUM_TAU)
    taus_target = torch.rand(batch_size, NUM_TAU)
    taus_policy_cur = torch.rand(batch_size, NUM_TAU)

    reference_loss, reference_priority = btr_reference.munchausen_iqn_loss(
        theirs, theirs, observations, actions.squeeze(1), rewards.squeeze(1),
        next_observations, dones.squeeze(1), weights.squeeze(1),
        gamma=gamma, n=n_step, entropy_tau=entropy_tau, alpha=alpha, lo=lo,
        num_tau=NUM_TAU, taus_online=taus_online.unsqueeze(-1),
        taus_target=taus_target.unsqueeze(-1), taus_policy_cur=taus_policy_cur.unsqueeze(-1),
    )

    # ---- the arithmetic as cleanrl/btr_atari.py performs it ----
    q_network = target_network = ours
    gamma_n = gamma ** n_step
    batch_indices = torch.arange(batch_size)
    with torch.no_grad():
        next_features = target_network.features(next_observations)
        next_z = target_network.quantile_values(next_features, taus_target)
        next_q = next_z.mean(1)
        current_q = q_network.quantile_values(
            q_network.features(observations), taus_policy_cur).mean(1)
        tau_log_pi_current = load_trainer('btr_atari').scaled_log_softmax(current_q, entropy_tau)
        munchausen_bonus = alpha * tau_log_pi_current.gather(1, actions).clamp(lo, 0.0)
        tau_log_pi_next = load_trainer('btr_atari').scaled_log_softmax(next_q, entropy_tau)
        pi_next = torch.softmax(next_q / entropy_tau, dim=-1)
        soft_values = (pi_next.unsqueeze(1) * (next_z - tau_log_pi_next.unsqueeze(1))).sum(2)
        target_quantiles = rewards + munchausen_bonus + gamma_n * soft_values * (1 - dones.float())

    features = q_network.features(observations)
    z = q_network.quantile_values(features, taus_online)
    old_quantiles = z[batch_indices, :, actions.flatten()]

    u = target_quantiles.unsqueeze(1) - old_quantiles.unsqueeze(2)
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    rho = (taus_online.unsqueeze(-1) - (u.detach() < 0).float()).abs() * huber / kappa
    loss_per_sample = rho.mean(2).sum(1)
    loss = (loss_per_sample * weights.squeeze()).mean()
    priority = abs_u.mean(2).sum(1)

    torch.testing.assert_close(loss, reference_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(priority, reference_priority, rtol=1e-5, atol=1e-6)


def test_hyperparameters_match_paper():
    """Defaults in `Args` against BTR's published hyperparameter table."""
    btr = load_trainer('btr_atari')
    args = btr.Args()
    assert args.learning_rate == 1e-4
    assert args.batch_size == 256
    assert args.num_envs == 64
    assert args.gamma == 0.997
    assert args.n_step == 3
    assert args.buffer_size == 2 ** 20
    assert args.learning_starts == 200_000
    assert args.target_network_frequency == 500
    assert args.prioritized_replay_alpha == 0.2
    assert args.munchausen_alpha == 0.9
    assert args.munchausen_tau == 0.03
    assert args.munchausen_clip == -1.0
    assert args.n_taus == args.n_target_taus == 8
    assert args.n_cos == 64
    assert args.model_size == 2
    assert args.maxpool_size == 6
    assert args.linear_size == 512
    assert args.noisy_std == 0.5
    assert args.max_grad_norm == 10
    assert args.epsilon_decay_steps == 2_000_000
    assert args.end_e == 0.01
