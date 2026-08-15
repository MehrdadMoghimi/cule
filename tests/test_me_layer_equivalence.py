"""Assert the mean-expansion layer against the claims of arXiv:2606.29806.

"Accelerating Q-learning through Efficient Value-Sharing across Actions",
Nagarajan, Daley, White & Machado, ICML 2026.

No official implementation is published (the paper's footnote points at
github.com/prabhatnagarajan/me_layer, which 404s as of 2026-08-08), so these
tests state the paper's propositions independently rather than diffing against
the authors' code. The one exception is `reference_me_layer`, which is the
twelve-line implementation printed verbatim in the paper's Appendix D.
"""

import pytest
import torch

from conftest import DiscreteEnvStub, device_params, load_trainer

TRAINERS = ["ibdqn_atari", "ibdqn_atari_torchcompile"]


def reference_me_layer(vec, mean_scaling_coefficient):
    """The paper's Appendix D listing, transcribed.

    `scale` is deliberately built without a dtype, exactly as the listing does:
    `torch.tensor(1 + k)` lands in the default float32, so a k that is not
    binary-representable (0.1, say) is rounded before it is ever used. Forcing
    float64 here instead would make this reference disagree with both the paper
    and the port at the 1e-8 level.
    """
    scale = torch.tensor(1 + mean_scaling_coefficient, device=vec.device)
    mean = vec.mean(dim=-1, keepdim=True)
    residual = vec - mean
    return scale * mean + residual


class BoxEnvStub(DiscreteEnvStub):
    def __init__(self, n_actions=6):
        super().__init__(n_actions)
        self.single_observation_space = type("Box", (), {"shape": (4, 84, 84)})()


@pytest.fixture(params=TRAINERS)
def trainer(request):
    return load_trainer(request.param)


# --- the layer itself ------------------------------------------------------


@pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 3.0, 6.0, 18.0])
def test_matches_the_papers_reference_listing(trainer, k):
    torch.manual_seed(0)
    z = torch.randn(64, 6, dtype=torch.float64)
    layer = trainer.MeanExpansionLayer(k).double()
    assert torch.allclose(layer(z), reference_me_layer(z, k), atol=0, rtol=0)


@pytest.mark.parametrize("k", [0.0, 0.5, 4.0, 6.0])
def test_equals_the_matrix_form(trainer, k):
    """q = (I + (k/n) J) z, Equation 5."""
    torch.manual_seed(1)
    n = 6
    z = torch.randn(32, n, dtype=torch.float64)
    identity = torch.eye(n, dtype=torch.float64)
    ones = torch.ones(n, n, dtype=torch.float64)
    expected = z @ (identity + (k / n) * ones).T
    assert torch.allclose(trainer.MeanExpansionLayer(k).double()(z), expected, atol=1e-12)


def test_k_zero_is_the_identity(trainer):
    """"IB-DQN generalizes DQN as special case when k = 0" (Section 4.1)."""
    torch.manual_seed(2)
    z = torch.randn(128, 18, dtype=torch.float64)
    assert torch.allclose(trainer.MeanExpansionLayer(0.0).double()(z), z, atol=1e-12)


def test_negative_k_is_rejected(trainer):
    with pytest.raises(ValueError):
        trainer.MeanExpansionLayer(-1.0)


def test_layer_has_no_learnable_parameters(trainer):
    """"introduces no additional learnable parameters to the model" (Section 4.1)."""
    layer = trainer.MeanExpansionLayer(4.0)
    assert list(layer.parameters()) == []
    assert list(layer.buffers()) != []


@pytest.mark.parametrize("k", [0.5, 1.0, 4.0])
def test_mean_component_is_scaled_by_k_plus_one(trainer, k):
    """"the mean-expansion layer scales the mean component by a factor of k + 1"."""
    torch.manual_seed(3)
    z = torch.randn(32, 4, dtype=torch.float64)
    q = trainer.MeanExpansionLayer(k).double()(z)
    assert torch.allclose(q.mean(-1), (k + 1) * z.mean(-1), atol=1e-12)


@pytest.mark.parametrize("k", [0.1, 1.0, 4.0, 6.0])
def test_implicit_baseline_proposition_3(trainer, k):
    """b = k * mu_z = sum(q_i) / (n + n/k), Equation 7."""
    torch.manual_seed(4)
    n = 6
    z = torch.randn(32, n, dtype=torch.float64)
    q = trainer.MeanExpansionLayer(k).double()(z)
    baseline = k * z.mean(-1)
    assert torch.allclose(baseline, q.sum(-1) / (n + n / k), atol=1e-12)


def test_k_equals_n_gives_the_norm_minimizing_baseline(trainer):
    """Proposition 1: b* = sum(q_i) / (n + 1), the argmin of ||u(q, b)||^2."""
    torch.manual_seed(5)
    n = 6
    z = torch.randn(64, n, dtype=torch.float64)
    q = trainer.MeanExpansionLayer(float(n)).double()(z)
    baseline = float(n) * z.mean(-1)
    assert torch.allclose(baseline, q.sum(-1) / (n + 1), atol=1e-12)

    # And it really is the minimizer of the baseline-residual norm.
    candidates = torch.linspace(-3, 3, 601, dtype=torch.float64)
    row = q[0]
    norms = ((row.unsqueeze(0) - candidates.unsqueeze(1)) ** 2).sum(1) + candidates**2
    assert abs(candidates[norms.argmin()] - baseline[0]) < 0.02


@pytest.mark.parametrize("k", [0.5, 1.0, 4.0])
def test_residual_has_lower_norm_than_q(trainer, k):
    """"||z||^2 < ||u(q, k mu_z)||^2 < ||q||^2" (Section 3.4), for non-zero mean."""
    torch.manual_seed(6)
    n = 6
    z = torch.randn(256, n, dtype=torch.float64) + 2.0  # shifted so the mean is non-zero
    q = trainer.MeanExpansionLayer(k).double()(z)
    baseline = k * z.mean(-1)
    residual_norm = (z**2).sum(-1)
    baseline_residual_norm = ((q - baseline.unsqueeze(-1)) ** 2).sum(-1) + baseline**2
    q_norm = (q**2).sum(-1)
    assert (residual_norm < baseline_residual_norm).all()
    assert (baseline_residual_norm < q_norm).all()


@pytest.mark.parametrize("k", [0.0, 0.5, 1.0, 4.0, 17.0])
def test_layer_is_invertible_with_condition_number_k_plus_one(trainer, k):
    """"the condition number of M_k is k + 1" (Section 3.5)."""
    n = 6
    matrix = torch.eye(n, dtype=torch.float64) + (k / n) * torch.ones(n, n, dtype=torch.float64)
    assert torch.linalg.cond(matrix).item() == pytest.approx(k + 1, rel=1e-9)
    torch.manual_seed(7)
    q = torch.randn(16, n, dtype=torch.float64)
    z = torch.linalg.solve(matrix, q.T).T
    assert torch.allclose(trainer.MeanExpansionLayer(k).double()(z), q, atol=1e-10)


@pytest.mark.parametrize("k", [0.5, 4.0])
def test_action_gaps_and_argmax_are_preserved(trainer, k):
    """The layer "preserv[es] their relative differences" (Section 4)."""
    torch.manual_seed(8)
    z = torch.randn(64, 6, dtype=torch.float64)
    q = trainer.MeanExpansionLayer(k).double()(z)
    gaps_z = z.unsqueeze(1) - z.unsqueeze(2)
    gaps_q = q.unsqueeze(1) - q.unsqueeze(2)
    assert torch.allclose(gaps_z, gaps_q, atol=1e-12)
    assert torch.equal(z.argmax(-1), q.argmax(-1))


@pytest.mark.parametrize("k", [0.5, 1.0, 4.0])
def test_gradient_spreads_the_td_error_over_every_action(trainer, k):
    """Equations 8 and 9: the taken action gets 1 + k/n, the others k/n."""
    n = 4
    z = torch.zeros(1, n, dtype=torch.float64, requires_grad=True)
    q = trainer.MeanExpansionLayer(k).double()(z)
    q[0, 1].backward()
    expected = torch.full((n,), k / n, dtype=torch.float64)
    expected[1] += 1.0
    assert torch.allclose(z.grad[0], expected, atol=1e-12)


# --- wiring into the trainers ---------------------------------------------


def test_default_k_is_the_action_count(trainer):
    """"we use k = n unless stated otherwise" (Section 3.4)."""
    assert trainer.resolve_mean_scaling_coefficient(-1.0, 18) == 18.0
    assert trainer.resolve_mean_scaling_coefficient(3.0, 18) == 3.0
    assert trainer.Args().mean_scaling_coefficient == -1.0


def test_ibdqn_network_ends_with_the_layer(trainer):
    network = trainer.QNetwork(BoxEnvStub(6)).network
    assert isinstance(network[-1], trainer.MeanExpansionLayer)
    assert network[-1].scale.item() == pytest.approx(7.0)  # k = n = 6


@pytest.mark.parametrize("device", device_params())
def test_ibdqn_matches_dqn_plus_the_layer(device):
    """IB-DQN is DQN with one extra layer, and nothing else moved."""
    ours = load_trainer("ibdqn_atari")
    baseline = load_trainer("dqn_atari")
    env = BoxEnvStub(6)

    torch.manual_seed(9)
    ib_network = ours.QNetwork(env, mean_scaling_coefficient=6.0).to(device)
    torch.manual_seed(9)
    plain = baseline.QNetwork(env).to(device)

    observations = torch.randint(0, 255, (5, 4, 84, 84), device=device).float()
    with torch.no_grad():
        expected = reference_me_layer(plain(observations), 6.0)
        assert torch.allclose(ib_network(observations), expected, atol=1e-5)


def test_ibdqn_with_k_zero_reproduces_dqn(trainer):
    env = BoxEnvStub(6)
    torch.manual_seed(10)
    ib_network = trainer.QNetwork(env, mean_scaling_coefficient=0.0).double()
    baseline = load_trainer("dqn_atari")
    torch.manual_seed(10)
    plain = baseline.QNetwork(env).double()
    observations = torch.randint(0, 255, (3, 4, 84, 84)).double()
    with torch.no_grad():
        assert torch.allclose(ib_network(observations), plain(observations), atol=1e-10)


# --- IB-IQN ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["iqn_atari", "iqn_atari_torchcompile"])
def test_iqn_is_untouched_by_default(name):
    """The flag defaults to 0, and at 0 the layer is not built at all, so the
    dopamine-confirmed IQN path stays bit-identical rather than merely close."""
    module = load_trainer(name)
    assert module.Args().mean_scaling_coefficient == 0.0
    assert module.QNetwork(BoxEnvStub(6)).me_layer is None


@pytest.mark.parametrize("name", ["iqn_atari", "iqn_atari_torchcompile"])
def test_ib_iqn_applies_the_layer_over_actions(name):
    """"IB-IQN refers to IQN where an ME layer is added at the end" (Section 6.2)."""
    module = load_trainer(name)
    env = BoxEnvStub(6)
    torch.manual_seed(11)
    plain = module.QNetwork(env).double()
    torch.manual_seed(11)
    ib_network = module.QNetwork(env, mean_scaling_coefficient=-1).double()
    assert ib_network.me_layer.scale.item() == pytest.approx(7.0)

    observations = torch.randint(0, 255, (3, 4, 84, 84)).double()
    taus = torch.rand(3, 8, dtype=torch.float64)
    with torch.no_grad():
        plain_quantiles = plain.quantile_values(plain.features(observations), taus)
        ib_quantiles = ib_network.quantile_values(ib_network.features(observations), taus)
    assert ib_quantiles.shape == plain_quantiles.shape  # distribution untouched
    assert torch.allclose(ib_quantiles, reference_me_layer(plain_quantiles, 6.0), atol=1e-10)
