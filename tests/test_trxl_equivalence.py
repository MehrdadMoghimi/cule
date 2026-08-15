"""PPO-TrXL: equivalence against CleanRL's `ppo_trxl`.

The reference (`~/cleanrl/cleanrl/ppo_trxl/ppo_trxl.py`, itself the CleanRL
contribution of MarcoMeter/episodic-transformer-memory-ppo) imports
`memory_gym`, `minigrid` and a local `pom_env`, so it cannot be imported. Its
transformer classes depend only on torch and einops, so they are extracted from
the source and executed — every module below is diffed against the authors' own
code running on shared weights.

Beyond that, the memory bookkeeping is checked as behaviour: the mask must hide
unwritten slots, the window must be the last `L` steps, gradients must not cross
timesteps, and the Atari memory bound must actually bound.
"""

import ast
import os
import textwrap

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import DiscreteEnvStub, load_trainer

TRAINER = 'ppo_trxl_atari'
UPSTREAM = os.path.join(os.path.expanduser('~'), 'cleanrl', 'cleanrl', 'ppo_trxl', 'ppo_trxl.py')


@pytest.fixture(scope='module')
def trxl():
    return load_trainer(TRAINER)


def _rearrange(tensor, pattern):
    """The two `einops.rearrange` calls the reference's transformer makes.

    Supplying them directly keeps the cross-check running without installing
    einops into the training environment; if einops *is* installed the real
    implementation is used instead.
    """
    if pattern == 'n -> n ()':
        return tensor.unsqueeze(-1)
    if pattern == 'd -> () d':
        return tensor.unsqueeze(0)
    raise AssertionError(f'unexpected rearrange pattern in upstream: {pattern!r}')


@pytest.fixture(scope='module')
def upstream():
    """Extract the reference's transformer stack and run it."""
    if not os.path.exists(UPSTREAM):
        pytest.skip('cleanrl ppo_trxl not present at ~/cleanrl')
    try:
        from einops import rearrange
    except ImportError:
        rearrange = _rearrange

    with open(UPSTREAM) as handle:
        source = handle.read()
    tree = ast.parse(source)
    namespace = {'nn': nn, 'torch': torch, 'np': np,
                 'rearrange': rearrange}
    wanted = {'layer_init', 'batched_index_select', 'PositionalEncoding',
              'MultiHeadAttention', 'TransformerLayer', 'Transformer'}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
    missing = wanted - set(namespace)
    assert not missing, missing
    return namespace


# ---------------------------------------------------------------------------
# module-for-module, against the reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dim', [64, 384])
def test_positional_encoding_matches_upstream(trxl, upstream, dim):
    ours = trxl.PositionalEncoding(dim)
    theirs = upstream['PositionalEncoding'](dim)
    assert torch.allclose(ours.inv_freqs, theirs.inv_freqs, rtol=0, atol=1e-12)
    for seq_len in (1, 7, 119):
        assert torch.allclose(ours(seq_len), theirs(seq_len), rtol=0, atol=1e-6)


def test_positional_encoding_counts_backwards_from_the_present(trxl):
    """`arange(seq_len - 1, -1, -1)`: table row 0 is the oldest slot."""
    encoding = trxl.PositionalEncoding(4)
    table = encoding(3)
    # The last row corresponds to seq value 0 -> sin 0, cos 1.
    assert torch.allclose(table[-1, :2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(table[-1, 2:], torch.ones(2), atol=1e-6)
    assert not torch.allclose(table[0, :2], torch.zeros(2), atol=1e-6)


@pytest.mark.parametrize('seed', range(3))
def test_multi_head_attention_matches_upstream(trxl, upstream, seed):
    torch.manual_seed(seed)
    dim, heads, batch, memory = 64, 4, 5, 11
    ours = trxl.MultiHeadAttention(dim, heads).double()
    theirs = upstream['MultiHeadAttention'](dim, heads).double()
    theirs.load_state_dict(ours.state_dict())

    generator = torch.Generator().manual_seed(seed + 100)
    values = torch.randn(batch, memory, dim, generator=generator, dtype=torch.float64)
    query = torch.randn(batch, 1, dim, generator=generator, dtype=torch.float64)
    mask = (torch.rand(batch, memory, generator=generator) < 0.7).double()
    mask[:, 0] = 1.0  # never fully masked

    got, got_weights = ours(values, values, query, mask)
    want, want_weights = theirs(values, values, query, mask)
    assert torch.allclose(got, want, rtol=0, atol=1e-12)
    assert torch.allclose(got_weights, want_weights, rtol=0, atol=1e-12)


def test_energy_is_scaled_by_embed_dim_not_head_size(trxl):
    """A deviation from Vaswani et al. that the reference makes; keep it.

    With `dim=64, heads=4` the two scalings differ by a factor of 2, which
    changes every attention distribution the model ever produces.
    """
    torch.manual_seed(0)
    dim, heads = 64, 4
    attention = trxl.MultiHeadAttention(dim, heads).double()
    values = torch.randn(1, 6, dim, dtype=torch.float64)
    query = torch.randn(1, 1, dim, dtype=torch.float64)

    _, weights = attention(values, values, query, None)

    head_size = dim // heads
    v = values.reshape(1, 6, heads, head_size)
    q = query.reshape(1, 1, heads, head_size)
    energy = torch.einsum('nqhd,nkhd->nhqk', [attention.queries(q), attention.keys(v)])
    by_embed = torch.softmax(energy / dim**0.5, dim=3)
    by_head = torch.softmax(energy / head_size**0.5, dim=3)

    assert torch.allclose(weights, by_embed, rtol=0, atol=1e-12)
    assert not torch.allclose(weights, by_head, rtol=1e-4, atol=1e-4)


def test_fully_masked_row_does_not_produce_nan(trxl):
    """Step 0 of an episode attends to nothing; `-1e20` keeps softmax finite."""
    torch.manual_seed(0)
    attention = trxl.MultiHeadAttention(64, 4).double()
    values = torch.randn(2, 5, 64, dtype=torch.float64)
    query = torch.randn(2, 1, 64, dtype=torch.float64)
    mask = torch.zeros(2, 5, dtype=torch.float64)

    out, weights = attention(values, values, query, mask)
    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()
    # Uniform, because every logit was filled with the same sentinel.
    assert torch.allclose(weights, torch.full_like(weights, 1 / 5), atol=1e-6)


@pytest.mark.parametrize('seed', range(3))
def test_transformer_layer_matches_upstream(trxl, upstream, seed):
    torch.manual_seed(seed)
    dim, heads, batch, memory = 64, 4, 3, 9
    ours = trxl.TransformerLayer(dim, heads).double()
    theirs = upstream['TransformerLayer'](dim, heads).double()
    theirs.load_state_dict(ours.state_dict())

    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(batch, memory, dim, generator=generator, dtype=torch.float64)
    query = torch.randn(batch, 1, dim, generator=generator, dtype=torch.float64)
    mask = torch.ones(batch, memory, dtype=torch.float64)

    got, _ = ours(value, value, query, mask)
    want, _ = theirs(value, value, query, mask)
    assert torch.allclose(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize('encoding', ['absolute', 'learned', ''])
def test_transformer_matches_upstream(trxl, upstream, encoding):
    torch.manual_seed(0)
    layers, dim, heads, max_steps, memory_length = 3, 64, 4, 64, 9
    batch = 5

    ours = trxl.Transformer(layers, dim, heads, max_steps, encoding).double()
    theirs = upstream['Transformer'](layers, dim, heads, max_steps, encoding).double()
    theirs.load_state_dict(ours.state_dict())

    generator = torch.Generator().manual_seed(1)
    x = torch.randn(batch, dim, generator=generator, dtype=torch.float64)
    memories = torch.randn(batch, memory_length, layers, dim, generator=generator, dtype=torch.float64)
    mask = torch.ones(batch, memory_length, dtype=torch.float64)
    indices = torch.stack([torch.arange(memory_length) for _ in range(batch)])

    got_x, got_mem = ours(x.clone(), memories.clone(), mask, indices)
    want_x, want_mem = theirs(x.clone(), memories.clone(), mask, indices)
    assert torch.allclose(got_x, want_x, rtol=0, atol=1e-11)
    assert torch.allclose(got_mem, want_mem, rtol=0, atol=1e-11)


def test_batched_index_select_matches_upstream(trxl, upstream):
    generator = torch.Generator().manual_seed(0)
    source = torch.randn(6, 20, 3, 8, generator=generator)
    index = torch.randint(0, 20, (6, 5), generator=generator)
    assert torch.equal(trxl.batched_index_select(source, 1, index.clone()),
                       upstream['batched_index_select'](source, 1, index.clone()))


def test_batched_index_select_picks_per_row_windows(trxl):
    source = torch.arange(2 * 5).reshape(2, 5, 1).float()
    index = torch.tensor([[0, 1], [3, 4]])
    got = trxl.batched_index_select(source, 1, index)
    assert torch.equal(got.squeeze(-1), torch.tensor([[0.0, 1.0], [8.0, 9.0]]))


# ---------------------------------------------------------------------------
# memory bookkeeping
# ---------------------------------------------------------------------------

def test_memory_mask_reveals_exactly_the_written_slots(trxl):
    """Row `t` of `tril(..., diagonal=-1)` has `t` ones."""
    mask = trxl.build_memory_mask(5)
    assert mask.shape == (5, 5)
    for t in range(5):
        assert mask[t].sum().item() == t
        assert torch.all(mask[t, :t] == 1)
        assert torch.all(mask[t, t:] == 0)


def test_memory_index_table_is_a_sliding_window_after_warmup(trxl):
    max_steps, length = 12, 4
    table = trxl.build_memory_index_table(max_steps, length)
    assert table.shape == (max_steps, length)
    # Warm-up: the first L-1 rows all point at slots 0..L-1.
    for t in range(length - 1):
        assert torch.equal(table[t], torch.arange(length))
    # After that, row t is [t-L+1, ..., t].
    for t in range(length - 1, max_steps):
        assert torch.equal(table[t], torch.arange(t - length + 1, t + 1))
    assert table.max().item() == max_steps - 1


def test_index_table_matches_the_reference_construction(trxl):
    """The reference builds it as `repetitions + stacked arange`; same result."""
    max_steps, length = 20, 6
    tail = torch.stack([torch.arange(i, i + length) for i in range(max_steps - length + 1)])
    repetitions = torch.repeat_interleave(
        torch.arange(0, length).unsqueeze(0), length - 1, dim=0).long()
    expected = torch.cat((repetitions, tail))
    assert torch.equal(trxl.build_memory_index_table(max_steps, length), expected)


def test_current_step_is_always_the_last_slot_of_its_window(trxl):
    """The query's own memory slot must be the newest one in view."""
    max_steps, length = 30, 7
    table = trxl.build_memory_index_table(max_steps, length)
    for t in range(length - 1, max_steps):
        assert table[t, -1].item() == t


def test_memory_is_not_a_gradient_path_across_timesteps(trxl):
    """`out_memories.append(x.detach())` — nothing backpropagates through time."""
    torch.manual_seed(0)
    transformer = trxl.Transformer(2, 32, 4, 16, 'absolute').double()
    x = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
    memories = torch.randn(3, 5, 2, 32, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.float64)
    indices = torch.stack([torch.arange(5) for _ in range(3)])

    _, new_memory = transformer(x, memories, mask, indices)
    assert not new_memory.requires_grad


def test_transformer_output_still_carries_gradient(trxl):
    """...but the *embedding* must, or nothing trains."""
    torch.manual_seed(0)
    transformer = trxl.Transformer(2, 32, 4, 16, 'absolute').double()
    x = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
    memories = torch.randn(3, 5, 2, 32, dtype=torch.float64, requires_grad=True)
    out, _ = transformer(x, memories, torch.ones(3, 5, dtype=torch.float64),
                         torch.stack([torch.arange(5) for _ in range(3)]))
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert memories.grad is not None and memories.grad.abs().sum() > 0


def test_memory_horizon_bounds_the_allocation(trxl):
    """The Atari adaptation: memory is O(max_episode_steps), not O(episode).

    The episodic memory is `num_envs * horizon * num_layers * dim` floats, so
    the horizon is the only term available to bound. At the repo's usual 128
    envs a full 27,000-step Atari episode would need ~16 GB for this tensor
    alone, on top of the per-rollout stored windows.
    """
    layers, dim, bytes_per_float = 3, 384, 4

    def megabytes(num_envs, horizon):
        return num_envs * horizon * layers * dim * bytes_per_float / 1e6

    assert megabytes(32, 1024) == pytest.approx(151, rel=0.02)
    assert megabytes(32, 27000) == pytest.approx(3981, rel=0.02)   # ~4 GB
    assert megabytes(128, 27000) == pytest.approx(15925, rel=0.02)  # ~16 GB


# ---------------------------------------------------------------------------
# annealing
# ---------------------------------------------------------------------------

def test_linear_anneal_runs_from_initial_to_final_then_holds(trxl):
    assert np.isclose(trxl.linear_anneal(1.0, 0.0, 0, 100), 1.0)
    assert np.isclose(trxl.linear_anneal(1.0, 0.0, 50, 100), 0.5)
    assert np.isclose(trxl.linear_anneal(1.0, 0.0, 100, 100), 0.0)
    assert np.isclose(trxl.linear_anneal(1.0, 0.0, 10_000, 100), 0.0)
    # Increasing schedules work too, and a zero horizon jumps straight to final.
    assert np.isclose(trxl.linear_anneal(0.0, 5.0, 25, 100), 1.25)
    assert np.isclose(trxl.linear_anneal(1.0, 0.5, 3, 0), 0.5)


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------

class _ArgsStub:
    trxl_num_layers = 2
    trxl_num_heads = 4
    trxl_dim = 64
    trxl_memory_length = 5
    trxl_positional_encoding = 'absolute'


def test_agent_forward_shapes(trxl):
    torch.manual_seed(0)
    envs = DiscreteEnvStub(6)
    agent = trxl.Agent(envs, _ArgsStub(), max_episode_steps=32)
    batch, length, layers, dim = 4, 5, 2, 64

    obs = torch.randint(0, 255, (batch, 4, 84, 84), dtype=torch.uint8).float()
    memory = torch.randn(batch, length, layers, dim)
    mask = torch.ones(batch, length, dtype=torch.bool)
    indices = torch.stack([torch.arange(length) for _ in range(batch)])

    action, logprob, entropy, value, new_memory = agent.get_action_and_value(obs, memory, mask, indices)
    assert action.shape == (batch,)
    assert logprob.shape == (batch,)
    assert entropy.shape == (batch,)
    assert value.shape == (batch,)
    assert new_memory.shape == (batch, layers, dim)
    assert torch.allclose(agent.get_value(obs, memory, mask, indices), value)


def test_agent_encoder_takes_the_four_frame_stack(trxl):
    """The Atari adaptation: 4 grayscale channels, not the reference's 3 RGB."""
    agent = trxl.Agent(DiscreteEnvStub(6), _ArgsStub(), max_episode_steps=32)
    first_conv = next(m for m in agent.encoder if isinstance(m, nn.Conv2d))
    assert first_conv.in_channels == 4


def test_memory_content_changes_the_action_distribution(trxl):
    """The memory has to matter, or this is an expensive feed-forward PPO."""
    torch.manual_seed(0)
    agent = trxl.Agent(DiscreteEnvStub(6), _ArgsStub(), max_episode_steps=32).double()
    obs = torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8).double()
    mask = torch.ones(2, 5, dtype=torch.bool)
    indices = torch.stack([torch.arange(5) for _ in range(2)])

    with torch.no_grad():
        _, _, _, value_a, _ = agent.get_action_and_value(
            obs, torch.zeros(2, 5, 2, 64, dtype=torch.float64), mask, indices)
        _, _, _, value_b, _ = agent.get_action_and_value(
            obs, torch.randn(2, 5, 2, 64, dtype=torch.float64), mask, indices)
    assert not torch.allclose(value_a, value_b, rtol=1e-3, atol=1e-3)


def test_masked_memory_slots_are_ignored(trxl):
    """Changing a masked-out slot must not change the output at all."""
    torch.manual_seed(0)
    agent = trxl.Agent(DiscreteEnvStub(6), _ArgsStub(), max_episode_steps=32).double()
    obs = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8).double()
    indices = torch.arange(5).unsqueeze(0)
    mask = torch.tensor([[True, True, False, False, False]])

    memory = torch.randn(1, 5, 2, 64, dtype=torch.float64)
    perturbed = memory.clone()
    perturbed[:, 2:] = torch.randn_like(perturbed[:, 2:])

    with torch.no_grad():
        _, _, _, value_a, _ = agent.get_action_and_value(obs, memory, mask, indices)
        _, _, _, value_b, _ = agent.get_action_and_value(obs, perturbed, mask, indices)
    assert torch.allclose(value_a, value_b, rtol=0, atol=1e-9)


def test_bad_head_count_is_rejected(trxl):
    with pytest.raises(ValueError, match='divisible'):
        trxl.MultiHeadAttention(65, 4)


def test_bad_positional_encoding_is_rejected(trxl):
    with pytest.raises(ValueError, match='unsupported positional encoding'):
        trxl.Transformer(1, 32, 4, 16, 'rotary')
