import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from cleanrl_utils.torchrl_replay import GpuPrioritizedSampler, NStepTransitionAccumulator


def _transition(step: int, dones: tuple[bool, bool] = (False, False)) -> TensorDict:
    return TensorDict(
        {
            "observations": torch.full((2, 1), step, dtype=torch.uint8),
            "next_observations": torch.full((2, 1), step + 1, dtype=torch.uint8),
            "actions": torch.zeros(2, dtype=torch.long),
            "rewards": torch.ones(2),
            "dones": torch.tensor(dones),
        },
        batch_size=[2],
    )


def test_torchrl_gpu_priority_sampler_keeps_invalid_windows_out_of_samples():
    sampler = GpuPrioritizedSampler(8, alpha=0.5, beta=0.4, eps=1e-6, device=torch.device("cpu"))
    replay = ReplayBuffer(storage=LazyTensorStorage(8, device="cpu"), sampler=sampler)
    indices = replay.extend(_transition(0))
    sampler.set_initial_priorities(indices, torch.tensor([True, False]))

    batch, info = replay.sample(32, return_info=True)
    assert set(info["index"].tolist()) == {0}
    assert batch["observations"].eq(0).all()
    assert torch.allclose(info["priority_weight"], torch.ones(32))

    # A replay sample can contain duplicate slots. Updating those priorities
    # must leave a finite, valid tree for the next sample.
    replay.update_priority(info["index"], torch.arange(1, 33, dtype=torch.float32))
    _, next_info = replay.sample(16, return_info=True)
    assert set(next_info["index"].tolist()) == {0}
    assert torch.isfinite(next_info["priority_weight"]).all()


def test_n_step_accumulator_rejects_early_terminal_and_keeps_final_terminal():
    accumulator = NStepTransitionAccumulator(n_step=3, gamma=0.5)
    assert accumulator.append(_transition(0)) is None
    assert accumulator.append(_transition(1, (False, True))) is None
    transition, valid = accumulator.append(_transition(2))

    assert transition["rewards"].tolist() == [1.75, 1.75]
    assert valid.tolist() == [True, False]
    assert transition["dones"].tolist() == [False, True]

    accumulator = NStepTransitionAccumulator(n_step=3, gamma=0.5)
    accumulator.append(_transition(0))
    accumulator.append(_transition(1))
    transition, valid = accumulator.append(_transition(2, (False, True)))

    assert transition["rewards"].tolist() == [1.75, 1.75]
    assert valid.tolist() == [True, True]
    assert transition["dones"].tolist() == [False, True]
