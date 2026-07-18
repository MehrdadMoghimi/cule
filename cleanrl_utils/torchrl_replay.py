"""GPU replay helpers for TorchRL-backed Atari learners."""

from __future__ import annotations

from collections import deque

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import Sampler


class GpuPrioritizedSampler(Sampler):
    """TorchRL sampler backed by CUDA sum/min segment trees.

    TorchRL's native prioritized sampler requires its optional segment-tree
    extension.  This sampler keeps both priorities and sampling on the learner
    device while the regular TorchRL ReplayBuffer retains ownership of the
    transition storage and circular writer.
    """

    def __init__(self, capacity: int, alpha: float, beta: float, eps: float, device: torch.device):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.device = device
        self.tree_capacity = 1 << max(0, (self.capacity - 1).bit_length())
        self.tree_depth = self.tree_capacity.bit_length() - 1
        self.sum_tree = torch.zeros(2 * self.tree_capacity, dtype=torch.float32, device=device)
        self.min_tree = torch.full(
            (2 * self.tree_capacity,), float("inf"), dtype=torch.float32, device=device
        )
        self.max_raw_priority = torch.ones((), dtype=torch.float32, device=device)

    def _indices(self, index: torch.Tensor | int) -> torch.Tensor:
        return torch.as_tensor(index, dtype=torch.long, device=self.device).reshape(-1)

    def _set_leaf_priorities(self, index: torch.Tensor | int, priority: torch.Tensor) -> None:
        leaves = self._indices(index) + self.tree_capacity
        priority = priority.to(device=self.device, dtype=self.sum_tree.dtype).reshape(-1)
        unique_leaves, inverse = torch.unique(leaves, return_inverse=True)
        # A sampled replay index can occur more than once. Keeping the largest
        # new error is deterministic and prevents duplicate writes racing.
        values = torch.full(
            (unique_leaves.numel(),), float("-inf"), dtype=self.sum_tree.dtype, device=self.device
        )
        values.scatter_reduce_(0, inverse, priority, reduce="amax", include_self=True)
        self.sum_tree[unique_leaves] = values
        self.min_tree[unique_leaves] = torch.where(
            values > 0, values, torch.full_like(values, float("inf"))
        )

        nodes = unique_leaves
        for _ in range(self.tree_depth):
            nodes = torch.unique(nodes >> 1)
            self.sum_tree[nodes] = self.sum_tree[nodes << 1] + self.sum_tree[(nodes << 1) + 1]
            self.min_tree[nodes] = torch.minimum(self.min_tree[nodes << 1], self.min_tree[(nodes << 1) + 1])

    def add(self, index: int) -> None:
        self._set_leaf_priorities(index, self.max_raw_priority.pow(self.alpha))

    def extend(self, index: torch.Tensor) -> None:
        # Rainbow immediately follows ReplayBuffer.extend() with
        # set_initial_priorities(), which also masks invalid n-step starts.
        # Updating the tree here would propagate the same leaves twice.
        return

    def set_initial_priorities(self, index: torch.Tensor, valid: torch.Tensor) -> None:
        """Set zero priority for n-step windows that cross an early terminal."""
        valid = valid.to(device=self.device, dtype=self.sum_tree.dtype).reshape(-1)
        self._set_leaf_priorities(index, valid * self.max_raw_priority.pow(self.alpha))

    def sample(self, storage, batch_size: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(storage) == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        total = self.sum_tree[1]
        mass = (torch.arange(batch_size, device=self.device) + torch.rand(batch_size, device=self.device)) * (
            total / batch_size
        )
        nodes = torch.ones(batch_size, dtype=torch.long, device=self.device)
        for _ in range(self.tree_depth):
            left = nodes << 1
            go_right = mass >= self.sum_tree[left]
            mass = mass - torch.where(go_right, self.sum_tree[left], torch.zeros_like(mass))
            nodes = left + go_right.long()
        index = nodes - self.tree_capacity
        # Normalize exactly as PER's min-probability normalization, without a
        # CPU scalar extraction or a full-buffer scan.
        weights = (self.sum_tree[nodes] / self.min_tree[1]).clamp_min(self.eps).pow(-self.beta)
        return index, {"priority_weight": weights}

    def update_priority(self, index, priority, *, storage=None) -> None:
        raw_priority = torch.as_tensor(priority, device=self.device, dtype=self.sum_tree.dtype).reshape(-1)
        raw_priority = raw_priority.abs().add(self.eps)
        self._set_leaf_priorities(index, raw_priority.pow(self.alpha))
        self.max_raw_priority = torch.maximum(self.max_raw_priority, raw_priority.max())

    def _empty(self) -> None:
        self.sum_tree.zero_()
        self.min_tree.fill_(float("inf"))
        self.max_raw_priority.fill_(1.0)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "sum_tree": self.sum_tree,
            "min_tree": self.min_tree,
            "max_raw_priority": self.max_raw_priority,
            "beta": self.beta,
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor | float]) -> None:
        self.sum_tree.copy_(torch.as_tensor(state_dict["sum_tree"], device=self.device))
        self.min_tree.copy_(torch.as_tensor(state_dict["min_tree"], device=self.device))
        self.max_raw_priority.copy_(torch.as_tensor(state_dict["max_raw_priority"], device=self.device))
        self.beta = float(state_dict["beta"])

    def dumps(self, path) -> None:
        torch.save(self.state_dict(), path)

    def loads(self, path) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


class NStepTransitionAccumulator:
    """Produces vectorized, valid n-step transitions without host copies."""

    def __init__(self, n_step: int, gamma: float):
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.transitions: deque[TensorDict] = deque(maxlen=n_step)

    def append(self, transition: TensorDict) -> tuple[TensorDict, torch.Tensor] | None:
        self.transitions.append(transition)
        if len(self.transitions) < self.n_step:
            return None

        first = self.transitions[0]
        rewards = torch.zeros_like(first["rewards"])
        dones = torch.zeros_like(first["dones"])
        valid = torch.ones_like(first["dones"], dtype=torch.bool)
        discount = 1.0
        for offset, item in enumerate(self.transitions):
            rewards.add_(item["rewards"], alpha=discount)
            dones |= item["dones"]
            # Reject n-step starts that cross a terminal before their final
            # step; a terminal on the final step is still a valid sample.
            if offset < self.n_step - 1:
                valid &= ~item["dones"]
            discount *= self.gamma

        return (
            TensorDict(
                {
                    "observations": first["observations"],
                    "actions": first["actions"],
                    "next_observations": self.transitions[-1]["next_observations"],
                    "rewards": rewards,
                    "dones": dones,
                },
                batch_size=first.batch_size,
                device=first.device,
            ),
            valid,
        )
