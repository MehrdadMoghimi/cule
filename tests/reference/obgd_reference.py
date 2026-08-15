"""NumPy reference for ObGD and SparseInit, written from the published equations.

Source of the specification: "Streaming Deep Reinforcement Learning Finally
Works", Elsayed, Vasan and Mahmood, ICLR 2025 (https://arxiv.org/abs/2410.14606)
-- Algorithm 3 (ObGD) and Section 4 (SparseInit).

The authors' implementation (https://github.com/mohmdelsayed/streaming-drl) is
released under CC BY-NC 4.0, so *no code is copied from it* and none is vendored
here.  What follows is an independent transcription of the paper's update rule:

    z_t     <- gamma * lambda * z_{t-1} + grad_t
    delta_bar <- max(|delta_t|, 1)
    M_t     <- delta_bar * ||z_t||_1 * alpha * kappa
    alpha_t <- alpha / M_t   if M_t > 1
               alpha         otherwise
    theta   <- theta - alpha_t * delta_t * z_t
    z_t     <- 0             if the episode ended or the action was non-greedy

Writing it out in NumPy, separately from the PyTorch trainer, is what makes
`tests/test_stream_equivalence.py` a real check rather than a restatement.
"""

import math

import numpy as np


class ObGDReference:
    """Single-stream ObGD over a flat list of parameter arrays."""

    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        self.params = [np.array(p, dtype=np.float64) for p in params]
        self.traces = [np.zeros_like(p) for p in self.params]
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa

    def step(self, grads, delta, reset=False):
        decay = self.gamma * self.lamda
        trace_l1 = 0.0
        for index, gradient in enumerate(grads):
            self.traces[index] = decay * self.traces[index] + np.asarray(gradient, dtype=np.float64)
            trace_l1 += np.abs(self.traces[index]).sum()

        delta_bar = max(abs(float(delta)), 1.0)
        bound = delta_bar * trace_l1 * self.lr * self.kappa
        step_size = self.lr / bound if bound > 1.0 else self.lr

        for index in range(len(self.params)):
            self.params[index] = self.params[index] - step_size * float(delta) * self.traces[index]
            if reset:
                self.traces[index] = np.zeros_like(self.traces[index])
        return step_size


def sparse_init_zero_count(fan_in, sparsity):
    """Number of incoming weights SparseInit zeroes per output unit."""
    return int(math.ceil(sparsity * fan_in))


def sparse_init_bound(fan_in):
    """SparseInit draws the surviving weights from U(-1/sqrt(fan_in), +1/sqrt(fan_in))."""
    return math.sqrt(1.0 / fan_in)


def layer_norm(x, eps=1e-5):
    """Parameter-free LayerNorm over every axis of `x`."""
    return (x - x.mean()) / np.sqrt(x.var() + eps)
