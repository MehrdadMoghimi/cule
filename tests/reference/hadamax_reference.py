"""NumPy transcription of Hadamax's official Flax encoder, for equivalence tests.

Transcribed from https://github.com/jacobkooi/hadamax, `purejaxql/networks.py`,
class `NatureCNN` with `config["ENCODER"] == "hadamax"` and
`norm_type == "layer_norm"`.

    Apache License 2.0
    Copyright the Hadamax authors (Kooi, Yang, Francois-Lavet)

JAX is not a dependency of this repo, so the reference is re-expressed in NumPy
rather than executed through Flax.  It is written against Flax/JAX *semantics*
rather than against `cleanrl/hadamax_pqn_atari_envpool.py`, which is what makes
the comparison meaningful:

  * tensors are NHWC, as in the original;
  * `nn.Conv(..., padding="SAME")` at stride 1 pads (k-1)//2 before and
    k-1-(k-1)//2 after, i.e. asymmetrically for even kernels;
  * `nn.LayerNorm()` reduces the trailing (channel) axis only, with the Flax
    default epsilon of 1e-6;
  * `nn.gelu` is `jax.nn.gelu`, whose default is `approximate=True` -- the tanh
    approximation, not the exact erf form PyTorch defaults to;
  * `max_pool(..., padding="SAME")` pads with -inf.
"""

import numpy as np

FLAX_LAYER_NORM_EPS = 1e-6


def gelu_tanh(x):
    """jax.nn.gelu(x, approximate=True)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm_channels(x, scale, bias, eps=FLAX_LAYER_NORM_EPS):
    """flax.linen.LayerNorm() over the trailing axis of an NHWC/ND array."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _same_padding(size, kernel, stride):
    """TF/Flax SAME padding for one spatial axis."""
    out = int(np.ceil(size / stride))
    total = max((out - 1) * stride + kernel - size, 0)
    before = total // 2
    return before, total - before


def conv2d_same(x, weight, bias, stride=1):
    """nn.Conv(features, kernel_size=(k, k), strides=(s, s), padding='SAME').

    `x` is NHWC and `weight` is HWIO, matching Flax's layouts.
    """
    n, h, w, _ = x.shape
    kh, kw, in_c, out_c = weight.shape
    ph = _same_padding(h, kh, stride)
    pw = _same_padding(w, kw, stride)
    padded = np.pad(x, ((0, 0), ph, pw, (0, 0)))
    out_h = int(np.ceil(h / stride))
    out_w = int(np.ceil(w / stride))
    patches = np.empty((n, out_h, out_w, kh * kw * in_c), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            window = padded[:, i * stride:i * stride + kh, j * stride:j * stride + kw, :]
            patches[:, i, j, :] = window.reshape(n, -1)
    flat_weight = weight.reshape(kh * kw * in_c, out_c)
    return patches @ flat_weight + bias


def max_pool_same(x, window, stride):
    """flax.linen.pooling.max_pool(x, window, strides, padding='SAME')."""
    n, h, w, c = x.shape
    ph = _same_padding(h, window, stride)
    pw = _same_padding(w, window, stride)
    padded = np.pad(x, ((0, 0), ph, pw, (0, 0)), constant_values=-np.inf)
    out_h = int(np.ceil(h / stride))
    out_w = int(np.ceil(w / stride))
    out = np.empty((n, out_h, out_w, c), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            out[:, i, j, :] = padded[
                :, i * stride:i * stride + window, j * stride:j * stride + window, :
            ].max(axis=(1, 2))
    return out


def hadamax_block(x, params, kernel, pool_window, pool_stride):
    """One `################## block` of the official encoder."""
    x1 = conv2d_same(x, params['w1'], params['b1'])
    x2 = conv2d_same(x, params['w2'], params['b2'])
    x1 = layer_norm_channels(x1, params['g1'], params['be1'])  # normalize before activation
    x2 = layer_norm_channels(x2, params['g2'], params['be2'])
    x1 = gelu_tanh(x1)
    x2 = gelu_tanh(x2)
    x = x1 * x2  # element-wise multiplication
    return max_pool_same(x, pool_window, pool_stride)


def hadamax_encoder(x, block_params):
    """The three-block trunk, up to (not including) the flatten."""
    x = hadamax_block(x, block_params[0], kernel=8, pool_window=4, pool_stride=4)
    x = hadamax_block(x, block_params[1], kernel=4, pool_window=2, pool_stride=2)
    x = hadamax_block(x, block_params[2], kernel=3, pool_window=3, pool_stride=1)
    return x


def hadamax_projection(flat, weight, bias, scale, norm_bias):
    """`nn.Dense(512)` -> `nn.LayerNorm()` -> `nn.gelu`; `weight` is (in, out)."""
    hidden = flat @ weight + bias
    hidden = layer_norm_channels(hidden, scale, norm_bias)
    return gelu_tanh(hidden)
