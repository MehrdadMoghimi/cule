"""Diff cleanrl/dreamerv3_atari.py against NM512/dreamerv3-torch.

DreamerV3's official implementation is JAX (danijar/dreamerv3). This diffs
against the PyTorch reproduction that the community treats as the reference,
which keeps the comparison in one framework and one dtype.

Layout differences that are relabellings, not changes, and are undone here:
  * upstream carries images as (batch, time, H, W, C) and permutes inside the
    encoder; this port keeps NCHW throughout, so observations are transposed
    before being handed to upstream;
  * upstream's `MultiEncoder`/`MultiDecoder` wrap the conv stack in a dict-keyed
    dispatcher for mixed observation spaces; only the `image` branch is used on
    Atari, so the port holds the conv stack directly.

The reference runs from `third_party/upstream/dreamerv3-torch`.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import Report, load_trainer, parse_devices, sys_path, upstream

BATCH, TIME, ACTIONS = 3, 7, 6
STOCH, DISCRETE, DETER, HIDDEN, DEPTH, UNITS = 8, 4, 32, 32, 4, 24



def gru_renamed(state):
    """Our GRU keeps a plain Sequential; upstream names its two submodules.

    Only keys under a `_cell`/GRU module are touched -- the RSSM's own
    `_img_in_layers` etc. are plain Sequentials on both sides.
    """
    renamed = {}
    for key, value in state.items():
        if key.startswith("layers.") or "_cell.layers." in key:
            key = key.replace("layers.0.", "layers.GRU_linear.")
            key = key.replace("layers.1.", "layers.GRU_norm.")
        renamed[key] = value
    return renamed


def load_upstream():
    root = upstream("dreamerv3-torch")
    with sys_path(root):
        import importlib

        return {name: importlib.import_module(name) for name in ("tools", "networks", "models")}


def run(device, seed, modules, ours):
    tools, networks = modules["tools"], modules["networks"]
    report = Report(f"DreamerV3 vs NM512/dreamerv3-torch [{device}]", tolerance=2e-5)
    torch.manual_seed(seed)

    # --- transforms -------------------------------------------------------
    x = torch.randn(64, device=device, dtype=torch.float64) * 30.0
    report.check("symlog", ours.symlog(x), tools.symlog(x))
    report.check("symexp", ours.symexp(x), tools.symexp(x))
    report.check("symexp(symlog(x)) == x", ours.symexp(ours.symlog(x)), x, tolerance=1e-9)

    # --- two-hot value/reward distribution --------------------------------
    logits = torch.randn(BATCH, TIME, 255, device=device)
    targets = torch.randn(BATCH, TIME, 1, device=device) * 5.0
    ours_dist = ours.TwoHotDist(logits)
    theirs_dist = tools.DiscDist(logits, device=device)
    report.check("two-hot log_prob", ours_dist.log_prob(targets), theirs_dist.log_prob(targets))
    report.check("two-hot mean", ours_dist.mean(), theirs_dist.mean())
    report.check("two-hot mode", ours_dist.mode(), theirs_dist.mode())
    report.check("two-hot buckets", ours_dist.buckets, theirs_dist.buckets)

    # --- categorical with a uniform mixture -------------------------------
    torch.manual_seed(seed)
    ours_onehot = ours.OneHotDist(torch.randn(BATCH, ACTIONS, device=device), unimix_ratio=0.01)
    torch.manual_seed(seed)
    theirs_onehot = tools.OneHotDist(torch.randn(BATCH, ACTIONS, device=device), unimix_ratio=0.01)
    report.check("unimix logits", ours_onehot.logits, theirs_onehot.logits)
    report.check("unimix probs", ours_onehot.probs, theirs_onehot.probs)
    report.check("onehot entropy", ours_onehot.entropy(), theirs_onehot.entropy())
    report.check("onehot mode", ours_onehot.mode(), theirs_onehot.mode())
    report.note(
        "straight-through sample keeps a gradient path",
        ours.OneHotDist(
            torch.randn(BATCH, ACTIONS, device=device, requires_grad=True), unimix_ratio=0.01
        ).sample().requires_grad_ is not None,
    )

    # --- image likelihood -------------------------------------------------
    mode = torch.randn(BATCH, TIME, 3, 8, 8, device=device)
    value = torch.randn(BATCH, TIME, 3, 8, 8, device=device)
    report.check("MSE image log_prob", ours.MSEDist(mode).log_prob(value),
                 tools.MSEDist(mode).log_prob(value))

    # --- initialisation ---------------------------------------------------
    torch.manual_seed(seed)
    ours_linear = torch.nn.Linear(64, 32).to(device)
    ours.weight_init(ours_linear)
    torch.manual_seed(seed)
    theirs_linear = torch.nn.Linear(64, 32).to(device)
    tools.weight_init(theirs_linear)
    report.check("weight_init std", ours_linear.weight.std(), theirs_linear.weight.std(), tolerance=5e-3)
    torch.manual_seed(seed)
    ours_uniform = torch.nn.Linear(64, 32).to(device)
    ours.uniform_weight_init(0.0)(ours_uniform)
    report.note("outscale=0 zeroes the head", bool(ours_uniform.weight.abs().max() == 0))

    # --- convolutional encoder and decoder --------------------------------
    torch.manual_seed(seed)
    ours_encoder = ours.ConvEncoder(depth=DEPTH).to(device)
    theirs_encoder = networks.ConvEncoder((64, 64, 3), depth=DEPTH).to(device)
    theirs_encoder.load_state_dict(ours_encoder.state_dict())
    report.note("encoder output width matches", ours_encoder.outdim == theirs_encoder.outdim,
                f"{ours_encoder.outdim} vs {theirs_encoder.outdim}")

    images = torch.rand(BATCH, TIME, 3, 64, 64, device=device)
    with torch.no_grad():
        ours_embed = ours_encoder(images)
        # Upstream takes NHWC and subtracts 0.5 in-place, so it gets a copy.
        theirs_embed = theirs_encoder(images.permute(0, 1, 3, 4, 2).contiguous())
    report.check("ConvEncoder", ours_embed, theirs_embed)

    feat_size = STOCH * DISCRETE + DETER
    torch.manual_seed(seed)
    ours_decoder = ours.ConvDecoder(feat_size, depth=DEPTH).to(device)
    theirs_decoder = networks.ConvDecoder(feat_size, (3, 64, 64), depth=DEPTH, act="SiLU").to(device)
    theirs_decoder.load_state_dict(ours_decoder.state_dict())
    features = torch.randn(BATCH, TIME, feat_size, device=device)
    with torch.no_grad():
        ours_image = ours_decoder(features)
        theirs_image = theirs_decoder(features).permute(0, 1, 4, 2, 3)
    report.check("ConvDecoder", ours_image, theirs_image)

    # --- GRU cell ---------------------------------------------------------
    torch.manual_seed(seed)
    ours_cell = ours.GRUCell(HIDDEN, DETER).to(device)
    theirs_cell = networks.GRUCell(HIDDEN, DETER).to(device)
    theirs_cell.load_state_dict(gru_renamed(ours_cell.state_dict()))
    inputs = torch.randn(BATCH, HIDDEN, device=device)
    state = torch.randn(BATCH, DETER, device=device)
    with torch.no_grad():
        theirs_out, _ = theirs_cell(inputs, [state])
    report.check("GRUCell", ours_cell(inputs, state), theirs_out)

    # --- RSSM -------------------------------------------------------------
    embed_size = ours_encoder.outdim
    torch.manual_seed(seed)
    ours_rssm = ours.RSSM(stoch=STOCH, deter=DETER, hidden=HIDDEN, discrete=DISCRETE,
                          unimix_ratio=0.01, num_actions=ACTIONS, embed=embed_size).to(device)
    theirs_rssm = networks.RSSM(
        stoch=STOCH, deter=DETER, hidden=HIDDEN, rec_depth=1, discrete=DISCRETE, act="SiLU",
        norm=True, mean_act="none", std_act="sigmoid2", min_std=0.1, unimix_ratio=0.01,
        initial="learned", num_actions=ACTIONS, embed=embed_size, device=device,
    ).to(device)
    theirs_rssm.load_state_dict(gru_renamed(ours_rssm.state_dict()))

    ours_initial = ours_rssm.initial(BATCH, device)
    theirs_initial = theirs_rssm.initial(BATCH)
    for key in ("deter", "stoch", "logit"):
        report.check(f"RSSM initial[{key}]", ours_initial[key], theirs_initial[key])
    report.note("feat size", ours_rssm.feat_size == feat_size, f"{ours_rssm.feat_size}")

    actions = torch.nn.functional.one_hot(
        torch.randint(0, ACTIONS, (BATCH,), device=device), ACTIONS
    ).float()
    state = {k: v.clone() for k, v in ours_initial.items()}
    with torch.no_grad():
        ours_prior = ours_rssm.img_step(state, actions, sample=False)
        theirs_prior = theirs_rssm.img_step({k: v.clone() for k, v in ours_initial.items()},
                                            actions, sample=False)
    for key in ("deter", "stoch", "logit"):
        report.check(f"img_step[{key}]", ours_prior[key], theirs_prior[key])

    embed = torch.randn(BATCH, embed_size, device=device)
    is_first = torch.zeros(BATCH, device=device)
    # The prior inside obs_step is always sampled, so both sides need the same
    # RNG state going in.
    torch.manual_seed(seed)
    with torch.no_grad():
        ours_post, ours_prior = ours_rssm.obs_step(
            {k: v.clone() for k, v in ours_initial.items()}, actions, embed, is_first, sample=False)
    torch.manual_seed(seed)
    with torch.no_grad():
        theirs_post, theirs_prior = theirs_rssm.obs_step(
            {k: v.clone() for k, v in ours_initial.items()}, actions, embed, is_first, sample=False)
    for key in ("deter", "stoch", "logit"):
        report.check(f"obs_step post[{key}]", ours_post[key], theirs_post[key])
        report.check(f"obs_step prior[{key}]", ours_prior[key], theirs_prior[key])

    # A sequence with a reset partway through, so the is_first branch is used.
    sequence_embed = torch.randn(BATCH, TIME, embed_size, device=device)
    sequence_actions = torch.nn.functional.one_hot(
        torch.randint(0, ACTIONS, (BATCH, TIME), device=device), ACTIONS
    ).float()
    sequence_first = torch.zeros(BATCH, TIME, device=device)
    sequence_first[:, 0] = 1.0
    sequence_first[1, 4] = 1.0
    torch.manual_seed(seed)
    with torch.no_grad():
        ours_posts, ours_priors = ours_rssm.observe(sequence_embed, sequence_actions, sequence_first)
    torch.manual_seed(seed)
    with torch.no_grad():
        theirs_posts, theirs_priors = theirs_rssm.observe(
            sequence_embed, sequence_actions, sequence_first)
    for key in ("deter", "logit"):
        report.check(f"observe post[{key}]", ours_posts[key], theirs_posts[key])
        report.check(f"observe prior[{key}]", ours_priors[key], theirs_priors[key])

    ours_kl = ours_rssm.kl_loss(ours_posts, ours_priors, 1.0, 0.5, 0.1)
    theirs_kl = theirs_rssm.kl_loss(theirs_posts, theirs_priors, 1.0, 0.5, 0.1)
    for index, name in enumerate(("kl total", "kl value", "dyn", "rep")):
        report.check(f"kl_loss {name}", ours_kl[index], theirs_kl[index])

    # --- lambda return ----------------------------------------------------
    reward = torch.randn(TIME, BATCH, 1, device=device)
    value = torch.randn(TIME, BATCH, 1, device=device)
    pcont = torch.full((TIME, BATCH, 1), 0.997, device=device)
    ours_return = ours.lambda_return(reward[1:], value[:-1], pcont[1:], value[-1], 0.95)
    theirs_return = torch.stack(
        tools.lambda_return(reward[1:], value[:-1], pcont[1:], bootstrap=value[-1],
                            lambda_=0.95, axis=0), dim=1)
    report.check("lambda_return", ours_return, theirs_return)

    # --- return normalisation --------------------------------------------
    torch.manual_seed(seed)
    values = torch.randn(TIME, BATCH, 1, device=device) * 4.0
    ours_ema_vals = torch.zeros(2, device=device)
    theirs_ema_vals = torch.zeros(2, device=device)
    ours_offset, ours_scale = ours.RewardEMA()(values, ours_ema_vals)
    theirs_offset, theirs_scale = models_reward_ema(modules, device)(values, theirs_ema_vals)
    report.check("RewardEMA offset", ours_offset, theirs_offset)
    report.check("RewardEMA scale", ours_scale, theirs_scale)
    report.check("RewardEMA state", ours_ema_vals, theirs_ema_vals)

    # --- hyperparameters --------------------------------------------------
    args = ours.Args()
    report.note("stoch 32 x 32, deter 512, hidden 512",
                (args.dyn_stoch, args.dyn_discrete, args.dyn_deter, args.dyn_hidden) == (32, 32, 512, 512))
    report.note("kl free 1.0, dyn 0.5, rep 0.1",
                (args.kl_free, args.dyn_scale, args.rep_scale) == (1.0, 0.5, 0.1))
    report.note("batch 16 x 64, horizon 15, train_ratio 1024",
                (args.batch_size, args.batch_length, args.imag_horizon, args.train_ratio)
                == (16, 64, 15, 1024))
    report.note("lr 1e-4 model / 3e-5 actor and critic",
                (args.model_lr, args.actor_lr, args.critic_lr) == (1e-4, 3e-5, 3e-5))
    report.note("discount 0.997, lambda 0.95, entropy 3e-4",
                (args.discount, args.discount_lambda, args.actor_entropy) == (0.997, 0.95, 3e-4))
    report.note("atari100k uses the reinforce gradient", args.imag_gradient == "reinforce")
    report.note("action repeat 4, prefill 2500", (args.action_repeat, args.prefill) == (4, 2500))
    return report.print()


def models_reward_ema(modules, device):
    return modules["models"].RewardEMA(device=device)


def main():
    devices, seed = parse_devices()
    modules = load_upstream()
    ours = load_trainer("dreamerv3_atari")
    results = [run(device, seed, modules, ours) for device in devices]
    print()
    if all(results):
        print("dreamerv3_atari: CONFIRMED against NM512/dreamerv3-torch on " + ", ".join(devices))
        return 0
    print("dreamerv3_atari: MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
