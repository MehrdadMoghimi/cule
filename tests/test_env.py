import pytest
import torch

from torchcule.atari import Env as AtariEnv

from conftest import device_params, requires_cuda

def make_env(device, game='PongNoFrameskip-v4', num_envs=8, **kwargs):
    defaults = dict(color_mode='gray', repeat_prob=0.0, rescale=True,
                    episodic_life=False, frameskip=4)
    defaults.update(kwargs)
    return AtariEnv(game, num_envs, device=device, **defaults)

@pytest.mark.parametrize('device', device_params())
def test_reset_and_step_shapes_gray(device):
    num_envs = 8
    env = make_env(device, num_envs=num_envs)
    obs = env.reset()

    assert obs.shape == (num_envs, 84, 84, 1)
    assert obs.dtype == torch.uint8
    assert obs.device.type == device

    actions = env.sample_random_actions()
    obs, reward, done, info = env.step(actions)

    assert obs.shape == (num_envs, 84, 84, 1)
    assert reward.shape == (num_envs,)
    assert reward.dtype == torch.float32
    assert done.shape == (num_envs,)
    assert done.dtype == torch.bool
    assert info['ale.lives'].shape == (num_envs,)
    # frames only render once the env steps (eval-mode reset returns zeros)
    assert (obs > 0).float().mean().item() > 0.5

@pytest.mark.parametrize('device', device_params())
def test_reset_and_step_shapes_rgb(device):
    num_envs = 4
    env = make_env(device, num_envs=num_envs, color_mode='rgb', rescale=False)
    obs = env.reset()

    assert obs.shape == (num_envs, env.height, env.width, 3)
    assert env.height == 210 and env.width == 160

    obs, reward, done, info = env.step(env.sample_random_actions())
    assert obs.shape == (num_envs, env.height, env.width, 3)
    assert (obs > 0).float().mean().item() > 0.1

def test_rgb_rescale_rejected():
    with pytest.raises(ValueError):
        make_env('cpu', color_mode='rgb', rescale=True)

@pytest.mark.parametrize('device', device_params())
def test_action_space_matches_minimal_actions(device):
    env = make_env(device, num_envs=2)
    n = env.action_space.n
    assert 2 <= n <= 18
    assert n == env.minimal_actions().size(0)
    actions = env.sample_random_actions()
    assert actions.max().item() < n

# games still broken in the CuLE emulator (see CHANGES.md). Almost all of the
# games historically listed here were actually mis-identified ROMs (their
# ale-py md5s were missing from rom_game_map, so they ran with bowling's
# decoder and action set) or victims of boot/flag bugs — all fixed and
# verified against ale-py under matched random play.
KNOWN_BROKEN_GAMES = [
    # agent input is ignored on both backends (the demo plays itself and
    # its scoring decodes as rewards); ale-py responds to input normally
    'double_dunk',
    # CPU backend produces no rewards while the GPU backend produces
    # implausible ones (backend-divergent emulation, unresolved)
    'elevator_action',
]

def run_game_health_check(device, game, steps=100):
    env = make_env(device, game=game, num_envs=4)
    env.train()
    obs = env.reset(initial_steps=20)
    done_events = 0
    since = torch.zeros(4)
    episode_lengths = []
    for _ in range(steps):
        obs, reward, done, info = env.step(env.sample_random_actions())
        assert torch.isfinite(reward).all()
        # garbage score decoding (uninitialized game RAM) shows up as huge
        # one-step rewards
        assert reward.abs().max().item() < 10000
        done_events += int(done.sum())
        since += 1
        for k in torch.nonzero(done.cpu()).flatten().tolist():
            episode_lengths.append(since[k].item())
            since[k] = 0
    assert (obs > 0).any(), 'frames should render non-background pixels'
    # a healthy env is not terminal on (nearly) every step; a done-loop here
    # means reset produced cached states the game itself considers terminal
    assert done_events < 0.5 * steps * 4, \
        '{} done events in {} steps x 4 envs looks like a reset loop'.format(done_events, steps)
    if episode_lengths:
        mean_len = sum(episode_lengths) / len(episode_lengths)
        assert mean_len > 15, 'episodes too short (mean {:.1f} steps)'.format(mean_len)
    return done_events

@pytest.mark.parametrize('device', device_params())
@pytest.mark.parametrize('game', [
    'BreakoutNoFrameskip-v4',           # 2K
    'PongNoFrameskip-v4',               # 2K
    'SeaquestNoFrameskip-v4',           # 4K
    'ChopperCommandNoFrameskip-v4',     # 4K
    'AsterixNoFrameskip-v4',            # F8
    'RoadRunnerNoFrameskip-v4',         # F6
    # E0-format coverage; montezuma steps mechanically but is in
    # KNOWN_BROKEN_GAMES for training (demo mode ignores agent input)
    'MontezumaRevengeNoFrameskip-v4',   # E0
    'RobotankNoFrameskip-v4',           # FE
])
def test_games_step_across_rom_formats(device, game):
    run_game_health_check(device, game)

@pytest.mark.parametrize('game', ['space_invaders', 'ms_pacman', 'tennis', 'qbert'])
def test_formerly_broken_games_stay_fixed(game):
    # these ROMs used to be mis-identified (md5 missing from rom_game_map ->
    # ran as bowling) or boot-deadlocked (qbert); guard the fixes
    run_game_health_check('cpu', game, steps=100)
    # rewards must flow under random play (they do in ale-py for all four;
    # episode termination is not asserted here -- a tennis match outlasts
    # any reasonable test budget)
    env = make_env('cpu', game=game, num_envs=8)
    env.train()
    env.reset(initial_steps=40)
    total = torch.zeros(8)
    for _ in range(300):
        _, reward, _, _ = env.step(env.sample_random_actions())
        total += reward
    assert total.abs().sum().item() > 0, 'no rewards under random play'

@pytest.mark.parametrize('game', KNOWN_BROKEN_GAMES)
@pytest.mark.xfail(strict=True,
                   reason='known emulator defect (see KNOWN_BROKEN_GAMES / CHANGES.md); '
                          'if this starts passing, move the game to the supported list')
def test_known_broken_games(game):
    # a healthy game rewards AND terminates under sustained random play
    # (double_dunk: demo rewards but no episodes; elevator_action: no CPU
    # rewards at all)
    env = make_env('cpu', game=game, num_envs=8)
    env.train()
    env.reset(initial_steps=40)
    total = torch.zeros(8)
    done_events = 0
    for _ in range(400):
        _, reward, done, _ = env.step(env.sample_random_actions())
        total += reward
        done_events += int(done.sum())
    assert total.abs().sum().item() > 0 and done_events > 0

@pytest.mark.parametrize('device', device_params())
def test_rewards_flow_under_random_play(device):
    # seaquest scores reliably under random play
    env = make_env(device, game='SeaquestNoFrameskip-v4', num_envs=32)
    env.train()
    env.reset(initial_steps=40)
    total = torch.zeros(32, device=env.device)
    for _ in range(200):
        _, reward, _, _ = env.step(env.sample_random_actions())
        total += reward
    assert total.sum().item() > 0

@pytest.mark.parametrize('device', device_params())
def test_episodic_life_and_lives_info(device):
    env = make_env(device, game='BreakoutNoFrameskip-v4', num_envs=16,
                   episodic_life=True)
    env.train()
    env.reset(initial_steps=40)
    saw_done = False
    max_lives = 0
    for _ in range(400):
        _, _, done, info = env.step(env.sample_random_actions())
        saw_done = saw_done or bool(done.any())
        max_lives = max(max_lives, int(info['ale.lives'].max().item()))
        if saw_done and max_lives > 0:
            break
    assert saw_done, 'episodic life should produce done events under random play'
    assert max_lives > 0, 'breakout should report remaining lives'

@requires_cuda
def test_gpu_rollout_deterministic_with_seeds():
    # the CUDA backend is reproducible given identical torch seeds and reset
    # seeds (the CPU backend is not: its thread pool consumes entropy in a
    # scheduling-dependent order upstream)
    def rollout():
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        env = make_env('cuda', game='BreakoutNoFrameskip-v4', num_envs=4)
        seeds = torch.arange(10, 14, dtype=torch.int32, device=env.device)
        env.reset(seeds=seeds)
        gen = torch.Generator().manual_seed(99)
        obs_seq, rew_seq = [], []
        for _ in range(40):
            actions = torch.randint(0, env.action_space.n, (4,), generator=gen)
            obs, rew, _, _ = env.step(actions.to(device=env.device, dtype=torch.uint8))
            obs_seq.append(obs.cpu().clone())
            rew_seq.append(rew.cpu().clone())
        return torch.stack(obs_seq), torch.stack(rew_seq)

    obs1, rew1 = rollout()
    obs2, rew2 = rollout()
    assert torch.equal(obs1, obs2)
    assert torch.equal(rew1, rew2)

@requires_cuda
def test_agent_input_affects_game():
    # demo-mode detector: with identical seeds, rollouts that differ only in
    # the (constant) action must diverge for a playable game. Games stuck in
    # their ROM's attract mode ignore input entirely (see KNOWN_BROKEN_GAMES).
    def rollout(game, action_idx, steps=150):
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        env = make_env('cuda', game=game, num_envs=2, color_mode='rgb',
                       rescale=False)
        env.train()
        env.reset(seeds=torch.tensor([11, 12], dtype=torch.int32,
                                     device=env.device))
        actions = torch.full((2,), min(action_idx, env.action_space.n - 1),
                             dtype=torch.uint8, device=env.device)
        for _ in range(steps):
            obs, _, _, _ = env.step(actions)
        return obs.cpu()

    for game in ('BreakoutNoFrameskip-v4', 'kaboom', 'qbert'):
        assert not torch.equal(rollout(game, 0), rollout(game, 3)), \
            '{} must respond to agent input'.format(game)

@requires_cuda
def test_training_mode_fire_reset_flag():
    env = make_env('cuda', game='BreakoutNoFrameskip-v4', num_envs=2)
    assert env.fire_reset, 'breakout has FIRE in its minimal action set'
    env2 = make_env('cuda', game='PongNoFrameskip-v4', num_envs=2)
    env2.train(frameskip=4)
    assert env2.is_training and env2.frameskip == 4

@requires_cuda
def test_render_lanes_bit_identical():
    """The warp-cooperative frame renderer must produce bit-identical frames
    for any CULE_RENDER_LANES value (lane count 1 executes the original
    serial store order, so 1 vs 32 catches any cooperative-rendering bug).
    The variable is latched on first kernel launch, hence the subprocesses."""
    import os
    import subprocess
    import sys
    import tempfile

    script = (
        "import sys, torch\n"
        "from torchcule.atari import Env\n"
        "torch.manual_seed(5)\n"
        "env = Env('BreakoutNoFrameskip-v4', 16, color_mode='gray',\n"
        "          device='cuda', rescale=True, frameskip=4,\n"
        "          episodic_life=True)\n"
        "env.train(4)\n"
        "env.reset(initial_steps=20)\n"
        "gen = torch.Generator().manual_seed(123)\n"
        "frames = []\n"
        "for _ in range(12):\n"
        "    a = torch.randint(env.action_space.n, (16,), generator=gen,\n"
        "                      dtype=torch.uint8).to('cuda')\n"
        "    obs, rew, done, _ = env.step(a)\n"
        "    frames.append(obs.cpu().clone())\n"
        "torch.save(torch.stack(frames), sys.argv[1])\n"
    )

    results = {}
    for lanes in (1, 32):
        fd, path = tempfile.mkstemp(suffix='.pt')
        os.close(fd)
        try:
            env_vars = dict(os.environ, CULE_RENDER_LANES=str(lanes))
            subprocess.run([sys.executable, '-c', script, path],
                           env=env_vars, check=True, timeout=300)
            results[lanes] = torch.load(path, weights_only=True)
        finally:
            os.unlink(path)

    assert torch.equal(results[1], results[32]), \
        'frame rendering must not depend on the render lane count'
