import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from utils.openai.atari_wrappers import translate_env_id
from utils.openai.envs import create_atari_env, create_vectorize_atari_env

@pytest.mark.parametrize('env_id, translated', [
    ('PongNoFrameskip-v4', 'ALE/Pong-v5'),
    ('BreakoutDeterministic-v4', 'ALE/Breakout-v5'),
    ('SpaceInvadersNoFrameskip-v4', 'ALE/SpaceInvaders-v5'),
    ('ALE/Pong-v5', 'ALE/Pong-v5'),
])
def test_translate_env_id(env_id, translated):
    assert translate_env_id(env_id) == translated

def test_single_env_old_api():
    env = create_atari_env('PongNoFrameskip-v4', seed=7, episode_life=True,
                           clip_rewards=True)()
    obs = env.reset()
    assert obs.shape == (1, 84, 84)
    assert obs.dtype == np.uint8

    for _ in range(50):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs.shape == (1, 84, 84)
        assert np.isfinite(reward)
        assert reward in (-1.0, 0.0, 1.0), 'rewards should be clipped'
        assert 'ale.lives' in info
        if done:
            obs = env.reset()
    env.close()

def test_single_env_accepts_array_actions():
    env = create_atari_env('PongNoFrameskip-v4', seed=7)()
    env.reset()
    # the a2c test harness sends shape-(1,) numpy arrays
    _, _, _, info = env.step(np.array([1], dtype=np.int64))
    assert 'ale.lives' in info
    env.close()

def test_time_limit_truncates():
    env = create_atari_env('PongNoFrameskip-v4', seed=7, max_frames=10)()
    env.reset()
    done = False
    for _ in range(11):
        _, _, done, _ = env.step(0)
        if done:
            break
    assert done, 'TimeLimit should truncate within max_frames steps'
    env.close()

def test_vector_env_old_api():
    num_envs = 3
    venv = create_vectorize_atari_env('BreakoutNoFrameskip-v4', seed=3,
                                      num_envs=num_envs, episode_life=True,
                                      clip_rewards=True)
    obs = venv.reset()
    assert obs.shape == (num_envs, 1, 84, 84)

    for _ in range(30):
        actions = np.random.randint(venv.action_space.n, size=num_envs)
        obs, rewards, dones, infos = venv.step(actions)

    assert obs.shape == (num_envs, 1, 84, 84)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    assert len(infos) == num_envs
    assert all('ale.lives' in i for i in infos)
    venv.close()
