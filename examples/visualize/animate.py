import os
import sys

_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from torchcule.atari import Env, Rom
from utils.openai.envs import create_vectorize_atari_env

def to_numpy(data):
    return data.cpu().numpy() if torch.is_tensor(data) else np.asarray(data)

def tile_frames(observations):
    obs = to_numpy(observations)
    if obs.ndim == 4 and obs.shape[1] in (1, 3) and obs.shape[-1] not in (1, 3):
        obs = obs.transpose(0, 2, 3, 1)  # NCHW (openai wrapper) -> NHWC
    return np.squeeze(np.hstack(obs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CuLE')
    parser.add_argument('--color', type=str, default='rgb', help='Color mode (rgb or gray)')
    parser.add_argument('--debug', action='store_true', help='Single step through frames for debugging')
    parser.add_argument('--env-name', type=str, help='Atari Game')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: 0)')
    parser.add_argument('--initial-steps', type=int, default=1000, help='Number of steps used to initialize the environment')
    parser.add_argument('--num-envs', type=int, default=5, help='Number of atari environments')
    parser.add_argument('--rescale', action='store_true', help='Resize output frames to 84x84 using bilinear interpolation')
    parser.add_argument('--training', action='store_true', help='Set environment to training mode')
    parser.add_argument('--use-cuda', action='store_true', help='Execute ALEs on GPU')
    parser.add_argument('--use-openai', action='store_true', default=False, help='Use OpenAI Gym environment')
    args = parser.parse_args()

    cmap   = None if (args.color == 'rgb') and not args.use_openai else 'gray'
    device = torch.device('cuda:{}'.format(args.gpu) if args.use_cuda else 'cpu')
    debug  = args.debug
    num_envs = args.num_envs

    if args.use_openai:
        env = create_vectorize_atari_env(args.env_name, seed=0, num_envs=args.num_envs,
                                         episode_life=False, clip_rewards=False)
        observations = env.reset()
    else:
        env = Env(args.env_name, args.num_envs, args.color, device=device,
                  rescale=args.rescale, episodic_life=True, repeat_prob=0.0)
        print(env.cart)

        if args.training:
            env.train()
        observations = env.reset(initial_steps=args.initial_steps, verbose=True)

    def sample_actions():
        if args.use_openai:
            return np.array([env.action_space.sample() for _ in range(num_envs)])
        return env.sample_random_actions()

    fig, ax = plt.subplots()
    # The reset frame can be all zeros, so fix the color limits instead of
    # letting imshow derive them from the first frame (grayscale would
    # otherwise render saturated). Note: animated=True must NOT be passed
    # here — without blitting, modern matplotlib skips animated artists
    # during regular draws, leaving an empty white canvas.
    first = tile_frames(observations)
    clim = {} if first.ndim == 3 else {'vmin': 0, 'vmax': 255}
    img = ax.imshow(first, cmap=cmap, **clim)
    ax.axis('off')

    frame = 0

    if debug:
        ax.set_title('frame: {}, rewards: {}, done: {}'.format(frame, [], []))
    else:
        fig.suptitle(frame)

    def updatefig(*fargs):
        global ax, debug, env, frame, img

        if debug:
            input('Press Enter to continue...')

        actions = sample_actions()

        observations, reward, done, info = env.step(actions)
        img.set_array(tile_frames(observations))

        if debug:
            ax.title.set_text('{}) rewards: {}, done: {}'.format(
                frame, to_numpy(reward), to_numpy(done)))
        else:
            fig.suptitle(frame)

        frame += 1

        return img,

    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=False,
                                  cache_frame_data=False)
    plt.tight_layout()
    plt.show()
