import os

from ale_py import roms

def atari_games():
    # legacy-style env names (e.g. 'SpaceInvadersNoFrameskip-v4') built from
    # the ale-py rom ids (e.g. 'space_invaders')
    return [''.join(part.title() for part in rom_id.split('_')) + 'NoFrameskip-v4'
            for rom_id in roms.get_all_rom_ids()]

env_names = atari_games()
for skip in ['QbertNoFrameskip-v4', 'ElevatorActionNoFrameskip-v4', 'DefenderNoFrameskip-v4']:
    if skip in env_names:
        env_names.remove(skip)
num_ales_list = [1024, 2048, 16, 4096] #[1, 32, 64, 128, 256, 512, 1024, 2048, 4096]

for num_ales in num_ales_list:
    for env_name in env_names:

        if num_ales < 1025:
            os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize --use-openai')
        os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize')
        os.system('python vtrace_main.py --benchmark --num-ales ' + str(num_ales) + ' --env-name ' + env_name + ' --num-steps 5 --num-minibatches 1 --num-steps-per-update 5 --normalize --use-cuda-env')
