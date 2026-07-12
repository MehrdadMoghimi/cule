"""CuLE (CUda Learning Environment module)

This module provides access to several RL environments that generate data
on the CPU or GPU.
"""

import os
import re

from ale_py import roms

from torchcule_atari import AtariRom

def rom_id_from_env_name(env_name):
    """Convert a Gym(nasium)-style environment name to an ale-py rom id.

    Handles e.g. 'PongNoFrameskip-v4', 'ALE/SpaceInvaders-v5',
    'BreakoutDeterministic-v0', or a plain rom id such as 'space_invaders'.
    """
    name = env_name.split('/')[-1]          # drop 'ALE/' namespace
    name = name.split('-')[0]               # drop '-v0/-v4/-v5' suffix
    for suffix in ('NoFrameskip', 'Deterministic'):
        name = name.replace(suffix, '')
    # CamelCase -> snake_case (no-op if already snake_case)
    name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', name)
    return name.lower()

class Rom(AtariRom):

    def __init__(self, env_name):
        rom_id = rom_id_from_env_name(env_name)
        game_path = roms.get_rom_path(rom_id)
        if (game_path is None) or (not os.path.exists(game_path)):
            raise IOError('Requested environment (%s) does not exist '
                          'in valid list of environments:\n%s' \
                          % (env_name, ', '.join(sorted(roms.get_all_rom_ids()))))
        super(Rom, self).__init__(str(game_path))

    def __repr__(self):
        return 'Name       : {}\n'\
               'Controller : {}\n'\
               'Swapped    : {}\n'\
               'Left Diff  : {}\n'\
               'Right Diff : {}\n'\
               'Type       : {}\n'\
               'Display    : {}\n'\
               'ROM Size   : {}\n'\
               'RAM Size   : {}\n'\
               'MD5        : {}\n'\
               .format(self.game_name(),
                       'Paddles' if self.use_paddles() else 'Joystick',
                       'Yes' if self.swap_paddles() or self.swap_ports() else 'No',
                       'B' if self.player_left_difficulty_B() else 'A',
                       'B' if self.player_right_difficulty_B() else 'A',
                       self.type(),
                       'NTSC' if self.is_ntsc() else 'PAL',
                       self.rom_size(),
                       self.ram_size(),
                       self.md5())
