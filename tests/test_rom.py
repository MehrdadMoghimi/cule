import pytest

from torchcule.atari import Rom
from torchcule.atari.rom import rom_id_from_env_name

@pytest.mark.parametrize('env_name, rom_id', [
    ('PongNoFrameskip-v4', 'pong'),
    ('BreakoutDeterministic-v0', 'breakout'),
    ('SpaceInvadersNoFrameskip-v4', 'space_invaders'),
    ('UpNDownNoFrameskip-v4', 'up_n_down'),
    ('MsPacmanNoFrameskip-v4', 'ms_pacman'),
    ('MontezumaRevengeNoFrameskip-v4', 'montezuma_revenge'),
    ('ALE/Pong-v5', 'pong'),
    ('ALE/KungFuMaster-v5', 'kung_fu_master'),
    ('space_invaders', 'space_invaders'),
    ('pong', 'pong'),
])
def test_rom_id_from_env_name(env_name, rom_id):
    assert rom_id_from_env_name(env_name) == rom_id

def test_pong_rom_properties():
    rom = Rom('PongNoFrameskip-v4')
    assert rom.is_supported()
    assert rom.ram_size() == 128
    assert rom.rom_size() == 2048
    assert rom.screen_width() == 160
    assert rom.screen_height() == 210
    assert rom.is_ntsc()
    assert len(rom.md5()) == 32
    assert rom.game_name()
    assert 2 <= len(rom.minimal_actions()) <= 18

@pytest.mark.parametrize('env_name', [
    'PongNoFrameskip-v4',        # 2K
    'SpaceInvadersNoFrameskip-v4',  # 4K
    'MsPacmanNoFrameskip-v4',    # F8
    'RoadRunnerNoFrameskip-v4',  # F6
    'MontezumaRevengeNoFrameskip-v4',  # E0
    'RobotankNoFrameskip-v4',    # FE
])
def test_rom_formats_load(env_name):
    rom = Rom(env_name)
    assert rom.is_supported()
    assert rom.rom_size() > 0

def test_unknown_game_raises():
    with pytest.raises(IOError):
        Rom('NotARealGameNoFrameskip-v4')
