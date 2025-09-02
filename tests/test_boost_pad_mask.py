import sys
import types
import pathlib
import numpy as np

# Ensure the project root is on the import path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# Create minimal stubs for rlgym_compat modules used by necto_obs
rlgym_compat = types.ModuleType("rlgym_compat")
common_values = types.ModuleType("common_values")
common_values.BLUE_TEAM = 0
common_values.ORANGE_TEAM = 1
game_state = types.ModuleType("game_state")
class GameState:  # pragma: no cover - simple stub
    pass
class PlayerData:  # pragma: no cover - simple stub
    pass
game_state.GameState = GameState
game_state.PlayerData = PlayerData
rlgym_compat.common_values = common_values
rlgym_compat.game_state = game_state
sys.modules['rlgym_compat'] = rlgym_compat
sys.modules['rlgym_compat.common_values'] = common_values
sys.modules['rlgym_compat.game_state'] = game_state

import importlib.util

necto_path = pathlib.Path(__file__).resolve().parents[1] / 'SkyForgeBot' / 'necto_obs.py'
spec = importlib.util.spec_from_file_location('necto_obs', necto_path)
necto_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(necto_module)
NectoObsBuilder = necto_module.NectoObsBuilder
BOOST_LOCATIONS = necto_module.BOOST_LOCATIONS

class Ball:
    position = np.zeros(3)
    linear_velocity = np.zeros(3)
    angular_velocity = np.zeros(3)

class DummyState:
    def __init__(self, boost_pads):
        self.ball = Ball()
        self.players = []
        self.boost_pads = boost_pads


def test_boost_pad_mask():
    builder = NectoObsBuilder()
    num_boosts = len(BOOST_LOCATIONS)
    state = DummyState(np.zeros(num_boosts))
    builder._maybe_update_obs(state)
    mask = builder.current_mask
    boost_slice = mask[0, 1:1 + num_boosts]
    assert np.all(boost_slice == 1)
