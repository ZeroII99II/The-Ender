import sys
import types
import pathlib
import numpy as np

# Ensure project root in path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# Stub rlgym_compat modules
rlgym_compat = types.ModuleType("rlgym_compat")
common_values = types.ModuleType("common_values")
common_values.BLUE_TEAM = 0
common_values.ORANGE_TEAM = 1
game_state = types.ModuleType("game_state")
class GameState:
    pass
class PlayerData:
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
BLUE = necto_module.BLUE_TEAM
ORANGE = necto_module.ORANGE_TEAM

# Helper classes for state construction
class CarData:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
    def forward(self):
        return np.array([1., 0., 0.])
    def up(self):
        return np.array([0., 0., 1.])

class Player(PlayerData):
    def __init__(self, team_num, position, car_id=0):
        self.team_num = team_num
        self.car_data = CarData(position)
        self.boost_amount = 0
        self.on_ground = 1
        self.has_flip = 1
        self.car_id = car_id

class Ball:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

class DummyState:
    def __init__(self, ball, players):
        self.ball = ball
        self.players = players
        self.boost_pads = np.zeros(len(necto_module.BOOST_LOCATIONS))


def test_build_obs_normalized_and_inverted():
    player_pos = np.array([1000., 2000., 100.])
    ball_pos = np.array([500., -500., 0.])

    # Blue team perspective
    player = Player(BLUE, player_pos, car_id=1)
    state = DummyState(Ball(ball_pos), [player])
    builder = NectoObsBuilder()
    q, kv, _ = builder.build_obs(player, state, np.zeros(5))
    assert np.allclose(q[0,0,5:8], player_pos/2300)
    expected_ball = (ball_pos - player_pos) / 2300
    assert np.allclose(kv[0,0,5:8], expected_ball)

    # Orange team perspective - coordinates should flip on x/y
    orange = Player(ORANGE, player_pos, car_id=2)
    state.players = [orange]
    builder = NectoObsBuilder()
    q_o, kv_o, _ = builder.build_obs(orange, state, np.zeros(5))
    invert = np.array([-1, -1, 1])
    assert np.allclose(q_o[0,0,5:8], player_pos * invert / 2300)
    expected_ball_o = (ball_pos - player_pos) * invert / 2300
    assert np.allclose(kv_o[0,0,5:8], expected_ball_o)
