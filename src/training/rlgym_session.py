from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# RLGym v2 / RocketSim imports (adapt to your installed API)
# If your package exposes different entrypoints, adjust here only.
from rlgym.envs import Match
from rlgym.sim import make_default_ball, make_default_cars, SimState
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import EventReward  # baseline; we’ll ignore and use SSLReward
from rlgym.utils import common_values as C


# ---- small utility container for actions for TWO controlled cars ----
@dataclass
class TwoActions:
    # continuous: [steer, throttle, pitch, yaw, roll] per car
    cont0: np.ndarray  # shape (5,)
    cont1: np.ndarray  # shape (5,)
    # discrete: [jump, boost, handbrake] per car (0/1)
    disc0: np.ndarray  # shape (3,)
    disc1: np.ndarray  # shape (3,)


class RLSession2v2:
    """
    Wrap an RLGym v2 Match (2v2) with helpers:
      - reset(seed) -> state dict
      - step(a_cont: (10,), a_disc: (6,)) -> state dict
    We always control BLUE team’s two players (indices 0,1).
    """

    def __init__(self, tick_skip: int = 8, game_speed: float = 1.0, episode_len_seconds: int = 300):
        self.tick_skip = tick_skip
        self.game_speed = game_speed
        self.episode_len_seconds = episode_len_seconds
        self._build_match()

    def _build_match(self):
        # cars: 2 blue (our hivemind-controlled copies), 2 orange (self-play opponents)
        self.match = Match(
            tick_skip=self.tick_skip,
            team_size=2,
            state_setter=DefaultState(),
            terminal_conditions=[GoalScoredCondition(), TimeoutCondition(self.episode_len_seconds)],
            reward_function=EventReward(),  # placeholder; we compute SSLReward outside
            obs_builder=None,               # we build obs ourselves
            game_speed=self.game_speed,
        )
        self._time = 0.0

    # ---- public API expected by env_factory.py ----
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            np.random.seed(seed)
        self.match.reset(seed=seed)
        self._time = 0.0
        return self._telemetry()

    def step(self, a: TwoActions) -> dict:
        # Convert our action heads to RLGym controller arrays for both blue cars.
        # RLGym expects per-player inputs typically in the order:
        # throttle, steer, pitch, yaw, roll, jump, boost, handbrake  (all floats/bools)
        def pack(cont, disc):
            steer, throttle, pitch, yaw, roll = cont.astype(np.float32)
            j, b, hb = (disc > 0.5).astype(np.float32)
            return np.array([throttle, steer, pitch, yaw, roll, j, b, hb], dtype=np.float32)

        blue0 = pack(a.cont0, a.disc0)
        blue1 = pack(a.cont1, a.disc1)

        # Opponent policy (self-play baseline): mirror with mild noise or zero controllers
        # You can swap this for a scripted/Nexto-like baseline later.
        orange0 = np.zeros(8, np.float32)
        orange1 = np.zeros(8, np.float32)

        actions = [blue0, blue1, orange0, orange1]
        done = self.match.step(actions)
        self._time += self.tick_skip / 120.0  # 120Hz sim when possible
        return self._telemetry(done=done)

    # ---- telemetry shaping for rewards + obs ----
    def _telemetry(self, done: bool = False) -> dict:
        state: SimState = self.match.get_state()  # ball & car kinematics
        ball = state.ball  # position, vel, etc.
        cars = state.cars  # list of 4 cars; 0,1 blue; 2,3 orange

        def car_blob(i):
            c = cars[i]
            return dict(
                pos=c.position.copy(),
                vel=c.linear_velocity.copy(),
                ang=c.angular_velocity.copy(),
                rot=c.rotation.copy(),          # euler or quaternion depending on API
                airborne=bool(c.on_ground == 0),
                has_flip=bool(c.has_jump),
                boost=float(c.boost),
                supersonic=bool(c.supersonic),
            )

        # events (fill simple defaults now; env reward will compute precise ones if you add detectors)
        events = dict(
            goal=bool(self.match._goal_scored if hasattr(self.match, "_goal_scored") else False),
            save=False,
            shot_on_target=False,
            clear_to_corner=False,
            demo_for=False,
            demo_against=False,
            mercy=False,
            time_up=done,
        )

        team = dict(both_challenging=False)  # can be computed from ETAs later
        touch = dict(by_self=False)
        combo = dict(double_tap=False, flip_reset=False)

        return dict(
            time=self._time,
            ball_pos=ball.position.copy(),
            ball_vel=ball.linear_velocity.copy(),
            ball_ang=ball.angular_velocity.copy(),
            self=car_blob(0),         # we define “self” as blue car index 0; obs builder will include teammate
            mate=car_blob(1),
            opp0=car_blob(2),
            opp1=car_blob(3),
            event=events,
            team=team,
            touch=touch,
            combo=combo,
        )
