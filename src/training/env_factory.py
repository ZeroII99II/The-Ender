from __future__ import annotations
from typing import Dict, Any, Callable
import numpy as np
from src.utils.gym_compat import gym
from src.rlbot_integration.observation_adapter import OBS_SIZE
from src.rlbot_integration.controller_adapter import CONT_DIM, DISC_DIM
from src.training.rewards_ssl import SSLReward

# TODO: replace with your real RLGym v2 session APIs.
# Provide these 3 hooks below:
#   _session_reset() -> state_dict
#   _session_step(a_cont: np.ndarray, a_disc: np.ndarray) -> state_dict
#   _build_obs(state_dict, prev_action) -> np.ndarray length 107


class RL2v2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            "cont": gym.spaces.Box(-1.0, 1.0, shape=(CONT_DIM,), dtype=np.float32),
            "disc": gym.spaces.MultiBinary(DISC_DIM)
        })
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._prev_action = np.zeros(CONT_DIM + DISC_DIM, dtype=np.float32)
        self._rew = SSLReward()

    # ----- RLGym v2 session stubs (IMPLEMENT ME) -----
    def _session_reset(self) -> Dict[str, Any]:
        # reset your RLGym v2 match here (kickoff state, seed)
        return dict(event=default_events(), self=default_self(), team=default_team(),
                    touch=default_touch(), combo=default_combo(),
                    ball_pos=np.zeros(3, np.float32))

    def _session_step(self, a_cont: np.ndarray, a_disc: np.ndarray) -> Dict[str, Any]:
        # advance simulation by one tick and return updated telemetry
        return self._session_reset()  # placeholder

    def _build_obs(self, s: Dict[str, Any], prev_action: np.ndarray) -> np.ndarray:
        # MUST mirror RLBot obs adapter ordering (length 107, in [-1,1])
        # Replace with your real observation builder
        return np.zeros((OBS_SIZE,), dtype=np.float32)
    # --------------------------------------------------

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        s = self._session_reset()
        self._rew.reset(s)
        self._prev_action[:] = 0
        obs = self._build_obs(s, self._prev_action)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Dict[str, np.ndarray]):
        a_cont = np.clip(action["cont"].astype(np.float32), -1.0, 1.0)
        a_disc = (action["disc"] > 0.5).astype(np.float32)
        s = self._session_step(a_cont, a_disc)

        obs = self._build_obs(s, self._prev_action)
        r = self._rew.get_reward(s, None, action, self._prev_action)

        terminated = bool(s.get("event", {}).get("goal") or s.get("event", {}).get("mercy"))
        truncated = bool(s.get("event", {}).get("time_up"))
        info: Dict[str, Any] = {}

        self._prev_action[:CONT_DIM] = a_cont
        self._prev_action[CONT_DIM:] = a_disc
        return obs, float(r), terminated, truncated, info


# Minimal defaults so code runs before you wire telemetry.
def default_events():
    return dict(goal=False, save=False, shot_on_target=False,
                clear_to_corner=False, demo_for=False, demo_against=False,
                mercy=False, time_up=False)


def default_self():
    return dict(airborne=False, spd_before_land=0.0, spd_after_land=0.0,
                small_pads_taken=0, own_half_last_man_killed=False,
                too_close_thirdman=False)


def default_team():
    return dict(both_challenging=False)


def default_touch():
    return dict(by_self=False)


def default_combo():
    return dict(double_tap=False, flip_reset=False)


def make_env(seed: int = 42) -> Callable[[], RL2v2Env]:
    def _thunk():
        return RL2v2Env(seed=seed)

    return _thunk

__all__ = ["RL2v2Env", "make_env"]
