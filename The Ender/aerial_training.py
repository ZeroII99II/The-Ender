"""Training script with aerial-focused reward shaping.

This module defines a reward function encouraging altitude, efficient
boost usage and aerial ball hits while discouraging excessive ground
time or low altitude touches.  It also contains a very small example
training loop showing how the reward can be plugged into an RLGym
environment.  The loop is intentionally lightweight and meant as a
starting point for more sophisticated training setups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from rlgym_compat.game_state import GameState, PlayerData
from rlgym_compat.reward_functions import RewardFunction


@dataclass
class AerialTrainingReward(RewardFunction):
    """Reward function promoting aerial play and efficient boost usage."""

    tick_skip: int = 8
    altitude_weight: float = 0.1
    aerial_hit_bonus: float = 1.0
    boost_efficiency: float = 0.05
    ground_penalty: float = 0.05
    low_touch_penalty: float = 0.1
    air_times: Dict[int, float] = field(default_factory=dict)
    ground_times: Dict[int, float] = field(default_factory=dict)

    def reset(self, initial_state: GameState) -> None:
        self.air_times = {p.car_id: 0.0 for p in initial_state.players}
        self.ground_times = {p.car_id: 0.0 for p in initial_state.players}

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
    ) -> float:
        car_id = player.car_id
        dt = self.tick_skip / 120

        if player.on_ground:
            self.ground_times[car_id] += dt
            self.air_times[car_id] = 0.0
        else:
            self.air_times[car_id] += dt
            self.ground_times[car_id] = 0.0

        reward = 0.0

        # Altitude provides a small positive reward encouraging aerial play.
        reward += (player.car_data.position[2] / 2000) * self.altitude_weight

        # Bonus when hitting the ball well above the car.
        if player.ball_touched and state.ball.position[2] - player.car_data.position[2] > 100:
            reward += self.aerial_hit_bonus

        # Reward boost-efficient flight.  Using little to no boost in the air
        # gives a positive reward while burning boost slowly penalizes.
        if not player.on_ground:
            boost_use = float(previous_action[6])
            if boost_use < 0.5:
                reward += self.boost_efficiency
            else:
                reward -= boost_use * self.boost_efficiency

        # Penalize sitting on the ground for extended periods.
        if self.ground_times[car_id] > 1.0:
            reward -= self.ground_penalty

        # Penalize dribbling or touching the ball close to the ground.
        if player.ball_touched and state.ball.position[2] < 200:
            reward -= self.low_touch_penalty

        return reward


def main() -> None:
    """Example training loop using the aerial reward."""

    try:
        from rlgym_compat import make
        from rlgym_compat.action_parsers import DefaultAction

        from necto_obs import NectoObsBuilder

        env = make(
            obs_builder=NectoObsBuilder(),
            reward_fn=AerialTrainingReward(),
            action_parser=DefaultAction(),
            tick_skip=8,
        )

        for _ in range(10):  # Minimal demonstration loop
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

