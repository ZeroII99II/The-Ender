import os
from typing import Dict

import numpy as np
import torch
from stable_baselines3 import PPO

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    KickoffMutator,
    FixedTeamSizeMutator,
    MutatorSequence,
)
from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import (
    VelocityPlayerToBallReward,
)

import gymnasium as gym


class SB3SingleAgentEnv(gym.Env):
    """Lightweight wrapper converting :class:`RLGym` to the Gym API."""

    def __init__(self, env: RLGym):
        super().__init__()
        self.env = env
        env.reset()
        self.agent = env.agents[0]
        obs_space = env.observation_space(self.agent)
        assert obs_space[0] == "real"
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space[1],), dtype=np.float32
        )
        act_space = env.action_space(self.agent)
        assert act_space[0] == "discrete"
        self.action_space = gym.spaces.Discrete(act_space[1])

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        obs = self.env.reset()
        return obs[self.agent], {}

    def step(self, action):
        action_dict = {self.agent: np.array([action])}
        for other in self.env.agents:
            if other != self.agent:
                action_dict[other] = np.array([0])
        obs, rew, term, trunc = self.env.step(action_dict)
        done = term[self.agent] or trunc[self.agent]
        return obs[self.agent], rew[self.agent], done, False, {}


def make_env() -> gym.Env:
    """Create a 2v2 RLGym environment matching RLBot tick rate."""
    reward_fn = CombinedReward(
        (GoalReward(), 1.0),
        (TouchReward(), 0.1),
        (VelocityPlayerToBallReward(), 0.1),
    )
    action_parser = RepeatAction(LookupTableAction(), repeats=8)
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=2, orange_size=2),
        KickoffMutator(),
    )
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=DefaultObs(),
        action_parser=action_parser,
        reward_fn=reward_fn,
        transition_engine=RocketSimEngine(),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(15),
    )
    return SB3SingleAgentEnv(env)


class AgentActor(torch.nn.Module):
    """Wrap a Stable-Baselines policy with the interface expected by Agent."""

    def __init__(self, policy, action_space: gym.Space):
        super().__init__()
        self.policy = policy
        self.n = action_space.n

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        dist = self.policy._get_action_dist_from_latent(latent_pi)
        logits = dist.distribution.logits
        split_logits = [logits]
        weights = torch.ones(1)
        return split_logits, weights


def main():
    env = make_env()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000)

    # Save regular checkpoint
    model.save("ppo_rlgym")

    # Export a TorchScript actor compatible with SkyForgeBot.Agent
    actor = AgentActor(model.policy, env.action_space)
    dummy = torch.zeros((1,) + env.observation_space.shape)
    scripted = torch.jit.trace(actor, dummy)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot")
    os.makedirs(out_dir, exist_ok=True)
    scripted.save(os.path.join(out_dir, "trained-model.pt"))


if __name__ == "__main__":
    main()
