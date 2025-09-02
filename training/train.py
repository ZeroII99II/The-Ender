import os
import torch
from stable_baselines3 import PPO
import rlgym
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import RandomState
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import (
    EventReward,
    TouchBallReward,
    VelocityReward,
)
from rlgym.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    TimeoutCondition,
)


def make_env() -> rlgym.RLGym:
    """Create a 2v2 rlgym environment matching RLBot tick rate."""
    reward_fn = CombinedReward(
        (
            EventReward(team_goal=1.0, concede=-1.0),
            TouchBallReward(),
            VelocityReward(),
        ),
        (1.0, 0.1, 0.1),
    )
    terminal_conditions = [TimeoutCondition(225), GoalScoredCondition()]
    env = rlgym.make(
        tick_skip=8,  # 120 / 8 = 15 Hz action rate like RLBot
        team_size=2,
        obs_builder=DefaultObs(),
        state_setter=RandomState(),
        reward_fn=reward_fn,
        action_parser=DiscreteAction(),
        terminal_conditions=terminal_conditions,
    )
    return env


class AgentActor(torch.nn.Module):
    """Wrap a Stable-Baselines policy with the interface expected by Agent."""

    def __init__(self, policy, action_space):
        super().__init__()
        self.policy = policy
        self.nvec = action_space.nvec

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        dist = self.policy._get_action_dist_from_latent(latent_pi)
        logits = dist.distribution.logits
        split_logits = torch.split(logits, self.nvec.tolist(), dim=-1)
        weights = torch.ones(len(split_logits))
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
