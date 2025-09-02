import os
import torch
from stable_baselines3 import PPO
import rlgym
from rlgym.utils.state_setters import RandomState
from rlgym.utils.action_parsers import NectoAction
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
from SkyForgeBot.necto_obs import NectoObsBuilder
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn


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
        obs_builder=NectoObsBuilder(),
        state_setter=RandomState(),
        reward_fn=reward_fn,
        action_parser=NectoAction(),
        terminal_conditions=terminal_conditions,
    )
    return env


class EARLPerceiverBlock(nn.Module):
    """Simplified perceiver attention block matching the pretrained model."""

    def __init__(self, d_model: int = 128, nhead: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, 2 * d_model)
        self.linear2 = nn.Linear(2 * d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, q, kv, mask):
        attn_out, _ = self.attention(q, kv, kv, key_padding_mask=mask.bool())
        x = q + attn_out
        ff = self.linear2(self.relu(self.linear1(x)))
        return x + ff


class EARLPerceiver(nn.Module):
    """Feature extractor used by the Necto policy."""

    def __init__(self):
        super().__init__()
        self.query_preprocess = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.key_value_preprocess = nn.Sequential(
            nn.Linear(24, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.blocks = nn.ModuleList([EARLPerceiverBlock()])
        self.postprocess = nn.Identity()

    def forward(self, q, kv, mask):
        q = self.query_preprocess(q)
        kv = self.key_value_preprocess(kv)
        for block in self.blocks:
            q = block(q, kv, mask)
        return self.postprocess(q)


class NectoFeatureExtractor(BaseFeaturesExtractor):
    """Stable-Baselines features extractor wrapping the perceiver."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.earl = EARLPerceiver()
        self.relu = nn.ReLU()

    def forward(self, obs):
        q, kv, mask = obs
        x = self.earl(q, kv, mask)
        x = self.relu(x)
        return x.squeeze(1)


class NectoPolicy(ActorCriticPolicy):
    """Actor-critic policy matching the architecture of ``necto-model.pt``."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[dict(pi=[], vf=[])],
            features_extractor_class=NectoFeatureExtractor,
            **kwargs,
        )

        action_dim = int(action_space.nvec.sum())
        # Reuse action_net so state dict matches ``net.output.linear``
        self.net = nn.Module()
        self.net.earl = self.features_extractor.earl
        self.net.relu = self.features_extractor.relu
        self.action_net = nn.Linear(self.features_extractor.features_dim, action_dim)
        self.net.output = nn.Module()
        self.net.output.linear = self.action_net

    def _load_pretrained(self, path: str):
        ts_model = torch.jit.load(path)
        state = ts_model.state_dict()
        policy_state = self.state_dict()
        for k in policy_state.keys():
            ts_key = None
            if k.startswith("net.earl"):
                ts_key = k
            elif k.startswith("action_net"):
                ts_key = "net.output.linear" + k[len("action_net"):]
            if ts_key is not None and ts_key in state:
                policy_state[k] = state[ts_key]
        self.load_state_dict(policy_state, strict=False)

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
    model = PPO(NectoPolicy, env, verbose=1)

    # Initialize from pretrained TorchScript weights
    ts_path = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot", "necto-model.pt")
    model.policy._load_pretrained(ts_path)

    model.learn(total_timesteps=1_000)

    # Save regular checkpoint
    model.save("ppo_rlgym")

    # Export a TorchScript actor compatible with SkyForgeBot.Agent
    actor = AgentActor(model.policy, env.action_space)
    dummy = (
        torch.zeros((1, 1, 32)),
        torch.zeros((1, 1, 24)),
        torch.zeros((1, 1), dtype=torch.bool),
    )
    scripted = torch.jit.trace(actor, dummy)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot")
    os.makedirs(out_dir, exist_ok=True)
    scripted.save(os.path.join(out_dir, "trained-model.pt"))


if __name__ == "__main__":
    main()
