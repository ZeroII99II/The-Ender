import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class NectoAction:
    """Action parser matching the discrete controls used by :class:`SkyForgeBot`.

    The action space is ``MultiDiscrete([3, 3, 2, 2, 2])`` corresponding to
    ``throttle/ pitch, steer/ yaw/ roll, jump, boost, handbrake``. The parser
    mirrors the logic in :mod:`SkyForgeBot.agent`.
    """

    def __init__(self):
        from gym.spaces import MultiDiscrete

        self.action_space = MultiDiscrete([3, 3, 2, 2, 2])

    def parse_actions(self, actions, _):
        import numpy as np

        actions = actions.reshape((-1, 5))
        parsed = np.zeros((actions.shape[0], 8))
        throttle = actions[:, 0] - 1
        steer = actions[:, 1] - 1
        parsed[:, 0] = throttle
        parsed[:, 1] = steer
        parsed[:, 2] = throttle
        parsed[:, 3] = steer * (1 - actions[:, 4])
        parsed[:, 4] = steer * actions[:, 4]
        parsed[:, 5] = actions[:, 2]
        parsed[:, 6] = actions[:, 3]
        parsed[:, 7] = actions[:, 4]
        return parsed


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
        self.attention = nn.MultiheadAttention(128, 1, batch_first=True)
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 128)

    def forward(self, q, kv, mask):
        attn_out, weights = self.attention(q, kv, kv, key_padding_mask=mask.bool())
        q = q + attn_out
        q = q + self.linear2(F.relu(self.linear1(q)))
        return q, weights


class EARLPerceiver(nn.Module):
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

        weights = []
        for block in self.blocks:
            q, w = block(q, kv, mask)
            weights.append(w)
        q = self.postprocess(q)
        return q, weights


class Necto(nn.Module):
    def __init__(self):
        super().__init__()
        self.earl = EARLPerceiver()
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 12)

    def forward(self, q, kv, mask):
        x, weights = self.earl(q, kv, mask)
        x = self.relu(x)
        logits = self.output(x.squeeze(1))
        split = torch.split(logits, [3, 3, 2, 2, 2], dim=-1)
        return split, weights


class DiscretePolicy(nn.Module):
    """Wrapper matching the structure expected by :mod:`SkyForgeBot.agent`."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, state):
        return self.net(*state)


from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class NectoPolicy(ActorCriticPolicy):
    """Stable-Baselines policy using the Necto network architecture."""

    def __init__(self, observation_space, action_space, lr_schedule: Schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], **kwargs)
        self.net = Necto()
        self.value_net = nn.Linear(128, 1)
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        # Override default build to skip feature extractor/mlp setup
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs, deterministic=False):
        q, kv, mask = obs
        split, _ = self.net(q, kv, mask)
        logits = torch.cat(split, dim=-1)
        distribution = self._get_action_dist_from_latent(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # Simple value estimate from mean features
        features = logits
        values = self.value_net(features)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions


def main():
    env = make_env()
    model = PPO(NectoPolicy, env, verbose=1)

    # Initialize from pretrained TorchScript weights
    ts_path = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot", "necto-model.pt")
    model.policy._load_pretrained(ts_path)

    # Load pretrained TorchScript weights
    script_path = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot", "necto-model.pt")
    if os.path.exists(script_path):
        scripted = torch.jit.load(script_path)
        state = {k.replace("net.", ""): v for k, v in scripted.state_dict().items()}
        model.policy.net.load_state_dict(state, strict=False)

    model.learn(total_timesteps=1_000)

    # Save regular checkpoint
    model.save("ppo_rlgym")

    # Export a TorchScript actor compatible with SkyForgeBot.Agent
    actor = DiscretePolicy(model.policy.net)
    dummy_q = torch.zeros(1, 1, 32)
    dummy_kv = torch.zeros(1, 40, 24)
    dummy_mask = torch.zeros(1, 40)
    scripted = torch.jit.trace(actor, (dummy_q, dummy_kv, dummy_mask))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot")
    os.makedirs(out_dir, exist_ok=True)
    scripted.save(os.path.join(out_dir, "necto-model.pt"))

    actor = AgentActor(model.policy, env.action_space)
    dummy = (
        torch.zeros((1, 1, 32)),
        torch.zeros((1, 1, 24)),
        torch.zeros((1, 1), dtype=torch.bool),
    )
    scripted = torch.jit.trace(actor, dummy)

    # Determine where to save the trained model.  RLBot can specify a target
    # location via the ``SKYFORGEBOT_MODEL_PATH`` environment variable which is
    # also used by the running agent.  When unset, default to placing the file
    # alongside ``SkyForgeBot/bot.cfg`` so it is immediately usable.
    model_path = os.getenv("SKYFORGEBOT_MODEL_PATH")
    cur_dir = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot")
    if model_path is None:
        model_path = os.path.join(cur_dir, "trained-model.pt")
    elif not os.path.isabs(model_path):
        model_path = os.path.join(cur_dir, model_path)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    scripted.save(model_path)

    # Save directly to the path expected by RLBot so the new model is
    # loaded without any manual renaming or moving.
    out_dir = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot")
    os.makedirs(out_dir, exist_ok=True)
    # Save directly to the path expected by ``bot.cfg`` so RLBot can load the
    # freshly trained model without any manual file moves.
    scripted.save(os.path.join(out_dir, "necto-model.pt"))

    scripted.save(os.path.join(out_dir, "necto-model.pt"))



if __name__ == "__main__":
    main()
