"""Training script for SkyForgeBot using rlgym 2.0 and PPO.

The script builds a 2v2 environment matching the configuration expected by
RLBot and continues training from the bundled TorchScript model.  After the
short training run a new TorchScript actor is written back to the path expected
by ``bot.cfg`` so the updated weights can be used immediately in matches.

Usage::

    python training/train.py            # saves to SkyForgeBot/necto-model.pt
    SKYFORGEBOT_MODEL_PATH=path/to/model.pt python training/train.py

The environment variable makes it easy for RLBot to direct training output to a
particular location.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from rlgym_compat.envs import make
from rlgym_compat.reward_functions import CombinedReward
from rlgym_compat.reward_functions.common_rewards import (
    EventReward,
    TouchBallReward,
    VelocityReward,
)
from rlgym_compat.state_setters import RandomState
from rlgym_compat.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    TimeoutCondition,
)

from SkyForgeBot.necto_obs import NectoObsBuilder


# ---------------------------------------------------------------------------
# Environment helpers


class NectoAction:
    """Discrete action parser matching :mod:`SkyForgeBot.agent`.

    The action space is ``MultiDiscrete([3, 3, 2, 2, 2])`` representing
    ``throttle/pitch, steer/yaw/roll, jump, boost, handbrake``.
    """

    def __init__(self) -> None:
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


def make_env():
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
    env = make(
        tick_skip=8,  # 120 / 8 = 15 Hz action rate like RLBot
        team_size=2,
        obs_builder=NectoObsBuilder(),
        state_setter=RandomState(),
        reward_fn=reward_fn,
        action_parser=NectoAction(),
        terminal_conditions=terminal_conditions,
    )
    return env


# ---------------------------------------------------------------------------
# Neural network architecture


class EARLPerceiverBlock(nn.Module):
    """Single attention block used by the Necto network."""

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
    """Feature extractor replicating the pretrained Necto architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.query_pre = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.kv_pre = nn.Sequential(
            nn.Linear(24, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.blocks = nn.ModuleList([EARLPerceiverBlock()])
        self.post = nn.Identity()

    def forward(self, q, kv, mask):
        q = self.query_pre(q)
        kv = self.kv_pre(kv)
        for block in self.blocks:
            q = block(q, kv, mask)
        return self.post(q), None


class Necto(nn.Module):
    """Policy network producing logits for discrete actions."""

    def __init__(self) -> None:
        super().__init__()
        self.earl = EARLPerceiver()
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 12)  # 3+3+2+2+2

    def forward(self, q, kv, mask):
        x, _ = self.earl(q, kv, mask)
        x = self.relu(x)
        logits = self.output(x.squeeze(1))
        split = torch.split(logits, [3, 3, 2, 2, 2], dim=-1)
        return split, None


class DiscretePolicy(nn.Module):
    """Wrapper module that matches :mod:`SkyForgeBot.agent` expectations."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, q, kv, mask):
        return self.net(q, kv, mask)


class NectoPolicy(ActorCriticPolicy):
    """Stable-Baselines policy using the Necto network."""

    def __init__(self, observation_space, action_space, lr_schedule: Schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], **kwargs)
        self.net = Necto()
        self.value_net = nn.Linear(128, 1)
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:  # type: ignore[override]
        # Override default build to skip feature extractor/mlp setup
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs, deterministic: bool = False):  # type: ignore[override]
        q, kv, mask = obs
        split, _ = self.net(q, kv, mask)
        logits = torch.cat(split, dim=-1)
        dist = self._get_action_dist_from_latent(logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(logits)
        return actions, values, log_prob

    def _predict(self, observation, deterministic: bool = False):  # type: ignore[override]
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions


def _load_pretrained(policy: NectoPolicy, path: str) -> None:
    """Load TorchScript weights into the policy network if present."""

    if os.path.exists(path):
        scripted = torch.jit.load(path)
        state = {k.replace("net.", ""): v for k, v in scripted.state_dict().items()}
        policy.net.load_state_dict(state, strict=False)


def main() -> None:
    env = make_env()
    model = PPO(NectoPolicy, env, verbose=1, n_steps=64)

    ts_path = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot", "necto-model.pt")
    _load_pretrained(model.policy, ts_path)

    # Run a small number of timesteps for demonstration purposes.
    model.learn(total_timesteps=1_000)

    # Export a TorchScript actor compatible with SkyForgeBot.Agent
    actor = DiscretePolicy(model.policy.net)
    builder = NectoObsBuilder()
    dummy_q = torch.zeros(1, 1, 32)
    dummy_kv = torch.zeros(1, 1 + 4 + len(builder._boost_locations), 24)
    dummy_mask = torch.zeros(1, dummy_kv.shape[1])
    scripted = torch.jit.trace(actor, (dummy_q, dummy_kv, dummy_mask))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "SkyForgeBot")
    model_path = os.getenv("SKYFORGEBOT_MODEL_PATH")
    if model_path is None:
        model_path = os.path.join(out_dir, "necto-model.pt")
    elif not os.path.isabs(model_path):
        model_path = os.path.join(out_dir, model_path)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    scripted.save(model_path)


if __name__ == "__main__":
    main()

