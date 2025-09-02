from __future__ import annotations
import torch
import torch.nn as nn
from src.utils.gym_compat import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        in_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, 1024),   nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, 512),    nn.LayerNorm(512),  nn.SiLU(),
        )
        self._features_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_vec(env_fns):
    return VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))


def policy_kwargs():
    return dict(
        features_extractor_class=MLPExtractor,
        net_arch=[dict(pi=[256], vf=[256])],
        activation_fn=nn.SiLU,
    )

__all__ = ["MLPExtractor", "make_vec", "policy_kwargs"]
