from __future__ import annotations
import numpy as np

# Observation size expected by both training and RLBot adapter
OBS_SIZE = 107


def build_observation(data: np.ndarray | None = None) -> np.ndarray:
    """Return observation array of length OBS_SIZE.

    This is a placeholder matching the contract. Replace with the actual
    adapter that flattens the Rocket League game state into the 107-length
    feature vector used for training and RLBot inference.
    """
    if data is None:
        return np.zeros((OBS_SIZE,), dtype=np.float32)
    obs = np.asarray(data, dtype=np.float32).reshape(-1)
    if obs.shape[0] != OBS_SIZE:
        raise ValueError(f"Observation must have length {OBS_SIZE}, got {obs.shape[0]}")
    return obs

__all__ = ["OBS_SIZE", "build_observation"]
