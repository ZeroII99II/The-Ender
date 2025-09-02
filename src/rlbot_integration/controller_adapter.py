from __future__ import annotations
import numpy as np

# Continuous inputs: steer, throttle, pitch, yaw, roll
CONT_DIM = 5
# Discrete/binary: jump, boost, handbrake
DISC_DIM = 3


def apply_cont(a):
    # [-1,1] already; just clamp for safety
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def apply_disc(b):
    return (b > 0.5).astype(np.float32)


def format_action(a_cont: np.ndarray, a_disc: np.ndarray) -> np.ndarray:
    """Return merged action vector expected by RLBot.

    Output order matches the RLGym session packer:
    [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
    """
    cont = apply_cont(np.asarray(a_cont, dtype=np.float32))
    disc = apply_disc(np.asarray(a_disc, dtype=np.float32))
    steer, throttle, pitch, yaw, roll = cont[:CONT_DIM]
    jump, boost, handbrake = disc[:DISC_DIM]
    return np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)


__all__ = ["CONT_DIM", "DISC_DIM", "apply_cont", "apply_disc", "format_action"]
