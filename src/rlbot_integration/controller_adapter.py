from __future__ import annotations
import numpy as np

# Continuous: steer, throttle, pitch, yaw, roll
CONT_DIM = 5
# Discrete/binary: jump, boost, handbrake
DISC_DIM = 3


def format_action(a_cont: np.ndarray, a_disc: np.ndarray) -> np.ndarray:
    """Return merged action vector expected by RLBot.

    Continuous controls are squashed with tanh into [-1, 1]; discrete logits or
    probabilities are thresholded at >0 to produce {0,1} outputs.
    """
    cont = np.tanh(np.asarray(a_cont, dtype=np.float32))
    disc = (np.asarray(a_disc, dtype=np.float32) > 0).astype(np.float32)
    return np.concatenate([cont[:CONT_DIM], disc[:DISC_DIM]])

__all__ = ["CONT_DIM", "DISC_DIM", "format_action"]
