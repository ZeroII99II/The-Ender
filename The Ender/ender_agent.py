import numpy as np

class EnderAgent:
    """A simple deterministic controller for comparison tests."""

    def act(self, obs):
        """Return a constant action regardless of observation.

        The action vector follows the order:
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
        """
        action = np.zeros(8)
        action[0] = 1.0  # full throttle forward
        action[5] = 1.0  # jump
        return action
