"""LQR (Linear Quadratic Regulator) agent with an SB3-compatible interface.

The gain matrix was calculated using control.lqr with Q=np.eye(4), R=0.0001
for the CartPole-BT family of environments. See:
https://python-control.readthedocs.io/en/latest/generated/control.lqr.html
"""

import numpy as np
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401 — registers CartPole-BT environments

# Optimal gain for CartPole-BT environments (Q=np.eye(4), R=0.0001)
DEFAULT_GAIN = np.array([-100.00, -197.54, 1491.28, 668.44])


class LQRAgent:
    """LQR controller compatible with the SB3 predict() interface.

    Control law: u = -K @ (obs - goal_state)
    """

    def __init__(self, goal_state, gain=DEFAULT_GAIN):
        self.goal_state = np.asarray(goal_state, dtype=np.float64)
        self.gain = np.asarray(gain, dtype=np.float64)

    @classmethod
    def from_env(cls, env, gain=DEFAULT_GAIN):
        """Construct using the goal state from a Gymnasium environment."""
        goal_state = env.unwrapped.goal_state
        return cls(goal_state=goal_state, gain=gain)

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        action = -self.gain @ (
            np.asarray(obs, dtype=np.float64) - self.goal_state
        )
        return np.array([action], dtype=np.float32), state
