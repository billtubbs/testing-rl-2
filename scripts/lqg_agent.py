"""LQG (Linear Quadratic Gaussian) agent for CartPole-BT p2 environments.

Combines LQR control with a Kalman filter for state estimation under
partial observation. Compatible with the SB3 predict() interface.

Gains computed for CartPole-BT system: m=1, M=5, L=2, g=-10, d=1, tau=0.05s.
Reference: https://github.com/billtubbs/gym-CartPole-bt-v0/blob/master/test_run_lqg.py
"""

import numpy as np

# Discrete-time state-space matrices (tau=0.05s)
_A = np.array(
    [
        [1.00000000e00, 4.97507794e-02, 2.49480674e-03, 4.15939085e-05],
        [0.00000000e00, 9.90045685e-01, 9.97511222e-02, 2.49480674e-03],
        [0.00000000e00, -1.24740337e-04, 1.00750522e00, 5.01250418e-02],
        [0.00000000e00, -4.98755611e-03, 3.00500770e-01, 1.00750522e00],
    ],
    dtype=np.float64,
)

_B = np.array(
    [[0.00024922], [0.00995432], [0.00012474], [0.00498756]], dtype=np.float64
)

# Kalman filter gain for p2 output [x, theta]
# (Q_kf=np.eye(4)*0.01, R_kf=np.eye(2)*0.1)
DEFAULT_KF_GAIN = np.array(
    [
        [1.03962720e00, 2.07302137e-03],
        [7.90472757e-01, 9.32428877e-02],
        [-7.72253147e-04, 1.06420691e00],
        [-1.75014331e-02, 1.44210117e00],
    ],
    dtype=np.float64,
)

# LQR gain (Q=np.eye(4), R=0.0001)
DEFAULT_LQR_GAIN = np.array(
    [-100.0, -197.5366, 1491.2808, 668.4449], dtype=np.float64
)


class LQGAgent:
    """LQG controller with Kalman filter for partially-observable CartPole-BT.

    Control law:  u[t] = -K @ x_est[t]
    KF update:    x_est[t+1] = A @ x_est[t] + B*u[t] + L @ (y[t] - C @ x_est[t])
    """

    def __init__(
        self,
        goal_state,
        C,
        lqr_gain=DEFAULT_LQR_GAIN,
        kf_gain=DEFAULT_KF_GAIN,
    ):
        self.goal_state = np.asarray(goal_state, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.lqr_gain = np.asarray(lqr_gain, dtype=np.float64)
        self.kf_gain = np.asarray(kf_gain, dtype=np.float64)
        self._C_pinv = np.linalg.pinv(self.C)
        self.x_est = np.zeros((4, 1))

    @classmethod
    def from_env(cls, env, **kwargs):
        goal_state = env.unwrapped.goal_state
        C = env.unwrapped.output_matrix.astype(np.float64)
        return cls(goal_state=goal_state, C=C, **kwargs)

    def reset(self, initial_obs):
        """Initialise state estimate from the first observation of an episode."""
        xp = self.goal_state.reshape(4, 1)
        ym = (
            np.asarray(initial_obs, dtype=np.float64).reshape(-1, 1)
            - self.C @ xp
        )
        self.x_est = self._C_pinv @ ym

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        if episode_start:
            self.reset(obs)

        xp = self.goal_state.reshape(4, 1)
        ym = np.asarray(obs, dtype=np.float64).reshape(-1, 1) - self.C @ xp

        # Control action
        u = float(-self.lqr_gain @ self.x_est)

        # Kalman filter update
        error = ym - self.C @ self.x_est
        self.x_est = _A @ self.x_est + _B * u + self.kf_gain @ error

        return np.array([u], dtype=np.float32), state
