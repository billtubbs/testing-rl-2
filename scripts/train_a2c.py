"""Train an A2C agent on CartPole-BT-vL-v1."""

import os
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401 — registers CartPole-BT environments
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

ENV_ID = "CartPole-BT-vL-v1"
SEED = 0
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10

LOG_DIR = "results/logs/a2c"
TB_LOG_DIR = "results/tb_logs"
MODEL_DIR = "results/models/a2c"


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = Monitor(gym.make(ENV_ID), LOG_DIR)
    eval_env = Monitor(gym.make(ENV_ID))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=1,
    )

    model = A2C(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )

    print(f"Training A2C on {ENV_ID} for {TOTAL_TIMESTEPS:,} timesteps (seed={SEED})...")
    print(f"TensorBoard logs: {TB_LOG_DIR}")
    print(f"Run: tensorboard --logdir {TB_LOG_DIR}\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(MODEL_DIR, "a2c_final"))

    env.close()
    eval_env.close()
    print("Done. Model saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
