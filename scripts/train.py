"""Train an RL agent on a Gymnasium environment."""

import argparse
import os
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401 — registers CartPole-BT environments
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

DEFAULT_ENV_ID = "CartPole-BT-vL-v1"
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10

ALGOS = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV_ID)
    parser.add_argument("--algo", choices=ALGOS, default="ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    log_dir = f"results/logs/{args.env}/{args.algo}/seed_{args.seed}"
    tb_log_dir = "results/tb_logs"
    model_dir = f"results/models/{args.env}/{args.algo}/seed_{args.seed}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = Monitor(gym.make(args.env), log_dir)
    eval_env = Monitor(gym.make(args.env))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=1,
    )

    model = ALGOS[args.algo](
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tb_log_dir,
    )

    print(f"Training {args.algo.upper()} on {args.env} for {args.timesteps:,} timesteps (seed={args.seed})...")
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"Run: tensorboard --logdir {tb_log_dir}\n")

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(model_dir, f"{args.algo}_final"))

    env.close()
    eval_env.close()
    print("Done. Model saved to", model_dir)


if __name__ == "__main__":
    main()
