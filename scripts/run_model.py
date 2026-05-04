"""Run a trained SB3 model on a Gymnasium environment."""

import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401 — registers CartPole-BT environments
from stable_baselines3 import A2C, PPO, SAC, TD3
from sb3_contrib import TQC
from lqr_agent import LQRAgent

ALGOS = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3, "tqc": TQC}
BENCHMARK_ALGOS = {"lqr"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", help="Gymnasium environment ID")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="path to saved model (.zip); not required for benchmark agents",
    )
    parser.add_argument(
        "--algo",
        choices=list(ALGOS) + list(BENCHMARK_ALGOS),
        default=None,
        help="algorithm (required if model path is ambiguous or using a benchmark agent)",
    )
    parser.add_argument("--seed", type=int, default=0, help="environment seed")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--save", metavar="FILE", default=None, help="save results to CSV file"
    )
    args = parser.parse_args()

    if args.algo == "lqr" or (args.algo is None and args.model is None):
        algo_name = "lqr"
    else:
        algo_name = args.algo
        if algo_name is None:
            parser.error(
                "provide --algo when the model path does not contain the algorithm name"
            )
        if args.model is None:
            parser.error("model path is required for trained agents")

    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    if algo_name == "lqr":
        model = LQRAgent.from_env(env)
    else:
        model = ALGOS[algo_name].load(args.model, env=env)

    print(
        f"Running {algo_name.upper()} on {args.env} for {args.episodes} episodes (seed={args.seed})..."
    )
    print()

    # Build column names from env spaces
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.shape[0]
    obs_cols = [f"obs_{i}" for i in range(n_obs)]
    act_cols = [f"action_{i}" for i in range(n_act)]

    rows = []
    rewards = []
    obs, _ = env.reset(seed=args.seed)
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if args.save:
                rows.append(
                    [episode, step]
                    + obs.tolist()
                    + np.atleast_1d(action).tolist()
                    + [reward]
                )
            total_reward += reward
            obs = next_obs
            done = terminated or truncated
            step += 1
        rewards.append(total_reward)
        print(
            f"  episode {episode + 1:>{len(str(args.episodes))}}: reward = {total_reward:.2f}"
        )

    mean = sum(rewards) / len(rewards)
    print(f"\nmean reward over {args.episodes} episodes: {mean:.2f}")

    if args.save:
        df = pd.DataFrame(
            rows,
            columns=["episode", "step"] + obs_cols + act_cols + ["reward"],
        )
        df.to_csv(args.save, index=False)
        print(f"Results saved to {args.save}")

    env.close()


if __name__ == "__main__":
    main()
