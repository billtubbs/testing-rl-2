"""Evaluate all trained models and baselines, saving results to CSV.

Runs each trained model from experiments.yaml with --n-seeds evaluation seeds (default 10),
plus LQR (full-state envs) and LQG (p2 envs) baselines.
Output: results/eval/combined.csv
"""

import argparse
import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401
from stable_baselines3 import A2C, PPO, SAC, TD3
from sb3_contrib import TQC

logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
logging.getLogger("sb3_contrib").setLevel(logging.WARNING)

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
from lqr_agent import LQRAgent
from lqg_agent import LQGAgent

ALGOS = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3, "tqc": TQC}
N_EVAL_STEPS = 100  # max_episode_steps for these envs


def expand_experiment(name, exp):
    if "seeds" not in exp:
        raise ValueError(f"Experiment '{name}': 'seeds' is required")
    lengths = {len(v) for v in exp.values() if isinstance(v, list)}
    if len(lengths) > 1:
        raise ValueError(
            f"Experiment '{name}': all lists must have the same length"
        )
    n = next(iter(lengths))
    return [
        {k: v[i] if isinstance(v, list) else v for k, v in exp.items()}
        for i in range(n)
    ]


def run_episode(env, agent, eval_seed):
    obs, _ = env.reset(seed=eval_seed)
    total_reward = 0.0
    terminated = truncated = False
    episode_start = True
    state = None
    n_steps = 0
    while not (terminated or truncated):
        action, state = agent.predict(
            obs,
            deterministic=True,
            state=state,
            episode_start=episode_start,
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_start = False
        n_steps += 1
    return total_reward, n_steps


def load_model(algo_name, model_path, env):
    cls = ALGOS[algo_name]
    return cls.load(model_path, env=env, verbose=0)


def find_model_path(exp_name, seed):
    return os.path.join(
        _PROJECT_DIR,
        "results",
        "models",
        exp_name,
        f"seed_{seed}",
        "best_model.zip",
    )


def is_partial_obs(env):
    """True if the env uses a non-identity output matrix."""
    C = env.unwrapped.output_matrix
    n = C.shape[1]
    return not np.array_equal(C, np.eye(n, dtype=C.dtype))


def make_baseline(env):
    """Return LQR or LQG agent depending on observability."""
    if is_partial_obs(env):
        return LQGAgent.from_env(env)
    else:
        return LQRAgent.from_env(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-seeds", type=int, default=100, help="Number of evaluation seeds"
    )
    args = parser.parse_args()
    eval_seeds = range(100, 100 + args.n_seeds)

    yaml_path = os.path.join(_PROJECT_DIR, "experiments.yaml")
    with open(yaml_path) as f:
        experiments = yaml.safe_load(f)["experiments"]

    results_dir = os.path.join(_PROJECT_DIR, "results", "eval")
    os.makedirs(results_dir, exist_ok=True)

    records = []

    print("Running evaluation experiments...")

    for exp_name, exp in experiments.items():
        env_id = exp["env"]
        algo_name = exp["algo"]
        env = gym.make(env_id)

        for run in expand_experiment(exp_name, exp):
            train_seed = run["seeds"]
            timesteps = run["timesteps"]
            model_path = find_model_path(exp_name, train_seed)

            if not os.path.exists(model_path):
                print(f"  Missing: {model_path} — skipping")
                continue

            model = load_model(algo_name, model_path, env)

            for eval_seed in eval_seeds:
                total_reward, n_steps = run_episode(env, model, eval_seed)
                records.append(
                    {
                        "experiment": exp_name,
                        "env": env_id,
                        "algo": algo_name,
                        "train_seed": train_seed,
                        "timesteps": timesteps,
                        "eval_seed": eval_seed,
                        "n_steps": n_steps,
                        "total_reward": total_reward,
                    }
                )
            print(
                f"  {exp_name}  {env_id}  {algo_name}  train_seed={train_seed}  n={args.n_seeds}"
            )

        # Baseline
        baseline = make_baseline(env)
        baseline_algo = "lqg" if is_partial_obs(env) else "lqr"
        for eval_seed in eval_seeds:
            total_reward, n_steps = run_episode(env, baseline, eval_seed)
            records.append(
                {
                    "experiment": exp_name,
                    "env": env_id,
                    "algo": baseline_algo,
                    "train_seed": None,
                    "timesteps": None,
                    "eval_seed": eval_seed,
                    "n_steps": n_steps,
                    "total_reward": total_reward,
                }
            )
        print(f"  {exp_name}  {env_id}  {baseline_algo}  n={args.n_seeds}")

        env.close()

    df = pd.DataFrame(records)
    out_path = os.path.join(results_dir, "combined.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(
        df.groupby(["experiment", "algo"])["total_reward"].mean().to_string()
    )


if __name__ == "__main__":
    main()
