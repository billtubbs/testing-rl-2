"""Run a trained SB3 model on a Gymnasium environment."""

import argparse
import re
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401 — registers CartPole-BT environments
from stable_baselines3 import A2C, PPO, SAC, TD3

ALGOS = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3}

# Matches paths of the form: .../results/models/{env}/{algo}/seed_{seed}/...
_PATH_RE = re.compile(r"results/models/(?P<env>[^/]+)/(?P<algo>[^/]+)/seed_(?P<seed>\d+)/")


def _infer_from_path(path):
    m = _PATH_RE.search(path)
    if m:
        return m.group("env"), m.group("algo"), int(m.group("seed"))
    return None, None, None


def _infer_algo_from_path(path):
    parts = path.lower().replace("\\", "/").split("/")
    for part in parts:
        for algo in ALGOS:
            if algo in part:
                return algo
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to saved model (.zip)")
    parser.add_argument("--env", default=None, help="environment ID (inferred from path if omitted)")
    parser.add_argument("--algo", choices=ALGOS, default=None, help="algorithm (inferred from path if omitted)")
    parser.add_argument("--seed", type=int, default=None, help="environment seed (defaults to training seed + 1)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    inferred_env, inferred_algo, inferred_seed = _infer_from_path(args.model)

    env_id = args.env or inferred_env
    algo_name = args.algo or inferred_algo or _infer_algo_from_path(args.model)
    seed = args.seed if args.seed is not None else (
        inferred_seed + 1 if inferred_seed is not None else 0
    )

    if env_id is None:
        parser.error("could not infer environment from path; provide --env")
    if algo_name is None:
        parser.error("could not infer algorithm from path or filename; provide --algo")

    render_mode = "human" if args.render else None
    env = gym.make(env_id, render_mode=render_mode)

    model = ALGOS[algo_name].load(args.model, env=env)

    print(f"Running {algo_name.upper()} on {env_id} for {args.episodes} episodes (seed={seed})...")
    if inferred_seed is not None and args.seed is None:
        print(f"(trained with seed={inferred_seed})")
    print()

    rewards = []
    obs, _ = env.reset(seed=seed)
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"  episode {episode + 1:>{len(str(args.episodes))}}: reward = {total_reward:.2f}")

    mean = sum(rewards) / len(rewards)
    print(f"\nmean reward over {args.episodes} episodes: {mean:.2f}")

    env.close()


if __name__ == "__main__":
    main()
