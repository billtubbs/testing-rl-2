"""Train an RL agent on a Gymnasium environment."""

import argparse
import glob
import os
import gymnasium as gym
import gym_CartPole_BT  # noqa: F401 — registers CartPole-BT environments
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import TQC

DEFAULT_ENV_ID = "CartPole-BT-vL-v1"
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10

ALGOS = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3, "tqc": TQC}


def find_latest_checkpoint(model_dir):
    """Return (path, timesteps) of the highest-timestep checkpoint, or (None, 0)."""
    checkpoints = []
    for path in glob.glob(os.path.join(model_dir, "*ts.zip")):
        stem = os.path.splitext(os.path.basename(path))[0]  # e.g. "150000ts"
        try:
            ts = int(stem.removesuffix("ts"))
            checkpoints.append((ts, path))
        except ValueError:
            pass
    if not checkpoints:
        return None, 0
    ts, path = max(checkpoints)
    return path, ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV_ID)
    parser.add_argument("--algo", choices=ALGOS, default="ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--name", default=None,
                        help="Experiment name, used as the results subdirectory. "
                             "Defaults to the algorithm name.")
    parser.add_argument(
        "--restart", action="store_true",
        help="Start from scratch even if a checkpoint exists.",
    )
    args = parser.parse_args()

    if args.name is None:
        args.name = args.algo

    log_dir = f"results/logs/{args.name}/seed_{args.seed}"
    tb_log_dir = "results/tb_logs"
    model_dir = f"results/models/{args.name}/seed_{args.seed}"

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

    checkpoint_path, checkpoint_ts = find_latest_checkpoint(model_dir)

    if not args.restart and checkpoint_path and args.timesteps <= checkpoint_ts:
        print(
            f"Warning: requested {args.timesteps:,} timesteps but checkpoint already "
            f"has {checkpoint_ts:,}. Restarting from scratch."
        )
        args.restart = True

    if args.restart or checkpoint_path is None:
        if args.restart and checkpoint_path:
            print(f"Restarting (ignoring checkpoint at {checkpoint_ts:,} timesteps).")
        model = ALGOS[args.algo](
            "MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log=tb_log_dir,
        )
        remaining = args.timesteps
        reset_num_timesteps = True
    else:
        print(f"Resuming from {checkpoint_path} ({checkpoint_ts:,} timesteps done).")
        model = ALGOS[args.algo].load(
            checkpoint_path, env=env, verbose=1, tensorboard_log=tb_log_dir,
        )
        remaining = args.timesteps - checkpoint_ts
        reset_num_timesteps = False

    print(
        f"Training {args.algo.upper()} on {args.env} for {remaining:,} timesteps"
        + (f" (resuming to {args.timesteps:,} total, seed={args.seed})..."
           if not reset_num_timesteps else f" (seed={args.seed})...")
    )
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"Run: tensorboard --logdir {tb_log_dir}\n")

    model.learn(
        total_timesteps=remaining,
        callback=eval_callback,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=args.name,
    )

    save_path = os.path.join(model_dir, f"{args.timesteps}ts")
    model.save(save_path)

    env.close()
    eval_env.close()
    print("Done. Model saved to", save_path + ".zip")


if __name__ == "__main__":
    main()
