# testing-rl-2

Testing common reinforcement learning algorithms on a realistic benchmark control problem — the CartPole-BT family of environments.

## Environment

The [gym-CartPole-bt](https://github.com/billtubbs/gym-CartPole-bt) environment extends the classic CartPole problem with a more realistic physical model, continuous action spaces, and multiple variants of varying difficulty. It provides a practical benchmark for continuous control algorithms.

## Algorithms

Experiments are run using [Stable Baselines3](https://stable-baselines3.readthedocs.io/) and [SB3-Contrib](https://sb3-contrib.readthedocs.io/):

| Algorithm | Type | Notes |
|-----------|------|-------|
| A2C | On-policy | Advantage Actor-Critic |
| PPO | On-policy | Proximal Policy Optimization |
| SAC | Off-policy | Soft Actor-Critic |
| TD3 | Off-policy | Twin Delayed DDPG |
| TQC | Off-policy | Truncated Quantile Critics (sb3-contrib) |
| LQR | Benchmark | Linear Quadratic Regulator (analytical, no training) |
| LQG | Benchmark | Linear Quadratic Gaussian (analytical, no training) |

## Installation

```bash
pip install -e ".[dev]"
```

The `gym-CartPole-bt` environment must be installed separately:

```bash
cd ~/path/to/gym-CartPole-bt && pip install -e .
```

## Running experiments

Experiments are managed with [Snakemake](https://snakemake.readthedocs.io/). Edit `Snakefile` to configure the environments, algorithms, seeds, and number of timesteps, then run:

```bash
# Dry run — shows what would be executed
snakemake --dry-run

# Run all experiments (adjust --cores to suit your machine)
snakemake --cores 4
```

Completed runs are skipped automatically on subsequent invocations.

Results are saved to `results/models/{env}/{algo}/seed_{seed}/`.

## Running a trained model

```bash
# Run one episode
python scripts/run_model.py <env_id> results/models/<env>/<algo>/seed_<seed>/<algo>_final.zip --algo <algo>

# Run with animation
python scripts/run_model.py <env_id> results/models/<env>/<algo>/seed_<seed>/<algo>_final.zip --algo <algo> --render

# Run the LQR benchmark (no model file required)
python scripts/run_model.py <env_id> --algo lqr --render

# Save full episode data (obs, actions, rewards) to CSV
python scripts/run_model.py <env_id> results/models/<env>/<algo>/seed_<seed>/<algo>_final.zip --algo <algo> --save results/run.csv
```

## TensorBoard

Training metrics are logged to `results/tb_logs/`. To view:

```bash
tensorboard --logdir results/tb_logs
```
