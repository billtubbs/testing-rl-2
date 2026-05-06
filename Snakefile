DEFAULT_SEEDS = list(range(5))

EXPERIMENTS = {
    "dL-vL-tqc": {
        "env": "CartPole-BT-dL-vL-v1",
        "algo": "tqc",
        "timesteps": 150_000,
    },
    "p2-dL-tqc": {
        "env": "CartPole-BT-p2-dL-v1",
        "algo": "tqc",
        "timesteps": 150_000,
    },
    "x2-dL-tqc": {
        "env": "CartPole-BT-x2-dL-v1",
        "algo": "tqc",
        "timesteps": 150_000,
    },
}


def all_targets():
    targets = []
    for name, exp in EXPERIMENTS.items():
        for seed in exp.get("seeds", DEFAULT_SEEDS):
            targets.append(
                f"results/models/{name}/seed_{seed}/{exp['timesteps']}ts.zip"
            )
    return targets


rule all:
    input:
        all_targets()


rule train:
    wildcard_constraints:
        timesteps=r"\d+"
    output:
        "results/models/{exp}/seed_{seed}/{timesteps}ts.zip"
    run:
        exp = EXPERIMENTS[wildcards.exp]
        shell(
            "python scripts/train.py"
            f" --env {exp['env']}"
            f" --algo {exp['algo']}"
            f" --name {wildcards.exp}"
            f" --seed {wildcards.seed}"
            f" --timesteps {wildcards.timesteps}"
        )
