configfile: "experiments.yaml"

EXPERIMENTS = config["experiments"]


def expand_experiment(name, exp):
    """Expand an experiment dict into a list of per-run configs.

    Scalar values are broadcast to match the length of any list values.
    All list values must have the same length.
    """
    if "seeds" not in exp:
        raise ValueError(f"Experiment '{name}': 'seeds' is required")
    lengths = {len(v) for v in exp.values() if isinstance(v, list)}
    if len(lengths) > 1:
        raise ValueError(
            f"Experiment '{name}': all lists must be the same length, got {lengths}"
        )
    n = next(iter(lengths))
    return [
        {k: v[i] if isinstance(v, list) else v for k, v in exp.items()}
        for i in range(n)
    ]


def all_targets():
    targets = []
    for name, exp in EXPERIMENTS.items():
        for run in expand_experiment(name, exp):
            targets.append(
                f"results/models/{name}/seed_{run['seeds']}/{run['timesteps']}ts.zip"
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
        runs = expand_experiment(wildcards.exp, EXPERIMENTS[wildcards.exp])
        cfg = next(r for r in runs if r["seeds"] == int(wildcards.seed))
        shell(
            "python scripts/train.py"
            f" --env {cfg['env']}"
            f" --algo {cfg['algo']}"
            f" --name {wildcards.exp}"
            f" --seed {wildcards.seed}"
            f" --timesteps {wildcards.timesteps}"
        )


rule eval_all:
    input:
        all_targets()
    output:
        "results/eval/combined.csv"
    shell:
        "python scripts/eval_all.py"
