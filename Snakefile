ENVS = ["CartPole-BT-vL-v1"]
ALGOS = ["a2c", "ppo", "sac", "td3"]
SEEDS = list(range(5))
TIMESTEPS = 100_000


rule all:
    input:
        expand(
            "results/models/{env}/{algo}/seed_{seed}/{algo}_final.zip",
            env=ENVS,
            algo=ALGOS,
            seed=SEEDS,
        )


rule train:
    output:
        "results/models/{env}/{algo}/seed_{seed}/{algo}_final.zip"
    shell:
        "python scripts/train.py"
        " --env {wildcards.env}"
        " --algo {wildcards.algo}"
        " --seed {wildcards.seed}"
        " --timesteps " + str(TIMESTEPS)
