import numpy as np
import matplotlib.pyplot as plt

from ucb import UCB
from Algorithm import Algorithm
from Experiment import Experiment
from bernoulli import Bernoulli
from Environment import Environment
from StructType import StructType
from Result import Result
import typing as tp
from omegaconf import DictConfig, OmegaConf
import hydra

import wandb

def execute(exp: Experiment,
            env: Environment,
            alg: Algorithm
            ) -> Result:

    # structure used to store the results
    res = Result(exp, env, alg)
    step_metrics: tp.Dict[str, float] = {}

    # number of steps of the experiment
    n_steps = exp.n_steps

    for t in range(n_steps):
        # if t % 100 == 0:
        #     print("Time step %d" % t)
        #     print(alg.__str__())

        # retrieve the arm the should be pulled
        arm_to_pull = alg.get_action(t)

        # pull the arm
        reward = env.pull_arm(arm_to_pull)

        # update the internal state of the algorithm and log it
        step_metrics.update(alg.update(t, arm_to_pull, reward))
        wandb.log(step_metrics)
        wandb.log({"reward": reward, "arm": arm_to_pull})

        # store this step
        res.store(t, arm_to_pull, reward)

    return res

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:

    # print config file
    # print(OmegaConf.to_yaml(cfg))

    # construct the arms and build the corresponding environment
    env = Environment([hydra.utils.instantiate(arm) for arm in cfg.environment.arms])
    # print("### Environment\n" + str(env))

    # prepare the algorithm
    alg = hydra.utils.instantiate(cfg.algorithm, n_arms=env.n_arms)
    # print("### Algorithm\n" + str(alg))

    # prepare the experiment
    exp = Experiment(cfg.experiment.n_steps)

    # initialize W&B
    wandb.init(
        project="MAB Project",
        group=f"{cfg.environment.name} - {cfg.algorithm._target_}",
        name=" ".join(map(str,list(cfg.algorithm.values()))),
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        )

    # execute the experiment
    execute(exp, env, alg)

    # finish W&B run
    wandb.finish()


if __name__ == '__main__':
    main()