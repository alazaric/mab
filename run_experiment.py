import numpy as np
import matplotlib.pyplot as plt

from UCB import UCB
from Algorithm import Algorithm
from Experiment import Experiment
from Bernoulli import Bernoulli
from Environment import Environment
from StructType import StructType
from Result import Result
import typing as tp

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
        if t % 100 == 0:
            print("Time step %d" % t)
            print(alg.__str__())

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


# set up the backend for the plot
plt.rcdefaults()

# initialize W&B
wandb.init(
    project="MAB Project",
    notes="First single run experiment",
    tags=["first attempt", "ucb", "2arms"]
    )

# construct the arms and build the corresponding environment
env = Environment([Bernoulli(p) for p in [0.7, 0.3]])
n_arms = env.n_arms
print("### Environment\n" + str(env))

# prepare the algorithm
ucb_params = StructType()
ucb_params.scaling = 1.0
alg = UCB(n_arms, ucb_params)
print("### Algorithm\n" + str(alg))

# prepare the experiment
exp = Experiment(1000)

# execute the experiment and collect the results
res = execute(exp, env, alg)  # type : Result

# compute statistics
stats = res.compute_statistics()

# plot a chart with the cumulative regret
cum_exp_regret = np.cumsum(stats.exp_regret)
# print(cum_exp_regret)

plt.plot(np.arange(exp.n_steps), cum_exp_regret, linewidth=2.5, linestyle="-")
plt.legend(loc='upper left', frameon=False)
plt.xlabel("steps")
plt.ylabel("cumulative regret")


plt.savefig("exercice_2.png",dpi=72)
plt.show()


