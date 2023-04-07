'''
Execute an experiment running an algorithm in an environment
'''

from Algorithm import Algorithm
from Environment import Environment
from Experiment import Experiment
from Result import Result
import typing as tp

def execute(exp: Experiment,
            env: Environment,
            alg: Algorithm
            ) -> Result:

    # structure used to store the results
    res = Result(exp, env, alg)
    step_metrics: tp.Dict[str, float] = {}
    experiment_metrics: tp.Dict[str, float] = {}

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

        # update the internal state of the algorithm and store it
        step_metrics.update(alg.update(t, arm_to_pull, reward))
        print(f"step {t}")
        print(step_metrics)

        # store this step
        res.store(t, arm_to_pull, reward)

    return res
