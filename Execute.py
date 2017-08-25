'''
Execute an experiment running an algorithm in an environment
'''

from Result import Result


def execute(exp,  # type : Experiment
            env,  # type : Environment
            alg,  # type : Algorithm
            ) -> Result:

    # number of steps of the experiment
    n_steps = exp.n_steps

    # structure used to store the results
    res = Result(exp, env, alg)

    for t in range(n_steps):
        if t % 100 == 0:
            print("Time step %d" % t)
            print(alg.__str__())

        # retrieve the arm the should be pulled
        arm_to_pull = alg.get_action(t)

        # pull the arm
        reward = env.pull_arm(arm_to_pull)

        # update the internal state of the algorithm
        alg.update(t, arm_to_pull, reward)

        # store this step
        res.store(t, arm_to_pull, reward)

    return res
