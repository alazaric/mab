'''
This class collects all the results of interest of an experiment
'''

import numpy as np

from StructType import StructType


class Result:

    def __init__(self,
                 exp,  # type : Experiment
                 env,  # type : Environment
                 alg,  # type : Algorithm
                 ):
        self.n_steps = exp.n_steps
        self.env = env
        self.alg = alg

        # stores all the sequence of arms and rewards observed over the whole experiment
        self.history = np.zeros((self.n_steps, 2))

    def store(self, t, arm, reward):
        # the time index t is assumed to be 0-based
        self.history[t][0] = arm
        self.history[t][1] = reward

    def compute_statistics(self):
        # store all expected values of arms in an array
        arm_rewards = np.zeros(self.env.n_arms)

        for i, arm in enumerate(self.env.arms):
            arm_rewards[i] = arm.expectation

        # compute the value and the index of the optimal arm (we assume only one arm is optimal)
        best_reward = np.amax(arm_rewards)
        best_arm = np.argmax(arm_rewards)

        # empirical and expected instantaneous regret
        emp_regret = np.zeros(self.n_steps)
        exp_regret = np.zeros(self.n_steps)

        # compute the empirical regret (using rewards) and expected regret (using expected rewards)
        for t in range(self.n_steps):
            arm_pulled = np.int(self.history[t][0])
            emp_regret[t] = best_reward - self.history[t][1]
            exp_regret[t] = best_reward - arm_rewards[arm_pulled]

        # collect all results in one structure
        statistics = StructType()
        statistics.emp_regret = emp_regret
        statistics.exp_regret = exp_regret

        # number of pulls to arms
        statistics.pulls = self.alg.get_pulls()

        return statistics
