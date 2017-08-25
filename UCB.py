'''
UCB bandit strategy
'''
from Algorithm import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class UCB(Algorithm):

    def __init__(self, n_arms, parameters):
        super().__init__(n_arms)

        # number of pulls per arm
        self.n_pulls = np.zeros(n_arms, np.int64)

        # cumulative reward per arm
        self.cum_rew = np.zeros(n_arms, np.float64)

        # scaling parameter for confidence intervals
        self.scaling = parameters.scaling

    def get_action(self, t):

        if t <= self.n_arms:
            # perform round robin over arms in the first self.n_arms steps
            best_arm = t % self.n_arms
        else:
            # compute the average reward
            avg_reward = self.cum_rew / self.n_pulls

            # compute the uncertainty
            conf = self.scaling * np.sqrt(np.log(t+1) / self.n_pulls)

            # compute arms indices
            score = avg_reward + conf

            # find the arm(s) with the largest score and select one at random
            best_arms = np.argwhere(score == np.amax(score))
            best_arm = np.random.choice(best_arms.flatten(), 1)[0]

        # return the arm with the largest score
        return best_arm

    def update(self, t, arm, reward):
        self.cum_rew[arm] += reward
        self.n_pulls[arm] += 1

    def get_pulls(self):
        return self.n_pulls

    def __str__(self):
        np.set_printoptions(precision=3)

        state = "## Parameters\n"
        state += "num arms : " + str(self.n_arms) + "\n"
        state += "scaling  : " + str(self.scaling) + "\n"
        state += "## Current state \n"

        state += "pulls " + str(self.n_pulls) + "\n"
        state += "cum_rwd " + str(self.cum_rew) + "\n"
        state += "avg_rwd " + str(self.cum_rew/self.n_pulls) + "\n"

        return state
