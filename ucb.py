'''
UCB bandit strategy
'''
from omegaconf import DictConfig
from Algorithm import *
import numpy as np
import typing as tp

np.seterr(divide='ignore', invalid='ignore')

class UCB(Algorithm):

    def __init__(self, 
                 n_arms: int, 
                 scaling: float):
        super().__init__(n_arms)

        # number of pulls per arm
        self.n_pulls = np.zeros(n_arms, np.int64)

        # cumulative reward per arm
        self.cum_rew = np.zeros(n_arms, np.float64)

        # average reward per arm
        self.avg_rew = np.zeros(n_arms, np.float64)

        # confidence intervals per arm
        self.conf_interval = np.ones(n_arms, np.float64) * np.inf

        # scaling parameter for confidence intervals
        self.scaling = scaling

    def get_action(self, 
                   t: int) -> int:

        if t <= self.n_arms:
            # perform round robin over arms in the first self.n_arms steps
            best_arm = t % self.n_arms
        else:
            # compute arms indices
            ucb_score = self.avg_reward + self.scaling * self.conf_interval

            # find the arm(s) with the largest score and select one at random
            best_arms = np.argwhere(ucb_score == np.amax(ucb_score))
            best_arm = np.random.choice(best_arms.flatten(), 1)[0]

        # return the arm with the largest score
        return best_arm

    def update(self, 
               t: int, 
               arm: int, 
               reward: float) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        # increment the cumulative reward and number of pulls for the observed arm
        self.cum_rew[arm] += reward
        self.n_pulls[arm] += 1

        # recompute the average reward and confidence interval for all arms
        self.avg_reward = self.cum_rew / self.n_pulls
        self.conf_interval = np.sqrt(np.log(t+1) / self.n_pulls)

        # returns internal variables for logging purposes
        for i in range(self.n_arms):
            metrics[f"avg_reward_arm{i}"] = self.avg_reward[i]
            metrics[f"n_pulls{i}"] = self.n_pulls[i]
            metrics[f"conf_interval{i}"] = self.conf_interval[i]
            metrics[f"ucb_score{i}"] = self.avg_reward[i] + self.scaling * self.conf_interval[i]
        return metrics

    def get_pulls(self) -> np.ndarray:
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
