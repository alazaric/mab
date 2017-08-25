'''
Arm distributed as a Bernoulli random variable
'''

from Arm import Arm
import random as r


class Bernoulli(Arm):

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.expectation = p

    def draw(self):
        return float(r.random() < self.p)

    def __str__(self):
        return str(self.expectation)