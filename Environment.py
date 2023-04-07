'''
This class contains the list of all arms and simulate pulls
'''
import typing as tp
from Arm import Arm

class Environment:

    def __init__(self, 
                 arms: tp.List[Arm]):
        self.arms = arms
        self.n_arms = len(arms)

    def pull_arm(self, 
                 arm_idx: int):
        return self.arms[arm_idx].draw()

    def __str__(self):
        out = "Number of arms: " + str(self.n_arms) + "\n"

        for i, a in enumerate(self.arms):
            out += "Arm " + str(i) + "  : " + a.__str__() + "\n"
        return out
