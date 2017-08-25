'''
This class contains the list of all arms and simulate pulls
'''


class Environment:

    def __init__(self, arms):
        self.arms = arms
        self.n_arms = len(arms)

    def pull_arm(self, arm):
        return self.arms[arm].draw()

    def __str__(self):
        out = "Number of arms: " + str(self.n_arms) + "\n"

        for i, a in enumerate(self.arms):
            out += "Arm " + str(i) + "  : " + a.__str__() + "\n"
        return out
