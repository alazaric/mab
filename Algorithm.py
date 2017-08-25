'''
Generic algorithm class
'''


class Algorithm:

    def __init__(self, n_arms):
        self.n_arms = n_arms

    def get_action(self): pass

    def update(self, arm, reward): pass
