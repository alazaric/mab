'''
Basic generic class used to define the interface
'''


class Arm:

    def __init__(self):
        self.expectation = 0.0

    def draw(self): pass

    def __str__(self):
        return str(self.expectation)
