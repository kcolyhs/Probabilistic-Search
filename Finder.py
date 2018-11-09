from Landscape import Landscape
import numpy as np


class LsFinder:
    default_dim = 50

    def __init__(self):
        self.new_landscape()
        default_chance = 1/(50*50)
        self.likelihood = np.ones(
                shape=(LsFinder.default_dim, LsFinder.default_dim))
        self.likelihood *= default_chance

    def new_landscape(self):
        self.ls = Landscape(LsFinder.default_dim)

    def bayesian_update(self, miss_tile):
        pass

    def find(self):
        pass


if __name__ == '__main__':
    finder = LsFinder()
