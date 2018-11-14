"""Module that contains the finder class and the main code to run trials
"""
from landscape import Landscape
from finder_utils import debug_print, set_debug_level, error_print
import numpy as np

DEFAULT_DIM = 10
DEFAULT_MOVE_VECTORS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])


class LsFinder:
    """Performs the probabilistic search on a randomly generated landscape
    """

    def __init__(self, dim_arg=DEFAULT_DIM,
                 move_vec_arg=DEFAULT_MOVE_VECTORS):
        self.dim = dim_arg
        self.move_vectors = move_vec_arg
        self.landscape = Landscape(self.dim)
        self.cur_location = ([int(self.dim/2), int(self.dim/2)])
        self.cur_location = None
        self.default_chance = 1/(self.dim*self.dim)
        self.likelihood = np.ones(
            shape=(self.dim, self.dim))
        self.likelihood *= self.default_chance
        self.query_counter = np.zeros((self.dim, self.dim))

    def reset_all(self):
        """Resets the finder with a new map and recenters the cur_location
        """
        self.landscape = Landscape(self.dim)
        self.cur_location = ([int(self.dim/2), int(self.dim/2)])
        self.likelihood = np.ones(
            shape=(self.dim, self.dim))
        self.likelihood *= self.default_chance
        self.query_counter = np.zeros((self.dim, self.dim))

    def reset_finder(self):
        """Resets the finder with the same map and recenters the cur_location
        """
        self.cur_location = ([int(self.dim/2), int(self.dim/2)])
        self.likelihood = np.ones(
            shape=(self.dim, self.dim))
        self.likelihood *= self.default_chance
        self.query_counter = np.zeros((self.dim, self.dim))

    def find(self, coords):
        debug_print(f"Searching in spot [{coords}]", 8)
        self.query_counter[coords] += 1
        if self.landscape.query_tile(coords):
            # print(f"Target has been found in [{x},{y}]")
            return True
        self.bayes_update_on_miss(coords)
        # print(self.likelihood)
        return False

    def bayes_update_on_miss(self, miss_coords):
        """Performs the bayesian update on a miss on a certain tile

            Parameters
            ----------
            miss_coords : (int, int)
                Coordinate tuple of where the miss observation occurred

            Returns
            -------
            float
                the new probability that is in the miss_coords spot
        """
        false_neg_chance = self.landscape.prob_map[miss_coords]
        original_belief = self.likelihood[miss_coords]
        new_belief = original_belief * false_neg_chance
        self.likelihood[miss_coords] = new_belief
        self.likelihood /= np.sum(self.likelihood)
        return new_belief

    # def bayes_update_on_move(self, )

    def search_target(self, rule_num, is_local=False, is_target_moving=False):
        """Runs a certain algorithm to find the target on a certain map
        """

        if rule_num == 1:
            def get_next_tile(self):
                index = np.argmax(self.likelihood)
                return np.unravel_index(index, (self.dim, self.dim))
        elif rule_num == 2:
            def get_next_tile(self):
                index = np.argmax(np.multiply(self.likelihood,
                                              1 - self.landscape.prob_map))
                return np.unravel_index(index, (self.dim, self.dim))

    def search_rule1(self):
        search_index = np.argmax(self.likelihood)
        x = int(search_index/self.dim)
        y = search_index % self.dim
        return self.find((x, y))

    def simple_wander_search_r1(self):
        moves = self.move_vectors + self.cur_location
        scores = np.array(list(map(self.rule1, moves)))
        next_move = moves[np.argmax(scores)]
        next_move = tuple(next_move)
        # print(next_move)
        self.cur_location = next_move
        return self.find(next_move)

    def simple_wander_search_r2(self):
        moves = self.move_vectors + self.cur_location
        scores = np.array(list(map(self.rule2, moves)))
        next_move = moves[np.argmax(scores)]
        next_move = tuple(next_move)
        # print(next_move)
        self.cur_location = next_move
        return self.find(next_move)

    def rule1(self, coords):
        if not self.in_bounds(coords[0], coords[1]):
            return 0
        return self.likelihood[coords[0]][coords[1]]

    def rule2(self, coords):
        if not self.in_bounds(coords[0], coords[1]):
            return 0
        return (np.multiply(self.likelihood, 1 - self.landscape.prob_map)
                [coords[0]][coords[1]])

    def in_bounds(self, x, y):
        return 0 <= x < self.dim and 0 <= y < self.dim

    def search_rule2(self):
        # prob of being found = prob of being in * (1 - false negative)
        search_index = np.argmax(np.multiply(
            self.likelihood, 1 - self.landscape.prob_map))
        x = int(search_index/self.dim)
        y = search_index % self.dim
        return self.find((x, y))

    def run_trials(self, num_trials, search_rule):
        total_steps = 0
        for __ in range(0, num_trials):
            self.reset_all()
            target_found = False
            while not target_found:
                target_found = search_rule()
                total_steps += 1
        return total_steps/num_trials

    def search_moving_target(self):
        pass

    def bayes_update_on_move(self):
        pass


if __name__ == '__main__':
    set_debug_level(8)
    finder = LsFinder()
    num_test = 100
    search_functions = [finder.search_rule1,
                        finder.search_rule2,
                        finder.simple_wander_search_r1,
                        finder.simple_wander_search_r2]
    for func in search_functions:
        avg_steps = finder.run_trials(num_test, func)
        print(f"Average number of steps for\
              {func.__name__} is [{avg_steps}]")
    print("done")
