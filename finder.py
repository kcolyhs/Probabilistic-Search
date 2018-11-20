"""Module that contains the finder class and the main code to run trials
"""
import numpy as np
from landscape import Landscape
from finder_utils import debug_print, set_debug_level, error_print

DEFAULT_DIM = 5
DEFAULT_MOVE_VECTORS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])


class LsFinder:
    """Performs the probabilistic search on a randomly generated landscape
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, dim_arg=DEFAULT_DIM,
                 move_vec_arg=DEFAULT_MOVE_VECTORS):
        self.dim = dim_arg
        self.move_vectors = move_vec_arg
        self.landscape = Landscape(self.dim)
        self.cur_location = ([int(self.dim/2), int(self.dim/2)])
        self.path_target = None
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

    def search_target(self, rule_num, search_approach,
                      target_moving_arg=False):
        """Runs a certain algorithm to find the target on a certain map
        """

        total_steps = 0
        self.landscape.is_target_moving = target_moving_arg
        if rule_num == 1:
            def score_matrix():
                return self.likelihood
        elif rule_num == 2:
            def score_matrix():
                return np.multiply(self.likelihood, 1-self.landscape.prob_map)
        else:
            error_print("Invalid rule number", 0)
            return -1

        def coords_to_chance(coords):
            coords = tuple(coords)
            if not self.in_bounds(coords):
                return 0
            scores = score_matrix()
            return scores[coords]

        def get_most_likely_global():
            return np.argmax(score_matrix())

        search_approach = search_approach.lower()
        if search_approach == 'global':
            def get_next_tile_func():
                index = get_most_likely_global()
                return np.unravel_index(index, (self.dim, self.dim))
            get_next_tile = get_next_tile_func
        elif search_approach == 'local':
            def get_next_tile_func():
                np.random.shuffle(self.move_vectors)
                moves = self.move_vectors + self.cur_location
                scores = np.array(list(map(coords_to_chance, moves)))
                next_move = moves[np.argmax(scores)]
                next_move = tuple(next_move)
                # print(next_move)
                self.cur_location = next_move
                return next_move
            get_next_tile = get_next_tile_func
        elif search_approach == 'path':
            # TODO implement path
            pass

        target_found = False
        while not target_found:
            total_steps += 1
            next_move = get_next_tile()
            print(self.likelihood[next_move])
            target_found = self.find(next_move)
            if target_found:
                return total_steps
            self.bayes_update_on_miss(next_move)
            if target_moving_arg is True:
                self.bayes_update_on_move()
        return total_steps

    def in_bounds(self, coords):
        return self.landscape.in_bounds(coords)

    def run_trials(self, num_trials, search_rule):
        total_steps = 0
        for __ in range(0, num_trials):
            self.reset_all()
            target_found = False
            while not target_found:
                target_found = search_rule()
                total_steps += 1
        return total_steps/num_trials

    def bayes_update_on_move(self):
        # Fetch the observation on boundry crossing from the landscape
        clue = self.landscape.get_last_transition()
        if clue is None:
            return
        target_move_vectors = Landscape.get_target_move_vectors()

        other_t_ids = np.zeros((self.dim, self.dim)) - 1
        num_moves_arr = np.zeros((self.dim, self.dim))
        for x in range(self.dim):
            for y in range(self.dim):
                coords = (x, y)
                t_id = self.landscape.t_id_map[coords]
                if t_id not in clue:
                    other_t_id = -1
                    continue
                if t_id == clue[0]:
                    other_t_id = clue[1]
                else:
                    other_t_id = clue[0]
                other_t_ids[coords] = other_t_id

                # Calculate the number of moves from (x,y) that match the
                # transition description
                neighbors = coords + target_move_vectors
                for neighbor in neighbors:
                    neighbor = tuple(neighbor)
                    if not self.in_bounds(neighbor):
                        continue
                    if self.landscape.t_id_map[neighbor] == other_t_id:
                        num_moves_arr[coords] += 1

        new_belief = np.zeros((self.dim, self.dim))
        for x in range(self.dim):
            for y in range(self.dim):
                coords = (x, y)
                # Get the t_id of the terrain we're on and the prev terrain
                t_id = self.landscape.t_id_map[coords]
                other_t_id = other_t_ids[coords]
                if other_t_id == -1:
                    continue

                poss_prev_locations = coords - target_move_vectors

                for neighbor in poss_prev_locations:
                    neighbor = tuple(neighbor)
                    if not self.in_bounds(neighbor):
                        continue
                    # neighbor is
                    if self.landscape.t_id_map[neighbor] != other_t_id:
                        # Filter out wrong t_id
                        continue
                    num_trans = num_moves_arr[neighbor]
                    new_belief[coords] += ((1/num_trans)
                                           * self.likelihood[neighbor])

        self.likelihood = new_belief
        self.likelihood /= np.sum(self.likelihood)


if __name__ == '__main__':
    set_debug_level(5)
    FINDER = LsFinder()
    num_test = 1
    # SEARCH_FUNCTIONS = [
    #     FINDER.search_rule1,
    #     FINDER.search_rule2,
    #     FINDER.simple_wander_search_r1,
    #     FINDER.simple_wander_search_r2,
    #     ]
    # for func in SEARCH_FUNCTIONS:
    #     avg_steps = FINDER.run_trials(num_test, func)
    #     print(f"Average number of steps for\
    #           {func.__name__} is [{avg_steps}]")
    print(f'''Target located at {FINDER.landscape.target}
          and is in t_id {FINDER.landscape.t_id_map[FINDER.landscape.target]}
          ''')
    # TEST 2 bayesian update on
    # print(f'Terrain:\n{FINDER.landscape.t_id_map}\n')
    # print(f'Likelihood:\n{FINDER.likelihood}\n')

    # FINDER.landscape.move_target()
    # print(f'Last transition: {FINDER.landscape.get_last_transition()}\n')
    # FINDER.bayes_update_on_move()
    # print(f'Likelihood:\n{FINDER.likelihood}\n')

    # FINDER.landscape.move_target()
    # print(f'Last transition: {FINDER.landscape.get_last_transition()}\n')
    # FINDER.bayes_update_on_move()
    # print(f'Likelihood:\n{FINDER.likelihood}\n')

    # Test 3
    avg = 0
    for _ in range(num_test):
        x = FINDER.search_target(2, "local", target_moving_arg=True)
        # FINDER.reset_finder()
        FINDER.reset_finder()
        avg += x
    avg /= num_test
    print(f'average = {avg}')
    print("done")
