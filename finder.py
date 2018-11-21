"""Module that contains the finder class and the main code to run trials
"""
import numpy as np
from landscape import Landscape
from finder_utils import debug_print, error_print, cell_dist, set_debug_level

DEFAULT_DIM = 10
DEFAULT_MOVE_VECTORS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])

KERNEL = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 5, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
        ])


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
        self.cur_path_target = None

    def reset_all(self):
        """Resets the finder with a new map and recenters the cur_location
        """
        self.landscape = Landscape(self.dim)
        self.cur_location = ([int(self.dim/2), int(self.dim/2)])
        self.likelihood = np.ones(
            shape=(self.dim, self.dim))
        self.likelihood *= self.default_chance
        self.query_counter = np.zeros((self.dim, self.dim))
        self.cur_path_target = None

    def reset_finder(self):
        """Resets the finder with the same map and recenters the cur_location
        """
        self.cur_location = ([int(self.dim/2), int(self.dim/2)])
        self.likelihood = np.ones(
            shape=(self.dim, self.dim))
        self.likelihood *= self.default_chance
        self.query_counter = np.zeros((self.dim, self.dim))
        self.cur_path_target = None

    def search_cell(self, coords):
        debug_print(f"Searching in spot [{coords}]", 8)
        self.query_counter[coords] += 1
        # searches the cell and moves the target if not found
        if self.landscape.query_tile(coords):
            # print(f"Target has been found in [{x},{y}]")
            return True
        self.bayes_update_on_miss(coords)
        if self.landscape.is_target_moving:
            self.bayes_update_on_move
        # print(self.likelihood)
        return False

    def bayes_update_on_miss(self, miss_coords):
        """Performs the bayesian update on a miss on a certain cell

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

    def move_on_path(self):
        # TODO implement path walking
        pos = self.cur_location
        target = self.cur_path_target
        if pos == target:
            return
        elif pos[0] < target[0]:
            self.cur_location = (pos[0]+1, pos[1])
            return
        elif pos[0] > target[0]:
            self.cur_location = (pos[0]-1, pos[1])
            return
        elif pos[1] < target[1]:
            self.cur_location = (pos[0], pos[1]+1)
            return
        elif pos[1] > target[1]:
            self.cur_location = (pos[0], pos[1]-1)
            return

    def search_target(self, rule_num, search_approach,
                      target_moving_arg=False):
        """Runs a certain algorithm to search for the target on a certain map
        """

        search_approach = search_approach.lower()
        total_steps = 0
        self.landscape.is_target_moving = target_moving_arg

        # Rule 1 -> likelihood that the target is in a cell
        # Rule 2 -> likelihood that the target is found in a cell
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
                return -1
            scores = score_matrix()
            return scores[coords]

        def best_global_cell():
            index = np.argmax(score_matrix())
            index = np.unravel_index(index, (self.dim, self.dim))
            return index

        def best_close_cell():
            '''Smarter algorithm that weighs in distance'''
            # TODO write this
            scores = np.copy(score_matrix())
            for x in range(self.dim):
                for y in range(self.dim):
                    coords = (x, y)
                    dist = cell_dist(coords, self.cur_location)
                    scores[coords] =

        # Set the decide_path and agent_search algorithm
        if search_approach == 'global':
            # No pathing always searches the best global
            def decide_path_global():
                self.cur_path_target = None
                return

            def agent_search_global():
                coords = best_global_cell()
                return self.search_cell(coords)

            decide_path = decide_path_global
            agent_search = agent_search_global
        elif search_approach == 'path_simple':
            # Travels to the best cell then searches it
            def decide_path_simple():
                if self.cur_path_target is None:
                    self.cur_path_target = best_global_cell()
                if self.cur_location == self.cur_path_target:
                    self.cur_path_target = None

            def agent_search_simple():
                return self.search_cell(self.cur_location)

            decide_path = decide_path_simple
            agent_search = agent_search_simple
        elif search_approach == 'path_smart':
            def decide_path_smart():
                if self.cur_path_target is None:
                    # TODO replace best_global_cell() with scoring func
                    self.cur_path_target = best_global_cell()
                if self.cur_location == self.cur_path_target:
                    self.cur_path_target = None

            def agent_search_smart():
                return self.search_cell(self.cur_location)

            decide_path = decide_path_smart
            agent_search = agent_search_smart

        target_found = False
        while not target_found:
            total_steps += 1
            # Sets the path to none if no movement needed
            decide_path()
            if self.cur_path_target is None:
                target_found = agent_search()
            else:
                self.move_on_path()
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

    def p_state(self):
        print(self.likelihood)
        print(self.cur_path_target)
        print(self.cur_location)


if __name__ == '__main__':
    set_debug_level(5)
    FINDER = LsFinder()
    num_test = 100
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

    # Test 3
    avg = 0
    for _ in range(num_test):
        x = FINDER.search_target(1, "global", target_moving_arg=True)
        # FINDER.reset_finder()
        FINDER.reset_finder()
        avg += x
    avg /= num_test
    print(f'average = {avg}')
    print("done")
