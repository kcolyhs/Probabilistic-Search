"""Module that handles the target and terrain functionality

    Globals
    -------
    TARGET_MOVE_VECTORS : np.array([int,int],...)
        These vectors describe the different movement options the target can
        take. You can add or remove diagnols by adding vectors to this array.
"""

import random
import numpy as np
from finder_utils import debug_print, set_debug_level, error_print

TARGET_MOVE_VECTORS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

# =============================================================================
# FLAT: P = .1  | DIST = .2
# HILLY: P = .3 | DIST = .3
# FOREST: P = .7| DIST = .3
# CAVE: P = .9  | DIST = .2
# =============================================================================


class Landscape:
    """Landscape handles the terrain map and target location.

        Attributes
        ---------

        dim : int
            dim is the dimensions of the square terrain map

        target : (int, int)
            Tuple containing the current location of the target

        is_target_moving : bool
            True when target is set to move after every query
    """
    terrain_names = ["Flat", "Hills", "Forested", "Caves"]
    # terrain_dist= [.2, .3, .3, .2]
    terrain_dist_thresh = [.2, .5, .8, 1]
    terrain_miss_prob = [.1, .3, .7, .9]

    def __init__(self, dim):
        self.dim = dim
        self.target = (0, 0)
        self.is_target_moving = False
        self.last_move = False
        self.generate_map()

    def generate_map(self):
        """Generates a new random map and places the target randomly inside
        """
        dim = self.dim
        landscape_dist = np.random.uniform(0, 1, size=(dim, dim))
        self.prob_map = np.zeros((dim, dim))
        self.t_id_map = np.zeros((dim, dim))
        # Maps the tiles to a false negative chance
        for map_x in range(0, dim):
            for map_y in range(0, dim):
                value = landscape_dist[map_x][map_y]
                t_id = 0
                for thresh in Landscape.terrain_dist_thresh:
                    if value >= thresh:
                        t_id += 1
                    else:
                        break
                debug_print(f'value:{value}, t_id: {t_id}', 10)
                self.t_id_map[map_x][map_y] = t_id
                self.prob_map[map_x][map_y] = Landscape.terrain_miss_prob[t_id]
        # Sets the target to a random tile in the map
        self.target = (random.randint(0, self.dim-1),
                       random.randint(0, self.dim-1))

    def move_target(self):
        """Randomly moves the target to an adjacent spot unbiased
        """
        old_loc = self.target
        new_locations = TARGET_MOVE_VECTORS + np.array(self.target)
        np.random.shuffle(new_locations)
        new_locations = list(map(tuple, new_locations))
        for new_loc in new_locations:
            if self.in_bounds(new_loc):
                self.target = new_loc
                self.last_move = (old_loc, new_loc)
                break

    def query_tile(self, coord):
        """Checks the tile indexed by coord for the target

            First checks if the target is in the right tile and then rolls for
            false positive. Next

            Parameters
            ----------
            coord : (int,int)
                coordinate tuple to check for target

            Returns
            -------
            bool
                Returns true if the search found the target
        """
        coord_x = coord[0]
        coord_y = coord[1]
        target_found = False
        if self.target == coord:
            if random.uniform(0, 1) > self.prob_map[coord_x][coord_y]:
                target_found = True
        # Moves the target if not found and if enabled
        if self.is_target_moving and not target_found:
            self.move_target()
        return target_found

    def in_bounds(self, coord):
        """Checks if coordinate tuple is within array bounds

            Parameters
            ----------
            coord: (int,int)
                coordinate tuple to be checked

            Returns
            -------
            bool
                A bool that is true if coord is a valid index in the landscape
        """
        coord_x = coord[0]
        coord_y = coord[1]
        return (0 <= coord_x < self.dim
                and 0 <= coord_y < self.dim)


if __name__ == '__main__':
    set_debug_level(5)
    LS = Landscape(50)
    LS.generate_map()
    LS.move_target()
    LS.query_tile((1, 1))
    x = LS
