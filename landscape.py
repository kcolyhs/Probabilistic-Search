"""Module that handles the target and terrain functionality"""

import random
import numpy as np

TARGET_MOVE_VECTORS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
NUM_TARGET_MOVES = np.size(TARGET_MOVE_VECTORS)

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
    def __init__(self, dim):
        self.dim = dim
        self.target = (0, 0)
        self.generate_map()
        self.is_target_moving = False

    def generate_map(self):
        """Generates a new random map and places the target randomly inside
        """
        dim = self.dim

        def val_to_prob(value):
            """Maps the uniform distribution to the distribution of false neg
            """
            if value < 0 or value >= 1:
                # print(f"ERROR: probability value ({value}) out of bounds")
                return None
            if value < .2:
                return .1
            if value < .5:
                return .3
            if value < .8:
                return .7
            return .9

        self.landscape_dist = np.random.uniform(0, 1, size=(dim, dim))
        self.prob_map = np.zeros((dim, dim))
        # Maps the tiles to a false negative chance
        for map_x in range(0, dim):
            for map_y in range(0, dim):
                value = self.landscape_dist[map_x][map_y]
                self.prob_map[map_x][map_y] = val_to_prob(value)
        # Sets the target to a random tile in the map
        self.target = (random.randint(0, self.dim-1),
                       random.randint(0, self.dim-1))

    def move_target(self):
        """Randomly moves the target to an adjacent spot unbiased
        """
        new_locations = TARGET_MOVE_VECTORS + np.array(self.target)
        np.random.shuffle(new_locations)
        new_locations = list(map(tuple, new_locations))
        for location in new_locations:
            if self.in_bounds(location):
                self.target = location

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
    LS = Landscape(50)
    LS.generate_map()
    LS.move_target()
    LS.query_tile((1, 1))
