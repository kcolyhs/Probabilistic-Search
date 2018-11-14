import random
import numpy as np

TARGET_MOVE_VECTORS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
num_target_moves = np.size(TARGET_MOVE_VECTORS)

# =============================================================================
# FLAT: P = .1  | DIST = .2
# HILLY: P = .3 | DIST = .3
# FOREST: P = .7| DIST = .3
# CAVE: P = .9  | DIST = .2
# =============================================================================


class Landscape:
    def __init__(self, dim):
        self.dim = dim
        self.target = (0, 0)
        self.generate_map()
        self.moving_target = False

    def generate_map(self):
        dim = self.dim

        def val_to_prob(value):
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
        for x in range(0, dim):
            for y in range(0, dim):
                value = self.landscape_dist[x][y]
                self.prob_map[x][y] = val_to_prob(value)

        self.target = (random.randint(0, self.dim-1),
                       random.randint(0, self.dim-1))

    def move_target(self):
        new_locations = TARGET_MOVE_VECTORS + np.array(self.target)
        np.random.shuffle(new_locations)
        new_locations = list(map(tuple, new_locations))
        for location in new_locations:
            if self.in_bounds(location):
                self.target = location

    def query_tile(self, coord):
        x = coord[0]
        y = coord[1]
        target_found = False

        if self.target == coord:
            if random.uniform(0, 1) > self.prob_map[x][y]:
                target_found = True

        if not target_found:
            self.move_target()

        return target_found

    def in_bounds(self, coord):
        x = coord[0]
        y = coord[1]
        return (0 <= x < self.dim
                and 0 <= y < self.dim)


if __name__ == '__main__':
    ls = Landscape(50)
    ls.generate_map()
    ls.move_target()
    ls.query_tile((1, 1))
