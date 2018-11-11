import numpy as np
import random
import matplotlib


class Landscape:
    def __init__(self, dim):
        self.dim = dim
        self.generate_map()

    def generate_map(self):
        # TODO randomly assign the tiles to each
        '''
        FLAT: P = .1  | DIST = .2
        HILLY: P = .3 | DIST = .3
        FOREST: P = .7| DIST = .3
        CAVE: P = .9  | DIST = .2
        '''
        dim = self.dim

        def val_to_prob(value):
            if value < 0 or value >= 1:
                #print(f"ERROR: probability value ({value}) out of bounds")
                return None
            elif value < .2:
                return .1
            elif value < .5:
                return .3
            elif value < .8:
                return .7
            else:
                return .9

        self.land_scape_dist = np.random.uniform(0,1,size=(dim, dim)) 
        self.prob_map = np.zeros((dim, dim)) # probabilities of finding
        # TODO set the prob_map values using assign_tile()
        for x in range(0, dim):
            for y in range(0, dim):
                value = self.land_scape_dist[x][y]
                self.prob_map[x][y] = val_to_prob(value)

        self.target = (random.randint(0,self.dim-1),random.randint(0,self.dim-1))

    def move_target(self):
        x = self.target[0]
        y = self.target[1]
        move_list = []

        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if (i >= 0 and i < self.dim and
                   j >= 0 and i < self.dim):
                    move_list.append((i, j))
        # TODO set target to a new location at random from the list
        pass

    def query_tile(self,x,y):
        if (x == self.target[0]) and (y == self.target[1]):
            if random.uniform(0,1) <= self.prob_map[x][y]:
                return False
            else:
                return True
        else:
            return False


if __name__ == '__main__':
    ls = Landscape(50)
    ls.generate_map()
