from Landscape import Landscape
import numpy as np


class LsFinder:
    default_dim = 3
    moves = np.array([0,1],[0,-1],[1,0],[-1,0])

    def __init__(self):
        self.dim = self.ls.dim
        self.new_landscape()
        default_chance = 1/(self.ls.dim*self.ls.dim)
        self.likelihood = np.ones(
                shape=(LsFinder.default_dim, LsFinder.default_dim))
        self.likelihood *= default_chance

    def new_landscape(self):
        self.ls = Landscape(self.dim)
        self.cur_location = np.array[int(self.dim/2),int(self.dim/2)]



    def find(self,x,y):
        if(self.ls.query_tile(x,y)):
            return True
        else:
            self.bayesian_update(x,y)
            print(self.likelihood)
            

    def bayesian_update(self, miss_x, miss_y):

        false_neg_chance = self.ls.prob_map[miss_x,miss_y]
        original_belief = self.likelihood[miss_x][miss_y]
        new_belief = original_belief * false_neg_chance

        remaining_belief = 1- new_belief
        scaling_factor = remaining_belief/(1-original_belief)

        self.likelihood *= scaling_factor
        self.likelihood[miss_x][miss_y] = new_belief


        pass

    def search_rule1(self):
        search_index = np.argmax(self.likelihood)
        x = int(search_index/self.dim)
        y = search_index%self.dim
        return self.find(x,y)

    def simple_wander_search_r1(self):
        # moves = np.arary([
        #     self.current_location,
        #     self.current_location,
        #     self.current_location,
        #     self.current_location,
        # ])
        # moves =
        # i = argmax()
        pass

    def in_bounds(self,x,y):
        return (x >=0 and x < self.dim and y >= 0 and y < self.dim)
    def search_rule2(self):
        #prob of being found = prob of being in * (1 - false negative)
        search_index = np.argmax(np.multiply(self.likelihood, 1 - self.ls.prob_map))
        x = int(search_index/self.dim)
        y = search_index%self.dim
        
        return self.find(x,y)

if __name__ == '__main__':
    finder = LsFinder()
    finder.find(1,1)
    print("Done")
