from Landscape import Landscape
import numpy as np


class LsFinder:
    default_dim = 5
    move_vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])

    def __init__(self):
        self.dim = self.default_dim
        self.new_landscape()
        default_chance = 1/(self.ls.dim*self.ls.dim)
        self.likelihood = np.ones(
                shape=(LsFinder.default_dim, LsFinder.default_dim))
        self.likelihood *= default_chance
        self.query_counter= np.zeros((self.dim,self.dim))

    def new_landscape(self):
        self.ls = Landscape(self.dim)
        self.cur_location = np.array([int(self.dim/2),int(self.dim/2)])



    def find(self,x,y):
        print(f"Searching in spot [{x},{y}]")
        self.query_counter[x][y]+=1
        if(self.ls.query_tile(x,y)):
            print(f"Target has been found in [{x},{y}]")
            return True
        else:
            self.bayesian_update(x,y)
            print(self.likelihood)
            return False
            

    def bayesian_update(self, miss_x, miss_y):

        false_neg_chance = self.ls.prob_map[miss_x,miss_y]
        original_belief = self.likelihood[miss_x][miss_y]
        new_belief = original_belief * false_neg_chance

        remaining_belief = 1- new_belief
        scaling_factor = remaining_belief/(1-original_belief)
        
        if(scaling_factor < 0):
            print("ERROR")

        self.likelihood *= scaling_factor
        self.likelihood[miss_x][miss_y] = new_belief
        
        # floating point error
        error = np.sum(self.likelihood) -1
        print(f"[Bayes]: floating point error = {error}")
        self.likelihood[miss_x][miss_y] = new_belief - error

    def search_rule1(self):
        search_index = np.argmax(self.likelihood)
        x = int(search_index/self.dim)
        y = search_index%self.dim
        return self.find(x,y)

    def simple_wander_search_r1(self):
        moves = self.move_vectors + self.cur_location
        scores = np.array(map(self.rule1, moves))
        next_move = moves[np.argmax(scores)]
        print(next_move)
        self.cur_location = next_move
        return self.find(next_move[0],next_move[1])

    def rule1(self,coords):
        if not self.in_bounds(coords[0],coords[1]):
            return 0
        return self.likelihood[coords[0],coords[1]]

    def in_bounds(self,x,y):
        return (x >= 0 and x < self.dim and y >= 0 and y < self.dim)
    
    def search_rule2(self):
        #prob of being found = prob of being in * (1 - false negative)
        search_index = np.argmax(np.multiply(self.likelihood, 1 - self.ls.prob_map))
        x = int(search_index/self.dim)
        y = search_index%self.dim
        
        return self.find(x,y)

if __name__ == '__main__':
    finder = LsFinder()
    x=0
    while((not finder.search_rule1()) and x < 1000 and
          np.sum(finder.likelihood) > .98):
        print(np.sum(finder.likelihood))
        print("===================================")
        x+=1
    print(f"Target was in [{finder.ls.target}]")
    print(f"Found target in {x} steps")
    print("# times searched each spot")
    print(finder.query_counter)
      
