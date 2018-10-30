import numpy as np
import random
import matplotlib


class Landscape:
	def __init__(self, dim):
		self.dim = dim
		pass

	def generate_map(self):
		# TODO randomly assign the tiles to each
		'''
		FLAT: P = .1  | DIST = .2
		HILLY: P = .3 | DIST = .3
		FOREST: P = .7| DIST = .3
		CAVE: P = .9  | DIST = .2
		'''
		def assign_tile(value):
			if value < 0 or value >= 1:
				print(f"ERROR: probability value ({value}) out of bounds")
				return None
			elif value < .2:
				return .1
			elif value < .5:
				return .3
			elif value < .8:
				return .7
			else:
				return .9

		self.land_scape_dist = np.random.uniform(size = (self.dim,self.dim))
		self.prob_map = np.zeros(size = (50,50))
		# TODO set the prob_map values using assign_tile()

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