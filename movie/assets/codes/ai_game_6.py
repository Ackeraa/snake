import random
from settings import *
import numpy as np
from nn import Net


class Game:

    def __init__(self, genes_list, seed, rows=ROWS, cols=COLS):
        self.Y = rows
        self.X = cols
        self.seed = seed if seed is not None else random.randint(-INF, INF)
        self.rand = random.Random(self.seed)

        self.snakes = []
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        for genes in genes_list:
            head = self.rand.choice(board)
            direction = DIRECTIONS[self.rand.randint(0, 3)]
            self.snakes.append(Snake(head, direction, genes, self.X, self.Y))
        
        self.food = self.rand.choice(board)

















#########################################################################
