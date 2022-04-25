import random
from settings import *
import numpy as np
from nn import Net


class Snake:

    def __init__(self, head, direction, genes, board_x, board_y):
        self.body = [head]
        self.direction = direction
        self.score = 0
        self.steps = 0
        self.dead = False
        self.board_x = board_x
        self.board_y = board_y
        self.uniq = [0] * board_x * board_y
        self.nn = Net(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, genes.copy())





















###########################################################################
