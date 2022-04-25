import random
import numpy as np
from ai_game import Game
from settings import *

class Individual:

    def __init__(self, genes):
        self.genes = genes
        self.score = 0
        self.steps = 0
        self.seed = None
    
    def get_fitness(self):
        game = Game([self.genes])
        self.score, self.steps, self.seed = game.play()
        self.fitness = self.score + 1 / self.steps






















############################################################################
