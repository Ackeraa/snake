import random
import numpy as np
from ai_game import Game
from settings import *

class GA:

    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, genes_len=GENES_LEN,
                     mutate_rate=MUTATE_RATE):
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []

        self.generate_ancestor()

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))


















############################################################################
