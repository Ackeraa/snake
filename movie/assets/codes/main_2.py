import random
import numpy as np
from ai_game import Game
from settings import *

class Individual:

    def __init__(self, genes):
        pass
    
    def get_fitness(self):
        pass

class GA:

    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, genes_len=GENES_LEN,
                     mutate_rate=MUTATE_RATE):
        pass

    def generate_ancestor(self):
        pass
    
    def crossover(self, c1_genes, c2_genes):
        pass

    def mutate(self, c_genes):  
        pass

    def elitism_selection(self, size):
        pass

    def roulette_wheel_selection(self, size):
        pass

    def evolve(self):
        pass



############################################################################
