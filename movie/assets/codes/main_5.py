import random
import numpy as np
from ai_game import Game
from settings import *

class GA:
    
    def crossover(self, c1_genes, c2_genes):
        p = np.random.randint(0, self.genes_len)
        c1_genes[:p+1], c2_genes[:p+1] = c2_genes[:p+1], c1_genens[:p+1]

    def mutate(self, c_genes):  
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_array] *= 0.2
        c_genes[mutation_array] += mutation[mutation_array]























############################################################################
