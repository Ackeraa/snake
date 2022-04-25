import random
import numpy as np
from ai_game import Game
from settings import *

class GA:
    
    def evolve(self):
        for individual in self.population:
            individual.get_fitness()

        self.population = self.elitism_selection(self.p_size)
        random.shuffle(self.population)

        children = []
        while len(children) < self.c_size:
            p1, p2 = self.roulette_wheel_selection(2)
            c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()
            self.crossover(c1_genes, c2_genes)
            self.mutate(c1_genes)
            self.mutate(c2_genes)
            c1, c2 = Individual(c1_genes), Individual(c2_genes)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children)













############################################################################
