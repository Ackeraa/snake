import random
import numpy as np
from ai_game import Game
from settings import *

class GA:
    
    def elitism_selection(self, size):
        population = sorted(self.population,
            key = lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def roulette_wheel_selection(self, size):
        selection = []
        wheel = sum(individual.fitness for individual in self.population)
        for _ in range(size):
            pick = np.random.uniform(0, wheel)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break
        
        return selection














############################################################################
