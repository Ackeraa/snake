
import random
import argparse
import numpy as np
from ai_game2 import Game
from settings import *
import os

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.score = 0
        self.steps = 0
    
    def get_fitness(self):
        game = Game([self.genes])
        self.score, self.steps, self.seed = game.play()
        self.fitness = self.score + 1 / self.steps

class GA:
    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, genes_len=GENES_LEN, mutate_rate=MUTATE_RATE):
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.population = []
        self.best_individual = None
        self.avg_score = 0

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))
    
    def crossover(self, c1_genes, c2_genes):
        p1_genes = c1_genes.copy()
        p2_genes = c2_genes.copy()

        point = np.random.randint(0, self.genes_len)
        c1_genes[:point + 1] = p2_genes[:point + 1]
        c2_genes[:point + 1] = p1_genes[:point + 1]

    def mutate(self, c_genes):  
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_array] *= 0.2
        c_genes[mutation_array] += mutation[mutation_array]

    def elitism_selection(self, size):
        population = sorted(self.population, key =lambda individual: individual.fitness, reverse=True)
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


if __name__ == '__main__':
    ga = GA()
    
    while True:
        generation += 1
        ga.evolve()
