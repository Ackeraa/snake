import random
import numpy as np
from nn import Net
from ai_game import Game
from ai_game_noui import Game as Game_Noui 
from settings import *

record = 0
class Individual:
    def __init__(self, genes):
        self.nn = Net(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT)
        self.genes = genes
        self.get_fitness()
    
    def get_fitness(self):
        self.nn.update(self.genes)
        game = Game_Noui()
        game.play(self.nn)
        steps = game.steps
        score = game.score
        global record
        if score > record:
            record = score
        self.fitness = steps + (2 ** score + 500 * (score ** 2.1)) - (((0.25 * steps) ** 1.3) * (score ** 1.2))
        self.fitness = max(self.fitness, 0.1)
 
class GA:
    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, genes_len=GENES_LEN, mutate_rate=MUTATE_RATE, cross_rate=CROSS_RATE, eta=ETA):
        self.p_size = p_size
        self.c_size = c_size
        self.genes_len = genes_len
        self.genes_m_len = N_BIAS_START
        self.mutate_rate = mutate_rate
        self.cross_rate = cross_rate
        self.eta = eta
        self.population = []
        self.age = 0

    def get_gene(self):
        return random.uniform(-1.0, 1.0)

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = np.random.uniform(-1, 1, self.genes_len)
            self.population.append(Individual(genes))
    
    def mutate(self, c1_genes, c2_genes):
        c1_genes = self.gaussian_mutate(c1_genes)
        c2_genes = self.gaussian_mutate(c2_genes)

        return c1_genes, c2_genes

    def gaussian_mutate(self, c_genes):
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        mutation = np.random.normal(size=c_genes.shape)
        c_genes[mutation_array] += mutation[mutation_array]

        return c_genes

    def crossover(self, p1_genes, p2_genes):
        rand = random.random()
        if rand < self.cross_rate:
            c1_genes, c2_genes = self.simulated_binary_crossover(p1_genes, p2_genes)
        else:
            c1_genes, c2_genes = self.single_point_binary_crossover(p1_genes, p2_genes)

        np.clip(c1_genes, -1, 1, out=c1_genes)
        np.clip(c2_genes, -1, 1, out=c2_genes)

        return c1_genes, c2_genes

    def single_point_binary_crossover(self, p1_genes, p2_genes):
        c1_genes = p1_genes.copy()
        c2_genes = p2_genes.copy()
        
        point = random.randint(0, self.genes_m_len)
        c1_genes[point:self.genes_m_len] = p2_genes[point:self.genes_m_len]
        c2_genes[point:self.genes_m_len] = p1_genes[point:self.genes_m_len]

        point = random.randint(self.genes_m_len, self.genes_len)
        c1_genes[point:self.genes_len] = p2_genes[point:self.genes_len]
        c2_genes[point:self.genes_len] = p1_genes[point:self.genes_len]

        return c1_genes, c2_genes
        
    def simulated_binary_crossover(self, p1_genes, p2_genes):
        rand = np.random.random(self.genes_len)
        gamma = np.empty(self.genes_len)

        gamma[rand <= 0.5] = (2 ** rand[rand <= 0.5]) ** (1.0 / (self.eta + 1))
        gamma[rand > 0.5] = (1.0 / (2.0 *(1 - rand[rand > 0.5]))) ** (1.0 / (self.eta + 1))

        c1_genes = (0.5 * ((1 + gamma) * p1_genes + (1 - gamma) * p2_genes))
        c2_genes = (0.5 * ((1 - gamma) * p1_genes + (1 + gamma) * p2_genes))

        return c1_genes, c2_genes

    def elitism_selection(self, size):
        population = sorted(self.population, key =lambda individual: individual.fitness, reverse=True)
        return population[:size]

    def evolve(self):
        self.age += 1
        self.population = self.elitism_selection(self.p_size)
        random.shuffle(self.population)

        children = []
        while len(children) < self.c_size:
            p1, p2 = self.roulette_wheel_selection(2)
            c1_genes, c2_genes = self.crossover(p1.genes, p2.genes)
            c1_genes, c2_genes = self.mutate(c1_genes, c2_genes)
            c1 = Individual(c1_genes)
            c2 = Individual(c2_genes)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children)

    def roulette_wheel_selection(self, size):
        selection = []
        wheel = sum(individual.fitness for individual in self.population)
        for _ in range(size):
            pick = random.uniform(0, wheel)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break

        return selection
    
    def get_best(self):
        best_fitness = 0
        for individual in self.population:
            if individual.fitness > best_fitness:
                best_fitness = individual.fitness
                best_individual = individual
        
        return best_individual
        
if __name__ == '__main__':
    random.seed()
    ga = GA()
    ga.generate_ancestor()
    game = Game()
    while True:
        best_individual = ga.get_best()
        nn = best_individual.nn
        game.play(nn)
        ga.evolve()
        print(record)