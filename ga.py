import random
import numpy as np
from nn import Net
from ai_game import Game
from ai_game_noui import Game as Game_Noui 
from settings import *
import copy

record = 0
best_fitness = 0

class Individual:
    def __init__(self, genes, net_struct):
        self.nn = Net(net_struct[0], net_struct[1], net_struct[2], net_struct[3])
        self.genes = genes
        self.score = 0
        self.steps = 0
    
    def get_fitness(self):
        self.nn.update(copy.deepcopy(self.genes))
        game = Game_Noui()
        game.play(self.nn)
        steps = game.steps
        score = game.score
        self.score = score
        self.steps = steps
        # print("steps", steps, "score", score)
        global record
        global best_fitness

        if score > record:
            record = score
        self.fitness = steps + (2 ** score + 500 * (score ** 2.1)) - (((0.25 * steps) ** 1.3) * (score ** 1.2))
        self.fitness = max(self.fitness, 0.1)
        if self.fitness > best_fitness:
            best_fitness = self.fitness
 
class GA:
    def __init__(self, p_size=P_SIZE, c_size=C_SIZE, mutate_rate=MUTATE_RATE, 
                 cross_rate=CROSS_RATE, eta=ETA, scale=SCALE, net_struct=NET_STRUCT):
        self.p_size = p_size
        self.c_size = c_size
        self.mutate_rate = mutate_rate
        self.cross_rate = cross_rate
        self.eta = eta
        self.scale = scale
        self.population = []
        self.age = 0
        self.net_struct = net_struct
        self.best_individual = None

    def get_gene(self):
        a, b, c, d = self.net_struct
        lengths = [a * b, b * c, c * d, b, c, d]
        genes = []
        for l in lengths:
            genes.append(np.random.uniform(-1, 1, l))
        
        return genes

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = self.get_gene()
            self.population.append(Individual(genes, self.net_struct))
    
    def crossover(self, c1_genes, c2_genes):
        for i in range(2 * len(self.net_struct) - 2):
            rand = np.random.random()
            if rand < self.cross_rate:
                c1_genes[i], c2_genes[i] = self.simulated_binary_crossover(c1_genes[i], c2_genes[i])
            else:
                c1_genes[i], c2_genes[i] = self.single_point_binary_crossover(c1_genes[i], c2_genes[i])
        return c1_genes, c2_genes

    def single_point_binary_crossover(self, c1_genes, c2_genes):
        p1_genes = copy.deepcopy(c1_genes)
        p2_genes = copy.deepcopy(c2_genes)
        genes_len = len(p1_genes)

        point = np.random.randint(0, genes_len)
        c1_genes[:point + 1] = p2_genes[:point + 1]
        c2_genes[:point + 1] = p1_genes[:point + 1]

        return c1_genes, c2_genes

    def simulated_binary_crossover(self, c1_genes, c2_genes):
        p1_genes = copy.deepcopy(c1_genes)
        p2_genes = copy.deepcopy(c2_genes)
        genes_len = len(p1_genes)

        rand = np.random.random(genes_len)
        gamma = np.empty(genes_len)

        gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (self.eta + 1))
        gamma[rand > 0.5] = (1.0 / (2.0 * (1 - rand[rand > 0.5]))) ** (1.0 / (self.eta + 1))

        c1_genes = 0.5 * ((1 + gamma) * p1_genes + (1 - gamma) * p2_genes)
        c2_genes = 0.5 * ((1 - gamma) * p1_genes + (1 + gamma) * p2_genes)

        return c1_genes, c2_genes

    def mutate(self, c1_genes, c2_genes):
        for i in range(2 * len(self.net_struct) - 2):
            c1_genes[i] = self.gaussian_mutate(c1_genes[i])
            c2_genes[i] = self.gaussian_mutate(c2_genes[i])
        
        return c1_genes, c2_genes

    def gaussian_mutate(self, c_genes):  
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_array] *= self.scale
        c_genes[mutation_array] += mutation[mutation_array]
           
        return c_genes

    def clip(self, c1_genes, c2_genes):
        for i in range(2 * len(self.net_struct) - 2):
            np.clip(c1_genes[i], -1, 1, out=c1_genes[i])
            np.clip(c2_genes[i], -1, 1, out=c2_genes[i])
    
        return c1_genes, c2_genes

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
        self.age += 1
        self.best_individual = self.population[0]
        for individual in self.population:
            individual.get_fitness()
            if individual.fitness > self.best_individual.fitness:
                self.best_individual = individual
            if individual.score == 97:
                self.save(individual, "final_genes.txt")
                return
        self.population = self.elitism_selection(self.p_size)

        random.shuffle(self.population)

        children = []
        while len(children) < self.c_size:
            p1, p2 = self.roulette_wheel_selection(2)
            c1_genes, c2_genes = copy.deepcopy(p1.genes), copy.deepcopy(p2.genes)
            c1_genes, c2_genes = self.crossover(c1_genes, c2_genes)
            c1_genes, c2_genes = self.mutate(c1_genes, c2_genes)
            c1_genes, c2_genes = self.clip(c1_genes, c2_genes)
            c1 = Individual(c1_genes, self.net_struct)
            c2 = Individual(c2_genes, self.net_struct)
            children.extend([c1, c2])

        random.shuffle(children)
        self.population.extend(children)

    def save(self, individual=None, fname="best_genes.txt"):
        if individual is None:
            individual = self.best_individual
        genes = self.best_individual.genes
        with open(fname, "w") as f:
            for gene in genes:
                for g in gene:
                    f.write(str(g) + " ")
                f.write("\n")

if __name__ == '__main__':
    random.seed()
    ga = GA()
    ga.generate_ancestor()
    game = Game()
    loop = 0
    while True:
        ga.evolve()
        # nn = ga.best_individual.nn
        # game.play(nn)
        loop += 1
        if loop % 20 == 0:
            ga.save()

        print(loop, record)
