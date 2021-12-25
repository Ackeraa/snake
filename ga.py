import random
from nn import Net
from ai_game import Game 
from settings import *

class Individual:
    def __init__(self, genes):
        self.nn = Net(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT)
        self.update(genes)
        self.age = 0
    
    def update(self, genes):
        self.genes = genes
        self.nn.update(genes)
        game = Game()
        while game.playing:
            state = game.get_state()
            action = self.nn.predict(state)
            game.move(action) 

        # Need to be fixed
        self.fitness = game.score

class GA:
    def __init__(self, p_size=P_SIZE, genes_len=GENES_LEN, mutate_rate=MUTATE_RATE, cross_rate=CROSS_RATE):
        self.p_size = p_size
        self.genes_len = genes_len
        self.mutate_rate = mutate_rate
        self.cross_rate = cross_rate
        self.population = []

    def get_gene(self):
        return random.uniform(-1.0, 1.0)

    def generate_ancestor(self):
        for i in range(self.p_size):
            genes = []
            for j in range(self.genes_len):
                genes.append(self.get_gene())
            self.population.append(Individual(genes))
    
    def mutate(self):
        for i in range(self.p_size):
            random_rate = random.random()
            random_pos = random.randint(0, self.genes_len - 1)
            if random_rate <= self.mutate_rate:
                genes = self.population[i].genes
                genes[random_pos] = self.get_gene()
                self.population[i].update(genes)

    def crossover(self):
        pass

    def simulated_binary_crossover(self, p1, p2, eta):
        c1, c2 = []
        for i in range(self.genes_len):
            rand = random.random()
            if rand <= 0.5:
                gamma = (2 * rand) ** (1.0 / (eta + 1))
            else:
                gamma = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1))
            c1.append(0.5 * ((1 + gamma) * p1[i] + (1 - gamma) * p2[i]))
            c2.append(0.5 * ((1 - gamma) * p1[i] + (1 + gamma) * p2[i]))
        
        return c1, c2

    def select(self, num):
        selection = []
        wheel = sum(individual.fitness for individual in self.population)
        for _ in range(num):
            pick = random.uniform(0, wheel)
            current = 0
            for individual in self.population:
                current += individual.fitness
                if current > pick:
                    selection.append(individual)
                    break

        return selection

if __name__ == '__main__':
    random.seed()
    ga = GA()
    ga.generate_ancestor()
    for _ in range(GA_LOOP):
        ga.select()
        ga.crossover()
        ga.mutate()
    
    best_individual = ga.get_best()
    nn = best_individual.nn
    


