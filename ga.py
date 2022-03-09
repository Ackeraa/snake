import random
import numpy as np
from nn import Net
from ai_game import Game
from ai_game_noui import Game as Game_Noui 
from settings import *
import torch
import os

class Individual:
    def __init__(self, genes):
        self.nn = Net(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT)
        self.genes = genes
        self.score = 0
        self.steps = 0
    
    def get_fitness(self):
        self.nn.update(self.genes.copy())
        game = Game_Noui()
        game.play(self.nn)
        steps = game.steps
        score = game.score
        self.score = score
        self.steps = steps
        self.seed = game.seed

        self.fitness = (score+0.5*(steps-steps/(score+1))/(steps+steps/(score+1)))*100000
 
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
    
    # load genes(nn model parameters) from file.
    def inherit_ancestor(self):
        for i in range(self.p_size):
            pth = os.path.join("model", "all_individual", str(i)+"_nn.pth")
            nn = torch.load(pth)
            genes = []
            with torch.no_grad():
                for parameters in nn.parameters():
                    genes.extend(parameters.numpy().flatten())
            self.population.append(Individual(np.array(genes)))

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
        sum_score = 0
        for individual in self.population:
            individual.get_fitness()
            sum_score += individual.score
        self.avg_score = sum_score / len(self.population)
        self.population = self.elitism_selection(self.p_size)
        self.best_individual = self.population[0]
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

    def save_best(self, score):
        model_pth= os.path.join("model", "best_individual", "nn20_"+str(score)+".pth")
        torch.save(self.best_individual.nn, model_pth)
        seed_pth = os.path.join("seed", "seed_"+str(score)+".txt")
        with open(seed_pth, "w") as f:
            f.write(str(self.best_individual.seed)) 
    
    def save_all(self):
        for individual in self.population:
            individual.get_fitness()
        population = self.elitism_selection(self.p_size)
        for i in range(len(population)):
            pth = os.path.join("model", "all_individual", str(i)+"_nn20.pth")
            torch.save(population[i].nn, pth)

if __name__ == '__main__':
    ga = GA()
    #ga.generate_ancestor()
    ga.inherit_ancestor()
    #game = Game()
    generation = 0
    record = 0
    while True:
        generation += 1
        ga.evolve()
        print("generation:", generation, ",record:", record, ",best score:", ga.best_individual.score, ",average score:", ga.avg_score)
        if ga.best_individual.score >= record:
            record = ga.best_individual.score 
            ga.save_best(ga.best_individual.score)
            # game.play(ga.best_individual.nn, ga.best_individual.seed, loop)
        if generation % 20 == 0:
            ga.save_all()


