import random
from settings import *
import numpy as np
from nn import Net
import torch
import os

class Snake:

    def __init__(self, head, direction, genes, board_x, board_y):
        self.body = [head]
        self.direction = direction
        self.score = 0
        self.steps = 0
        self.dead = False
        self.uniq = [0] * 100
        self.board_x = board_x
        self.board_y = board_y
        self.nn = Net(N_INPUT, N_HIDDEN1, N_HIDDEN2, N_OUTPUT, genes.copy())

    def move(self, food):
        self.steps += 1
        state = self.get_state(food)
        action = self.nn.predict(state) 

        self.direction = DIRECTIONS[action]
        head = (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1])

        has_eat = False
        if (head[0] < 0 or head[0] >= self.board_x or head[1] < 0 or head[1] >= self.board_y
                or head in self.body):
            self.dead = True
        else:
            self.body.insert(0, head)
            if head == food:                  
                self.score += 1
                has_eat = True
            else:                           
                self.body.pop()
                if (head, food) not in self.uniq:
                    self.uniq.append((head, food))
                    del self.uniq[0]
                else:                       
                    self.dead = True

        return has_eat

    def get_state(self, food):
        i = DIRECTIONS.index(self.direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        if len(self.body) == 1:
            tail_direction = self.direction
        else:
            tail_direction = (self.body[-2][0] - self.body[-1][0], self.body[-2][1] - self.body[-1][1])
        i = DIRECTIONS.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0

        state = head_dir + tail_dir
        
        dirs = [[0, -1], [1, -1], [1, 0], [1, 1], 
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        
        for dir in dirs:
            x = self.body[0][0] + dir[0]
            y = self.body[0][1] + dir[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            while x >= 0 and x < self.board_x and y >= 0 and y < self.board_y:
                if (x, y) == food:
                    see_food = 1.0  
                elif (x, y) in self.body:
                    see_self = 1.0 
                dis += 1
                x += dir[0]
                y += dir[1]
            state += [1.0/dis, see_food, see_self]
        
        return state

class Game:

    def __init__(self, genes_list, seed, show=False, rows=ROWS, cols=COLS):
        self.Y = rows
        self.X = cols
        self.show = show
        self.seed = seed if seed is not None else random.randint(-INF, INF)
        self.rand = random.Random(self.seed)

        self.snakes = []
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        for genes in genes_list:
            head = self.rand.choice(board)
            direction = DIRECTIONS[self.rand.randint(0, 3)]
            self.snakes.append(Snake(head, direction, genes, self.X, self.Y))
        
        self.food = self.rand.choice(board)
        self.best_score = 0

    def play(self, genes_list, seed=None):
        self.new(genes_list, seed)
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        alive_snakes_set = set(self.snakes)
        while alive_snakes_set:
            for snake in alive_snakes_set:
                has_eat = snake.move(self.food)
                if has_eat:
                    self.food = self.rand.choice(board)
                if snake.score > self.best_score:
                    self.best_score = snake.score
            alive_snakes = [snake for snake in alive_snakes_set if not snake.dead]
            alive_snakes_set = set(alive_snakes)


        score = [snake.score for snake in self.snakes]
        steps = [snake.steps for snake in self.snakes]

        return score, steps, self.seed

if __name__ == '__main__':
    game = Game(show=True)
    game.play_saved_model(50)
