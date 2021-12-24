import random
from random import choice, randrange
from settings import *
import numpy as np

def get_xy(pos):
    x = pos[1] * GRID_SIZE + GRID_SIZE // 2 
    y = pos[0] * GRID_SIZE + GRID_SIZE // 2 + BLANK_SIZE
    return (x, y)

class Snake():
    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction
        self.last_move = 0

    def update(self):
        self.pos = (self.pos[0] + self.direction[0],
                    self.pos[1] + self.direction[1])

class Food():
    def __init__(self, pos):
        self.pos = pos

class Game:
    def __init__(self):
        self.rows = (HEIGHT - BLANK_SIZE * 2) // GRID_SIZE
        self.cols = WIDTH // GRID_SIZE

        self.score = 0
        self.generation = 0
        self.snake = []
        self.food = None
        self.empty_cells = {}
        
        self.reward = 0
        self.new()

    def _create_food(self):
        idx = random.randint(0, len(self.empty_cells) - 1)
        pos = list(self.empty_cells.keys())[idx]
        self.empty_cells.pop(pos)
        self.food = Food(pos)

    def new(self):
        self.playing = True
        self.iter = 0
        self.snake = []
        self.empty_cells = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.empty_cells[(i, j)] = 1

        # create new snake
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        direction = directions[random.randint(0, 3)]
        pos = (self.rows // 2, self.cols // 2)
        self.snake.append(Snake(pos, direction))
        self.empty_cells.pop(pos)

        self._create_food()
        self.score = 0
        self.generation += 1

    def move(self, action):
        '''
        # [up, down, left. right]
        #print(action)
        if action == [1, 0, 0, 0] and self.snake[-1].direction != (1, 0):
            self.snake[-1].direction = (-1, 0)
        elif action == [0, 1, 0, 0] and self.snake[-1].direction != (-1, 0):
            self.snake[-1].direction = (1, 0)
        elif action == [0, 0, 1, 0] and self.snake[-1].direction != (0, 1):
            self.snake[-1].direction = (0, -1)
        elif action == [0, 0, 0, 1] and self.snake[-1].direction != (0, -1):
            self.snake[-1].direction = (0, 1)    
        '''

        self.iter += 1
        # [straight, left, right]
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        i = dirs.index(self.snake[-1].direction)
        if np.array_equal(action, [1, 0, 0]):
            # keep original direction
            pass
        elif np.array_equal(action, [0, 1, 0]):
            self.snake[-1].direction = dirs[(i - 1) % 4]
        else:
            self.snake[-1].direction = dirs[(i + 1) % 4]

        self._update()
        #print("reward: ", self.reward, "playing: ", self.playing, "score: ", self.score)
        return self.reward, self.playing, self.score

    def _update(self):
        #print(self.snake[-1].direction, self.snake[-1].pos, self.food.pos)
        self.reward = 0
        # check if eat the food
        pos = (self.snake[-1].pos[0] + self.snake[-1].direction[0],
               self.snake[-1].pos[1] + self.snake[-1].direction[1])
        if self.food.pos == pos:
            self.score += 1
            self.reward = 10
            self.snake.append(Snake(pos, self.snake[-1].direction))
            self._create_food()
        else:
            for snake in self.snake:
                snake.update()
            lost_cell = self.snake[-1].pos 
            if not lost_cell in self.empty_cells or self.iter > 300 * len(self.snake):
                #collides or out of range
                self.reward = -10
                self.playing = False
            else:
                self.empty_cells.pop(lost_cell)

            got_cell = (self.snake[0].pos[0] - self.snake[0].direction[0],
                        self.snake[0].pos[1] - self.snake[0].direction[1])
            self.empty_cells[got_cell] = 1
            for i in range(0, len(self.snake) - 1):
                self.snake[i].direction = self.snake[i + 1].direction

if __name__ == '__main__':
    game = Game()
    while True:
        a = list(map(int, input().split()))
        game.move(a)
