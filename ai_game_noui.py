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

    def update(self):
        self.pos = (self.pos[0] + self.direction[0],
                    self.pos[1] + self.direction[1])

class Food():
    def __init__(self, pos):
        self.pos = pos

class Game:
    def __init__(self, rows=ROWS, cols=COLS):
        self.rows = rows
        self.cols = cols
        self.score = 0
        self.generation = 0
        self.snake = []
        self.food = None
        self.empty_cells = {}
        self.gap_steps = 0
        

    def _create_food(self):
        idx = random.randint(0, len(self.empty_cells) - 1)
        pos = list(self.empty_cells.keys())[idx]
        self.empty_cells.pop(pos)
        self.food = Food(pos)

    def new(self):
        self.playing = True
        self.steps = 0
        self.gap_steps = 0
        self.snake = []
        self.empty_cells = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.empty_cells[(i, j)] = 1

        # create new snake
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        direction = directions[random.randint(0, 3)]
        pos = (self.rows // 2, self.cols // 2)
        pos1 = (pos[0] - direction[0], pos[1] - direction[1])
        pos2 = (pos1[0] - direction[0], pos1[1] - direction[1])
        self.snake.append(Snake(pos2, direction))
        self.snake.append(Snake(pos1, direction))
        self.snake.append(Snake(pos, direction))
        self.empty_cells.pop(pos)
        self.empty_cells.pop(pos1)
        self.empty_cells.pop(pos2)

        self._create_food()
        self.score = 0
        self.generation += 1

    def move(self, action):
        self.steps += 1
        self.gap_steps += 1

        '''
        # 0: straight, 1: left, 2: right
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        i = dirs.index(self.snake[-1].direction)
        if action == 0:
            # keep original direction
            pass
        elif action == 1:
            self.snake[-1].direction = dirs[(i - 1) % 4]
        else:
            self.snake[-1].direction = dirs[(i + 1) % 4]
        '''

        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.snake[-1].direction = dirs[action]
        self._update()

    def get_state(self):
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # head direction
        i = dirs.index(self.snake[-1].direction)
        head_dir = [0, 0, 0, 0]
        head_dir[i] = 1

        # tail direction
        i = dirs.index(self.snake[0].direction)
        tail_dir = [0, 0, 0, 0]
        tail_dir[i] = 1

        state = head_dir + tail_dir
        # vision
        dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], 
                [-1, 0], [-1, -1], [0, -1], [1, -1]]
        
        for dir in dirs:
            r = self.snake[-1].pos[0] + dir[0]
            c = self.snake[-1].pos[1] + dir[1]
            dis = 1
            see_food = 0
            see_self = 0
            while r < self.rows and r >= 0 and c < self.cols and c >= 0:
                if self.food.pos == (r, c):
                    see_food = 1  
                elif not (r, c) in self.empty_cells:
                    see_self = 1 
                dis += 1
                r += dir[0]
                c += dir[1]
            state += [1.0/dis, see_food, see_self]

        return state

    def play(self, nn):
        self.new()
        while self.playing:
            state = self.get_state()
            action = nn.predict(state)
            self.move(action) 

    def _update(self):
        # check if eat the food
        pos = (self.snake[-1].pos[0] + self.snake[-1].direction[0],
               self.snake[-1].pos[1] + self.snake[-1].direction[1])
        if self.food.pos == pos:
            self.gap_steps = 0
            self.score += 1
            self.snake.append(Snake(pos, self.snake[-1].direction))
            self._create_food()
        else:
            for snake in self.snake:
                snake.update()
            lost_cell = self.snake[-1].pos 
            if not lost_cell in self.empty_cells or self.gap_steps > GAME_LOOP:
                #collides or out of range
                self.playing = False
                self.food = None
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
