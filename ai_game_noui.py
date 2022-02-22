from email.errors import FirstHeaderLineIsContinuationDefect
import random
from settings import *
import numpy as np
from nn import Net

class Game:
    def __init__(self, rows=ROWS, cols=COLS):
        self.Y = rows
        self.X = cols
        self.score = 0
        self.generation = 0
        self.snake = []
        self.food = None
        self.gap_steps = 0
        self.available_places = {}   
        self.game_over = False
        self.win = False
        self.apple_seed = None
        self.rand_apple = None
        self.uniq = None

    def play(self, nn):
        self._new()
        while not self.game_over:
            state = self._get_state()   
            action = nn.predict(state)
            self._move(action)

    def play_nn(self):
        genes = []
        with open("genes.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                gene = list(map(float, line.split()))
                genes.append(gene)

        nn = Net(NET_STRUCT[0], NET_STRUCT[1], NET_STRUCT[2], NET_STRUCT[3])
        nn.update(genes)
        self.play(nn)
        print(self.score)

    def _new(self):
        self.game_over = False
        self.win = False
        self.apple_seed = np.random.randint(-1000000000, 1000000000)
        self.rand_apple = random.Random(self.apple_seed)
        self.snake = []
        self.steps = 0
        self.gap_steps = 0
        self.available_places = {}
        self.uniq = [0] * (self.X * self.Y - 2)
        for x in range(self.X):
            for y in range(self.Y):
                self.available_places[(x, y)] = 1

        # create new snake
        x = random.randint(2, self.X - 3)
        y = random.randint(2, self.Y - 3)
        self.head = (x, y)
        direction = DIRECTIONS[random.randint(0, 3)]
        body1 = (self.head[0] - direction[0], self.head[1] - direction[1])
        body2 = (body1[0] - direction[0], body1[1] - direction[1])
        self.snake.append(self.head)
        self.snake.append(body1)
        self.snake.append(body2)
        self.available_places.pop(self.head)
        self.available_places.pop(body1)
        self.available_places.pop(body2)
        self._place_food()
        self.score = 0
        self.generation += 1

    def _place_food(self):
        if len(self.available_places) == 0:
            self.game_over = True
            self.win = True
            return
        possible_places = sorted(list(self.available_places.keys()))
        self.food = self.rand_apple.choice(possible_places)
        self.available_places.pop(self.food)

    def _move(self, action):
        self.steps += 1
        self.gap_steps += 1

        direction = DIRECTIONS[action]
        self.head = (self.head[0] + direction[0], self.head[1] + direction[1])
        self.snake.insert(0, self.head)
        
        if self.head == self.food:
            self.gap_steps = 0
            self.score += 1
            self._place_food()
        else:
            tail = self.snake.pop()
            self.available_places[tail] = 1
            if not self.head in self.available_places:
                self.game_over = True  
            else:
                self.available_places.pop(self.head)
            
            # infinate loop
            if (self.head, self.food) not in self.uniq:
                self.uniq.append((self.head,self.food))
                del self.uniq[0]
            else:
                self.game_over = True

    def _get_state(self):
        # head direction
        head_direction = (self.snake[0][0] - self.snake[1][0], self.snake[0][1] - self.snake[1][1])
        i = DIRECTIONS.index(head_direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        # tail direction
        tail_direction = (self.snake[-2][0] - self.snake[-1][0], self.snake[-2][1] - self.snake[-1][1])
        i = DIRECTIONS.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0

        state = []
        # vision
        dirs = [[0, -1], [1, -1], [1, 0], [1, 1], 
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        
        for dir in dirs:
            x = self.head[0] + dir[0]
            y = self.head[1] + dir[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            dis_to_food = np.inf
            dis_to_self = np.inf
            while x < self.X and x >= 0 and y < self.Y and y >= 0:
                if self.food == (x, y):
                    see_food = 1.0  
                    dis_to_food = dis
                elif not (x, y) in self.available_places:
                    see_self = 1.0 
                    dis_to_self = dis
                dis += 1
                x += dir[0]
                y += dir[1]
            state += [1.0/dis, see_food, see_self]
        state += head_dir + tail_dir

        return state

if __name__ == '__main__':
    game = Game()
    game.play_nn()

