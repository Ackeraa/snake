import random
import pygame as pg
from os import path
from settings import *
from sprites import *
import numpy as np

class Game:
    def __init__(self, rows=ROWS, cols=COLS):
        pg.init()
        self.Y = rows
        self.X = cols
        self.width = cols * GRID_SIZE
        self.height = rows * GRID_SIZE + BLANK_SIZE
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.font_name = pg.font.match_font(FONT_NAME)

        self.score = 0
        self.generation = 0
        self.step = 0
        self.gap_steps = 0
        self.snake = []
        self.food = None
        self.direction = None
        self.available_places = {}

    def _place_food(self):
        self.food = random.choice(list(self.available_places.keys()))
        self.available_places.pop(self.food)

    def _new(self):
        self.game_over = False
        self.snake = []
        self.steps = 0
        self.gap_steps = 0
        self.available_places = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.available_places[(i, j)] = 1

        # create new snake
        self.direction = DIRECTIONS[random.randint(0, 3)]
        self.head = (self.cols // 2, self.rows // 2)
        body = (self.head[0] - self.direction[0], self.head[1] - self.direction[1])
        self.snake.append(self.head)
        self.snake.append(body)
        self.available_places.pop(self.head)
        self.available_places.pop(body)

        self._place_food()
        self.score = 0
        self.generation += 1

    def play(self):
        self._new()
        while not self.game_over:
            state = self._get_state()
            #action = nn.predict(state)
            action = int(input())
            self._move(action)
            self._draw()

    def _get_state(self):
        # head direction
        i = DIRECTIONS.index(self.direction)
        head_dir = [0, 0, 0, 0]
        head_dir[i] = 1

        state = head_dir 
        # vision
        dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], 
                [-1, 0], [-1, -1], [0, -1], [1, -1]]
        
        for dir in dirs:
            r = self.head[0] + dir[0]
            c = self.head[1] + dir[1]
            dis = 0
            dis_to_food = 0
            dis_to_body = 1
            see_body = False
            while r < self.rows and r >= 0 and c < self.cols and c >= 0:
                if self.food == (r, c):
                    dis_to_food = 1 - dis / (self.rows - 1)
                elif (not (r, c) in self.available_places) and (not see_body):
                    see_body = True
                    dis_to_body = dis / (self.rows - 1)
                dis += 1.0
                r += dir[0]
                c += dir[1]
            dis_to_wall = dis / self.rows
            state += [dis_to_wall, dis_to_body, dis_to_food]

        #print(state)
        return state

    def _move(self, action):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        self.steps += 1
        self.gap_steps += 1

        self.direction = DIRECTIONS[action]
        self.head = (self.head[0] + self.direction[0], self.head[1] + self.direction[1])
        self.snake.insert(0, self.head)
        
        if self.head == self.food:
            self.gap_steps = 0
            self.score += 1
            self._place_food()
        else:
            tail = self.snake.pop()
            self.available_places[tail] = 1
            if (not self.head in self.available_places) or (self.gap_steps > GAME_LOOP):
                self.game_over = True  
            else:
                self.available_places.pop(self.head)
        
        self.clock.tick(FPS)

    def _get_xy(self, pos):
        x = pos[1] * GRID_SIZE
        y = pos[0] * GRID_SIZE + BLANK_SIZE
        return (x, y)

    def _draw(self):
        self.screen.fill(BLACK)
        
        # draw snake
        for s in self.snake:
            x, y = self._get_xy(s)
            pg.draw.rect(self.screen, BLUE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, BLUE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))
        
        # draw food
        x, y = self._get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        
        # draw text
        text = "score: " + str(self.score) + "     generation: " + str(self.generation)
        font = pg.font.Font(self.font_name, 20)
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.midtop = ((self.width / 2, 5))
        self.screen.blit(text_surface, text_rect)


        # draw grid
        n = (self.height - BLANK_SIZE) // GRID_SIZE + 1
        m = self.width // GRID_SIZE
        for i in range(0, n):
            pg.draw.line(self.screen, LINE_COLOR, (0, i * GRID_SIZE + BLANK_SIZE), 
                         (self.width, i * GRID_SIZE + BLANK_SIZE), 1)
        for i in range(0, m):
            pg.draw.line(self.screen, LINE_COLOR, (i * GRID_SIZE, BLANK_SIZE), 
                         (i * GRID_SIZE, (n - 1) * GRID_SIZE + BLANK_SIZE), 1)

        pg.display.flip()

if __name__ == '__main__':

    g = Game() 
    g.play()
    pg.quit()
