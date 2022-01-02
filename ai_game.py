import random
import pygame as pg
from os import path
from settings import *
from sprites import *
import numpy as np

class Game:
    def __init__(self, rows=ROWS, cols=COLS):
        pg.init()
        self.rows = rows
        self.cols = cols
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
        self.empty_cells = {}
        
    def _create_food(self):
        idx = random.randint(0, len(self.empty_cells) - 1)
        pos = list(self.empty_cells.keys())[idx]
        self.empty_cells.pop(pos)
        if self.food is not None:
            self.food.kill()
        self.food = Food(self, pos)

    def new(self):
        self.all_sprites = pg.sprite.LayeredUpdates()
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
        self.snake.append(Snake(self, pos2, direction))
        self.snake.append(Snake(self, pos1, direction))
        self.snake.append(Snake(self, pos, direction))
        self.snake[-1].image.fill(HEAD_COLOR)
        self.empty_cells.pop(pos)
        self.empty_cells.pop(pos1)
        self.empty_cells.pop(pos2)

        self._create_food()
        self.score = 0
        self.generation += 1

    def move(self, action):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

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
        self._draw()
        self.clock.tick(FPS)

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
            self.snake[-1].image.fill(BODY_COLOR)
            self.snake.append(Snake(self, pos, self.snake[-1].direction))
            self.snake[-1].image.fill(HEAD_COLOR)
            self._create_food()
        else:
            self.all_sprites.update()
            lost_cell = self.snake[-1].pos 
            if not lost_cell in self.empty_cells or self.gap_steps > GAME_LOOP:
                #collides or out of range
                self.playing = False
                self.food.kill() 
            else:
                self.empty_cells.pop(lost_cell)

            got_cell = (self.snake[0].pos[0] - self.snake[0].direction[0],
                        self.snake[0].pos[1] - self.snake[0].direction[1])
            self.empty_cells[got_cell] = 1
            for i in range(0, len(self.snake) - 1):
                self.snake[i].direction = self.snake[i + 1].direction

    def _draw(self):
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        self._draw_text("s: " + str(self.score) + "     g: " + str(self.generation), 30, WHITE, self.width / 2, 5)
        self._draw_grid()
        pg.display.flip()

    def _draw_text(self, text, size, color, x, y):
        font = pg.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(text_surface, text_rect)
    
    def _draw_grid(self):
        n = (self.height - BLANK_SIZE) // GRID_SIZE + 1
        m = self.width // GRID_SIZE
        for i in range(0, n):
            pg.draw.line(self.screen, LINE_COLOR, (0, i * GRID_SIZE + BLANK_SIZE), (self.width, i * GRID_SIZE + BLANK_SIZE), 1)

        for i in range(0, m):
            pg.draw.line(self.screen, LINE_COLOR, (i * GRID_SIZE, BLANK_SIZE), (i * GRID_SIZE, (n - 1) * GRID_SIZE + BLANK_SIZE), 1)

if __name__ == '__main__':

    g = Game() 
    pg.quit()
