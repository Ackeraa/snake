import random
import pygame as pg
from os import path
from settings import *
from sprites import *

class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.font_name = pg.font.match_font(FONT_NAME)
        self.rows = (HEIGHT - BLANK_SIZE * 2) // GRID_SIZE
        self.cols = WIDTH // GRID_SIZE

        self.running = True
        self.score = 0
        self.generation = 0
        self.snake = []
        self.last_move = 0
        self.if_moved = False
        self.pause = False
        self.food = None
        self.empty_cells = {}

    def _create_food(self):
        idx = random.randint(0, len(self.empty_cells) - 1)
        pos = list(self.empty_cells.keys())[idx]
        self.empty_cells.pop(pos)
        if self.food is not None:
            self.food.kill()
        self.food = Food(self, pos)
        ## need to be fixed.

    def new(self):
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.if_moved = True
        self.snake = []
        self.empty_cells = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.empty_cells[(i, j)] = 1

        # create new snake
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        direction = directions[random.randint(0, 3)]
        pos = (self.cols // 2, self.rows // 2)
        self.snake.append(Snake(self, pos, direction))
        self.snake[-1].image.fill(HEAD_COLOR)
        self.empty_cells.pop(pos)

        self._create_food()
        self.score = 0
        self.generation += 1
        self._run()

    def _run(self):
        self.clock.tick(FPS)
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self._events()
            if not self.pause:
                self._update()
            self._draw()

    def _update(self):
        now = pg.time.get_ticks()
        if now - self.last_move > MOVE_GAP:
            self.last_move = now

            # check if eat the food
            pos = (self.snake[-1].pos[0] + self.snake[-1].direction[0],
                   self.snake[-1].pos[1] + self.snake[-1].direction[1])
            if self.food.pos == pos:
                self.score += 1
                self.snake[-1].image.fill(BODY_COLOR)
                self.snake.append(Snake(self, pos, self.snake[-1].direction))
                self.snake[-1].image.fill(HEAD_COLOR)
                self._create_food()
            else:
                self.all_sprites.update()
                self.if_moved = 1
                lost_cell = self.snake[-1].pos
                if lost_cell in self.empty_cells:
                    self.empty_cells.pop(lost_cell)
                else:
                    #collides or out of range
                    self.playing = False
                    self.food.kill() 

                got_cell = (self.snake[0].pos[0] - self.snake[0].direction[0],
                            self.snake[0].pos[1] - self.snake[0].direction[1])
                self.empty_cells[got_cell] = 1
                for i in range(0, len(self.snake) - 1):
                    self.snake[i].direction = self.snake[i + 1].direction
    
    def _events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False
            elif event.type == pg.KEYDOWN and self.if_moved:
                self.if_moved = 0
                if event.key == pg.K_UP and self.snake[-1].direction != (1, 0):
                    self.snake[-1].direction = (-1, 0)
                elif event.key == pg.K_DOWN and self.snake[-1].direction != (-1, 0):
                    self.snake[-1].direction = (1, 0)
                elif event.key == pg.K_LEFT and self.snake[-1].direction != (0, 1):
                    self.snake[-1].direction = (0, -1)
                elif event.key == pg.K_RIGHT and self.snake[-1].direction != (0, -1):
                    self.snake[-1].direction = (0, 1)    
            if event.type == pg.KEYDOWN and event.key == pg.K_q:
                self.pause = not self.pause

    def _draw(self):
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        self._draw_text("score: " + str(self.score) + "     generation: " + str(self.generation), 30, WHITE, WIDTH / 2, 5)
        self._draw_grid()
        pg.display.flip()

    def _draw_text(self, text, size, color, x, y):
        font = pg.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(text_surface, text_rect)
    
    def _draw_grid(self):
        n = (HEIGHT - BLANK_SIZE) // GRID_SIZE + 1
        m = WIDTH // GRID_SIZE
        for i in range(0, n):
            pg.draw.line(self.screen, LINE_COLOR, (0, i * GRID_SIZE + BLANK_SIZE), (WIDTH, i * GRID_SIZE + BLANK_SIZE), 1)

        for i in range(0, m):
            pg.draw.line(self.screen, LINE_COLOR, (i * GRID_SIZE, BLANK_SIZE), (i * GRID_SIZE, (n - 1) * GRID_SIZE + BLANK_SIZE), 1)

if __name__ == '__main__':

    g = Game() 
    while g.running:
        g.new()

    pg.quit()
