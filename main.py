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
        self.running = True
        self.font_name = pg.font.match_font(FONT_NAME)
        self.score = 0
        self.generation = 1

        self.rows = (HEIGHT - BLANK_SIZE) // GRID_SIZE
        self.columns = WIDTH // GRID_SIZE

        self.border_left = 0
        self.border_right = 0
        self.border_up = BLANK_SIZE
        self.border_down = self.rows * GRID_SIZE + BLANK_SIZE


    def new(self):
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.snake = Snake(self)
        #self.food = Food(self)
        self.score = 0
        self.generation += 1
        self.run()

    def run(self):
        self.clock.tick(FPS)
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()

    def update(self):
        self.all_sprites.update()
    
    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False

    def draw(self):
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        self.draw_text("score: " + str(self.score) + "     generation: " + str(self.generation), 35, WHITE, WIDTH / 2, 5)
        self.draw_grid()
        pg.display.flip()

    def draw_text(self, text, size, color, x, y):
        font = pg.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(text_surface, text_rect)
    
    def draw_grid(self):
        n = (HEIGHT - BLANK_SIZE) // GRID_SIZE + 1
        m = WIDTH // GRID_SIZE
        for i in range(0, n):
            pg.draw.line(self.screen, LINE_COLOR, (0, i * GRID_SIZE + BLANK_SIZE), (WIDTH, i * GRID_SIZE + BLANK_SIZE), 1)

        for i in range(0, m):
            pg.draw.line(self.screen, LINE_COLOR, (i * GRID_SIZE, BLANK_SIZE), (i * GRID_SIZE, (n - 1) * GRID_SIZE + BLANK_SIZE), 1)




g = Game()

while g.running:
    g.new()

pg.quit()