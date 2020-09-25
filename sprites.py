import pygame as pg
from settings import *
from random import choice, randrange
from os import path

class Snake(pg.sprite.Sprite):
    def __init__(self, game):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.image = pg.Surface((GRID_SIZE, GRID_SIZE))
        self.image.fill((255,255,255))
        self.rect = self.image.get_rect()
        self.rect.center = (game.columns // 2 * GRID_SIZE - GRID_SIZE // 2, 
                            BLANK_SIZE + game.rows // 2 * GRID_SIZE - GRID_SIZE // 2)

        self.direction = [-1, 0]
        self.last_move = 0

    def update(self):
        now = pg.time.get_ticks()
        if now - self.last_move > MOVE_GAP:
            self.last_move = now
            center = list(self.rect.center)
            center[0] += self.direction[0] * GRID_SIZE
            center[1] += self.direction[1] * GRID_SIZE
            self.rect.center = center
            

class Food(pg.sprite.Sprite):
    def __init__(self, game):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.image = pg.Surface((GRID_SIZE, GRID_SIZE))
        self.image.fill((255,255,255))
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH / 3, HEIGHT / 2)

    def update(self):
        self.rect = self.image.get_rect()