import pygame as pg
from settings import *
from random import choice, randrange
from os import path

def get_xy(pos):
    x = pos[1] * GRID_SIZE + GRID_SIZE // 2 
    y = pos[0] * GRID_SIZE + GRID_SIZE // 2 + BLANK_SIZE
    return (x, y)

class Snake(pg.sprite.Sprite):
    def __init__(self, game, pos, direction):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.image = pg.Surface((GRID_SIZE, GRID_SIZE))
        self.image.fill((255,255,255))
        self.rect = self.image.get_rect()
        self.pos = pos
        self.rect.center = get_xy(self.pos)

        self.direction = direction
        self.last_move = 0

    def update(self):
        self.pos = (self.pos[0] + self.direction[0],
                    self.pos[1] + self.direction[1])
        self.rect.center = get_xy(self.pos)

class Food(pg.sprite.Sprite):
    def __init__(self, game, pos):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.image = pg.Surface((GRID_SIZE, GRID_SIZE))
        self.image.fill((255,255,255))
        self.rect = self.image.get_rect()
        self.pos = pos
        self.rect.center = get_xy(self.pos)

    def update(self):
        pass
