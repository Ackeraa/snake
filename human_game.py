import random
import pygame as pg
from settings import *

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
        self.step = 0
        self.snake = []
        self.food = None
        self.direction = None
        self.available_places = {}
        self.game_over = False
        self.win = False
        self.playing = True

    def play(self):
        while self.playing:
            self._new()
            while not self.game_over:
                self._event()
                self._move()
                self._draw()

    def _new(self):
        self.game_over = False
        self.win = False
        self.snake = []
        self.steps = 0
        self.score = 0
        self.available_places = {}
        for i in range(self.X):
            for j in range(self.Y):
                self.available_places[(i, j)] = 1

        # create new snake
        x = random.randint(2, self.X - 3)
        y = random.randint(2, self.Y - 3)
        self.head = (x, y)
        self.direction = DIRECTIONS[random.randint(0, 3)]
        body1 = (self.head[0] - self.direction[0], self.head[1] - self.direction[1])
        body2 = (body1[0] - self.direction[0], body1[1] - self.direction[1])
        self.snake.append(self.head)
        self.snake.append(body1)
        self.snake.append(body2)
        self.available_places.pop(self.head)
        self.available_places.pop(body1)
        self.available_places.pop(body2)

        self._place_food()

    def _place_food(self):
        if len(self.available_places) == 0:
            self.game_over = True
            self.win = True
            return
        self.food = random.choice(list(self.available_places.keys()))
        self.available_places.pop(self.food)

    def _move(self):
        self.steps += 1
        self.head = (self.head[0] + self.direction[0], self.head[1] + self.direction[1])
        self.snake.insert(0, self.head)
        
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            tail = self.snake.pop()
            self.available_places[tail] = 1
            if not self.head in self.available_places:
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
        
        # draw head
        x, y = self._get_xy(self.snake[0])
        pg.draw.rect(self.screen, WHITE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        pg.draw.rect(self.screen, WHITE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # draw body
        for s in self.snake[1:]:
            x, y = self._get_xy(s)
            pg.draw.rect(self.screen, BLUE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, BLUE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))
        
        # draw food
        x, y = self._get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        
        # draw text
        text = "score: " + str(self.score)
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

    def _event(self):
        self.clock.tick(FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.game_over  = True
                self.playing = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP and self.direction != (1, 0):
                    self.direction = (-1, 0)
                elif event.key == pg.K_DOWN and self.direction != (-1, 0):
                    self.direction = (1, 0)
                elif event.key == pg.K_LEFT and self.direction != (0, 1):
                    self.direction = (0, -1)
                elif event.key == pg.K_RIGHT and self.direction != (0, -1):
                    self.direction = (0, 1)    

if __name__ == '__main__':

    g = Game() 
    g.play()
    pg.quit()
