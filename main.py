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
        self.generation = 0

        self.rows = (HEIGHT - BLANK_SIZE) // GRID_SIZE
        self.columns = WIDTH // GRID_SIZE

        self.empty_cells = {}

        self.snake = []
        self.last_move = 0
        self.if_moved = 0

    def new(self):
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.last_pressed = 0
        self.if_moved = 1
        self.snake = []
        for i in range(self.rows):
            for j in range(self.columns):
                x = j * GRID_SIZE + GRID_SIZE // 2 
                y = i * GRID_SIZE + GRID_SIZE // 2 + BLANK_SIZE
                self.empty_cells[(x, y)] = 1

        x = self.columns // 2 * GRID_SIZE - GRID_SIZE // 2
        y = BLANK_SIZE + self.rows // 2 * GRID_SIZE - GRID_SIZE // 2
        directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        direction = directions[random.randint(0, 3)]
        # create new snake
        if direction == [1, 0]:
            self.snake.append(Snake(self, (x - 2 * GRID_SIZE, y), direction))
            self.snake.append(Snake(self, (x - GRID_SIZE, y), direction))
            self.snake.append(Snake(self, (x, y), direction))
            self.snake[-1].image.fill(HEAD_COLOR)
            self.empty_cells.pop((x - 2 * GRID_SIZE, y))
            self.empty_cells.pop((x - GRID_SIZE, y))
            self.empty_cells.pop((x, y))
        elif direction == [-1, 0]:
            self.snake.append(Snake(self, (x + 2 * GRID_SIZE, y), direction))
            self.snake.append(Snake(self, (x + GRID_SIZE, y), direction))
            self.snake.append(Snake(self, (x, y), direction))
            self.snake[-1].image.fill(HEAD_COLOR)
            self.empty_cells.pop((x + 2 * GRID_SIZE, y))
            self.empty_cells.pop((x + GRID_SIZE, y))
            self.empty_cells.pop((x, y))
        elif direction == [0, 1]:
            self.snake.append(Snake(self, (x, y - 2 * GRID_SIZE), direction))
            self.snake.append(Snake(self, (x, y - GRID_SIZE), direction))
            self.snake.append(Snake(self, (x, y), direction))
            self.snake[-1].image.fill(HEAD_COLOR)
            self.empty_cells.pop((x, y - 2 * GRID_SIZE))
            self.empty_cells.pop((x, y - GRID_SIZE))
            self.empty_cells.pop((x, y))
        else:
            self.snake.append(Snake(self, (x, y + 2 * GRID_SIZE), direction))
            self.snake.append(Snake(self, (x, y + GRID_SIZE), direction))
            self.snake.append(Snake(self, (x, y), direction))
            self.snake[-1].image.fill(HEAD_COLOR)
            self.empty_cells.pop((x, y + 2 * GRID_SIZE))
            self.empty_cells.pop((x, y + GRID_SIZE))
            self.empty_cells.pop((x, y))

        x = random.randint(0, len(self.empty_cells))
        food_pos = list(self.empty_cells.keys())[x]
        self.empty_cells.pop(food_pos)
        self.food = Food(self, food_pos)
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
        now = pg.time.get_ticks()
        if now - self.last_move > MOVE_GAP:
            self.last_move = now
            self.all_sprites.update()
            self.if_moved = 1
            # maintain the empty_cells
            lost_cell = (self.snake[-1].rect.center[0], self.snake[-1].rect.center[1])
            if lost_cell in self.empty_cells:
                self.empty_cells.pop(lost_cell)

            got_cel = (self.snake[0].rect.center[0] - self.snake[0].direction[0] * GRID_SIZE, 
                        self.snake[0].rect.center[1] - self.snake[0].direction[1] * GRID_SIZE)
            self.empty_cells[got_cel] = 1

            for i in range(0, len(self.snake) - 1):
                self.snake[i].direction = self.snake[i + 1].direction
        # check if eat the food
        if self.food.pos == self.snake[-1].rect.center:
            x = self.snake[-1].rect.center[0] + self.snake[-1].direction[0] * GRID_SIZE
            y = self.snake[-1].rect.center[1] + self.snake[-1].direction[1] * GRID_SIZE
            self.snake[-1].image.fill(BODY_COLOR)
            self.snake.append(Snake(self, (x, y), self.snake[-1].direction))
            self.snake[-1].image.fill(HEAD_COLOR)
            x = random.randint(0, len(self.empty_cells))
            food_pos = list(self.empty_cells.keys())[x]
            self.food.kill()
            self.food = Food(self, food_pos)
        
        # check if collides
        x = self.snake[-1].rect.center[0]
        y = self.snake[-1].rect.center[1]
        if (x, y) in self.empty_cells == False or x < 0 or x > WIDTH or y < BLANK_SIZE or y > BLANK_SIZE + self.rows * GRID_SIZE:
            self.playing = False
            self.food.kill()
    
    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False
            elif event.type == pg.KEYDOWN and self.if_moved:
                self.if_moved = 0
                if event.key == pg.K_UP and self.snake[-1].direction != [0, 1]:
                    self.snake[-1].direction = [0, -1]
                elif event.key == pg.K_DOWN and self.snake[-1].direction != [0, -1]:
                    self.snake[-1].direction = [0, 1]
                elif event.key == pg.K_LEFT and self.snake[-1].direction != [1, 0]:
                    self.snake[-1].direction = [-1, 0]
                elif event.key == pg.K_RIGHT and self.snake[-1].direction != [-1, 0]:
                    self.snake[-1].direction = [1, 0]    
                

    def draw(self):
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        self.draw_text("score: " + str(self.score) + "     generation: " + str(self.generation), 35, WHITE, WIDTH / 2, 5)
        #self.draw_empty()
        self.draw_grid()
        pg.display.flip()

    def draw_empty(self):
        for i in range(self.columns):
            for j in range(self.rows):
                x = i * GRID_SIZE
                y = BLANK_SIZE + j * GRID_SIZE
                if (x + GRID_SIZE // 2, y + GRID_SIZE // 2 ) in self.empty_cells:
                    pg.draw.rect(self.screen, LIGHTBLUE, (x, y, GRID_SIZE, GRID_SIZE))
                else:
                    pg.draw.rect(self.screen, WHITE, (x, y, GRID_SIZE, GRID_SIZE))

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




if __name__ == '__main__':

    g = Game() 
    while g.running:
        g.new()

    pg.quit()