import random
import pygame as pg
from settings import *
import numpy as np
import os
import torch

class Game:
    """This Class is for visualization of the AI snake movement.

       It gives the state of the game to Neural Network and gets the next move back.

    Attributes:
        X: Columns of the game board.
        Y: Rows of the game board.
        width: Width of the game board.
        height: Height of the game board.
        screen: Pygame screen.
        clock: Pygame clock.
        font_name: Name of the font.
        score: Food eat by the snake.
        steps: Steps moved of the snake.
        snake: postion of the snake.
        head: head of the snake.
        food: Position of the food.
        direction: The direction of the snake's head.
        available_places: # Places available for snake to move or place food. 
        game_over: A boolean if the game is over.
        win: A boolean if the game is winning.
    """

    def __init__(self, rows=ROWS, cols=COLS):
        pg.init()

        self.Y = rows
        self.X = cols
        self.width = cols * GRID_SIZE
        self.height = rows * GRID_SIZE + BLANK_SIZE

        pg.display.set_caption(TITLE)
        self.screen = pg.display.set_mode((self.width, self.height))
        self.clock = pg.time.Clock()
        self.font_name = pg.font.match_font(FONT_NAME)

        self.score = 0
        self.steps = 0
        self.snake = []
        self.head = None
        self.food = None
        self.direction = None
        self.available_places = {}
        self.game_over = False
        self.win = False

    def play(self, nn, seed=None):
        """Use the Neural Network to play the game.

        Args:
            nn: Neural Network.
            seed: The random seed to generate food serials and initial position of snake.
                  It is used for reproduction.      
        """
        self.rand = random.Random(seed)
        self.new()
        while not self.game_over:
            self._event()
            state = self.get_state()
            action = nn.predict(state)
            self.move(action)
            self._draw()

    def play_saved_model(self, score):
        """Use the saved Neural Network model play the game.

        Args:
            score: Specify which model to load, also indicates the highest score it can get.
        """
        model_pth = os.path.join("model", "best_individual", "nn_"+str(score)+'.pth')
        nn = torch.load(model_pth)

        seed_pth = os.path.join("seed", "seed_"+str(score)+'.txt')  # Get the seed for reproduction.
        with open(seed_pth, "r") as f:
            seed = int(f.read())
 
        self.play(nn, seed)

    def new(self):
        self.game_over = False
        self.win = False
        self.snake = []
        self.steps = 0
        self.score = 0
        self.available_places = {}
        for x in range(self.X):
            for y in range(self.Y):
                self.available_places[(x, y)] = 1

        # Create new snake.
        x = self.rand.randint(2, self.X - 3)
        y = self.rand.randint(2, self.Y - 3)
        self.head = (x, y)
        direction = DIRECTIONS[self.rand.randint(0, 3)]
        body1 = (self.head[0] - direction[0], self.head[1] - direction[1])
        body2 = (body1[0] - direction[0], body1[1] - direction[1])
        self.snake.append(self.head)
        self.snake.append(body1)
        self.snake.append(body2)

        # Update places available to move or place food.
        self.available_places.pop(self.head)
        self.available_places.pop(body1)
        self.available_places.pop(body2)
        
        self.place_food()
  
    def place_food(self):
        if len(self.available_places) == 0:
            self.game_over = True
            self.win = True
            return 
        possible_places = sorted(list(self.available_places.keys()))
        self.food = self.rand.choice(possible_places)
        self.available_places.pop(self.food)

    def move(self, action):      
        """Take a direction to move.
        
        Args:
            action: The indics of the direction to move, between 0 and 3.
        """
        self.steps += 1
        self.direction = DIRECTIONS[action]
        self.head = (self.head[0] + self.direction[0], self.head[1] + self.direction[1])
        self.snake.insert(0, self.head)
        
        if self.head == self.food:  # Eat the food.
            self.score += 1
            self.place_food()
        else:
            tail = self.snake.pop()
            self.available_places[tail] = 1
            if not self.head in self.available_places:  # Hit the wall or itself.
                self.game_over = True  
            else:
                self.available_places.pop(self.head)

    def get_state(self):
        # Head direction.
        head_direction = (self.snake[0][0] - self.snake[1][0], self.snake[0][1] - self.snake[1][1])
        i = DIRECTIONS.index(head_direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        # Tail direction.
        tail_direction = (self.snake[-2][0] - self.snake[-1][0], self.snake[-2][1] - self.snake[-1][1])
        i = DIRECTIONS.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0

        state = []
        
        # Vision.
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

    def _get_xy(self, pos):
        """Transform pos to the coordinates of pygame."""
        x = pos[0] * GRID_SIZE
        y = pos[1] * GRID_SIZE + BLANK_SIZE
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
        x = self.width // GRID_SIZE
        y = (self.height - BLANK_SIZE) // GRID_SIZE + 1
        for i in range(0, x):
            pg.draw.line(self.screen, LINE_COLOR, (i * GRID_SIZE, BLANK_SIZE), 
                         (i * GRID_SIZE, (y - 1) * GRID_SIZE + BLANK_SIZE), 1)
        for i in range(0, y):
            pg.draw.line(self.screen, LINE_COLOR, (0, i * GRID_SIZE + BLANK_SIZE), 
                         (self.width, i * GRID_SIZE + BLANK_SIZE), 1)

        pg.display.flip()

    def _event(self):
        """Get the event from user interface."""
        self.clock.tick(FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

if __name__ == '__main__':

    g = Game() 
    g.play_saved_model(97)
    pg.quit()
