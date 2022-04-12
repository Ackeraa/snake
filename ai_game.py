import random
import pygame as pg
from settings import *
import numpy as np
import os
import torch

class Snake:

    def __init__(self, id, pos, direction, nn):
        self.id = id
        self.snake = [pos]
        self.direction = direction
        self.nn = nn
        self.score = 0
        self.steps = 0

    def move(self, board):
        """Take a direction to move.
        
        Args:
            action: The indics of the direction to move, between 0 and 3.
        """
        self.steps += 1
        state = self.get_state(board)
        action = self.nn.predict(state) 
        self.direction = DIRECTIONS[action]
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, head)
        
        if board[head[0]][head[1]] == 2:  # Eat the food.
            self.score += 1
            board[head[0]][head[1]] = self.id

            return True
        else:
            tail = self.snake.pop()
            board[tail[0]][tail[1]] = -1
            if (head[0] < 0 or head[0] >= len(board) or head[1] < 0 or head[1] >= len(board[0]) or
                board[head[0]][head[1]] != -1):  # Hit the wall or itself or other.
                self.game_over = True
            else:
                board[head[0]][head[1]] = self.id

            return False

    def get_state(self, board):
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

        state = [head_dir, tail_dir]
        
        # Vision.
        dirs = [[0, -1], [1, -1], [1, 0], [1, 1], 
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        
        for dir in dirs:
            x = self.snake[0][0] + dir[0]
            y = self.snake[0][1] + dir[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            see_other = 0.0
            dis_to_food = np.inf
            dis_to_self = np.inf
            dis_to_other = np.inf
            while x < self.X and x >= 0 and y < self.Y and y >= 0:
                if board[x][y] == FOOD:
                    see_food = 1.0  
                    dis_to_food = dis
                elif board[x][y] = self.id:
                    see_self = 1.0 
                    dis_to_self = dis
                elif board[x][y] != -1:
                    see_other = 1.0
                    dis_to_other = dis
                dis += 1
                x += dir[0]
                y += dir[1]
            state += [1.0/dis, see_food, see_self, see_other]
        
        return state

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

        self.snakes = []
        self.food = None
        self.available_places = {}
        self.board = []
        self.game_over = False
        self.win = False

    def play(self, nn1, nn2, seed=None):
        """Use the Neural Network to play the game.

        Args:
            nn: Neural Network.
            seed: The random seed to generate food serials and initial position of snake.
                  It is used for reproduction.      
        """
        self.rand = random.Random(seed)
        self.new(nn1, nn2)
        while not self.game_over:
            self._event()
            first_to_move = random.randint(0, 1)

            snake1 = self.snakes[first_to_move]
            has_eat1 = snake1.move(self.board)

            snake2 = self.snakes[first_to_move^1]
            has_eat2 = snake2.move(self.board)

            if has_eat1 or has_eat2:
                self.place_sth(FOOD)  # Place food.

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

    def new(self, nn1, nn2):
        self.game_over = False
        self.win = False

        # empty: -1, snake1: 0, snake2: 1, food: 2
        self.board = [[-1 for _ in range(self.X)] for _ in range(self.Y)]

        # Create new snakes, both with 1 length.
        self.snakes = []
        head1 = self.place_sth(0)
        direction1 = DIRECTIONS[self.rand.randint(0, 3)]
        self.snakes.append(Snake(head1, direction1, nn1))
                                                                 
        head2 = self.place_sth(1)
        direction2 = DIRECTIONS[self.rand.randint(0, 3)]
        self.snakes.append(Snake(head2, direction2, nn2))
        
        place_sth(FOOD)  # Place food.
  
    def place_sth(self, sth):
        empty_cells = []
        for x in range(self.X):
            for y in range(self.Y):
                if self.board[x][y] == -1:
                    empty_cells.append((x, y))
        if empty_cells == []:
            self.game_over = True
            return

        cell = self.rand.choice(empty_cells)
        self.board[cell[0]][cell[1]] = sth

    def _get_xy(self, pos):
        """Transform pos to the coordinates of pygame."""
        x = pos[0] * GRID_SIZE
        y = pos[1] * GRID_SIZE + BLANK_SIZE
        return (x, y)

    def _draw(self):
        self.screen.fill(BLACK)
        
        # Draw head.
        x, y = self._get_xy(self.snakes[0][0])
        pg.draw.rect(self.screen, WHITE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        pg.draw.rect(self.screen, WHITE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # Draw body.
        for s in self.snake[1:]:
            x, y = self._get_xy(s)
            pg.draw.rect(self.screen, BLUE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, BLUE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))
        
        # Draw food.
        x, y = self._get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        
        # Draw text.
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
