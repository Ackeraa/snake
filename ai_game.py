import random
import pygame as pg
from settings import *
import numpy as np
import os
import torch
from enum import Enum

class Snake:

    def __init__(self, id, head, direction, nn):
        self.id = id
        self.snake = [head]
        self.direction = direction
        self.nn = nn
        self.score = 0
        self.steps = 0
        self.dead = False
        self.uniq = [0] * 100

    def move(self, board, food):
        """Take a direction to move.
        
        Args:
            action: The the action of next move, 0: keeep straight, 1: turn left, 2: turn right.
        """
        self.steps += 1
        state = self.get_state(board)
        action = self.nn.predict(state) 
        # idx = DIRECTIONS.index(self.direction)
        # if action == 1:    # Turn left.
        #     self.direction = DIRECTIONS[(idx - 1 + 4) % 4]
        # elif action == 2:  # Turn right.
        #     self.direction = DIRECTIONS[(idx + 1) % 4]
        # else keep straight.

        self.direction = DIRECTIONS[action]
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, head)

        has_eat = False
        if (head[0] < 0 or head[0] >= len(board) or head[1] < 0 or head[1] >= len(board[0]) or
            board[head[0]][head[1]] == self.id):  # Hit the wall or itself or other.
            self.snake.pop()
            self.dead = True
        else:
            if board[head[0]][head[1]] == FOOD:  # Eat the food.
                self.score += 1
                has_eat = True
            else:
                tail = self.snake.pop()
                board[tail[0]][tail[1]] = -1

                # Check if arises infinate loop.
                if (head, food) not in self.uniq:
                    self.uniq.append((head, food))
                    del self.uniq[0]
                else:                         # Infinate loop.
                    self.dead = True

            board[head[0]][head[1]] = self.id

        return has_eat

    def get_state(self, board):
        # Head direction.
        i = DIRECTIONS.index(self.direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        '''
        # Tail direction.
        tail_direction = (self.snake[-2][0] - self.snake[-1][0], self.snake[-2][1] - self.snake[-1][1])
        i = DIRECTIONS.index(tail_direction)
        tail_dir = [0.0, 0.0, 0.0, 0.0]
        tail_dir[i] = 1.0
        '''
        state = head_dir

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
            while x < len(board) and x >= 0 and y < len(board[0]) and y >= 0:
                if board[x][y] == FOOD:
                    see_food = 1.0  
                    dis_to_food = dis
                elif board[x][y] == self.id:
                    see_self = 1.0 
                    dis_to_self = dis
                elif board[x][y] != -1:
                    see_other = 1.0
                    dis_to_other = dis
                dis += 1
                x += dir[0]
                y += dir[1]
            state += [1.0/dis, see_food, see_self]
        
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
        self.board = []

    def play(self, nn, seed=None):
        """Use the Neural Network to play the game.

        Args:
            nn: Neural Network.
            seed: The random seed to generate food serials and initial position of snake.
                  It is used for reproduction.      
        """
        self.rand = random.Random(seed)
        self.new(nn)
        while True:
            self._event()
            has_eat = False

            snake = self.snake
            if not snake.dead:
                has_eat = snake.move(self.board, self.food)

            if snake.dead:
                break

            if has_eat:
                self.food = self.place_sth(FOOD)

            self._draw()

        return self.snake.score, self.snake.steps, None

    def play_saved_model(self, score):
        """Use the saved Neural Network model play the game.

        Args:
            score: Specify which model to load, also indicates the highest score it can get.
        """
        model_pth = os.path.join("model0", "best_individual", "nn_"+str(score)+'.pth')
        nn = torch.load(model_pth)

        seed_pth = os.path.join("seed0", "seed_"+str(score)+'.txt')  # Get the seed for reproduction.
        with open(seed_pth, "r") as f:
            seed = int(f.read())
 
        self.play(nn, seed)

    def new(self, nn):
        # empty: -1, snake1: 0, snake2: 1, food: 2
        self.board = [[-1 for _ in range(self.X)] for _ in range(self.Y)]

        # Create new snakes, both with 1 length.
        head = self.place_sth(0)
        direction = DIRECTIONS[self.rand.randint(0, 3)]
        self.snake = Snake(0, head, direction, nn)
        
        self.food = self.place_sth(FOOD)
  
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

        return cell

    def _get_xy(self, pos):
        """Transform pos to the coordinates of pygame."""
        x = pos[0] * GRID_SIZE
        y = pos[1] * GRID_SIZE + BLANK_SIZE
        return (x, y)

    def _draw(self):
        self.screen.fill(BLACK)
        
        # Draw head1.
        if not self.snake.dead:
            x, y = self._get_xy(self.snake.snake[0])
            pg.draw.rect(self.screen, WHITE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, WHITE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # Draw body1.
        for s in self.snake.snake[1:]:
            x, y = self._get_xy(s)
            pg.draw.rect(self.screen, BLUE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, BLUE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # Draw food.
        x, y = self._get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        
        # Draw text.
        text = "score: " + str(self.snake.score)
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
    g.play_saved_model(15)
    pg.quit()
