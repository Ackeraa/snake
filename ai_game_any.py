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

        self.direction = DIRECTIONS[action]

        # idx = DIRECTIONS.index(self.direction)
        # if action == 1:    # Turn left.
        #     self.direction = DIRECTIONS[(idx - 1 + 4) % 4]
        # elif action == 2:  # Turn right.
        #     self.direction = DIRECTIONS[(idx + 1) % 4]
        # else keep straight.
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

    def play(self, nn1, nn2, seed=None):
        """Use the Neural Network to play the game.

        Args:
            nn: Neural Network.
            seed: The random seed to generate food serials and initial position of snake.
                  It is used for reproduction.      
        """
        self.rand = random.Random(seed)
        self.new(nn1, nn2)
        while self.food is not None:
            self._event()
            first_to_move = self.rand.randint(0, 1)
            has_eat1 = has_eat2 = False

            snake1 = self.snakes[first_to_move]
            snake2 = self.snakes[first_to_move^1]

            if not snake1.dead:
                has_eat1 = snake1.move(self.board, self.food)
                if snake1.dead:
                    for x in range(self.X):
                        for y in range(self.Y):
                            if self.board[x][y] == snake1.id:
                                self.board[x][y] = -1
                if not snake2.dead:
                    for pos in snake2.snake:
                        self.board[pos[0]][pos[1]] = snake2.id

            if not snake2.dead:
                has_eat2 = snake2.move(self.board, self.food)
                if snake2.dead:
                    for x in range(self.X):
                        for y in range(self.Y):
                            if self.board[x][y] == snake2.id:
                                self.board[x][y] = -1
                if not snake1.dead:
                    for pos in snake1.snake:
                        self.board[pos[0]][pos[1]] = snake1.id

            if snake1.dead and snake2.dead:
                break

            if has_eat1 or has_eat2:
                self.food = self.place_sth(FOOD)

            self._draw()

        return self.snakes[0].score, self.snakes[0].steps,\
               self.snakes[1].score, self.snakes[1].steps, None 

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

    def new(self, nn1, nn2):
        # empty: -1, snake1: 0, snake2: 1, food: 2
        self.board = [[-1 for _ in range(self.X)] for _ in range(self.Y)]

        # Create new snakes, both with 1 length.
        self.snakes = []
        head1 = self.place_sth(0)
        direction1 = DIRECTIONS[self.rand.randint(0, 3)]
        self.snakes.append(Snake(0, head1, direction1, nn1))
                                                                 
        head2 = self.place_sth(1)
        direction2 = DIRECTIONS[self.rand.randint(0, 3)]
        self.snakes.append(Snake(1, head2, direction2, nn2))
        
        self.food = self.place_sth(FOOD)
  
    def place_sth(self, sth):
        empty_cells = []
        for x in range(self.X):
            for y in range(self.Y):
                if self.board[x][y] == -1:
                    empty_cells.append((x, y))
        if empty_cells == []:
            return None

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
        if not self.snakes[0].dead:
            x, y = self._get_xy(self.snakes[0].snake[0])
            pg.draw.rect(self.screen, WHITE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, WHITE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # Draw body1.
        for s in self.snakes[0].snake[1:]:
            x, y = self._get_xy(s)
            pg.draw.rect(self.screen, BLUE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, BLUE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # Draw head2.
        if not self.snakes[1].dead:
            x, y = self._get_xy(self.snakes[1].snake[0])
            pg.draw.rect(self.screen, WHITE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
            pg.draw.rect(self.screen, WHITE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

            # Draw body2.
            for s in self.snakes[1].snake[1:]:
                x, y = self._get_xy(s)
                pg.draw.rect(self.screen, GREEN1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
                pg.draw.rect(self.screen, GREEN2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # Draw food.
        if self.food is not None:
            x, y = self._get_xy(self.food)
            pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        
        # Draw text.
        text = "score1: " + str(self.snakes[0].score) +  "   score2: " + str(self.snakes[1].score)
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
