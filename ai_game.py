import pygame as pg
import random
from settings import *
import numpy as np
import torch
import os

class Snake:

    def __init__(self, head, direction, nn, board_x, board_y):
        self.body = [head]
        self.direction = direction
        self.nn = nn
        self.score = 0
        self.steps = 0
        self.gap_steps = 0
        self.dead = False
        self.uniq = [0] * 100
        self.board_x = board_x
        self.board_y = board_y

    def move(self, food):
        """Take a direction to move.
        
        Args:
            action: The the action of next move, 0: keeep straight, 1: turn left, 2: turn right.
        """
        self.steps += 1
        self.gap_steps += 1
        state = self.get_state(food)
        action = self.nn.predict(state) 

        self.direction = DIRECTIONS[action]
        head = (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1])

        has_eat = False
        if (head[0] < 0 or head[0] >= self.board_x or head[1] < 0 or head[1] >= self.board_y
                or head in self.body):  # Hit the wall or itself.
            self.dead = True
        else:
            self.body.insert(0, head)
            if head == food:  # Eat the food.
                self.score += 1
                has_eat = True
                self.gap_steps = 0
            else:                             # Nothing happened.
                self.body.pop()
                # Check if arises infinate loop.
                if (head, food) not in self.uniq:
                    self.uniq.append((head, food))
                    del self.uniq[0]
                else:                         # Infinate loop.
                    self.dead = True
                # if self.gap_steps > MAX_STEPS:
                #     self.dead = True

        return has_eat

    def get_state(self, food):
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
            x = self.body[0][0] + dir[0]
            y = self.body[0][1] + dir[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            see_other = 0.0
            dis_to_food = np.inf
            dis_to_self = np.inf
            dis_to_other = np.inf
            while x >= 0 and x < self.board_x and y >= 0 and y < self.board_y:
                if (x, y) == food:
                    see_food = 1.0  
                    dis_to_food = dis
                elif (x, y) in self.body:
                    see_self = 1.0 
                    dis_to_self = dis
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
        score: Food eat by the snake.
        steps: Steps moved of the snake.
        snake: postion of the snake.
        head: head of the snake.
        food: Position of the food.
        available_places: # Places available for snake to move or place food. 
        game_over: A boolean if the game is over.
        win: A boolean if the game is winning.
        uniq: Hash table to detect infinate loop.
        seed: The random seed to generate food serials and initial position of snake.
        rand: Random function.
    """

    def __init__(self, show=False, rows=ROWS, cols=COLS):
        self.Y = rows
        self.X = cols
        self.show = show
        self.snakes = []
        self.food = None
        self.best_score = 0
        self.seed = random.randint(-1000000000, 1000000000)
        self.rand = random.Random(self.seed)

        if show:
            pg.init()
            self.width = cols * GRID_SIZE
            self.height = rows * GRID_SIZE + BLANK_SIZE

            pg.display.set_caption(TITLE)
            self.screen = pg.display.set_mode((self.width, self.height))
            self.clock = pg.time.Clock()
            self.font_name = pg.font.match_font(FONT_NAME)


    # need to deleted.
    def new(self, nns, seed):
        if seed is not None:
            self.seed = seed
            self.rand = random.Random(self.seed)

        # Create new snakes, both with 1 length.
        self.snakes = []
        self.best_score = 0
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        for nn in nns:
            head = self.rand.choice(board)
            direction = DIRECTIONS[self.rand.randint(0, 3)]
            self.snakes.append(Snake(head, direction, nn, self.X, self.Y))
        
        self.food = self.rand.choice(board)

    def play(self, nns, seed=None):
        self.new(nns, seed)
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        #alive_snakes_set = set(self.rand.sample(self.snakes, len(self.snakes)))
        alive_snakes_set = set(self.snakes)
        while alive_snakes_set:
            if self.show:
                self._event()
                self._draw()

            for snake in alive_snakes_set:
                has_eat = snake.move(self.food)
                if has_eat:
                    self.food = self.rand.choice(board)
                if snake.score > self.best_score:
                    self.best_score = snake.score
            alive_snakes = [snake for snake in alive_snakes_set if not snake.dead]
            #alive_snakes_set = set(self.rand.sample(alive_snakes, len(alive_snakes)))
            alive_snakes_set = set(alive_snakes)


        score = [snake.score for snake in self.snakes]
        steps = [snake.steps for snake in self.snakes]

        return score, steps, self.seed

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
 
        self.play([nn], seed)

    def _draw(self):
        self.screen.fill(BLACK)

        # Transform pos to the coordinates of pygame.
        get_xy = lambda pos: (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE + BLANK_SIZE)
        
        # Draw snake.
        for snake in self.snakes:
            if not snake.dead:
                x, y = get_xy(snake.body[0])
                pg.draw.rect(self.screen, WHITE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
                pg.draw.rect(self.screen, WHITE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

            for s in snake.body[1:]:
                x, y = get_xy(s)
                pg.draw.rect(self.screen, BLUE1, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
                pg.draw.rect(self.screen, BLUE2, pg.Rect(x+4, y+4, GRID_SIZE - 8, GRID_SIZE - 8))

        # Draw food.
        x, y = get_xy(self.food)
        pg.draw.rect(self.screen, RED, pg.Rect(x, y, GRID_SIZE, GRID_SIZE))
        
        # Draw text.
        text = "best score: " + str(self.best_score)
        font = pg.font.Font(self.font_name, 20)
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.midtop = ((self.width / 2, 5))
        self.screen.blit(text_surface, text_rect)

        # Draw grid.
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
    game = Game(show=True)
    game.play_saved_model(50)
