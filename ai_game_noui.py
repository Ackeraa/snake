import random
from settings import *
import numpy as np

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
        idx = DIRECTIONS.index(self.direction)
        if action == 1:    # Turn left.
            self.direction = DIRECTIONS[(idx - 2 + 4) % 4]
        elif action == 2:  # Turn right.
            self.direction = DIRECTIONS[(idx + 1) % 4]
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
            else:                             # Nothing happened.
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

    def __init__(self, rows=ROWS, cols=COLS):
        self.Y = rows
        self.X = cols

        self.snakes = []
        self.food = None
        self.board = []
        self.seed = random.randint(-1000000000, 1000000000)
        self.rand = random.Random(self.seed)

    def play(self, nn):
        self.new(nn)
        while True:
            has_eat = False

            snake = self.snakes[0]
            if not snake.dead:
                has_eat = snake.move(self.board, self.food)

            if snake.dead:
                break

            if has_eat:
                self.food = self.place_sth(FOOD)

        return self.snakes[0].score, self.snakes[0].steps, self.seed

    def new(self, nn):
        # empty: -1, snake1: 0, snake2: 1, food: 2
        self.board = [[-1 for _ in range(self.X)] for _ in range(self.Y)]

        # Create new snakes, both with 1 length.
        self.snakes = []
        head1 = self.place_sth(0)
        direction1 = DIRECTIONS[self.rand.randint(0, 3)]
        self.snakes.append(Snake(0, head1, direction1, nn))
        
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

if __name__ == '__main__':
    game = Game()
