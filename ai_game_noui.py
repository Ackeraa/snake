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
            action: The indics of the direction to move, between 0 and 3.
        """
        self.steps += 1
        state = self.get_state(board)
        action = self.nn.predict(state) 
        self.direction = DIRECTIONS[action]
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, head)

        has_eat = False
        if (head[0] < 0 or head[0] >= len(board) or head[1] < 0 or head[1] >= len(board[0]) or
            board[head[0]][head[1]] != -1):  # Hit the wall or itself or other.
            self.snake.pop()
            self.dead = True
        else:
            tail = self.snake.pop()
            board[tail[0]][tail[1]] = -1
            if board[head[0]][head[1]] == 2:  # Eat the food.
                self.score += 1
                has_eat = True
            else:                             # Nothing happened.
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
            state += [1.0/dis, see_food, see_self, see_other]
        
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

    def play(self, nn1, nn2):
        self.new(nn1, nn2)
        while True:
            first_to_move = random.randint(0, 1)
            has_eat1 = has_eat2 = False

            snake1 = self.snakes[first_to_move]
            if not snake1.dead:
                has_eat1 = snake1.move(self.board, self.food)

            snake2 = self.snakes[first_to_move^1]
            if not snake2.dead:
                has_eat2 = snake2.move(self.board, self.food)

            if snake1.dead and snake2.dead:
                break

            if has_eat1 or has_eat2:
                self.food = self.place_sth(FOOD)

        return self.snakes[0].score, self.snakes[0].steps,\
               self.snakes[1].score, self.snakes[1].steps, 

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
            self.game_over = True
            return

        cell = self.rand.choice(empty_cells)
        self.board[cell[0]][cell[1]] = sth

        return cell

    def move(self, action):
        """Take a direction to move.
        
        Args:
            action: The indics of the direction to move, between 0 and 3.
        """
        self.steps += 1
        direction = DIRECTIONS[action]
        self.head = (self.head[0] + direction[0], self.head[1] + direction[1])
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
            
            # Check if arises infinate loop.
            if (self.head, self.food) not in self.uniq:
                self.uniq.append((self.head,self.food))
                del self.uniq[0]
            else:
                self.game_over = True

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

if __name__ == '__main__':
    game = Game()
