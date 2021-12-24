import torch
import random
import numpy as np
from collections import deque
from ai_game import Game
from ai_game1 import SnakeGameAI, Direction, Point
from model1 import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def get_state0(self, game):

        # dir : up, down, left, right,
        # food: up, down, left, right,
        # coll: up, down, left, right

        s_dir = game.snake[-1].direction
        s_pos = game.snake[-1].pos
        s_dir_u = s_dir == (-1, 0)
        s_dir_d = s_dir == (1, 0)
        s_dir_l = s_dir == (0, -1)
        s_dir_r = s_dir == (0, 1)

        f_pos = game.food.pos
        f_dir_u = f_pos[0] < s_pos[0]
        f_dir_d = f_pos[0] > s_pos[0]
        f_dir_l = f_pos[1] < s_pos[1]
        f_dir_r = f_pos[1] > s_pos[1]

        empty = game.empty_cells
        coll_pos_u = (s_pos[0] - 1, s_pos[1])
        coll_pos_d = (s_pos[0] + 1, s_pos[1])
        coll_pos_l = (s_pos[0], s_pos[1] - 1)
        coll_pos_r = (s_pos[0], s_pos[1] + 1)
        coll_s = (s_dir_u and coll_pos_u not in empty) or\
                 (s_dir_d and coll_pos_d not in empty) or\
                 (s_dir_l and coll_pos_l not in empty) or\
                 (s_dir_r and coll_pos_r not in empty) 
        coll_l = (s_dir_u and coll_pos_l not in empty) or\
                 (s_dir_d and coll_pos_r not in empty) or\
                 (s_dir_l and coll_pos_d not in empty) or\
                 (s_dir_r and coll_pos_u not in empty) 
        coll_r = (s_dir_u and coll_pos_r not in empty) or\
                 (s_dir_d and coll_pos_l not in empty) or\
                 (s_dir_l and coll_pos_u not in empty) or\
                 (s_dir_r and coll_pos_d not in empty) 

        state = [
            coll_s, coll_l, coll_r,
            s_dir_u, s_dir_d, s_dir_l, s_dir_r,
            f_dir_u, f_dir_d, f_dir_l, f_dir_r
        ]

        s = np.array(state, dtype=int)
        print(s[0:3], s[3:7], s[7:11])
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 240 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    train()
