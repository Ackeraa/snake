import torch
import random
import numpy as np
from collections import deque
from ai_game_noui import Game
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

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
            s_dir_u, s_dir_d, s_dir_l, s_dir_r,
            f_dir_u, f_dir_d, f_dir_l, f_dir_r,
            coll_s, coll_l, coll_r
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, playing):
        self.memory.append((state, action, reward, next_state, playing))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, playings = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, playings)

    def train_short_memory(self, state, action, reward, next_state, playing):
        self.trainer.train_step(state, action, reward, next_state, playing)

    def get_action(self, state, num):
        self.epsilon = num - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train(num):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()

    while agent.n_games < 100:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, num)
        reward, playing, score = game.move(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, playing)

        agent.remember(state_old, final_move, reward, state_new, playing)

        if not playing:
            game.new()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()
    return record

if __name__ == "__main__":
    ans_num = 0
    record_num = 0
    for num in range(101, 120):
        record = train(num)
        print(record)
        if record > record_num:
            record_num = record
            ans_num = num
    print(ans_num, record_num)
