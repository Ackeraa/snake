import random
from settings import *
import numpy as np
from nn import Net


class Snake:

    def move(self, food):
        self.steps += 1
        state = self.get_state(food)
        action = self.nn.predict(state) 

        self.direction = DIRECTIONS[action]
        head = (self.body[0][0] + self.direction[0],
                self.body[0][1] + self.direction[1])

        has_eat = False
        if (head[0] < 0 or head[0] >= self.board_x or head[1] < 0 or\
                head[1] >= self.board_y or head in self.body[:-1]):
            self.dead = True
        else:
            self.body.insert(0, head)
            if head == food:                  
                self.score += 1
                has_eat = True
            else:                           
                self.body.pop()
                if (head, food) not in self.uniq:
                    self.uniq.append((head, food))
                    del self.uniq[0]
                else:                       
                    self.dead = True

        return has_eat




###########################################################################
