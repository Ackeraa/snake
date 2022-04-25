import random
from settings import *
import numpy as np
from nn import Net


class Snake:

    def get_state(self, food):
        i = DIRECTIONS.index(self.direction)
        head_dir = [0.0, 0.0, 0.0, 0.0]
        head_dir[i] = 1.0

        state = head_dir
        
        dirs = [[0, -1], [1, -1], [1, 0], [1, 1], 
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        
        for dir in dirs:
            x = self.body[0][0] + dir[0]
            y = self.body[0][1] + dir[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            while x >= 0 and x < self.board_x and y >= 0 and y < self.board_y:
                if (x, y) == food:
                    see_food = 1.0  
                elif (x, y) in self.body:
                    see_self = 1.0 
                dis += 1
                x += dir[0]
                y += dir[1]
            state += [1.0/dis, see_food, see_self]
        
        return state




###########################################################################
