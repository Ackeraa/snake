import random
from settings import *
import numpy as np
from nn import Net


class Game:

    def play(self):
        board = [(x, y) for x in range(self.X) for y in range(self.Y)]
        alive_snakes_set = set(self.snakes)
        while alive_snakes_set:
            for snake in alive_snakes_set:
                has_eat = snake.move(self.food)
                if has_eat:
                    self.food = self.rand.choice(board)
            alive_snakes = [snake for snake in alive_snakes_set
                                if not snake.dead]
            alive_snakes_set = set(alive_snakes)

        score = [snake.score for snake in self.snakes]
        steps = [snake.steps for snake in self.snakes]

        return score, steps, self.seed















###########################################################################
