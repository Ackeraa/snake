import sys
sys.path.append('../')
from manim import *
from helper import *
from ai_game_noui import Game 
import random
import torch
import copy

class Fig1(Scene):
    def construct(self):
        self.play_game(10, 6)
        self.wait()

    def play_game(self, size, score):
        model_pth = os.path.join("../", "model", "best_individual", "nn_"+str(score)+'.pth')
        nn = torch.load(model_pth)

        seed_pth = os.path.join("../seed", "seed_"+str(score)+'.txt')
        with open(seed_pth, "r") as f:
            seed = int(f.read())
 
        matrix = get_matrix(size, 0.7, GRAY)
        self.add(matrix)
        game = Game()
        game.rand = random.Random(seed)
        game.new()
        while not game.game_over:
            state = game.get_state()
            action = nn.predict(state)
            game.move(action)
            
            head = game.snake[0]
            bodys = game.snake[1:]
            food = game.food
            
            if game.game_over:
                break

            for i in range(size):
                for j in range(size):
                    matrix[i][j].set_fill(BLACK, opacity=1)

            matrix[head[1]][head[0]].set_fill(WHITE, opacity=1)
            matrix[food[1]][food[0]].set_fill(PURE_RED, opacity=1)
            for body in bodys: 
                matrix[body[1]][body[0]].set_fill(PURE_BLUE, opacity=1)
            
            self.wait(0.1)

class Fig2(Scene):
    def construct(self):
        self.add_nn()
        self.wait()

    def add_nn(self):
        self.nn = VGroup()
        struct = [32, 20, 4]
        for i in range(3):
            self.nn.add(VGroup(*[Circle(0.06, color=WHITE, fill_opacity=1) 
                            for _ in range(struct[i])]).arrange(DOWN, buff=0.1)).arrange(RIGHT, 3)
        self.add(self.nn)

        self.nn_edges = []
        for i in range(2):
            edges = []
            for j in range(struct[i]):
                edge = []
                for k in range(struct[i + 1]):
                    e = Line(self.nn[i][j], self.nn[i + 1][k], stroke_width=0.5, color=PURE_RED)
                    self.add(e)
                    edge.append(e)
                edges.append(edge)
            self.nn_edges.append(edges)
        
class Fig3(Scene):
    def construct(self):
        n = 10
        v1 = [round(random.random(),3) for _ in range(n)]
        v1[5] = '...'
        v2 = [round(random.random(),3) for _ in range(n)]
        v2[5] = '...'

        p1 = Array(n, v1)
        p2 = Array(n, v2)

        vg = VGroup()
        vg.add(p1, p2).arrange(RIGHT, buff=1)
        vg.move_to(2*UP)
        self.add(vg)
        #self.play(FadeIn(vg))
    
        idx = 3
        for i in range(idx):
            self.play(p1[i][0].animate.set_fill(BLUE, opacity=0.4), 
                      p2[i][0].animate.set_fill(GREEN, opacity=0.4), run_time=0.1)
        for i in range(idx, n):
            self.play(p1[i][0].animate.set_fill(MAROON, opacity=0.4), 
                      p2[i][0].animate.set_fill(PINK, opacity=0.4), run_time=0.1)

        c1_l = copy.deepcopy(p1[:idx])
        c1_l.next_to(p1[idx//2], DOWN * 7)
        self.play(Transform(p1[:idx].copy(), c1_l))

        c1_r = copy.deepcopy(p2[idx:])
        c1_r.next_to(c1_l, RIGHT, buff=0)
        self.play(Transform(p2[idx:].copy(), c1_r))

        c2_l = copy.deepcopy(p2[:idx])
        c2_l.next_to(p2[idx//2], DOWN * 7)
        self.play(Transform(p2[:idx].copy(), c2_l))

        c2_r = copy.deepcopy(p1[idx:])
        c2_r.next_to(c2_l, RIGHT, buff=0)
        self.play(Transform(p1[idx:].copy(), c2_r))

        self.wait()
