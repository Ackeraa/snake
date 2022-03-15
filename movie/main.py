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
        self.play(ReplacementTransform(p1[:idx].copy(), c1_l))

        c1_r = copy.deepcopy(p2[idx:])
        c1_r.next_to(c1_l, RIGHT, buff=0)
        self.play(ReplacementTransform(p2[idx:].copy(), c1_r))

        c2_l = copy.deepcopy(p2[:idx])
        c2_l.next_to(p2[idx//2], DOWN * 7)
        self.play(ReplacementTransform(p2[:idx].copy(), c2_l))

        c2_r = copy.deepcopy(p1[idx:])
        c2_r.next_to(c2_l, RIGHT, buff=0)
        self.play(ReplacementTransform(p1[idx:].copy(), c2_r))

        self.play(FadeOut(c2_l, shift=RIGHT), FadeOut(c2_r, shift=RIGHT), 
                  c1_l.animate.shift(RIGHT * 3.3), c1_r.animate.shift(RIGHT * 3.3))


        c = VGroup()
        for i in range(idx):
            c.add(copy.deepcopy(c1_l[i])).arrange(RIGHT, buff=0)
        for i in range(n - idx):
            c.add(copy.deepcopy(c1_r[i])).arrange(RIGHT, buff=0)

        self.add(c)
        c.move_to(c1_r[1])
        c.shift(RIGHT*0.3)
        self.remove(c1_l, c1_r)

        for i in range(n):
            c[i][0].set_fill(WHITE, opacity=0)
        self.wait()

        self.play(c[0][0].animate.set_fill(TEAL, opacity=0.4),
                  c[2][0].animate.set_fill(TEAL, opacity=0.4),
                  c[4][0].animate.set_fill(TEAL, opacity=0.4),
                  c[6][0].animate.set_fill(TEAL, opacity=0.4), run_time=0.4)
       
        cc = copy.deepcopy(c)
        cc.next_to(c, DOWN * 7)
        cc[0][1].text= "12"
        self.play(ReplacementTransform(c.copy(), cc))

        self.remove(p1, p2, c)
        self.play(cc.animate.shift(UP * 6))

        self.genes = cc
        self.wait()

class Final(Scene):
    def construct(self):
        #self.add_sound("bgm.mp3")
        #self.play_game(10, 6)
        self.add_genes()
        #self.add_nn()
        #self.transform_genes_to_nn_edges()
        #self.train_process()
        self.wait()

    def train_process(self):
        self.play(self.nn.animate.shift(RIGHT * 5).scale(0.9),
                  self.nn_edges.animate.shift(RIGHT * 5).scale(0.9))
        self.play_game(10, 3)
        self.play(self.matrix.animate.shift(LEFT * 4, UP * 1.5).scale(0.6))

    def play_game(self, size, score):
        model_pth = os.path.join("../", "model", "best_individual", "nn_"+str(score)+'.pth')
        nn = torch.load(model_pth)

        seed_pth = os.path.join("../seed", "seed_"+str(score)+'.txt')
        with open(seed_pth, "r") as f:
            seed = int(f.read())
 
        self.matrix = get_matrix(size, 0.7, GRAY)
        self.add(self.matrix)
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
                    self.matrix[i][j].set_fill(BLACK, opacity=1)

            self.matrix[head[1]][head[0]].set_fill(WHITE, opacity=1)
            self.matrix[food[1]][food[0]].set_fill(PURE_RED, opacity=1)
            for body in bodys: 
                self.matrix[body[1]][body[0]].set_fill(PURE_BLUE, opacity=1)
            self.wait(0.3)

        self.wait(1)
        self.play(FadeOut(self.matrix))
            
    def transform_genes_to_nn_edges(self):
        anims = [ReplacementTransform(self.genes, self.nn_edges)]
        for e in self.nn_edges:
            anims.append(e.animate.set_color(BLUE))

        self.play(*anims)

    def add_nn(self):
        self.nn = VGroup()
        struct = [32, 20, 12, 4]
        for i in range(4):
            self.nn.add(VGroup(*[Circle(0.06, color=WHITE, fill_opacity=1) 
                            for _ in range(struct[i])]).arrange(DOWN, buff=0.1)).arrange(RIGHT, 2)
        self.nn.shift(LEFT * 2)
        self.add(self.nn)

        self.nn_edges = VGroup()
        for i in range(3):
            for j in range(struct[i]):
                for k in range(struct[i + 1]):
                    e = Line(self.nn[i][j], self.nn[i + 1][k], stroke_width=0.5, color=PURE_RED)
                    self.add(e)
                    self.nn_edges.add(e)

        text1 = Text("Input", font_size=18).next_to(self.nn[0], UP * 0.5)
        always(text1.next_to, self.nn[0], UP * 0.5)
        text2 = Text("Hidden1", font_size=18).next_to(self.nn[1], UP * 0.5)
        always(text1.next_to, self.nn[0], UP * 0.5)
        text3 = Text("Hidden2", font_size=18).next_to(self.nn[2], UP * 0.5)
        always(text1.next_to, self.nn[0], UP * 0.5)
        text4 = Text("Output", font_size=18).next_to(self.nn[3], UP * 0.5)
        always(text1.next_to, self.nn[0], UP * 0.5)
        self.play(FadeIn(text1), 
                  FadeIn(text2),
                  FadeIn(text3),
                  FadeIn(text4), run_time=0.3)

    def add_genes(self):
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
        anims = []
        for i in range(idx):
            anims.append(p1[i][0].animate.set_fill(BLUE, opacity=0.4))
            anims.append(p2[i][0].animate.set_fill(GREEN, opacity=0.4))
        for i in range(idx, n):
            anims.append(p1[i][0].animate.set_fill(MAROON, opacity=0.4))
            anims.append(p2[i][0].animate.set_fill(PINK, opacity=0.4))
        self.play(*anims)


        c1_l = copy.deepcopy(p1[:idx])
        c1_l.next_to(p1[idx//2], DOWN * 7)
        self.play(ReplacementTransform(p1[:idx].copy(), c1_l))

        c1_r = copy.deepcopy(p2[idx:])
        c1_r.next_to(c1_l, RIGHT, buff=0)
        self.play(ReplacementTransform(p2[idx:].copy(), c1_r))

        c2_l = copy.deepcopy(p2[:idx])
        c2_l.next_to(p2[idx//2], DOWN * 7)
        self.play(ReplacementTransform(p2[:idx].copy(), c2_l))

        c2_r = copy.deepcopy(p1[idx:])
        c2_r.next_to(c2_l, RIGHT, buff=0)
        self.play(ReplacementTransform(p1[idx:].copy(), c2_r))

        self.play(FadeOut(c2_l, shift=RIGHT), FadeOut(c2_r, shift=RIGHT), 
                  c1_l.animate.shift(RIGHT * 3.3), c1_r.animate.shift(RIGHT * 3.3))


        c = VGroup()
        for i in range(idx):
            c.add(copy.deepcopy(c1_l[i])).arrange(RIGHT, buff=0)
        for i in range(n - idx):
            c.add(copy.deepcopy(c1_r[i])).arrange(RIGHT, buff=0)

        self.add(c)
        c.move_to(c1_r[1])
        c.shift(RIGHT*0.3)
        self.remove(c1_l, c1_r)

        for i in range(n):
            c[i][0].set_fill(WHITE, opacity=0)
        self.wait()

        self.play(c[0][0].animate.set_fill(TEAL, opacity=0.4),
                  c[2][0].animate.set_fill(TEAL, opacity=0.4),
                  c[4][0].animate.set_fill(TEAL, opacity=0.4),
                  c[6][0].animate.set_fill(TEAL, opacity=0.4), run_time=0.4)
       
        cc = copy.deepcopy(c)
        cc.next_to(c, DOWN * 7)
        cc[0].update_text("da")
        self.play(ReplacementTransform(c.copy(), cc))

        self.remove(p1, p2, c)
        self.play(cc.animate.shift(UP * 6))

        self.genes = cc
        self.wait()

class Fig4(Scene):
    def construct(self):
        text = Text("ad")
        text.add_updater(lambda x: x.set_text("1"))
        text.set_text("1")
        self.add(text)
        self.wait()

