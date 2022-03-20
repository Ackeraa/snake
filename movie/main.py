from re import S
import sys
sys.path.append('../')
from manim import *
from helper import *
from ai_game_noui import Game 
import random
import numpy as np
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
        #self.add_genes()
        self.add_nn()
        #self.transform_genes_to_nn_edges()
        #self.train_process()
        self.whole_picture()
        self.wait()

    def whole_picture(self):
        n = 10
        m = 5
        vg0 = VGroup()
        for i in range(m):
            v1 = [round(random.random(),3) for _ in range(n)]
            v1[5] = '...'
            v2 = [round(random.random(),3) for _ in range(n)]
            v2[5] = '...'

            p1 = Array(n, v1, size=0.2)
            p2 = Array(n, v2, size=0.2)
            for i in range(n):
                p1[i][0].set(stroke_width=1)
                p2[i][0].set(stroke_width=1)

            vg = VGroup()
            vg.add(p1, p2).arrange(RIGHT, buff=0.5)
            vg0.add(vg).arrange(DOWN, buff=1)

        others = VGroup(Text("......").scale(0.5))
        vg0.add(VGroup(others)).arrange(DOWN, buff=1)
        vg0.shift(UP*0.7+LEFT*4.5)
        self.add(vg0)
        vg1 = VGroup()

        anims1, anims2, anims3, anims4, anims5 = [], [], [], [], []
        for i in range(m):
            idx = random.randint(1, n - 2)
            if idx % 2 == 0:
                idx += 1
            p1 = vg0[i][0]
            p2 = vg0[i][1]

            c1 = VGroup(*copy.deepcopy(p1[:idx]), *copy.deepcopy(p2[idx:])).arrange(RIGHT, buff=0) 
            c1.next_to(p1, DOWN*1.5)
            anims1.append(ReplacementTransform(p1[:idx].copy(), c1[:idx]))
            anims2.append(ReplacementTransform(p2[idx:].copy(), c1[idx:]))

            c2 = VGroup(*copy.deepcopy(p2[:idx]), *copy.deepcopy(p1[idx:])).arrange(RIGHT, buff=0) 
            c2.next_to(p2, DOWN*1.5)
            anims1.append(ReplacementTransform(p1[:idx].copy(), c2[:idx]))
            anims2.append(ReplacementTransform(p1[idx:].copy(), c2[idx:]))

            anims5.append(p2.animate.next_to(p1, DOWN*0.3))
            anims5.append(c2.animate.next_to(c1, DOWN*0.3))
            vg1.add(VGroup(c1, c2))

            ls = [i for i in range(n)]
            num = random.randint(0, 4)
            cs = random.sample(ls, num)
            for j in cs:
                if c1[j][1].text != "...": 
                    anims3.append(c1[j][0].animate.set_fill(TEAL, opacity=0.4))
                    new_t1 = str(round(float(c1[j][1].text) + np.random.normal()*0.2, 2))
                    anims4.append(c1[j].animate.update_text(new_t1, size=0.1))
            num = random.randint(0, n)
            cs = random.sample(ls, 4)
            for j in cs:
                if c2[j][1].text != "...": 
                    anims3.append(c2[j][0].animate.set_fill(TEAL, opacity=0.4))
                    new_t1 = str(round(float(c2[j][1].text) + np.random.normal()*0.2, 2))
                    c2[j].animate.update_text(new_t1, size=0.15)
                    anims4.append(c2[j].animate.update_text(new_t1, size=0.1))

        '''
        self.play(*anims1, run_time=1)
        self.play(*anims2, run_time=1)
        self.play(*anims3, run_time=1)
        self.play(*anims4, run_time=1)
        self.play(*anims5, run_time=1)
        '''
        
        self.wait()

    def train_process(self):
        self.play(self.nn.animate.shift(RIGHT * 5).scale(0.9),
                  self.nn_edges.animate.shift(RIGHT * 5).scale(0.9))
        t1 = Text("上", font_size=12).next_to(self.nn[3][0], RIGHT, buff=0.1)
        t2 = Text("下", font_size=12).next_to(self.nn[3][1], RIGHT, buff=0.1)
        t3 = Text("左", font_size=12).next_to(self.nn[3][2], RIGHT, buff=0.1)
        t4 = Text("右", font_size=12).next_to(self.nn[3][3], RIGHT, buff=0.1)
        self.add(t1, t2, t3, t4)

        text_l = [Text("1.蛇首方向: ", font_size=18),
                Text("2.蛇尾方向: ", font_size=18),
                Text("3.蛇首八个方向(从垂直向上起，顺时针45')上,", font_size=18),
                Text("是否有食物: ", font_size=18),
                Text("是否有自身: ", font_size=18),
                Text("与墙的距离(取倒数): ", font_size=18)]
        self.text_vg = VGroup(*text_l).arrange(DOWN, center=False, aligned_edge=LEFT)  
        self.text_vg[3].shift(RIGHT*0.3)
        self.text_vg[4].shift(RIGHT*0.3)
        self.text_vg[5].shift(RIGHT*0.3)

        self.text_vg.shift(LEFT * 6 + DOWN)

        self.add(self.text_vg)
        self.play_game(10, 6, True)
 
    def play_game(self, size, score, is_train=False):
        model_pth = os.path.join("../", "model", "best_individual", "nn_"+str(score)+'.pth')
        nn = torch.load(model_pth)

        seed_pth = os.path.join("../seed", "seed_"+str(score)+'.txt')
        with open(seed_pth, "r") as f:
            seed = int(f.read())
 
        self.matrix = get_matrix(size, 0.7, GRAY)
        if is_train:
            self.matrix.shift(LEFT * 4.5, UP * 1.5).scale(0.6)
        self.add(self.matrix)
        game = Game()
        game.rand = random.Random(seed)
        game.new()
        self.text_vg2 = None
        while not game.game_over:
            state = game.get_state()
            action = nn.predict(state)
            if is_train:
                self.update_state(state, action)
            
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
            if is_train:
                self.wait(1)
            else:
                self.wait(0.3)
            game.move(action)

        self.wait(1)
        self.play(FadeOut(self.matrix))

    def update_state(self, state, action):
        if self.text_vg2 is not None:
            self.remove(self.text_vg2)
            for i in range(4):
                for node in self.nn[i]:
                    node.set_color(WHITE)

        see_food, see_self, dis = [], [], []
        for i in range(8):
            dis.append(state[i * 3])
            see_food.append(state[i * 3 + 1])
            see_self.append(state[i * 3 + 2])

        head_dir = state[24:28]
        tail_dir = state[28:32]
        def get_text(a, y):
            s = "["
            for x in a:
                c = str(round(x, 2)) if y else str(int(x))
                s += c + ",   "
            return s[:-4]+"]"

        text = [Text(get_text(head_dir, 0), font_size=18).next_to(self.text_vg[0], RIGHT),
                Text(get_text(tail_dir, 0), font_size=18).next_to(self.text_vg[1], RIGHT),
                Text(get_text(see_food, 0), font_size=18).next_to(self.text_vg[3], RIGHT),
                Text(get_text(see_self, 0), font_size=18).next_to(self.text_vg[4], RIGHT),
                Text(get_text(dis, 1), font_size=18).next_to(self.text_vg[5], RIGHT)]
        self.text_vg2 = VGroup(*text)
        self.add(self.text_vg2)
        #self.play(Transform(self.text_vg2.copy(), self.nn[0]))
        ids = head_dir + tail_dir + see_food + see_self + dis
        for i in range(len(ids)):
            if ids[i] != 0:
                self.nn[0][i].set_color(BLUE)

        pre = 0
        for i in range(3):
            edges = self.nn_edges[pre : pre + self.struct[i] * self.struct[i + 1]]
            anims = []
            for edge in edges:
                e = edge.copy().set_color(PINK)
                anims.append(ShowPassingFlash(e.copy().set_color(PINK),
                                              run_time=1,
                                              time_width=1))
            #self.play(*anims)

            if i != 2:
                ids = random.sample([i for i in range(self.struct[i + 1])], 
                                      random.randint(4, self.struct[i + 1] - 1))
                for j in ids:
                    self.nn[i + 1][j].set_color(BLUE)
            else:
                self.nn[i + 1][action].set_color(BLUE)

            pre += self.struct[i] * self.struct[i + 1]

    def transform_genes_to_nn_edges(self):
        anims = [ReplacementTransform(self.genes, self.nn_edges)]
        colors = [random_color() for _ in range(10)]
        for e in self.nn_edges:
            anims.append(e.animate.set_stroke_width(random.random())\
                         .set_color(random.choice(colors)))

        self.play(*anims)

    def add_nn(self):
        self.nn = VGroup()
        struct = [32, 20, 12, 4]
        self.struct = struct
        for i in range(4):
            self.nn.add(VGroup(*[Circle(0.06, color=WHITE, stroke_width=1.5) 
                            for _ in range(struct[i])]).arrange(DOWN, buff=0.1)).arrange(RIGHT, 2)
        self.nn.shift(LEFT * 2)
        self.add(self.nn)

        self.nn_edges = VGroup()
        for i in range(3):
            for j in range(struct[i]):
                for k in range(struct[i + 1]):
                    e = Line(self.nn[i][j], self.nn[i + 1][k], stroke_width=0.3, color=WHITE)
                    self.add(e)
                    self.nn_edges.add(e)

        text1 = Text("Input", font_size=18).next_to(self.nn[0], UP * 0.5)
        always(text1.next_to, self.nn[0], UP * 0.5)
        text2 = Text("Hidden1", font_size=18).next_to(self.nn[1], UP * 0.5)
        always(text2.next_to, self.nn[1], UP * 0.5)
        text3 = Text("Hidden2", font_size=18).next_to(self.nn[2], UP * 0.5)
        always(text3.next_to, self.nn[2], UP * 0.5)
        text4 = Text("Output", font_size=18).next_to(self.nn[3], UP * 0.5)
        always(text4.next_to, self.nn[3], UP * 0.5)
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

        arrow1 = Arrow(start=0.5*UP, end=DOWN).shift(UP)
        text1 = Text("交叉", font_size=18).next_to(arrow1, RIGHT)
        self.play(FadeIn(arrow1), FadeIn(text1))

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
        self.wait()
        arrow2 = Arrow(start=0.5*UP, end=DOWN).shift(DOWN * 1.2)
        text2 = Text("变异", font_size=18).next_to(arrow2, RIGHT)
        self.play(FadeIn(arrow2), FadeIn(text2))

        cc = copy.deepcopy(c)
        cc.next_to(c, DOWN * 7)
        new_t1 = str(round(float(cc[0][1].text) + np.random.normal()*0.2, 2))
        new_t2 = str(round(float(cc[2][1].text) + np.random.normal()*0.2, 2))
        new_t3 = str(round(float(cc[4][1].text) + np.random.normal()*0.2, 2))
        new_t4 = str(round(float(cc[6][1].text) + np.random.normal()*0.2, 2))
        cc[0].update_text(new_t1)
        cc[2].update_text(new_t2)
        cc[4].update_text(new_t3)
        cc[6].update_text(new_t4)
        self.play(ReplacementTransform(c.copy(), cc))
        self.wait(2)

        self.remove(p1, p2, c, arrow1, text1, arrow2, text2)
        self.play(cc.animate.shift(UP * 6),
                  cc[0][0].animate.shift(UP * 6).set_fill(BLACK, opacity=0),
                  cc[2][0].animate.shift(UP * 6).set_fill(BLACK, opacity=0),
                  cc[4][0].animate.shift(UP * 6).set_fill(BLACK, opacity=0),
                  cc[6][0].animate.shift(UP * 6).set_fill(BLACK, opacity=0))

        self.genes = cc
        self.wait()

class Fig4(Scene):
    def construct(self):
        circle = Circle(0.06, color=WHITE, stroke_width=1.2)
        self.add(circle)
        self.wait()

