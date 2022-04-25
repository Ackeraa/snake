from manim import *
import math
from queue import Queue
import random

COLORS = ['#FF8080', '#00FF00', '#A6CF8C', '#699C52', '#58C4DD', '#F0AC5F', '#E8C11C', '#B189C6', '#FF0000', '#C55F73']
def random_color1(visible):
    if not visible:
        return BLACK
    colors = ['#FF8080', '#00FF00', '#A6CF8C', '#699C52', '#58C4DD', '#F0AC5F', '#E8C11C', '#B189C6', '#FF0000', '#C55F73']
    return random.choice(colors)

def get_array(size, square_size=1, color=WHITE, stroke_width=3):
    return VGroup(*[Square(square_size, stroke_width=stroke_width).set_color(color) for _ in range(size)]).arrange(RIGHT, buff=0)

def get_matrix(size, square_size=1, color=WHITE, stroke_width=3):
    return VGroup(*[get_array(size, square_size, color, stroke_width) for _ in range(size)]).arrange(DOWN, buff=0)

def put_values_up_array(array, values, color=WHITE):
    return VGroup(*[Text(str(values[i]), color=color).scale(0.6).move_to(array[i].get_center()).shift(UP) for i in range(len(values))])

def put_values_down_array(array, values, color=WHITE):
    return VGroup(*[Text(str(values[i]), color=color).scale(0.6).move_to(array[i].get_center()).shift(DOWN) for i in range(len(values))])

def put_values_in_array(array, values, color=WHITE):
    return VGroup(*[Text(str(values[i]), color=color).scale(0.3).move_to(array[i].get_center()) for i in range(len(values))])

def get_node(size=1, color=WHITE):
    node = Circle(size).set_color(WHITE)
    node.set_fill(color, opacity=1)

    return node


class NN(VGroup):
    def __init__(self, struct, size=0.05, color=BLUE):
        vg = VGroup()
        for i in range(3):
            vg.add(VGroup(*[Circle(size, color=WHITE, fill_opacity=1) 
                            for _ in range(struct[i])]).arrange(DOWN, buff=0.1)).arrange(RIGHT, 2)
        
        e = Line(vg[0][0], vg[1][1], color=WHITE)
        vg.add(e)
        super().__init__(vg)

class Grid(VGroup):
    def __init__(self, size=0.6, text="", color=WHITE, stroke_width=3):
        square = Square(size, stroke_width=stroke_width).set_color(color)
        self.number = text
        text = Text(str(text), color=color).scale(size/2).move_to(square.get_center())
        self.text = text
        self.square = square
        super().__init__(self.square, self.text)

    def update_text(self, text, size=0.3):
        self.number = text
        self.text.become(Text(str(text), color=WHITE).scale(size).move_to(self.square.get_center()))

class Info(VGroup):
    def __init__(self, text="", pos=DOWN*3, color=RED):
        self.pos = pos
        text = self.enhance(text)
        self.text = Text(text, color=color, font_size=16)
        self.text.shift(self.pos)
        super().__init__(self.text)
    
    def update_text(self, text, pos=DOWN*3, font_size=16):
        if pos is not None:
            self.pos = pos
        text = self.enhance(text)
        self.text=self.text.become(Text(text, color=RED, font_size=font_size))
        self.text.shift(self.pos)

    def enhance(self, text):
        text1 = ""
        for t in text:
            text1 += t + " "
        return text1[:-1]

class Array(VGroup):
    def __init__(self, length, texts, size=0.6, color=WHITE, stroke_width=3, direction=RIGHT):
        super().__init__()
        for i in range(length):
            self.add(Grid(size, texts[i], color, stroke_width=stroke_width)).arrange(direction, buff=0)

class Individual(VGroup):
    def __init__(self, happy=True, visible=True):
        super().__init__()
        head = Circle(stroke_width=2).set_color(random_color1(visible))
        eye_color = random_color1(visible)
        eyes = VGroup(Line(LEFT*0.2, RIGHT*0.2, stroke_width=1, color = eye_color),
                      Line(LEFT*0.2, RIGHT*0.2, stroke_width=1, color = eye_color))\
                      .arrange(RIGHT, buff=0.6).shift(UP*0.3)
        mouth = Triangle(stroke_width=1).scale(0.3).shift(DOWN*0.5).set_color(random_color1(visible))
        self.happy = happy
        if happy:
            mouth.rotate(180*DEGREES).shift(DOWN*0.1)
        arm_color = random_color1(visible)
        arm1 = Line(UP*0.24+RIGHT*0.84, LEFT*0.7+DOWN*0.2, stroke_width=2,
                    color = arm_color).shift(DOWN*1.25+LEFT*0.85)
        arm2 = Line(UP*0.24+LEFT*0.84, RIGHT*0.7+DOWN*0.2, stroke_width=2,
                    color = arm_color).shift(DOWN*1.25+RIGHT*0.85)
        leg_color = random_color1(visible)
        leg1 = Line(UP, DOWN*0.8, stroke_width=2, color = leg_color).shift(DOWN*2.2+LEFT*0.7)
        leg2 = Line(UP, DOWN*0.8, stroke_width=2, color = leg_color).shift(DOWN*2.2+RIGHT*0.7)

        self.add(head, eyes[0], eyes[1], mouth, arm1, arm2, leg1, leg2)

    def turn_mouth(self):
        if self.happy:
            self[3].rotate(180*DEGREES)
        else:
            self[3].rotate(180*DEGREES).shift(DOWN*0.1)


class Genes(VGroup):
    def __init__(self, n=10):
        super().__init__()
        v = [round(random.random(),3) for _ in range(n)]
        v[n//2] = '...'
        p = Array(n, v, stroke_width=1, direction=DOWN)
        self.add(p)

class NN(VGroup):
    def __init__(self):
        super().__init__()
        nn = VGroup()
        struct = [21, 14, 8, 4]
        #buffs = [0.1, 0.3, 0.3, 0.3]
        for i in range(4):
            nn.add(VGroup(*[Circle(0.06, color=WHITE, stroke_width=1.5) 
                for _ in range(struct[i])]).arrange(DOWN, buff=0.1)).arrange(RIGHT, 1)
        nn.shift(LEFT * 2)
        self.add(nn)

        nn_edges = VGroup()
        for i in range(3):
            for j in range(struct[i]):
                for k in range(struct[i + 1]):
                    e = Line(nn[i][j], nn[i + 1][k], stroke_width=0.3, color=WHITE)
                    self.add(e)

class Game(VGroup):
    def __init__(self):
        super().__init__()
        matrix = get_matrix(10, 0.7, GRAY, stroke_width=1).scale(0.2)
        matrix[5][5].set_fill(WHITE, opacity=1)
        matrix[5][6].set_fill(PURE_BLUE, opacity=1)
        matrix[4][6].set_fill(PURE_BLUE, opacity=1)
        matrix[2][2].set_fill(PURE_RED, opacity=1)
        self.add(matrix)

class Ellipsi(VGroup):
    def __init__(self, color=WHITE):
        super().__init__()
        text = Text("...", color=color)
        self.add(text)

class MyCode(Code):
    def __init__(self, code, font_size=12, bkg="rectangle"):
        super().__init__(
            code,
            font_size=font_size,
            tab_width=4,
            insert_line_no=False,
            background=bkg,
            style=Code.styles_list[13],
            font="Monospace",
            background_stroke_color=BLUE_A,
            language="Python",
            line_spacing=0.5,
            margin=0.2,
        )
        self.code[-1].set_fill(RED, opacity=0)
        #self.background_mobject.set_fill(BLACK, opacity=0.1)

class CodeArr(Triangle):
    def __init__(self, code, line):
        super().__init__(color=RED_E)
        self.set_fill(RED, opacity=1).rotate(-90*DEGREES).scale(.03)
        self.code = code
        self.h = code.code[0].height
        self.move(line, False)

    def move(self, line, animate=True):
        if animate:
            return self.animate.next_to(self.code.code[line], LEFT*0.5).shift(
                                DOWN*(self.code.code[line].height/2-self.h/1.8))
        else:
            return self.next_to(self.code.code[line], LEFT*0.5).shift(
                                DOWN*(self.code.code[line].height/2-self.h/1.8))
