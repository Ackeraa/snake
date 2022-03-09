from manim import *
import math
from queue import Queue

def get_array(size, square_size=1, color=WHITE):
    return VGroup(*[Square(square_size).set_color(color) for _ in range(size)]).arrange(RIGHT, buff=0)

def get_matrix(size, square_size=1, color=WHITE):
    return VGroup(*[get_array(size, square_size, color) for _ in range(size)]).arrange(DOWN, buff=0)

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
    def __init__(self, size=0.6, text="", color=WHITE):
        square = Square(size).set_collor(color)
        text = Text(str(text), color=color).scale(0.3).move_to(square.get_center())
        super().__init__(square, text)

class Array(VGroup):
    def __init__(self, length, texts, size=0.6, color=WHITE):
        super().__init__()
        for i in range(length):
            self.add(Grid(size, texts[i], color)).arrange(RIGHT, buff=0)
