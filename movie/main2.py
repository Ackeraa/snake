from manim import *
from helper import *
import random
import math

class CodeFromString(Scene):
    def construct(self):
        code = Code(
            "test.py",
            tab_width=4,
            insert_line_no=False,
            background="Window",
            style=Code.styles_list[14],
            font="Monospace",
            language="Python",
        )
        code = MyCode("test.py")
        code.shift(LEFT * 2 + UP * 2)
        code.width=6
        self.add(code)
        self.wait()

class Main(Scene):
    def construct(self):
        self.info = Info("")
        #self.add_flow()
        self.add_code_pic()
        self.add_snake()

        self.wait()

    def info_update(self, text):
        self.play(self.info.animate.update_text(text))

    def add_flow(self):

        #self.info_update("生成种群")
        n = 25
        m = int(math.sqrt(n))
        population = VGroup()
        for i in range(n):
            if i == n // 2:
                population.add(Ellipsi().scale(0.5))
            else:
                population.add(Individual(happy=random.random()>0.5).scale(0.2))

        population.arrange_in_grid(rows=m).shift(UP*0.5)
        #self.play(FadeIn(population))
        self.add(population)

        # Individual.
        #self.info_update("")
        #self.play(population.animate.shift(LEFT*5.5).scale(0.5))
        population.shift(LEFT*5.5).scale(0.5)

        arr1 = Arrow(start=LEFT+UP, end=RIGHT, stroke_width=2,
                     max_tip_length_to_length_ratio=0.05, color=BLUE)
        arr2 = Arrow(start=LEFT, end=RIGHT, stroke_width=2,
                     max_tip_length_to_length_ratio=0.05, color=BLUE)
        arr3 = Arrow(start=LEFT+DOWN, end=RIGHT, stroke_width=2,
                     max_tip_length_to_length_ratio=0.05, color=BLUE)
        arr_vg = VGroup(arr1, arr2, arr3).arrange(DOWN, buff=2.3)
        arr_vg.shift(LEFT*1.9)

        individuals = VGroup()
        for i in range(m):
            if i == m // 2:
                individuals.add(Ellipsi().scale(0.5))
            else:
                individuals.add(Individual(happy=random.random()>0.5).scale(0.2))
        individuals.arrange(DOWN, buff=0.6).shift(LEFT*2.5)

        arr = VGroup()
        dot = Dot().move_to(population[14])
        for i in range(m):
            arr.add(Arrow(start=dot, end=individuals[i], stroke_width=1.5,
                          max_tip_length_to_length_ratio=0.05, color=GRAY_B))
        #self.play(FadeIn(arr), FadeIn(individuals))
        self.add(arr, individuals)

        # Genes.
        genes = VGroup()
        anims = []
        for i in range(m):
            if i == m // 2:
                genes.add(Ellipsi(color=BLACK).scale(0.5))
            else:
                genes.add(Genes(5).scale(0.3))
                #anims.append(ReplacementTransform(individuals[i][2].copy(), genes[i]))
        genes.arrange(DOWN, buff=0.6).shift(LEFT*2)
        #self.play(*anims)
        self.add(genes)

        # NN.
        nns = VGroup()
        anims = []
        for i in range(m):
            if i == m // 2:
                nns.add(Ellipsi().scale(0.5))
            else:
                nns.add(NN().scale(0.25))
            #anims.append(ReplacementTransform(genes[i].copy(), nns[i]))
        nns.arrange(DOWN, buff=0.6)
        #self.play(*anims)

        arr1 = VGroup()
        for i in range(m):
            dot1 = Dot().move_to(genes[i]).shift(LEFT*0.2)
            dot2 = Dot().move_to(nns[i]).shift(LEFT*0.3)
            arr1.add(Arrow(start=dot1, end=dot2, stroke_width=1.5,
                          max_tip_length_to_length_ratio=0.05, color=GRAY_B))
        self.add(arr1)
        self.add(nns)

        # Game.
        games = VGroup()
        for i in range(m):
            if i == m // 2:
                games.add(Ellipsi().scale(0.5))
            else:
                games.add(Game().scale(0.7))
        games.arrange(DOWN, buff=0.7).shift(RIGHT*2)

        arr2 = VGroup()
        for i in range(m):
            if i == m // 2:
                arr2.add(Ellipsi().scale(0.5))
            else:
                dot1 = Dot().move_to(nns[i]).shift(RIGHT*0.5+DOWN*0.3)
                dot2 = Dot().move_to(games[i]).shift(LEFT*0.6+DOWN*0.3)
                ar = CurvedArrow(start_point=[dot1.get_x(), dot1.get_y(), 0],
                                  end_point=[dot2.get_x(), dot2.get_y(), 0],
                                  stroke_width=1.5, color=GRAY_B)
                x0, y0 = ar.get_x(), ar.get_y()
                x1, y1 = 4 * dot1.get_x() - 3 * x0, 4 * dot1.get_y() - 3 * y0
                x2, y2 = 4 * dot2.get_x() - 3 * x0, 4 * dot2.get_y() - 3 * y0
                ar = CurvedArrow(start_point=[x1, y1, 0], end_point=[x2, y2, 0],
                                 stroke_width=1.5, color=GRAY_B).scale(0.25)
                ar1 = CurvedArrow(start_point=[x2, y2, 0], end_point=[x1, y1, 0],
                                 stroke_width=1.5, color=GRAY_B).scale(0.25).shift(DOWN*0.5)

                arr2.add(ar, ar1)
        self.add(arr2)
        self.add(games)

        # Simulate.
        rects = VGroup()
        for i in range(m):
            if i != m // 2:
                rects.add(Rectangle(stroke_width=1.5, color=BLUE_C).surround(VGroup(nns[i], games[i])))
        self.add(rects)

        # Show passing flash.
        anims = []
        anims2 = []
        ids = [0, 2, 5, 7]
        ids2 = [1, 3, 6, 8]
        for i in range(0, m - 1):
            e2 = arr2[ids[i]].copy()
            e1 = arr2[ids2[i]].copy()
            run_time = 1
            color = BLUE_E
            anims.append(ShowPassingFlash(e1.copy().set_color(color),
                                          run_time=run_time,
                                          time_width=run_time))
            color = RED_E
            anims2.append(ShowPassingFlash(e2.copy().set_color(color),
                                          run_time=run_time,
                                          time_width=run_time))
        '''
        for _ in range(5):
            self.play(*anims)
            self.play(*anims2)
        '''
        
        # Fitness.
        fits = VGroup()
        for i in range(m):
            if i == m // 2:
                fits.add(Ellipsi(BLACK).scale(0.5))
            else:
                fits.add(MathTex("f(score, steps)").scale(0.4).next_to(games[i], RIGHT))
        
        self.add(fits)

        # Envolve.

        #self.play(ReplacementTransform(fits, population))
        #self.play(FadeOut(*self.mobjects[1:]), population.animate.shift(RIGHT*3).scale(2))
        saved_mobjects = self.mobjects[1:].copy()
        self.remove(*self.mobjects[1:])
        population.shift(RIGHT*3).scale(2)

        n_die = 16
        n_alive = n - n_die
        indics = [i for i in range(len(population)) if i != n//2]
        dies = random.sample(indics, n_die)
        alives = list(set(indics) - set(dies))
        self.wait()
        for x in dies:
            self.remove(population[x])

        children = VGroup(*[Individual(visible=False).scale(0.2) for _ in range(n_die)])\
                    .arrange_in_grid(rows=int(math.sqrt(n_die))).next_to(population, RIGHT * 3)

        num = len(population[0])
        indics = [i for i in range(num)]

        # Cross.
        anims = []
        for i in range(n_die):
            i1, i2 = random.sample(alives, 2)
            p1, p2 = population[i1], population[i2]
            n_from_p1 = random.randint(1, num - 1)
            from_p1 = random.sample(indics, n_from_p1)
            from_p2 = list(set(indics) - set(from_p1))
            
            vp1 = VGroup()
            vc1 = VGroup()
            for x in from_p1:
                children[i][x].set_color(p1[x].color)
                if x == 3 and children[i].happy != p1.happy:
                    children[i].turn_mouth()
                vp1.add(p1[x])
                vc1.add(children[i][x])
            anims.append(ReplacementTransform(vp1.copy(), vc1))

            vp2 = VGroup()
            vc2 = VGroup()
            for x in from_p2:
                children[i][x].set_color(p2[x].color)
                if x == 3 and children[i].happy != p2.happy:
                    children[i].turn_mouth()
                vp2.add(p2[x])
                vc2.add(children[i][x])
            anims.append(ReplacementTransform(vp2.copy(), vc2))

        self.add(children) #to be deleted.

        # Mutate.
        indics = [1, 7, 8, 13]
        anims = []
        # mouth.
        anims.append(children[1][3].animate.become(
                Circle(stroke_width=1).scale(0.06).next_to(children[1][2], DOWN*0.3).shift(LEFT*0.1)))
        # head
        anims.append(children[7][0].animate.become(
                Circle(stroke_width=2).scale(0.25).move_to(children[7][3]).shift(UP*0.1)))

        # head
        anims.append(children[8][0].animate.become(
                Triangle(stroke_width=2).scale(0.4).move_to(children[8][3]).shift(UP*0.2)))

        # arm
        anims.append(FadeOut(children[13][6]))

        self.play(*anims)
        
        # New population.
        anims = []
        for i, x in enumerate(dies):
            population[x] = children[i].copy().move_to(population[x])
            anims.append(ReplacementTransform(children[i], population[x]))
        self.play(*anims)

        # Next generation.
        population.shift(LEFT*3).scale(0.5)
        self.add(*saved_mobjects)

        self.remove(*self.m_objects)

    def add_code_pic(self):

        code1 = MyCode("snake_1.py")
        code2 = MyCode("nn_1.py")
        code3 = MyCode("main_1.py")
        self.codes = VGroup(code1, code2, code3).arrange(RIGHT, buff=1).scale(0.7)

        title1 = MathTex("snake.py", font_size=22).set_color([BLUE_E, TEAL_D]).next_to(code1, UP*0.2)
        title2 = MathTex("nn.py", font_size=22).set_color([BLUE_E, TEAL_D]).next_to(code2, UP*0.2)
        title3 = MathTex("main.py", font_size=22).set_color([BLUE_E, TEAL_D]).next_to(code3, UP*0.2)
        self.codes_title = VGroup(title1, title2, title3)
        self.add(self.codes_title)

        t1 = Text("实现贪吃蛇", font_size=16, color=TEAL_D).next_to(title1, UP)
        t2 = Text("实现神经网络", font_size=16, color=TEAL_D).next_to(title2, UP)
        t3 = Text("实现遗传算法", font_size=16, color=TEAL_D).next_to(title3, UP)
        self.codes_desc = VGroup()
        self.codes_desc.add(t1, t2, t3)
        #self.play(FadeIn(t1))

        self.add(self.codes, self.codes_desc)
    
    def add_snake(self):
    
        title = MathTex("snake.py", font_size=32).set_color([BLUE_E, TEAL_D]).shift(UP*3.5)
        anims = []
        anims.append(self.codes[0].animate.shift(LEFT*0.2+UP).scale(1.1))
        anims.append(ReplacementTransform(self.codes_title[0], title))
        anims.append(FadeOut(self.codes[1:]))
        anims.append(FadeOut(self.codes_title[1:]))
        anims.append(FadeOut(self.codes_desc))
        self.play(*anims)

class Test(Scene):
    def construct(self):
        c = Circle()
        hair = VGroup(Line(UP*0.24+LEFT*0.84, RIGHT*0.7+DOWN*0.2, stroke_width=2),
                      Line(UP*0.24+RIGHT*0.84, LEFT*0.7+DOWN*0.2, stroke_width=2).shift(RIGHT*1.4)).scale(0.3)
        self.add(hair)
