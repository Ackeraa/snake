from manim import *
sys.path.append('../')
from ai_game import Game as AiGame
from helper import *
import random
import math
import copy


class Main(Scene):
    def construct(self):
        self.info = Info("")
        self.add_sound("bgm.mp3")
        self.add_begin()
        self.add_flow()
        self.add_code_pic()
        self.add_ai_game()
        self.add_nn()
        self.add_main()
        self.add_end()

    def add_end(self):
        t = Text("The End.", font_size=24)
        self.play(Create(t))
        self.wait(1.5)

    def info_update(self, text, pos=DOWN*3, font_size=16):
        self.play(self.info.animate.update_text(text, pos, font_size))

    def info_clear(self):
        return self.info.animate.update_text("", pos=None, font_size=20)

    def add_begin(self):
        genes_list = []
        n = 100
        size = 20
        for i in range(n):
            genes_pth = os.path.join("../", "genes", "all", str(i))
            with open(genes_pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                genes_list.append(genes)
 
        self.matrix = get_matrix(size, 0.3, stroke_width=1.5, color=GRAY_C)
        self.add(self.matrix)
        game = AiGame(genes_list=genes_list)
        for snake in game.snakes:
            snake.color = random_color()

        board = [(x, y) for x in range(game.X) for y in range(game.Y)]
        alive_snakes_set = set(game.snakes)
        steps = 0

        set_food = []
        set_head = []
        set_body = []
        set_color = []
        while alive_snakes_set:
            steps += 1
            for snake in alive_snakes_set:
                has_eat = snake.move(game.food)
                if has_eat:
                    game.food = game.rand.choice(board)
                if snake.score > game.best_score:
                    game.best_score = snake.score
            alive_snakes = [snake for snake in alive_snakes_set if not snake.dead]
            alive_snakes_set = set(alive_snakes)

            set_food.append(game.food)
            bodys = []
            heads = []
            colors = []
            for snake in alive_snakes:
                tmp_body = []
                for body in snake.body:
                    tmp_body.append(body)
                bodys.append(tmp_body)
                heads.append(snake.body[0])
                colors.append(snake.color)
            set_head.append(heads)
            set_body.append(bodys)
            set_color.append(colors)

        nums = 2000
        shows = [[i for i in range(nums)]]
        #shows.append([i for i in range(steps//2-2*nums, steps//2)])
        #shows.append([i for i in range(steps-nums, steps-20)])
        for show in shows:
            for step in show:
                for i in range(size):
                    for j in range(size):
                        self.matrix[i][j].set_fill(BLACK, opacity=1)
                food = set_food[step]
                heads = set_head[step]
                bodys = set_body[step]
                colors = set_color[step]
                self.matrix[food[1]][food[0]].set_fill(PURE_RED, opacity=1)
                # for head in heads:
                    # self.matrix[head[1]][head[0]].set_fill(WHITE, opacity=1)
                for i in range(len(bodys)):
                    for body in bodys[i]:
                        self.matrix[body[1]][body[0]].set_fill(colors[i], opacity=1)
                self.wait(0.04)
                # if step == show[0] and show != shows[0]:
                    # self.play(FadeIn(self.matrix), run_time=1)
            # if show != shows[-1]:
                # self.play(FadeOut(self.matrix), run_time=1)

        self.play(FadeOut(self.matrix))

    def add_flow(self):

        sm = VGroup()
        n = 25
        m = int(math.sqrt(n))
        population = VGroup()
        for i in range(n):
            if i == n // 2:
                population.add(Ellipsi().scale(0.5))
            else:
                population.add(Individual(happy=random.random()>0.5).scale(0.2))

        population.arrange_in_grid(rows=m).shift(UP*0.5)
        self.info_update("种群", pos=RIGHT*5, font_size=16)
        self.play(FadeIn(population))

        # Individual.
        self.play(population.animate.shift(LEFT*5.5).scale(0.5),
                  self.info_clear())

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
        self.info_update("个体", pos=RIGHT*5, font_size=16)
        self.play(FadeIn(arr), FadeIn(individuals))
        sm.add(individuals)
        sm.add(arr)

        # Genes.
        genes = VGroup()
        anims = []
        for i in range(m):
            if i == m // 2:
                genes.add(Ellipsi(color=BLACK).scale(0.5))
            else:
                genes.add(Genes(5).scale(0.3))
                anims.append(ReplacementTransform(individuals[i][2].copy(), genes[i]))
        genes.arrange(DOWN, buff=0.6).shift(LEFT*2)
        self.info_update("基因", pos=RIGHT*5, font_size=16)
        self.wait(0.3)
        self.play(*anims)
        sm.add(genes)

        # NN.
        nns = VGroup()
        anims = []
        for i in range(m):
            if i == m // 2:
                nns.add(Ellipsi().scale(0.5))
            else:
                nns.add(NN().scale(0.25))
            anims.append(ReplacementTransform(genes[i].copy(), nns[i]))
        nns.arrange(DOWN, buff=0.6)
        sm.add(nns)

        arr1 = VGroup()
        for i in range(m):
            dot1 = Dot().move_to(genes[i]).shift(LEFT*0.2)
            dot2 = Dot().move_to(nns[i]).shift(LEFT*0.3)
            arr1.add(Arrow(start=dot1, end=dot2, stroke_width=1.5,
                          max_tip_length_to_length_ratio=0.05, color=GRAY_B))
        self.info_update("转化为神经网络参数", pos=RIGHT*5, font_size=16)
        self.play(FadeIn(arr1))
        self.play(*anims)
        sm.add(arr1)

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
        self.info_update("进行游戏，直至蛇死亡", pos=RIGHT*5, font_size=16)
        self.play(FadeIn(games))
        sm.add(games)

        # Simulate.
        rects = VGroup()
        for i in range(m):
            if i != m // 2:
                rects.add(Rectangle(stroke_width=1.5, color=BLUE_C).surround(VGroup(nns[i], games[i])))
        self.play(FadeIn(rects))
        self.play(FadeIn(arr2))
        sm.add(rects)
        sm.add(arr2)

        # Show passing flash.
        anims = []
        anims2 = []
        ids = [0, 2, 5, 7]
        ids2 = [1, 3, 6, 8]
        t1s = []
        t2s = []
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
            t1s.append(Text("传入状态", font_size=11).next_to(arr2[ids2[i]],UP*0.1))
            t2s.append(Text("传回结果", font_size=11).next_to(arr2[ids[i]],DOWN*0.1))
        self.play(*anims)
        self.play(FadeIn(*t1s))
        self.play(*anims2)
        self.play(FadeIn(*t2s))
        sm.add(*t1s)
        sm.add(*t2s)
        for _ in range(2):
            self.play(*anims)
            self.play(*anims2)
        
        # Fitness.
        fits = VGroup()
        for i in range(m):
            if i == m // 2:
                fits.add(Ellipsi(BLACK).scale(0.5))
            else:
                fits.add(MathTex("f(score, steps)").scale(0.4).next_to(games[i], RIGHT))
        
        self.info_update("奖励函数, 计算适应度", pos=RIGHT*5, font_size=16)
        self.play(FadeIn(fits))
        self.wait()

        # Envolve.

        self.play(Transform(fits, population), self.info_clear())
        self.wait(0.4)
        self.play(FadeOut(*sm, fits), population.animate.shift(RIGHT*3).scale(2))

        n_die = 16
        n_alive = n - n_die
        indics = [i for i in range(len(population)) if i != n//2]
        dies = random.sample(indics, n_die)
        alives = [i for i in indics if i not in dies]
        self.info_update("淘汰个体", pos=RIGHT*5, font_size=16)
        run_time = 1
        for x in dies:
            self.play(FadeOut(population[x]), run_time=run_time)
            run_time -= 0.05

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
        self.info_update("交叉", pos=RIGHT*5, font_size=16)
        for anim in anims:
            self.play(anim, run_time=0.3)

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

        self.info_update("变异", pos=RIGHT*5, font_size=16)
        for i in range(len(anims)):
            self.play(Indicate(children[indics[i]], color=RED))
            self.play(anims[i])
        
        # New population.
        anims = []
        for i, x in enumerate(dies):
            population[x] = children[i].copy().move_to(population[x])
            anims.append(ReplacementTransform(children[i], population[x]))
        self.info_update("生成下一代", pos=RIGHT*5, font_size=16)
        self.play(*anims)
        self.wait()

        # Next generation.
        self.play(population.animate.shift(LEFT*3).scale(0.5),
                  FadeIn(*sm), self.info_clear())
        self.info_update("进行下一轮", pos=RIGHT*5, font_size=16)
        self.wait()
        self.play(FadeOut(*self.mobjects))

    def add_code_pic(self):

        code1 = MyCode("ai_game_1.py")
        code2 = MyCode("nn_1.py")
        code3 = MyCode("main_1.py")
        self.codes = VGroup(code1, code2, code3).arrange(RIGHT, buff=1)

        title1 = MathTex("ai\_game.py", font_size=22).set_color([BLUE_E, TEAL_D]).next_to(code1, UP*0.2)
        title2 = MathTex("nn.py", font_size=22).set_color([BLUE_E, TEAL_D]).next_to(code2, UP*0.2)
        title3 = MathTex("main.py", font_size=22).set_color([BLUE_E, TEAL_D]).next_to(code3, UP*0.2)
        self.codes_title = VGroup(title1, title2, title3)

        color = RED
        t1 = Text("实 现 贪 吃 蛇", font_size=16, color=color).next_to(title1, UP)
        t2 = Text("实 现 神 经 网 络", font_size=16, color=color).next_to(title2, UP)
        t3 = Text("实 现 遗 传 算 法", font_size=16, color=color).next_to(title3, UP)
        self.codes_desc = VGroup()
        self.codes_desc.add(t1, t2, t3)
        for i in range(3):
            self.play(FadeIn(self.codes_desc[i], shift=DOWN))
            self.play(FadeIn(self.codes[i], self.codes_title[i], scale=0.5))
            self.wait(1)
    
    def add_ai_game(self):
    
        title = MathTex("ai\_game.py", font_size=24).set_color([BLUE_E, TEAL_D]).shift(UP*3.5+RIGHT*3)
        anims = []
        code = MyCode("ai_game_2.py")
        anims.append(ReplacementTransform(self.codes[0], code))
        anims.append(ReplacementTransform(self.codes_title[0], title))
        anims.append(FadeOut(self.codes[1:]))
        anims.append(FadeOut(self.codes_title[1:]))
        anims.append(FadeOut(self.codes_desc))
        self.play(*anims)

        anims = []
        lines = [8, 11, 14, 19, 22]
        infos = ["初始化蛇", "蛇移动", "获取当前状态（作为神经网络输入）",
                 "初始化游戏（可含有多条蛇)", "进行一轮游戏"]
        arr = CodeArr(code, lines[0])
        self.play(FadeIn(arr))
        self.info_update(infos[0])
        for i in range(1, len(lines)):
            self.play(arr.move(lines[i]))
            self.info_update(infos[i])
            self.wait(1.5)
        # snake
        ## __init__
        code3 = MyCode("ai_game_3.py").code
        self.play(FadeOut(code.code[8:], shift=DOWN),
                  FadeOut(arr), self.info_clear())
        code.code[8:] = code3[8:]
        self.play(FadeIn(code.code[8:], shift=UP))
        self.play(Indicate(code.code[9:16], color=RED))
        self.info_update("初始化蛇的位置、方向、得分等")
        self.wait(1.5)
        self.play(Indicate(code.code[16], color=RED))
        self.info_update("用于判断蛇是否进入死循环")
        self.wait(2)
        self.play(Indicate(code.code[17], color=RED))
        self.info_update("根据基因初始化神经网络")
        self.wait(2)
        
        ## get_state
        code4 = MyCode("ai_game_4.py").code
        self.play(FadeOut(code.code[8:], shift=DOWN),
                  self.info.animate.update_text("", RIGHT*4, font_size=16))
        code.code[8:] = code4[8:]
        self.play(FadeIn(code.code[8:14], shift=UP))
        self.play(Indicate(code.code[9:12], color=RED))
        self.info_update("蛇头方向")
        self.wait(2)
        self.info_update("")
        self.play(Create(code.code[15:35]), run_time=6)
        self.info_update("蛇头8个方向上是否有食物、自身、墙")
        self.wait(2)

        ## move
        code5 = MyCode("ai_game_5.py").code
        self.play(FadeOut(code.code[8:], shift=DOWN), self.info_clear())
        code.code[8:] = code5[8:]
        self.play(FadeIn(code.code[8:], shift=UP))
        arr.move(8, False)
        self.play(FadeIn(arr))
        indics = [10, 11, 14, 18, 22, 23, 27, 28, 34]
        infos = ["获取当前状态", "将状态输入神经网络，预测下一步移动方向", "蛇头下一步的位置",
                 "判断是否撞墙或撞自己", "没有撞墙，插入新的蛇头",  "是否吃到食物",
                 "舍去旧的蛇尾", "判断是否出现死循环", "返回是否吃到了食物"]
        for i in range(len(indics)):
            self.play(arr.move(indics[i]))
            self.info_update(infos[i])
            self.wait(2)

        # game
        ## __init__
        code6 = MyCode("ai_game_6.py").code
        self.play(FadeOut(code.code[6:], shift=DOWN),
                  FadeOut(arr), self.info_clear())
        code.code[6:] = code6[6:]
        self.play(FadeIn(code.code[6:], shift=UP))
        self.play(Indicate(code.code[9:11], color=RED))
        self.info_update("初始化界面大小")
        self.wait(1)
        self.play(Indicate(code.code[11:13], color=RED))
        self.info_update("初始化随机函数（用于训练过程的复现）")
        self.wait(2)
        self.play(Indicate(code.code[14:21], color=RED))
        self.info_update("根据基因列表生成若干条蛇")
        self.wait(2)
        self.play(Indicate(code.code[22], color=RED))
        self.info_update("生成初始食物")
        self.wait(1)

        ## play
        code7 = MyCode("ai_game_7.py").code
        self.play(FadeOut(code.code[8:], shift=DOWN), self.info_clear())
        code.code[8:] = code7[8:]
        self.play(FadeIn(code.code[8:], shift=UP))
        arr.move(11, False)
        self.play(FadeIn(arr))
        self.info_update("当还有蛇存活时")
        self.wait()
        self.play(arr.move(13))
        self.info_update("每条蛇依次移动")
        self.wait()
        self.play(arr.move(15))
        self.info_update("如果食物被吃掉，生成新食物")
        self.wait(1.5)
        self.play(arr.move(16))
        self.info_update("舍弃死亡的蛇")
        self.wait(1.5)
        self.play(arr.move(23))
        self.info_update("返回得分、步数及随机数种子")
        self.wait(1.5)
        self.play(FadeOut(*self.mobjects))

    def add_nn(self):

        # Introduce code structure.
        anims = []
        anims.append(FadeIn(self.codes[1:]))
        anims.append(FadeIn(self.codes_title[1:]))
        anims.append(FadeIn(self.codes_desc[1:]))
        self.play(*anims)
        self.wait()

        anims = []
        title = MathTex("nn.py", font_size=24).set_color([BLUE_E, TEAL_D]).shift(UP*3.7+RIGHT*3)
        code = MyCode("nn_2.py")
        anims.append(ReplacementTransform(self.codes[1], code))
        anims.append(ReplacementTransform(self.codes_title[1], title))
        anims.append(FadeOut(self.codes[2]))
        anims.append(FadeOut(self.codes_title[2]))
        anims.append(FadeOut(self.codes_desc[1:]))
        self.play(*anims)

        anims = []
        lines = [5, 8, 11, 14]
        infos = ["初始化神经网络", "设置神经网络权重", "前向传播", "进行预测"]
        arr = CodeArr(code, lines[0])
        self.play(FadeIn(arr))
        self.info = Info("")
        self.info_update(infos[0])
        for i in range(1, len(lines)):
            self.play(arr.move(lines[i]))
            self.info_update(infos[i])
            self.wait()

        # Introduce code detail.

        ## __init__
        code3 = MyCode("nn_3.py")

        self.play(FadeOut(code.code[5:], shift=DOWN), FadeOut(arr),
                  self.info_clear())
        code.code[5:] = code3.code[5:]
        self.play(FadeIn(code.code[5:34], shift=UP))
        self.play(FadeIn(code.code[34:], shift=UP))
        w_bs = [0.2, 0.3, -0.3, 0.5, 0.5, 0.1, 0.3, 0.6, 0.2, 0.1, 0.8]
        ws = [0.2, 0.3, 0.5, 0.1, 0.6, 0.2]
        bs = [-0.3, 0.5, 0.3, 0.1, 0.8] 
        weights = Array(11, w_bs, size=0.5, stroke_width=2).shift(RIGHT*4+UP*3)
        self.play(code.animate.shift(LEFT*3),
                  code3.animate.shift(LEFT*3),
                  title.animate.shift(LEFT*3))
        arr = CodeArr(code, 36)
        self.play(FadeIn(arr))
        self.play(ReplacementTransform(code.code[36].copy(), weights))
        self.wait()
        self.play(arr.move(37))
        self.wait()
        self.play(arr.move(5))

        # nn
        nn = VGroup()
        struct = [1, 2, 1, 2]
        for i in range(4):
            nn.add(VGroup(*[Circle(0.3, color=WHITE, stroke_width=2) 
                for _ in range(struct[i])]).arrange(DOWN, buff=1)).arrange(RIGHT, 0.5)
        nn.shift(RIGHT * 4 + UP * 0.2)

        nn_edges = VGroup()
        ws_vg = VGroup()
        cnt = 0
        for i in range(3):
            edges = VGroup()
            vg = VGroup()
            for j in range(struct[i]):
                for k in range(struct[i + 1]):
                    e = Line(nn[i][j], nn[i + 1][k], stroke_width=2, color=WHITE)
                    text = MathTex(str(ws[cnt]), color=WHITE, font_size=16).next_to(e, DOWN * 0.05)
                    cnt += 1
                    vg.add(text)
                    edges.add(e)
                ws_vg.add(vg)

            nn_edges.add(edges)

        b = Circle(0.2, color=WHITE, stroke_width=2)
        t = Text("b", font_size=14).move_to(b)
        nn_b = VGroup(*[VGroup(b.copy(), t.copy()) for _ in range(3)]).arrange(RIGHT, buff=0.68)
        nn_b.shift(RIGHT * 3.7 + UP*2)
        nn_bedges = VGroup()
        bs_vg = VGroup()
        cnt = 0
        for i in range(3):
            vg = VGroup()
            vg1 = VGroup()
            for j in range(struct[i + 1]):
                e = DashedLine(nn_b[i], nn[i + 1][j], stroke_width=2, color=WHITE)
                text = MathTex(str(bs[cnt]), color=WHITE, font_size=16).move_to(e).shift(UP * 0.3)
                cnt += 1
                vg.add(e)
                vg1.add(text)
            bs_vg.add(vg1)
            nn_bedges.add(vg)

        self.play(arr.move(6))
        for i in range(4):
            self.play(arr.move(8 + i))
            if i != 3:
                self.play(FadeIn(nn[i]), FadeIn(nn_b[i]))
            else:
                self.play(FadeIn(nn[i]))

        for i in range(3):
            self.play(arr.move(13 + i))
            self.play(FadeIn(nn_edges[i]), FadeIn(nn_bedges[i]))

        # activate function
        self.play(arr.move(17))
        ax = Axes(
            x_range=(-0.2, 1.2, 0.2),
            y_range=(-0.2, 1.2, 0.2),
            axis_config={
                'color': GREY_A,
                'stroke_width': 2,
            },
        )
        ax.scale(0.2).shift(RIGHT*2.5+DOWN*2)
        relu = ax.plot(
            lambda x: max(x, 0),
            use_smoothing=False,
            color=BLUE,
            stroke_width=2,
        )
        self.play(Write(ax, lag_ratio=0.01, run_time=1))
        self.play(Create(relu))

        self.play(arr.move(18))
        ax2 = Axes(
            x_range=(-5, 5, 1),
            y_range=(-0.2, 1, 0.2),
            axis_config={
                'color': GREY_A,
                'stroke_width': 2,
            },
        )
        ax2.scale(0.2).shift(RIGHT*5.5+DOWN*2)
        sigmod = ax2.plot(
            lambda x: 1.0 / (1.0 + math.exp(-x)),
            color=BLUE,
            stroke_width=2,
        )
        self.play(Write(ax2, lag_ratio=0.01, run_time=1))
        self.play(Create(sigmod))

        ## set_weight
        self.play(arr.move(20))
        self.play(FadeOut(code.code[5:21], shift=DOWN), FadeOut(arr))
        code4 = MyCode("nn_4.py").code.shift(LEFT * 3)
        code.code[5:21] = code4[5:21]
        self.play(FadeIn(code.code[5:21], shift=UP))
        arr.move(5, animate=False)
        self.play(arr.move(6))
        self.play(Indicate(weights, color=RED))
        self.play(arr.move(7))
        indics = [2, 4, 6, 7, 9]
        texts = ["x", "xx", "y", "yy", "z"]
        ar = Arrow(start=UP*0.2, end=DOWN*0.1, stroke_width=2,
                     max_tip_length_to_length_ratio=0.3)
        for i in range(5):
            self.play(arr.move(8 + i))
            vg = VGroup(MathTex(texts[i], font_size=18, color=RED),
                        ar.copy()).arrange(DOWN, buff=0.1)
            vg.next_to(weights[indics[i]], UP*0.1).shift(weights[i].width / 2 * LEFT)
            self.play(FadeIn(vg, shift=DOWN))

        indics = [0, 2, 4, 6, 7, 9, 11]
        for i in range(3):
            l = indics[2*i]
            r = indics[2*i+1]
            rr = indics[2*i+2]
            self.play(arr.move(13 + 2 * i))
            self.play(ReplacementTransform(weights[l:r].copy(), ws_vg[i]))
            self.play(arr.move(13 + 2 * i + 1))
            self.play(ReplacementTransform(weights[r:rr].copy(), bs_vg[i]))

        ## predict & forward
        self.play(arr.move(38))
        self.play(FadeOut(code.code[5:21], shift=DOWN))
        code5 = MyCode("nn_5.py").code.shift(LEFT * 3)
        code.code[5:20] = code5[5:20]
        self.play(FadeIn(code.code[5:9], shift=UP))
        self.play(arr.move(5))
        self.play(arr.move(6))
        t_in = Text("1", font_size=18, color=RED).move_to(nn[0][0])
        self.wait(0.4)
        self.play(arr.move(7))
        self.wait(0.4)
        self.play(ReplacementTransform(code.code[7][9:14].copy(), t_in))
        self.play(FadeIn(code.code[9:19], shift=UP), arr.move(10))

        color = BLUE_E
        run_time = 1
        xs = [[-0.1, 0.8], [0.38], [0.328, 0.876]]
        ys = [[0.0, 0.8], [0.38], [0.5813, 0.7060]]
        for i in range(3):
            axs = ax if i < 2 else ax2
            self.play(arr.move(11 + 2 * i))
            self.wait()
            self.play(ShowPassingFlash(nn_edges[i].copy().set_color(color),
                                              run_time=run_time,
                                              time_width=run_time),
                      ShowPassingFlash(nn_bedges[i].copy().set_color(color),
                                              run_time=run_time,
                                              time_width=run_time),
                      )
            node_texts = VGroup()
            x_dots = VGroup()
            y_dots = VGroup()
            lines = VGroup()
            node_texts1 = VGroup()
            for j in range(len(nn[i + 1])):
                node_texts.add(MathTex(str(xs[i][j]), font_size = 17, color=GREEN_C).move_to(nn[i + 1][j]))
                x_dots.add(Dot(axs.coords_to_point(xs[i][j], 0), color=GREEN).scale(0.5))
                y_dots.add(Dot(axs.coords_to_point(0, ys[i][j]), color=RED).scale(0.5))
                lines.add(axs.get_lines_to_point(axs.c2p(xs[i][j],ys[i][j])))
                node_texts1.add(MathTex(str(ys[i][j]), font_size = 17, color=RED).move_to(nn[i + 1][j]))

            self.play(FadeIn(node_texts))
            self.wait(0.5)
            self.play(arr.move(12 + 2 * i))
            self.wait()
            self.play(ReplacementTransform(node_texts, x_dots))
            self.play(FadeIn(lines), FadeIn(y_dots))
            self.wait(0.5)
            self.play(ReplacementTransform(y_dots, node_texts1))
            self.remove(x_dots, lines)
            self.wait(0.5)

        self.play(arr.move(8))
        self.wait()
        self.play(Indicate(nn[3][1], color=RED))
        self.wait()
        self.play(FadeOut(*self.mobjects))

    def add_main(self):
        # Introduce code structure.
        anims = []
        anims.append(FadeIn(self.codes[2:]))
        anims.append(FadeIn(self.codes_title[2:]))
        anims.append(FadeIn(self.codes_desc[2:]))
        self.play(*anims)
        self.wait()

        anims = []
        title = MathTex("main.py", font_size=24).set_color([BLUE_E, TEAL_D]).shift(UP*3.7+RIGHT*3)
        code = MyCode("main_2.py")
        anims.append(ReplacementTransform(self.codes[2], code))
        anims.append(ReplacementTransform(self.codes_title[2], title))
        anims.append(FadeOut(self.codes_desc[2:]))
        self.play(*anims)
        anims = []
        lines = [7, 10, 15, 19, 22, 25, 28, 31, 34]
        infos = ["初始化个体", "计算个体适应度", "初始化遗传算法", "生成初始种群",
                     "交叉", "变异", "精英选择", "轮盘赌选择", "进化"]
        arr = CodeArr(code, lines[0])
        self.play(FadeIn(arr))
        self.info_update(infos[0])
        for i in range(1, len(lines)):
            self.play(arr.move(lines[i]))
            self.info_update(infos[i])
            self.wait(0.8)

        # Individual
        self.play(FadeOut(code.code[13:37], shift=DOWN), FadeOut(arr), self.info_clear())
        code3 = MyCode("main_3.py")
        self.play(ReplacementTransform(code.code[5:13], code3.code[5:18]))
        code.code[5:18] = code3.code[5:18]
        self.wait()
        self.play(Indicate(code3.code[8:12], color=RED))
        self.info_update("初始化个体基因、得分、步数、随机数种子")
        self.wait(2)
        self.play(Indicate(code3.code[14:17], color=RED))
        self.info_update("进行一轮游戏，根据得分、步数计算适应度")
        self.wait(2)
        code4 = MyCode("main_4.py")
        code.code[5:21] = code4.code[5:21]
        self.play(FadeOut(code3.code[5:18], shift=DOWN),
                  FadeIn(code.code[5:21], shift=UP),
                  self.info_clear())

        # GA
        self.play(Indicate(code.code[9:14], color=RED))
        self.info_update("初始化种群大小、基因长度、变异概率等")
        self.wait(2)
        self.play(Indicate(code.code[18:21], color=RED))
        self.info_update("生成初始种群, 基因是(-1,1)区间均匀分布")
        self.wait(2)
        self.play(FadeOut(code.code[7:22], shift=DOWN), self.info_clear())

        ## crossover
        code5 = MyCode("main_5.py")
        code.code[7:17] = code5.code[7:17]   
        self.play(FadeIn(code.code[7:17], shift=UP)) 
        self.play(code.animate.shift(LEFT*3),
                  code5.animate.shift(LEFT*3),
                  title.animate.shift(LEFT*3))
        
        arr.move(7, False) 
        self.play(FadeIn(arr))
        n = 5
        p1 = Array(n, [round(random.random(), 3) for _ in range(n)],
                   size=0.4, stroke_width=2)
        p2 = Array(n, [round(random.random(), 3) for _ in range(n)],
                   size=0.4, stroke_width=2)
        vg_genes = VGroup(p1, p2).arrange(RIGHT, buff=0.5)
        vg_genes.shift(RIGHT*4)
        self.play(FadeIn(vg_genes))

        self.play(arr.move(8))
        ar = Arrow(start=UP*0.2, end=DOWN*0.1, stroke_width=2,
                     max_tip_length_to_length_ratio=0.3)
        ar1 = ar.copy()
        ar.next_to(p1[1], UP*0.1).shift(p1[2].width / 2 * RIGHT)
        ar1.next_to(p2[1], UP*0.1).shift(p2[2].width / 2 * RIGHT)
        self.play(FadeIn(VGroup(ar, ar1), shift=DOWN))

        self.play(arr.move(9))
        pos1 = p1[3].get_center()
        pos2 = p2[3].get_center()
        path1 = ArcBetweenPoints(pos1, pos2, angle=-PI/2)
        path2 = ArcBetweenPoints(pos2, pos1, angle=-PI/2)
        self.play(MoveAlongPath(p1[-3:], path1),
                  MoveAlongPath(p2[-3:], path2), run_time=1)

        self.play(FadeOut(VGroup(ar, ar1)), arr.move(11))

        ## mutation
        self.play(arr.move(12))
        self.wait(0.8)
        mutation_array1 = Array(5, [0, 0, 1, 0, 0], size=0.4, stroke_width=2)
        mutation_array2 = Array(5, [0, 1, 0, 0, 0], size=0.4, stroke_width=2)
        mu_vg = VGroup(mutation_array1, mutation_array2).arrange(RIGHT, buff=0.5)
        mu_vg.shift(RIGHT*4+UP)
        m1 = np.round(np.random.normal(size=5), 3)
        m2 = np.round(np.random.normal(size=5), 3)
        mutation1 = Array(5, list(m1), size=0.4, stroke_width=2) 
        mutation2 = Array(5, list(m2), size=0.4, stroke_width=2) 
        muu_vg = VGroup(mutation1, mutation2).arrange(RIGHT, buff=0.5)
        muu_vg.shift(RIGHT*4+UP*2)

        self.play(FadeIn(mu_vg, shift=DOWN))
        self.play(arr.move(13))
        self.wait(0.5)
        self.play(FadeIn(muu_vg, shift=DOWN))
        self.play(arr.move(14))
        self.wait(0.5)
        self.play(Indicate(mu_vg[0][2], color=RED),
                  Indicate(muu_vg[0][2], color=RED),
                  Indicate(mu_vg[1][1], color=RED),
                  Indicate(muu_vg[1][1], color=RED),
                  )
        self.wait(0.5)
        self.play(mutation1[2].animate.update_text(str(round(m1[2]*0.2, 3)), size=0.2),
                    mutation2[1].animate.update_text(str(round(m1[1]*0.2, 3)), size=0.2))

        self.play(arr.move(15))
        self.wait(0.5)
        self.play(Indicate(mu_vg[0][2], color=RED),
                  Indicate(muu_vg[0][2], color=RED),
                  Indicate(p2[2], color=RED),
                  Indicate(mu_vg[1][1], color=RED),
                  Indicate(muu_vg[1][1], color=RED),
                  Indicate(p2[1], color=RED))
        self.wait(0.5)
        self.play(p2[2].animate.update_text(str(round(m1[2]*0.2+float(p2[2][1].text), 3)), size=0.2),
                    p2[1].animate.update_text(str(round(m1[1]*0.2+float(p2[1][1].text), 3)), size=0.2))

        ## select
        self.play(FadeOut(code.code[7:17], shift=DOWN), FadeOut(arr))
        code6 = MyCode("main_6.py").shift(LEFT*3)
        code.code[7:25] = code6.code[7:25]   
        self.play(FadeIn(code.code[7:25], shift=UP))
        self.wait(0.5)
        self.play(Indicate(code.code[8:11], color=RED))
        self.info = Info("")
        self.info_update("精英选择: 对种群按适应度大小排序，返回前#size个", pos=LEFT*3+DOWN*3)
        self.wait(1.5)
        self.play(Indicate(code.code[13:25], color=RED))
        self.info_update("轮盘赌选择: 适应度越大被选概率越大", pos=LEFT*3+DOWN*3)
        self.wait(1.5)

        n = 10
        angles = [0.05, 0.01, 0.02, 0.4, 0.05, 0.07, 0.01, 0.2, 0.1, 0.09]

        wheel = VGroup()
        lines = VGroup()
        secs = VGroup()
        angle = 1
        line = Line(ORIGIN, RIGHT*2, stroke_width=1, color=GREY_C)
        for i in range(10):
            sec = AnnularSector(outer_radius=2, inner_radius=0, angle=2*PI*angle)
            sec.set_color(COLORS[i])
            secs.add(sec)
            line = line.copy().rotate(-2*PI*angles[i], about_point=RIGHT-RIGHT)
            lines.add(line)
            angle -= angles[i]
        dot = Dot(color=WHITE)
        circle = Circle(2, color=GREY_C, stroke_width=2)
        arrow = Arrow(buff=0, start=ORIGIN, end=RIGHT*1.5, stroke_width=3,
                      color=GREY_E, max_tip_length_to_length_ratio=0.1)
        wheel.add(secs, lines, circle, arrow, dot)
        wheel.shift(RIGHT*4)
        self.play(FadeIn(secs, scale=0.5), FadeIn(lines, scale=0.5), FadeIn(circle, scale=0.5),
                  FadeOut(p1, p2, mu_vg, muu_vg))
        self.play(Create(dot), Create(arrow))
        self.wait(0.3)
        self.play(Rotate(arrow, angle=8*PI+0.75*PI, about_point=dot.get_center(),
                         rate_func=rush_from), run_time=3)

        self.play(FadeOut(wheel), self.info_clear(), FadeOut(code.code[7:25]))

        ## envole
        code7 = MyCode("main_7.py").shift(LEFT*3)
        code.code[7:26] = code7.code[7:26]   
        self.play(FadeIn(code.code[7:26], shift=UP))
        self.wait(0.5)
        arr = CodeArr(code, 7)
        self.play(FadeIn(arr))
        self.info = Info("")
        self.info_update("进化", pos=LEFT*3+DOWN*3)
        self.play(FadeOut(arr))
        self.wait(1)
        self.play(Indicate(code.code[8:13], color=RED))
        self.info_update("计算个体适应度，并选取适应度高的个体作为父代", pos=LEFT*3+DOWN*3)
        self.wait(2)
        self.play(Indicate(code.code[15:23], color=RED))
        self.info_update("每次轮盘赌选择两个父代个体，并交叉、变异生成两个子代个体",
                         pos=LEFT*3+DOWN*3)
        self.wait(2.3)
        self.remove(*self.mobjects)

class Test(Scene):
    def construct(self):
        genes_list = []
        n = 100
        size = 20
        for i in range(n):
            genes_pth = os.path.join("../", "genes", "all", str(i))
            with open(genes_pth, "r") as f:
                genes = np.array(list(map(float, f.read().split())))
                genes_list.append(genes)
 
        self.matrix = get_matrix(size, 0.3, stroke_width=1.5, color=GRAY_C)
        self.add(self.matrix)
        game = AiGame(genes_list=genes_list)
        for snake in game.snakes:
            snake.color = random_color()

        board = [(x, y) for x in range(game.X) for y in range(game.Y)]
        alive_snakes_set = set(game.snakes)
        steps = 0

        set_food = []
        set_head = []
        set_body = []
        set_color = []
        while alive_snakes_set:
            steps += 1
            for snake in alive_snakes_set:
                has_eat = snake.move(game.food)
                if has_eat:
                    game.food = game.rand.choice(board)
                if snake.score > game.best_score:
                    game.best_score = snake.score
            alive_snakes = [snake for snake in alive_snakes_set if not snake.dead]
            alive_snakes_set = set(alive_snakes)

            set_food.append(game.food)
            bodys = []
            heads = []
            colors = []
            for snake in alive_snakes:
                tmp_body = []
                for body in snake.body:
                    tmp_body.append(body)
                bodys.append(tmp_body)
                heads.append(snake.body[0])
                colors.append(snake.color)
            set_head.append(heads)
            set_body.append(bodys)
            set_color.append(colors)

        print(steps)
        nums = 1500
        shows = [[i for i in range(nums)]]
        #shows.append([i for i in range(steps//2-2*nums, steps//2)])
        #shows.append([i for i in range(steps-nums, steps-20)])
        for show in shows:
            for step in show:
                for i in range(size):
                    for j in range(size):
                        self.matrix[i][j].set_fill(BLACK, opacity=1)
                food = set_food[step]
                heads = set_head[step]
                bodys = set_body[step]
                colors = set_color[step]
                self.matrix[food[1]][food[0]].set_fill(PURE_RED, opacity=1)
                # for head in heads:
                    # self.matrix[head[1]][head[0]].set_fill(WHITE, opacity=1)
                for i in range(len(bodys)):
                    for body in bodys[i]:
                        self.matrix[body[1]][body[0]].set_fill(colors[i], opacity=1)
                self.wait(0.04)
                # if step == show[0] and show != shows[0]:
                    # self.play(FadeIn(self.matrix), run_time=1)
            # if show != shows[-1]:
                # self.play(FadeOut(self.matrix), run_time=1)

        self.play(FadeOut(self.matrix))
        self.wait()

class Test2(Scene):
    def construct(self):
        #ax = Axes(x_range=(-2, 2), y_range=(-5, 0))
        color = TEAL
        face = ParametricFunction(self.func, t_range=(-PI, PI),
                                  fill_color=color, fill_opacity=1,
                                  color=color, stroke_width=0.1)
        face.shift(UP * 3)

        start=PI/4
        end=PI/2
        eye = Arc(radius=0.15, start_angle=start, angle=end, stroke_width=2, color=BLACK)
        eyes = VGroup(eye.copy(), eye.copy()).arrange(RIGHT, buff=0.7)
        eyes.move_to(face)

        end = -PI/2 + 5 * DEGREES
        start = -30 * DEGREES
        mouth = Arc(radius=1, start_angle=start, angle=end, stroke_width=2.5, color=BLACK)
        mouth.move_to(face).shift(DOWN * 1.2)

        water = VGroup(face, mouth, eyes).scale(0.5)

        self.add(water)


    def func(self, t):
        a = 1
        b = 2.3
        x = a * (1 - math.sin(t)) * math.cos(t)
        y = b * (math.sin(t) - 1)
        return [x, y, 0]

