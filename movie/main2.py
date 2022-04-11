from manim import *

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
        code.shift(LEFT * 2 + UP * 2)
        code.width=6
        self.add(code)
        self.wait()
