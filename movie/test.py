from manim import *
from manim_rubikscube import *

class Main(ThreeDScene):
    def construct(self):
         cube = RubiksCube().scale(0.6)
         self.add(cube)

         self.move_camera(phi=50*DEGREES, theta=160*DEGREES)
         self.renderer.camera.frame_center = cube.get_center()

         self.wait(4)
