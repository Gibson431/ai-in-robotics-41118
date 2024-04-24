import pybullet as p
import os


class Cone:
    def __init__(self, client, base, color):
        f_name = os.path.join(
            os.path.dirname(__file__), f"models/cone_{color}/cone_{color}.urdf"
        )
        self.goal = client.loadURDF(f_name, [base[0], base[1], 0])
        # self.goal = client.load(f_name)
