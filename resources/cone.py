import pybullet as p
import os


class Cone:
    def __init__(self, client, base, color="yellow"):
        f_name = os.path.join(
            os.path.dirname(__file__), f"models/cone_{color}/model.sdf"
        )
        # self.goal = client.loadSDF(f_name, [base[0], base[1], 0])
        self.goal = client.loadSDF(f_name)
