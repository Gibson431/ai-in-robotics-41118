import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import agents
import time
from fsae.envs import StaticTrackEnv
from pybullet_utils import bullet_client as bc
import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from resources.car import Car
from resources.plane import Plane
from resources.goal import Goal
from resources.cone import Cone
import matplotlib.pyplot as plt
import os
import time

p = bc.BulletClient(connection_mode=p.GUI)
# f_name = os.path.join(
#     os.path.dirname(__file__), f"resources/models/cone_yellow/cone_yellow.urdf"
# )
# self.goal = client.loadSDF(f_name, [base[0], base[1], 0])
# goal = p.loadURDF(f_name)

c = Cone(p, (0, 0), color="blue")

while True:
    time.sleep(1)
