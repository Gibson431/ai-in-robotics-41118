import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from fsae.resources.car import Car
from fsae.resources.plane import Plane
from fsae.resources.goal import Goal
from fsae.resources.cone import Cone
import matplotlib.pyplot as plt
import time
from fsae.track_generator import TrackGenerator

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class RandomTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "fp_camera", "tp_camera"]}

    def __init__(self, render_mode=None):

        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -0.6], dtype=np.float32),
            high=np.array([1, 0.6], dtype=np.float32),
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40], dtype=np.float32),
            high=np.array([40, 40], dtype=np.float32),
        )
        self.np_random, _ = gym.utils.seeding.np_random()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_mode:
            self._p = bc.BulletClient(connection_mode=p.GUI)
            self._renders = True
        else:
            self._renders = False
            self._p = bc.BulletClient()

        self._track_generator = TrackGenerator()
        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 20
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0

    def step(self, action):
        # Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
            goalpos, goalorn = self._p.getBasePositionAndOrientation(
                self.goal_object.goal
            )
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
        # (car_ob[1] - self.goal[1]) ** 2))
        dist_to_goal = math.sqrt(
            ((carpos[0] - goalpos[0]) ** 2 + (carpos[1] - goalpos[1]) ** 2)
        )
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        reward = -dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        # Done by reaching goal
        if dist_to_goal < 1.5 and not self.reached_goal:
            # print("reached goal")
            self.done = True
            self.reached_goal = True
            reward += 50

        ob = car_ob
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0
        # Set the goal to a random target
        x = (
            self.np_random.uniform(5, 9)
            if self.np_random.integers(2)
            else self.np_random.uniform(-9, -5)
        )
        y = (
            self.np_random.uniform(5, 9)
            if self.np_random.integers(2)
            else self.np_random.uniform(-9, -5)
        )
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False
        self.goal_object = Goal(self._p, self.goal)
        # Get observation to return
        carpos = self.car.get_observation()
        (start_cones, l_cones, r_cones) = self._track_generator()
        self._track_generator.write_to_csv(
            "~test.csv",
            start_cones,
            l_cones,
            r_cones,
            True,
        )

        self.cones = []
        for c in l_cones:
            self.cones.append(Cone(self._p, (c.real, c.imag), color="blue"))
        for c in r_cones:
            self.cones.append(Cone(self._p, (c.real, c.imag), color="yellow"))
        # Visual element of the goal
        # self.cones = [
        #     Cone(self._p, (carpos[0], carpos[1] + 1), color="blue"),
        #     Cone(self._p, (carpos[0], carpos[1] - 1), color="yellow"),
        # ]

        self.prev_dist_to_goal = math.sqrt(
            ((carpos[0] - self.goal[0]) ** 2 + (carpos[1] - self.goal[1]) ** 2)
        )
        car_ob = self.getExtendedObservation()
        return np.array(car_ob, dtype=np.float32)

    def render(self, mode="human"):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=80, aspect=1, nearVal=0.01, farVal=100
            )
            pos, ori = [list(l) for l in self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.3

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=20.0,
                yaw=40.0,
                pitch=-35,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                nearVal=0.1,
                farVal=100.0,
            )
            (_, _, px, _, _) = self._p.getCameraImage(
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        # self._observation = []  #self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(
            invCarPos, invCarOrn, goalpos, goalorn
        )

        observation = [goalPosInCar[0], goalPosInCar[1]]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()
