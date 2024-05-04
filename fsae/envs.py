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
from queue import Queue

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class RandomTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "fp_camera", "tp_camera"]}

    def __init__(self, render_mode=None, seed=None):

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

        if seed:
            self._track_generator = TrackGenerator(config={"seed": seed})
        else:
            self._track_generator = TrackGenerator()
        self._path = None
        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 20
        self.car = None
        self.done = False
        self.prev_dist = None
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
            # goalpos, goalorn = self._p.getBasePositionAndOrientation(
            #     self.goal_object.goal
            # )
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        dist = self.projectAndFindDistance(
            self.centres.queue[0],
            self.centres.queue[1],
            self.car.get_observation()[0:2],
        )

        reward = dist - self.prev_dist
        self.prev_dist = dist

        length_of_segment = self.calcDistanceBetweenPoints(
            self.centres.queue[0],
            self.centres.queue[1],
        )
        if dist > length_of_segment:
            self.centres.put(self.centres.get())
            self.prev_dist = self.projectAndFindDistance(
                self.centres.queue[0],
                self.centres.queue[1],
                self.car.get_observation()[0:2],
            )

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

        self.done = False
        # self.goal_object = Goal(self._p, self.goal)
        # Get observation to return
        carpos = self.car.get_observation()

        margin = (
            self._track_generator.config["track_width"] / 2
            + self._track_generator.config["margin"]
        )
        while True:
            self._path = TrackGenerator.generate_path_w_params(
                rng=self._track_generator.rng,
                n_points=self._track_generator.config["resolution"],
                min_corner_radius=self._track_generator.config["min_corner_radius"],
                max_frequency=self._track_generator.config["max_frequency"],
                amplitude=self._track_generator.config["amplitude"],
            )
            if not (
                self._track_generator.config["check_self_intersection"]
                and TrackGenerator.self_intersects(*self._path[:2], margin)
            ):
                break
        self._path = TrackGenerator.pick_starting_point(
            *self._path,
            starting_straight_length=self._track_generator.config[
                "starting_straight_length"
            ],
            downsample=self._track_generator.config["starting_straight_downsample"],
        )

        (start_cones, l_cones, r_cones, centres) = TrackGenerator.place_cones(
            *self._path,
            self._track_generator.config["min_corner_radius"],
            min_cone_spacing=self._track_generator.config["min_cone_spacing"],
            max_cone_spacing=self._track_generator.config["max_cone_spacing"],
            track_width=self._track_generator.config["track_width"],
            cone_spacing_bias=self._track_generator.config["cone_spacing_bias"],
            start_offset=self._track_generator.config["starting_straight_length"],
            starting_cone_spacing=self._track_generator.config["starting_cone_spacing"],
        )

        TrackGenerator.write_to_csv(
            "~test.csv",
            start_cones,
            l_cones,
            r_cones,
            True,
        )
        # print(self._path)

        self.cones = []
        for c in l_cones:
            self.cones.append(Cone(self._p, (c.real, c.imag), color="blue"))
        for c in r_cones:
            self.cones.append(Cone(self._p, (c.real, c.imag), color="yellow"))
        for c in start_cones:
            self.cones.append(Cone(self._p, (c.real, c.imag), color="orange"))

        self.centres = Queue()
        for i, p in enumerate(centres):
            # if i % 50 == 0:
            self.centres.put((p.real, p.imag))

        # Visualise the centre pos
        # self.centre_obj = []
        # for c in self.centres.queue:
        #     self.centre_obj.append(Goal(self._p, c))

        self.prev_dist = 0
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
            # frame = np.array(px)
            # frame = frame[:, :, :3]
            return px
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
            # frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        # self._observation = []  #self._racecar.getObservation()
        # carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        # goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        # invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        # goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(
        #     invCarPos, invCarOrn, goalpos, goalorn
        # )
        # observation = [goalPosInCar[0], goalPosInCar[1]]
        # return observation

        carpos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        return [carpos[0], carpos[1]]

    @staticmethod
    def calcDistanceBetweenPoints(point1, point2):
        dist = math.sqrt(((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))
        return dist

    def _termination(self):
        dist = self.normal_distance(
            self.centres.queue[0],
            self.centres.queue[1],
            self.car.get_observation()[0:2],
        )
        return dist > 1.5

    def close(self):
        self._p.disconnect()

    @staticmethod
    def projectAndFindDistance(point1, point2, p):
        """
        Project point p onto the line defined by point1 and point2, and find the distance from point1
        to the projected point along the line segment.

        :param point1: A tuple (x1, y1) representing the first point.
        :param point2: A tuple (x2, y2) representing the second point.
        :param p: A tuple (px, py) representing the point to be projected.
        :return: The distance from point1 to the projected point along the line segment.
        """
        x1, y1 = point1
        x2, y2 = point2
        px, py = p

        # Vector from point1 to point2
        dx, dy = x2 - x1, y2 - y1
        # Vector from point1 to p
        dx1, dy1 = px - x1, py - y1

        # Calculate the dot product of vectors (point1 to point2) and (point1 to p)
        dot_product = dx * dx1 + dy * dy1
        # Length squared of the vector (point1 to point2)
        length_squared = dx * dx + dy * dy

        if length_squared == 0:
            # The points point1 and point2 are the same. The distance is just the distance to point1
            return 0

        projected_distance = abs(dot_product) ** 0.5
        projected_distance = (
            projected_distance if dot_product >= 0 else -projected_distance
        )

        return projected_distance

    @staticmethod
    def normal_distance(point1, point2, p):
        """
        Calculate the perpendicular distance from point p to the line defined by point1 and point2.

        :param point1: A tuple (x1, y1) representing the first point.
        :param point2: A tuple (x2, y2) representing the second point.
        :param p: A tuple (px, py) representing the point from which the distance is to be calculated.
        :return: The perpendicular distance from p to the line segment defined by point1 and point2.
        """
        x1, y1 = point1
        x2, y2 = point2
        px, py = p

        # Calculate the numerator (area of the triangle formed by the points multiplied by 2)
        numerator = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))

        # Calculate the denominator (length of the base of the triangle)
        denominator = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        if denominator == 0:
            # Point1 and Point2 are the same point
            return float(
                0
            )  # or some large number, as the concept of a line doesn't exist here

        # Distance is numerator divided by denominator
        distance = numerator / denominator

        return distance
