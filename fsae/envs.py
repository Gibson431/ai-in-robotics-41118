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
from collections import deque
from fsae.detect import object_detection
import cv2

def draw_bounding_boxes(image_array, bboxes, colors=None):
    """
    Draw bounding boxes on an RGB image array.

    Args:
        image_array (numpy.ndarray): The input RGB image as a NumPy array.
        bboxes (list): A list of bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        colors (list, optional): A list of colors for the bounding boxes in BGR format. If None, random colors will be used.

    Returns:
        numpy.ndarray: The image array with bounding boxes drawn.
    """
    
    # Convert the RGB image array to BGR format (OpenCV convention)
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # If colors is None, generate random colors for each bounding box
    if colors is None:
        colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(len(bboxes))]
    
    # Draw the bounding boxes
    for bbox, color in zip(bboxes, colors):
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
    
    return image

class RandomTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "fp_camera", "tp_camera", "detections"]}
    RENDER_HEIGHT = 720
    RENDER_WIDTH = 1280

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
        if self.render_mode == 'detections':
            self.detects_ = True
        else: 
            self.detects_ = False

        if render_mode == 'detections':
            self._renders = True
            self._p = bc.BulletClient()#connection_mode=p.GUI)
        elif render_mode:
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
        if self.detects_:
            self.detector = object_detection('fsae/cones.pt',(0.23, 0.31))
            self.detection_window = cv2.namedWindow('Detections')
    
    def step(self, action):
        """
        Steps the simulation, applying the action.

        :param action: A tuple (t, s) representing the throttle(-1,1) and steering(-0.6,0.6).
        :return: observation(closest 4 cones in FOV), reward (distance travelled in step), done flag, info (empty dict)
        """
        # Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                if self.detects_:
                    rgb, depth = self.render()
                    results = self.detector.detect(rgb)
                    cv2.imshow(
                        "Detections",
                        cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR),
                    )
                    cv2.waitKey(1)
                pos, ori = self._p.getBasePositionAndOrientation(self.car.car)
                ori = self._p.getEulerFromQuaternion(ori)
                self._p.resetDebugVisualizerCamera(
                    cameraDistance=10.0,
                    cameraYaw=math.degrees(ori[2]) - 90,
                    cameraPitch=-45,
                    cameraTargetPosition=pos,
                )
                # time.sleep(self._timeStep)

            carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
            # goalpos, goalorn = self._p.getBasePositionAndOrientation(
            #     self.goal_object.goal
            # )
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        if self._renders and self.detects_:
            box = []
            for i in range(4):
                boxes = results.pred[0][i, :4].cpu()
                box.append(boxes)
                print(self.detector.reproject_object_to_3d(boxes, depth))
                print(results.pred[0][i, 5])
            cv2.imshow('valid',draw_bounding_boxes(rgb, box))
            cv2.waitKey(0)

        dist = self.projectAndFindDistance(
            list(self.centres)[0],
            list(self.centres)[1],
            self.car.get_observation()[0:2],
        )

        if dist < self.prev_dist: #if the car sarts reversing
            self.done = True
            

        if (dist - self.prev_dist) < 0.0001: #if the car is stopped
            self.done = True
            

        reward = dist - self.prev_dist
        self.prev_dist = dist

        length_of_segment = self.calcDistanceBetweenPoints(
            list(self.centres)[0],
            list(self.centres)[1],
        )
        if dist > length_of_segment:
            self.centres.append(self.centres.popleft())
            self.prev_dist = self.projectAndFindDistance(
                list(self.centres)[0],
                list(self.centres)[1],
                self.car.get_observation()[0:2],
            )
        if dist < 0:
            self.centres.appendleft(self.centres.pop())
            self.prev_dist = self.projectAndFindDistance(
                list(self.centres)[0],
                list(self.centres)[1],
                self.car.get_observation()[0:2],
            )

        ob = car_ob
        visual_cones = np.asarray(
            self.getConesTransformedAndSorted(4), dtype=np.float32
        )
        return visual_cones, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
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

        if seed is not None:
            self._track_generator = TrackGenerator(config={"seed": seed})

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

        self.centres = deque()
        for i, p in enumerate(centres):
            # if i % 50 == 0:
            self.centres.append((p.real, p.imag))

        # Visualise the centre pos
        # self.centre_obj = []
        # for c in self.centres.queue:
        #     self.centre_obj.append(Goal(self._p, c))

        self.prev_dist = 0
        # car_ob = self.getExtendedObservation()
        visual_cones = self.getConesTransformedAndSorted(4)
        return visual_cones

    def render(self, mode=None):
        """
        Computes and returns the camera image based on the specified mode.

        This function supports two modes of camera views in a simulation environment:
        1. "fp_camera" - First person camera view from the perspective of a car.
        2. "tp_camera" - Third person camera view, providing an overhead perspective.
        3. "detector" 

        :param mode (str): Specifies the camera mode. It can be either "fp_camera" for first person
                    view or "tp_camera" for third person view. The default is "fp_camera".

        :return (tuple): This function returns a tuple (rgb, depth) where 'rgb' is an array
            representing the RGB image and 'depth' is an array representing the depth
            information from the camera perspective. If an unsupported mode is specified,
            it returns two empty lists, ([], []).

        Details:
            - "fp_camera": Calculates the projection and view matrices based on the car's
                           current position and orientation. It uses these matrices to render
                           the camera image showing what is directly in front of the car.
            - "tp_camera": Similar to "fp_camera", but sets the camera at a fixed distance from
                           the car, looking at it from an angle, providing a broader view of
                           the surroundings.
        """
        if mode == None:
            mode = self.render_mode
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=16/9, nearVal=0.01, farVal=100
            )
            pos, ori = [list(l) for l in self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.4

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # self.detector.set_intrinsics(proj_matrix.)
            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, rgb, depth, _) = self._p.getCameraImage(
                width=self.RENDER_WIDTH,
                height=self.RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            ) 
            return (rgb, depth)
        
        elif mode == "detections":
            # Base information
            car_id = self.car.get_ids()
            fov = 60
            aspect = 16/9
            near = 0.01
            far = 100
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=fov, aspect=aspect, nearVal=near, farVal=far
            )
            pos, ori = [list(l) for l in self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.4

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            self.detector.set_intrinsics(fov,aspect, self.RENDER_WIDTH, self.RENDER_HEIGHT, near, far)
            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, rgb, depth, _) = self._p.getCameraImage(
                width=self.RENDER_WIDTH,
                height=self.RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            return (rgb, depth)

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
                aspect=float(self.RENDER_WIDTH) / self.RENDER_HEIGHT,
                nearVal=0.1,
                farVal=100.0,
            )
            (_, _, rgb, depth, _) = self._p.getCameraImage(
                width=self.RENDER_WIDTH,
                height=self.RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            return rgb, depth
        else:
            return [], []

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

    # def getConeObservation(self):
    #     cones = self.getConesTransformedAndSorted(4)
    #     return cones

    @staticmethod
    def calcDistanceBetweenPoints(point1, point2):
        dist = math.sqrt(((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))
        return dist

    def _termination(self):
        dist = self.normal_distance(
            list(self.centres)[0],
            list(self.centres)[1],
            self.car.get_observation()[0:2],
        )

        if dist == float("inf"):
            return False  # this should never happen, but just incase

        return dist > 1.3

    def close(self):
        cv2.destroyAllWindows()
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

        # Point1 and Point2 are the same point
        if denominator == 0:
            return float("inf")

        # Distance is numerator divided by denominator
        distance = numerator / denominator

        return distance

    def getConesTransformedAndSorted(self, num_cones, detected_cones=None):
        l_cones = [c for c in self.cones if c.color == "blue"]
        r_cones = [c for c in self.cones if c.color == "yellow"]

        # Convert all cones to car reference frame
        l_cones = [(self.reframeToCar(c.cone)[:2, 3], c.color) for c in l_cones]
        r_cones = [(self.reframeToCar(c.cone)[:2, 3], c.color) for c in r_cones]

        # Filter out cones outside of 90deg FOV
        l_cones = [
            (p, c) for p, c in l_cones if (abs(p[0]) >= abs(p[1])) and (p[0] > 0)
        ]
        r_cones = [
            (p, c) for p, c in r_cones if (abs(p[0]) >= abs(p[1])) and (p[0] > 0)
        ]
        # if detected_cones: 
        #     for 
        # Calculate the magnitude of each coordinate and pair it with the corresponding tuple
        l_magnitudes = [(np.linalg.norm(xy), xy, name) for xy, name in l_cones]
        r_magnitudes = [(np.linalg.norm(xy), xy, name) for xy, name in r_cones]

        # Sort the list of tuples based on the calculated magnitude
        l_sorted_magnitudes = sorted(l_magnitudes, key=lambda x: x[0])
        r_sorted_magnitudes = sorted(r_magnitudes, key=lambda x: x[0])

        l_cones = [(xy, name) for _, xy, name in l_sorted_magnitudes][
            : int(num_cones / 2)
        ]
        r_cones = [(xy, name) for _, xy, name in r_sorted_magnitudes][
            : int(num_cones / 2)
        ]

        cones_mapped = []
        for c, _ in l_cones:
            cones_mapped.append(c)
        while len(cones_mapped) < num_cones / 2:
            cones_mapped.append([0, 0])
        for c, _ in r_cones:
            cones_mapped.append(c)
        while len(cones_mapped) < num_cones:
            cones_mapped.append([0, 0])

        cones_stacked = np.hstack([np.hstack(detection) for detection in cones_mapped])
        return cones_stacked

    @staticmethod
    def get_transformation_matrix(position, orientation):
        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # Create transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position
        return transformation_matrix

    def reframeToCar(self, target_object):
        # Get the world position and orientation of the car object
        car_pos, car_ori = p.getBasePositionAndOrientation(self.car.car)
        # Get the world position and orientation of the target object
        target_pos, target_ori = p.getBasePositionAndOrientation(target_object)

        # Compute the transformation matrices
        target_world_matrix = self.get_transformation_matrix(car_pos, car_ori)
        world_matrix = self.get_transformation_matrix(target_pos, target_ori)

        # Compute the inverse of the target's world transformation matrix
        target_world_matrix_inv = np.linalg.inv(target_world_matrix)

        # Compute the transformation from world object to the target object frame
        transformation_to_target_frame = np.dot(target_world_matrix_inv, world_matrix)

        return transformation_to_target_frame
