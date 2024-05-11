import yolov5
import numpy as np

class object_detection():
    """class to handle object_detection and reprojection for known object

        Args: 
            weights (string): Path to model weights
            object_size (tuple): width and height in metres of object to detect
            intrinsics (matrix): fx, fy; cx, cy
    """

    def __init__(self, weights, object_size):
        self.model = yolov5.load(weights)
        # set model parameters
        self.model.conf = 0.5  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

        self.width = object_size[0]
        self.height = object_size[1]
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def set_intrinsics(self, horizontal_fov, vertical_fov, image_width, image_height):
        # Calculate focal length
        focal_length_x = (image_width / 2) / np.tan(horizontal_fov / 2)
        focal_length_y = (image_height / 2) / np.tan(vertical_fov / 2)

        # Calculate principal point
        principal_point_x = image_width / 2
        principal_point_y = image_height / 2

        # Return camera intrinsics
        self.fx = focal_length_x
        self.fy = focal_length_y
        self.cx = principal_point_x
        self.cy = principal_point_y
        print("set")

    def detect(self,image):
        return self.model(image)

    
    def reproject_object_to_3d(self, bbox):
        """
        Reprojects an object into 3D coordinates from the camera frame.

        Args:
            bbox (list or tuple): Bounding box coordinates in the format [x, y, w, h].

        Returns:
            np.ndarray: 3D coordinates of the object in the camera frame.
        """
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Calculate the center of the bounding box
        center_x = x_min + (x_max - x_min)/2
        center_y = y_min + (y_max - y_min)/2

        # Estimate depth using the known object width
        depth = (self.width * self.fx) / (x_max - x_min)

        # Reproject the center of the bounding box to 3D
        X = (center_x - self.cx) * depth / self.fx
        Y = (center_y - self.cy) * depth / self.fy
        Z = depth

        # Adjust Y coordinate based on the known object height
        object_height_pixels = (self.height * self.fy) / depth
        Y -= ((y_max-y_min) / 2 - object_height_pixels / 2)

        return np.array([X, Y, Z])
    
        

    