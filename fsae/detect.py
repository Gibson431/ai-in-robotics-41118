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

    def set_intrinsics(self, fov, aspect, image_width, image_height, near, far):
        # Calculate focal length
        focal_length_x = image_width/(2 * aspect * np.tan(np.radians(fov / 2)))
        focal_length_y = image_height/(2 * np.tan(np.radians(fov/ 2)))

        # Calculate principal point
        principal_point_x = image_width / 2
        principal_point_y = image_height / 2

        # Return camera intrinsics
        self.fx = focal_length_x
        self.fy = focal_length_y
        self.cx = principal_point_x
        self.cy = principal_point_y


            # Compute the near and far plane distances from the projection matrix
        self.near_plane = near
        self.far_plane = far

    def detect(self,image):
        return self.model(image, size=1280)

    
    # def reproject_object_to_3d(self, bbox, depth_im):
    #     """
    #     Reprojects an object into 3D coordinates from the camera frame.

    #     Args:
    #         bbox (list or tuple): Bounding box coordinates in the format [x, y, w, h].

    #     Returns:
    #         np.ndarray: 3D coordinates of the object in the camera frame.
    #     """
    #     # Extract bounding box coordinates
    #     x_min, y_min, x_max, y_max = bbox

    #     # Calculate the center of the bounding box
    #     centre_x = x_min + (x_max - x_min)/2
    #     centre_y = y_min + (y_max - y_min)/2

    #     # Estimate depth using the known object width
    #     # print(depth_im)
    #     depth = depth_im[int(centre_y), int(centre_x)]
    #     near=0.01
    #     far=100
    #     depth = far * near / (far - (far - near) * depth)

    #     # depth = (self.width * self.fx) / (x_max - x_min)

    #     # Reproject the center of the bounding box to 3D
    #     X = (centre_x - self.cx) * depth / self.fx
    #     Y = (centre_y - self.cy) * depth / self.fy
    #     Z = depth

    #     # Adjust Y coordinate based on the known object height
    #     # object_height_pixels = (self.height * self.fy) / depth
    #     # Y -= ((y_max-y_min) / 2 - object_height_pixels / 2)

    #     return X, Y, Z
    

    def reproject_object_to_3d(self, bbox, depth_image):
        """
        Reprojects an object into 3D coordinates from the camera frame using depth values from a depth image
        and considering the perspective projection model.

        Args:
            bbox (list or tuple): Bounding box coordinates in the format [x, y, w, h].
            depth_image (np.ndarray): Depth image where each pixel value represents the linear depth.
            projection_matrix (np.ndarray): The camera's 4x4 projection matrix.

        Returns:
            np.ndarray: 3D coordinates of the object in the camera frame.
        """
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Calculate the center of the bounding box
        center_x = x_min + (x_max - x_min) / 2
        center_y = y_min + (y_max - y_min) / 2

        # Get the linear depth value at the center of the bounding box
        linear_depth = depth_image[int(center_y), int(center_x)]

        # Convert linear depth to depth in meters
        depth = self.far_plane * self.near_plane / (self.far_plane - (self.far_plane - self.near_plane) * linear_depth)

        # Compute the normalized device coordinates
        x_ndc = (center_x - self.cx) / self.fx
        y_ndc = (center_y - self.cy) / self.fy

        # Reproject the center of the bounding box to 3D
        X = x_ndc * depth
        Y = y_ndc * depth
        Z = depth

        # Adjust Y coordinate based on the known object height
        object_height_pixels = (self.height * self.fy) / depth
        Y -= ((y_max - y_min) / 2 - object_height_pixels / 2)

        return X, Y, Z
        

    