import torch
import numpy as np

class object_detection():
    """class to handle object_detection and reprojection for known object

        Args: 
            weights (string): Path to model weights
            object_size (tuple): width and height in metres of object to detect
            intrinsics (matrix): fx, fy; cx, cy
    """

    def __init__(self, weights, object_size, intrinsics):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights) 
        self.model = self.model.to(self.device)
        self.width = object_size[0]
        self.height = object_size[1]
        self.fx = intrinsics[0,0]
        self.fy = intrinsics[0,1]
        self.cx = intrinsics[1,0]
        self.cy = intrinsics[1,1]

    def detect(self,image):
        """
        Runs the object detector to find all cones within the image. 

        Args:
            image (np.ndarray): 3D Numpy array containing rgb values for an image.

        Returns:
            yolov5 result: yolov5 pip package result object.
        """
        image = torch.from_numpy(image).to(self.device)
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
    
        

    