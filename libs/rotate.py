import cv2
import numpy as np
from libs.utils import get_corners, rotate_box, rotate_im, get_enclosing_box, clip_box

class Rotate(object):
    def __init__(self, angle):
        """
        Initialize the Rotate data augmentation object.

        Args:
            angle (float): The specified rotation angle in degrees for rotating the image and bounding boxes.
        """
        self.angle = angle

    def __call__(self, img, bboxes):
        """
        Rotate the image and adjust the bounding box coordinates accordingly.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): Bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The rotated image.
            numpy.ndarray: The adjusted bounding boxes.
        """
        # Get the rotation angle from the object's attribute
        angle = self.angle

        # Get the width and height of the image
        w, h = img.shape[1], img.shape[0]

        # Calculate the center coordinates of the image
        cx, cy = w // 2, h // 2

        # Get the corners of the bounding boxes and add the class labels
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))

        # Rotate the image and the bounding boxes
        img = rotate_im(img, angle)
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        # Get the new bounding box that encloses the rotated boxes
        new_bbox = get_enclosing_box(corners)

        # Calculate the scale factors to resize the image back to its original size
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        # Resize the image back to its original size
        img = cv2.resize(img, (w, h))

        # Scale the bounding boxes back to their original size
        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        # Update the bounding boxes with the resized ones
        bboxes = new_bbox

        # Clip the bounding boxes to stay within the image boundaries
        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

        # Return the rotated image and the adjusted bounding boxes
        return img, bboxes