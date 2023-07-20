import numpy as np
from libs.horizontal_flip import HorizontalFlip
import cv2

class Shear(object):
    def __init__(self, shear_factor=0.2):
        """
        Initialize the Shear data augmentation object.

        Args:
            shear_factor (float): The shear factor to apply to the image. Positive values shear the image towards the
                                  right, and negative values shear the image towards the left.
        """
        # Initialize the object with the shear factor
        self.shear_factor = shear_factor

    def __call__(self, img, bboxes):
        """
        Apply shear transformation to the input image and bounding boxes.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): Bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The sheared image.
            numpy.ndarray: The adjusted bounding boxes.
        """
        # Retrieve the shear factor
        shear_factor = self.shear_factor

        # Apply horizontal flip if the shear factor is negative
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        # Create an affine transformation matrix for shear
        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        # Calculate the new width after shearing
        nW = img.shape[1] + abs(shear_factor * img.shape[0])

        # Adjust bounding boxes based on shear factor
        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

        # Apply the shear transformation to the image
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        # Undo horizontal flip if the shear factor was negative
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        return img, bboxes