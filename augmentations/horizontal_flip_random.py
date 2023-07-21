import numpy as np
import random

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        """
        Initialize the object with the probability of flipping.

        Args:
            p (float): The probability of horizontally flipping the image.
        """
        self.p = p

    def __call__(self, img, bboxes):
        """
        Apply random horizontal flip to the input image.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): An array of bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The horizontally flipped image.
            numpy.ndarray: The bounding boxes adjusted after the horizontal flip.
        """
        # Get the center of the image
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))
        
        # Check if flipping should be applied based on the probability
        if random.random() < self.p:
            # Flip the image horizontally
            img = img[:, ::-1, :]
            
            # Adjust the bounding box coordinates after flipping
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
            box_w = np.abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

        # Return the flipped image and adjusted bounding boxes
        return img, bboxes