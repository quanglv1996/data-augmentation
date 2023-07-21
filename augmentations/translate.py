import numpy as np
from utils.utils import clip_box

class Translate(object):
    def __init__(self, translate_x=0.2, translate_y=0.2, diff=False):
        """
        Initialize the Translate augmentation class.

        Args:
            translate_x (float): Translation factor for the x-axis. Should be between 0 and 1.
            translate_y (float): Translation factor for the y-axis. Should be between 0 and 1.
            diff (bool): If True, use different translation factors for x and y axes. Default is False.
        """
        self.translate_x = translate_x
        self.translate_y = translate_y

        # Check that the provided translation factors are within valid ranges
        assert 0 < self.translate_x < 1, "Translate factor for x-axis should be between 0 and 1"
        assert 0 < self.translate_y < 1, "Translate factor for y-axis should be between 0 and 1"

    def __call__(self, img, bboxes):
        """
        Apply translation to the input image and its bounding boxes.

        Args:
            img (numpy array): The input image as a numpy array.
            bboxes (numpy array): An array of bounding boxes in the format [xmin, ymin, xmax, ymax, class_id].

        Returns:
            numpy array: Translated image.
            numpy array: Updated bounding boxes after translation.
        """
        img_shape = img.shape

        # Extract the translation factors for x and y axes
        translate_factor_x = self.translate_x
        translate_factor_y = self.translate_y

        canvas = np.zeros(img_shape, dtype=np.uint8)

        # Calculate the translation distance for both x and y axes
        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        # Determine the original coordinates of the translated region
        orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1], corner_x + img.shape[1])]

        # Copy the translated region to the canvas
        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
        img = canvas

        # Update the bounding boxes' coordinates to match the translation
        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

        # Clip the bounding boxes to ensure they remain within the image boundaries
        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

        return img, bboxes