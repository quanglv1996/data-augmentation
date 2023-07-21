import random
import numpy as np

class RandomHSV(object):
    def __init__(self, hue=None, saturation=None, brightness=None):
        """
        Initialize the RandomHSV data augmentation object.

        Args:
            hue (tuple or None): Range of hue adjustment. If None, hue will not be adjusted.
            saturation (tuple or None): Range of saturation adjustment. If None, saturation will not be adjusted.
            brightness (tuple or None): Range of brightness adjustment. If None, brightness will not be adjusted.
        """
        # Initialize the object with optional hue, saturation, and brightness ranges
        self.hue = hue if hue is not None else 0
        self.saturation = saturation if saturation is not None else 0
        self.brightness = brightness if brightness is not None else 0

        # Ensure the ranges are in tuple format
        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)
        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)
        if type(self.brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)

    def __call__(self, img, bboxes):
        """
        Apply random hue, saturation, and brightness adjustments to the input image.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): An array of bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The augmented image with random hue, saturation, and brightness adjustments.
            numpy.ndarray: The original bounding boxes (no adjustment is made on the bounding boxes).
        """
        # Randomly sample hue, saturation, and brightness from their ranges
        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)

        # Convert the image to integer type for numerical manipulation
        img = img.astype(int)

        # Adjust the hue, saturation, and brightness of the image
        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1, 1, 3))

        # Clip the pixel values to the valid range
        img = np.clip(img, 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        # Convert the image back to unsigned 8-bit integer type
        img = img.astype(np.uint8)

        # Return the augmented image and original bounding boxes (no adjustment on bounding boxes)
        return img, bboxes