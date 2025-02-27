import random
import numpy as np

class RandomHSV(object):
    def __init__(self, hue=None, saturation=None, brightness=None):
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

    def transform(self, img, bboxes):
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