import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random

class AdjustBrightness:
    def __init__(self, brightness_min=1.0, brightness_max=1.5):
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max

    def transform(self, img, bboxes):
        if self.brightness_min == self.brightness_max:
            brightness_factor = self.brightness_min
        else:
            brightness_factor = random.uniform(self.brightness_min, self.brightness_max)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img[..., ::-1])

        img = F.adjust_brightness(img, brightness_factor)
        img = np.asarray(img)[..., ::-1]

        return img, bboxes
