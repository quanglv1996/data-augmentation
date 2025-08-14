import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random

class AdjustBrightness:
    def __init__(self, brightness_min=0.8, brightness_max=1.2):
        assert 0.5 <= brightness_min <= brightness_max <= 2.0, "Ngưỡng sáng nên nằm trong [0.5, 2.0]"
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max

    def transform(self, img, bboxes):
        brightness_factor = self.brightness_min if self.brightness_min == self.brightness_max \
            else random.uniform(self.brightness_min, self.brightness_max)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img[..., ::-1])
            img = F.adjust_brightness(img, brightness_factor)
            img = np.asarray(img)[..., ::-1]
        else:
            img = F.adjust_brightness(img, brightness_factor)

        return img, bboxes