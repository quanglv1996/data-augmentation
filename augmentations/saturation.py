import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import random

class AdjustSaturation:
    def __init__(self, saturation_min=0.8, saturation_max=1.2):
        assert 0.5 <= saturation_min <= saturation_max <= 2.0, "Ngưỡng saturation nên nằm trong [0.5, 2.0]"
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max

    def transform(self, img, bboxes):
        saturation_factor = self.saturation_min if self.saturation_min == self.saturation_max \
            else random.uniform(self.saturation_min, self.saturation_max)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = F.adjust_saturation(img, saturation_factor)
            img = np.asarray(img)[..., ::-1]
        else:
            img = F.adjust_saturation(img, saturation_factor)

        return img, bboxes
