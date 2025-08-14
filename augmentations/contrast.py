import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import random

class AdjustContrast:
    def __init__(self, contrast_min=0.8, contrast_max=1.2):
        assert 0.5 <= contrast_min <= contrast_max <= 2.0, "Ngưỡng contrast nên nằm trong [0.5, 2.0]"
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max

    def transform(self, img, bboxes):
        contrast_factor = self.contrast_min if self.contrast_min == self.contrast_max \
            else random.uniform(self.contrast_min, self.contrast_max)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = F.adjust_contrast(img, contrast_factor)
            img = np.asarray(img)[..., ::-1]
        else:
            img = F.adjust_contrast(img, contrast_factor)

        return img, bboxes