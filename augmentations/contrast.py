import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import random

class AdjustContrast(object):
    def __init__(self, contrast_min=1.0, contrast_max=1.5):
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max

    def transform(self, img, bboxes):
        if self.contrast_min == self.contrast_max:
            contrast_factor = self.contrast_min
        else:
            contrast_factor = random.uniform(self.contrast_min, self.contrast_max)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img = F.adjust_contrast(img, contrast_factor)

        img = np.array(img)[:, :, ::-1].copy()

        return img, bboxes
