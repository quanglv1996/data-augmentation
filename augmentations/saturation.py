import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import random

class AdjustSaturation(object):
    def __init__(self, saturation_min=1.0, saturation_max=1.5):
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max

    def transform(self, img, bboxes):
        # Chọn giá trị saturation factor ngẫu nhiên trong khoảng min-max
        if self.saturation_min == self.saturation_max:
            saturation_factor = self.saturation_min
        else:
            saturation_factor = random.uniform(self.saturation_min, self.saturation_max)

        # Convert the image to a PIL Image object if it is a numpy array
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Adjust the saturation of the image
        img = F.adjust_saturation(img, saturation_factor)

        # Convert the Image object back to a numpy array and reverse the color channels
        img = np.array(img)[:, :, ::-1].copy()

        # Return the image with adjusted saturation and the bounding boxes
        return img, bboxes
