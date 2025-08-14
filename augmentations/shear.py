import numpy as np
import cv2
import random
from augmentations.horizontal_flip import HorizontalFlip

class Shear:
    def __init__(self, shear_min=-0.2, shear_max=0.2):
        assert -0.5 <= shear_min <= shear_max <= 0.5, "shear nên nằm trong [-0.5, 0.5]"
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.hori_flip = HorizontalFlip()

    def transform(self, img, bboxes):
        shear_factor = random.uniform(self.shear_min, self.shear_max)
        img_out, bboxes_out = img, np.asarray(bboxes, dtype=np.float32)

        if shear_factor < 0:
            img_out, bboxes_out = self.hori_flip.transform(img_out, bboxes_out)
            shear_factor = abs(shear_factor)

        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        new_width = int(img_out.shape[1] + abs(shear_factor * img_out.shape[0]))
        img_sheared = cv2.warpAffine(img_out, M, (new_width, img_out.shape[0]))

        bboxes_out[:, [0, 2]] += bboxes_out[:, [1, 3]] * shear_factor

        if shear_factor < 0:
            img_sheared, bboxes_out = self.hori_flip.transform(img_sheared, bboxes_out)

        return img_sheared, bboxes_out
