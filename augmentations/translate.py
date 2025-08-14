import numpy as np
import random
from utils.utils import clip_box

class Translate:
    def __init__(self, translate_min=0.1, translate_max=0.2, diff=False):
        assert 0 < translate_min < translate_max < 1, "Translate factors phải nằm trong khoảng (0,1)"
        self.translate_min = translate_min
        self.translate_max = translate_max
        self.diff = diff

    def transform(self, img, bboxes):
        img_shape = img.shape
        translate_factor_x = random.uniform(self.translate_min, self.translate_max)
        translate_factor_y = random.uniform(self.translate_min, self.translate_max) if self.diff else translate_factor_x

        shift_x = int(translate_factor_x * img_shape[1])
        shift_y = int(translate_factor_y * img_shape[0])

        canvas = np.zeros(img_shape, dtype=np.uint8)
        y1, y2 = max(0, shift_y), min(img_shape[0], img_shape[0] + shift_y)
        x1, x2 = max(0, shift_x), min(img_shape[1], img_shape[1] + shift_x)
        mask = img[0:img_shape[0]-shift_y, 0:img_shape[1]-shift_x, :]
        canvas[y1:y2, x1:x2, :] = mask
        img_out = canvas

        bboxes = np.asarray(bboxes, dtype=np.float32)
        bboxes[:, :4] += [shift_x, shift_y, shift_x, shift_y]
        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

        return img_out, bboxes
