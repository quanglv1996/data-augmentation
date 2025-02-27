import numpy as np
import random
from utils.utils import clip_box

class Translate(object):
    def __init__(self, translate_min=0.1, translate_max=0.2, diff=False):
        self.translate_min = translate_min
        self.translate_max = translate_max
        self.diff = diff

        # Kiểm tra giá trị hợp lệs
        assert 0 < self.translate_min < self.translate_max < 1, "Translate factors phải nằm trong khoảng (0,1)"

    def transform(self, img, bboxes):
        img_shape = img.shape

        translate_factor_x = random.uniform(self.translate_min, self.translate_max)
        translate_factor_y = random.uniform(self.translate_min, self.translate_max) if self.diff else translate_factor_x

        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        canvas = np.zeros(img_shape, dtype=np.uint8)

        y1, y2 = max(0, corner_y), min(img_shape[0], img_shape[0] + corner_y)
        x1, x2 = max(0, corner_x), min(img_shape[1], img_shape[1] + corner_x)

        mask = img[max(-corner_y, 0):y2-corner_y, max(-corner_x, 0):x2-corner_x, :]
        canvas[y1:y2, x1:x2, :] = mask
        img = canvas

        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

        return img, bboxes
