import cv2
import numpy as np
import random
from utils.utils import clip_box

class Scale:
    def __init__(self, scale_x_min=0.8, scale_x_max=1.2, scale_y_min=0.8, scale_y_max=1.2):
        assert 0.5 <= scale_x_min <= scale_x_max <= 2.0, "scale_x nên nằm trong [0.5, 2.0]"
        assert 0.5 <= scale_y_min <= scale_y_max <= 2.0, "scale_y nên nằm trong [0.5, 2.0]"
        self.scale_x_min = scale_x_min
        self.scale_x_max = scale_x_max
        self.scale_y_min = scale_y_min
        self.scale_y_max = scale_y_max

    def transform(self, img, bboxes):
        img_shape = img.shape
        scale_x = self.scale_x_min if self.scale_x_min == self.scale_x_max else random.uniform(self.scale_x_min, self.scale_x_max)
        scale_y = self.scale_y_min if self.scale_y_min == self.scale_y_max else random.uniform(self.scale_y_min, self.scale_y_max)

        img_scaled = cv2.resize(img, None, fx=scale_x, fy=scale_y)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        bboxes[:, :4] *= [scale_x, scale_y, scale_x, scale_y]

        # Nếu ảnh đã scale nhỏ hơn hoặc bằng ảnh gốc, chèn vào canvas
        if img_scaled.shape[0] <= img_shape[0] and img_scaled.shape[1] <= img_shape[1]:
            canvas = np.zeros(img_shape, dtype=np.uint8)
            y_lim = img_scaled.shape[0]
            x_lim = img_scaled.shape[1]
            canvas[:y_lim, :x_lim, :] = img_scaled
            img_out = canvas
        else:
            # Nếu ảnh lớn hơn, crop về đúng shape gốc
            img_out = img_scaled[:img_shape[0], :img_shape[1], :]

        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.2)
        return img_out, bboxes
