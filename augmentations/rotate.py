import cv2
import numpy as np
import random
from utils.utils import get_corners, rotate_box, rotate_im, get_enclosing_box, clip_box

class Rotate:
    def __init__(self, angle_min=-10, angle_max=10):
        assert angle_min <= angle_max, "angle_min phải <= angle_max"
        self.angle_min = angle_min
        self.angle_max = angle_max

    def transform(self, img, bboxes):
        angle = self.angle_min if self.angle_min == self.angle_max else random.uniform(self.angle_min, self.angle_max)
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:])) if bboxes.shape[1] > 4 else corners

        img_rot = rotate_im(img, angle)
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        new_bbox = get_enclosing_box(corners)
        scale_x, scale_y = img_rot.shape[1] / w, img_rot.shape[0] / h
        img_rot = cv2.resize(img_rot, (w, h))
        new_bbox[:, :4] /= [scale_x, scale_y, scale_x, scale_y]

        bboxes_out = clip_box(new_bbox, [0, 0, w, h], 0.25)
        return img_rot, bboxes_out
