import cv2
import numpy as np
import random
from utils.utils import get_corners, rotate_box, rotate_im, get_enclosing_box, clip_box

class Rotate(object):
    def __init__(self, angle_min=-10, angle_max=10):
        self.angle_min = angle_min
        self.angle_max = angle_max

    def transform(self, img, bboxes):
        # Chọn một góc ngẫu nhiên trong khoảng min-max
        if self.angle_min == self.angle_max:
            angle = self.angle_min
        else:
            angle = random.uniform(self.angle_min, self.angle_max)

        # Kích thước ảnh gốc
        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        # Lấy góc bounding box
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))

        # Xoay ảnh và bounding box
        img = rotate_im(img, angle)
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        # Tạo bounding box mới bao quanh vùng xoay
        new_bbox = get_enclosing_box(corners)

        # Tính tỉ lệ scale lại ảnh về kích thước gốc
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        # Resize ảnh về kích thước ban đầu
        img = cv2.resize(img, (w, h))

        # Scale lại tọa độ bounding box
        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        # Cập nhật bounding box và cắt các phần nằm ngoài ảnh
        bboxes = clip_box(new_bbox, [0, 0, w, h], 0.25)

        return img, bboxes
