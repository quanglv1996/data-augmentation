import cv2
import numpy as np
import random
from utils.utils import clip_box

class Scale(object):
    def __init__(self, scale_x_min=0.8, scale_x_max=1.2, scale_y_min=0.8, scale_y_max=1.2):
        self.scale_x_min = scale_x_min
        self.scale_x_max = scale_x_max
        self.scale_y_min = scale_y_min
        self.scale_y_max = scale_y_max

    def transform(self, img, bboxes):
        img_shape = img.shape

        # Chọn hệ số scale ngẫu nhiên trong khoảng min-max
        if self.scale_x_min == self.scale_x_max:
            scale_x = self.scale_x_min
        else:
            scale_x = random.uniform(self.scale_x_min, self.scale_x_max)

        if self.scale_y_min == self.scale_y_max:
            scale_y = self.scale_y_min
        else:
            scale_y = random.uniform(self.scale_y_min, self.scale_y_max)

        # Resize ảnh
        img = cv2.resize(img, None, fx=scale_x, fy=scale_y)

        # Điều chỉnh tọa độ bounding box
        bboxes[:, :4] *= [scale_x, scale_y, scale_x, scale_y]

        # Tạo canvas có cùng kích thước với ảnh gốc
        canvas = np.zeros(img_shape, dtype=np.uint8)

        # Giới hạn kích thước ảnh sau scale để không vượt quá khung gốc
        y_lim = min(img.shape[0], img_shape[0])
        x_lim = min(img.shape[1], img_shape[1])

        # Chèn ảnh đã scale vào canvas
        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

        # Cập nhật bounding boxes và cắt phần vượt ra ngoài ảnh
        img = canvas
        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.2)

        return img, bboxes
