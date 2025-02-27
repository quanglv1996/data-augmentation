import numpy as np
import cv2
import random
from augmentations.horizontal_flip import HorizontalFlip

class Shear(object):
    def __init__(self, shear_min=-0.2, shear_max=0.2):
        self.shear_min = shear_min
        self.shear_max = shear_max
        self.hori_flip = HorizontalFlip()

    def transform(self, img, bboxes):
        # Chọn hệ số shear ngẫu nhiên trong khoảng shear_min - shear_max
        shear_factor = random.uniform(self.shear_min, self.shear_max)

        # Áp dụng flip nếu shear_factor âm để đảm bảo shear luôn dương
        flip_applied = False
        if shear_factor < 0:
            img, bboxes = self.hori_flip.transform(img, bboxes)
            flip_applied = True

        # Tạo ma trận biến đổi affine để shear
        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        # Tính toán chiều rộng mới sau khi shear
        new_width = img.shape[1] + abs(shear_factor * img.shape[0])

        # Điều chỉnh tọa độ bounding box theo shear
        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

        # Biến đổi ảnh bằng warpAffine
        img = cv2.warpAffine(img, M, (int(new_width), img.shape[0]))

        # Nếu đã flip trước đó, ta cần flip lại để đưa về trạng thái ban đầu
        if flip_applied:
            img, bboxes = self.hori_flip.transform(img, bboxes)

        return img, bboxes
