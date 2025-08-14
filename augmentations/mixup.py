import torchvision.transforms.functional as F
import numpy as np
import torch
from PIL import Image

class Mixup:
    def __init__(self, lambd=0.3):
        assert 0.0 <= lambd <= 1.0, "lambda nên nằm trong [0, 1]"
        self.lambd = lambd

    def transform(self, img1, bboxes1, img2, bboxes2):
        # Đảm bảo hai ảnh cùng kích thước
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)

        t1 = F.to_tensor(img1)
        t2 = F.to_tensor(img2)
        mix_img = t1 * self.lambd + t2 * (1. - self.lambd)
        img = F.to_pil_image(mix_img)
        img = np.array(img)[:, :, ::-1]

        mix_bboxes = np.concatenate((bboxes1, bboxes2), axis=0)
        return img, mix_bboxes