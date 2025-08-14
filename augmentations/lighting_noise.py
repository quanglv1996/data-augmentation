import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import random

class LightingNoise:
    def __init__(self):
        pass

    def transform(self, img, bboxes):
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        swap = random.choice(perms)

        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        img_tensor = F.to_tensor(img)
        img_tensor = img_tensor[swap, :, :]
        img = F.to_pil_image(img_tensor)
        img = np.array(img)[:, :, ::-1]

        return img, bboxes