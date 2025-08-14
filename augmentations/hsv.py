import random
import numpy as np

class RandomHSV:
    def __init__(self, hue=10, saturation=30, brightness=30):
        # Đảm bảo các tham số là tuple (min, max)
        self.hue = (-hue, hue) if not isinstance(hue, tuple) else hue
        self.saturation = (-saturation, saturation) if not isinstance(saturation, tuple) else saturation
        self.brightness = (-brightness, brightness) if not isinstance(brightness, tuple) else brightness

    def transform(self, img, bboxes):
        # Random giá trị
        h = random.randint(*self.hue) if self.hue[0] != self.hue[1] else self.hue[0]
        s = random.randint(*self.saturation) if self.saturation[0] != self.saturation[1] else self.saturation[0]
        v = random.randint(*self.brightness) if self.brightness[0] != self.brightness[1] else self.brightness[0]

        img = img.astype(int)
        img[..., 0] = np.clip(img[..., 0] + h, 0, 179)
        img[..., 1] = np.clip(img[..., 1] + s, 0, 255)
        img[..., 2] = np.clip(img[..., 2] + v, 0, 255)
        img = img.astype(np.uint8)

        return img, bboxes