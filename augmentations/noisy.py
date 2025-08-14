import numpy as np
import cv2

class Noisy:
    def __init__(self, noise_type="gauss", mean=0, std=10, prob=0.05):
        self.noise_type = noise_type
        self.mean = mean
        self.std = std
        self.prob = prob

    def transform(self, img, bboxes):
        image = img.copy()
        if self.noise_type == "gauss":
            gauss = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            return noisy, bboxes

        elif self.noise_type == "sp":
            probs = np.random.rand(*image.shape[:2])
            if image.ndim == 2:
                black, white = 0, 255
            else:
                black = np.zeros(image.shape[2], dtype='uint8')
                white = np.full(image.shape[2], 255, dtype='uint8')
            image[probs < (self.prob / 2)] = black
            image[probs > 1 - (self.prob / 2)] = white
            return image, bboxes