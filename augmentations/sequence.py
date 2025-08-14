import random

class Sequence:
    def __init__(self, augmentations, probs=1):
        self.augmentations = augmentations
        self.probs = probs

    def transform(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            prob = self.probs[i] if isinstance(self.probs, list) else self.probs
            if prob > 0 and random.random() < prob:
                images, bboxes = augmentation.transform(images, bboxes)
        return images, bboxes
