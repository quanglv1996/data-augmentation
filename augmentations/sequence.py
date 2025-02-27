import random

class Sequence(object):
    def __init__(self, augmentations, probs=1):
        self.augmentations = augmentations
        self.probs = probs

    def transform(self, images, bboxes):
        # Loop through the list of augmentations and apply them based on their probabilities
        for i, augmentation in enumerate(self.augmentations):
            # Check if the probability is a single value or a list
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            # Check if the augmentation should be applied based on the probability
            if random.random() < prob:
                images, bboxes = augmentation.transform(images, bboxes)

        # Return the augmented images and bounding boxes
        return images, bboxes
