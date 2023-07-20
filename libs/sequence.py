import random

class Sequence(object):
    def __init__(self, augmentations, probs=1):
        """
        Initialize the Sequence data augmentation object.

        Args:
            augmentations (list): A list of data augmentation techniques to be applied in sequence.
            probs (float or list): The probability/probabilities of applying each augmentation. If a single value is
                                   provided, the same probability is applied to all augmentations. If a list of
                                   probabilities is provided, it must have the same length as the 'augmentations' list.
        """
        self.augmentations = augmentations
        self.probs = probs

    def __call__(self, images, bboxes):
        """
        Apply the sequence of augmentations to the input images and bounding boxes.

        Args:
            images (list of numpy.ndarray): The list of input images.
            bboxes (list of numpy.ndarray): The list of corresponding bounding boxes.

        Returns:
            list of numpy.ndarray: The augmented images.
            list of numpy.ndarray: The adjusted bounding boxes.
        """
        # Loop through the list of augmentations and apply them based on their probabilities
        for i, augmentation in enumerate(self.augmentations):
            # Check if the probability is a single value or a list
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            # Check if the augmentation should be applied based on the probability
            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)

        # Return the augmented images and bounding boxes
        return images, bboxes
