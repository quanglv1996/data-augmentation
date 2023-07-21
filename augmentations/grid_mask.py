import numpy as np
from PIL import Image

class GridMask(object):
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=0, prob=0.7):
        """
        Initialize the GridMask augmentation class with various parameters.

        Args:
            use_h (bool): If True, use grid masking along the height (vertical) dimension.
            use_w (bool): If True, use grid masking along the width (horizontal) dimension.
            rotate (int): The maximum rotation angle (in degrees) to apply to the mask.
            offset (bool): If True, apply an offset to the image after masking.
            ratio (float): The ratio of the mask size to the grid size.
            mode (int): The mode of the mask. If 0, the mask will be black; if 1, the mask will be white.
            prob (float): The probability of applying GridMask augmentation to an image.
        """
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, prob=0.5):
        """
        Set the probability of applying GridMask augmentation.

        Args:
            prob (float): The probability of applying GridMask augmentation to an image.
        """
        self.prob = prob

    def __call__(self, img, bboxes):
        """
        Apply GridMask augmentation to the input image.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): An array of bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The augmented image with GridMask applied.
            numpy.ndarray: The unchanged array of bounding boxes.
        """
        # Check if GridMask augmentation should be applied based on the probability
        if np.random.rand() > self.prob:
            return img, bboxes

        # Get image dimensions
        h = img.shape[0]
        w = img.shape[1]

        # Calculate the size of the grid
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # Create the mask
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        # Apply rotation to the mask
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask.astype(np.float), axis=2)
        mask = np.tile(mask, [1, 1, 3])
        mask = mask.astype(np.uint8)

        # Apply the GridMask augmentation to the image
        if self.offset:
            offset = np.float(2 * (np.random.rand(h, w) - 0.5))
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        # Return the augmented image and unchanged bounding boxes
        return img, bboxes