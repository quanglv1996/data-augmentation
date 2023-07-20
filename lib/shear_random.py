import cv2
import numpy as np
import random
from horizontal_flip import HorizontalFlip
from utils import draw_rect, get_info_bbox

class RandomShear(object):
    def __init__(self, shear_factor=0.2):
        """
        Initialize the RandomShear data augmentation object.

        Args:
            shear_factor (float or tuple): The shear factor range. If a single value is provided, the same shear factor
                                          will be applied along the x-axis and y-axis. If a tuple of two values is
                                          provided, the shear factor will be randomly sampled from this range for
                                          both x-axis and y-axis shearing.
        """
        # Initialize the object with the shear factor range (can be a single value or tuple)
        if type(shear_factor) == tuple:
            assert len(shear_factor) == 2, "Invalid range for shear factor"
        else:
            shear_factor = (-shear_factor, shear_factor)
        self.shear_factor = shear_factor

    def __call__(self, img, bboxes):
        """
        Apply random shear transformation to the input image and bounding boxes.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): Bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The sheared image.
            numpy.ndarray: The adjusted bounding boxes.
        """
        # Randomly sample a shear factor from the defined range
        shear_factor = random.uniform(*self.shear_factor)

        w, h = img.shape[1], img.shape[0]

        # Apply horizontal flip if the shear factor is negative
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        # Create an affine transformation matrix for shear
        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        # Calculate the new width after shearing
        nW = img.shape[1] + abs(shear_factor * img.shape[0])

        # Adjust bounding boxes based on shear factor
        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

        # Apply the shear transformation to the image
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        # Undo horizontal flip if the shear factor was negative
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        # Resize the image back to its original size
        img = cv2.resize(img, (w, h))

        # Adjust the bounding box coordinates based on the scaling factor
        scale_factor_x = nW / w
        bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]

        return img, bboxes
    
def main():
    label_mapping = {
        'disc': 0,
        'adapter':1,
        'guide':2,
        'qr':3,
        'gun':4,
        'boom': 5,
        'head': 6,
    }
    
    path_img = 'D:/data-augmentation-for-object-detection/data/1a7ff59a026f50acbf91d546e8048637.jpg'
    img = cv2.imread(path_img)
    path_xml = 'D:/data-augmentation-for-object-detection/data/1a7ff59a026f50acbf91d546e8048637.xml'
    bboxes = get_info_bbox(path_xml, label_mapping)
    
    
    img_res, bboxes_res = RandomShear(0.2)(img.copy(), bboxes.copy())
    draw_rect(img_res, bboxes_res, img)
    
if __name__ == '__main__':
    main()