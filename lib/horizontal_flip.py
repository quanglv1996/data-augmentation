import numpy as np
import cv2
from utils import draw_rect, get_info_bbox

class HorizontalFlip(object):
    def __init__(self):
        """
        Initialize the HorizontalFlip data augmentation object.
        """
        pass

    def __call__(self, img, bboxes):
        """
        Apply horizontal flip to the input image and adjust bounding box coordinates.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): An array of bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The horizontally flipped image.
            numpy.ndarray: The bounding boxes adjusted after the horizontal flip.
        """
        # Get the center of the image
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        # Flip the image horizontally
        img = img[:, ::-1, :]

        # Adjust the bounding box coordinates after flipping
        bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
        box_w = np.abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w

        # Return the flipped image and adjusted bounding boxes
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
    
    img_res, bboxes_res = HorizontalFlip()(img.copy(), bboxes.copy())
    draw_rect(img_res, bboxes_res, img)
    
if __name__ == '__main__':
    main()