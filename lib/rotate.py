import sys
sys.path.append('../.')
import cv2
import numpy as np
from lib.utils import get_corners, rotate_box, rotate_im, get_enclosing_box, clip_box
from utils import draw_rect, get_info_bbox

class Rotate(object):
    def __init__(self, angle):
        """
        Initialize the Rotate data augmentation object.

        Args:
            angle (float): The specified rotation angle in degrees for rotating the image and bounding boxes.
        """
        self.angle = angle

    def __call__(self, img, bboxes):
        """
        Rotate the image and adjust the bounding box coordinates accordingly.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): Bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The rotated image.
            numpy.ndarray: The adjusted bounding boxes.
        """
        # Get the rotation angle from the object's attribute
        angle = self.angle

        # Get the width and height of the image
        w, h = img.shape[1], img.shape[0]

        # Calculate the center coordinates of the image
        cx, cy = w // 2, h // 2

        # Get the corners of the bounding boxes and add the class labels
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))

        # Rotate the image and the bounding boxes
        img = rotate_im(img, angle)
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        # Get the new bounding box that encloses the rotated boxes
        new_bbox = get_enclosing_box(corners)

        # Calculate the scale factors to resize the image back to its original size
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        # Resize the image back to its original size
        img = cv2.resize(img, (w, h))

        # Scale the bounding boxes back to their original size
        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        # Update the bounding boxes with the resized ones
        bboxes = new_bbox

        # Clip the bounding boxes to stay within the image boundaries
        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

        # Return the rotated image and the adjusted bounding boxes
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
    
    
    img_res, bboxes_res = Rotate(10)(img.copy(), bboxes.copy())
    draw_rect(img_res, bboxes_res, img)
    
if __name__ == '__main__':
    main()