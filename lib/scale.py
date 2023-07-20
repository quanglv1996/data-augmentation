import sys
sys.path.append('../.')
import cv2
import numpy as np
from lib.utils import clip_box
from utils import draw_rect, get_info_bbox

class Scale(object):

    def __init__(self, scale_x=0.2, scale_y=0.2):
        """
        Initialize the Scale data augmentation object.

        Args:
            scale_x (float): Scaling factor for the x direction. Default is 0.2, meaning a 20% increase in width.
            scale_y (float): Scaling factor for the y direction. Default is 0.2, meaning a 20% increase in height.
        """
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, img, bboxes):
        """
        Scale the input image and adjust the bounding box coordinates accordingly.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): Bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The scaled image.
            numpy.ndarray: The adjusted bounding boxes.
        """
        # Get the shape of the input image
        img_shape = img.shape

        # Calculate the resize scale factors for x and y
        resize_scale_x = 1 + self.scale_x
        resize_scale_y = 1 + self.scale_y

        # Resize the image using the specified scale factors
        img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

        # Adjust the bounding boxes accordingly
        bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        # Create a canvas to paste the resized image on
        canvas = np.zeros(img_shape, dtype=np.uint8)

        # Calculate the limiting boundaries for the resized image
        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])

        # Paste the resized image on the canvas
        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

        # Update the image and bounding boxes with the resized and clipped ones
        img = canvas
        bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.2)

        # Return the scaled image and adjusted bounding boxes
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
    
    
    img_res, bboxes_res = Scale(0.2, 0.2)(img.copy(), bboxes.copy())
    draw_rect(img_res, bboxes_res, img)
    
if __name__ == '__main__':
    main()