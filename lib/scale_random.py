import sys
sys.path.append('../.')
import cv2
import numpy as np
import random
from lib.utils import clip_box
from utils import draw_rect, get_info_bbox
class RandomScale(object):

    def __init__(self, scale=0.2, diff=False):
        """
        Initialize the RandomScale data augmentation object.

        Args:
            scale (float or tuple): The scaling factor or a range of scaling factors. If a tuple, it should contain two values (min_scale, max_scale).
                The scaling factor should be positive, and if it's a tuple, the values should be greater than -1.
            diff (bool): If True, use different scaling factors for x and y directions. Otherwise, use the same scaling factor for both directions.
        """
        self.scale = scale

        # If scale is a tuple, ensure it is a valid range
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            # Ensure scale is a positive float
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff

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

        # Calculate random scale factors for x and y
        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        # Calculate the resize scale factors for x and y
        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        # Resize the image using the calculated scale factors
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
        bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

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
    
    
    img_res, bboxes_res = RandomScale(0.2)(img.copy(), bboxes.copy())
    draw_rect(img_res, bboxes_res, img)
    
if __name__ == '__main__':
    main()