import numpy as np
import random
from utils.utils import clip_box

class RandomTranslate(object):
    def __init__(self, translate=0.2, diff=False):
        """
        Randomly translates the input image and its corresponding bounding boxes.

        Args:
            translate (float or tuple): If float, the same translation factor will be used for both x and y axes.
                                       If tuple, it should be (min_factor, max_factor) specifying the range of
                                       translation factors for both x and y axes. The factors should be between 0 and 1.
            diff (bool): If True, uses different translation factors for x and y axes. Default is False.
        """
        self.translate = translate
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert 0 < self.translate[0] < 1, "Translate factor should be between 0 and 1"
            assert 0 < self.translate[1] < 1, "Translate factor should be between 0 and 1"
        else:
            assert 0 < self.translate < 1, "Translate factor should be between 0 and 1"
            self.translate = (-self.translate, self.translate)
            
        self.diff = diff

    def __call__(self, img, bboxes):        
        """
        Applies random translation to the input image and its bounding boxes.

        Args:
            img (numpy array): The input image as a numpy array.
            bboxes (numpy array): An array of bounding boxes in the format [xmin, ymin, xmax, ymax, class_id].

        Returns:
            numpy array: Translated image.
            numpy array: Updated bounding boxes after translation.
        """
        img_shape = img.shape
     
        # Randomly sample translation factors for x and y axes
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)
        
        # If self.diff is True, use different translation factors for x and y axes
        if not self.diff:
            translate_factor_y = translate_factor_x
            
        # Create a canvas of zeros with the same shape as the input image
        canvas = np.zeros(img_shape, dtype=np.uint8)
    
        # Calculate the corner coordinates of the translated image region
        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])
   
        # Calculate the coordinates of the original image region that will be placed in the canvas
        orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1], corner_x + img.shape[1])]
    
        # Extract the image region to be translated
        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
        # Paste the translated region in the canvas
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
        img = canvas
        
        # Update the bounding box coordinates based on the translation
        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]
        
        # Clip the bounding boxes to stay within the image boundaries
        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)
        
        return img, bboxes