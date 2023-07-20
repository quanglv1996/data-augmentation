import random
import cv2
from utils import draw_rect, get_info_bbox

class Cutout(object):
    def __init__(self, amount=0.3):
        """
        Initialize the Cutout augmentation class.

        Args:
            amount (float): Proportion of bounding boxes to apply cutout. Default is 0.3.
        """
        # Store the provided cutout amount
        self.amount = amount
        
    def __call__(self, img, bboxes):
        """
        Apply the cutout augmentation to the input image.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): An array of bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The augmented image with cutout applied.
            list: The list of bounding boxes after cutout.
        """
        # Make a copy of the input image and bounding boxes
        img = img.copy()
        bboxes = list(bboxes.copy())
        
        # Randomly select a portion of bounding boxes based on the cutout amount
        ran_select = random.sample(bboxes, round(self.amount * len(bboxes)))

        # Apply cutout to the selected bounding boxes
        for box in ran_select:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            # Calculate the size and position of the cutout region
            mask_w = int((x2 - x1) * random.uniform(0, 0.7))
            mask_h = int((y2 - y1) * random.uniform(0, 0.7))
            mask_x1 = random.randint(x1, x2 - mask_w)
            mask_y1 = random.randint(y1, y2 - mask_h)
            mask_x2 = mask_x1 + mask_w
            mask_y2 = mask_y1 + mask_h
            
            # Apply cutout by setting the selected region to black (zero)
            img[mask_y1:mask_y2, mask_x1:mask_x2, :] = 0
        
        # Return the image with cutout applied and the bounding boxes
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
    
    img_res, bboxes_res = Cutout(.3)(img.copy(), bboxes.copy())
    draw_rect(img_res, bboxes_res, img)
    
if __name__ == '__main__':
    main()