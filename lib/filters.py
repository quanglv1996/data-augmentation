import cv2
import random
from utils import draw_rect, get_info_bbox

class Filters(object):
    def __init__(self):
        """
        Initialize the Filters augmentation class.
        """
        pass
    
    def __call__(self, img, bboxes):
        """
        Apply a random filter to the input image.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): An array of bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The augmented image with the selected filter applied.
            numpy.ndarray: The unchanged array of bounding boxes.
        """
        # Define a list of available filters
        filter_list = ["blur", "gaussian", "median"]
        # Randomly choose a filter type from the list
        f_type = random.choice(filter_list)
        
        # Calculate the filter size based on the image size (nearest odd number)
        temp = int(img.shape[0] / 100)
        fsize = temp if temp % 2 == 1 else temp + 1
    
        # Apply the selected filter to the image based on the chosen filter type
        if f_type == "blur":
            image = img.copy()
            return cv2.blur(image, (fsize, fsize)), bboxes
        elif f_type == "gaussian":
            image = img.copy()
            return cv2.GaussianBlur(image, (fsize, fsize), 0), bboxes
        elif f_type == "median":
            image = img.copy()
            return cv2.medianBlur(image, fsize), bboxes

        
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
    
    img_res, bboxes_res = Filters()(img.copy(), bboxes.copy())
    draw_rect(img_res, bboxes_res, img)
    
if __name__ == '__main__':
    main()