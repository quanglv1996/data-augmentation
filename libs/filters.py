import cv2
import random

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