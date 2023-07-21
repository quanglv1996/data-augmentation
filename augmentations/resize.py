import numpy as np
from utils.utils import letterbox_image

class Resize(object):
    def __init__(self, inp_dim):
        """
        Initialize the Resize data augmentation object.

        Args:
            inp_dim (int): The desired input dimension (height and width) for resizing the image while maintaining aspect ratio.
        """
        self.inp_dim = inp_dim
        
    def __call__(self, img, bboxes):
        """
        Resize the input image and adjust the bounding box coordinates accordingly.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): Bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The resized image.
            numpy.ndarray: The adjusted bounding boxes.
        """
        # Get the original width and height of the image
        w, h = img.shape[1], img.shape[0]
        
        # Resize the image to fit the desired input dimension while maintaining the aspect ratio
        img = letterbox_image(img, self.inp_dim)
        
        # Calculate the scale factor to adjust the bounding box coordinates accordingly
        scale = min(self.inp_dim/h, self.inp_dim/w)
        bboxes[:, :4] *= scale
        
        # Calculate the new width and height after scaling
        new_w = scale * w
        new_h = scale * h
        inp_dim = self.inp_dim   
    
        # Calculate the amount to add to the bounding box coordinates to center them in the resized image
        del_h = (inp_dim - new_h) / 2
        del_w = (inp_dim - new_w) / 2
        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
        bboxes[:, :4] += add_matrix
        
        # Convert the resized image back to uint8 data type
        img = img.astype(np.uint8)
        
        # Return the resized image and adjusted bounding boxes
        return img, bboxes