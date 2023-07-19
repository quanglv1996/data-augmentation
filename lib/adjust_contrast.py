import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image

class Adjust_Contrast(object):
    def __init__(self, contrast_factor=1.5):
        self.contrast_factor = contrast_factor
        
    def __call__(self, img, bboxes):
        if type(img) == np.ndarray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
    
        img = F.adjust_contrast(img,self.contrast_factor)
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 
    
        return img, bboxes 