import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image

class Adjust_Saturation(object):
    def __init__(self, saturation_factor=1.5):
        self.saturation_factor = saturation_factor
        
    def __call__(self, img, bboxes):
        if type(img) == np.ndarray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
    
        img = F.adjust_saturation(img,self.saturation_factor)
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 
        return img, bboxes