import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import random

class Lighting_Noise(object):
    def __init__(self):
        pass
        
    def __call__(self, img, bboxes):
        img = img.copy()
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
                (1, 2, 0), (2, 0, 1), (2, 1, 0))
        swap = perms[random.randint(0, len(perms)- 1)]
        
        if type(img) == np.ndarray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        img = F.to_tensor(img)
        img = img[swap, :, :]
        img = F.to_pil_image(img)
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 
        
        return img, bboxes