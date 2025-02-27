import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import random

class LightingNoise(object):
    def __init__(self):
        pass
        
    def transform(self, img, bboxes):
        # Create a copy of the image
        img = img.copy()

        # Define permutations to swap color channels
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
                (1, 2, 0), (2, 0, 1), (2, 1, 0))
        swap = perms[random.randint(0, len(perms) - 1)]
        
        # Convert the image to RGB format if it is a numpy array
        if type(img) == np.ndarray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        # Convert the image to a PyTorch tensor
        img = F.to_tensor(img)

        # Perform color channel swapping
        img = img[swap, :, :]

        # Convert the tensor back to a PIL image and then to a NumPy array
        img = F.to_pil_image(img)
        img = np.array(img)[:, :, ::-1].copy()
        
        # Return the augmented image and original bounding boxes (no adjustment on bounding boxes)
        return img, bboxes