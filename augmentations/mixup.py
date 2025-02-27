import torchvision.transforms.functional as F
import numpy as np
import torch

class Mixup(object):
    def __init__(self, lambd=0.3):
        self.lambd = lambd
        
    def transform(self, img1, bboxes1, img2, bboxes2):
        # Convert the input images to PyTorch tensors
        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        
        # Determine the dimensions of the mixed image
        mixup_width = max(img1.shape[2], img2.shape[2])
        mixup_height = max(img1.shape[1], img2.shape[1])
        
        # Create an empty tensor to hold the mixed image
        mix_img = torch.zeros(3, mixup_height, mixup_width)
        
        # Apply mixing to the images based on the lambda factor
        mix_img[:, :img1.shape[1], :img1.shape[2]] = img1 * self.lambd
        mix_img[:, :img2.shape[1], :img2.shape[2]] += img2 * (1. - self.lambd)
        
        # Convert the mixed image tensor back to a PIL image and then to a NumPy array
        img = F.to_pil_image(mix_img)
        img = np.array(img)[:, :, ::-1].copy()
        
        # Concatenate the bounding boxes from both images
        mix_bboxes = np.concatenate((bboxes1, bboxes2))
    
        # Return the mixed image and mixed bounding boxes
        return img, mix_bboxes