import torchvision.transforms.functional as F
import numpy as np
import torch

class Mixup(object):
    def __init__(self, lambd=.3):
        self.lambd = lambd
        
    def __call__(self, img1, bboxes1, img2, bboxes2):
        img1 = F.to_tensor(img1)  #Tensor
        img2 = F.to_tensor(img2)  #Tensor
        
        mixup_width = max(img1.shape[2], img2.shape[2])
        mix_up_height = max(img1.shape[1], img2.shape[1])
        
        mix_img = torch.zeros(3, mix_up_height, mixup_width)
        mix_img[:, :img1.shape[1], :img1.shape[2]] = img1 * self.lambd
        mix_img[:, :img2.shape[1], :img2.shape[2]] += img2 * (1. - self.lambd)
        
        img = F.to_pil_image(mix_img)
        img = np.array(img) 
        img = img[:, :, ::-1].copy()

        mix_bboxes = np.concatenate((bboxes1, bboxes2))
    
        return img, mix_bboxes