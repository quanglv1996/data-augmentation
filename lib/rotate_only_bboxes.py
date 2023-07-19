import torchvision.transforms.functional as F
import numpy as np


class Rotate_Only_Bboxes(object):
    def __init__(self, angle=5):
        self.angle = angle
        
    def __call__(self, img, bboxes):
        new_image = img.copy()
        boxes = bboxes.copy()
        new_image = F.to_tensor(new_image)
        for i in range(boxes.shape[0]):
            x_min, y_min, x_max, y_max, id = map(int, boxes[i])
            bbox = new_image[:,  y_min:y_max+1, x_min:x_max+1]
            bbox = F.to_pil_image(bbox)
            bbox = bbox.rotate(self.angle)
            
            new_image[:,y_min:y_max+1, x_min:x_max+1] = F.to_tensor(bbox)
            img = F.to_pil_image(new_image)
            img = np.array(img) 
            img = img[:, :, ::-1].copy() 
        return img , boxes