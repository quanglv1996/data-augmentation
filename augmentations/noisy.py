import numpy as np
import cv2

class Noisy(object):
    def __init__(self, noise_type="gauss"):
        self.noise_type = noise_type
        
    def transform(self, img, bboxes):
        # Apply Gaussian noise to the image
        if self.noise_type == "gauss":
            image = img.copy() 
            mean = 0
            st = 0.7
            gauss = np.random.normal(mean, st, image.shape)
            gauss = gauss.astype('uint8')
            image = cv2.add(image, gauss)
            return image, bboxes
    
        # Apply salt-and-pepper noise to the image
        elif self.noise_type == "sp":
            image = img.copy() 
            prob = 0.05
            if len(image.shape) == 2:
                black = 0
                white = 255            
            else:
                colorspace = image.shape[2]
                if colorspace == 3:  # RGB
                    black = np.array([0, 0, 0], dtype='uint8')
                    white = np.array([255, 255, 255], dtype='uint8')
                else:  # RGBA
                    black = np.array([0, 0, 0, 255], dtype='uint8')
                    white = np.array([255, 255, 255, 255], dtype='uint8')
            probs = np.random.random(image.shape[:2])
            image[probs < (prob / 2)] = black
            image[probs > 1 - (prob / 2)] = white
            return image, bboxes