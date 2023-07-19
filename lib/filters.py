import cv2
import random

class Filters(object):
    def __init__(self):
        pass
    def __call__(self, img, bboxes):
        filter_list = ["blur", "gaussian", "median"]
        f_type = random.choice(filter_list)
        temp = int(img.shape[0]/100)
        if temp%2 == 1: 
            fsize = temp 
        else:
            fsize = temp +1
    
        if f_type == "blur":
            image=img.copy()
            return cv2.blur(image,(fsize,fsize)), bboxes
        elif f_type == "gaussian":
            image=img.copy()
            return cv2.GaussianBlur(image, (fsize, fsize), 0), bboxes
        elif f_type == "median":
            image=img.copy()
            return cv2.medianBlur(image, fsize), bboxes