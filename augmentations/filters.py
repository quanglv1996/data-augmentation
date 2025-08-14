import cv2
import random

class Filters:
    def __init__(self):
        pass

    def transform(self, img, bboxes):
        filter_list = ["blur", "gaussian", "median"]
        f_type = random.choice(filter_list)

        temp = max(3, int(img.shape[0] / 100))
        fsize = temp if temp % 2 == 1 else temp + 1

        image = img.copy()
        if fsize < 3:
            return image, bboxes

        if f_type == "blur":
            return cv2.blur(image, (fsize, fsize)), bboxes
        elif f_type == "gaussian":
            return cv2.GaussianBlur(image, (fsize, fsize), 0), bboxes
        elif f_type == "median":
            return cv2.medianBlur(image, fsize), bboxes
        else:
            return image, bboxes