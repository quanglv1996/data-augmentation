import random

class Cutout(object):
    def __init__(self, amount=.3):
        self.amount = amount
        
    def __call__(self, img, bboxes):
        img = img.copy()
        bboxes = list(bboxes.copy())
        ran_select = random.sample(bboxes, round(self.amount*len(bboxes)))

        for box in ran_select:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            mask_w = int((x2 - x1)*random.uniform(0, 0.7))
            mask_h = int((y2 - y1)*random.uniform(0, 0.7))
            mask_x1 = random.randint(x1, x2 - mask_w)
            mask_y1 = random.randint(y1, y2 - mask_h)
            mask_x2 = mask_x1 + mask_w
            mask_y2 = mask_y1 + mask_h
            
            img[mask_y1:mask_y2, mask_x1:mask_x2, :] = 0
        return img, bboxes