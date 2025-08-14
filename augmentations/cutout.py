import random

class Cutout:
    def __init__(self, amount=0.3):
        assert 0.0 <= amount <= 1.0, "amount nên nằm trong [0.0, 1.0]"
        self.amount = amount

    def transform(self, img, bboxes):
        img = img.copy()
        bboxes = list(bboxes)
        num_select = max(1, round(self.amount * len(bboxes)))
        if len(bboxes) == 0:
            return img, bboxes

        ran_select = random.sample(bboxes, min(num_select, len(bboxes)))

        for box in ran_select:
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            mask_w = int(w * random.uniform(0.2, 0.7))
            mask_h = int(h * random.uniform(0.2, 0.7))
            if mask_w == 0 or mask_h == 0:
                continue
            mask_x1 = random.randint(x1, x2 - mask_w)
            mask_y1 = random.randint(y1, y2 - mask_h)
            mask_x2 = mask_x1 + mask_w
            mask_y2 = mask_y1 + mask_h
            img[mask_y1:mask_y2, mask_x1:mask_x2, :] = 0

        return img, bboxes