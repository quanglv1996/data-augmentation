import numpy as np
from PIL import Image

class GridMask:
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=0, prob=0.7):
        assert 0 < ratio < 1, "ratio nên nằm trong (0, 1)"
        assert 0 <= prob <= 1, "prob nên nằm trong [0, 1]"
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = max(1, int(rotate))
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def set_prob(self, prob=0.5):
        assert 0 <= prob <= 1, "prob nên nằm trong [0, 1]"
        self.prob = prob

    def transform(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes

        h, w = img.shape[:2]
        d1, d2 = 2, min(h, w)
        hh, ww = int(1.5 * h), int(1.5 * w)
        d = np.random.randint(d1, d2)
        l = np.random.randint(1, d) if self.ratio == 1 else min(max(int(d * self.ratio + 0.5), 1), d - 1)

        mask = np.ones((hh, ww), np.float32)
        st_h, st_w = np.random.randint(d), np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s, t = d * i + st_h, min(d * i + st_h + l, hh)
                mask[s:t, :] = 0
        if self.use_w:
            for i in range(ww // d):
                s, t = d * i + st_w, min(d * i + st_w + l, ww)
                mask[:, s:t] = 0

        r = np.random.randint(self.rotate)
        mask_img = Image.fromarray(np.uint8(mask * 255))
        mask_img = mask_img.rotate(r)
        mask = np.asarray(mask_img) / 255.0
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask.astype(np.float32), axis=2)
        mask = np.tile(mask, [1, 1, 3])

        if self.offset:
            offset = 2 * (np.random.rand(h, w, 3) - 0.5)
            img = img * mask + (1 - mask) * offset
        else:
            img = img * mask

        img = img.astype(np.uint8)
        return img, bboxes