import numpy as np

class HorizontalFlip:
    def __init__(self):
        pass

    def transform(self, img, bboxes):
        img_flipped = img[:, ::-1, :]
        if bboxes is None or len(bboxes) == 0:
            return img_flipped, bboxes

        bboxes = np.asarray(bboxes)
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(-1, 4)

        w = img.shape[1]
        # Lật bbox theo chiều ngang
        x_min = w - bboxes[:, 2]
        x_max = w - bboxes[:, 0]
        bboxes[:, 0] = np.minimum(x_min, x_max)
        bboxes[:, 2] = np.maximum(x_min, x_max)

        return img_flipped, bboxes