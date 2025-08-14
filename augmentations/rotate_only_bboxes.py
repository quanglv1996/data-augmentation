import torchvision.transforms.functional as F
import numpy as np

class RotateOnlyBboxes:
    def __init__(self, angle=5):
        self.angle = angle

    def transform(self, img, bboxes):
        img_tensor = F.to_tensor(img)
        boxes = np.asarray(bboxes).copy()

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            # Giới hạn bbox trong biên ảnh
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_tensor.shape[2] - 1, x_max)
            y_max = min(img_tensor.shape[1] - 1, y_max)
            if x_max <= x_min or y_max <= y_min:
                continue

            bbox = img_tensor[:, y_min:y_max+1, x_min:x_max+1]
            bbox_img = F.to_pil_image(bbox)
            bbox_img = bbox_img.rotate(self.angle)
            img_tensor[:, y_min:y_max+1, x_min:x_max+1] = F.to_tensor(bbox_img)

        img_out = F.to_pil_image(img_tensor)
        img_out = np.array(img_out)[:, :, ::-1]
        return img_out, boxes