import numpy as np

class HorizontalFlip(object):
    def __init__(self):
        pass

    def transform(self, img, bboxes):
        if bboxes is None or len(bboxes) == 0:
            return img[:, ::-1, :], bboxes  # Chỉ lật ảnh nếu không có bbox
        
        bboxes = np.asarray(bboxes)  # Đảm bảo bboxes là mảng NumPy 2D
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(-1, 4)  # Chuyển đổi thành (N, 4) nếu cần
        
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]  # Lật ảnh ngang
        
        # Điều chỉnh tọa độ bounding box
        bboxes[:, [0, 2]] = 2 * img_center[[0, 2]] - bboxes[:, [0, 2]]

        # Sắp xếp lại giá trị bbox để đảm bảo min-max đúng thứ tự
        bboxes[:, [0, 2]] = np.sort(bboxes[:, [0, 2]], axis=1)

        return img, bboxes