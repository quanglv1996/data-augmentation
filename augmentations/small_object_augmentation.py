import numpy as np
import random

class SmallObjectAugmentation:
    def __init__(self, thresh=256*256, prob=0.7, copy_times=3, epochs=30, all_objects=False, one_object=False):
        self.thresh = thresh
        self.prob = prob
        self.copy_times = 1 if (all_objects or one_object) else copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object

    def is_small_object(self, h, w):
        return h * w <= self.thresh

    def compute_overlap(self, annot_a, annot_b):
        if annot_a is None:
            return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, right_min - left_max) * max(0, bottom_min - top_max)
        return inter != 0

    def do_not_overlap(self, new_annot, annots):
        return not any(self.compute_overlap(new_annot, annot) for annot in annots)

    def create_copy_annot(self, h, w, annot, annots):
        annot = annot.astype(np.int32)
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        for _ in range(self.epochs):
            random_x = np.random.randint(int(annot_w / 2), int(w - annot_w / 2))
            random_y = np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            xmin, ymin = random_x - annot_w // 2, random_y - annot_h // 2
            xmax, ymax = xmin + annot_w, ymin + annot_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_annot = np.array([xmin, ymin, xmax, ymax, annot[4]], dtype=np.int32)
            if self.do_not_overlap(new_annot, annots):
                return new_annot
        return None

    def add_patch_in_img(self, annot, copy_annot, image):
        copy_annot = copy_annot.astype(np.int32)
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
        return image

    def transform(self, img, bboxes):
        if self.all_objects and self.one_object:
            return img, bboxes
        if np.random.rand() > self.prob:
            return img, bboxes

        h, w = img.shape[:2]
        annots = np.asarray(bboxes)
        small_object_list = [idx for idx, annot in enumerate(annots)
                             if self.is_small_object(annot[3] - annot[1], annot[2] - annot[0])]
        l = len(small_object_list)
        if l == 0:
            return img, bboxes

        if self.all_objects:
            copy_object_num = l
        elif self.one_object:
            copy_object_num = 1
        else:
            copy_object_num = np.random.randint(1, l + 1)

        random_list = random.sample(small_object_list, copy_object_num)
        select_annots = annots[random_list, :]
        annots_list = annots.tolist()
        for annot in select_annots:
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
            if not self.is_small_object(annot_h, annot_w):
                continue
            for _ in range(self.copy_times):
                new_annot = self.create_copy_annot(h, w, annot, annots_list)
                if new_annot is not None:
                    img = self.add_patch_in_img(new_annot, annot, img)
                    annots_list.append(new_annot)
        return img, np.array(annots_list, dtype=np.int32)