import numpy as np
import random

class SmallObjectAugmentation(object):
    def __init__(self, thresh=256*256, prob=0.7, copy_times=3, epochs=30, all_objects=False, one_object=False):
        """
        Initialize the SmallObjectAugmentation data augmentation object.

        Args:
            thresh (int): The threshold to determine whether an object is considered small based on its area.
            prob (float): The probability of applying the small object augmentation.
            copy_times (int): The number of times to copy the small object to different locations in the image.
            epochs (int): The maximum number of attempts to find a valid location for copying the small object.
            all_objects (bool): If True, the small object augmentation is applied to all small objects in the image.
            one_object (bool): If True, only one small object is selected for augmentation.
        """
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        if self.all_objects or self.one_object:
            self.copy_times = 1

    def is_small_object(self, h, w):
        """
        Check if an object is considered small based on its height and width.

        Args:
            h (int): Height of the object.
            w (int): Width of the object.

        Returns:
            bool: True if the object is small, False otherwise.
        """
        if h * w <= self.thresh:
            return True
        else:
            return False

    def compute_overlap(self, annot_a, annot_b):
        """
        Compute the overlap between two bounding boxes.

        Args:
            annot_a (numpy.ndarray): Bounding box coordinates (xmin, ymin, xmax, ymax) of the first box.
            annot_b (numpy.ndarray): Bounding box coordinates (xmin, ymin, xmax, ymax) of the second box.

        Returns:
            bool: True if the bounding boxes overlap, False otherwise.
        """
        if annot_a is None:
            return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
        if inter != 0:
            return True
        else:
            return False

    def do_not_overlap(self, new_annot, annots):
        """
        Check if a new bounding box does not overlap with existing bounding boxes.

        Args:
            new_annot (numpy.ndarray): New bounding box coordinates (xmin, ymin, xmax, ymax).
            annots (list): List of existing bounding box coordinates.

        Returns:
            bool: True if the new bounding box does not overlap with existing ones, False otherwise.
        """
        for annot in annots:
            if self.compute_overlap(new_annot, annot):
                return False
        return True

    def create_copy_annot(self, h, w, annot, annots):
        """
        Create a new copy of a small object bounding box.

        Args:
            h (int): Height of the image.
            w (int): Width of the image.
            annot (numpy.ndarray): Bounding box coordinates (xmin, ymin, xmax, ymax) of the small object.
            annots (list): List of existing bounding box coordinates.

        Returns:
            numpy.ndarray or None: New bounding box coordinates (xmin, ymin, xmax, ymax) if a valid location is found, else None.
        """
        annot = annot.astype(np.int)
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        for epoch in range(self.epochs):
            random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                 np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
            xmax, ymax = xmin + annot_w, ymin + annot_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_annot = np.array([xmin, ymin, xmax, ymax, annot[4]]).astype(np.int)

            if self.do_not_overlap(new_annot, annots) is False:
                continue

            return new_annot
        return None

    def add_patch_in_img(self, annot, copy_annot, image):
        """
        Add a patch of a small object to the image.

        Args:
            annot (numpy.ndarray): Bounding box coordinates (xmin, ymin, xmax, ymax) of the small object.
            copy_annot (numpy.ndarray): Bounding box coordinates (xmin, ymin, xmax, ymax) of the copy of the small object.
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image with the patch of the small object added.
        """
        copy_annot = copy_annot.astype(np.int)
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
        return image

    def __call__(self, img, bboxes):
        """
        Apply the SmallObjectAugmentation to the input image and bounding boxes.

        Args:
            img (numpy.ndarray): The input image.
            bboxes (numpy.ndarray): Bounding boxes associated with the image.

        Returns:
            numpy.ndarray: The augmented image.
            list: The adjusted bounding boxes.
        """
        if self.all_objects and self.one_object:
            return img, bboxes
        if np.random.rand() > self.prob:
            return img, bboxes

        img, annots = img, bboxes
        h, w = img.shape[0], img.shape[1]

        small_object_list = []
        for idx in range(annots.shape[0]):
            annot = annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
            if self.is_small_object(annot_h, annot_w):
                small_object_list.append(idx)

        l = len(small_object_list)
        # No Small Object
        if l == 0:
            return img, bboxes

        # Refine the copy_object by the given policy
        # Policy 2:
        copy_object_num = np.random.randint(0, l)
        # Policy 3:
        if self.all_objects:
            copy_object_num = l
        # Policy 1:
        if self.one_object:
            copy_object_num = 1

        random_list = random.sample(range(l), copy_object_num)
        annot_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        select_annots = annots[annot_idx_of_small_object, :]
        annots = annots.tolist()
        for idx in range(copy_object_num):
            annot = select_annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]

            if self.is_small_object(annot_h, annot_w) is False:
                continue

            for i in range(self.copy_times):
                new_annot = self.create_copy_annot(h, w, annot, annots)
                if new_annot is not None:
                    img = self.add_patch_in_img(new_annot, annot, img)
                    annots.append(new_annot)

        return img, annots