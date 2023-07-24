import sys
sys.path.append('')
sys.path.append('../../')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import random
import cv2
from tqdm import tqdm
from shutil import copyfile
from argparse import ArgumentParser


from augmentations.adjust_brightness import AdjustBrightness
from augmentations.adjust_contrast import AdjustContrast
from augmentations.adjust_saturation import AdjustSaturation
from augmentations.cutout import Cutout
from augmentations.filters import Filters
from augmentations.grid_mask import GridMask
from augmentations.horizontal_flip import HorizontalFlip
from augmentations.horizontal_flip_random import RandomHorizontalFlip
from augmentations.hsv_random import RandomHSV
from augmentations.lighting_noise import LightingNoise
from augmentations.mixup import Mixup
from augmentations.noisy import Noisy
from augmentations.resize import Resize
from augmentations.rotate_only_bboxes import RotateOnlyBboxes
from augmentations.rotate_random import RandomRotate
from augmentations.rotate import Rotate
from augmentations.scale_random import RandomScale
from augmentations.scale import Scale
from augmentations.sequence import Sequence
from augmentations.shear_random import RandomShear
from augmentations.shear import Shear
from augmentations.small_object_augmentation import SmallObjectAugmentation
from augmentations.translate_random import RandomTranslate
from utils.utils import create_folder
from utils.utils import save_sample
from utils.utils import get_info_bbox_yolo, get_info_bbox_pascalvoc

from config import config_augmentation, config_data


class DataAugmentation(object):
    def __init__(self, path_dataset, path_save, label_mapping, scale=[0.6, 0.2, 0.2], src_type_dataset='voc', dest_type_dataset='yolo'):
        """
        Initialize the DataAugmentationForYoloV5 object with the provided parameters.

        Parameters:
            path_dataset (str): The path to the dataset directory.
            path_save (str): The path to save augmented data.
            label_mapping (dict): A dictionary that maps class names to their corresponding integer labels.
            scale (list, optional): A list specifying the scale for training, validation, and testing sets, respectively.
                                Default is [0.6, 0.2, 0.2].
            src_type_dataset (str, optional): The source data format, which can be either 'voc' or 'yolo'. 
                                            Default is 'voc'.
            dest_type_dataset (str, optional): The destination data format to be converted, which can be either 'voc' or 'yolo'.
                                            Default is 'yolo'.
        """
        # Initialize instance variables with provided parameters
        self.path_dataset = path_dataset
        self.path_save = path_save
        self.label_mapping = label_mapping
        self.scale = scale
        self.aug_conf = config_augmentation
        self.src_type_dataset = src_type_dataset
        self.dest_type_dataset = dest_type_dataset

        # Create folders to save augmented data
        create_folder(self.path_save)
        self.train_path = os.path.join(self.path_save, 'train')
        create_folder(self.train_path)
        self.val_path = os.path.join(self.path_save, 'val')
        create_folder(self.val_path)
        self.test_path = os.path.join(self.path_save, 'test')
        create_folder(self.test_path)

        # Default image type is 'jpg'
        self.type_img = 'jpg'
        
    def create_yaml(self):
        """
        Create a YAML file containing information about the dataset.

        The YAML file will include:
        - Paths to the training and validation datasets.
        - The number of classes (nc).
        - The names of the classes (names).

        The file will be saved in the path specified by self.path_save as 'data.yaml'.
        """
        print('Create yaml file...')
        
        # Get the number of classes and names from the label_mapping dictionary
        nc = len(self.label_mapping)
        names = list(self.label_mapping.keys())

        # Create the path to save the YAML file
        path_save_yaml = os.path.join(self.path_save, 'data.yaml')

        # Write the dataset information to the YAML file
        with open(path_save_yaml, "w") as f:
            f.write('train: {}\n'.format(self.train_path))  # Path to the training dataset
            f.write('val: {}\n'.format(self.val_path))      # Path to the validation dataset
            f.write('nc: {}\n'.format(nc))                  # Number of classes
            f.write('names: {}\n'.format(str(names)))       # Names of the classes
        
    def split_dataset(self):
        """
        Split the dataset into training, validation, and testing sets.

        Returns:
            tuple: A tuple containing lists of filenames for training, validation, and testing sets, respectively.
        """
        # Determine the file extension for label files based on the source data type
        if self.src_type_dataset == 'yolo':
            self.type_format_label = 'txt'
        elif self.src_type_dataset == 'voc':
            self.type_format_label = 'xml'
        
        # Get the list of all filenames in the dataset directory
        filenames = os.listdir(self.path_dataset)
        
        # Get the image file extension
        self.type_img = filenames[0].split('.')[-1]
        
        # Separate image and label filenames based on their file extensions
        label_filenames = [i for i in filenames if i.split('.')[-1] == self.type_format_label]
        img_filenames = [i for i in filenames if i.split('.')[-1] == self.type_img]

        # Identify images with and without corresponding label files
        temp = [i.split('.')[0] for i in label_filenames if i.split('.')[-1] == self.type_format_label]
        img_without_label = [i for i in img_filenames if i.split('.')[0] not in temp]
        img_with_label = [i for i in img_filenames if i.split('.')[0] in temp]

        # Shuffle the image lists to randomize data splitting
        random.shuffle(img_without_label)
        random.shuffle(img_with_label)

        # Calculate the number of images for training, validation, and testing sets
        index_cut_train_wol = int(len(img_without_label) * self.scale[0])
        index_cut_val_wol = int(len(img_without_label) * (self.scale[0] + self.scale[1]))

        index_cut_train_wl = int(len(img_with_label) * self.scale[0])
        index_cut_val_wl = int(len(img_with_label) * (self.scale[0] + self.scale[1]))

        # Create lists of filenames for training, validation, and testing sets
        train_img = img_without_label[:index_cut_train_wol] + img_with_label[:index_cut_train_wl]
        val_img = img_without_label[index_cut_train_wol:index_cut_val_wol] + img_with_label[index_cut_train_wl:index_cut_val_wl]
        test_img = img_without_label[index_cut_val_wol:] + img_with_label[index_cut_val_wl:]

        # Shuffle the data within each set to randomize the order
        random.shuffle(train_img)
        random.shuffle(val_img)
        random.shuffle(test_img)

        return train_img, val_img, test_img
        
    def augment_data(self, img_paths, data_set='train'):
        """
        Augment the data in the specified dataset.

        Args:
            img_paths (list): List of image file names to be augmented.
            data_set (str): Type of dataset to augment, should be one of ['train', 'val', 'test'].

        The augmented data will be saved in the corresponding 'images' and 'labels' folders in the specified dataset.

        If the data_set provided is not valid, an error message will be displayed.

        For each image in img_paths, the bounding box information will be extracted from the corresponding label file
        based on the self.src_type_dataset, and augmentation will be applied to the image and bounding boxes.

        If the self.src_type_dataset is 'yolo', the bounding boxes will be read using the YOLO format.
        If the self.src_type_dataset is 'voc', the bounding boxes will be read using the Pascal VOC format.

        The augmented data will be saved in the YOLO format (if self.dest_type_dataset is 'yolo') or
        the Pascal VOC format (if self.dest_type_dataset is 'voc').

        The augmented images and labels will be saved in the 'images' and 'labels' folders, respectively,
        in the corresponding dataset path (train, val, or test).
        """
        # Dictionary to map data type to corresponding paths
        data_paths = {
            'train': self.train_path,
            'val': self.val_path,
            'test': self.test_path
        }
        
        # Check if the data_type is valid
        if data_set not in data_paths:
            print('Invalid data type')
            return
        
        # Get the path to save images and labels
        path_dataset = data_paths[data_set]
        img_path_save = os.path.join(path_dataset, 'images')
        label_path_save = os.path.join(path_dataset, 'labels')
        
        # Create the required folders
        create_folder(img_path_save)
        create_folder(label_path_save)

        print('Create {} dataset...'.format(data_set))
        for filename in tqdm(img_paths):
            src_img = os.path.join(self.path_dataset, filename)
            img = cv2.imread(src_img)
            src_label = src_img.replace(self.type_img, self.type_format_label)

            # Extract bounding box information based on the data format
            if self.src_type_dataset == 'yolo':
                bboxes = get_info_bbox_yolo(img, src_label)
            elif self.src_type_dataset == 'voc':
                bboxes = get_info_bbox_pascalvoc(src_label, self.label_mapping)
            
            if os.path.exists(src_label):
                # Save the original sample without augmentation
                save_sample(self.dest_type_dataset, img, bboxes, img_path_save, label_path_save, self.label_mapping)

                # Continue processing for training and validation sets
                if data_set == 'test':
                    continue
                else:
                    # Data augmentation options for the training and validation sets

                    # 1. AdjustBrightneess
                    if self.aug_conf.AdjustBrightneess['used']:
                        brightness_factor = self.aug_conf.AdjustBrightneess['brightness_factor']
                        img_, bboxes_ = AdjustBrightness(brightness_factor)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 2. AdjustContrast
                    if self.aug_conf.AdjustContrast['used']:
                        contrast_factor = self.aug_conf.AdjustContrast['contrast_factor']
                        img_, bboxes_ = AdjustContrast(contrast_factor)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 3. AdjustSaturation
                    if self.aug_conf.AdjustSaturation['used']:
                        saturation_factor = self.aug_conf.AdjustSaturation['saturation_factor']
                        img_, bboxes_ = AdjustSaturation(saturation_factor)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 4. Cutout
                    if self.aug_conf.Cutout['used']:
                        amount = self.aug_conf.Cutout['amount']
                        img_, bboxes_ = Cutout(amount)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 5. Filters
                    if self.aug_conf.Filters['used']:
                        img_, bboxes_ = Filters()(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 6. GridMask
                    if self.aug_conf.GridMask['used']:
                        use_h = self.aug_conf.GridMask['use_h']
                        use_w = self.aug_conf.GridMask['use_w']
                        rotate = self.aug_conf.GridMask['rotate']
                        offset = self.aug_conf.GridMask['offset']
                        ratio = self.aug_conf.GridMask['ratio']
                        mode = self.aug_conf.GridMask['mode']
                        prob = self.aug_conf.GridMask['prob']
                        img_, bboxes_ = GridMask(use_h, use_w, rotate, offset, ratio, mode, prob)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 7. RandomHorizontalFliP
                    if self.aug_conf.RandomHorizontalFlip['used']:
                        p = self.aug_conf.RandomHorizontalFlip['p']
                        img_, bboxes_ = RandomHorizontalFlip(p)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 8. HorizontalFliP
                    if self.aug_conf.HorizontalFlip['used']:
                        img_, bboxes_ = HorizontalFlip()(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 9. RandomHSV
                    if self.aug_conf.RandomHSV['used']:
                        hue = self.aug_conf.RandomHSV['hue']
                        saturation = self.aug_conf.RandomHSV['saturation']
                        brightness = self.aug_conf.RandomHSV['brightness']
                        img_, bboxes_ = RandomHSV(hue, saturation, brightness)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 10. Lighting_Noise
                    if self.aug_conf.LightingNoise['used']:
                        img_, bboxes_ = LightingNoise()(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 11. Noisy
                    if self.aug_conf.Noisy['used']:
                        noise_type = self.aug_conf.Noisy['noise_type']
                        img_, bboxes_ = Noisy(noise_type)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 12. Resize
                    if self.aug_conf.Resize['used']:
                        inp_dim = self.aug_conf.Resize['inp_dim']
                        img_, bboxes_ = Resize(inp_dim)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 13. RotateOnlyBboxes
                    if self.aug_conf.RotateOnlyBboxes['used']:
                        angle = self.aug_conf.RotateOnlyBboxes['angle']
                        img_, bboxes_ = RotateOnlyBboxes(angle)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 14. RandomRotate
                    if self.aug_conf.RandomRotate['used']:
                        angle = self.aug_conf.RandomRotate['angle']
                        img_, bboxes_ = RandomRotate(angle)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 15. Rotate
                    if self.aug_conf.Rotate['used']:
                        angle = self.aug_conf.Rotate['angle']
                        img_, bboxes_ = Rotate(angle)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 16. RandomScale
                    if self.aug_conf.RandomScale['used']:
                        scale = self.aug_conf.RandomScale['scale']
                        diff = self.aug_conf.RandomScale['diff']
                        img_, bboxes_ = RandomScale(scale, diff)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 17. Scale
                    if self.aug_conf.Scale['used']:
                        scale_x = self.aug_conf.Scale['scale_x']
                        scale_y = self.aug_conf.Scale['scale_y']
                        img_, bboxes_ = Scale(scale_x, scale_y)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 18. RandomShear
                    if self.aug_conf.RandomShear['used']:
                        shear_factor = self.aug_conf.RandomShear['shear_factor']
                        img_, bboxes_ = RandomShear(shear_factor)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 19. Shear
                    if self.aug_conf.Shear['used']:
                        shear_factor = self.aug_conf.Shear['shear_factor']
                        img_, bboxes_ = Shear(shear_factor)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 20. SmallObjectAugmentation
                    if self.aug_conf.SmallObjectAugmentation['used']:
                        img_, bboxes_ = SmallObjectAugmentation()(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 21. RandomTranslate
                    if self.aug_conf.RandomTranslate['used']:
                        translate = self.aug_conf.RandomTranslate['translate']
                        diff = self.aug_conf.RandomTranslate['diff']
                        img_, bboxes_ = RandomTranslate(translate, diff)(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)

                    # 22. Sequence
                    if self.aug_conf.Sequence['used']:
                        img_, bboxes_ = Sequence([RandomHSV(40, 40, 30), RandomHorizontalFlip(), RandomScale(), RandomTranslate(),
                                        RandomRotate(10), RandomShear()])(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_, bboxes_, img_path_save, label_path_save, self.label_mapping)
            else:
                # If src_label does not exist, copy the image file to the destination folder
                copyfile(src_img, os.path.join(img_path_save, format(random.getrandbits(128), 'x') + '.jpg'))

                

    def create(self):
        # Split the dataset into training, validation, and test sets
        train, val, test = self.split_dataset()

        # Augment the data for the training set
        self.augment_data(train, 'train')

        # Augment the data for the validation set
        self.augment_data(val, 'val')

        # Augment the data for the test set
        self.augment_data(test, 'test')

        # If the destination dataset type is 'yolo', create a YAML file for YOLO format
        if self.dest_type_dataset == 'yolo':
            self.create_yaml()

        # Print a message indicating that dataset creation is complete
        print('Create augmentation dataset complete...')

def main():
    # Load configuration data
    task = config_data

    # Create an argument parser to handle command-line arguments
    parser = ArgumentParser(description='Run data creation...')
    
    # Define command-line arguments
    parser.add_argument('--path_raw', type=str, default=task.path_data_raw, help='Path to raw data')
    parser.add_argument('--path_save', type=str, default=task.path_save, help='Path to save the dataset')
    parser.add_argument('--train_scale', type=float, default=task.train_scale, help='Scale of the training set')
    parser.add_argument('--val_scale', type=float, default=task.val_scale, help='Scale of the validation set')
    parser.add_argument('--src_type_dataset', type=str, default=task.src_type_dataset, help='Source data format (options: ["yolo", "voc"])')
    parser.add_argument('--dest_type_dataset', type=str, default=task.dest_type_dataset, help='Destination data format (options: ["yolo", "voc"])')

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Extract relevant arguments
    path_data = args.path_raw
    path_save = args.path_save
    scale = [args.train_scale, args.val_scale, 1 - args.train_scale - args.val_scale]
    src_type_dataset = args.src_type_dataset
    dest_type_dataset = args.dest_type_dataset
    
    # Load label mapping from the configuration data
    label_mapping = task.label_mapping
    
    # Create the DataAugmentation object with the specified parameters
    dataset = DataAugmentation(path_data, path_save, label_mapping, scale, src_type_dataset, dest_type_dataset)
    
    # Run the data creation process
    dataset.create()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()