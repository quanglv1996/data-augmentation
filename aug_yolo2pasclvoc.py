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
from augmentations.translate import Translate
from utils.utils import create_folder, save_yolo_format, get_info_bbox_pascalvoc

from config import config_augmentation, config_data


class DataAugmentationForPascalVOC(object):
    def __init__(self, path_dataset, path_save, label_mapping, scale=[0.6, 0.2, 0.2]):
        # Initialize the DataAugmentationForPascalVOC object with provided parameters
        self.path_dataset = path_dataset
        self.path_save = path_save
        self.label_mapping = label_mapping
        self.scale = scale
        self.aug_conf = config_augmentation

        # Create folder to save augmented data
        create_folder(self.path_save)
        self.train_path = os.path.join(self.path_save, 'train')
        create_folder(self.train_path)
        self.val_path = os.path.join(self.path_save, 'val')
        create_folder(self.val_path)
        self.test_path = os.path.join(self.path_save, 'test')
        create_folder(self.test_path)

        # Default image type is 'jpg'
        self.type_img = 'jpg'


    def split_dataset(self):
            filenames = os.listdir(self.path_dataset)
            
            self.type_img = filenames[0].split('.')[-1]
            
            txt_filenames = [i for i in filenames if i.split('.')[-1] == 'txt']
            img_filenames = [i for i in filenames if i.split('.')[-1] == self.type_img]

            temp = [i.split('.')[0] for i in txt_filenames if i.split('.')[-1] == 'txt']
            img_without_label = [i for i in img_filenames if i.split('.')[0] not in temp]
            img_with_label = [i for i in img_filenames if i.split('.')[0] in temp]

            random.shuffle(img_without_label)
            random.shuffle(img_with_label)

            index_cut_train_wol = int(len(img_without_label) * self.scale[0])
            index_cut_val_wol = int(len(img_without_label)* (self.scale[0] + self.scale[1]))

            index_cut_train_wl = int(len(img_with_label) * self.scale[0])
            index_cut_val_wl = int(len(img_with_label) *(self.scale[0] + self.scale[1]))

            train_img = img_without_label[:index_cut_train_wol] + img_with_label[:index_cut_train_wl]
            val_img = img_without_label[index_cut_train_wol:index_cut_val_wol] + img_with_label[index_cut_train_wl:index_cut_val_wl]
            test_img = img_without_label[index_cut_val_wol:] + img_with_label[index_cut_val_wl:]

            random.shuffle(train_img)
            random.shuffle(val_img)
            random.shuffle(test_img)

            return train_img, val_img, test_img
        
    def augment_data(self, img_paths, data_set='train'):
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
            
            src_txt = src_img.replace(self.type_img, 'txt')
            bboxes = get_info_bbox_pascalvoc(src_txt, self.label_mapping)
            
            if os.path.exists(src_txt):
                save_yolo_format(img, bboxes, img_path_save, label_path_save)
                if data_set == 'test':
                    continue
                else:
                    # 1.AdjustBrightneess
                    if self.aug_conf.AdjustBrightneess['used']:
                        brightness_factor = self.aug_conf.AdjustBrightneess['brightness_factor']
                        img_, bboxes_ = AdjustBrightness(brightness_factor)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                    # 2.AdjustContrast
                    if self.aug_conf.AdjustContrast['used']:
                        contrast_factor = self.aug_conf.AdjustContrast['contrast_factor']
                        img_, bboxes_ = AdjustContrast(contrast_factor)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 3.AdjustSaturation
                    if self.aug_conf.AdjustSaturation['used']:
                        saturation_factor = self.aug_conf.AdjustSaturation['saturation_factor']
                        img_, bboxes_ = AdjustSaturation(saturation_factor)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 4.Cutout
                    if self.aug_conf.Cutout['used']:
                        amount = self.aug_conf.Cutout['amount']
                        img_, bboxes_ = Cutout(amount)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                    # 4.Filters
                    if self.aug_conf.Filters['used']:
                        img_, bboxes_ = Filters()(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 4.GridMask
                    if self.aug_conf.GridMask['used']:
                        use_h = self.aug_conf.GridMask['use_h']
                        use_w = self.aug_conf.GridMask['use_w']
                        rotate = self.aug_conf.GridMask['rotate']
                        offset = self.aug_conf.GridMask['offset']
                        ratio = self.aug_conf.GridMask['ratio']
                        mode = self.aug_conf.GridMask['mode']
                        prob = self.aug_conf.GridMask['prob']
                        img_, bboxes_ = GridMask(use_h, use_w, rotate, offset, ratio, mode, prob)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 5.RandomHorizontalFliP
                    if self.aug_conf.RandomHorizontalFlip['used']:
                        p = self.aug_conf.RandomHorizontalFlip['p']
                        img_, bboxes_ = RandomHorizontalFlip(p)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                    # 6.HorizontalFliP
                    if self.aug_conf.HorizontalFlip['used']:
                        img_, bboxes_ = HorizontalFlip()(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                     # 7.RandomHSV
                    if self.aug_conf.RandomHSV['used']:
                        hue = self.aug_conf.RandomHSV['hue']
                        saturation = self.aug_conf.RandomHSV['saturation']
                        brightness = self.aug_conf.RandomHSV['brightness']
                        img_, bboxes_ = RandomHSV(hue, saturation, brightness)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)

                    # 8.Lighting_Noise
                    if self.aug_conf.LightingNoise['used']:
                        img_, bboxes_ = LightingNoise()(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)

                    # 9.Noisy
                    if self.aug_conf.Noisy['used']:
                        noise_type = self.aug_conf.Noisy['noise_type']
                        img_, bboxes_ = Noisy(noise_type)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                    # 10.Resize
                    if self.aug_conf.Resize['used']:
                        inp_dim = self.aug_conf.Resize['inp_dim']
                        img_, bboxes_ = Resize(inp_dim)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                    # 11.RotateOnlyBboxes
                    if self.aug_conf.RotateOnlyBboxes['used']:
                        angle = self.aug_conf.RotateOnlyBboxes['angle']
                        img_, bboxes_ = RotateOnlyBboxes(angle)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 12.RandomRotate
                    if self.aug_conf.RandomRotate['used']:
                        angle = self.aug_conf.RandomRotate['angle']
                        img_, bboxes_ = RandomRotate(angle)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 13.Rotate
                    if self.aug_conf.Rotate['used']:
                        angle = self.aug_conf.Rotate['angle']
                        img_, bboxes_ = Rotate(angle)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 14.RandomScale
                    if self.aug_conf.RandomScale['used']:
                        scale = self.aug_conf.RandomScale['scale']
                        diff = self.aug_conf.RandomScale['diff']
                        img_, bboxes_ = RandomScale(scale, diff)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                    # 14.Scale
                    if self.aug_conf.Scale['used']:
                        scale_x = self.aug_conf.Scale['scale_x']
                        scale_y = self.aug_conf.Scale['scale_y']
                        img_, bboxes_ = Scale(scale_x, scale_y)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 15.RandomShear
                    if self.aug_conf.RandomShear['used']:
                        shear_factor = self.aug_conf.RandomShear['shear_factor']
                        img_, bboxes_ = RandomShear(shear_factor)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 15.Shear
                    if self.aug_conf.Shear['used']:
                        shear_factor = self.aug_conf.Shear['shear_factor']
                        img_, bboxes_ = Shear(shear_factor)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                        
                    # 16.SmallObjectAugmentation
                    if self.aug_conf.SmallObjectAugmentation['used']:
                        img_, bboxes_ = SmallObjectAugmentation()(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 17.RandomTranslate
                    if self.aug_conf.RandomTranslate['used']:
                        translate = self.aug_conf.RandomTranslate['translate']
                        diff = self.aug_conf.RandomTranslate['diff']
                        img_, bboxes_ = RandomTranslate(translate, diff)(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
                    
                    # 18.Sequence
                    if self.aug_conf.Sequence['used']:
                        img_, bboxes_ = Sequence([RandomHSV(40, 40, 30), RandomHorizontalFlip(), RandomScale(), RandomTranslate(),
                                        RandomRotate(10), RandomShear()])(img.copy(), bboxes.copy())
                        save_yolo_format(img_, bboxes_, img_path_save, label_path_save)
            else:
                copyfile(src_img, os.path.join(img_path_save, format(random.getrandbits(128), 'x') + '.jpg'))
                

    def create(self):
        train, val, test = self.split_dataset()
        self.augment_data(train, 'train')
        self.augment_data(val, 'val')
        self.augment_data(test, 'test')
        self.create_yaml()
        print('Create dataset complete...')

        
def main():
    task = config_data
    parser = ArgumentParser(description='Run data creation...')
    parser.add_argument('--path_raw', type=str,
                        default=task.path_data_raw, help='Path data raw')
    parser.add_argument('--path_save', type=str,
                        default=task.path_save, help='Path save dataset')
    parser.add_argument('--train_scale', type=float, default=task.train_scale,
                        help='Scale of training set')
    parser.add_argument('--val_scale', type=float,
                        default=task.val_scale, help='Scale of validation set')
    args = parser.parse_args()
    path_data = args.path_raw
    path_save = args.path_save
    label_mapping = task.label_mapping
    scale = [args.train_scale, args.val_scale,
             1 - args.train_scale - args.val_scale]
    dataset = DataAugmentationForPascalVOC(path_data, path_save, label_mapping, scale)
    dataset.create()

if __name__ == '__main__':
    main()