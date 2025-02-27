import os
import random
import yaml
import cv2
import ast
import configparser
from tqdm import tqdm
from shutil import copyfile

# Import augmentation modules
from augmentations.brightness import AdjustBrightness
from augmentations.contrast import AdjustContrast
from augmentations.saturation import AdjustSaturation
from augmentations.cutout import Cutout
from augmentations.filters import Filters
from augmentations.grid_mask import GridMask
from augmentations.horizontal_flip import HorizontalFlip
from augmentations.hsv import RandomHSV
from augmentations.lighting_noise import LightingNoise
from augmentations.mixup import Mixup
from augmentations.noisy import Noisy
from augmentations.resize import Resize
from augmentations.rotate_only_bboxes import RotateOnlyBboxes
from augmentations.rotate import Rotate
from augmentations.scale import Scale
from augmentations.sequence import Sequence
from augmentations.shear import Shear
from augmentations.small_object_augmentation import SmallObjectAugmentation
from augmentations.translate import Translate

# Import utility functions
from utils.utils import create_folder, save_sample, get_info_bbox_yolo, get_info_bbox_pascalvoc


class Augmentation:
    def __init__(self, config_path):
        self.config_path = config_path
        
        self.config_dict = self._load_config()
        self._create_augmentation_object()
        
        # Paths
        self.path_dataset = self.config_dict['MAIN']['path_dataset']
        self.path_save = self.config_dict['MAIN']['path_save']
        self.src_type_dataset = self.config_dict['MAIN']['src_type_dataset']
        self.dest_type_dataset = self.config_dict['MAIN']['dest_type_dataset']
        self.label_mapping = self.config_dict['MAIN']['label_mapping']
        self.scale_dataset = self.config_dict['MAIN']['scale']

        self.train_path = os.path.join(self.path_save, 'train')
        self.val_path = os.path.join(self.path_save, 'val')
        self.test_path = os.path.join(self.path_save, 'test')
        for folder in [self.path_save, self.train_path, self.val_path, self.test_path]:
            create_folder(folder)
        
        self.type_img = 'jpg'
        

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)

        config_dict = {
            section: {key: self._convert_type(value) for key, value in config.items(section)}
            for section in config.sections()
        }

        if 'MAIN' in config_dict:
            if 'label_mapping' in config_dict['MAIN']:
                label_str = config_dict['MAIN']['label_mapping'].strip("[]") 
                label_pairs = [item.split(":") for item in label_str.split(",")]
                config_dict['MAIN']['label_mapping'] = {k.strip().strip("'").strip('"'): int(v.strip()) for k, v in label_pairs}

            if 'scale' in config_dict['MAIN']:
                scale_str = config_dict['MAIN']['scale'].strip("[]")  
                config_dict['MAIN']['scale'] = [float(item.strip()) for item in scale_str.split(",")]

        return config_dict

    @staticmethod
    def _convert_type(value):
        value_lower = value.lower()
        
        if value_lower in {"true", "false"}:
            return value_lower == "true"
        
        try:
            if "." in value or "e" in value.lower():  
                return float(value)
            return int(value) 
        except ValueError:
            pass
        
        return value


    def create_yaml(self):
        print('Create yaml file...')
        
        data = {
            "train": self.train_path,
            "val": self.val_path,
            "test": self.test_path,
            "nc": len(self.label_mapping),
            "names": list(self.label_mapping.keys())
        }
        
        path_save_yaml = os.path.join(self.path_save, 'data.yaml')
        
        with open(path_save_yaml, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
        copyfile(self.config_path, os.path.join(self.path_save, 'config.cfg'))


    def _create_augmentation_object(self):
        self.adjust_brightness_used = self.config_dict['AdjustBrightness']['used']
        self.adjust_brightness = AdjustBrightness(brightness_min=self.config_dict['AdjustBrightness']['brightness_min'],
                                                   brightness_max=self.config_dict['AdjustBrightness']['brightness_max'])
        
        self.adjust_contrast_used = self.config_dict['AdjustContrast']['used']
        self.adjust_contrast = AdjustContrast(contrast_min=self.config_dict['AdjustContrast']['contrast_min'],
                                            contrast_max=self.config_dict['AdjustContrast']['contrast_max'])
        
        self.adjust_saturation_used = self.config_dict['AdjustSaturation']['used']
        self.adjust_saturation = AdjustSaturation(saturation_min=self.config_dict['AdjustSaturation']['saturation_min'],
                                                 saturation_max=self.config_dict['AdjustSaturation']['saturation_max'])
        
        self.cutout_used = self.config_dict['Cutout']['used']
        self.cutout = Cutout(amount=self.config_dict['Cutout']['amount'])
        
        self.filters_used = self.config_dict['Filters']['used']
        self.filters = Filters()
        
        self.grid_mask_uesd = self.config_dict['GridMask']['used']
        self.grid_mask = GridMask(use_h=self.config_dict['GridMask']['use_h'],
                                  use_w=self.config_dict['GridMask']['use_w'],
                                  rotate=self.config_dict['GridMask']['rotate'],
                                  offset=self.config_dict['GridMask']['offset'],
                                  ratio=self.config_dict['GridMask']['ratio'],
                                  mode=self.config_dict['GridMask']['mode'],
                                  prob=self.config_dict['GridMask']['prob'])
        
        self.horizontal_flip_used = self.config_dict['HorizontalFlip']['used']
        self.horizontal_flip = HorizontalFlip()
        
        self.random_hsv_used = self.config_dict['RandomHSV']['used']
        self.random_hsv = RandomHSV(hue=self.config_dict['RandomHSV']['hue'],
                                    saturation=self.config_dict['RandomHSV']['saturation'],
                                    brightness=self.config_dict['RandomHSV']['brightness'])
        
        self.lighting_noise_used = self.config_dict['LightingNoise']['used']
        self.lighting_noise = LightingNoise()
        
        self.mixup_used = self.config_dict['Mixup']['used']
        self.mixup = Mixup(lambd=self.config_dict['Mixup']['lambd'])
        
        self.noisy_used = self.config_dict['Noisy']['used']
        self.noisy = Noisy(noise_type=self.config_dict['Noisy']['noise_type'])
        
        self.resize_used = self.config_dict['Resize']['used']
        self.resize = Resize(inp_dim=self.config_dict['Resize']['inp_dim'])
        
        self.rotate_only_bboxes_used = self.config_dict['RotateOnlyBboxes']['used']
        self.rotate_only_bboxes = RotateOnlyBboxes(angle=self.config_dict['RotateOnlyBboxes']['angle'])
        
        self.rotate_used = self.config_dict['Rotate']['used']
        self.rotate = Rotate(angle_min=self.config_dict['Rotate']['angle_min'],
                             angle_max=self.config_dict['Rotate']['angle_max'])
        
        self.scale_used = self.config_dict['Scale']['used']
        self.scale = Scale(scale_x_min=self.config_dict['Scale']['scale_x_min'],
                            scale_x_max=self.config_dict['Scale']['scale_x_max'],
                            scale_y_min=self.config_dict['Scale']['scale_y_min'],
                            scale_y_max=self.config_dict['Scale']['scale_y_max'])
        
        self.shear_used = self.config_dict['Shear']['used']
        self.shear = Shear(shear_min = self.config_dict['Shear']['shear_min'],
                           shear_max = self.config_dict['Shear']['shear_max'])
        
        self.small_object_augmentation_used = self.config_dict['SmallObjectAugmentation']['used']
        self.small_object_augmentation = SmallObjectAugmentation()
        
        self.translate_used = self.config_dict['Translate']['used']
        self.translate = Translate(translate_min = self.config_dict['Translate']['translate_min'],
                                   translate_max = self.config_dict['Translate']['translate_max'],
                                   diff = self.config_dict['Translate']['diff'])

        # self.sequence_used = self.config_dict['Sequence']['used']
        # self.sequence = Sequence()
    


    def split_dataset(self):
        self.type_format_label = {'yolo': 'txt', 'voc': 'xml', 'labelme': 'json'}.get(self.src_type_dataset, 'txt')
        
        filenames = os.listdir(self.path_dataset)
        
        self.type_img = filenames[0].split('.')[-1]
        
        label_filenames = [f for f in filenames if f.endswith(self.type_format_label)]
        img_filenames = [f for f in filenames if f.endswith(self.type_img)]
        
        label_stems = {f.rsplit('.', 1)[0] for f in label_filenames}
        img_with_label = [f for f in img_filenames if f.rsplit('.', 1)[0] in label_stems]
        img_without_label = [f for f in img_filenames if f.rsplit('.', 1)[0] not in label_stems]
        
        random.shuffle(img_with_label)
        random.shuffle(img_without_label)
        
        train_cut_wol = int(len(img_without_label) * self.scale_dataset[0])
        val_cut_wol = int(len(img_without_label) * sum(self.scale_dataset[:2]))
        
        train_cut_wl = int(len(img_with_label) * self.scale_dataset[0])
        val_cut_wl = int(len(img_with_label) * sum(self.scale_dataset[:2]))
        
        train_img = img_without_label[:train_cut_wol] + img_with_label[:train_cut_wl]
        val_img = img_without_label[train_cut_wol:val_cut_wol] + img_with_label[train_cut_wl:val_cut_wl]
        test_img = img_without_label[val_cut_wol:] + img_with_label[val_cut_wl:]
    
        random.shuffle(train_img)
        random.shuffle(val_img)
        random.shuffle(test_img)
        
        return train_img, val_img, test_img


    def augment_data(self, img_paths, data_set='train'):
        save_path = {
            'train': self.train_path,
            'val': self.val_path,
            'test': self.test_path
        }.get(data_set)
        if not save_path:
            print(f"Invalid dataset type: {data_set}")
            return

        img_save = os.path.join(save_path, 'images')
        label_save = os.path.join(save_path, 'labels')
        create_folder(img_save)
        create_folder(label_save)

        print(f'Processing {data_set} dataset...')
        for filename in tqdm(img_paths):
            src_img = os.path.join(self.path_dataset, filename)
            img = cv2.imread(src_img)
            src_label = src_img.replace(self.type_img, self.type_format_label)
            if self.src_type_dataset == 'yolo':
                bboxes = get_info_bbox_yolo(img, src_label)
            elif self.src_type_dataset == 'voc':
                bboxes = get_info_bbox_pascalvoc(src_label, self.label_mapping)

            
            if os.path.exists(src_label):
                save_sample(self.dest_type_dataset, img, bboxes, img_save, label_save, self.label_mapping)
                if data_set == 'test':
                    continue
                
                augmentations = [
                    (self.config_dict['AdjustBrightness']['used'], self.adjust_brightness),
                    (self.config_dict['AdjustContrast']['used'], self.adjust_contrast),
                    (self.config_dict['AdjustSaturation']['used'], self.adjust_saturation),
                    (self.config_dict['Cutout']['used'], self.cutout),
                    (self.config_dict['Filters']['used'], self.filters),
                    (self.config_dict['GridMask']['used'], self.grid_mask),
                    (self.config_dict['HorizontalFlip']['used'], self.horizontal_flip),
                    (self.config_dict['RandomHSV']['used'], self.random_hsv),
                    (self.config_dict['LightingNoise']['used'], self.lighting_noise),
                    (self.config_dict['Noisy']['used'], self.noisy),
                    (self.config_dict['Resize']['used'], self.resize),
                    (self.config_dict['RotateOnlyBboxes']['used'], self.rotate_only_bboxes),
                    (self.config_dict['Rotate']['used'], self.rotate),
                    (self.config_dict['Scale']['used'], self.scale),
                    (self.config_dict['Shear']['used'], self.shear),
                    (self.config_dict['SmallObjectAugmentation']['used'], self.small_object_augmentation),
                    (self.config_dict['Translate']['used'], self.translate)
                ]

                for used, aug_object in augmentations:
                    if used:
                        img_aug, bboxes_aug = aug_object.transform(img.copy(), bboxes.copy())
                        save_sample(self.dest_type_dataset, img_aug, bboxes_aug, img_save, label_save, self.label_mapping)
            else:
                copyfile(src_img, os.path.join(img_save, f'{random.getrandbits(128):x}.jpg'))
    

    def create(self):
        train, val, test = self.split_dataset()
        self.augment_data(train, 'train')
        self.augment_data(val, 'val')
        self.augment_data(test, 'test')
        if self.dest_type_dataset == 'yolo':
            self.create_yaml()
        print('Create augmentation dataset complete...')

if __name__ == "__main__":
    config_path = './config.cfg'
    Augmentation(config_path=config_path).create()
