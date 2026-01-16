import os
import cv2
import numpy as np
import random
import json
import sys
from pathlib import Path

# Import augmentation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentations.brightness import AdjustBrightness
from augmentations.contrast import AdjustContrast
from augmentations.saturation import AdjustSaturation
from augmentations.cutout import Cutout
from augmentations.filters import Filters
from augmentations.grid_mask import GridMask
from augmentations.horizontal_flip import HorizontalFlip
from augmentations.hsv import RandomHSV
from augmentations.lighting_noise import LightingNoise
from augmentations.noisy import Noisy
from augmentations.resize import Resize
from augmentations.rotate import Rotate
from augmentations.scale import Scale
from augmentations.sequence import Sequence
from augmentations.shear import Shear
from augmentations.translate import Translate
from utils.utils import draw_rect, get_info_bbox_yolo, get_info_bbox_pascalvoc
import xml.etree.ElementTree as ET


class AugmentationService:
    def __init__(self):
        self.augmentation_classes = {
            'brightness': {'class': AdjustBrightness, 'params': {'brightness_min': 0.8, 'brightness_max': 1.2}, 'name': 'Brightness', 'description': 'Điều chỉnh độ sáng'},
            'contrast': {'class': AdjustContrast, 'params': {'contrast_min': 0.8, 'contrast_max': 1.2}, 'name': 'Contrast', 'description': 'Điều chỉnh độ tương phản'},
            'saturation': {'class': AdjustSaturation, 'params': {'saturation_min': 0.8, 'saturation_max': 1.2}, 'name': 'Saturation', 'description': 'Điều chỉnh độ bão hòa'},
            'horizontal_flip': {'class': HorizontalFlip, 'params': {}, 'name': 'Horizontal Flip', 'description': 'Lật ảnh theo chiều ngang'},
            'hsv': {'class': RandomHSV, 'params': {}, 'name': 'Random HSV', 'description': 'Điều chỉnh HSV ngẫu nhiên'},
            'noisy': {'class': Noisy, 'params': {}, 'name': 'Noise', 'description': 'Thêm nhiễu vào ảnh'},
            'rotate': {'class': Rotate, 'params': {'angle_min': -15, 'angle_max': 15}, 'name': 'Rotate', 'description': 'Xoay ảnh'},
            'scale': {'class': Scale, 'params': {'scale_x_min': 0.8, 'scale_x_max': 1.2, 'scale_y_min': 0.8, 'scale_y_max': 1.2}, 'name': 'Scale', 'description': 'Thay đổi tỷ lệ ảnh'},
            'shear': {'class': Shear, 'params': {'shear_min': -0.2, 'shear_max': 0.2}, 'name': 'Shear', 'description': 'Biến dạng nghiêng'},
            'translate': {'class': Translate, 'params': {'translate_min': 0.1, 'translate_max': 0.2}, 'name': 'Translate', 'description': 'Dịch chuyển ảnh'},
            'cutout': {'class': Cutout, 'params': {'amount': 0.3}, 'name': 'Cutout', 'description': 'Tạo vùng che ngẫu nhiên'},
            'grid_mask': {'class': GridMask, 'params': {'prob': 0.7}, 'name': 'Grid Mask', 'description': 'Tạo lưới che'},
        }
    
    def get_available_augmentations(self):
        """Return list of available augmentations"""
        return [
            {
                'id': key,
                'name': value['name'],
                'description': value['description']
            }
            for key, value in self.augmentation_classes.items()
        ]
    
    def create_augmentation_pipeline(self, selected_augmentations):
        """Create augmentation pipeline from selected augmentations"""
        augmentations = []
        for aug_id in selected_augmentations:
            if aug_id in self.augmentation_classes:
                aug_config = self.augmentation_classes[aug_id]
                aug_obj = aug_config['class'](**aug_config['params'])
                augmentations.append(aug_obj)
        
        return Sequence(augmentations, probs=1)
    
    def create_single_augmentation(self, aug_id):
        """Create a single augmentation object"""
        if aug_id in self.augmentation_classes:
            aug_config = self.augmentation_classes[aug_id]
            return aug_config['class'](**aug_config['params'])
        return None
    
    def read_image_and_label(self, img_path, label_path, label_format):
        """Read image and label"""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        
        # Read label
        bboxes = None
        if label_path and os.path.exists(label_path):
            if label_format == 'yolo':
                bboxes = get_info_bbox_yolo(img, label_path)
            elif label_format == 'voc':
                # For VOC format, we need label mapping
                label_mapping = {'person': 0, 'car': 1, 'dog': 2, 'cat': 3}  # Default mapping
                bboxes = get_info_bbox_pascalvoc(label_path, label_mapping)
        
        if bboxes is None or len(bboxes) == 0:
            bboxes = np.array([])
        
        return img, bboxes
    
    def save_yolo_label(self, bboxes, img_shape, output_path):
        """Save bounding boxes in YOLO format"""
        h, w = img_shape[:2]
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                if len(bbox) >= 5:
                    x_min, y_min, x_max, y_max, class_id = bbox[:5]
                    # Convert to YOLO format
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
    
    def save_voc_label(self, bboxes, img_shape, output_path, img_filename):
        """Save bounding boxes in Pascal VOC format"""
        h, w = img_shape[:2]
        
        # Create XML structure
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'filename').text = img_filename
        
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(w)
        ET.SubElement(size, 'height').text = str(h)
        ET.SubElement(size, 'depth').text = str(3)
        
        for bbox in bboxes:
            if len(bbox) >= 5:
                x_min, y_min, x_max, y_max, class_id = bbox[:5]
                
                obj = ET.SubElement(annotation, 'object')
                ET.SubElement(obj, 'name').text = f'class_{int(class_id)}'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(int(x_min))
                ET.SubElement(bndbox, 'ymin').text = str(int(y_min))
                ET.SubElement(bndbox, 'xmax').text = str(int(x_max))
                ET.SubElement(bndbox, 'ymax').text = str(int(y_max))
        
        # Write to file
        tree = ET.ElementTree(annotation)
        tree.write(output_path)
    
    def generate_preview(self, task_folder, selected_augmentations, label_format):
        """Generate preview with a random sample for each augmentation"""
        images_folder = os.path.join(task_folder, 'images')
        labels_folder = os.path.join(task_folder, 'labels')
        
        # Get random image
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            return {'error': 'No images found'}
        
        random_image = random.choice(image_files)
        img_path = os.path.join(images_folder, random_image)
        
        # Find corresponding label
        base_name = os.path.splitext(random_image)[0]
        label_ext = '.txt' if label_format == 'yolo' else '.xml'
        label_path = os.path.join(labels_folder, base_name + label_ext)
        
        # Read image and label
        img, bboxes = self.read_image_and_label(img_path, label_path, label_format)
        if img is None:
            return {'error': 'Failed to read image'}
        
        # Save preview images
        preview_folder = os.path.join('outputs', 'previews')
        os.makedirs(preview_folder, exist_ok=True)
        
        preview_id = str(random.randint(100000, 999999))
        
        # Draw bboxes on original image
        img_with_bbox = draw_rect(img, bboxes, img) if len(bboxes) > 0 else img
        original_path = os.path.join(preview_folder, f'original_{preview_id}.jpg')
        cv2.imwrite(original_path, img_with_bbox)
        
        # Generate preview for each augmentation
        augmented_images = []
        for aug_id in selected_augmentations:
            aug_obj = self.create_single_augmentation(aug_id)
            if aug_obj is None:
                continue
            
            # Apply single augmentation
            if len(bboxes) > 0:
                bboxes_copy = bboxes.copy()
                aug_img, aug_bboxes = aug_obj.transform(img.copy(), bboxes_copy)
            else:
                # If no bboxes, just transform the image with empty bbox array
                aug_img = img.copy()
                aug_bboxes = np.array([]).reshape(0, 5)
                # For augmentations that don't affect bbox, still transform the image
                try:
                    aug_img, aug_bboxes = aug_obj.transform(img.copy(), np.array([]).reshape(0, 5))
                except:
                    # If augmentation fails with empty bbox, just use original image
                    aug_img = img.copy()
            
            # Draw bboxes
            aug_img_with_bbox = draw_rect(aug_img, aug_bboxes, aug_img) if len(aug_bboxes) > 0 else aug_img
            
            # Save augmented image
            aug_filename = f'aug_{aug_id}_{preview_id}.jpg'
            aug_path = os.path.join(preview_folder, aug_filename)
            cv2.imwrite(aug_path, aug_img_with_bbox)
            
            augmented_images.append({
                'augmentation_id': aug_id,
                'augmentation_name': self.augmentation_classes[aug_id]['name'],
                'image_path': f'previews/{aug_filename}',
                'bbox_count': len(aug_bboxes)
            })
        
        return {
            'original_image': f'previews/original_{preview_id}.jpg',
            'original_bbox_count': len(bboxes),
            'augmented_images': augmented_images
        }
    
    def apply_augmentations(self, task_folder, output_folder, selected_augmentations, label_format):
        """Apply each augmentation separately to all images"""
        images_folder = os.path.join(task_folder, 'images')
        labels_folder = os.path.join(task_folder, 'labels')
        
        # Create output folders
        output_images_folder = os.path.join(output_folder, 'images')
        output_labels_folder = os.path.join(output_folder, 'labels')
        os.makedirs(output_images_folder, exist_ok=True)
        os.makedirs(output_labels_folder, exist_ok=True)
        
        # Get all images
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        processed_count = 0
        
        # First, copy original images and labels
        for img_file in image_files:
            img_path = os.path.join(images_folder, img_file)
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            label_ext = '.txt' if label_format == 'yolo' else '.xml'
            label_path = os.path.join(labels_folder, base_name + label_ext)
            
            # Copy original image
            output_img_path = os.path.join(output_images_folder, f'original_{img_file}')
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(output_img_path, img)
            
            # Copy original label
            if os.path.exists(label_path):
                output_label_path = os.path.join(output_labels_folder, f'original_{base_name}{label_ext}')
                import shutil
                shutil.copy2(label_path, output_label_path)
            
            processed_count += 1
        
        # Apply each augmentation separately
        for aug_id in selected_augmentations:
            aug_obj = self.create_single_augmentation(aug_id)
            if aug_obj is None:
                continue
            
            aug_name = self.augmentation_classes[aug_id]['name'].replace(' ', '_').lower()
            
            for img_file in image_files:
                img_path = os.path.join(images_folder, img_file)
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                label_ext = '.txt' if label_format == 'yolo' else '.xml'
                label_path = os.path.join(labels_folder, base_name + label_ext)
                
                # Read image and label
                img, bboxes = self.read_image_and_label(img_path, label_path, label_format)
                if img is None:
                    continue
                
                # Apply augmentation
                if len(bboxes) > 0:
                    bboxes_copy = bboxes.copy()
                    aug_img, aug_bboxes = aug_obj.transform(img.copy(), bboxes_copy)
                else:
                    # If no bboxes, just transform the image
                    aug_img = img.copy()
                    aug_bboxes = np.array([]).reshape(0, 5)
                    try:
                        aug_img, aug_bboxes = aug_obj.transform(img.copy(), np.array([]).reshape(0, 5))
                    except:
                        # If augmentation fails with empty bbox, just use original image
                        aug_img = img.copy()
                
                # Save augmented image with augmentation name prefix
                output_img_name = f'{aug_name}_{base_name}{ext}'
                output_img_path = os.path.join(output_images_folder, output_img_name)
                cv2.imwrite(output_img_path, aug_img)
                
                # Save augmented label
                output_label_name = f'{aug_name}_{base_name}{label_ext}'
                output_label_path = os.path.join(output_labels_folder, output_label_name)
                if label_format == 'yolo':
                    self.save_yolo_label(aug_bboxes, aug_img.shape, output_label_path)
                else:
                    self.save_voc_label(aug_bboxes, aug_img.shape, output_label_path, output_img_name)
                
                processed_count += 1
        
        return {
            'processed_count': processed_count,
            'output_folder': output_folder,
            'original_count': len(image_files),
            'augmented_count': len(image_files) * len(selected_augmentations),
            'total_count': len(image_files) + len(image_files) * len(selected_augmentations)
        }
