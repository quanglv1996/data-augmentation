# Augmentation for Object Detection

This project provides a data augmentation pipeline for object detection tasks, supporting various augmentation techniques to enhance training datasets. The implementation allows dataset splitting, augmentation, and saving in different formats.

## Features
- Supports multiple augmentation techniques such as brightness adjustment, contrast enhancement, flipping, scaling, and more.
- Compatible with different dataset formats like YOLO and Pascal VOC.
- Automatically splits the dataset into training, validation, and test sets.
- Saves augmented images and labels efficiently.
- Generates a YAML file for YOLO dataset format.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install opencv-python tqdm pyyaml
```

## Usage

### 1. Configuration
Modify the `config.cfg` file to specify dataset paths and augmentation parameters.

### 2. Run the Augmentation Pipeline
Execute the following command:
```bash
python augmentation.py
```

### 3. Output Structure
The augmented dataset will be saved in the specified output directory:
```
output/
  ├── train/
  │   ├── images/
  │   ├── labels/
  ├── val/
  │   ├── images/
  │   ├── labels/
  ├── test/
  │   ├── images/
  │   ├── labels/
  ├── data.yaml (for YOLO format)
```

## Configuration File (config.cfg)
Example of `config.cfg`:
```
[MAIN]
path_dataset = path/to/dataset
path_save = path/to/output
src_type_dataset = yolo
dest_type_dataset = yolo
```

## Customizing Augmentation
Modify `_create_augmentation_object` in `augmentation.py` to add or remove augmentation methods.

## License
This project is licensed under the MIT License.

## Author
Developed by [quanglv1996]

