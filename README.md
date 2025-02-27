# Data Augmentation for Object Detection

This project provides a comprehensive data augmentation pipeline for object detection tasks. It supports multiple augmentation techniques and formats, enabling better dataset preparation.

## Features
- Supports various augmentation techniques like brightness adjustment, contrast enhancement, flipping, scaling, and more.
- Compatible with different dataset formats such as YOLO and Pascal VOC.
- Automatically splits the dataset into training, validation, and test sets based on user-defined proportions.
- Saves augmented images and labels in the desired format.
- Generates a `data.yaml` file for YOLO dataset compatibility.
- Stores configuration files alongside augmented data for reproducibility.

## Requirements
Ensure you have the required dependencies installed:
```bash
pip install opencv-python tqdm pyyaml
```

## Usage

### 1. Configuration
Modify the `config.cfg` file to specify dataset paths, augmentation settings, and dataset parameters.

### 2. Run the Augmentation Pipeline
Use the following command:
```bash
python augmentation.py --config_path ./config.cfg
```

### 3. Output Structure
After running the script, the augmented dataset will be saved in the specified output directory:
```
data/
  ├── processed/
  │   ├── train/
  │   │   ├── images/
  │   │   ├── labels/
  │   ├── val/
  │   │   ├── images/
  │   │   ├── labels/
  │   ├── test/
  │   │   ├── images/
  │   │   ├── labels/
  │   ├── data.yaml (for YOLO format)
  │   ├── config.cfg (backup of configuration file)
```

## Configuration File (`config.cfg`)
Example of `config.cfg`:
```
[MAIN]
path_dataset = ./data/raw
path_save = ./data/processed
src_type_dataset = voc
dest_type_dataset = yolo
label_mapping = ['bike':0, 'dog':1, 'car':2]
scale = [0.6, 0.2, 0.2]
```

## Customizing Augmentation
Modify the `config.cfg` file to enable/disable augmentation techniques or adjust parameters as needed.

## License
This project is licensed under the MIT License.

## Author
Developed by [quanglv1996]

