# Data Augmentation Tool 🚀

This repository contains a Python tool for data augmentation of image datasets in YOLO or Pascal VOC format. The tool offers various augmentation techniques to create augmented datasets for training computer vision models.

## Installation 🛠️

1. Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/data-augmentation-tool.git
```

2. Change into the repository directory:
```bash
pip install -r requirements.txt
```

## Usage 📋

To use the data augmentation tool, follow these steps:

1. Prepare your dataset in either YOLO or Pascal VOC format. Ensure that your dataset images and label files are placed in the appropriate folders.

2. Open the `config.py` file and configure the augmentation settings and dataset paths according to your requirements.

3. Run the data augmentation script by executing the `main.py` file with the appropriate command-line arguments:

```bash
python main.py --path_raw /path/to/dataset --path_save /path/to/save --train_scale 0.6 --val_scale 0.2 --src_type_dataset voc --dest_type_dataset yolo
```


The arguments are as follows:
- `path_raw`: Path to the raw dataset directory.
- `path_save`: Path to save the augmented dataset.
- `train_scale`: Proportion of the dataset to be used for training.
- `val_scale`: Proportion of the dataset to be used for validation.
- `src_type_dataset`: Source data format, either 'voc' (Pascal VOC) or 'yolo' (YOLO).
- `dest_type_dataset`: Destination data format, either 'voc' (Pascal VOC) or 'yolo' (YOLO).

The tool will split the dataset into training, validation, and testing sets based on the specified proportions, apply the selected augmentation techniques, and save the augmented data in the desired format.

## Augmentation Techniques 🌟

The tool supports various augmentation techniques, which can be configured in the `config.py` file. The available techniques include:
- AdjustBrightness
- AdjustContrast
- AdjustSaturation
- Cutout
- Filters
- GridMask
- HorizontalFlip
- RandomHorizontalFlip
- RandomHSV
- LightingNoise
- Mixup
- Noisy
- Resize
- RotateOnlyBboxes
- RandomRotate
- Rotate
- RandomScale
- Scale
- RandomShear
- Shear
- SmallObjectAugmentation
- RandomTranslate
- Sequence

## Output 📁

The augmented dataset will be saved in the specified `path_save` directory in the format specified by `dest_type_dataset`. The augmented images will be stored in the 'images' folder, and the corresponding label files will be stored in the 'labels' folder.

Additionally, a `data.yaml` file will be generated in the `path_save` directory, containing information about the dataset, such as the paths to the training and validation sets, the number of classes, and the class names.

## License 📜

This tool is released under the MIT License. See [LICENSE](LICENSE) for details.

Please feel free to use and modify this tool for your data augmentation needs. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Happy data augmenting! 😊
