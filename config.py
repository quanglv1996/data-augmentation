class Map(dict):
    """
    A custom dictionary class that allows accessing dictionary keys as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


'''
Config for augmentation object detection
'''
# Configuration settings for various data augmentation techniques
config_augmentation = {
    'AdjustBrightneess': {
        'used': True,
        'brightness_factor': 1.5,
    },
    'AdjustContrast': {
        'used': True,
        'contrast_factor': 1.5,
    },
    'AdjustSaturation': {
        'used': True,
        'saturation_factor': 1.5,
    },
    'Cutout': {
        'used': False,
        'amount': .3,
    },
    'Filters': {
        'used': True
    },
    'GridMask': {
        'used': False,
        'use_h': True,
        'use_w': True,
        'rotate': 1,
        'offset': False,
        'ratio': .5,
        'mode': 0,
        'prob': .7
    },
    'RandomHorizontalFlip': {
        'used': True,
        'p': 1
    },
    # Other augmentation techniques and their configurations...
}

# Convert the augmentation configuration to a Map object
config_augmentation = Map(config_augmentation)

'''
Config for generation data
'''
# Configuration settings for generating data
config_data = {
    'path_data_raw': 'D:/data-augmentation-for-object-detection/dataset_pascalvoc',
    'path_save': 'D:/data-augmentation-for-object-detection/dataset',
    'label_mapping': {
        'disc': 0,
        'adapter': 1,
        'guide': 2,
        'qr': 3,
        'gun': 4,
        'boom': 5,
        'head': 6,
    },
    'train_scale': 0.7,
    'val_scale': 0.2,
    'src_type_dataset': 'voc',
    'dest_type_dataset': 'yolo',
}

# Convert the data generation configuration to a Map object
config_data = Map(config_data)