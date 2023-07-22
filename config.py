class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
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
config_augmentation = {
    'AdjustBrightneess':{
        'used': True,
        'brightness_factor': 1.5,
    },
    'AdjustContrast':{
        'used': True,
        'contrast_factor': 1.5,
    },
    'AdjustSaturation':{
        'used': True,
        'saturation_factor': 1.5,
    },
    'Cutout':{
        'used': False,
        'amount': .3,
    },
    'Filters':{
        'used': True
    },
    'GridMask':{
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
        'p': 1},
    'HorizontalFlip':{
        'used': False
    },
    'RandomHSV': {
        'used': True,
        'hue':100,
        'saturation':100,
        'brightness': 100
    },
    'LightingNoise':{
        'used': True,
    },
    'Mixup':{
        'used': False,
        'lambd': .3,
    },
    'Noisy':{
        'used': True,
        'noise_type': "gauss",
    },
    'Resize':{
        'used': True,
        'inp_dim': 512,
    },
    'RotateOnlyBboxes':{
        'used': False,
        'angle': 5.,
    },
    'RandomRotate':{
        'used': True,
        'angle': 5.,
    },
    'Rotate':{
        'used': False,
        'angle': 5,
    },
    'RandomScale': {
        'used': True,
        'scale': 0.5,
        'diff': True
    },
    'Scale': {
        'used': False,
        'scale_x': 0.5,
        'scale_y': True
    },
    'Sequence': {
        'used': True
    },
    'RandomShear': {
        'used': True,
        'shear_factor': 0.5
    },
    'Shear': {
        'used': False,
        'shear_factor': 0.5
    },
    'SmallObjectAugmentation': {
        'used': False,
    },
    'RandomTranslate': {
        'used': True,
        'translate': 0.2,
        'diff': True
    },
    'Translate': {
        'used': False,
        'translate': 0.2,
        'diff': True
    },
}
config_augmentation = Map(config_augmentation)

'''
Config for generation data
'''
config_data = {
    'path_data_raw': 'D:/data-augmentation-for-object-detection/dataset_pascalvoc',
    'path_save': 'D:/data-augmentation-for-object-detection/dataset',
    'label_mapping': {
        'disc': 0,
        'adapter':1,
        'guide':2,
        'qr':3,
        'gun':4,
        'boom': 5,
        'head': 6,
    },
    'train_scale': 0.7,
    'val_scale': 0.2,
    'src_type_dataset': 'voc',
    'dest_type_dataset': 'yolo',
    
}
config_data = Map(config_data)
