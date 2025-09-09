import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu

origin_img_size = 512
input_img_size = 480

class LiverDataSet(BaseDataset):
    
    ALL_CLASSES = ['background', 'lesion']
    
    def __init__(
            self, 
            images_dir, 
            origin_mask_dir,
            prior_mask_dir,
            post_mask_dir,
            image_ids_txt,
            masks_count, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        # self.ids = os.listdir(images_dir)
        fp = open(image_ids_txt, 'r')
        self.ids = fp.readlines()
        fp.close()
        
        random.shuffle(self.ids)
        
        self.images_fps = [os.path.join(images_dir, image_id.replace('\n', '') + '.png') for image_id in self.ids]
        
        self.masks_count = masks_count
        self.origin_masks_fps = [os.path.join(origin_mask_dir, image_id.replace('\n', '') + '.png') for image_id in self.ids]
        # self.prior_masks_fps = [os.path.join(prior_mask_dir, image_id.replace('\n', '') + '.png') for image_id in self.ids]
        self.post_masks_fps = [os.path.join(post_mask_dir, image_id.replace('\n', '') + '.png') for image_id in self.ids]

        self.prior_mask_dir = prior_mask_dir
        
        # convert str names to class values on masks
        self.class_values = [self.ALL_CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        rescale_len = origin_img_size
        
        # print (self.images_fps[i])
        
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (rescale_len, rescale_len), interpolation = cv2.INTER_LINEAR)
        
        mask = np.zeros((rescale_len, rescale_len, self.masks_count), dtype='float')
        
        origin_mask = cv2.imread(self.origin_masks_fps[i], 0)
        origin_mask = cv2.resize(origin_mask, (rescale_len, rescale_len), interpolation = cv2.INTER_NEAREST)
        
        prior_mask = np.zeros(origin_mask.shape, dtype='float')
        prior_mask_fp = os.path.join(self.prior_mask_dir, self.ids[i].replace('\n', '') + '.png')
        if os.path.exists(prior_mask_fp):
            prior_mask = cv2.imread(prior_mask_fp, 0)
            # print (prior_masks_fp)
        prior_mask = cv2.resize(prior_mask, (rescale_len, rescale_len), interpolation = cv2.INTER_LINEAR)
        
        post_mask = np.zeros(origin_mask.shape, dtype='float')
        if os.path.exists(self.post_masks_fps[i]):
            post_mask = cv2.imread(self.post_masks_fps[i], 0)
        post_mask = cv2.resize(post_mask, (rescale_len, rescale_len), interpolation = cv2.INTER_LINEAR)
        
        
        mask[:, :, 0] = origin_mask
        # mask[:, :, 0] = origin_mask * 0.8 ##test one-hot LS
        mask[:, :, 1] = prior_mask
        mask[:, :, 2] = post_mask
        mask = mask / 255.0

        # print ('1. ', mask.shape, type(mask))
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        pos = self.images_fps[i].rfind('/')
        image_id = self.images_fps[i][pos+1:-4]
        
        # print (image_id)    
        
        return image, mask, image_id
        
    def __len__(self):
        return len(self.ids)

    def set_prior_mask_path(self, input_path):
        self.prior_mask_dir = input_path
        print ('set prior mask dir: ', self.prior_mask_dir)
    
    
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=input_img_size, min_width=input_img_size, always_apply=True, border_mode=0),
        albu.RandomCrop(height=input_img_size, width=input_img_size, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)