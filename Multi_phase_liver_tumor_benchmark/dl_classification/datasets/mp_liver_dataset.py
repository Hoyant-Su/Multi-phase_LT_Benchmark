import torch
import numpy as np
from functools import partial
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler
try:
    from datasets.transforms import *
except:
    from transforms import *

import csv
import numpy as np
import sys
import models.my_globals



class MultiPhaseLiverDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.size = args.img_size
        self.is_training = is_training
        img_list = []
        lab_list = []
        phase_list = ['pvp', 'art', 'nc', 'delay']
        #phase_list = ['pvp', 'art', 'nc']
        #phase_list = ['pvp']

        if is_training:
            anno = np.loadtxt(args.train_anno_file, dtype=np.str_)
        else:
            anno = np.loadtxt(args.val_anno_file, dtype=np.str_)

        for item in anno:
            mp_img_list = []
            for phase in phase_list:
                mp_img_list.append(f'{args.data_dir}/{item[0]}_{phase}.nii.gz')

            img_list.append(mp_img_list)
            lab_list.append(item[1])

        self.img_list = img_list
        self.lab_list = lab_list


    def __getitem__(self, index):
        args = self.args
        image = self.load_mp_images(self.img_list[index])
        if self.is_training:
            image = self.transforms(image, args.train_transform_list)
        else:
            image = self.transforms(image, args.val_transform_list)
        image = image.copy()
        label = int(self.lab_list[index])
        return (image, label)

    def load_feats(self, meta_feat_item):
        meta_feat = []
        with open(f'{self.args.feat_csv_dir}', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[0] == meta_feat_item:
                    meta_feat.append(row[1:-1])
        return meta_feat


    def load_mp_images(self, mp_img_list):
        mp_image = []
        for img in mp_img_list:
            image = load_nii_file(img)
            image = resize3D(image, self.size)

            image = image_normalization(image, win = [-100, 300])
            mp_image.append(image[None, ...])
        mp_image = np.concatenate(mp_image, axis=0)
        return mp_image

    def transforms(self, mp_image, transform_list):
        args = self.args
        if 'center_crop' in transform_list:
            mp_image = center_crop(mp_image, args.crop_size)
        # if 'random_crop' in transform_list:
        #     mp_image = random_crop(mp_image, args.crop_size)
        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)
        if 'image_net_autoaugment' in transform_list:
           mp_image = image_net_autoaugment(mp_image)
        return mp_image

    def __len__(self):
        return len(self.img_list)

def create_loader(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
):

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader
