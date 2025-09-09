import os
from os.path import join, exists

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import segmentation_models_pytorch as smp
import SimpleITK as sitk
import scipy
from Data_processing.tumor_seg.test_unet import pred_one_image, do_tumorseg_one_slice

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def test_one_volume_tumorseg(volume_path, liver_seg_model, tumor_seg_model, label_value = 1, save_segment_path = None):
    print('start process: ', volume_path)

    volume = sitk.ReadImage(volume_path)
    volume_data = sitk.GetArrayFromImage(volume)

    print('shape info: ', volume_data.shape)

    spacing = volume.GetSpacing()
    NORM_SPACING_Z = 2.5
    zoom_flag = False
    zoom_scale = spacing[2] / NORM_SPACING_Z
    print('spacing info: ', spacing, ', zoom_scale = ', zoom_scale)

    zoom_volume_data = None
    if spacing[2] > NORM_SPACING_Z:
        zoom_flag = True
        zoom_volume_data = scipy.ndimage.zoom(volume_data, [zoom_scale, 1, 1])
        print('zoom_volume_data shape: ', zoom_volume_data.shape)

    min_HU = -100.0
    max_HU = 300.0
    print('min_HU = ', min_HU, ', max_HU = ', max_HU)

    volume_data = 255.0 * (volume_data - min_HU) / (max_HU - min_HU)
    volume_data[volume_data <= 0] = 0
    volume_data[volume_data >= 255] = 255

    if zoom_flag:
        zoom_volume_data = 255.0 * (zoom_volume_data - min_HU) / (max_HU - min_HU)
        zoom_volume_data[zoom_volume_data <= 0] = 0
        zoom_volume_data[zoom_volume_data >= 255] = 255

    pred_tumor_seg_map = np.zeros_like(volume_data)

    for i in range(volume_data.shape[0]):
        print('process slice ', i)

        current_slice_data = volume_data[i, :, :]
        slice_3c = cv2.merge((current_slice_data, current_slice_data, current_slice_data))

        former_slice_data = None
        next_slice_data = None

        if not zoom_flag:
            former_slice_index = round(i - 1.0 / zoom_scale)
            next_slice_index = round(i + 1.0 / zoom_scale)

            if former_slice_index < 0:
                former_slice_index = 0
            if next_slice_index > volume_data.shape[0] - 1:
                next_slice_index = volume_data.shape[0] - 1

            former_slice_data = volume_data[former_slice_index, :, :]
            next_slice_data = volume_data[next_slice_index, :, :]
        else:
            former_slice_index = round(i * zoom_scale - 1)
            next_slice_index = round(i * zoom_scale + 1)

            if former_slice_index < 0:
                former_slice_index = 0
            if next_slice_index > zoom_volume_data.shape[0] - 1:
                next_slice_index = zoom_volume_data.shape[0] - 1

            former_slice_data = zoom_volume_data[former_slice_index, :, :]
            next_slice_data = zoom_volume_data[next_slice_index, :, :]

        img_3c = cv2.merge((former_slice_data, current_slice_data, next_slice_data))

        #### 1. predict liver mask via liver seg model
        pred_liver_slice_mask, pred_liver_slice_prob = pred_one_image(img_3c.copy().astype('float32'), liver_seg_model, 320)

        #### 2. predict tumor mask via tumor seg model on the basis of predicted liver mask
        pred_tumor_slice_mask, pred_tumor_slice_prob = do_tumorseg_one_slice(img_3c, pred_liver_slice_mask, tumor_seg_model)
        # print (pred_tumor_slice_mask.shape, pred_tumor_seg_map.shape)
        pred_tumor_seg_map[i, :, :] = pred_tumor_slice_mask

    if save_segment_path is not None:
        pred_tumor_seg_map *= label_value
        segment = sitk.GetImageFromArray(pred_tumor_seg_map.astype('uint8'))
        segment.SetSpacing(volume.GetSpacing())
        segment.SetOrigin(volume.GetOrigin())
        segment.SetDirection(volume.GetDirection())
        sitk.WriteImage(segment, save_segment_path)



if __name__ == '__main__':
    print('start process xiehe data ...')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    liver_seg_model = torch.load('Data_processing/pre_processing_weights/lits_liver_mitb0/epoch_17_iou_0.922.pth', DEVICE)
    tumor_seg_model = torch.load('Data_processing/pre_processing_weights/lits_tumor/epoch_35_iou_0.627.pth', DEVICE)

    data_dir = "" #fill in the blank
    mask_dir = "" #fill in the blank

    label_value = 1

    case_folds = os.listdir(data_dir)
    for fold in case_folds:
        if '230218c2' not in fold: #230408a10, 230218c2
            continue
        print ('start process ', fold)
        case_dir = os.path.join(data_dir, fold)
        nii_files = os.listdir(case_dir)

        case_mask_dir = os.path.join(mask_dir, fold)
        
        for one_file in nii_files:
            if 'pvp' in one_file or 'PVP' in one_file:
                volume_path = os.path.join(case_dir, one_file)       
                make_dir(case_mask_dir)
                save_segment_path = os.path.join(case_mask_dir, 'mask_' + one_file)
                test_one_volume_tumorseg(volume_path, liver_seg_model, tumor_seg_model, label_value, save_segment_path)
