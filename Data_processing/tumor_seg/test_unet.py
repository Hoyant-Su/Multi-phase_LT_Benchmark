import os
from os.path import join, exists

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import numpy as np
from Data_processing.tumor_seg.maskvis import draw_mask
import SimpleITK as sitk
import scipy

import time
from medpy import metric

import segmentation_models_pytorch as smp
from Data_processing.tumor_seg.liver import LiverDataSet, get_preprocessing, get_training_augmentation
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils as smp_utils

import argparse

input_img_size = 480

def cal_dice(pred, gt, epsilon=1e-6):
    d = -1
    if np.sum(gt) > 0:
        inter = np.sum((gt) & (pred))
        sets_sum = np.sum(gt.reshape(-1)) + np.sum(pred.reshape(-1))
        if sets_sum == 0:
            sets_sum = 2*inter
            print("tn mask......")
        d = (2 * inter + epsilon) / (sets_sum + epsilon)
    return d

def cal_iou(pred, gt, epsilon=1e-6):
    d = -1
    if np.sum(gt) > 0:
        inter = np.sum((gt) & (pred))
        union = np.sum((gt) | (pred))
        d = (inter + epsilon) / (union + epsilon)
    return d

### compute metric according to transunet code
def calculate_metric_percase(pred, gt, spacing, metric_list):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = 0
        if 'dice' in metric_list:
            dice = metric.binary.dc(pred, gt)
        
        iou = 0
        if 'iou' in metric_list:
            iou = metric.binary.jc(pred, gt)
        
        hd95 = 0
        if 'hd95' in metric_list:
            hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        
        return dice, iou, hd95
    elif gt.sum()==0:
        return 1, 1, 0
    else:
        return 0, 0, 0


def pred_one_image(ori_img, seg_model, input_img_size=480):
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rez_img = cv2.resize(ori_img, (input_img_size, input_img_size), interpolation = cv2.INTER_LINEAR)
    image = cv2.cvtColor(rez_img, cv2.COLOR_BGR2RGB)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    max_pixel_value = 255.0
    r, g, b = cv2.split(image)
    r = (r - mean[0]*max_pixel_value) / (std[0]*max_pixel_value)
    g = (g - mean[1]*max_pixel_value) / (std[1]*max_pixel_value)
    b = (b - mean[2]*max_pixel_value) / (std[2]*max_pixel_value)
    
    input_image = cv2.merge((r, g, b))
    
    input_image = input_image.transpose(2, 0, 1).astype('float32')
    
    x_tensor = torch.from_numpy(input_image).to(DEVICE).unsqueeze(0)
    pr_map = seg_model.predict(x_tensor)
    pr_map = pr_map.squeeze().cpu().numpy()
    pr_mask = pr_map.round().astype('uint8')
    
    # print (ori_img.shape[0], ori_img.shape[1])
    pred_res_prob = cv2.resize(pr_map, (ori_img.shape[1], ori_img.shape[0]))
    pred_res_mask = cv2.resize(pr_mask, (ori_img.shape[1], ori_img.shape[0]), interpolation = cv2.INTER_NEAREST)  
    
    return pred_res_mask, pred_res_prob

def do_tumorseg_one_slice(slice_image, liver_mask, tumor_seg_model):
    
    pred_tumor_slice_mask = np.zeros((slice_image.shape[0], slice_image.shape[1]), dtype='int')
    pred_tumor_slice_prob = np.zeros((slice_image.shape[0], slice_image.shape[1]), dtype='float')
    
    if np.sum(liver_mask) > 30:    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        liver_mask = cv2.dilate(liver_mask.astype('uint8'), kernel)

        slice_image = cv2.add(slice_image.astype('uint8'), np.zeros(np.shape(slice_image), dtype=np.uint8), mask=liver_mask.astype('uint8'))

        pred_tumor_slice_mask, pred_tumor_slice_prob = pred_one_image(slice_image, tumor_seg_model)

        # print (slice_image.shape, pred_tumor_slice_mask.shape)
        
    return pred_tumor_slice_mask, pred_tumor_slice_prob

def do_tumorseg_one_volume(volume_path, segmentation_path, proc_config, tumor_seg_model, eval_metrics=[], dataset_type='LiTS'):
    
    # print ('start process: ', volume_path)

    # print (proc_config)
    
    tumor_dice_score = 0
    tumor_iou_score = 0
    
    volume = sitk.ReadImage(volume_path)
    volume_data = sitk.GetArrayFromImage(volume)
    
    segmentation = sitk.ReadImage(segmentation_path)
    gt_seg_map = sitk.GetArrayFromImage(segmentation)

    spacing = volume.GetSpacing()

    if dataset_type == 'KiTS':
        volume_data = volume_data.transpose(2,1,0)
        gt_seg_map = gt_seg_map.transpose(2,1,0)
        spacing = [spacing[2], spacing[1], spacing[0]]
        
    # print('shape info: ', volume_data.shape, gt_seg_map.shape)
    # print ('spacing info: ', spacing)
    zoom_flag = False
    norm_spacing_z = proc_config['norm_spacing_z']
    zoom_scale = spacing[2] / norm_spacing_z
    zoom_volume_data = None
    if spacing[2] > norm_spacing_z:
        zoom_flag = True
        zoom_volume_data = scipy.ndimage.zoom(volume_data, [zoom_scale, 1, 1])
    
    ### 腹部软组织窗HU范围
    min_HU = proc_config['min_HU']
    max_HU = proc_config['max_HU']
    ## 腹部软组织窗HU值变换
    volume_data = 255.0 * (volume_data - min_HU) / (max_HU - min_HU)
    volume_data[volume_data <= 0 ] = 0
    volume_data[volume_data >= 255 ] = 255
    
    if zoom_flag:
        zoom_volume_data = 255.0 * (zoom_volume_data - min_HU) / (max_HU - min_HU)
        zoom_volume_data[zoom_volume_data <= 0 ] = 0
        zoom_volume_data[zoom_volume_data >= 255 ] = 255
    
    ## organ+tumor -> organ
    gt_organ = gt_seg_map.copy()
    gt_organ[gt_organ >= 1] = 1
    
    gt_tumor = gt_seg_map.copy()
    gt_tumor[gt_tumor == 1] = 0
    gt_tumor[gt_tumor == 2] = 1
    
    pred_tumor_seg_map = np.zeros_like(gt_tumor)
    pred_tumor_seg_prob = np.zeros(gt_tumor.shape, dtype='float')
    
    for i in range(volume_data.shape[0]):
        # print ('process slice ', i)
        
        current_slice_data = volume_data[i,:,:]
        
        former_slice_data = None
        next_slice_data = None
        
        if not zoom_flag:
            former_slice_index = i-1
            next_slice_index = i+1
            # former_slice_index = round(i - 1.0 / zoom_scale)
            # next_slice_index = round(i - 1.0 / zoom_scale)

            if former_slice_index < 0:
                former_slice_index = 0
            if next_slice_index > volume_data.shape[0] - 1:
                next_slice_index = volume_data.shape[0] - 1
                
            former_slice_data = volume_data[former_slice_index,:,:]
            next_slice_data = volume_data[next_slice_index,:,:]
        else:
            former_slice_index = round(i*zoom_scale - 1)
            next_slice_index = round(i*zoom_scale + 1)
    
            if former_slice_index < 0:
                former_slice_index = 0
            if next_slice_index > zoom_volume_data.shape[0] - 1:
                next_slice_index = zoom_volume_data.shape[0] - 1
                
            former_slice_data = zoom_volume_data[former_slice_index,:,:]
            next_slice_data = zoom_volume_data[next_slice_index,:,:]
            
        img_3c = cv2.merge((former_slice_data, current_slice_data, next_slice_data))

        # print (img_3c.shape)
        
        #### predict tumor mask via tumor seg model on the basis of organ mask (gt or pred)
        organ_slice_mask = gt_organ[i,:,:]
        pred_tumor_slice_mask, pred_tumor_slice_prob = do_tumorseg_one_slice(img_3c, organ_slice_mask, tumor_seg_model)

        # print (pred_tumor_slice_mask.shape)

        pred_tumor_seg_map[i,:,:] = pred_tumor_slice_mask
        pred_tumor_seg_prob[i,:,:] = pred_tumor_slice_prob  

    tumor_dice_score, tumor_iou_score, tumor_hd95_score = calculate_metric_percase(pred_tumor_seg_map.astype('uint8'), gt_tumor.astype('uint8'), spacing, eval_metrics)
    
    return tumor_dice_score, tumor_iou_score, tumor_hd95_score, pred_tumor_seg_prob


def parse_args():
    parser = argparse.ArgumentParser(description='Test a liver segmentation model')
    parser.add_argument('--model', default=None)
    parser.add_argument('--dataset', default='LiTS')            ### LiTS / KiTS
    
    args = parser.parse_args()
    
    return args

def test():    
    
    args = parse_args()  
    print(args)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### load new train model
    model = torch.load(args.model, DEVICE)

    proc_config = {}
    proc_config['min_HU'] = -100
    proc_config['max_HU'] = 300
    proc_config['norm_spacing_z'] = 2.5


    ###LiTS / KiTS
    dataset_type = args.dataset
    data_root_dir = '/mnt/data/smart_health_02/wanglilong/data/'

    volume_dir = os.path.join(data_root_dir, dataset_type + '/volumes/')
    segmentation_dir = os.path.join(data_root_dir, dataset_type + '/labels/')
   
    volume_list_txt = os.path.join(data_root_dir, dataset_type + '/test_vol.txt')
    fp = open(volume_list_txt, 'r')
    volume_list = fp.readlines()
    print (len(volume_list), volume_list[0])
    fp.close()
    
    mean_dice = 0
    mean_iou = 0
    mean_hd95 = 0
    nCount = 0
    
    avg_runtime = 0
    
    for item in volume_list:
        volume_path = join(volume_dir, item.replace('\n', '') + '.nii.gz')
        segmentation_path = join(segmentation_dir, item.replace('\n', '').replace('volume', 'segmentation') + '.nii.gz')     ###LiTS

        t0 = time.time()
        dice_score, iou_score, hd95_score, pred_prob = do_tumorseg_one_volume(volume_path, segmentation_path, proc_config, model, eval_metrics=['dice', 'iou', 'hd95'], dataset_type=dataset_type)
        pred_time = time.time() - t0        
        
        print (item.replace('\n', ''), 'dice = {:.3f}, iou = {:.3f}, hd95 = {:.3f}, time = {:.3f}s'.format(dice_score, iou_score, hd95_score, pred_time))

        if hd95_score >= 180:
            hd95_score = 180
        
        avg_runtime += pred_time
        
        mean_dice += dice_score
        mean_iou += iou_score
        mean_hd95 += hd95_score
        nCount += 1
    
    mean_dice /= nCount
    mean_iou /= nCount
    mean_hd95 /= nCount
    
    avg_runtime /= len(volume_list)
    
    print ('nCount = {}, mean_dice = {:.2f}, mean_iou = {:.2f}， mean_hd95 = {:.2f}， avg_runtime = {:.2f}'.format(nCount, mean_dice*100, mean_iou*100, mean_hd95, avg_runtime))

