import os
import shutil

import numpy as np
import csv
import random
import SimpleITK as sitk
from radiomics import featureextractor

# File which contains the extraction parameters for Pyradiomics
params_file = 'Multi_phase_liver_tumor_benchmark/radiomics_extract_classification/src/feature_extraction_parameters.yaml'
# Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def write_csv(input_info_set, save_file):
    with open(save_file, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['image', 'mask'])
        for item in input_info_set:
            csv_writer.writerow([item['image'], item['mask']])

def read_csv(list_file):
    dict_list = []
    with open(list_file) as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        for row in csv_reader:
            # print (row)
            hcp_info_dict = dict()
            hcp_info_dict['image'] = row[0]
            hcp_info_dict['mask'] = row[1]
            dict_list.append(hcp_info_dict)

    return dict_list

def crop_tumor_patch(image_file, mask_file, tumor_dir, case_id):
    mask = sitk.ReadImage(mask_file)
    mask_data = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()

    image = sitk.ReadImage(image_file)
    image_data = sitk.GetArrayFromImage(image)

    print (mask_data.shape, spacing)

    if mask_data.shape != image_data.shape:
        print ('mask and image shape mismatch')

    ### find tumors, give up too small / single slice
    input_mask = sitk.GetImageFromArray(mask_data)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(input_mask)
    num_connected_regions = cc_filter.GetObjectCount()

    tumor_count = 0
    print ('num_connected_regions = ', num_connected_regions)
    if num_connected_regions >= 1:
        lss_filter = sitk.LabelShapeStatisticsImageFilter()
        lss_filter.Execute(output_mask)
        np_output_mask = sitk.GetArrayFromImage(output_mask)
        for idx in range(1, num_connected_regions + 1):
            area = lss_filter.GetNumberOfPixels(idx)
            # print (idx, area)
            if area <= 50:
                print('this tumor is too small ...')
                continue

            tumor_mask_data = np.zeros_like(np_output_mask)
            tumor_mask_data[np_output_mask == idx] = 1

            lesion_3d_pts = np.nonzero(tumor_mask_data)
            min_z = np.min(lesion_3d_pts[0])
            max_z = np.max(lesion_3d_pts[0])
            center_z = int((min_z + max_z) / 2)
            min_y = np.min(lesion_3d_pts[1])
            max_y = np.max(lesion_3d_pts[1])
            center_y = int((min_y + max_y) / 2)
            min_x = np.min(lesion_3d_pts[2])
            max_x = np.max(lesion_3d_pts[2])
            center_x = int((min_x + max_x) / 2)
            print('min z = {}, max z = {}, min y = {}, max y = {}, min x = {}, max x = {}'.format(min_z, max_z, min_y, max_y, min_x, max_x))

            if min_z == max_z:
                print ('this tumor only contain one slice ...')
                continue

            tumor_mask = sitk.GetImageFromArray(tumor_mask_data)
            tumor_mask.SetSpacing(mask.GetSpacing())
            tumor_mask.SetOrigin(mask.GetOrigin())
            tumor_mask.SetDirection(mask.GetDirection())
            save_tumor_mask_dir = os.path.join(tumor_dir, case_id)
            make_dir(save_tumor_mask_dir)
            sitk.WriteImage(tumor_mask, os.path.join(save_tumor_mask_dir, 'tumor_{}.nii.gz'.format(tumor_count)))

            tumor_count += 1

    return tumor_count

def prepare_tumor_data():
    image_dir = "" ##fill in the blank
    mask_dir = "" ##fill in the blank
    tumor_dir = "" ##fill in the blank

    total_tumor_count = 0
    meta_tumor_count = 0
    case_set = os.listdir(mask_dir)
    for case in case_set:
        case_dir = os.path.join(mask_dir, case)
        mask_name = os.listdir(case_dir)[0]
        print('start process ', case, mask_name)
        mask_file = os.path.join(case_dir, mask_name)
        image_file = os.path.join(image_dir, case, mask_name)
        if not os.path.exists(image_file):
            print('{} not exist ...'.format(image_file))
            continue

        tumor_count = crop_tumor_patch(image_file, mask_file, tumor_dir, case)
        total_tumor_count += tumor_count
        if 'a' in case:
            meta_tumor_count += tumor_count
        # break
    print('total_tumor_count = ', total_tumor_count, ', meta_tumor_count = ', meta_tumor_count)


if __name__ == '__main__':
    prepare_tumor_data()
