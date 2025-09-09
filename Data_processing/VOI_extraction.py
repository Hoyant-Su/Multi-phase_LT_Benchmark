import os
import shutil

import numpy as np
import csv
import random
import SimpleITK as sitk
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor, as_completed

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
            # print(row)
            hcp_info_dict = dict()
            hcp_info_dict['image'] = row[0]
            hcp_info_dict['mask'] = row[1]
            dict_list.append(hcp_info_dict)

    return dict_list


def crop_tumor_patch(image_file, mask_file, tumor_dir, case_id):
    case_id = os.path.basename(case_id)
    mask = sitk.ReadImage(mask_file)
    mask_data = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()


    image = sitk.ReadImage(image_file)
    image_data = sitk.GetArrayFromImage(image)

    #print(mask_data.shape, spacing)

    tumor_count = 0
    if mask_data.shape != image_data.shape:
        print(f'{case_id}: mask and image shape mismatch')
        return 0

    ### find tumors, give up too small / single slice
    input_mask = sitk.GetImageFromArray(mask_data)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(input_mask)
    num_connected_regions = cc_filter.GetObjectCount()

    #print('num_connected_regions = ', num_connected_regions)
    if num_connected_regions >= 1:
        lss_filter = sitk.LabelShapeStatisticsImageFilter()
        lss_filter.Execute(output_mask)
        np_output_mask = sitk.GetArrayFromImage(output_mask)
        for idx in range(1, num_connected_regions + 1):
            area = lss_filter.GetNumberOfPixels(idx)
            # print(idx, area)
            if area <= 50: ## optional
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
            #print('min z = {}, max z = {}, min y = {}, max y = {}, min x = {}, max x = {}'.format(min_z, max_z, min_y, max_y, min_x, max_x))

            if min_z == max_z:
                print('this tumor only contain one slice ...')
                continue

            tumor_mask = sitk.GetImageFromArray(tumor_mask_data)
            tumor_mask.SetSpacing(mask.GetSpacing())
            tumor_mask.SetOrigin(mask.GetOrigin())
            tumor_mask.SetDirection(mask.GetDirection())
            # save_tumor_mask_dir = os.path.join(tumor_dir, 'total_mask', case_id)
            # make_dir(save_tumor_mask_dir)
            # sitk.WriteImage(tumor_mask, os.path.join(save_tumor_mask_dir, 'tumor_{}.nii.gz'.format(tumor_count)))

            # patch_xy_size = max_x - min_x
            # if patch_xy_size < max_y - min_y:
            #     patch_xy_size = max_y - min_y
            # patch_xy_size = patch_xy_size * spacing[0] * 1.5
            # patch_xy_half = int(0.5 * patch_xy_size / spacing[0])
            
            # patch_x0 = center_x - patch_xy_half
            # patch_x1 = center_x + patch_xy_half
            # patch_y0 = center_y - patch_xy_half
            # patch_y1 = center_y + patch_xy_half
            # patch_z0 = min_z - int((max_z-min_z+1)/2 + 0.5)
            # patch_z1 = max_z + int((max_z-min_z+1)/2 + 0.5)
            # if patch_z0 <= 0:
            #     patch_z0 = 0
            # if patch_z1 >= mask_data.shape[0] - 1:
            #     patch_z1 = mask_data.shape[0] - 1

            patch_size = (30, 160, 160)  
            half_z, half_y, half_x = patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2  

            patch_x0 = center_x - half_x
            patch_x1 = center_x + half_x
            patch_y0 = center_y - half_y
            patch_y1 = center_y + half_y
            patch_z0 = center_z - half_z
            patch_z1 = center_z + half_z

            #print(patch_x0, patch_x1, patch_y0, patch_y1, patch_z0, patch_z1)
            
            patch_x0 = max(0, patch_x0)
            patch_y0 = max(0, patch_y0)
            patch_z0 = max(0, patch_z0)

            patch_x1 = min(mask_data.shape[2] - 1, patch_x1)
            patch_y1 = min(mask_data.shape[1] - 1, patch_y1)
            patch_z1 = min(mask_data.shape[0] - 1, patch_z1)
            
            #print(patch_x0, patch_x1, patch_y0, patch_y1, patch_z0, patch_z1)
            if patch_z1 <= patch_z0 or patch_y1 <= patch_y0 or patch_x1 <= patch_x0:
                print(f"Invalid patch size: {(patch_z1 - patch_z0, patch_y1 - patch_y0, patch_x1 - patch_x0)}")
                continue

            patch_mask_data = mask_data[patch_z0:patch_z1, patch_y0:patch_y1, patch_x0:patch_x1]
            patch_mask = sitk.GetImageFromArray(patch_mask_data)
            patch_mask.SetSpacing(mask.GetSpacing())
            patch_mask.SetOrigin(mask.GetOrigin())
            patch_mask.SetDirection(mask.GetDirection())
            save_tumor_mask_dir = os.path.join(tumor_dir, 'mask')
            make_dir(save_tumor_mask_dir)
            save_tumor_mask = f"{save_tumor_mask_dir}/{case_id}_tumor_{tumor_count}.nii.gz"
            sitk.WriteImage(patch_mask, save_tumor_mask)


            for phase in ['pvp', 'art', 'nc', 'delay']:
                img_file = image_file.replace('pvp.nii.gz', '{}.nii.gz'.format(phase))

                image = sitk.ReadImage(img_file)
                image_data = sitk.GetArrayFromImage(image)

                patch_volume_data = image_data[patch_z0:patch_z1, patch_y0:patch_y1, patch_x0:patch_x1]

                patch_volume = sitk.GetImageFromArray(patch_volume_data)
                patch_volume.SetSpacing(image.GetSpacing())
                patch_volume.SetOrigin(image.GetOrigin())
                patch_volume.SetDirection(image.GetDirection())
                save_tumor_data_dir = os.path.join(tumor_dir, 'image')
                make_dir(save_tumor_data_dir)
                save_tumor_data = f"{save_tumor_data_dir}/{case_id}_tumor_{tumor_count}_{phase}.nii.gz"
                sitk.WriteImage(patch_volume, save_tumor_data)
                
            tumor_count += 1

    return tumor_count


def process_case(case, image_dir, mask_dir, tumor_dir):
    case_dir = os.path.join(mask_dir, case)
    mask_name = os.listdir(case_dir)[0]
    #print(f"Start process {case}, {mask_name}")
    mask_file = os.path.join(case_dir, mask_name)
    image_file = os.path.join(image_dir, case, mask_name.split('mask_')[-1])
    if not os.path.exists(image_file):
        print(f"{case}: {image_file} not exist, skipped")
        return case, 0
    return case, crop_tumor_patch(image_file, mask_file, tumor_dir, case)

def prepare_tumor_data():
    image_dir = ""
    mask_dir = ""
    tumor_dir = ""

    total_tumor_count = 0
    meta_tumor_count = 0
    case_set = os.listdir(mask_dir)

    existed_case_list = [item.split('_tumor')[0] for item in os.listdir(f"{tumor_dir}/mask")]
    futures = []
    with ProcessPoolExecutor(max_workers=64) as executor:
        for case in case_set:
            if case in existed_case_list:
                continue
            futures.append(executor.submit(process_case, case, image_dir, mask_dir, tumor_dir))

        for future in as_completed(futures):
            result_case, result_tumor_count = future.result()
            if result_tumor_count == 0:
                print(f"{result_case}: no tumor extracted")
            #print(f"Future result: {result_case}")
            case, tumor_count = future.result()
            total_tumor_count += tumor_count
            if 'a' in case:
                meta_tumor_count += tumor_count

    print('total_tumor_count = ', total_tumor_count, ', meta_tumor_count = ', meta_tumor_count)

if __name__ == '__main__':
    prepare_tumor_data()