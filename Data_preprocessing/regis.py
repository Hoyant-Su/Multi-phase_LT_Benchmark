import itk
import os

def do_registration(fixed_img_path, moving_img_path, save_img_path):
    fixed_image = itk.imread(fixed_img_path, itk.F)
    moving_image = itk.imread(moving_img_path, itk.F)

    parameter_object = itk.ParameterObject.New()
    parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", 3, 20.0)
    parameter_object.AddParameterMap(parameter_map_bspline)

    result_image, _ = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=False)

    itk.imwrite(result_image, save_img_path)
    print("Saved:", save_img_path)

def main():
    fixed_image_folder = "" #provide PVP phase file root
    moving_image_parent_folder = "" #provide NC, AP, DP phases file root
    result_dir = "" #fill in the blank


    for case in os.listdir(fixed_image_folder):
        fixed_pvp_path = os.path.join(fixed_image_folder, case)
        fixed_image_path = os.path.join(fixed_pvp_path, "pvp.nii.gz")

        patient_folder_name = os.path.splitext(case)[0].replace('.nii', '')
        moving_images_folder = os.path.join(moving_image_parent_folder, patient_folder_name)
        
        if os.path.exists(moving_images_folder):
            moving_images = []
            for moving_image_name in ["art.nii.gz", "nc.nii.gz","delay.nii.gz"]:
                moving_image_path = os.path.join(moving_images_folder, moving_image_name)
                if os.path.exists(moving_image_path):
                    moving_images.append(moving_image_path)

            for moving_image_path in moving_images:
                result_image_name = os.path.basename(moving_image_path)
                result_image_path = os.path.join(result_dir, patient_folder_name, result_image_name)
                os.makedirs(os.path.dirname(result_image_path), exist_ok=True)
                do_registration(fixed_image_path, moving_image_path, result_image_path)


if __name__ == '__main__':
    main()
