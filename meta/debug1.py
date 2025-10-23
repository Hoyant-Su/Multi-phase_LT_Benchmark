import nibabel as nib

def get_nii_gz_dimensions(file_path):
    """
    读取 .nii.gz 文件并返回图像的尺寸（shape）。

    参数:
        file_path (str): .nii.gz 文件的路径

    返回:
        tuple: 图像数据的维度
    """
    try:
        # 读取 .nii.gz 文件
        img = nib.load(file_path)

        # 获取图像数据
        data = img.get_fdata()  # 获取原始数组（float格式）

        # 返回数据尺寸
        return data.shape
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

# 示例调用
file_path = "/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/tumor_radiomics/Data/final_volume/230218a1/pvp.nii.gz"  # 请替换为您实际的文件路径
dimensions = get_nii_gz_dimensions(file_path)

if dimensions:
    print(f"图像尺寸: {dimensions}")
else:
    print("无法获取文件的尺寸")
