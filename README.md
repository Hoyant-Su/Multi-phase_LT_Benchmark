# MCT-LTDiag
📝 A code repository developed for benchmark construction on differential diagnosis of liver tumors.




## What's New
- [2025-09] Released the first version of our specialized liver tumor classification model **RU-Net**, trained on **MCT_LTDiag**.  👉 [Model Link](https://github.com/Hoyant-Su/RU-Net)
- [2025-05] Officially released our liver tumor NIfTI dataset on the Harvard Dataverse platform.  👉 [Dataset Link](https://doi.org/10.7910/DVN/S3RW15)

## Environment Setup

⚙️ Follow the steps below to set up the PyTorch environment required for this project:

```bash
cd Multi_phase_liver_tumor_benchmark
conda env create -f environment.yml
conda activate LT_benchmark
```

## Data Preparation
🗂️ Prepare the data in the following structure:
```bash
Data/
  └── mask/
    ├── tumor_type/
      ├── case1/
        ├── mask_pvp.nii.gz
      ├── .../
    ├── .../
  └── volume/
    ├── tumor_type/
      ├── case1/
        ├── phase1.nii.gz
        ├── phase2.nii.gz
        ├── ...
      ├── .../
    ├── .../
    
Label/
  └── exp/
    └── <exp_name>/<exp_date>/ # Experiment directory with name and date
      ├── fold_1_xxx.csv 
      ├── fold_2_xxx.csv 
      └── ... 
```

The dataset, along with feature and label files with `.csv` and `.txt` suffixes, can be accessed at 👉 **[Dataset Repository]()**

## Data Preprocessing Pipeline
Here we present a standardized data preprocessing workflow with executable scripts, comprising two core components:

### 📦 Multi-phase Registration Protocol
```text
 Input: NC, AP, DP phase scans  
 Process: Multiphase registration using ITKElastix (v0.20.0)
          Hierarchical B-spline deformation
 Output: PVP-phase aligned volumes
```

### 📦 AI-driven Segmentation Workflow
processing steps: Model mask generation -> Expert refinement -> VOI extraction

Here we provided workflow related scripts `regis.py`, `seg_for_mask.py`, `VOI_extraction.py` to ensure reproducibility.

Comprehensive flowchart can be referred: 
![image](https://github.com/Hoyant-Su/Multi-phase_LT_Benchmark/blob/main/flow_chart_v0425.png)


## Benchmark and Implementation
We follow existing literature on liver tumor diagnosis to implement model architectures that have proven effective for differential diagnosis of liver tumors. Follow the instruction bellow to run the implementation codes.
### Training & Validation
```bash
# Machine Learning Baseline
python radiomics_extract_classification/train_random_forest_K_fold.py
# Deep Learning Baseline
bash dl_classification/do_train_liver_parallel.sh
```
### Testing
```bash
# Machine Learning Baseline
python radiomics_extract_classification/test_random_forest_K_fold.py
python radiomics_extract_classification/train_svm_K_fold.py
# Deep Learning Baseline
bash /dl_classification/do_test_liver_K.sh
```


## Contributing Institutions

The following institutions have contributed to the development of this project:

<p align="center">
  <img src="./assets/logos/PUMCH.jpg" alt="Peking Union Medical College Hospital" style="height:120px; margin:0 15px; object-fit: contain;">
  <img src="/assets/logos/Fudan_logo.svg" alt="Fudan University" style="height:100px; margin:0 15px; object-fit: contain;"/>
  <img src="/assets/logos/SII_logo.svg" alt="Shanghai Innovation Institute" style="height:100px; margin:0 15px; object-fit: contain;"/>
  <img src="/assets/logos/AI_LAB_logo.svg" alt="Shanghai Artificial Intelligence Laboratory" style="height:200px; margin:0 15px; object-fit: contain;"/>
</p>
