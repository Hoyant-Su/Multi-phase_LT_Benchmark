# Multi-phase_LT_Benchmark
A code repository developed for benchmark construction on differential diagnosis of liver tumors.

## Environment Setup

Follow the steps below to set up the PyTorch environment required for this project:

```bash
cd Multi_phase_liver_tumor_benchmark
conda env create -f environment.yml
conda activate LT_benchmark
```

## Data Preparation
Prepare the data in the following structure:
Data/ 

  ├── mask/ # Tumor mask files 
  
  ├── volume/ # Registered CT volumes 
  
Label/ 

  └── exp/ 
  
  └── <exp_name>/<exp_date>/
  
    ├── fold_0_xxx.csv 
    
    ├── fold_1_xxx.csv 
    
    └── ... 
    
All images, along with radiomics feature files and label files with `.csv` and `.txt` suffixes, can be accessed at:  
**[Dataset Repository]()**


## Implementation
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
