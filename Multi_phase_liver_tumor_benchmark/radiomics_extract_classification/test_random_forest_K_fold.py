from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.src.model_evaluation import adjusted_prediction, eval_fpr_tpr, eval_sensitivity_specificity
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.phase_feat_concat import radiomics_data_parse
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.metric_evaluation import compute_metrics
import os 
import json
from sklearn.metrics import roc_curve, auc

import globals

# Feature selection
corr_threshold = globals.corr_threshold

# Fix seed
seed = 42
np.random.seed(seed)
random.seed(seed)
exp = globals.exp

exp_class_num_dict = {'multi_cls': 5, 'binary_cls_23':2, 'Breast_Colon':2, 'multi_cls_4':4, 'multi_cls_3_except_for_Colon': 3, 'multi_cls_4_dataset_essay': 4}
n_classes = exp_class_num_dict[exp]

date = globals.date
fold_name = globals.fold_name
do_extension = globals.do_extension

model = 'rf'
file_root_path = f'../../tumor_radiomics/Label/exp/{exp}/{fold_name}'

#phases_num = 3
phases_num = 4

misclassified_samples = []

if __name__ == '__main__':
    fold_num = 5
    for fold_index in range(1, fold_num+1):

        feature_file_test_set_art = os.path.join(file_root_path, f'fold_{fold_index}_test_features_art.csv')
        feature_file_test_set_delay = os.path.join(file_root_path, f'fold_{fold_index}_test_features_delay.csv')
        feature_file_test_set_pvp = os.path.join(file_root_path, f'fold_{fold_index}_test_features_pvp.csv')
        feature_file_test_set_nc = os.path.join(file_root_path, f'fold_{fold_index}_test_features_nc.csv')

        feature_file_test_set_dilated_ring_pvp = os.path.join(file_root_path, f'fold_{fold_index}_test_features_pvp_dilated_ring.csv')
        feature_file_test_set_dilated_ring_art = os.path.join(file_root_path, f'fold_{fold_index}_test_features_art_dilated_ring.csv')
        feature_file_test_set_dilated_ring_nc = os.path.join(file_root_path, f'fold_{fold_index}_test_features_nc_dilated_ring.csv')
        feature_file_test_set_dilated_ring_delay = os.path.join(file_root_path, f'fold_{fold_index}_test_features_delay_dilated_ring.csv')

        feature_file_test_set_eroded_ring_pvp = os.path.join(file_root_path, f'fold_{fold_index}_test_features_pvp_eroded_ring.csv')
        feature_file_test_set_eroded_ring_art = os.path.join(file_root_path, f'fold_{fold_index}_test_features_art_eroded_ring.csv')
        feature_file_test_set_eroded_ring_nc = os.path.join(file_root_path, f'fold_{fold_index}_test_features_nc_eroded_ring.csv')
        feature_file_test_set_eroded_ring_delay = os.path.join(file_root_path, f'fold_{fold_index}_test_features_delay_eroded_ring.csv')



        df_test = pd.read_csv(feature_file_test_set_art)
        test_tumor_id = df_test.iloc[:,0]
        #print(test_tumor_id.shape)

        if do_extension:
            df_features_test = radiomics_data_parse(feature_file_test_set_pvp, feature_file_test_set_art, 
                                                        feature_file_test_set_nc, feature_file_test_set_delay,
                                                        feature_file_test_set_dilated_ring_pvp, feature_file_test_set_dilated_ring_art,
                                                        feature_file_test_set_dilated_ring_nc, feature_file_test_set_dilated_ring_delay
                                                    )
        else:
            if phases_num == 4:
                df_features_test = radiomics_data_parse(feature_file_test_set_pvp, feature_file_test_set_art, 
                                                        feature_file_test_set_nc, feature_file_test_set_delay)
            elif phases_num == 3:
                df_features_test = radiomics_data_parse(feature_file_test_set_pvp, feature_file_test_set_art, 
                                                        feature_file_test_set_nc)



        label_file_test_set = os.path.join(file_root_path, f'fold_{fold_index}_test_tumor_label.txt')
        labels_test = np.loadtxt(label_file_test_set)

        if do_extension:
            dropped_features = np.genfromtxt(f'feature_selection/{exp}/{fold_name}/fold_{fold_index}/dropped_features.csv', delimiter=',', dtype=str)
        else:
            dropped_features = np.genfromtxt(f'feature_selection/{exp}/{fold_name}_without_extension/fold_{fold_index}/dropped_features.csv', delimiter=',', dtype=str)

        # Drop features from the test set
        df_features_test_uncorr = df_features_test.drop(columns=dropped_features[:-1])
        print (df_features_test_uncorr.shape)

        # Load previously trained random forest model
        if do_extension:
            rf = load(f'trained_model_0821/{fold_name}/fold_{fold_index}/random_forest_model_{exp}.joblib')
        else:
            rf = load(f'trained_model_0821/{fold_name}_without_extension/fold_{fold_index}/random_forest_model_{exp}.joblib')            

        # Prepare test dataset
        X_test = df_features_test_uncorr.to_numpy()
        y_test = labels_test

        # Predict on test set
        y_score = rf.predict_proba(X_test)
        y_pred = np.argmax(y_score, axis=1)

        
        accuracy, kappa, specificity, sensitivity, f1_scores, cls_aucs, scores_json =  compute_metrics(y_test, y_pred, y_score, test_tumor_id)

        # Save the sample metrics to a JSON file
        if do_extension:
            os.makedirs(f'../output/{exp}/{fold_name}/fold_{fold_index}',exist_ok=True)
            with open(f'../output/{exp}/{fold_name}/fold_{fold_index}/{model}_score.json', 'w') as f:
                json.dump(scores_json, f, indent=4)
        else:
            os.makedirs(f'../output/{exp}/{fold_name}_without_extension/fold_{fold_index}',exist_ok=True)
            with open(f'../output/{exp}/{fold_name}_without_extension/fold_{fold_index}/{model}_score.json', 'w') as f:
                json.dump(scores_json, f, indent=4)

        df = pd.DataFrame({
            'Class': np.arange(n_classes),
            'Specificity': specificity,
            'Sensitivity': sensitivity,
            'F1-Score': f1_scores,
            'AUC': cls_aucs
        })

        overall_metrics = pd.DataFrame({
            'Class': ['Overall'],
            'Specificity': [np.nan],
            'Sensitivity': [np.nan],
            'F1-Score': [np.nan],
            'AUC': [np.nan],
            'Accuracy': [accuracy],
            'Kappa': [kappa]
        })
        df = pd.concat([df, overall_metrics], ignore_index=True)

        if do_extension:
            os.makedirs(f'../output/{exp}/{fold_name}/fold_{fold_index}',exist_ok=True)
            df.to_csv(f'../output/{exp}/{fold_name}/fold_{fold_index}/results_{model}_{date}.csv', index=False)
        else:
            os.makedirs(f'../output/{exp}/{fold_name}_without_extension/fold_{fold_index}',exist_ok=True)
            df.to_csv(f'../output/{exp}/{fold_name}_without_extension/fold_{fold_index}/results_{model}_{date}.csv', index=False)
        
        plt.figure(figsize=(10, 8))
        if exp == 'multi_cls_4_dataset_essay':
            class_dict = {'Class 0': 'BCLM_CRLM', 'Class 1': 'HCC', 'Class 2': 'HH', 'Class 3': 'ICC'}

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test, y_score[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)

            class_name = class_dict[f'Class {i}']
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate',fontsize=22)
        plt.ylabel('True Positive Rate',fontsize=22)
        plt.title(f'{exp}_{model}_{date}_fold_{fold_index}_ROC curve',fontsize=22)
        plt.legend(loc='lower right',fontsize=22)
        if do_extension:
            plt.savefig(f'../output/{exp}/{fold_name}/fold_{fold_index}/{exp}_{model}_{date}_AUC.png')
        else:
            plt.savefig(f'../output/{exp}/{fold_name}_without_extension/fold_{fold_index}/{exp}_{model}_{date}_AUC.png')           


        # Identify and collect misclassified samples
        misclassified_ids = df_features_test_uncorr.index[y_pred != y_test].tolist()

        misclassified_labels = y_test[y_pred != y_test]
        misclassified_predictions = y_pred[y_pred != y_test]

        for idx, true_label, pred_label in zip(misclassified_ids, misclassified_labels, misclassified_predictions):
            misclassified_samples.append({'ID': test_tumor_id[idx], 'True Label': true_label, 'Predicted Label': pred_label})

    # Save all misclassified samples to a single TXT file
    misclassified_df = pd.DataFrame(misclassified_samples)

    if do_extension:
        os.makedirs(f'../output/{exp}/{fold_name}/misclassified_samples/{model}', exist_ok=True)
        misclassified_df.to_csv(f'../output/{exp}/{fold_name}/misclassified_samples/{model}/misclassified_samples_summary.txt', index=False, sep='\t')
    else:
        os.makedirs(f'../output/{exp}/{fold_name}_without_extension/misclassified_samples/{model}', exist_ok=True)
        misclassified_df.to_csv(f'../output/{exp}/{fold_name}_without_extension/misclassified_samples/{model}/misclassified_samples_summary.txt', index=False, sep='\t')       