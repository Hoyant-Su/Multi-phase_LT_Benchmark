import numpy as np
import pandas as pd
import random
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.src.feature_selection import remove_correlated_features
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.phase_feat_concat import radiomics_data_parse
from sklearn import svm
from sklearn import preprocessing
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.metric_evaluation import compute_metrics
from joblib import dump, load
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE

import json
import globals


# Feature selection
corr_threshold = globals.corr_threshold

# Fix seed
seed = 42
np.random.seed(seed)
random.seed(seed)
date = globals.date

model = 'svm'
exp = globals.exp

exp_class_num_dict = {'multi_cls': 5, 'binary_cls_23':2, 'Breast_Colon':2, 'multi_cls_4':4, 'multi_cls_3_except_for_Colon': 3, 'multi_cls_4_dataset_essay': 4}
n_classes = exp_class_num_dict[exp]

phases_num = globals.phases_num
misclassified_samples = []

fold_name = globals.fold_name
do_extension = globals.do_extension

file_root_path = f'../../tumor_radiomics/Label/exp/{exp}/{fold_name}'

do_oversample = 'False'

if __name__ == '__main__':
    fold_num = 5
    for fold_index in range(1, fold_num+1):
        
        #### step 1. prepare training and test data
        feature_file_training_set_art = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_art.csv')
        feature_file_training_set_delay = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_delay.csv')
        feature_file_training_set_nc = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_nc.csv')
        feature_file_training_set_pvp = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_pvp.csv')

        feature_file_training_set_dilated_ring_pvp = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_pvp_dilated_ring.csv')
        feature_file_training_set_dilated_ring_art = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_art_dilated_ring.csv')
        feature_file_training_set_dilated_ring_nc = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_nc_dilated_ring.csv')
        feature_file_training_set_dilated_ring_delay = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_delay_dilated_ring.csv')

        feature_file_training_set_eroded_ring_pvp = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_pvp_eroded_ring.csv')
        feature_file_training_set_eroded_ring_art = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_art_eroded_ring.csv')
        feature_file_training_set_eroded_ring_nc = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_nc_eroded_ring.csv')
        feature_file_training_set_eroded_ring_delay = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_delay_eroded_ring.csv')



        if do_extension:
            df_features_training = radiomics_data_parse(feature_file_training_set_pvp, feature_file_training_set_art, 
                                                        feature_file_training_set_nc, feature_file_training_set_delay,
                                                        feature_file_training_set_dilated_ring_pvp, feature_file_training_set_dilated_ring_art,
                                                        feature_file_training_set_dilated_ring_nc, feature_file_training_set_dilated_ring_delay
                                                        )
        else:
            if phases_num == 4:
                df_features_training = radiomics_data_parse(feature_file_training_set_pvp, feature_file_training_set_art, 
                                                            feature_file_training_set_nc, feature_file_training_set_delay)
            elif phases_num == 3:
                df_features_training = radiomics_data_parse(feature_file_training_set_pvp, feature_file_training_set_art, 
                                                            feature_file_training_set_nc)

        label_file_training_set = os.path.join(file_root_path, f'fold_{fold_index}_trainval_tumor_label.txt')
             
        labels_training = np.loadtxt(label_file_training_set)
        print('train dim: ', df_features_training.shape, len(labels_training))


        feature_file_test_set_art = os.path.join(file_root_path, f'fold_{fold_index}_test_features_art.csv')
        feature_file_test_set_delay = os.path.join(file_root_path, f'fold_{fold_index}_test_features_delay.csv')
        feature_file_test_set_pvp = os.path.join(file_root_path, f'fold_{fold_index}_test_features_pvp.csv')
        feature_file_test_set_nc = os.path.join(file_root_path, f'fold_{fold_index}_test_features_nc.csv')

        feature_file_test_set_eroded_ring_pvp = os.path.join(file_root_path, f'fold_{fold_index}_test_features_pvp_eroded_ring.csv')
        feature_file_test_set_eroded_ring_art = os.path.join(file_root_path, f'fold_{fold_index}_test_features_art_eroded_ring.csv')
        feature_file_test_set_eroded_ring_nc = os.path.join(file_root_path, f'fold_{fold_index}_test_features_nc_eroded_ring.csv')
        feature_file_test_set_eroded_ring_delay = os.path.join(file_root_path, f'fold_{fold_index}_test_features_delay_eroded_ring.csv')


        df_test = pd.read_csv(feature_file_test_set_art)
        test_tumor_id = df_test.iloc[:,0]


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



        # df_features_test = radiomics_data_parse(feature_file_test_set_pvp, feature_file_test_set_art,
        #                                         feature_file_test_set_nc, feature_file_test_set_delay)



        label_file_test_set = os.path.join(file_root_path, f'fold_{fold_index}_test_tumor_label.txt')
        #label_file_test_set = '../radiomics_fea_example/test_label.txt'
        labels_test = np.loadtxt(label_file_test_set)
        print('test dim: ', df_features_test.shape, len(labels_test))

        #### step 2. drop features
        #dropped_features = np.genfromtxt(f'feature_selection/{exp}/{fold_name}/fold_{fold_index}/dropped_features.csv', delimiter=',', dtype=str)
        if do_extension:
            dropped_features = np.genfromtxt(f'feature_selection/{exp}/{fold_name}/fold_{fold_index}/dropped_features.csv', delimiter=',', dtype=str)
        else:
            dropped_features = np.genfromtxt(f'feature_selection/{exp}/{fold_name}_without_extension/fold_{fold_index}/dropped_features.csv', delimiter=',', dtype=str)


        df_features_training_uncorr = df_features_training.drop(columns=dropped_features[:-1])
        df_features_test_uncorr = df_features_test.drop(columns=dropped_features[:-1])
        print('feature dim after drop: ', df_features_training_uncorr.shape, df_features_test_uncorr.shape)

        #### step 3. train SVM model
        X_train = df_features_training_uncorr.to_numpy()
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        y_train = labels_training

        smote = SMOTE(random_state=seed)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        clf = svm.SVC(C=2, kernel='rbf', gamma='auto', decision_function_shape='ovr', probability=True)

        if do_oversample == True:
            clf.fit(X_train_resampled, y_train_resampled)
        else:
            clf.fit(X_train_scaled, y_train)

        print ('model train over')

        #### step 4. test model
        X_test = df_features_test_uncorr.to_numpy()
        X_test_scaled = scaler.transform(X_test)

        y_test = labels_test

        y_score = clf.predict_proba(X_test_scaled)
        y_pred = np.argmax(y_score, axis=1)

        accuracy, kappa, specificity, sensitivity, f1_scores, cls_aucs, scores_json =  compute_metrics(y_test, y_pred, y_score, test_tumor_id)
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
        plt.title(f'{exp}_{model}_{date}_{fold_name}_fold_{fold_index}_ROC curve',fontsize=22)
        plt.legend(loc='lower right',fontsize=22)
        if do_extension:
            plt.savefig(f'../output/{exp}/{fold_name}/fold_{fold_index}/{exp}_{model}_{date}_AUC.png')
        else:
            plt.savefig(f'../output/{exp}/{fold_name}_without_extension/fold_{fold_index}/{exp}_{model}_{date}_AUC.png')
        #plt.show()
        ###
        misclassified_ids = df_features_test_uncorr.index[y_pred != y_test].tolist()

        misclassified_labels = y_test[y_pred != y_test]
        misclassified_predictions = y_pred[y_pred != y_test]

        for idx, true_label, pred_label in zip(misclassified_ids, misclassified_labels, misclassified_predictions):
            misclassified_samples.append({'ID': test_tumor_id[idx], 'True Label': true_label, 'Predicted Label': pred_label})

    misclassified_df = pd.DataFrame(misclassified_samples)
    if do_extension:
        os.makedirs(f'../output/{exp}/{fold_name}/misclassified_samples/{model}', exist_ok=True)
        misclassified_df.to_csv(f'../output/{exp}/{fold_name}/misclassified_samples/{model}/misclassified_samples_summary.txt', index=False, sep='\t')
    else:
        os.makedirs(f'../output/{exp}/{fold_name}_without_extension/misclassified_samples/{model}', exist_ok=True)
        misclassified_df.to_csv(f'../output/{exp}/{fold_name}_without_extension/misclassified_samples/{model}/misclassified_samples_summary.txt', index=False, sep='\t')       